import os
import logging
import time
import json
import numpy as np
import cv2
from PIL import Image
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import base64
from openai import OpenAI
from pydantic import BaseModel
from jose import jwt, JWTError
from datetime import datetime, timedelta
import secrets

# Force Tesseract path
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ---------------- Config ----------------
SECRET_KEY = "supersecretkey"  # Replace for production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("receipt-ocr")

# ---------------- Load ENV ----------------
load_dotenv()

def _clean_env(val: str | None) -> str | None:
    if val is None:
        return None
    v = val.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    if v.endswith("\\"):
        v = v[:-1]
    return v or None

def _mask(k: str | None) -> str:
    if not k:
        return "(missing)"
    return k[:4] + "..." + k[-4:]

OPENAI_API_KEY = _clean_env(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL   = _clean_env(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"

log.info(f"OpenAI model: {OPENAI_MODEL}")
log.info(f"OpenAI key tail (masked): {_mask(OPENAI_API_KEY)}")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- FastAPI app ----------------
app = FastAPI(title="Receipt OCR API (Tesseract + OpenAI + Auth)", version="2.0.0")

# ---------------- Data Stores ----------------
users_db = {}         # email -> {password, api_keys: [], receipts: {}}
receipts_store = {}   # receipt_id -> {status, result}

# ---------------- Models ----------------
class SignupModel(BaseModel):
    email: str
    password: str

class LoginModel(BaseModel):
    email: str
    password: str

class APIKeyModel(BaseModel):
    name: str

class UploadMetaModel(BaseModel):
    filename: str
    file_extension: str
    file_size: int

# ---------------- Auth Helpers ----------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        log.error("[AUTH] Missing or invalid JWT header")
        raise HTTPException(status_code=401, detail="Missing JWT")

    token = auth_header.split("Bearer ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None or email not in users_db:
            log.error("[AUTH] Invalid JWT payload or unknown user")
            raise HTTPException(status_code=401, detail="Invalid JWT")
        return email
    except JWTError:
        log.exception("[AUTH] JWT decode error")
        raise HTTPException(status_code=401, detail="Invalid JWT")

def get_user_by_api_key(api_key: str):
    for email, user in users_db.items():
        if api_key in user["api_keys"]:
            return email
    log.error("[AUTH] Invalid API key used")
    raise HTTPException(status_code=401, detail="Invalid API key")

# ---------------- Image helpers ----------------
def _resize_if_huge(bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    log.info(f"[PREPROCESS] Starting preprocessing. Input size: {len(image_bytes)} bytes")
    t0 = time.perf_counter()
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        log.error("[PREPROCESS] Failed to decode image.")
        raise ValueError("Invalid image data")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )
    elapsed = time.perf_counter() - t0
    log.info(f"[PREPROCESS] Completed in {elapsed:.3f} seconds.")
    return thr

def pil_from_cv2(mat: np.ndarray) -> Image.Image:
    if len(mat.shape) == 2:
        return Image.fromarray(mat)
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))

# ---------------- OCR with Tesseract ----------------
def run_tesseract(img_bin: np.ndarray, lang: str, psm: int, oem: int) -> Dict[str, Any]:
    log.info(f"[TESSERACT] Starting OCR with lang={lang}, psm={psm}, oem={oem}")
    t0 = time.perf_counter()
    config = f"--oem {oem} --psm {psm}"
    try:
        text = pytesseract.image_to_string(pil_from_cv2(img_bin), lang=lang, config=config)
    except Exception:
        log.exception("[TESSERACT] Error during OCR")
        raise
    elapsed = time.perf_counter() - t0
    log.info(f"[TESSERACT] Completed OCR in {elapsed:.3f} seconds. Extracted {len(text)} characters.")
    log.info(f"[TESSERACT] OCR Preview: {text[:200].replace(chr(10), ' ')} ...")
    return {"text": text, "time_sec": round(elapsed, 3)}

# ---------------- OpenAI extraction ----------------
SYSTEM_INSTRUCTIONS = """You are a careful information extraction system.
You receive OCR output of a retail receipt. Fix obvious OCR errors and return ONLY strict JSON:
- Normalize numbers (no commas), use floats where appropriate.
- If dates are present, prefer 'YYYY-MM-DD HH:MM:SS' 24h.
- Include Items as an array (name/qty/price/tax when detectable).
- If any field cannot be found, look again in the OCR output extract.
- You can also use the image to extract the correct JSON value along with the OCR extract.
- You are an intelligent LLM, be createive and effective in populating the values in JSON.
- Return ONLY the JSON per the provided schema—no extra text.
"""

def _safe_json_loads(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        cleaned = txt.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
        return json.loads(cleaned)

def extract_receipt_to_json(ocr_text: str, image_b64: str) -> Dict[str, Any]:
    log.info("[OPENAI] Starting JSON extraction from OCR text + image.")
    t0 = time.perf_counter()
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ocr_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }
    try:
        chat = client.chat.completions.create(**payload)
        text = chat.choices[0].message.content
    except Exception:
        log.exception("[OPENAI] API request failed")
        raise
    elapsed = time.perf_counter() - t0
    log.info(f"[OPENAI] Extraction completed in {elapsed:.3f} seconds.")
    return _safe_json_loads(text)

# ---------------- Auth Endpoints ----------------
@app.post("/api/auth/signup")
def signup(data: SignupModel):
    if data.email in users_db:
        log.warning(f"[SIGNUP] Attempt to register existing email: {data.email}")
        raise HTTPException(status_code=400, detail="User exists")
    users_db[data.email] = {"password": data.password, "api_keys": [], "receipts": {}}
    token = create_access_token({"sub": data.email})
    log.info(f"[SIGNUP] New user registered: {data.email}")
    return {"email": data.email, "token": token}

@app.post("/api/auth/signin")
def signin(data: LoginModel):
    user = users_db.get(data.email)
    if not user or user["password"] != data.password:
        log.warning(f"[LOGIN] Failed login attempt for email: {data.email}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": data.email})
    log.info(f"[LOGIN] Successful login: {data.email}")
    return {"email": data.email, "token": token}

# ---------------- API Key ----------------
@app.post("/api/keys/generate")
def generate_api_key(data: APIKeyModel, email: str = Depends(get_current_user)):
    key = secrets.token_hex(16)
    users_db[email]["api_keys"].append(key)
    log.info(f"[API KEY] Generated new key for {email}")
    return {"name": data.name, "key": key}

# ---------------- Receipt Upload Flow ----------------
@app.post("/api/receipts/upload")
def upload_receipt_meta(meta: UploadMetaModel, x_api_key: str = Query(...)):
    email = get_user_by_api_key(x_api_key)
    receipt_id = secrets.token_hex(8)
    receipts_store[receipt_id] = {"status": "processing", "result": None}
    upload_url = f"https://fake-upload-url.com/{receipt_id}"
    log.info(f"[UPLOAD] Meta received for {email} — Receipt ID: {receipt_id}")
    return {"id": receipt_id, "upload_url": upload_url}

@app.get("/api/receipts/{rid}/status")
def get_status(rid: str, x_api_key: str = Query(...)):
    get_user_by_api_key(x_api_key)
    rec = receipts_store.get(rid)
    if not rec:
        log.warning(f"[STATUS] Receipt not found: {rid}")
        raise HTTPException(status_code=404, detail="Not found")
    return {"id": rid, "status": rec["status"]}

@app.get("/api/receipts/{rid}/result")
def get_result(rid: str, x_api_key: str = Query(...)):
    get_user_by_api_key(x_api_key)
    rec = receipts_store.get(rid)
    if not rec:
        log.warning(f"[RESULT] Receipt not found: {rid}")
        raise HTTPException(status_code=404, detail="Not found")
    return rec["result"] or {}

# ---------------- OCR Direct Endpoint ----------------
@app.post("/ocr-extract")
async def ocr_extract_one_shot(file: UploadFile = File(...), lang: str = Query("eng"), psm: int = Query(6), oem: int = Query(3)):
    log.info(f"[ENDPOINT] /ocr-extract called. Filename: {file.filename}")
    try:
        content = await file.read()
        if not content:
            log.error("[ENDPOINT] Uploaded file is empty.")
            raise HTTPException(status_code=400, detail="Empty file")
        log.info(f"[ENDPOINT] File size: {len(content)} bytes.")

        image_b64 = base64.b64encode(content).decode("utf-8")
        pre_bin = preprocess_image(content)
        tesseract_res = run_tesseract(pre_bin, lang, psm, oem)
        extracted = extract_receipt_to_json(tesseract_res.get("text", ""), image_b64)

        log.info("[ENDPOINT] OCR and extraction completed successfully.")
        return JSONResponse({
            "filename": file.filename,
            "ocr": {"tesseract": tesseract_res},
            "extracted": extracted
        })
    except HTTPException:
        raise
    except Exception as e:
        log.exception("[ENDPOINT] Unexpected error during OCR processing.")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Health ----------------
@app.api_route("/", methods=["GET", "HEAD"], tags=["Health"])
def root():
    return {"status": "ok", "message": "Receipt OCR API is running. Visit /docs for API docs."}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
