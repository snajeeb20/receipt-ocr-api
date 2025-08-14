import os
import logging
import time
import json
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel
from jose import jwt, JWTError
from datetime import datetime, timedelta
import secrets

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
app = FastAPI(title="Receipt OCR API (OpenAI Vision + Auth)", version="3.0.0")

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
        raise HTTPException(status_code=401, detail="Missing JWT")

    token = auth_header.split("Bearer ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None or email not in users_db:
            raise HTTPException(status_code=401, detail="Invalid JWT")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid JWT")

def get_user_by_api_key(api_key: str):
    for email, user in users_db.items():
        if api_key in user["api_keys"]:
            return email
    raise HTTPException(status_code=401, detail="Invalid API key")

# ---------------- OpenAI extraction ----------------
SYSTEM_INSTRUCTIONS = """You are a careful information extraction system.
You receive OCR output of a retail receipt. Fix obvious OCR errors and return ONLY strict JSON:
- Normalize numbers (no commas), use floats where appropriate.
- If dates are present, prefer 'YYYY-MM-DD HH:MM:SS' 24h.
- Include Items as an array (name/qty/price/tax when detectable).
- If any field cannot be found, look again in the OCR output extract.
- You can also use the image to extract the correct JSON value along with the OCR extract.
- You are an intelligent LLM, be createive and effective in populating the values in JOSN.
- Return ONLY the JSON per the provided schema—no extra text.
- Ensure the extraction also includes shop name, grand total, fbr invoice, address, datetime, grand total and bill number.
- Remember dates are all of the current year.
- This is important make sure you dont read 2025 as 2023 or 2024.
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

# ---------------- Auth Endpoints ----------------
@app.post("/api/auth/signup")
def signup(data: SignupModel):
    if data.email in users_db:
        raise HTTPException(status_code=400, detail="User exists")
    users_db[data.email] = {"password": data.password, "api_keys": [], "receipts": {}}
    token = create_access_token({"sub": data.email})
    return {"email": data.email, "token": token}

@app.post("/api/auth/signin")
def signin(data: LoginModel):
    user = users_db.get(data.email)
    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": data.email})
    return {"email": data.email, "token": token}

# ---------------- API Key ----------------
@app.post("/api/keys/generate")
def generate_api_key(data: APIKeyModel, email: str = Depends(get_current_user)):
    key = secrets.token_hex(16)
    users_db[email]["api_keys"].append(key)
    return {"name": data.name, "key": key}

# ---------------- Receipt Upload Flow ----------------
@app.post("/api/receipts/upload")
def upload_receipt_meta(meta: UploadMetaModel, x_api_key: str = Query(...)):
    email = get_user_by_api_key(x_api_key)
    receipt_id = secrets.token_hex(8)
    receipts_store[receipt_id] = {"status": "processing", "result": None}
    upload_url = f"https://fake-upload-url.com/{receipt_id}"
    return {"id": receipt_id, "upload_url": upload_url}

@app.get("/api/receipts/{rid}/status")
def get_status(rid: str, x_api_key: str = Query(...)):
    get_user_by_api_key(x_api_key)
    rec = receipts_store.get(rid)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return {"id": rid, "status": rec["status"]}

@app.get("/api/receipts/{rid}/result")
def get_result(rid: str, x_api_key: str = Query(...)):
    get_user_by_api_key(x_api_key)
    rec = receipts_store.get(rid)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return rec["result"] or {}

# ---------------- OCR Direct Endpoint (OpenAI Vision only) ----------------
@app.post("/ocr-extract")
async def ocr_extract_one_shot(file: UploadFile = File(...)):
    try:
        log.info(f"[UPLOAD] Received file: {file.filename}")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        image_b64 = base64.b64encode(content).decode("utf-8")

        log.info("[OCR] Sending image directly to OpenAI Vision API")
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }

        chat = client.chat.completions.create(**payload)
        text = chat.choices[0].message.content

        extracted = _safe_json_loads(text)
        log.info(f"[OCR] Extraction completed successfully for {file.filename}")

        return JSONResponse({
            "filename": file.filename,
            "extracted": extracted
        })

    except Exception as e:
        log.exception("[ENDPOINT] Unexpected error during OCR processing.")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Health Endpoints ----------------
@app.api_route("/", methods=["GET", "HEAD"], tags=["Health"])
def root():
    return {"status": "ok", "message": "Receipt OCR API is running. Visit /docs for API docs."}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
