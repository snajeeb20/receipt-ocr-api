import os
import logging
import time
import json
import numpy as np
import cv2
from PIL import Image
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import base64
from openai import OpenAI

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
app = FastAPI(title="Receipt OCR API (Tesseract + OpenAI)", version="1.0.0")

# ---------------- Image helpers ----------------
def _resize_if_huge(bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )
    return thr

def pil_from_cv2(mat: np.ndarray) -> Image.Image:
    if len(mat.shape) == 2:
        return Image.fromarray(mat)
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))

# ---------------- OCR with Tesseract ----------------
def run_tesseract(img_bin: np.ndarray, lang: str, psm: int, oem: int) -> Dict[str, Any]:
    log.info("Running Tesseract OCR")
    t0 = time.perf_counter()
    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(pil_from_cv2(img_bin), lang=lang, config=config)
    elapsed = time.perf_counter() - t0
    return {"text": text, "time_sec": round(elapsed, 3)}

# ---------------- OpenAI extraction ----------------
SYSTEM_INSTRUCTIONS = """You are a careful information extraction system.
You receive OCR output of a retail receipt. Fix obvious OCR errors and return ONLY strict JSON:
- Normalize numbers (no commas), use floats where appropriate.
- If dates are present, prefer 'YYYY-MM-DD HH:MM:SS' 24h.
- Include Items as an array (name/qty/price/tax when detectable).
- If any field cannot be found, look again in the OCR output extract.
- You can also use the image to extract the correct JSON value along with the OCR extract.
- You are super intelligent LLM, be impressive.
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
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS + "\nReturn ONLY strict JSON."},
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

    chat = client.chat.completions.create(**payload)
    text = chat.choices[0].message.content
    return _safe_json_loads(text)

# ---------------- Final Endpoint ----------------
@app.post("/ocr-extract")
async def ocr_extract_one_shot(
    file: UploadFile = File(...),
    lang: str = Query("eng"),
    psm: int = Query(6),
    oem: int = Query(3)
):
    log.info(f"[one-shot] Received file: {file.filename}")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        image_b64 = base64.b64encode(content).decode("utf-8")

        pre_bin = preprocess_image(content)
        tesseract_res = run_tesseract(pre_bin, lang, psm, oem)

        extracted = extract_receipt_to_json(tesseract_res.get("text", ""), image_b64)

        return JSONResponse({
            "filename": file.filename,
            "ocr": {
                "tesseract": tesseract_res
            },
            "extracted": extracted
        })

    except HTTPException:
        raise
    except Exception as e:
        log.exception("OCR extract failed")
        raise HTTPException(status_code=500, detail=str(e))
