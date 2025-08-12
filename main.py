import os
# Windows CPU + BLAS env guards
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import json
import numpy as np
import cv2
from PIL import Image
import pytesseract
from typing import Optional, Dict, Any
import base64

# OpenAI SDK
from openai import OpenAI

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("receipt-ocr")

# ---------------- Env helpers ----------------
def _clean_env(val: str | None) -> str | None:
    """Trim spaces, strip quotes, drop accidental trailing backslash."""
    if val is None:
        return None
    v = val.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    if v.endswith("\\"):  # common paste bug on Windows
        v = v[:-1]
    return v or None

def _mask(k: str | None) -> str:
    if not k:
        return "(missing)"
    return k[:4] + "..." + k[-4:]

OPENAI_API_KEY = _clean_env(os.getenv("OPENAI_API_KEY"))
# Pick a broadly supported chat model
OPENAI_MODEL   = _clean_env(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"

log.info(f"OpenAI model: {OPENAI_MODEL}")
log.info(f"OpenAI key tail (masked): {_mask(OPENAI_API_KEY)}")
if OPENAI_API_KEY and OPENAI_API_KEY.endswith("\\"):
    log.warning("OPENAI_API_KEY ends with a backslash — this will break auth!")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- Version helpers ----------------
try:
    from importlib.metadata import version as _pkgver
except Exception:
    from importlib_metadata import version as _pkgver  # py<3.8 backport

def _safe_ver(pkg: str) -> str:
    try:
        return _pkgver(pkg)
    except Exception:
        return "unknown"

# ---------------- Optional EasyOCR ----------------
EASYOCR_AVAILABLE = False
EASYOCR_VER: Optional[str] = None
EASY_READER = None
EASY_INIT_ERROR: Optional[str] = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    EASYOCR_VER = _safe_ver("easyocr") or getattr(easyocr, "__version__", "unknown")
except ImportError as e:
    EASY_INIT_ERROR = f"easyocr import error: {e}"

# ---------------- FastAPI app ----------------
app = FastAPI(title="Receipt OCR API (Tesseract + EasyOCR + OpenAI)", version="1.1.0")

# ---------------- Startup checks ----------------
@app.on_event("startup")
async def startup_checks():
    log.info("==== Startup: environment & engine checks ====")
    log.info(f"OpenCV: {getattr(cv2, '__version__', 'unknown')}")
    log.info(f"Pillow: {_safe_ver('Pillow')}")
    log.info(f"pytesseract: {getattr(pytesseract, '__version__', 'unknown')}")
    log.info(f"openai sdk: {_safe_ver('openai')} (model={OPENAI_MODEL})")

    try:
        tesseract_path = getattr(pytesseract.pytesseract, "tesseract_cmd", "auto(PATH)")
        tesseract_ver = str(pytesseract.get_tesseract_version())
    except Exception as e:
        tesseract_path = "not found"
        tesseract_ver = f"error: {e}"
    log.info(f"Tesseract binary: {tesseract_path}")
    log.info(f"Tesseract version: {tesseract_ver}")
    log.info(f"TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX', '(not set)')}")

    global EASY_READER, EASY_INIT_ERROR
    log.info(f"EasyOCR installed: {EASYOCR_AVAILABLE} (version: {EASYOCR_VER})")
    if EASYOCR_AVAILABLE and EASY_READER is None:
        try:
            t0 = time.perf_counter()
            EASY_READER = easyocr.Reader(['en'], gpu=False)
            t1 = time.perf_counter()
            log.info(f"EasyOCR initialized in {(t1 - t0):.3f}s")
        except Exception as e:
            EASY_INIT_ERROR = f"easyocr init error: {e}"
            log.exception("EasyOCR init failed")

    log.info("==== Startup checks complete ====")

# ---------------- Image helpers ----------------
def _resize_if_huge(bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    resized = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    log.info(f"Resized image from {w}x{h} to {resized.shape[1]}x{resized.shape[0]}")
    return resized

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

# ---------------- OCR engines ----------------
def run_tesseract(img_bin: np.ndarray, lang: str, psm: int, oem: int) -> Dict[str, Any]:
    log.info("Running Tesseract OCR")
    t0 = time.perf_counter()
    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(pil_from_cv2(img_bin), lang=lang, config=config)
    elapsed = time.perf_counter() - t0
    log.info(f"Tesseract finished in {elapsed:.3f}s")
    return {"text": text, "time_sec": round(elapsed, 3)}

def run_easyocr(original_bgr: np.ndarray, lang: str) -> Dict[str, Any]:
    if not EASYOCR_AVAILABLE:
        return {"text": "EasyOCR not installed", "time_sec": 0.0}
    if EASY_READER is None:
        return {"text": EASY_INIT_ERROR or "EasyOCR not initialized", "time_sec": 0.0}
    log.info("Running EasyOCR (cached reader)")
    t0 = time.perf_counter()
    results = EASY_READER.readtext(original_bgr)
    text = "\n".join([t for _, t, _ in results])
    elapsed = time.perf_counter() - t0
    log.info(f"EasyOCR finished in {elapsed:.3f}s")
    return {"text": text, "time_sec": round(elapsed, 3)}

# ---------------- OpenAI extraction ----------------
SYSTEM_INSTRUCTIONS = """You are a careful information extraction system.
You receive two OCR outputs of the same retail receipt. Merge both sources,
fixing obvious OCR errors, and return ONLY strict JSON:
- Normalize numbers (no commas), use floats where appropriate.
- If dates are present, prefer 'YYYY-MM-DD HH:MM:SS' 24h.
- Include Items as an array (name/qty/price/tax when detectable).
- If any field cannot be found, look again in the extract you will find it in either one of the OCR output extract.
- You can also use the image to extract the correct JSON value along with the extracts of the OCR models.
- You are super intelligent LLM, be impressive.
- Return ONLY the JSON per the provided schema—no extra text.
"""

def _build_user_content(ocr_a: str, ocr_b: str) -> str:
    def _clip(s: str, limit: int = 5000) -> str:
        if not s:
            return ""
        return s if len(s) <= limit else (s[:limit] + "\n...[TRUNCATED]...")
    return (
        "OCR A (Tesseract):\n" + _clip(ocr_a) +
        "\n\n---\n\nOCR B (EasyOCR):\n" + _clip(ocr_b) + "\n"
    )

def _safe_json_loads(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        cleaned = txt.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
        return json.loads(cleaned)

def _raise_with_body(e: Exception):
    err_resp = getattr(e, "response", None)
    body = None
    if err_resp is not None:
        try:
            body = err_resp.json()
        except Exception:
            try:
                body = err_resp.text
            except Exception:
                body = None
    raise RuntimeError(f"OpenAI request failed: {getattr(e, 'message', str(e))} | Body: {body}") from e

def _chat_call(payload: Dict[str, Any]):
    try:
        return client.chat.completions.create(**payload)
    except Exception as e:
        _raise_with_body(e)

def extract_receipt_to_json(ocr_a: str, ocr_b: str, image_b64: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    user_text = _build_user_content(ocr_a, ocr_b)

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS + "\nReturn ONLY strict JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        chat = _chat_call(payload)
        text = chat.choices[0].message.content
        return _safe_json_loads(text)
    except Exception as e_primary:
        log.warning(f"Primary json_object call failed; trying fallback. Detail: {e_primary}")

        payload_fallback = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS + "\nReturn ONLY a single valid JSON object."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }
        try:
            chat = _chat_call(payload_fallback)
            text = chat.choices[0].message.content
            return _safe_json_loads(text)
        except Exception as e_fb:
            _raise_with_body(e_fb)

# ---------------- Final Endpoint ----------------
@app.post(
    "/ocr-extract",
    summary="One-shot: OCR (Tesseract + EasyOCR + OpenAI) -> structured JSON with image"
)
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

        original = np.frombuffer(content, np.uint8)
        original = cv2.imdecode(original, cv2.IMREAD_COLOR)
        if original is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        pre_bin  = preprocess_image(content)
        original = _resize_if_huge(original, max_side=1024)

        tesseract_res = run_tesseract(pre_bin, lang, psm, oem)
        easyocr_res   = run_easyocr(original, lang)

        ocr_a = tesseract_res.get("text", "") or ""
        ocr_b = easyocr_res.get("text", "") or ""
        extracted = extract_receipt_to_json(ocr_a, ocr_b, image_b64)

        return JSONResponse({
            "filename": file.filename,
            "ocr": {
                "tesseract": tesseract_res,
                "easyocr": easyocr_res
            },
            "extracted": extracted
        })

    except HTTPException:
        raise
    except Exception as e:
        log.exception("one-shot /ocr-extract failed")
        raise HTTPException(status_code=500, detail=str(e))
