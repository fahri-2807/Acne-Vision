"""
routers/acne.py
Endpoint acne Level Classification (CNN + TensorFlow)
Python : 3.10.9

Endpoint:
  POST /acne/predict           Prediksi satu gambar
  POST /acne/predict/batch     Prediksi banyak gambar
  GET  /acne/model/info        Info model CNN
"""

import os
import io
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from pydantic import BaseModel

import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
import os
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingModuleSource]

# 1. Cari lokasi file main.py saat ini (ada di /app/api/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Naik satu tingkat ke folder utama (/app/)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 3. Gabungkan dengan folder output yang benar
MODEL_PATH = os.path.join(BASE_DIR, "output", "best_model.keras")

print(f"JALUR MODEL YANG BENAR: {MODEL_PATH}")

# Contoh pemuatan model keras
try:
    model = load_model(MODEL_PATH)
    acne_cnn_model_status = True
except Exception as e:
    print(f"Gagal memuat model: {e}")
    acne_cnn_model_status = False
    
IMG_SIZE      = (224, 224)
CLASS_NAMES   = ['Level 0', 'Level 1', 'Level 2', 'Level 3']
CLASS_DESC    = {
    'Level 0': 'Kulit normal atau sangat ringan',
    'Level 1': 'Jerawat ringan',
    'Level 2': 'Jerawat sedang',
    'Level 3': 'Jerawat berat',
}
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# ---------------- INFO MODEL CNN ----------------
print("MODEL PATH:", MODEL_PATH)
if os.path.exists(MODEL_PATH):
    print("MODEL SIZE:", os.path.getsize(MODEL_PATH), "bytes")
else:
    print("MODEL NOT FOUND")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = None

def load_acne_model():
    global model
    if not Path(MODEL_PATH).exists():
        logger.warning(f"[acne] Model tidak ditemukan di '{MODEL_PATH}'")
        return
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info(f"[acne] Model dimuat: {MODEL_PATH}")
        logger.info(f"[acne] Input  : {model.input_shape}")
        logger.info(f"[acne] Output : {model.output_shape}")
    except Exception as e:
        logger.error(f"[acne] Gagal memuat model: {e}")

load_acne_model()

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
router = APIRouter()

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PredictionResult(BaseModel):
    filename          : str
    predicted_class   : str
    class_index       : int
    confidence        : float
    confidence_pct    : str
    description       : str
    probabilities     : dict
    inference_time_ms : float

class BatchPredictionResponse(BaseModel):
    total_images  : int
    results       : List[PredictionResult]
    total_time_ms : float

class acneModelInfo(BaseModel):
    model_loaded  : bool
    model_path    : str
    input_shape   : Optional[str]
    output_shape  : Optional[str]
    num_classes   : int
    class_names   : List[str]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Mengubah gambar menjadi format tensor yang kompatibel dengan EfficientNetB0."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    
    # Konversi ke float32 tanpa dibagi 255.0 secara manual karena preprocess_input
    # dari EfficientNet menerima nilai rentang 0-255 lalu menyesuaikannya secara internal
    arr = np.array(img, dtype=np.float32)
    
    # Gunakan preprocessing bawaan dari Keras EfficientNet
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def run_inference(tensor: np.ndarray) -> tuple:
    t0    = time.perf_counter()
    probs = model.predict(tensor, verbose=0)[0]
    ms    = (time.perf_counter() - t0) * 1000
    return int(np.argmax(probs)), probs, round(ms, 2)

def validate_file(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipe file tidak didukung: '{file.content_type}'. Gunakan JPEG atau PNG."
        )

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@router.get("/model/info", response_model=acneModelInfo)
def acne_model_info():
    """Informasi model CNN acne level."""
    return acneModelInfo(
        model_loaded  = model is not None,
        model_path    = MODEL_PATH,
        input_shape   = str(model.input_shape)  if model else None,
        output_shape  = str(model.output_shape) if model else None,
        num_classes   = len(CLASS_NAMES),
        class_names   = CLASS_NAMES,
    )


@router.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(..., description="File gambar (JPEG/PNG)")):
    """
    Prediksi tingkat keparahan dari **satu gambar**.

    - Format: JPEG atau PNG, maks 10 MB
    - Output: kelas prediksi + probabilitas semua kelas
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model CNN belum dimuat.")

    validate_file(file)
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran file melebihi 10 MB.")

    try:
        tensor          = preprocess_image(image_bytes)
        idx, probs, ms  = run_inference(tensor)
    except Exception as e:
        logger.error(f"[acne] Error '{file.filename}': {e}")
        raise HTTPException(status_code=422, detail=f"Gagal memproses gambar: {e}")

    logger.info(f"[acne] '{file.filename}' → {CLASS_NAMES[idx]} ({probs[idx]*100:.1f}%) [{ms}ms]")

    return PredictionResult(
        filename          = file.filename,
        predicted_class   = CLASS_NAMES[idx],
        class_index       = idx,
        confidence        = round(float(probs[idx]), 4),
        confidence_pct    = f"{probs[idx]*100:.2f}%",
        description       = CLASS_DESC[CLASS_NAMES[idx]],
        probabilities     = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
        inference_time_ms = ms,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Prediksi tingkat keparahan dari **banyak gambar** sekaligus."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model CNN belum dimuat.")
    if not files:
        raise HTTPException(status_code=400, detail="Tidak ada file yang dikirim.")

    t_start = time.perf_counter()
    results = []

    for file in files:
        validate_file(file)
        image_bytes = await file.read()
        try:
            tensor         = preprocess_image(image_bytes)
            idx, probs, ms = run_inference(tensor)
            results.append(PredictionResult(
                filename          = file.filename,
                predicted_class   = CLASS_NAMES[idx],
                class_index       = idx,
                confidence        = round(float(probs[idx]), 4),
                confidence_pct    = f"{probs[idx]*100:.2f}%",
                description       = CLASS_DESC[CLASS_NAMES[idx]],
                probabilities     = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
                inference_time_ms = ms,
            ))
        except Exception as e:
            logger.warning(f"[acne] Gagal '{file.filename}': {e}")
            results.append(PredictionResult(
                filename=file.filename, predicted_class="ERROR",
                class_index=-1, confidence=0.0, confidence_pct="0%",
                description=str(e), probabilities={}, inference_time_ms=0.0,
            ))

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return BatchPredictionResponse(
        total_images=len(files), results=results, total_time_ms=total_ms
    )