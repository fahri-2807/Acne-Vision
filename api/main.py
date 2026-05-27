"""
FastAPI — Multi-Model API
Menggabungkan dua model dalam satu server:
  1. acne Level Classification (CNN + TensorFlow)
  2. Skincare Recommendation    (Content-Based + scikit-learn)

Python   : 3.10.9
Struktur :
  main.py
  routers/
    acne.py  
    skincare.py
  models/
    best_model.keras
    skincare_model.pkl 
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import acne, skincare, acne_skin

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# INISIALISASI FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "🧠 Multi-Model API",
    description = (
        "API dengan dua model machine learning:\n\n"
        "### 👥 acne Level Classification\n"
        "Mengklasifikasikan tingkat keparahan jerawat dari gambar (CNN).\n\n"
        "### 💆 Skincare Recommendation\n"
        "Merekomendasikan produk skincare berdasarkan masalah dan jenis kulit.\n\n"
        "### 🔍 Acne + Skincare \n"
        "Deteksi tingkat keparahan jerawat dari foto wajah sekaligus mendapat "
        "rekomendasi produk skincare yang sesuai — dalam satu request.\n\n"
        "---\n"
        "**Python:** 3.10.9 | **TensorFlow:** 2.13.0 | **scikit-learn:** 1.3.2"
    ),
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# DAFTARKAN ROUTER
# ─────────────────────────────────────────────
app.include_router(acne.router,     prefix="/acne",    tags=["acne Level"])
app.include_router(skincare.router,  prefix="/skincare", tags=["Skincare"])
app.include_router(acne_skin.router, prefix="/acne",     tags=["Acne + Skincare"])


# ─────────────────────────────────────────────
# ROOT & HEALTH
# ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """Informasi umum API."""
    return {
        "api"      : "Multi-Model API",
        "version"  : "2.0.0",
        "status"   : "running",
        "modules"  : {
            "acne_skin"    : "Acne Detection + Skincare Recommendation (TERPADU)",
            "acne_level"  : "Acne Level Classification (CNN) — standalone",
            "skincare"     : "Skincare Recommendation — standalone",
        },
        "endpoints": {
            "docs"                      : "/docs",
            "redoc"                     : "/redoc",
            "health"                    : "/health",
            "⭐ acne_analyze"           : "/acne/analyze",
            "⭐ acne_analyze_batch"     : "/acne/analyze/batch",
            "acne_info"                 : "/acne/info",
            "acne_predict"             : "/acne/predict",
            "acne_predict_batch"       : "/acne/predict/batch",
            "skincare_recommend"        : "/skincare/recommend",
            "skincare_skin_types"       : "/skincare/skin-types",
        }
    }


@app.get("/health", tags=["Info"])
def health():
    """Status keseluruhan API dan kedua model."""
    return {
        "status"          : "ok",
        "acne_cnn_model"  : acne_skin.cnn_model is not None,
        "acne_skin_bundle": acne_skin.skin_bundle is not None,
        "acne_model"     : acne.model is not None,
        "skincare_model"  : skincare.model_bundle is not None,
    }

