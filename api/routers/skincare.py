"""
routers/skincare.py
Endpoint Skincare Recommendation (Content-Based + scikit-learn)
Python : 3.10.9

Endpoint:
  POST /skincare/recommend      Rekomendasi produk
  GET  /skincare/skin-types     Daftar pilihan input yang valid
  GET  /skincare/model/info     Info model skincare
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
SKINCARE_MODEL_PATH = os.getenv("SKINCARE_MODEL_PATH", "../output/skincare_cnn_integrated.pkl")

# Bobot scoring
WEIGHTS = {
    'cosine' : 0.30,
    'skor'   : 0.30,
    'level'  : 0.25,
    'kulit'  : 0.15,
}

# ─────────────────────────────────────────────
# LOAD MODEL BUNDLE
# ─────────────────────────────────────────────
model_bundle = None

def load_skincare_model():
    global model_bundle
    if not Path(SKINCARE_MODEL_PATH).exists():
        logger.warning(f"[Skincare] Model tidak ditemukan di '{SKINCARE_MODEL_PATH}'")
        return
    try:
        with open(SKINCARE_MODEL_PATH, "rb") as f:
            model_bundle = pickle.load(f)
        meta = model_bundle.get("metadata", {})
        logger.info(f"[Skincare] Model dimuat: {SKINCARE_MODEL_PATH}")
        logger.info(f"[Skincare] Total produk : {meta.get('total_produk', '?')}")
    except Exception as e:
        logger.error(f"[Skincare] Gagal memuat model: {e}")

load_skincare_model()

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
router = APIRouter()

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class RecommendRequest(BaseModel):
    masalah_kulit   : str
    jenis_kulit     : str
    bahan_preferred : Optional[str] = None
    jenis_produk    : Optional[str] = None
    top_n           : int           = 5
    exclude_warning : bool          = False

    class Config:
        json_schema_extra = {
            "example": {
                "masalah_kulit"   : "Jerawat",
                "jenis_kulit"     : "Berminyak",
                "bahan_preferred" : "Niacinamide",
                "jenis_produk"    : None,
                "top_n"           : 5,
                "exclude_warning" : False,
            }
        }

class ProductResult(BaseModel):
    rank           : int
    brand          : str
    produk         : str
    jenis_produk   : str
    bahan_aktif    : str
    masalah_kulit  : str
    untuk_kulit    : str
    level_utama    : int
    peringatan     : str
    skor_dataset   : int
    final_score    : float
    catatan        : str

class RecommendResponse(BaseModel):
    query          : dict
    total_hasil    : int
    rekomendasi    : List[ProductResult]

class SkincareModelInfo(BaseModel):
    model_loaded    : bool
    model_path      : str
    total_produk    : Optional[int]
    masalah_options : Optional[List[str]]
    kulit_options   : Optional[List[str]]
    bahan_options   : Optional[List[str]]
    produk_options  : Optional[List[str]]

class SkintypeOptions(BaseModel):
    masalah_kulit   : List[str]
    jenis_kulit     : List[str]
    bahan_aktif     : List[str]
    jenis_produk    : List[str]

# ─────────────────────────────────────────────
# CORE INFERENCE
# ─────────────────────────────────────────────
def run_recommendation(
    masalah_kulit   : str,
    jenis_kulit     : str,
    bahan_preferred : Optional[str],
    jenis_produk    : Optional[str],
    top_n           : int,
    exclude_warning : bool,
) -> List[dict]:

    df      = model_bundle["df"]
    tfidf   = model_bundle["tfidf"]
    matrix  = model_bundle["tfidf_matrix"]

    # --- TF-IDF Query -----------------------------------------------
    parts = [masalah_kulit, jenis_kulit]
    if bahan_preferred: parts.append(bahan_preferred)
    if jenis_produk:    parts.append(jenis_produk)
    query_vec = tfidf.transform([" ".join(parts).lower()])
    cos_sim_all = cosine_similarity(query_vec, matrix).flatten()

    # --- Rule-based Score -------------------------------------------
    mask_masalah = (df["Masalah Kulit"].str.lower() == masalah_kulit.lower()).astype(float)
    mask_kulit   = (df["Untuk Kulit"].str.lower() == jenis_kulit.lower()).astype(float)
    mask_bahan   = (df["Tipe_Bahan_Aktif_Final"].str.lower() == (bahan_preferred or "").lower()).astype(float) * 0.3 if bahan_preferred else 0
    mask_produk  = (df["Jenis_Produk_Final"].str.lower() == (jenis_produk or "").lower()).astype(float) * 0.2 if jenis_produk else 0

    rule_score = np.clip(
        mask_masalah * 0.4 + mask_kulit * 0.3 + mask_bahan + mask_produk, 0, 1
    )

    # --- Final Weighted Score ----------------------------------------
    # Untuk endpoint skincare standalone, gunakan mask_masalah sebagai proxy level
    final_score = (
        WEIGHTS['cosine'] * cos_sim_all                +
        WEIGHTS['skor']   * df['skor_norm'].values     +
        WEIGHTS['kulit']  * rule_score                 +
        WEIGHTS['level']  * mask_masalah.values
    )

    result = df.copy()
    result["final_score"] = final_score

    # --- Filter Opsional -----------------------------------------------
    if exclude_warning:
        result = result[result["warning1"].str.contains("Tidak ada|Aman", case=False, na=True)]
    if jenis_produk:
        f2 = result[result["Jenis_Produk_Final"].str.lower() == jenis_produk.lower()]
        if len(f2) >= top_n:
            result = f2

    result = (
        result
        .sort_values("final_score", ascending=False)
        .drop_duplicates(subset=["Brand", "Produk"])
        .head(top_n)
        .reset_index(drop=True)
    )

    return [
        {
            "rank"          : i + 1,
            "brand"         : row["Brand"],
            "produk"        : row["Produk"],
            "jenis_produk"  : row["Jenis_Produk_Final"],
            "bahan_aktif"   : row["Tipe_Bahan_Aktif_Final"],
            "masalah_kulit" : row["Masalah Kulit"],
            "untuk_kulit"   : row["Untuk Kulit"],
            "level_utama"   : int(row["Level_Utama"]),
            "peringatan"    : row["warning1"],
            "skor_dataset"  : int(row["Skor_Rekomendasi"]),
            "final_score"   : round(float(row["final_score"]), 4),
            "catatan"       : str(row["Catatan_Rekomendasi"])[:300],
        }
        for i, (_, row) in enumerate(result.iterrows())
    ]

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):

    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model skincare belum dimuat.")

    # Validasi input terhadap opsi yang tersedia
    meta    = model_bundle.get("metadata", {})
    valid_m = [x.lower() for x in meta.get("masalah_options", [])]
    valid_k = [x.lower() for x in meta.get("kulit_options",   [])]

    if valid_m and req.masalah_kulit.lower() not in valid_m:
        raise HTTPException(
            status_code=422,
            detail=f"Masalah kulit '{req.masalah_kulit}' tidak dikenali. "
                   f"Pilihan: {meta.get('masalah_options', [])}"
        )
    if valid_k and req.jenis_kulit.lower() not in valid_k:
        raise HTTPException(
            status_code=422,
            detail=f"Jenis kulit '{req.jenis_kulit}' tidak dikenali. "
                   f"Pilihan: {meta.get('kulit_options', [])}"
        )

    try:
        hasil = run_recommendation(
            masalah_kulit   = req.masalah_kulit,
            jenis_kulit     = req.jenis_kulit,
            bahan_preferred = req.bahan_preferred,
            jenis_produk    = req.jenis_produk,
            top_n           = req.top_n,
            exclude_warning = req.exclude_warning,
        )
    except Exception as e:
        logger.error(f"[Skincare] Error rekomendasi: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses rekomendasi: {e}")

    logger.info(
        f"[Skincare] {req.masalah_kulit}|{req.jenis_kulit} "
        f"→ {len(hasil)} rekomendasi"
    )

    return RecommendResponse(
        query={
            "masalah_kulit"   : req.masalah_kulit,
            "jenis_kulit"     : req.jenis_kulit,
            "bahan_preferred" : req.bahan_preferred,
            "jenis_produk"    : req.jenis_produk,
            "exclude_warning" : req.exclude_warning,
        },
        total_hasil  = len(hasil),
        rekomendasi  = hasil,
    )


@router.get("/skin-types", response_model=SkintypeOptions)
def skin_types():
    """Daftar semua pilihan input yang valid untuk endpoint /recommend."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model skincare belum dimuat.")

    meta = model_bundle.get("metadata", {})
    return SkintypeOptions(
        masalah_kulit = meta.get("masalah_options", []),
        jenis_kulit   = meta.get("kulit_options",   []),
        bahan_aktif   = meta.get("bahan_options",   []),
        jenis_produk  = meta.get("produk_options",  []),
    )


@router.get("/model/info", response_model=SkincareModelInfo)
def skincare_model_info():
    """Informasi model rekomendasi skincare."""
    if model_bundle is None:
        return SkincareModelInfo(
            model_loaded=False, model_path=SKINCARE_MODEL_PATH,
            total_produk=None, masalah_options=None,
            kulit_options=None, bahan_options=None, produk_options=None,
        )
    meta = model_bundle.get("metadata", {})
    return SkincareModelInfo(
        model_loaded    = True,
        model_path      = SKINCARE_MODEL_PATH,
        total_produk    = meta.get("total_produk"),
        masalah_options = meta.get("masalah_options"),
        kulit_options   = meta.get("kulit_options"),
        bahan_options   = meta.get("bahan_options"),
        produk_options  = meta.get("produk_options"),
    )