"""
routers/acne_skin.py
Endpoint TERPADU: Deteksi Tingkat Keparahan Jerawat + Rekomendasi Skincare
Python : 3.10.9
 
Alur:
  Gambar → CNN (NNEW Acne Level 0-3) → Profil skincare → Rekomendasi produk

Endpoint:
  POST /acne/analyze             Deteksi jerawat + rekomendasi sekaligus
  POST /acne/analyze/batch       Banyak gambar sekaligus
  GET  /acne/info                Info kedua model
"""

import os
import io
import time
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
CNN_MODEL_PATH      = os.getenv("CNN_MODEL_PATH",      "../output/best_model.keras")
SKINCARE_MODEL_PATH = os.getenv("SKINCARE_MODEL_PATH", "../output/skincare_cnn_integrated.pkl")

IMG_SIZE      = (224, 224)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB

# ═══════════════════════════════════════════════════════════
# MAPPING CNN LEVEL → PROFIL SKINCARE
# ═══════════════════════════════════════════════════════════
ACNE_LEVEL_INFO = {
    0: {
        'label'             : 'Level 0 — Kulit Normal atau Sangat Ringan',
        'deskripsi'         : 'Kulit normal atau sangat ringan. Fokus pada maintenance, hidrasi, dan pencegahan.',
        'skincare_levels'   : [0],
        'masalah_default'   : ['Dehidrasi', 'Kusam', 'Penuaan'],
        'bahan_rekomendasi' : ['Hyaluronic Acid', 'Ceramide', 'Bakuchiol'],
        'warna'             : '#55A868',
        'emoji'             : '🟢',
        'saran_dokter'      : False,
    },
    1: {
        'label'             : 'Level 1 — Jerawat Ringan',
        'deskripsi'         : 'Jerawat ringan. Perlu skincare rutin dengan bahan aktif ringan.',
        'skincare_levels'   : [1],
        'masalah_default'   : ['Jerawat', 'Pori besar', 'Flek hitam'],
        'bahan_rekomendasi' : ['Niacinamide', 'Tea Tree', 'Salicylic Acid'],
        'warna'             : '#4C72B0',
        'emoji'             : '🔵',
        'saran_dokter'      : False,
    },
    2: {
        'label'             : 'Level 2 — Jerawat Sedang',
        'deskripsi'         : 'Masalah kulit sedang. Butuh bahan aktif yang lebih targeted dan konsisten.',
        'skincare_levels'   : [2],
        'masalah_default'   : ['Jerawat', 'Bekas jerawat', 'Iritasi'],
        'bahan_rekomendasi' : ['Niacinamide', 'Ceramide', 'Tea Tree', 'Salicylic Acid'],
        'warna'             : '#DD8452',
        'emoji'             : '🟠',
        'saran_dokter'      : False,
    },
    3: {
        'label'             : 'Level 3 — Jerawat Berat',
        'deskripsi'         : 'Masalah kulit berat. Perlu skincare intensif. Disarankan konsultasi dokter kulit.',
        'skincare_levels'   : [3],
        'masalah_default'   : ['Jerawat', 'Iritasi', 'Bekas jerawat'],
        'bahan_rekomendasi' : ['Ceramide', 'Niacinamide'],
        'warna'             : '#C44E52',
        'emoji'             : '🔴',
        'saran_dokter'      : True,
    },
}


WEIGHTS = {
    'cosine': 0.30,  
    'skor'  : 0.30, 
    'level' : 0.25,  
    'kulit' : 0.15,  
}

# ─────────────────────────────────────────────
# LOAD KEDUA MODEL
# ─────────────────────────────────────────────
cnn_model     = None
skin_bundle   = None

def load_models():
    global cnn_model, skin_bundle

    # Load CNN
    if Path(CNN_MODEL_PATH).exists():
        try:
            cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
            logger.info(f"[Acne-Skin] CNN dimuat     : {CNN_MODEL_PATH}")
        except Exception as e:
            logger.error(f"[Acne-Skin] Gagal load CNN : {e}")
    else:
        logger.warning(f"[Acne-Skin] CNN tidak ditemukan di '{CNN_MODEL_PATH}'")

    # Load Skincare bundle
    if Path(SKINCARE_MODEL_PATH).exists():
        try:
            with open(SKINCARE_MODEL_PATH, "rb") as f:
                skin_bundle = pickle.load(f)
            n = skin_bundle.get("metadata", {}).get("total_produk", "?")
            logger.info(f"[Acne-Skin] Skincare dimuat : {SKINCARE_MODEL_PATH} ({n} produk)")
        except Exception as e:
            logger.error(f"[Acne-Skin] Gagal load skincare : {e}")
    else:
        logger.warning(f"[Acne-Skin] Skincare tidak ditemukan di '{SKINCARE_MODEL_PATH}'")

load_models()

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
router = APIRouter()

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class AcneDetectionResult(BaseModel):
    acne_level       : int
    acne_label       : str
    acne_deskripsi   : str
    confidence       : float
    confidence_pct   : str
    probabilities    : dict   # {"Tingkat 0": 0.12, ...}
    saran_dokter     : bool

class SkincareProduct(BaseModel):
    rank          : int
    brand         : str
    produk        : str
    jenis_produk  : str
    bahan_aktif   : str
    untuk_kulit   : str
    level_utama   : int
    peringatan    : str
    skor_dataset  : int
    final_score   : float
    catatan       : str

class AnalyzeResponse(BaseModel):
    filename         : str
    inference_time_ms: float
    # Hasil deteksi jerawat
    acne             : AcneDetectionResult
    # Hasil rekomendasi skincare
    jenis_kulit      : str
    total_rekomendasi: int
    rekomendasi      : List[SkincareProduct]

class BatchAnalyzeResponse(BaseModel):
    total_images  : int
    total_time_ms : float
    results       : List[AnalyzeResponse]

class AcneModelInfo(BaseModel):
    cnn_loaded          : bool
    cnn_path            : str
    skincare_loaded     : bool
    skincare_path       : str
    acne_levels         : dict
    skincare_total_produk: Optional[int]
    skincare_kulit_options: Optional[List[str]]
    skincare_masalah_options: Optional[List[str]]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Bytes → numpy array (1, 224, 224, 3) siap untuk EfficientNetB0 CNN."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    
    # Menyesuaikan dengan standar input layer EfficientNetB0
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def run_cnn(tensor: np.ndarray) -> tuple:
    """Jalankan CNN → (level, probabilities, time_ms)."""
    t0    = time.perf_counter()
    probs = cnn_model.predict(tensor, verbose=0)[0]
    ms    = round((time.perf_counter() - t0) * 1000, 2)
    return int(np.argmax(probs)), probs, ms


def run_skincare(acne_level: int, jenis_kulit: str, top_n: int = 5) -> dict:

    profile_map = skin_bundle.get("cnn_to_skincare_profile", {})
    df          = skin_bundle["df"]
    tfidf       = skin_bundle["tfidf"]
    matrix      = skin_bundle["tfidf_matrix"]

    # Ambil profil dari mapping berdasarkan CNN level
    profile       = profile_map.get(acne_level, profile_map.get(0))
    target_levels = profile.get("skincare_levels", [0])
    masalah_list  = profile.get("masalah_default", ["Jerawat"])
    bahan_list    = profile.get("bahan_rekomendasi", ["Ceramide"])

    # ── Filter dataset berdasarkan Level_Utama ────────────────
    df_filtered = df[df["Level_Utama"].isin(target_levels)].copy()

    # Jika hasil filter terlalu sedikit, gunakan semua data
    if len(df_filtered) < top_n:
        df_filtered = df.copy()

    # ── TF-IDF Cosine Similarity ──────────────────────────────
    query_parts = masalah_list + [jenis_kulit] + bahan_list
    query_text  = " ".join(query_parts).lower()
    query_vec   = tfidf.transform([query_text])

    # Hitung cosine similarity hanya untuk baris yang difilter
    filtered_idx = df_filtered.index
    cos_sim_all  = cosine_similarity(query_vec, matrix).flatten()
    cos_sim      = cos_sim_all[filtered_idx]

    # ── Rule-based Score ──────────────────────────────────────
    mask_masalah = df_filtered["Masalah Kulit"].isin(masalah_list).astype(float)
    mask_kulit   = (df_filtered["Untuk Kulit"].str.lower() == jenis_kulit.lower()).astype(float)
    mask_bahan   = df_filtered["Tipe_Bahan_Aktif_Final"].isin(bahan_list).astype(float) * 0.3
    mask_level   = df_filtered["Level_Utama"].isin(target_levels).astype(float)

    rule_score = np.clip(
        mask_masalah * 0.4 + mask_kulit * 0.3 + mask_bahan, 0, 1
    )

    # ── Final Weighted Score ──────────────────────────────────
    final_score = (
        WEIGHTS["cosine"] * cos_sim                           +
        WEIGHTS["skor"]   * df_filtered["skor_norm"].values   +
        WEIGHTS["level"]  * mask_level.values                 +
        WEIGHTS["kulit"]  * rule_score.values
    )

    result = df_filtered.copy()
    result["cosine_sim"]  = cos_sim
    result["rule_score"]  = rule_score.values
    result["final_score"] = final_score

    result = (
        result
        .sort_values("final_score", ascending=False)
        .drop_duplicates(subset=["Brand", "Produk"])
        .head(top_n)
        .reset_index(drop=True)
    )
    result.index += 1

    # Format output
    rekomendasi = [
        {
            "rank"        : i,
            "brand"       : row["Brand"],
            "produk"      : row["Produk"],
            "jenis_produk": row["Jenis_Produk_Final"],
            "bahan_aktif" : row["Tipe_Bahan_Aktif_Final"],
            "untuk_kulit" : row["Untuk Kulit"],
            "level_utama" : int(row["Level_Utama"]),
            "peringatan"  : row["warning1"],
            "skor_dataset": int(row["Skor_Rekomendasi"]),
            "final_score" : round(float(row["final_score"]), 4),
            "catatan"     : str(row["Catatan_Rekomendasi"])[:300],
        }
        for i, (_, row) in enumerate(result.iterrows(), 1)
    ]

    return {
        "cnn_level"       : acne_level,
        "profile"         : profile,
        "masalah_dipakai" : masalah_list,
        "bahan_dipakai"   : bahan_list,
        "jenis_kulit"     : jenis_kulit,
        "rekomendasi"     : rekomendasi,
    }


def validate_file(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Format tidak didukung: '{file.content_type}'. Gunakan JPEG atau PNG."
        )


def build_response(filename: str, image_bytes: bytes,
                   jenis_kulit: str, top_n: int) -> AnalyzeResponse:
    """Jalankan full pipeline untuk satu gambar dan kembalikan AnalyzeResponse."""
    t_start = time.perf_counter()

    # Step 1: CNN
    tensor             = preprocess_image(image_bytes)
    level, probs, _    = run_cnn(tensor)
    info               = ACNE_LEVEL_INFO[level]

    # Step 2: Skincare
    skincare_result = run_skincare(level, jenis_kulit, top_n)
    rekomendasi     = skincare_result["rekomendasi"]

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    level_names = {0: "Level 0", 1: "Level 1", 2: "Level 2", 3: "Level 3"}

    return AnalyzeResponse(
        filename          = filename,
        inference_time_ms = total_ms,
        acne = AcneDetectionResult(
            acne_level     = level,
            acne_label     = info["label"],
            acne_deskripsi = info["deskripsi"],
            confidence     = round(float(probs[level]), 4),
            confidence_pct = f"{probs[level]*100:.2f}%",
            probabilities  = {level_names[i]: round(float(p), 4) for i, p in enumerate(probs)},
            saran_dokter   = info["saran_dokter"],
        ),
        jenis_kulit       = jenis_kulit,
        total_rekomendasi = len(rekomendasi),
        rekomendasi       = rekomendasi,
    )

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file       : UploadFile = File(..., description="Foto wajah (JPEG/PNG)"),
    jenis_kulit: str        = "Berminyak",
    top_n      : int        = 5,
):

    if cnn_model is None:
        raise HTTPException(503, "Model CNN belum dimuat. Pastikan best_model.h5 tersedia.")
    if skin_bundle is None:
        raise HTTPException(503, "Model skincare belum dimuat. Pastikan skincare_cnn_integrated.pkl tersedia.")

    validate_file(file)
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, "Ukuran file melebihi 10 MB.")

    try:
        result = build_response(file.filename, image_bytes, jenis_kulit, top_n)
    except Exception as e:
        logger.error(f"[Acne-Skin] Error '{file.filename}': {e}")
        raise HTTPException(422, f"Gagal memproses gambar: {e}")

    logger.info(
        f"[Acne-Skin] '{file.filename}' → Level {result.acne.acne_level} "
        f"({result.acne.confidence_pct}) | {len(result.rekomendasi)} rekomendasi "
        f"[{result.inference_time_ms}ms]"
    )
    return result


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(
    files      : List[UploadFile] = File(..., description="Beberapa foto wajah"),
    jenis_kulit: str              = "Berminyak",
    top_n      : int              = 3,
):

    if cnn_model is None:
        raise HTTPException(503, "Model CNN belum dimuat.")
    if skin_bundle is None:
        raise HTTPException(503, "Model skincare belum dimuat.")
    if not files:
        raise HTTPException(400, "Tidak ada file yang dikirim.")

    t_start = time.perf_counter()
    results = []

    for file in files:
        validate_file(file)
        image_bytes = await file.read()
        try:
            r = build_response(file.filename, image_bytes, jenis_kulit, top_n)
            results.append(r)
        except Exception as e:
            logger.warning(f"[Acne-Skin] Gagal '{file.filename}': {e}")

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(f"[Acne-Skin] Batch {len(files)} gambar selesai [{total_ms}ms]")

    return BatchAnalyzeResponse(
        total_images  = len(files),
        total_time_ms = total_ms,
        results       = results,
    )


@router.get("/info", response_model=AcneModelInfo)
def acne_info():
    """Status dan informasi kedua model yang digunakan endpoint ini."""
    meta = skin_bundle.get("metadata", {}) if skin_bundle else {}
    return AcneModelInfo(
        cnn_loaded               = cnn_model is not None,
        cnn_path                 = CNN_MODEL_PATH,
        skincare_loaded          = skin_bundle is not None,
        skincare_path            = SKINCARE_MODEL_PATH,
        acne_levels              = {
            str(k): v["label"] for k, v in ACNE_LEVEL_INFO.items()
        },
        skincare_total_produk    = meta.get("total_produk"),
        skincare_kulit_options   = meta.get("kulit_options"),
        skincare_masalah_options = meta.get("masalah_options"),
    )
