# 📋 Dokumentasi API — Acne Level Detection & Skincare Recommendation

**Versi:** 2.0.0  
**Base URL:** `https://web-production-e014.up.railway.app`  
**Python:** 3.10.9 | **Framework:** FastAPI 0.104.1  
**Dokumentasi interaktif:** `https://web-production-e014.up.railway.app/docs`

---

## Daftar Isi

1. [Gambaran Umum](#gambaran-umum)
2. [Struktur Project](#struktur-project)
3. [Cara Menjalankan Server](#cara-menjalankan-server)
4. [Endpoint Terpadu ⭐](#endpoint-terpadu)
5. [Endpoint CNN Standalone](#endpoint-cnn-standalone)
6. [Endpoint Skincare Standalone](#endpoint-skincare-standalone)
7. [Endpoint Umum](#endpoint-umum)
8. [Kode Error](#kode-error)
9. [Contoh Integrasi](#contoh-integrasi)

---

## Gambaran Umum

API ini menggabungkan dua model machine learning:

| Model       | File                          | Fungsi                                                           |
| ----------- | ----------------------------- | ---------------------------------------------------------------- |
| CNN NNEW    | `best_model.h5`               | Mendeteksi tingkat keparahan jerawat dari foto wajah (level 0–3) |
| Skincare ML | `skincare_cnn_integrated.pkl` | Merekomendasikan produk skincare berdasarkan level jerawat       |

### Alur utama

```
📸 Foto wajah  →  POST /acne/analyze  →  Level jerawat + Rekomendasi skincare
```

### Tingkat keparahan jerawat

| Level  | Kondisi                   | Saran                                 |
| ------ | ------------------------- | ------------------------------------- |
| `0` 🟢 | Tidak ada / sangat ringan | Maintenance & hidrasi                 |
| `1` 🔵 | Ringan                    | Perawatan rutin                       |
| `2` 🟠 | Sedang                    | Bahan aktif targeted                  |
| `3` 🔴 | Berat                     | Skincare intensif + konsultasi dokter |

---

## Struktur Project

```
project/
├── api/
│   ├── main.py
│   └── routers/
│       ├── __init__.py
│       ├── acne_skin.py     ← endpoint terpadu ⭐
│       ├── acne.py          ← endpoint CNN standalone
│       └── skincare.py      ← endpoint skincare standalone
└── output/
    ├── best_model.h5
    └── skincare_cnn_integrated.pkl
```

---

## Cara Menjalankan Server

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan server

```bash
cd api
uvicorn main:app --reload --port 8000
```

### 3. Verifikasi server berjalan

```bash
curl https://web-production-e014.up.railway.app/
```

Response yang diharapkan:

```json
{
  "api": "Multi-Model API",
  "version": "2.0.0",
  "status": "running",
  "modules": {
    "acne_skin": "Acne Detection + Skincare Recommendation (TERPADU)",
    "acne_level": "Acne Level Classification (CNN) — standalone",
    "skincare": "Skincare Recommendation — standalone"
  }
}
```

> ⚠️ Pastikan model `best_model.h5` dan `skincare_cnn_integrated.pkl` tersedia agar endpoint utama berfungsi.

---

## Endpoint Terpadu

> ### ⭐ Ini endpoint utama yang direkomendasikan untuk dipakai tim web/backend

---

### `POST /acne/analyze`

Mendeteksi tingkat keparahan jerawat dari foto wajah **sekaligus** mengembalikan rekomendasi produk skincare yang sesuai dalam **satu request**.

#### Request

| Parameter     | Tipe      | Lokasi    | Wajib | Default     | Keterangan                             |
| ------------- | --------- | --------- | ----- | ----------- | -------------------------------------- |
| `file`        | `file`    | form-data | ✅    | —           | Foto wajah format JPEG/PNG, maks 10 MB |
| `jenis_kulit` | `string`  | query     | ✅    | `Berminyak` | Jenis kulit pengguna                   |
| `top_n`       | `integer` | query     | ❌    | `5`         | Jumlah rekomendasi yang dikembalikan   |

**Pilihan nilai `jenis_kulit`:**
`Berminyak` · `Kering` · `Normal` · `Kombinasi` · `Sensitif`

#### Contoh request — JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append("file", fotoWajah); // File object dari input[type=file]

const response = await fetch(
  "https://web-production-e014.up.railway.app/acne/analyze?jenis_kulit=Berminyak&top_n=5",
  {
    method: "POST",
    body: formData,
  },
);

const data = await response.json();
```

#### Contoh request — Python (requests)

```python
import requests

with open('foto_wajah.jpg', 'rb') as f:
    response = requests.post(
        'https://web-production-e014.up.railway.app/acne/analyze',
        files={'file': ('foto.jpg', f, 'image/jpeg')},
        params={'jenis_kulit': 'Berminyak', 'top_n': 5}
    )

data = response.json()
```

#### Contoh request — curl

```bash
curl -X POST "https://web-production-e014.up.railway.app/acne/analyze?jenis_kulit=Berminyak&top_n=5" \
     -F "file=@foto_wajah.jpg"
```

#### Response `200 OK`

```json
{
  "filename": "foto_wajah.jpg",
  "inference_time_ms": 135.4,

  "acne": {
    "acne_level": 2,
    "acne_label": "Tingkat 2 — Jerawat sedang",
    "acne_deskripsi": "Jerawat cukup banyak. Perlu bahan aktif yang lebih targeted.",
    "confidence": 0.7812,
    "confidence_pct": "78.12%",
    "probabilities": {
      "Tingkat 0": 0.0321,
      "Tingkat 1": 0.1245,
      "Tingkat 2": 0.7812,
      "Tingkat 3": 0.0622
    },
    "saran_dokter": false
  },

  "jenis_kulit": "Berminyak",
  "total_rekomendasi": 5,

  "rekomendasi": [
    {
      "rank": 1,
      "brand": "Emina",
      "produk": "Niacinamide 10% + Zinc 1%",
      "jenis_produk": "Serum",
      "bahan_aktif": "Niacinamide",
      "untuk_kulit": "Sensitif",
      "level_utama": 2,
      "peringatan": "Aman",
      "skor_dataset": 3,
      "final_score": 0.6241,
      "catatan": "Produk dapat dipertimbangkan untuk jerawat sedang..."
    }
  ]
}
```

#### Penjelasan field response

**Objek `acne`:**

| Field            | Tipe      | Keterangan                                        |
| ---------------- | --------- | ------------------------------------------------- |
| `acne_level`     | `integer` | Tingkat jerawat: `0`, `1`, `2`, atau `3`          |
| `acne_label`     | `string`  | Label deskriptif tingkat jerawat                  |
| `acne_deskripsi` | `string`  | Penjelasan kondisi kulit                          |
| `confidence`     | `float`   | Skor kepercayaan model (0.0–1.0)                  |
| `confidence_pct` | `string`  | Persentase kepercayaan, contoh `"78.12%"`         |
| `probabilities`  | `object`  | Probabilitas untuk setiap tingkat (0–3)           |
| `saran_dokter`   | `boolean` | `true` jika level 3, artinya disarankan ke dokter |

**Tiap objek dalam array `rekomendasi`:**

| Field          | Tipe      | Keterangan                          |
| -------------- | --------- | ----------------------------------- |
| `rank`         | `integer` | Urutan rekomendasi (1 = terbaik)    |
| `brand`        | `string`  | Nama brand produk                   |
| `produk`       | `string`  | Nama produk                         |
| `jenis_produk` | `string`  | Kategori produk (Serum, Toner, dll) |
| `bahan_aktif`  | `string`  | Bahan aktif utama                   |
| `untuk_kulit`  | `string`  | Jenis kulit yang cocok              |
| `level_utama`  | `integer` | Level keparahan jerawat yang dituju |
| `peringatan`   | `string`  | Warning penggunaan produk           |
| `skor_dataset` | `integer` | Skor dari dataset (0–4)             |
| `final_score`  | `float`   | Skor akhir rekomendasi (0.0–1.0)    |
| `catatan`      | `string`  | Catatan penggunaan dari dataset     |

---

### `POST /acne/analyze/batch`

Mendeteksi jerawat dan merekomendasikan skincare untuk **beberapa foto sekaligus**.

#### Request

| Parameter     | Tipe      | Lokasi    | Wajib | Default     | Keterangan                   |
| ------------- | --------- | --------- | ----- | ----------- | ---------------------------- |
| `files`       | `file[]`  | form-data | ✅    | —           | Beberapa foto wajah JPEG/PNG |
| `jenis_kulit` | `string`  | query     | ✅    | `Berminyak` | Berlaku untuk semua foto     |
| `top_n`       | `integer` | query     | ❌    | `3`         | Jumlah rekomendasi per foto  |

#### Contoh request — JavaScript

```javascript
const formData = new FormData();
formData.append("files", foto1);
formData.append("files", foto2);
formData.append("files", foto3);

const response = await fetch(
  "https://web-production-e014.up.railway.app/acne/analyze/batch?jenis_kulit=Kering&top_n=3",
  { method: "POST", body: formData },
);
const data = await response.json();
```

#### Contoh request — curl

```bash
curl -X POST "https://web-production-e014.up.railway.app/acne/analyze/batch?jenis_kulit=Berminyak&top_n=3" \
     -F "files=@foto1.jpg" \
     -F "files=@foto2.jpg"
```

#### Response `200 OK`

```json
{
  "total_images": 2,
  "total_time_ms": 284.7,
  "results": [
    {
      "filename": "foto1.jpg",
      "inference_time_ms": 130.2,
      "acne": { "...": "..." },
      "jenis_kulit": "Berminyak",
      "total_rekomendasi": 3,
      "rekomendasi": ["..."]
    }
  ]
}
```

---

### `GET /acne/info`

Mengembalikan status kedua model dan daftar pilihan input yang valid.

```bash
curl https://web-production-e014.up.railway.app/acne/info
```

```json
{
  "cnn_loaded": true,
  "cnn_path": "../output/best_model.h5",
  "skincare_loaded": true,
  "skincare_path": "../output/skincare_cnn_integrated.pkl",
  "acne_levels": {
    "0": "Tingkat 0 — Tidak ada jerawat / sangat ringan",
    "1": "Tingkat 1 — Jerawat ringan",
    "2": "Tingkat 2 — Jerawat sedang",
    "3": "Tingkat 3 — Jerawat berat"
  },
  "skincare_total_produk": 150,
  "skincare_kulit_options": [
    "Berminyak",
    "Kering",
    "Kombinasi",
    "Normal",
    "Sensitif"
  ],
  "skincare_masalah_options": [
    "Bekas jerawat",
    "Dehidrasi",
    "Flek hitam",
    "Iritasi",
    "Jerawat",
    "Kusam",
    "Penuaan",
    "Pori besar"
  ]
}
```

---

## Endpoint CNN Standalone

> Gunakan endpoint ini jika hanya butuh **deteksi tingkat jerawat saja** tanpa rekomendasi skincare.

---

### `POST /acne/predict`

Prediksi tingkat keparahan jerawat dari satu foto.

#### Request

| Parameter | Tipe   | Lokasi    | Wajib | Keterangan                      |
| --------- | ------ | --------- | ----- | ------------------------------- |
| `file`    | `file` | form-data | ✅    | Foto wajah JPEG/PNG, maks 10 MB |

#### Contoh request

```javascript
const formData = new FormData();
formData.append("file", fotoWajah);

const response = await fetch(
  "https://web-production-e014.up.railway.app/acne/predict",
  {
    method: "POST",
    body: formData,
  },
);
```

#### Response `200 OK`

```json
{
  "filename": "foto.jpg",
  "predicted_class": "Level 2",
  "class_index": 2,
  "confidence": 0.7812,
  "confidence_pct": "78.12%",
  "description": "Kepadatan sedang (count 21–42)",
  "probabilities": {
    "Level 0": 0.0321,
    "Level 1": 0.1245,
    "Level 2": 0.7812,
    "Level 3": 0.0622
  },
  "inference_time_ms": 98.3
}
```

---

### `POST /acne/predict/batch`

Prediksi tingkat jerawat dari beberapa foto sekaligus.

```bash
curl -X POST https://web-production-e014.up.railway.app/acne/predict/batch \
     -F "files=@foto1.jpg" \
     -F "files=@foto2.jpg"
```

---

## Endpoint Skincare Standalone

> Gunakan endpoint ini jika sudah punya level jerawat dan hanya butuh **rekomendasi skincare saja**.

---

### `POST /skincare/recommend`

Rekomendasi produk skincare berdasarkan input manual (tanpa foto).

#### Request Body (JSON)

```json
{
  "masalah_kulit": "Jerawat",
  "jenis_kulit": "Berminyak",
  "bahan_preferred": "Niacinamide",
  "jenis_produk": null,
  "top_n": 5,
  "exclude_warning": false
}
```

| Field             | Tipe      | Wajib | Pilihan nilai                                                                                          |
| ----------------- | --------- | ----- | ------------------------------------------------------------------------------------------------------ |
| `masalah_kulit`   | `string`  | ✅    | Jerawat · Kusam · Dehidrasi · Flek hitam · Iritasi · Bekas jerawat · Pori besar · Penuaan              |
| `jenis_kulit`     | `string`  | ✅    | Berminyak · Kering · Normal · Kombinasi · Sensitif                                                     |
| `bahan_preferred` | `string`  | ❌    | Niacinamide · Ceramide · Tea Tree · Salicylic Acid · Hyaluronic Acid · Vitamin C · Retinol · Bakuchiol |
| `jenis_produk`    | `string`  | ❌    | Serum · Toner · Cleanser · Moisturizer · Sunscreen · Essence · dll                                     |
| `top_n`           | `integer` | ❌    | 1–20 (default `5`)                                                                                     |
| `exclude_warning` | `boolean` | ❌    | `true` = filter produk yang punya peringatan iritasi                                                   |

#### Contoh request — JavaScript

```javascript
const response = await fetch(
  "https://web-production-e014.up.railway.app/skincare/recommend",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      masalah_kulit: "Jerawat",
      jenis_kulit: "Berminyak",
      top_n: 5,
    }),
  },
);
const data = await response.json();
```

---

### `GET /skincare/skin-types`

Daftar semua pilihan input valid.

```bash
curl https://web-production-e014.up.railway.app/skincare/skin-types
```

```json
{
  "masalah_kulit": [
    "Bekas jerawat",
    "Dehidrasi",
    "Flek hitam",
    "Iritasi",
    "Jerawat",
    "Kusam",
    "Penuaan",
    "Pori besar"
  ],
  "jenis_kulit": ["Berminyak", "Kering", "Kombinasi", "Normal", "Sensitif"],
  "bahan_aktif": [
    "Bakuchiol",
    "Ceramide",
    "Hyaluronic Acid",
    "Niacinamide",
    "Retinol",
    "Salicylic Acid",
    "Tea Tree",
    "Vitamin C"
  ],
  "jenis_produk": [
    "Cleanser",
    "Essence",
    "Gel Moisturizer",
    "Moisturizer",
    "Night Cream",
    "Serum",
    "Spot Treatment",
    "Sunscreen",
    "Toner"
  ]
}
```

---

## Endpoint Umum

| Endpoint     | Fungsi                                     |
| ------------ | ------------------------------------------ |
| `GET /`      | Info API dan daftar semua endpoint         |
| `GET /docs`  | Swagger UI — bisa coba langsung di browser |
| `GET /redoc` | Dokumentasi ReDoc                          |

---

## Kode Error

| Kode  | Nama                     | Penyebab                         | Solusi                                    |
| ----- | ------------------------ | -------------------------------- | ----------------------------------------- |
| `200` | OK                       | Request berhasil                 | —                                         |
| `400` | Bad Request              | Tidak ada file yang dikirim      | Pastikan field `file` atau `files` terisi |
| `413` | Request Entity Too Large | File melebihi 10 MB              | Kompres atau resize foto sebelum dikirim  |
| `415` | Unsupported Media Type   | Format file bukan JPEG/PNG       | Konversi file ke JPEG atau PNG            |
| `422` | Unprocessable Entity     | Gambar rusak / input tidak valid | Periksa file dan nilai parameter          |
| `503` | Service Unavailable      | Model belum dimuat               | Periksa path file model, restart server   |

Contoh response error:

```json
{
  "detail": "Format tidak didukung: 'image/webp'. Gunakan JPEG atau PNG."
}
```

---

## Contoh Integrasi

### JavaScript — Upload foto dari form HTML

```javascript
async function analyzeAcne() {
  const file = document.getElementById("foto").files[0];
  const jenisSkin = document.getElementById("kulit").value;

  if (!file) return alert("Pilih foto terlebih dahulu");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(
      `https://web-production-e014.up.railway.app/acne/analyze?jenis_kulit=${jenisSkin}&top_n=5`,
      { method: "POST", body: formData },
    );

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail);
    }

    const data = await response.json();

    console.log("Level jerawat :", data.acne.acne_level);
    console.log("Confidence    :", data.acne.confidence_pct);

    data.rekomendasi.forEach((p) => {
      console.log(`#${p.rank} ${p.brand} — ${p.produk} (${p.bahan_aktif})`);
    });

    if (data.acne.saran_dokter) {
      alert("⚠️ Tingkat jerawat berat. Disarankan konsultasi dokter kulit.");
    }
  } catch (error) {
    console.error("Error:", error.message);
  }
}
```

### Axios (React / Vue / Node.js)

```javascript
import axios from "axios";

const analyzeAcne = async (file, jenisSkin = "Berminyak", topN = 5) => {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await axios.post(
    "https://web-production-e014.up.railway.app/acne/analyze",
    formData,
    {
      params: { jenis_kulit: jenisSkin, top_n: topN },
      headers: { "Content-Type": "multipart/form-data" },
    },
  );

  return data;
};
```

### Python — Backend ke backend

```python
import requests

def analyze_acne(image_path: str, jenis_kulit: str = "Berminyak", top_n: int = 5):
    with open(image_path, "rb") as f:
        response = requests.post(
            "https://web-production-e014.up.railway.app/acne/analyze",
            files={"file": (image_path, f, "image/jpeg")},
            params={"jenis_kulit": jenis_kulit, "top_n": top_n}
        )
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.json()['detail']}")
    return response.json()

hasil = analyze_acne("foto_wajah.jpg", jenis_kulit="Berminyak")
print(f"Level  : {hasil['acne']['acne_level']}")
print(f"Conf   : {hasil['acne']['confidence_pct']}")
for p in hasil['rekomendasi']:
    print(f"  #{p['rank']} {p['brand']} — {p['produk']}")
```

---

## Ringkasan Endpoint

| Method | Endpoint               | Fungsi                                 | Prioritas  |
| ------ | ---------------------- | -------------------------------------- | ---------- |
| `POST` | `/acne/analyze`        | Deteksi jerawat + rekomendasi skincare | ⭐ Utama   |
| `POST` | `/acne/analyze/batch`  | Batch beberapa foto                    | ⭐ Utama   |
| `GET`  | `/acne/info`           | Status model + pilihan input           | Info       |
| `POST` | `/acne/predict`        | Deteksi jerawat saja                   | Opsional   |
| `POST` | `/acne/predict/batch`  | Deteksi jerawat batch                  | Opsional   |
| `POST` | `/skincare/recommend`  | Rekomendasi skincare manual            | Opsional   |
| `GET`  | `/skincare/skin-types` | Daftar pilihan input skincare          | Info       |
| `GET`  | `/`                    | Info API umum                          | Monitoring |
| `GET`  | `/docs`                | Swagger UI                             | Dev tools  |