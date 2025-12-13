# MLOps Speech SLU (Retail) – Plan

Kerangka tugas besar MLOps untuk SLU retail (produk & quantity) tanpa intent. Fokus pada reproducible pipeline, experiment tracking sederhana, CI, dan API inference (tanpa Docker).

### Kenapa ada Typer?
- Typer adalah library Python untuk bikin CLI dengan type hints sehingga argumen otomatis punya help/validasi.
- Dipakai supaya semua langkah pipeline (ingest → preprocess → train → evaluate → hparam-search) konsisten lewat satu perintah `slu ...`, tanpa mengingat modul/flag terpisah.
- Jalankan `slu --help` atau `slu train --help` untuk melihat opsi yang tersedia.
- Kalau entrypoint `slu` belum terdaftar, jalankan lewat `python -m slu.cli ...` (fungsi sama).

## Dataset & Artefak
- Sumber metadata/label: `metadata-google-tts.csv`, `metadata-google-tts-split.csv`, `label_final_revisi-manual.csv` (taruh di direktori induk repo ini).
- Output ingest: `data/processed/metadata.csv` dan label maps (`product2id.json`, `qty2id.json`).
- Output preprocess: `data/features/*.npy` + `metadata_features.csv`.
- Registry: `artifacts/registry/latest.json` berisi path model, config, metrics.

## Arsitektur Model
- M1 (utama): log-mel → CNN blocks (downsample time) → Transformer encoder kecil → attention pooling → 2 head (product, quantity).
- M2 (pembanding): log-mel → linear/pos-enc → Transformer encoder kecil → attention pooling → 2 head.
- Tujuan: bandingkan manfaat CNN front-end vs murni Transformer di dataset kecil.

## Rencana Pengerjaan (checklist)
1) **Ingestion**
	- Gabungkan metadata sumber, bersihkan duplikat, simpan `data/processed/metadata.csv`.
	- Bangun label map produk/qty, simpan ke `data/processed`.
2) **EDA**
	- Statistik dasar: durasi, sebaran kelas produk/qty, imbalance.
	- Catat keputusan (class weight, sampling) di catatan EDA.
3) **Preprocess**
	- Ekstrak log-mel sesuai `configs/preprocess.yaml` → simpan `.npy` dan `metadata_features.csv`.
	- Validasi durasi min/max, normalisasi konsisten.
4) **Modeling & Training**
	- Implement loop training M1 & M2 (loss CE ganda, class weight opsional, early stopping, scheduler cosine/warmup).
	- Simpan checkpoint + metrics ke `artifacts/`.
5) **Tuning**
	- Grid kecil: lr, dropout, n_layers, d_model, batch_size (lihat `configs/model_*.yaml` + `train.yaml`).
	- Pilih best run berdasar macro-F1 produk (utama) + qty.
6) **Evaluation & Error Analysis**
	- Hitung accuracy/F1 produk & qty, confusion matrix produk.
	- Ambil 5–10 contoh salah prediksi + analisis singkat.
7) **Registry**
	- Simpan artifact final: `model.pt`, config, label_maps, `metrics.json` via `registry.py` → `artifacts/registry/latest.json`.
8) **API Inference (FastAPI)**
	- Endpoint `/health`, `/predict` (upload audio → preprocess → infer → JSON).
	- Logging sederhana: latency, error rate, distribusi confidence.
9) **CI**
	- GitHub Actions: install deps → lint (py_compile/ruff opsional) → pytest → smoke-train kecil (opsional jika waktu cukup).
10) **Monitoring minimal**
	 - Log latency/error/confidence ke `logs/` atau stdout (cukup untuk demo).
11) **PPT Deliverable**
	 - Workflow diagram, ML canvas, screenshot evidence (EDA, run, CI, API test).

## Struktur Repo (ringkas)
```
configs/           # data/preprocess/model/train configs
src/slu/           # ingest, preprocess, eda, train, evaluate, registry, models
api/               # FastAPI app + schema
tests/             # unit + smoke
.github/workflows/ # CI
artifacts/         # checkpoints + registry (output)
data/              # processed/features (output)
```

## Cara Jalan Lokal (baseline)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# (opsional lebih rapi) install editable supaya modul `slu` dikenali
pip install -e .

# 1) Ingest (gabung metadata + label map)
slu ingest --config configs/data.yaml

# 2) Preprocess (extract log-mel)
slu preprocess --metadata data/processed/metadata.csv --config configs/preprocess.yaml

# 3) Train
slu train --model-cfg configs/model_m1.yaml --train-cfg configs/train.yaml

# 4) Evaluate
slu evaluate --metadata-features data/processed/metadata_features.csv

# 5) Optuna search (opsional)
slu hparam-search --n-trials 5

# 6) API
uvicorn api.app:app --reload
```

## Catatan Penting
- Tidak memakai intent label (hanya produk & qty).
- Docker tidak wajib; fokus ke FastAPI + CI + reproducibility.
- Pastikan preprocessing di training dan API identik (pakai config yang sama).
