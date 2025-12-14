# MLOps Speech SLU (Retail) — Kelompok 2 RC

Spoken Language Understanding untuk retail (prediksi `product` & `quantity`) dengan log-mel → CNN+Transformer (M1) atau Transformer-tiny (M2). Proyek Tugas Besar MLOps: data pipeline, reproducible training, Optuna tuning, registry, dan FastAPI inference.

## Ringkas Hasil
- Model default: **M1 best** (`artifacts/m1_best/best_model.pt`, config `configs/model_m1.yaml`, train cfg `configs/train_m1_best.yaml`).
- Metrics (val): acc_product 0.68, acc_quantity 0.71, f1_product 0.676, f1_quantity 0.704, val_loss 1.573.
- Registry: `artifacts/registry/latest.json` menunjuk ke M1 best (label maps di `data/processed/product2id.json`, `data/processed/qty2id.json`).
- Optuna study & plot: `optuna.db`, `artifacts/optuna-hyperparameter.png`, `artifacts/m1/optuna_*.html`.

## Stack
- Python, PyTorch, FastAPI, Optuna, librosa.
- CLI via Typer (`python -m slu.cli ...`).
- Logging inference ke `logs/inference.log`.

## Struktur Singkat
```
configs/            # data/preprocess/model/train configs
src/slu/            # ingest, preprocess, train, eval, registry, models
api/                # FastAPI app + schema (UI upload/record + docs)
artifacts/          # checkpoints, metrics, registry, optuna plots/db
data/               # processed metadata + label maps + features (out)
```

## Setup Lokal
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .   # optional for editable import slu
```

## Pipeline Ringkas
1) Preprocess (butuh metadata & audio features sudah tersedia jika tidak diulang):
```
slu preprocess --metadata data/processed/metadata.csv --config configs/preprocess.yaml
```
2) Train (baseline):
```
python -m slu.cli train --model-cfg configs/model_m1.yaml --train-cfg configs/train.yaml --metadata-features data/processed/metadata_features.csv
```
3) Train M1 best (hasil Optuna):
```
python -m slu.cli train --model-cfg configs/model_m1.yaml --train-cfg configs/train_m1_best.yaml --metadata-features data/processed/metadata_features.csv
```
4) Evaluate M1 best:
```
python -m slu.cli evaluate --registry artifacts/registry/m1_best.json --model-cfg configs/model_m1.yaml --train-cfg configs/train_m1_best.yaml --metadata-features data/processed/metadata_features.csv --metrics-path artifacts/m1_best/metrics.json --confusion-product artifacts/m1_best/confusion_product.png --confusion-quantity artifacts/m1_best/confusion_qty.png --report-path artifacts/m1_best/classification_report.txt
```
5) Optuna tuning (example):
```
python -m slu.cli hparam-search --study-name slu-optuna-m1 --storage sqlite:///optuna.db --n-trials 20 --model-cfg configs/model_m1.yaml --train-cfg configs/train.yaml --metadata-features data/processed/metadata_features.csv
```

## Registry
- Default inference registry: `artifacts/registry/latest.json` → `artifacts/m1_best/best_model.pt` + configs/label maps.
- Alternate entry: `artifacts/registry/m1_best.json` (same target).

## API (FastAPI)
Run local:
```
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
- Homepage `/` memiliki UI upload dan rekam mic (MediaRecorder → WAV client-side) yang memanggil `/predict`.
- Health: `GET /health`
- Predict: `POST /predict` with form-data `file` (wav/mp3/flac/webm/ogg → dikonversi ke WAV di klien jika rekam).
- Docs: `/docs`, OpenAPI: `/openapi.json`.

Contoh cURL (PowerShell):
```
curl -X POST "http://localhost:8000/predict" -F "file=@sample.wav"
```

## Model & Artefak Penting
- Checkpoint: `artifacts/m1_best/best_model.pt`
- Metrics/report: `artifacts/m1_best/metrics.json`, `classification_report.txt`, `confusion_product.png`, `confusion_qty.png`.
- Hyperparams (best): `configs/train_m1_best.yaml` (lr=2.726e-4, weight_decay=1.03e-6, batch_size=8, max_epochs=30, patience=5, dropout=0.2, cosine scheduler, augment on).

## Deployment Catatan Singkat
- Pastikan `ffmpeg` tersedia untuk decoding audio (librosa/audioread).
- DigitalOcean App Platform (disarankan Dockerfile): build image → expose port 8000 → start `uvicorn api.app:app --host 0.0.0.0 --port 8000` → set `PORT=8000`. Lampirkan `artifacts/` + `configs/` + `data/processed/label_maps` di repo/volume.
- Droplet alternatif: install Python 3.11 + ffmpeg, `pip install -r requirements.txt`, jalankan uvicorn di belakang Nginx/HTTPS.

## Logging
- Inference events dicatat ke `logs/inference.log` (latency_ms, status, confidence, error jika ada).

## Tim
- Kelompok 2 RC: Gymnastiar Al Khoarizmy (122450096), Diana Syafithri (122450141), Eksanty F Islamiaty (122450001), dhea amelia putri (122450004).
