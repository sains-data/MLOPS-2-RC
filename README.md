# MLOps Speech SLU (Retail) — Kelompok 2 RC

Spoken Language Understanding untuk retail (prediksi `product` & `quantity`) dengan log-mel → CNN+Transformer (M1) atau Transformer-tiny (M2). Proyek Tugas Besar MLOps: data pipeline, reproducible training, Optuna tuning, registry, FastAPI inference, monitoring.
## Ringkas Hasil
- Model default: **M1 best** (`artifacts/m1_best/best_model.pt`, config `configs/model_m1.yaml`, train cfg `configs/train_m1_best.yaml`).
- Metrics (val): acc_product 0.68, acc_quantity 0.71, f1_product 0.676, f1_quantity 0.704, val_loss 1.573.
- Registry: `artifacts/registry/latest.json` menunjuk ke M1 best (label maps di `data/processed/product2id.json`, `data/processed/qty2id.json`).
- Optuna study & plot: `optuna.db`, `artifacts/optuna-hyperparameter.png`, `artifacts/m1/optuna_*.html`.
- Deploy (DO App Platform): `https://orca-app-u6dem.ondigitalocean.app/` (UI upload/record), `/docs`, `/metrics`.

## Stack & Dev
- Python, PyTorch, FastAPI, Optuna, librosa, matplotlib.
- CLI via Typer (`python -m slu.cli ...`).
- Scripting: package in `src/slu/` (ingest, preprocess, train, eval, registry, models, eda, monitor_report).
- Logging: inference JSONL at `logs/inference.log`.
- Reproducibility: GitHub versioning + registry JSON for model/version; tag `v0.12` (monitoring release).

## Struktur Singkat
```
configs/            # data/preprocess/model/train configs
src/slu/            # ingest, preprocess, train, eval, registry, models, eda, monitor_report
api/                # FastAPI app + schema (UI upload/record + docs + /metrics)
artifacts/          # checkpoints, metrics, registry, optuna plots/db
data/               # processed metadata + label maps + features (out)
logs/               # inference.log (JSONL)
```

## Setup Lokal
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .   # optional for editable import slu
```

## Komponen & Langkah
### Data
- **Preparation**: metadata/labels di `data/processed/metadata.csv`, label maps `product2id.json`, `qty2id.json`.
- **EDA**: `python src/slu/eda.py --metadata data/processed/metadata.csv --output-dir artifacts/eda` → summary + plots.
- **Preprocessing**: log-mel sesuai `configs/preprocess.yaml` → `data/processed/metadata_features.csv` + feature `.npy`.

### Modeling (dua model: M1 CNN+Transformer, M2 Transformer tiny)
- **Training**: `python -m slu.cli train --model-cfg configs/model_m1.yaml --train-cfg configs/train.yaml --metadata-features data/processed/metadata_features.csv` (baseline). M1 best gunakan `configs/train_m1_best.yaml`.
- **Experiment Tracking**: W&B enabled in configs (`wandb.enable: true`, project `slu-mlops`).
- **Hyperparameter Tuning**: Optuna (`python -m slu.cli hparam-search --study-name slu-optuna-m1 --storage sqlite:///optuna.db ...`).
- **Evaluation**: `python -m slu.cli evaluate --registry artifacts/registry/m1_best.json ...` menghasilkan metrics, confusion, report di `artifacts/m1_best/`.
- **Model Serving**: FastAPI (`api/app.py`) at `/predict`, `/health`, `/docs`, `/metrics`; UI upload/record di `/`. Dockerfile for deployment.

### Developing
- **Scripting org**: modular package under `src/slu/` (train/eval/models/registry/eda/monitor_report).
- **CLI**: Typer entrypoints in `slu.cli` (ingest, preprocess, train, evaluate, hparam-search).

### Logging & Monitoring
- Inference logs (JSONL) at `logs/inference.log`: request_id, latency_ms, audio_duration, mel_mean/std, frames, prediction, confidence, model_version.
- Prometheus metrics at `/metrics`: counters + latency summary.
- Offline report: `python src/slu/monitor_report.py --log logs/inference.log --output-dir artifacts/monitoring --max-classes 20` → latency/duration/confidence hist, class counts, `summary.json` (example: rows=24, mean latency ~1.94s, p95 ~7.09s, mean confidence ~0.52).
- Rationale: online drift proxy via input stats (duration, mel_mean/std) + prediction/confidence distribution because ground truth unavailable.

### Reproducibility
- Code/versioning: GitHub, tags (latest monitoring tag `v0.12`).
- Models/configs: registry JSON points to checkpoint/config/label maps.
- Data artifacts tracked in repo paths (`data/processed`, `artifacts/m1_best`).

### Production
- **CI/CD**: GitHub Actions runs on push (install + tests). Dockerfile for DO App Platform; CPU-only torch wheel to reduce build size.
- **Monitoring**: `/metrics` for Prometheus scrape; `logs/inference.log` + `monitor_report.py` for periodic plots; UI served at DO URL.

## Pipeline Ringkas (perintah utama)
- Preprocess: `slu preprocess --metadata data/processed/metadata.csv --config configs/preprocess.yaml`
- Train baseline: `python -m slu.cli train --model-cfg configs/model_m1.yaml --train-cfg configs/train.yaml --metadata-features data/processed/metadata_features.csv`
- Train best: `python -m slu.cli train --model-cfg configs/model_m1.yaml --train-cfg configs/train_m1_best.yaml --metadata-features data/processed/metadata_features.csv`
- Evaluate best: `python -m slu.cli evaluate --registry artifacts/registry/m1_best.json --model-cfg configs/model_m1.yaml --train-cfg configs/train_m1_best.yaml --metadata-features data/processed/metadata_features.csv --metrics-path artifacts/m1_best/metrics.json --confusion-product artifacts/m1_best/confusion_product.png --confusion-quantity artifacts/m1_best/confusion_qty.png --report-path artifacts/m1_best/classification_report.txt`
- Optuna tuning: `python -m slu.cli hparam-search --study-name slu-optuna-m1 --storage sqlite:///optuna.db --n-trials 20 --model-cfg configs/model_m1.yaml --train-cfg configs/train.yaml --metadata-features data/processed/metadata_features.csv`
- Monitoring report: `python src/slu/monitor_report.py --log logs/inference.log --output-dir artifacts/monitoring --max-classes 20`

## API (FastAPI)
- Run local: `uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload`
- Homepage `/`: upload + mic record (MediaRecorder → WAV) hitting `/predict`.
- Health: `GET /health`; Predict: `POST /predict` form-data `file` (wav/mp3/flac/webm/ogg).
- Docs: `/docs`; OpenAPI: `/openapi.json`; Metrics: `/metrics` (Prometheus).
- Deploy: `https://orca-app-u6dem.ondigitalocean.app/`

## Model & Artefak Penting
- Checkpoint: `artifacts/m1_best/best_model.pt`
- Metrics/report: `artifacts/m1_best/metrics.json`, `classification_report.txt`, `confusion_product.png`, `confusion_qty.png`.
- Hyperparams (best): `configs/train_m1_best.yaml` (lr=2.726e-4, weight_decay=1.03e-6, batch_size=8, max_epochs=30, patience=5, dropout=0.2, cosine scheduler, augment on).

## Tim
- Kelompok 2 RC: Gymnastiar Al Khoarizmy (122450096), Diana Syafithri (122450141), Eksanty F Islamiaty (122450001), dhea amelia putri (122450004).
