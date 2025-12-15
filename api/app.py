from __future__ import annotations

import io
import json
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Summary, generate_latest

from .schema import HealthResponse, PrediksiResponse
from slu.models.m1_cnn_transformer import CNNTransformerSLU
from slu.models.m2_transformer_tiny import TransformerTinySLU

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = PROJECT_ROOT / "artifacts" / "registry"
DEFAULT_PREPROCESS_CFG = PROJECT_ROOT / "configs" / "preprocess.yaml"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Prometheus metrics (labeled by model)
REQUEST_COUNTER = Counter("predict_requests_total", "Total predict requests", ["model"])
REQUEST_ERRORS = Counter("predict_errors_total", "Total predict errors", ["model"])
REQUEST_LATENCY = Summary("predict_latency_seconds", "Predict latency seconds", ["model"])

app = FastAPI(
    title="SLU Inference API — Kelompok 2 RC",
    description=(
        "API inferensi Spoken Language Understanding (SLU) untuk memprediksi produk dan kuantitas dari audio. "
        "Proyek Tugas Besar Mata Kuliah Machine Learning Operations."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def homepage() -> HTMLResponse:
    return HTMLResponse(
                """
                <!doctype html>
                <html lang="id">
                    <head>
                        <meta charset="utf-8" />
                        <meta name="viewport" content="width=device-width, initial-scale=1" />
                        <title>SLU Inference API — Kelompok 2 RC</title>
                        <style>
                            :root { color-scheme: light dark; font-family: 'Segoe UI', Arial, sans-serif; }
                            body { margin: 0; padding: 24px; background: #0b1021; color: #e6e9f2; }
                            .card { max-width: 720px; margin: 0 auto; background: #111831; border: 1px solid #1f2a4d; border-radius: 14px; padding: 24px; box-shadow: 0 8px 30px rgba(0,0,0,0.35); }
                            h1 { margin-top: 0; font-size: 24px; letter-spacing: 0.3px; }
                            p { color: #c8cede; line-height: 1.5; }
                            .section { margin-top: 18px; padding: 14px; border: 1px dashed #2c3c66; border-radius: 10px; background: #0d142a; }
                            label { display: block; margin-bottom: 8px; font-weight: 600; }
                            input[type="file"], button { font-size: 14px; }
                            button { cursor: pointer; border: none; border-radius: 10px; padding: 10px 14px; background: linear-gradient(135deg, #5dd4ff, #7b7bff); color: #0b1021; font-weight: 700; }
                            button:disabled { opacity: 0.6; cursor: not-allowed; }
                            .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
                            .status { margin-top: 10px; font-family: 'Consolas', monospace; background: #0a1125; padding: 10px; border-radius: 10px; border: 1px solid #1d2d52; white-space: pre-wrap; word-break: break-word; }
                            .links a { color: #7bd7ff; text-decoration: none; }
                            .links a:hover { text-decoration: underline; }
                        </style>
                    </head>
                    <body>
                        <div class="card">
                            <h1>SLU Inference API — Kelompok 2 RC</h1>
                            <p>Halaman ini untuk mencoba <em>online inference</em> dengan mengunggah audio (atau merekam dari mikrofon). Permintaan akan memanggil endpoint <code>/predict</code> yang sama seperti saat deployment.</p>
                            <div class="section">
                                <strong>Identitas Kelompok</strong>
                                <p style="margin: 10px 0 0 0;">
                                    <b>Kelompok 2 RC</b><br/>
                                    Anggota:
                                </p>
                                <ol style="margin: 8px 0 0 18px; color: #c8cede; line-height: 1.6;">
                                    <li>Gymnastiar Al Khoarizmy (122450096)</li>
                                    <li>Diana Syafithri (122450141)</li>
                                    <li>Eksanty F Islamiaty (122450001)</li>
                                    <li>dhea amelia putri (122450004)</li>
                                </ol>
                            </div>


                            <div class="section">
                                <label for="file">Unggah audio (wav/mp3/flac)</label>
                                <div class="row">
                                    <input id="file" type="file" accept="audio/*" />
                                    <button id="uploadBtn">Prediksi</button>
                                </div>
                            </div>

                            <div class="section">
                                <label>Rekam dari Mikrofon</label>
                                <div class="row">
                                    <button id="recordBtn">Mulai Rekam</button>
                                    <button id="stopBtn" disabled>Stop & Prediksi</button>
                                    <span id="recStatus">Idle</span>
                                </div>
                            </div>

                            <div class="section">
                                <strong>Hasil</strong>
                                <div id="result" class="status">Menunggu input...</div>
                                <div class="row" style="margin-top: 10px;">
                                    <button id="resetBtn">Reset</button>
                                    <span style="font-size: 12px; color: #9aa6c6;">Reset akan menghentikan rekaman, mengosongkan input, dan membersihkan hasil.</span>
                                </div>
                            </div>

                            <p class="links">Dokumentasi: <a href="/docs">/docs</a> · OpenAPI: <a href="/openapi.json">/openapi.json</a></p>
                        </div>

                        <script>
                            const resultBox = document.getElementById('result');
                            const uploadBtn = document.getElementById('uploadBtn');
                            const fileInput = document.getElementById('file');
                            const recordBtn = document.getElementById('recordBtn');
                            const stopBtn = document.getElementById('stopBtn');
                            const resetBtn = document.getElementById('resetBtn');
                            const recStatus = document.getElementById('recStatus');
                            let mediaRecorder = null;
                            let chunks = [];
                            let mediaStream = null;

                            async function blobToWav(blob) {
                                const arrayBuf = await blob.arrayBuffer();
                                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                                const audioBuf = await audioCtx.decodeAudioData(arrayBuf);
                                const { numberOfChannels, sampleRate, length } = audioBuf;
                                const interleaved = new Float32Array(length * numberOfChannels);
                                for (let ch = 0; ch < numberOfChannels; ch++) {
                                    interleaved.set(audioBuf.getChannelData(ch), ch * length);
                                }
                                const wavBuf = new ArrayBuffer(44 + interleaved.length * 2);
                                const view = new DataView(wavBuf);
                                function writeString(v, offset, str) { for (let i = 0; i < str.length; i++) v.setUint8(offset + i, str.charCodeAt(i)); }
                                writeString(view, 0, 'RIFF');
                                view.setUint32(4, 36 + interleaved.length * 2, true);
                                writeString(view, 8, 'WAVE');
                                writeString(view, 12, 'fmt ');
                                view.setUint32(16, 16, true);
                                view.setUint16(20, 1, true);
                                view.setUint16(22, numberOfChannels, true);
                                view.setUint32(24, sampleRate, true);
                                view.setUint32(28, sampleRate * numberOfChannels * 2, true);
                                view.setUint16(32, numberOfChannels * 2, true);
                                view.setUint16(34, 16, true);
                                writeString(view, 36, 'data');
                                view.setUint32(40, interleaved.length * 2, true);
                                let offset = 44;
                                for (let i = 0; i < interleaved.length; i++, offset += 2) {
                                    let s = Math.max(-1, Math.min(1, interleaved[i]));
                                    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
                                }
                                return new Blob([view], { type: 'audio/wav' });
                            }

                            async function sendAudio(blob) {
                                resultBox.textContent = 'Mengunggah...';
                                const form = new FormData();
                                const wavBlob = await blobToWav(blob);
                                form.append('file', wavBlob, 'audio.wav');
                                try {
                                    const res = await fetch('/predict', { method: 'POST', body: form });
                                    if (!res.ok) throw new Error(await res.text());
                                    const data = await res.json();
                                    resultBox.textContent = JSON.stringify(data, null, 2);
                                } catch (err) {
                                    resultBox.textContent = `Error: ${err}`;
                                }
                            }

                            uploadBtn.addEventListener('click', () => {
                                if (!fileInput.files.length) {
                                    resultBox.textContent = 'Silakan pilih file audio terlebih dahulu.';
                                    return;
                                }
                                sendAudio(fileInput.files[0]);
                            });

                            recordBtn.addEventListener('click', async () => {
                                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                                    resultBox.textContent = 'Perekaman tidak didukung di browser ini.';
                                    return;
                                }
                                try {
                                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                                    mediaStream = stream;
                                    chunks = [];
                                    mediaRecorder = new MediaRecorder(stream);
                                    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
                                    mediaRecorder.onstop = () => {
                                        const blob = new Blob(chunks, { type: 'audio/webm' });
                                        sendAudio(blob);
                                        stream.getTracks().forEach(t => t.stop());
                                        mediaStream = null;
                                    };
                                    mediaRecorder.start();
                                    recStatus.textContent = 'Merekam...';
                                    recordBtn.disabled = true;
                                    stopBtn.disabled = false;
                                } catch (err) {
                                    resultBox.textContent = `Tidak bisa mulai merekam: ${err}`;
                                }
                            });

                            stopBtn.addEventListener('click', () => {
                                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                                    mediaRecorder.stop();
                                    recStatus.textContent = 'Memproses...';
                                    recordBtn.disabled = false;
                                    stopBtn.disabled = true;
                                }
                            });

                            function resetAll() {
                                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                                    mediaRecorder.stop();
                                }
                                if (mediaStream) {
                                    mediaStream.getTracks().forEach(t => t.stop());
                                    mediaStream = null;
                                }
                                chunks = [];
                                fileInput.value = '';
                                recStatus.textContent = 'Idle';
                                recordBtn.disabled = false;
                                stopBtn.disabled = true;
                                resultBox.textContent = 'Menunggu input...';
                            }

                            resetBtn.addEventListener('click', resetAll);
                        </script>
                    </body>
                </html>
                """
        )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=2)
def load_registry(model_key: str) -> dict:
    registry_path = REGISTRY_DIR / f"{model_key}.json"
    if not registry_path.exists():
        raise RuntimeError(f"Registry not found for model '{model_key}': {registry_path}")
    return json.loads(registry_path.read_text())


@lru_cache(maxsize=1)
def load_label_maps(product_path: Path, qty_path: Path) -> Tuple[dict, dict, dict, dict]:
    product_map = json.loads(product_path.read_text())
    qty_map = json.loads(qty_path.read_text())
    inv_product = {int(v): k for k, v in product_map.items()}
    inv_qty = {int(v): k for k, v in qty_map.items()}
    return product_map, qty_map, inv_product, inv_qty


def build_model(model_cfg_path: Path, num_products: int, num_quantities: int) -> torch.nn.Module:
    cfg = load_yaml(model_cfg_path)
    name = cfg.get("model_name", "cnn_transformer")
    if name == "cnn_transformer":
        model = CNNTransformerSLU(
            num_products=num_products,
            num_quantities=num_quantities,
            d_model=cfg.get("transformer", {}).get("d_model", 256),
            nhead=cfg.get("transformer", {}).get("nhead", 4),
            num_layers=cfg.get("transformer", {}).get("num_layers", 2),
        )
    elif name == "transformer_tiny":
        model = TransformerTinySLU(
            num_products=num_products,
            num_quantities=num_quantities,
            d_model=cfg.get("transformer", {}).get("d_model", 128),
            nhead=cfg.get("transformer", {}).get("nhead", 4),
            num_layers=cfg.get("transformer", {}).get("num_layers", 2),
        )
    else:
        raise ValueError(f"Unknown model_name: {name}")
    return model


@lru_cache(maxsize=2)
def load_model_cached(model_key: str) -> Tuple[torch.nn.Module, dict, dict, dict, dict]:
    reg = load_registry(model_key)
    model_path_raw = reg.get("model_path")
    config_path_raw = reg.get("config_path")
    product_map_raw = reg.get("label_map_product")
    qty_map_raw = reg.get("label_map_quantity")

    if not (model_path_raw and config_path_raw and product_map_raw and qty_map_raw):
        raise RuntimeError("Registry missing required keys: model_path, config_path, label_map_product, label_map_quantity")

    model_path = PROJECT_ROOT / model_path_raw if not Path(model_path_raw).is_absolute() else Path(model_path_raw)
    config_path = PROJECT_ROOT / config_path_raw if not Path(config_path_raw).is_absolute() else Path(config_path_raw)
    product_map_path = PROJECT_ROOT / product_map_raw if not Path(product_map_raw).is_absolute() else Path(product_map_raw)
    qty_map_path = PROJECT_ROOT / qty_map_raw if not Path(qty_map_raw).is_absolute() else Path(qty_map_raw)

    product_map, qty_map, inv_product, inv_qty = load_label_maps(product_map_path, qty_map_path)

    model = build_model(config_path, num_products=len(product_map), num_quantities=len(qty_map))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device, product_map, qty_map, {"product": inv_product, "quantity": inv_qty}


def load_audio_to_logmel(data: bytes, preprocess_cfg: Path = DEFAULT_PREPROCESS_CFG) -> tuple[np.ndarray, int, float]:
    cfg = load_yaml(preprocess_cfg)
    sr = cfg.get("sample_rate", 16000)
    trim_top_db = cfg.get("trim_top_db", 30)
    n_mels = cfg.get("n_mels", 128)
    n_fft = cfg.get("n_fft", 1024)
    hop_length = cfg.get("hop_length", 256)
    win_length = cfg.get("win_length", 1024)
    power = cfg.get("power", 2.0)
    min_dur = cfg.get("min_duration", 0.5)
    max_dur = cfg.get("max_duration", 5.0)

    buf = io.BytesIO(data)
    waveform, _ = librosa.load(buf, sr=sr, mono=True)
    if trim_top_db is not None:
        waveform, _ = librosa.effects.trim(waveform, top_db=trim_top_db)
    duration = waveform.shape[0] / sr
    if duration < min_dur or duration > max_dur:
        raise ValueError(f"Audio duration {duration:.2f}s di luar rentang {min_dur}-{max_dur}s")
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=power,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    frames = logmel.shape[1]
    return logmel.astype(np.float32), frames, duration


def infer(logmel: np.ndarray, model_key: str):
    model, device, _, _, inv_maps = load_model_cached(model_key)
    tensor = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(device)
    lengths = torch.tensor([logmel.shape[1]], device=device)
    with torch.no_grad():
        outputs = model(tensor)
        prod_logits = outputs["product"]
        qty_logits = outputs["quantity"]
        prod_prob = torch.softmax(prod_logits, dim=-1)
        qty_prob = torch.softmax(qty_logits, dim=-1)
        prod_idx = int(prod_prob.argmax(dim=-1).item())
        qty_idx = int(qty_prob.argmax(dim=-1).item())
        prod_conf = float(prod_prob.max().item())
        qty_conf = float(qty_prob.max().item())
    return {
        "product": inv_maps["product"].get(prod_idx, str(prod_idx)),
        "quantity": inv_maps["quantity"].get(qty_idx, str(qty_idx)),
        "confidence": min(prod_conf, qty_conf),
    }


def log_event(event: dict) -> None:
    timestamp = int(time.time() * 1000)
    path = LOG_DIR / "inference.log"
    line = {"ts": timestamp, **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PrediksiResponse)
async def predict(
    file: UploadFile = File(...),
    model_version: str | None = Query(None, description="Pilih model: m1 atau m2"),
    model_header: str | None = Header(None, alias="X-Model-Version"),
) -> PrediksiResponse:
    request_id = str(uuid.uuid4())
    model_key = (model_version or model_header or "m1").lower()
    if model_key not in {"m1", "m2"}:
        raise HTTPException(status_code=400, detail="model_version must be m1 or m2")
    try:
        reg = load_registry(model_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    allowed_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/flac",
        "audio/mp3",
        "audio/wave",
        "audio/webm",
        "audio/ogg",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported audio type")
    data = await file.read()
    t0 = time.perf_counter()
    try:
        logmel, frames, duration = load_audio_to_logmel(data)
        result = infer(logmel, model_key=model_key)
        latency = time.perf_counter() - t0
        REQUEST_COUNTER.labels(model_key).inc()
        REQUEST_LATENCY.labels(model_key).observe(latency)
        log_event(
            {
                "event": "predict",
                "request_id": request_id,
                "status": "ok",
                "latency_ms": round(latency * 1000, 2),
                "audio_duration": round(duration, 3),
                "mel_mean": round(float(logmel.mean()), 6),
                "mel_std": round(float(logmel.std()), 6),
                "frames": int(frames),
                "product": result["product"],
                "quantity": result["quantity"],
                "confidence": result["confidence"],
                "model_version": model_key,
                "model_path": reg.get("model_path", "unknown"),
            }
        )
    except Exception as exc:  # pragma: no cover
        latency = time.perf_counter() - t0
        REQUEST_ERRORS.labels(model_key).inc()
        log_event(
            {
                "event": "predict",
                "request_id": request_id,
                "status": "error",
                "latency_ms": round(latency * 1000, 2),
                "model_version": model_key,
                "error": str(exc),
            }
        )
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}")
    return PrediksiResponse(product=result["product"], quantity=result["quantity"], confidence=result["confidence"])


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
