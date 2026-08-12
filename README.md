# 🔥 Fire Detection from Satellite Images

**A tested, containerized fire-detection toolkit — Streamlit UI, REST API, and CLI — that classifies satellite/aerial imagery as Fire or No Fire and explains *why*.**

A Keras CNN drives the prediction; Grad-CAM explainability, a tunable decision threshold, batch CSV triage, a FastAPI REST endpoint, a scriptable CLI, and hardened input validation turn a simple classifier demo into a small, production-shaped computer vision tool. All three interfaces (`app.py`, `api.py`, `cli.py`) share one framework-free inference core (`core.py`), so there's exactly one place the prediction logic can go wrong.

[![CI](https://github.com/bharat3645/fire_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/bharat3645/fire_detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/model-TensorFlow%2FKeras-FF6F00.svg)](https://www.tensorflow.org/)

---

## Overview

Upload a satellite or aerial image and the app predicts whether it shows fire, using a pretrained Keras CNN. Beyond the base classifier, this project focuses on the things that separate a weekend demo from something you'd actually trust:

- **Explainability** — a real gradient-based Grad-CAM heatmap shows which pixels drove the call, instead of a bare probability.
- **Usability at scale** — batch-upload a folder of images and export a CSV report in one pass.
- **Robustness** — corrupt files, oversized uploads, and model-load failures fail safely with a clear message, never a crash or a silent wrong answer.
- **Engineering hygiene** — the inference/validation logic lives in a framework-free module with a real test suite, linting, CI, and a Dockerfile, not just a single UI script.

## Features

| Feature | Description |
|---|---|
| 🖼️ **Single image detection** | Upload one image and get a Fire / No Fire verdict with a confidence score. |
| 🔬 **Grad-CAM explainability** | A gradient-based heatmap (computed from the model's own convolutional layers, verified against the shipped `fire_detection_model.h5`) overlaid on the image, showing exactly which regions pushed the prediction toward "Fire." |
| 🎚️ **Adjustable decision threshold** | A sidebar slider controls the P(Fire) cutoff (default 0.5) live, useful for demonstrating sensitivity vs. false-positive tradeoffs instead of a hardcoded cutoff. |
| 📦 **Batch upload mode** | Upload many images at once and get a results table — filename, prediction, confidence, P(Fire), and per-file error — exportable as CSV for triaging a whole folder of tiles. |
| 🛡️ **Input validation & safe failure** | Empty files, oversized uploads (10 MB cap), corrupt/non-image files, and model-load failures at startup are all caught and surfaced as clear messages; one bad file in a batch never kills the run. |
| 🌐 **REST API** | A FastAPI service (`api.py`) exposes `/predict` (with a `?threshold=` override) and `/health`, with interactive OpenAPI docs at `/docs` — for calling the model from a script or another service instead of a browser. |
| 💻 **CLI** | `cli.py` runs batch inference over a single image or a whole directory (recursively) with no browser involved, writes CSV or JSON, and exits with a script-friendly status code (0 = clean, 2 = fire flagged, 1 = fatal error) — built for cron jobs and scheduled scans. |
| 📝 **Prediction audit log** | Every prediction made through the UI, API, or CLI appends a timestamped JSON-line record (`core.log_prediction`), giving a consistent, greppable trail across all three interfaces. |
| ✅ **Test suite + CI + Docker** | 44 pytest tests across `core.py`, `api.py`, and `cli.py` (validation, preprocessing, classification, Grad-CAM, REST endpoints, CLI exit codes — plus an end-to-end test against the real `.h5` model), a ruff lint pass, and a GitHub Actions workflow that runs both plus a Docker build on every push/PR. |

## Tech stack

- **Model / inference:** TensorFlow / Keras (CNN, 64×64 RGB input, single sigmoid output)
- **UI:** Streamlit
- **REST API:** FastAPI + Uvicorn
- **CLI:** argparse
- **Image handling:** Pillow, NumPy
- **Testing:** pytest
- **Linting:** ruff
- **CI/CD:** GitHub Actions
- **Containerization:** Docker

## Architecture

Three thin entry points (Streamlit UI, REST API, CLI) all sit on top of one framework-free logic layer, so prediction, validation, and explainability can be unit-tested — and stay in sync across all three interfaces — without booting Streamlit or a web server:

```mermaid
flowchart LR
    U["User"] -->|"upload image(s)"| APP["app.py\n(Streamlit UI)"]
    S["Script / service"] -->|"HTTP POST /predict"| API["api.py\n(FastAPI)"]
    CRON["cron job / shell"] -->|"batch of files"| CLI["cli.py\n(argparse)"]

    APP -->|"validate + preprocess"| CORE["core.py\n(framework-free logic)"]
    API -->|"validate + preprocess"| CORE
    CLI -->|"validate + preprocess"| CORE

    CORE -->|"predict"| MODEL["fire_detection_model.h5\n(Keras CNN)"]
    MODEL -->|"P(Fire)"| CORE
    CORE -->|"classify + Grad-CAM"| APP
    CORE -->|"classify"| API
    CORE -->|"classify"| CLI
    CORE -.->|"log_prediction()"| LOG["logs/predictions.jsonl\n(audit trail)"]

    APP -->|"verdict, heatmap, CSV"| U
    API -->|"JSON verdict"| S
    CLI -->|"CSV / JSON + exit code"| CRON

    TESTS["tests/\n(44 pytest tests)"] -.->|"exercises"| CORE
    CI["GitHub Actions CI"] -.->|"lint + test + docker build"| TESTS
```

### Project structure

```
fire_detection/
├── app.py                      # Streamlit UI: upload, single/batch inference, results display
├── api.py                      # FastAPI REST service: /predict, /health, OpenAPI docs
├── cli.py                      # Headless batch-inference CLI: file/dir in, CSV/JSON out
├── core.py                     # Framework-free logic: validation, preprocessing, classification, Grad-CAM
├── fire_detection_model.h5     # Pretrained Keras CNN (64x64 RGB input, sigmoid output)
├── tests/
│   ├── conftest.py             # Shared tiny in-memory Keras model fixture (fast, no 19MB model load)
│   ├── test_core.py            # Validation, preprocessing, classification, Grad-CAM, logging
│   ├── test_api.py             # FastAPI TestClient tests for /predict and /health
│   └── test_cli.py             # CLI batch runs, exit codes, CSV/JSON output
├── .github/workflows/ci.yml    # Lint + test + Docker build on every push/PR
├── Dockerfile                  # Container build for one-command deployment (UI, API, or CLI)
├── .dockerignore
├── pyproject.toml              # ruff + pytest configuration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # + pytest, ruff
└── README.md
```

## Getting started

### Prerequisites

- Python 3.11+
- (Optional) Docker, for containerized runs

### Installation

```bash
git clone https://github.com/bharat3645/fire_detection.git
cd fire_detection
pip install -r requirements.txt
```

### Run the Streamlit UI

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (defaults to `http://localhost:8501`).

- **Single Image mode:** upload a satellite image and click **Detect Fire** to see the verdict, confidence, and Grad-CAM heatmap.
- **Batch Upload mode:** upload multiple images, run them all, review the results table, and download it as CSV.

Use the sidebar to adjust the **Fire decision threshold** and toggle the **Grad-CAM heatmap** on or off.

### Run the REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`, or call it directly:

```bash
curl -F "file=@image.jpg" "http://localhost:8000/predict?threshold=0.5"
```

`GET /health` reports whether the model loaded successfully — useful for container/orchestrator health checks.

### Run the CLI

```bash
python cli.py --input path/to/images/                       # CSV to stdout
python cli.py --input image.jpg --format json
python cli.py --input path/to/images/ --output results.csv --threshold 0.3
```

Exits `0` if nothing was flagged as Fire, `2` if at least one image was, and `1` on a fatal error (bad path, no images found, model failed to load) — designed to be checked from a cron job or shell script.

### Run with Docker

```bash
docker build -t fire-detection .
docker run -p 8501:8501 fire-detection          # Streamlit UI (default)
docker run -p 8000:8000 --entrypoint uvicorn fire-detection api:app --host 0.0.0.0 --port 8000   # REST API
docker run --entrypoint python -v /path/to/images:/data fire-detection cli.py --input /data       # CLI
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v            # unit + integration tests for core.py, api.py, and cli.py
ruff check .          # lint
```

Tests run automatically on every push/PR via GitHub Actions (`.github/workflows/ci.yml`), which also builds the Docker image to catch container regressions.

## Notes / limitations

- The model expects 64×64 RGB input, matching its saved input shape; images are resized automatically before inference.
- The model ships pretrained in this repo — no training script or dataset is included, so it can't be retrained or fine-tuned from this repo alone.
- Prediction quality depends entirely on the (unknown) training dataset used to produce `fire_detection_model.h5`; treat outputs as a demo, not a production wildfire-detection system.
- Grad-CAM is computed on the fly per image and is best-effort — if it fails for a given image, the app still shows the prediction and just skips the heatmap.
- Max upload size is 10 MB per image (configurable via `core.MAX_UPLOAD_SIZE_MB`), to keep the app responsive.
- Model path and audit-log path are overridable via the `FIRE_MODEL_PATH` and `FIRE_DETECTION_LOG_PATH` env vars (the API and CLI read these directly; the CLI also has `--model`).

## Roadmap

- [ ] Training/fine-tuning script with a documented dataset
- [ ] Multi-class severity levels (e.g. smoke-only vs. active fire)

## Contributing

Issues and pull requests are welcome. Please run `ruff check .` and `pytest -v` before submitting a PR — both are enforced in CI.

## License

Licensed under the [MIT License](LICENSE) © 2026 [bharat3645](https://github.com/bharat3645).
