# VisionSpeak — AI Lip Reading System

A complete browser application for visual-only speech-recognition experiments,
using Python, TensorFlow, OpenCV, MediaPipe, Flask, HTML, CSS and JavaScript.

## Start

```cmd
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python prepare_video.py --download-model
python fetch_reference_model.py
python serve.py
```

Open **http://127.0.0.1:8080/** and create a private account. `python app.py` starts
the same application. Keep the existing `models/checkpoint.weights.h5` file in place.

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for upload, webcam, live lip detection,
prediction history, export/delete, test instructions and deployment status.

## Features

- Video upload with automatic lip detection or original GRID framing.
- Camera-only capture, live lip-region preview and repeated short recognition cycles.
- Existing TensorFlow checkpoint with greedy or optional beam decoding.
- Explicit unreliable-result handling; no fabricated confidence percentages.
- Private accounts, signed sessions, CSRF protection and persistent per-user history.
- Responsive desktop/mobile interface, search, JSON export and deletion.
- Bounded inference, authenticated API, production server and PostgreSQL support.

**Model update:** the browser now offers a local Auto-AVSR English sentence model
when its optional runtime and weights are installed. The supplied greeting decoded
as **HI I AM JENNIFER**: the greeting works, but the name is wrong. This is an
experimental feature, not validated general-speech accuracy. See
[GENERAL_MODEL.md](GENERAL_MODEL.md) for setup, measured results and limitations.

The public LipNet GRID model remains available alongside the original checkpoint.
On eleven public regression clips, word error rate
fell from 100% to 10.6% (four exact sentences). These are not held-out accuracy
results. That GRID model does not support general conversation.
No model was trained. See [MODEL_PROGRESS.md](MODEL_PROGRESS.md) for sources,
reproduction commands, measured limitations and remaining work.

## Project layout

- `service.py`, `web_features.py`: Flask API, browser sessions and capture routes.
- `history_store.py`: SQLite/PostgreSQL persistence and request limits.
- `templates/index.html`, `static/`: complete website.
- `recognition.py`, `lip_reader.py`: inference using existing weights.
- `reference_lipnet.py`: public RGB/GRU model adapter for GRID commands.
- `input_quality.py`: advisory image-quality and webcam timing diagnostics.
- `fetch_reference_model.py`, `fetch_public_samples.py`, `benchmark_public.py`: reproducible public model evaluation.
- `preprocessing.py`, `prepare_video.py`: Part 1 video preparation.
- `recognize.py`, `evaluate_model.py`: command-line inference/evaluation.
- `serve.py`, `Dockerfile`, `compose.yaml`: deployment entry points and recipes.
- `tests/`: inference-only software verification.
- `lip_reader.ipynb`, `train_model.py`: historical development artifacts, not used
  by the current application or tests.

Local state is stored under ignored `instance/`; data and outputs are also ignored.
Raw video and webcam images are removed after processing. Only result records are
retained in history. [DEPLOYMENT.md](DEPLOYMENT.md) describes production secrets,
HTTPS, shared database, backups, scaling and unverified deployment checks.
