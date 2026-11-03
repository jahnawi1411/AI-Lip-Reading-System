FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    VISIONSPEAK_ENV=production PORT=8080 \
    MPLCONFIGDIR=/tmp/matplotlib TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 libegl1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-part1.txt requirements-part2.txt requirements-server.txt requirements-audio.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt -r requirements-audio.txt
RUN python -m venv --system-site-packages /app/.venv-general && /app/.venv-general/bin/pip install --no-cache-dir torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
COPY service.py serve.py recognition.py preprocessing.py input_quality.py reference_lipnet.py general_bridge.py general_lipreading.py audiovisual.py setup_audio.py deployment_preflight.py setup_general_model.py lip_reader.py web_features.py history_store.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/checkpoint.weights.h5 models/face_landmarker.task ./models/
COPY models/lipnet-reference/ ./models/lipnet-reference/
COPY vendor/auto_avsr/ ./vendor/auto_avsr/
COPY models/auto-avsr/ ./models/auto-avsr/
COPY THIRD_PARTY_LICENSE_LIPNET.txt ./
RUN python deployment_preflight.py
RUN useradd --uid 10001 --create-home appuser && mkdir -p /app/instance /app/models/whisper && chown appuser /app/instance /app/models/whisper
USER appuser
# Download audio weights at build time, not during a user request.
RUN python setup_audio.py
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/ready', timeout=4)"
CMD ["python", "serve.py"]
