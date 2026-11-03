"""Stateless inference API. Each process owns one bounded inference slot."""
import hmac
import os
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

from flask import Flask, g, jsonify, request, render_template
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.update(
        ENVIRONMENT=os.getenv("VISIONSPEAK_ENV", "development"),
        API_TOKEN=os.getenv("VISIONSPEAK_API_TOKEN", ""),
        WEIGHTS_PATH=os.getenv("VISIONSPEAK_WEIGHTS", str(ROOT / "models/checkpoint.weights.h5")),
        REFERENCE_WEIGHTS_PATH=os.getenv("VISIONSPEAK_REFERENCE_WEIGHTS", str(ROOT / "models/lipnet-reference/unseen-weights178.h5")),
        UPLOAD_ROOT=os.getenv("VISIONSPEAK_UPLOAD_ROOT", tempfile.gettempdir()),
        MAX_CONTENT_LENGTH=32 * 1024 * 1024,
        # Werkzeug reads multipart bodies in 64 KiB chunks and can retain a
        # partial boundary between reads. Leave room for both, including files.
        MAX_FORM_MEMORY_SIZE=128 * 1024,
        MAX_FORM_PARTS=10,
    )
    if overrides:
        app.config.update(overrides)
    if app.config["ENVIRONMENT"] not in {"development", "production"}:
        raise ValueError("VISIONSPEAK_ENV must be development or production.")
    if app.config["ENVIRONMENT"] == "production" and len(app.config["API_TOKEN"]) < 32:
        raise ValueError("Production requires VISIONSPEAK_API_TOKEN with at least 32 characters.")
    Path(app.config["UPLOAD_ROOT"]).mkdir(parents=True, exist_ok=True)
    slot = threading.BoundedSemaphore(1)
    app.extensions["inference_slot"] = slot
    ready = False

    def warmup():
        nonlocal ready
        if ready:
            return
        from lip_reader import get_model
        from preprocessing import LipDetector
        get_model(Path(app.config["WEIGHTS_PATH"]))
        with LipDetector():
            pass
        ready = True

    app.extensions["warmup"] = warmup
    from web_features import install_web
    authorize = install_web(app, slot, warmup)

    @app.before_request
    def identify_and_authorize():
        g.request_id = uuid4().hex
        g.started = time.monotonic()
        return authorize()

    @app.after_request
    def response_headers(response):
        response.headers["X-Request-ID"] = g.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(self)"
        response.headers["Referrer-Policy"] = "same-origin"
        app.logger.info("request_id=%s method=%s path=%s status=%s elapsed_ms=%.0f",
                        g.request_id, request.method, request.path, response.status_code,
                        (time.monotonic() - g.started) * 1000)
        return response

    @app.errorhandler(Exception)
    def errors(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.name, request_id=g.request_id), exc.code
        app.logger.exception("request_id=%s inference failure", g.request_id)
        return jsonify(error="internal_error", message="Request failed; use the request ID for support.",
                       request_id=g.request_id), 500

    @app.get("/")
    def index():
        from general_bridge import available
        return render_template("index.html", general_available=available())

    @app.get("/health/live")
    def live():
        return jsonify(status="alive")

    @app.get("/health/ready")
    def readiness():
        # A probe must not compete with video inference or repeatedly load models.
        if ready:
            from sqlalchemy import text
            try:
                with app.extensions["store"].engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
            except Exception:
                return jsonify(status="database_unavailable", recognition_accuracy="unvalidated"), 503
        return jsonify(status="ready" if ready else "starting", recognition_accuracy="unvalidated"), 200 if ready else 503

    @app.post("/api/v1/recognize")
    def recognize():
        if not slot.acquire(blocking=False):
            response = jsonify(error="busy", message="This instance is processing another clip; retry later.")
            response.headers["Retry-After"] = "5"
            return response, 429
        try:
            upload = request.files.get("video")
            if upload is None or not upload.filename:
                return jsonify(error="missing_video"), 400
            extension = Path(upload.filename).suffix.lower()
            if extension not in {".mp4", ".mpg", ".mpeg", ".avi", ".webm"}:
                return jsonify(error="unsupported_format"), 400
            mode = request.form.get("mode", "mediapipe")
            decoder = request.form.get("decoder", "greedy")
            engine = request.form.get('engine', 'original')
            if engine not in {'original', 'lipnet-reference', 'auto-avsr', 'audio-video', 'lip-expression'} or (engine != 'original' and mode != 'mediapipe'):
                return jsonify(error='invalid_options', message='The selected model requires automatic lip detection.'), 400
            if mode not in {"mediapipe", "grid"} or decoder not in {"greedy", "beam"}:
                return jsonify(error="invalid_options"), 400
            try:
                start = float(request.form.get("start", "0"))
                duration = float(request.form["duration"]) if "duration" in request.form else None
            except ValueError:
                return jsonify(error="invalid_time_selection"), 400
            # Never use client paths, retain uploads, or persist transcripts here.
            with tempfile.TemporaryDirectory(prefix="visionspeak-", dir=app.config["UPLOAD_ROOT"]) as directory:
                video_path = Path(directory) / ("input" + extension)
                upload.save(video_path)
                if engine == 'lip-expression':
                    from audiovisual import recognize_expressions
                    try:
                        result = recognize_expressions(video_path, start, duration)
                    except ValueError as exc:
                        return jsonify(error='invalid_video', message=str(exc)), 422
                    except RuntimeError as exc:
                        return jsonify(error='expression_unavailable', message=str(exc)), 503
                    result['request_id'] = g.request_id
                    return jsonify(app.extensions['save_result'](result, Path(upload.filename.replace('\\', '/')).name))
                if engine == 'audio-video':
                    from audiovisual import recognize_audiovisual
                    try:
                        result = recognize_audiovisual(video_path, start, duration)
                    except ValueError as exc:
                        return jsonify(error='invalid_video', message=str(exc)), 422
                    except RuntimeError as exc:
                        return jsonify(error='audio_unavailable', message=str(exc)), 503
                    result['request_id'] = g.request_id
                    return jsonify(app.extensions['save_result'](result, Path(upload.filename.replace('\\', '/')).name))
                if engine != 'auto-avsr':
                    warmup()
                from recognition import recognize_file
                try:
                    if engine == 'auto-avsr':
                        from general_bridge import available, recognize_general
                        if not available():
                            return jsonify(error='model_unavailable', message='The English sentence model is not installed.'), 503
                        result = recognize_general(video_path,start,duration)
                    else:
                        weight_key = 'REFERENCE_WEIGHTS_PATH' if engine == 'lipnet-reference' else 'WEIGHTS_PATH'
                        if not Path(app.config[weight_key]).exists() and engine == 'lipnet-reference':
                            return jsonify(error='model_unavailable', message='The public reference model is not installed.'), 503
                        result = recognize_file(video_path, mode, Path(app.config[weight_key]),
                                                decoder=decoder, start=start, duration=duration, engine=engine)
                except ValueError:
                    return jsonify(error="invalid_video", message="Check video format, visible face, frame count and selected interval."), 422
                from audiovisual import attach_lip_expressions
                attach_lip_expressions(result, video_path, start, duration)
            # Do not disclose server filesystem paths or image geometry to clients.
            response = {key: value for key, value in result.items()
                        if key not in {"source", "weights", "input_report"}}
            response["request_id"] = g.request_id
            return jsonify(app.extensions["save_result"](response, Path(upload.filename.replace('\\', '/')).name))
        finally:
            slot.release()

    return app
