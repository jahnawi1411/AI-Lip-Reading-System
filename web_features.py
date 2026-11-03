"""Browser sessions, per-user history and codec-independent webcam input."""
import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from datetime import timedelta
from pathlib import Path

from flask import abort, g, jsonify, request, session
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash

from history_store import Store


def install_web(app, slot, warmup):
    instance = Path(app.instance_path)
    instance.mkdir(parents=True, exist_ok=True)
    token = app.config["API_TOKEN"]
    secret = os.getenv("VISIONSPEAK_SECRET_KEY")
    if not secret:
        if token:
            secret = hmac.new(token.encode(), b"visionspeak-browser-session", hashlib.sha256).hexdigest()
        elif app.testing:
            secret = secrets.token_hex(32)
        else:
            path = instance / "session.key"
            try:
                with path.open("x") as file:
                    file.write(secrets.token_hex(32))
            except FileExistsError:
                pass
            secret = path.read_text()
    if len(secret) < 32:
        raise ValueError("VISIONSPEAK_SECRET_KEY must contain at least 32 characters.")
    app.secret_key = secret
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_SECURE=app.config["ENVIRONMENT"] == "production",
                      PERMANENT_SESSION_LIFETIME=timedelta(days=1))
    url = app.config.get("DATABASE_URL") or os.getenv("DATABASE_URL") or (
        "sqlite://" if app.testing else "sqlite:///" + (instance / "visionspeak.db").as_posix())
    store = Store(url)
    app.extensions["store"] = store

    def csrf():
        if not session.get("csrf") or not hmac.compare_digest(request.headers.get("X-CSRF-Token", "").encode(), session["csrf"].encode()):
            abort(403)

    def guard():
        g.owner = None
        if not request.path.startswith("/api/"):
            return
        if request.path.startswith("/api/auth/"):
            if request.method != "GET":
                csrf()
            return
        header = request.headers.get("Authorization", "")
        if token and hmac.compare_digest(header.encode(), ("Bearer " + token).encode()):
            return
        user = store.user(uid=session["uid"]) if session.get("uid") else None
        if user:
            g.owner = user["id"]
            if request.method not in {"GET", "HEAD"}:
                csrf()
            if request.path in {"/api/v1/recognize", "/api/v1/recognize-frames"}:
                if not store.allow("infer:" + g.owner, maximum=10):
                    return jsonify(error="rate_limited", message="Please wait a minute before another recognition."), 429
            return
        # Preserve localhost CLI use; private history/preview still require login.
        if app.config["ENVIRONMENT"] == "development" and not token and request.path == "/api/v1/recognize":
            return
        return jsonify(error="unauthorized", message="Sign in to continue."), 401

    @app.get("/api/auth/session")
    def auth_session():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(32)
        user = store.user(uid=session["uid"]) if session.get("uid") else None
        return jsonify(csrf=session["csrf"], user={"username": user["username"]} if user else None)

    @app.post("/api/auth/<action>")
    def authenticate(action):
        if action == "logout":
            session.clear()
            return jsonify(ok=True)
        if action not in {"login", "register"}:
            abort(404)
        # No forwarded IP is trusted by default. Configure rate limiting at the gateway too.
        address = hashlib.sha256((request.remote_addr or "unknown").encode()).hexdigest()
        if not store.allow("auth:" + address, maximum=12, seconds=900):
            return jsonify(error="rate_limited", message="Too many attempts. Try again in 15 minutes."), 429
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            abort(400)
        username, password = body.get("username", ""), body.get("password", "")
        if not isinstance(username, str) or not isinstance(password, str):
            abort(400)
        username = username.strip().lower()
        if not re.fullmatch(r"[a-z0-9_]{3,32}", username) or not 10 <= len(password) <= 128:
            return jsonify(error="invalid_credentials", message="Use 3–32 letters, digits or underscores for the username and a password of 10–128 characters."), 400
        if action == "register":
            try:
                user = store.register(username, generate_password_hash(password))
            except IntegrityError:
                return jsonify(error="username_unavailable", message="That username is unavailable."), 409
        else:
            user = store.user(username=username)
            if not user or not check_password_hash(user["password"], password):
                return jsonify(error="invalid_credentials", message="Username or password is incorrect."), 401
        session.clear(); session.permanent = True
        session.update(uid=user["id"], csrf=secrets.token_hex(32))
        return jsonify(user={"username": user["username"]}, csrf=session["csrf"])

    def require_owner():
        if not g.owner:
            abort(401)

    @app.get("/api/v1/history")
    def history():
        require_owner()
        return jsonify(items=store.list(g.owner))

    @app.delete("/api/v1/history")
    @app.delete("/api/v1/history/<identifier>")
    def delete_history(identifier=None):
        require_owner()
        removed = store.remove(g.owner, identifier)
        if identifier and not removed:
            abort(404)
        return jsonify(deleted=removed)

    def save_result(result, source):
        if g.owner:
            result["source_label"] = source[:120]
            result["history_id"] = store.save(g.owner, result)
        return result

    app.extensions["save_result"] = save_result

    def image_from_base64(value):
        import cv2
        import numpy as np
        if not isinstance(value, str) or len(value) > 600000:
            raise ValueError("Invalid image size")
        raw = base64.b64decode(value, validate=True)
        # Inspect dimensions before native decoding to bound image allocation.
        from PIL import Image
        from io import BytesIO
        with Image.open(BytesIO(raw)) as im:
            if im.format != "JPEG" or im.width > 640 or im.height > 480 or im.width < 32 or im.height < 32:
                raise ValueError("Expected JPEG up to 640x480")
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image")
        return frame

    preview_detector = None

    @app.post("/api/v1/lips")
    def lips():
        nonlocal preview_detector
        require_owner()
        if not store.allow("preview:" + g.owner, maximum=120):
            return jsonify(error="rate_limited"), 429
        if not slot.acquire(blocking=False):
            return jsonify(error="busy"), 429
        try:
            import mediapipe as mp
            import cv2
            from preprocessing import DEFAULT_MODEL, crop_lips
            if preview_detector is None:
                preview_detector = mp.tasks.vision.FaceLandmarker.create_from_options(
                    mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(DEFAULT_MODEL)), num_faces=2, output_face_blendshapes=True))
            try:
                body = request.get_json(silent=True) or {}
                if not isinstance(body, dict):
                    raise ValueError("Expected image object")
                frame = image_from_base64(body.get("image"))
                detected = preview_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if len(detected.face_landmarks) != 1:
                    return jsonify(box=None, message="Keep exactly one face in view.")
                _, box = crop_lips(frame, detected.face_landmarks[0])
                from audiovisual import lip_expression
                shapes = detected.face_blendshapes[0] if detected.face_blendshapes else []
                expression = lip_expression(detected.face_landmarks[0], frame.shape[1], frame.shape[0], shapes)
                return jsonify(box=box, width=frame.shape[1], height=frame.shape[0], expression=expression, message=expression['label'])
            except (ValueError, OSError):
                return jsonify(box=None, message="Center your face and keep your lips visible.")
        finally:
            slot.release()

    @app.post("/api/v1/recognize-frames")
    def recognize_frames_route():
        require_owner()
        if not slot.acquire(blocking=False):
            return jsonify(error="busy", message="Recognition is busy. Try again shortly."), 429
        try:
            body = request.get_json(silent=True) or {}
            if not isinstance(body, dict):
                abort(400)
            images = body.get("frames")
            timestamps = body.get("timestamps_ms")
            engine = body.get('engine', 'original')
            if not isinstance(engine, str) or engine not in {'original', 'lipnet-reference', 'auto-avsr'}:
                return jsonify(error='invalid_options'), 400
            if engine == 'auto-avsr':
                from general_bridge import available
                if not available():
                    return jsonify(error='model_unavailable', message='The English sentence model is not installed.'), 503
            if engine == 'lipnet-reference' and not Path(app.config['REFERENCE_WEIGHTS_PATH']).is_file():
                return jsonify(error='model_unavailable', message='The public reference model is not installed.'), 503
            if not isinstance(images, list) or len(images) != 75 or not isinstance(timestamps, list) or len(timestamps) != 75:
                return jsonify(error="invalid_frames", message="Capture exactly 75 frames."), 400
            import numpy as np
            try:
                from input_quality import assess_timing, assess_crops
                timing = assess_timing(timestamps)
                if engine == 'auto-avsr':
                    import tempfile
                    import cv2
                    from general_bridge import recognize_general
                    decoded = [image_from_base64(value) for value in images]
                    shape = decoded[0].shape
                    if any(frame.shape != shape for frame in decoded):
                        raise ValueError('Camera dimensions changed during capture.')
                    if not 1 <= timing['effective_fps'] <= 120:
                        raise ValueError('Unsupported capture rate.')
                    with tempfile.TemporaryDirectory(prefix='visionspeak-camera-',dir=app.config['UPLOAD_ROOT']) as directory:
                        video = Path(directory)/'capture.avi'
                        writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),timing['effective_fps'],(shape[1],shape[0]))
                        try:
                            if not writer.isOpened():
                                raise ValueError('Could not prepare camera sequence.')
                            for frame in decoded:
                                writer.write(frame)
                        finally:
                            writer.release()
                        result = recognize_general(video)
                    result.pop('input_report',None)
                    result['capture_timing']=timing
                    result['capture_duration_seconds']=timing['duration_seconds']
                    result['warnings'].extend(timing['warnings'])
                    result['request_id']=g.request_id
                    return jsonify(save_result(result,'Webcam capture'))
                from preprocessing import LipDetector, normalize_sequence
                warmup()
                if engine == 'lipnet-reference':
                    from reference_lipnet import ReferenceCropper, normalize_reference, recognize_reference_frames
                    detector_context = LipDetector(cropper=ReferenceCropper())
                else:
                    detector_context = LipDetector()
                with detector_context as detector:
                    crops = [detector.detect(image_from_base64(value), index * 40)[0] for index, value in enumerate(images)]
                from recognition import recognize_frames
                if engine == 'lipnet-reference':
                    result = recognize_reference_frames(normalize_reference(crops), Path(app.config['REFERENCE_WEIGHTS_PATH']))
                else:
                    result = recognize_frames(normalize_sequence(crops), Path(app.config["WEIGHTS_PATH"]), mode="webcam")
            except (ValueError, OSError):
                return jsonify(error="invalid_capture", message="Capture failed. Keep one face centered in good light, then try again."), 422
            result.pop("weights", None)
            if engine == 'lipnet-reference':
                import cv2
                quality = assess_crops(np.stack([cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) for crop in crops]))
            else:
                quality = assess_crops(np.asarray(crops, dtype=np.uint8))
            result["capture_duration_seconds"] = timing["duration_seconds"]
            result["capture_timing"] = timing
            result["input_quality"] = quality
            result.setdefault("warnings", []).extend(quality["warnings"] + timing["warnings"])
            result["request_id"] = g.request_id
            return jsonify(save_result(result, "Webcam capture"))
        finally:
            slot.release()

    return guard
