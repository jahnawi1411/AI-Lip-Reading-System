"""VisionSpeak Part 1: visual-only lip detection and sequence preparation."""
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
LIP_IDS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
           409, 270, 269, 267, 0, 37, 39, 40, 185)


def crop_lips(frame, landmarks):
    """Return a grayscale 46x140 crop and an exclusive pixel bounding box."""
    height, width = frame.shape[:2]
    points = np.array([(landmarks[i].x * width, landmarks[i].y * height)
                       for i in LIP_IDS])
    if not np.isfinite(points).all():
        raise ValueError("Invalid lip landmarks.")
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    # Preserve aspect ratio; include context around the lips.
    crop_width = max((high[0] - low[0]) * 1.5,
                     (high[1] - low[1]) * 1.5 * 140 / 46, 4)
    crop_height = crop_width * 46 / 140
    x1, y1 = np.floor(center - [crop_width / 2, crop_height / 2]).astype(int)
    x2, y2 = np.ceil(center + [crop_width / 2, crop_height / 2]).astype(int)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError("Lips are too close to the image edge; center the speaker.")
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(gray, (140, 46), interpolation=cv2.INTER_AREA)
    return crop, (int(x1), int(y1), int(x2), int(y2))


class LipDetector:
    """One detector per clip, with increasing timestamps in milliseconds."""
    def __init__(self, model_path=DEFAULT_MODEL, cropper=crop_lips):
        if not Path(model_path).is_file():
            raise FileNotFoundError("Face model missing. Run: python prepare_video.py --download-model")
        import mediapipe as mp
        self.mp = mp
        self.cropper = cropper
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if len(result.face_landmarks) != 1:
            raise ValueError("Exactly one visible face is required in every sampled frame.")
        return self.cropper(frame, result.face_landmarks[0])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.detector.close()


def normalize_sequence(crops):
    frames = np.asarray(crops, dtype=np.float32)[..., np.newaxis]
    if frames.shape != (75, 46, 140, 1):
        raise ValueError("Expected exactly 75 grayscale crops of size 46 x 140.")
    return (frames - frames.mean()) / max(float(frames.std()), 1e-6)


def prepare_video(path, model_path=DEFAULT_MODEL, detector_factory=LipDetector,
                  start=0.0, duration=None, normalizer=normalize_sequence):
    """Sample a short constant-frame-rate clip uniformly, preserving frame order.

    Outputs match the legacy model's shape only. The crop distribution must be
    evaluated or used for retraining before connecting it to that model.
    """
    path = Path(path)
    if not np.isfinite(start) or not 0 <= start <= 60:
        raise ValueError("Start time must be between 0 and 60 seconds.")
    if duration is not None and (not np.isfinite(duration) or not 0 < duration <= 10):
        raise ValueError("Selected duration must be greater than 0 and at most 10 seconds.")
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open this video. Try an MP4 or AVI file.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or not 1 <= fps <= 120:
            raise ValueError("Video must report a frame rate between 1 and 120 FPS.")
        # First pass counts actual decoded frames instead of trusting metadata.
        count = 0
        # Avoid selecting an extra frame from floating-point boundary noise.
        first_index = int(np.ceil(start * fps - 1e-9))
        stop_index = int(np.ceil((start + duration) * fps - 1e-9)) if duration is not None else None
        while True:
            if stop_index is not None and count >= stop_index:
                break
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            if (count - first_index) / fps > 10:
                raise ValueError("Use a short clip of at most 10 seconds.")
            if frame.shape[0] * frame.shape[1] > 3840 * 2160:
                raise ValueError("Video resolution must not exceed 4K.")
        if count - first_index < 2:
            raise ValueError("Selected video interval contains fewer than two readable frames.")
        if stop_index is not None and count < stop_index:
            raise ValueError("Selected interval extends beyond the readable video; shorten --duration.")
    finally:
        cap.release()

    indices = np.rint(np.linspace(first_index, count - 1, 75)).astype(np.int32)
    selected = set(indices.tolist())
    crops, boxes = {}, {}
    cap = cv2.VideoCapture(str(path))
    try:
        with detector_factory(model_path) as detector:
            for index in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Video decoding ended unexpectedly on the second pass.")
                if index in selected:
                    try:
                        crops[index], boxes[index] = detector.detect(frame, round(index * 1000 / fps))
                    except ValueError as exc:
                        raise ValueError(f"Frame {index}: {exc}") from exc
    finally:
        cap.release()
    ordered = np.stack([crops[int(i)] for i in indices])
    normalized = normalizer(ordered)
    report = {
        "schema_version": 1, "source": path.name, "source_fps": fps,
        "source_frames": count, "duration_seconds": (count - first_index) / fps,
        "selected_start_seconds": first_index / fps,
        "selected_end_seconds": count / fps,
        "selected_source_frames": count - first_index,
        "sampled_frames": 75, "unique_sampled_frames": len(selected),
        "shape": list(normalized.shape), "dtype": "float32",
        "sampling": "uniform nearest frame over full clip; repeats for short clips",
        "normalization": "(grayscale - sequence mean) / max(sequence std, 1e-6)",
        "audio_used": False, "recognition_performed": False,
        "source_indices": indices.tolist(),
        "lip_boxes": [boxes[int(i)] for i in indices],
    }
    from input_quality import assess_crops
    quality_crops = ordered if ordered.ndim == 3 else np.stack([cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) for crop in ordered])
    report["input_quality"] = assess_crops(quality_crops)
    return normalized, ordered, report
