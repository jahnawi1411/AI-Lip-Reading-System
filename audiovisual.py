"""Local audio transcription and observable lip-shape measurements."""
import math
import os
from functools import lru_cache
from itertools import chain
from pathlib import Path

import numpy as np


def lip_expression(landmarks, width, height, blendshapes=None):
    """Map mouth-only blendshape coefficients to observable gestures.

    Thresholds are application heuristics, not calibrated probabilities.
    Pout means rounded/projected lips; it does not imply sadness.
    Keep the geometric opening separately because a smile can be open or closed.
    """
    def point(i):
        return np.array([landmarks[i].x * width, landmarks[i].y * height])
    mouth_width = float(np.linalg.norm(point(61) - point(291)))
    opening = float(np.linalg.norm(point(13) - point(14))) / max(mouth_width, 1)
    if not math.isfinite(opening) or not math.isfinite(mouth_width) or mouth_width < 2:
        raise ValueError("Invalid mouth landmarks")
    opening_label = "closed lips" if opening < .08 else "slightly open lips" if opening < .25 else "open mouth"
    names = ('mouthSmileLeft', 'mouthSmileRight', 'mouthPucker', 'mouthFunnel')
    scores = {shape.category_name: float(shape.score) for shape in (blendshapes or [])
              if shape.category_name in names}
    available = all(name in scores and math.isfinite(scores[name]) and 0 <= scores[name] <= 1 for name in names)
    label = opening_label
    if available:
        smile = min(scores['mouthSmileLeft'], scores['mouthSmileRight'])
        rounded = max(scores['mouthPucker'], scores['mouthFunnel'])
        # Require both corners for a smile, and a margin over conflicting signals.
        if smile >= .5 and smile - rounded >= .15:
            label = 'smiling lips'
        elif rounded >= .55 and rounded - smile >= .15:
            label = 'pursed lips' if opening < .08 and scores['mouthPucker'] >= .55 else 'pouting lips'
    return {"label": label, "opening_label": opening_label,
            "opening_ratio": round(opening, 3), "gesture_model_available": available,
            "mouth_scores": {name: round(score, 4) for name, score in scores.items()
                             if math.isfinite(score) and 0 <= score <= 1},
            "method": "mouth blendshapes + geometry" if available else "mouth geometry only"}


def expression_timeline(path, start, duration):
    import cv2
    from preprocessing import LipDetector
    samples = []
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 1 <= fps <= 120:
            raise ValueError("Invalid video frame rate")
        def measure(frame, landmarks, blendshapes):
            return lip_expression(landmarks, frame.shape[1], frame.shape[0], blendshapes)
        with LipDetector(cropper=measure, include_blendshapes=True) as detector:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            for index in range(math.ceil(duration * fps)):
                # Decode in sequence, but only materialize the five frames per
                # second that the expression timeline actually uses.
                if not cap.grab():
                    break
                if index % max(1, round(fps / 5)):
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if frame.shape[0] * frame.shape[1] > 3840 * 2160:
                    raise ValueError("Video resolution must not exceed 4K")
                scale = min(1., 640 / frame.shape[1], 480 / frame.shape[0])
                if scale < 1:
                    frame = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)),
                                              max(1, round(frame.shape[0] * scale))),
                                       interpolation=cv2.INTER_AREA)
                timestamp = start + index / fps
                try:
                    shape = detector.detect(frame, round(timestamp * 1000))
                except ValueError:
                    shape = {"label": "face unavailable", "opening_ratio": None}
                samples.append({"time": round(timestamp, 3), **shape})
    finally:
        cap.release()
    return {"samples": samples, "description": "Visible lip gestures estimated from mouth movement: smiling, pouting, pursed, closed, slightly open or open. These labels do not identify emotions. Speech movements can resemble gestures; scores are not confidence probabilities."}


def summarize_expressions(expressions):
    """Use the same gesture summary for combined and expression-only results."""
    from collections import Counter
    samples = expressions['samples']
    counts = Counter(sample['label'] for sample in samples if sample.get('opening_ratio') is not None)
    expressions['summary'] = dict(counts)
    expressions['detected_samples'] = sum(counts.values())
    expressions['total_samples'] = len(samples)
    expressions['gesture_samples'] = sum(sample.get('gesture_model_available', False) for sample in samples)
    return counts.most_common(1)[0][0] if counts else None


def attach_lip_expressions(result, path, start=0, duration=None):
    """Analyze the same selected clip without replacing its recognized words."""
    try:
        shapes = expression_timeline(path, start, 16.0 if duration is None else duration)
        dominant = summarize_expressions(shapes)
    except (ValueError, OSError, ImportError, RuntimeError):
        shapes = {'samples': [], 'summary': {}, 'description': 'Lip expressions could not be read from this clip. Try keeping your face clearly in view.'}
        dominant = None
    result['lip_expressions'] = shapes
    result['dominant_expression'] = dominant
    return result


def recognize_expressions(path, start=0, duration=None):
    """Analyze visible lip shapes independently of speech or audio models."""
    duration = 300.0 if duration is None else duration
    if not math.isfinite(start) or not 0 <= start <= 60 or not math.isfinite(duration) or not 0 < duration <= 300:
        raise ValueError("Choose a start from 0–60 seconds and a duration up to 300 seconds.")
    try:
        expressions = expression_timeline(path, start, duration)
    except (OSError, ImportError) as exc:
        raise RuntimeError("Lip analysis unavailable. Install the application dependencies and run prepare_video.py --download-model.") from exc
    samples = expressions['samples']
    if not samples:
        raise ValueError("No readable frames in the selected interval. Choose an earlier start or another video.")
    dominant = summarize_expressions(expressions)
    return {
        'engine': 'lip-expression', 'task': 'lip-expression', 'text': '', 'usable_text': None,
        'status': 'measured' if dominant else 'face_unavailable',
        'dominant_expression': dominant, 'audio_used': False,
        'lip_expressions': expressions, 'review_required': True,
        'message': ('Visible lip gestures estimated. Review the timeline; mouth movements during speech can resemble gestures.'
                    if dominant else 'No single visible face found. Face the camera with your mouth unobstructed.'),
        'selected_start_seconds': start, 'selected_duration_seconds': duration,
    }


@lru_cache(maxsize=1)
def speech_model():
    from faster_whisper import WhisperModel
    return WhisperModel(os.getenv("VISIONSPEAK_AUDIO_MODEL", "base"), device="cpu",
                        compute_type="int8", cpu_threads=4,
                        download_root=str(Path(__file__).parent / "models" / "whisper"))


def read_audio(path, start, duration):
    """Decode only the bounded interval, preserving the container's audio offset."""
    import av
    audio = np.zeros(round(duration * 16000), dtype=np.float32)
    found = False
    last_sample = 0
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        origin = float(container.start_time or 0) / av.time_base
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        cursor = float(stream.start_time or 0) * float(stream.time_base) - origin
        for frame in chain(container.decode(stream), [None]):
            for chunk in resampler.resample(frame):
                timestamp = float(chunk.pts * chunk.time_base) - origin if chunk.pts is not None else cursor
                values = chunk.to_ndarray().reshape(-1)
                cursor = timestamp + len(values) / 16000
                offset = round((timestamp - start) * 16000)
                lo, hi = max(0, offset), min(len(audio), offset + len(values))
                if hi > lo:
                    audio[lo:hi] = values[lo-offset:hi-offset]
                    found = True
                    last_sample = max(last_sample, hi)
                if timestamp >= start + duration:
                    return audio[:last_sample] if found else None
    # Preserve the audio offset and real silence, but don't send unrecorded
    # padding (e.g. 13 seconds after a 3-second webcam clip) through inference.
    return audio[:last_sample] if found else None


def recognize_audiovisual(path, start=0, duration=None):
    duration = 300.0 if duration is None else duration
    if not math.isfinite(start) or not 0 <= start <= 60 or not math.isfinite(duration) or not 0 < duration <= 300:
        raise ValueError("Choose a start from 0–60 seconds and a duration up to 300 seconds.")
    try:
        audio = read_audio(path, start, duration)
    except ImportError as exc:
        raise RuntimeError("Install requirements-audio.txt in the application environment.") from exc
    except Exception as exc:
        raise ValueError("Could not decode this video's audio.") from exc
    words, text = [], ""
    language = None
    status = "no_audio" if audio is None else "no_speech"
    if audio is not None and np.max(np.abs(audio)) > .0001:
        try:
            model = speech_model()
        except Exception as exc:
            raise RuntimeError("Audio model unavailable. Install requirements-audio.txt and run setup_audio.py to download the model.") from exc
        beam_size = max(1, min(5, int(os.getenv("VISIONSPEAK_AUDIO_BEAM_SIZE", "5"))))
        segments, info = model.transcribe(audio, beam_size=beam_size, best_of=1, vad_filter=True,
                                         word_timestamps=True, condition_on_previous_text=False, language="en")
        parts = []
        for segment in segments:
            parts.append(segment.text.strip())
            for word in segment.words or []:
                words.append({"word": word.word.strip(), "start": round(start + word.start, 3),
                              "end": round(start + word.end, 3)})
        text = " ".join(parts).strip()
        language = info.language
        status = "candidate" if text else "no_speech"
    try:
        expressions = expression_timeline(path, start, duration)
    except (ValueError, OSError, ImportError):
        expressions = {"samples": [], "description": "Lip analysis unavailable. Check the face model and visible face."}
    dominant = summarize_expressions(expressions)
    message = {"no_audio": "No audio in the selected interval. Choose a video with sound or use a visual model.",
               "no_speech": "No speech detected in the selected interval.",
               "candidate": "Transcript from audio. Lip shapes are measured separately; review the words for accuracy."}[status]
    return {"engine": "audio-video", "text": text, "usable_text": text or None, "status": status,
            "audio_used": audio is not None, "language": language, "words": words,
            "lip_expressions": expressions, "dominant_expression": dominant, "message": message, "review_required": True,
            "selected_start_seconds": start, "selected_duration_seconds": duration}
