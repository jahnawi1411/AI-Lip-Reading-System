"""Part 2 recognition interface shared by the CLI and future Flask integration."""
import hashlib
from pathlib import Path

import numpy as np

VOCAB = "abcdefghijklmnopqrstuvwxyz'?!123456789 "
BLANK = len(VOCAB) + 1  # zero is the legacy StringLookup unknown token


def validate_frames(frames):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.shape != (75, 46, 140, 1) or not np.isfinite(frames).all():
        raise ValueError("Expected finite float frames with shape (75, 46, 140, 1).")
    if float(frames.std()) < 1e-6:
        raise ValueError("The sequence has no visual variation; recognition is not meaningful.")
    return frames


def load_sequence(path):
    with np.load(path, allow_pickle=False) as archive:
        if "frames" not in archive:
            raise ValueError("Sequence archive must contain 'frames'.")
        return validate_frames(archive["frames"])


def decode_probabilities(probabilities):
    """Greedy CTC collapse; score is emitted-run support, not accuracy."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] != BLANK + 1:
        raise ValueError("Unexpected model output shape.")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Model output must contain normalized probabilities.")
    path = p.argmax(axis=1)
    characters, supports = [], []
    unknown = False
    start = 0
    while start < len(path):
        label = int(path[start])
        end = start + 1
        while end < len(path) and path[end] == label:
            end += 1
        if label != BLANK:
            characters.append(VOCAB[label - 1] if label else "\ufffd")
            unknown |= label == 0
            supports.append(float(p[start:end, label].max()))
        start = end
    text = "".join(characters).strip()
    score = float(np.exp(np.log(np.maximum(supports, 1e-12)).mean())) if text else None
    return {"text": text, "model_score": score, "confidence": None,
            "confidence_status": "uncalibrated",
            "score_definition": "geometric mean of peak probabilities in emitted nonblank CTC runs",
            "status": "unknown_tokens" if unknown else ("decoded" if text else "no_speech_decoded")}


def decode_beam(probabilities, beam_width=50, top_paths=3):
    """Rank CTC sequences using the existing network's probabilities only."""
    baseline = decode_probabilities(probabilities)
    if not 1 <= top_paths <= beam_width <= 100:
        raise ValueError("Require 1 <= top_paths <= beam_width <= 100.")
    import tensorflow as tf
    p = np.asarray(probabilities, dtype=np.float32)
    # TensorFlow expects logits; log(softmax probabilities) is equivalent up to
    # a per-frame constant. Blank is the last class in the original checkpoint.
    logits = np.log(np.maximum(p, np.finfo(np.float32).tiny))[:, None, :]
    paths, scores = tf.nn.ctc_beam_search_decoder(
        logits, [len(p)], beam_width=beam_width, top_paths=top_paths)
    alternatives = []
    for index, path in enumerate(paths):
        labels = path.values.numpy().tolist()
        text = "".join(VOCAB[label - 1] if label else "\ufffd" for label in labels).strip()
        log_probability = float(scores.numpy()[0, index])
        if not np.isfinite(log_probability):
            continue
        alternatives.append({"text": text, "ctc_log_probability": log_probability})
    if not alternatives:
        raise ValueError("The decoder returned no finite candidate sequences.")
    best = alternatives[0]
    return {"text": best["text"], "model_score": None, "confidence": None,
            "confidence_status": "uncalibrated",
            "score_definition": "CTC sequence log probabilities; higher ranks first, not calibrated accuracy",
            "ctc_log_probability": best["ctc_log_probability"],
            "alternatives": alternatives, "greedy_text": baseline["text"],
            "status": "unknown_tokens" if "\ufffd" in best["text"] else
                      ("decoded" if best["text"] else "no_speech_decoded")}


def weight_fingerprint(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.parent.glob(path.name + ".*"))
    if not files:
        raise FileNotFoundError(f'Weights not found: {path}')
    digest = hashlib.sha256()
    for file in files:
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def recognize_frames(frames, weights=None, mode="prepared", decoder="greedy", beam_width=50):
    from lip_reader import configure_tensorflow, get_model
    frames = validate_frames(frames)
    if decoder not in {"beam", "greedy"} or not 3 <= beam_width <= 100:
        raise ValueError("Decoder must be beam or greedy; beam width must be between 3 and 100.")
    configure_tensorflow()
    model, loaded = get_model(weights)
    probabilities = model(frames[np.newaxis], training=False).numpy()[0]
    result = (decode_beam(probabilities, beam_width) if decoder == "beam"
              else decode_probabilities(probabilities))
    result["decoder"] = decoder
    result["engine"] = 'original'
    result["review_required"] = True
    result["usable_text"] = result["text"] if result["status"] == "decoded" else None
    result["message"] = ("Candidate text needs review." if result["usable_text"] else
                         "Could not reliably recognize speech from this clip.")
    result["review_reasons"] = ["This checkpoint has no verified accuracy for this input."]
    if result["status"] != "decoded":
        result["review_reasons"].append("The output contains unknown tokens or no decoded speech.")
    if decoder == "beam" and result["text"] != result["greedy_text"]:
        result["review_reasons"].append("Beam and greedy decoding disagree.")
    result.update({"weights": str(loaded.resolve()), "weights_sha256": weight_fingerprint(loaded),
                   "preprocessing": mode, "audio_used": False,
                   "warnings": ["Recognition accuracy and confidence have not been validated on a labelled test set."]})
    if mode != "grid":
        result["warnings"].append("Lip crops and temporal sampling differ from the original GRID pipeline. Output is an experimental candidate from the unchanged checkpoint.")
    return result


def recognize_file(path, mode="auto", weights=None, decoder="greedy", beam_width=50,
                   start=0.0, duration=None, engine="original"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    if engine == 'auto-avsr':
        if mode not in {'auto','mediapipe'} or path.suffix.lower() == '.npz':
            raise ValueError('Sentence recognition requires video with automatic detection.')
        if weights is not None:
            raise ValueError('The sentence worker uses its separately verified checkpoint.')
        from general_bridge import recognize_general
        return recognize_general(path,start,duration)
    if engine == 'lipnet-reference':
        if mode not in {'auto', 'mediapipe'} or path.suffix.lower() == '.npz':
            raise ValueError('Public LipNet requires a video with automatic RGB mouth detection.')
        from reference_lipnet import recognize_reference_file
        return recognize_reference_file(path, weights, decoder, start, duration)
    if engine != 'original':
        raise ValueError('Unknown recognition engine.')
    if mode == "auto":
        mode = "prepared" if path.suffix.lower() == ".npz" else "mediapipe"
    if mode != "mediapipe" and (start != 0 or duration is not None):
        raise ValueError("Time selection is available only for video with mediapipe/auto mode.")
    report = None
    if mode == "prepared":
        frames = load_sequence(path)
    elif mode == "mediapipe":
        from preprocessing import prepare_video
        frames, _, report = prepare_video(path, start=start, duration=duration)
    elif mode == "grid":
        from lip_reader import load_video
        frames = load_video(path).numpy()
    else:
        raise ValueError("Mode must be prepared, mediapipe or grid.")
    result = recognize_frames(frames, weights, mode, decoder, beam_width)
    result["source"] = str(path.resolve())
    if report is not None:
        result["input_report"] = report
        result["input_quality"] = report["input_quality"]
        result["warnings"].extend(report["input_quality"]["warnings"])
        if not 2.5 <= report["duration_seconds"] <= 3.5:
            result["warnings"].append("The selected duration differs from the model's approximately 3-second GRID clips; uniform resampling changes speech timing.")
    return result


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def error_counts(reference, hypothesis):
    # Commas and periods are not in the checkpoint vocabulary; ignore them for scoring.
    reference, hypothesis = (" ".join(value.lower().replace(",", "").replace(".", "").split())
                             for value in (reference, hypothesis))
    if not reference:
        raise ValueError("A nonempty reference transcript is required.")
    return {"word_errors": edit_distance(reference.split(), hypothesis.split()),
            "reference_words": len(reference.split()),
            "character_errors": edit_distance(reference, hypothesis),
            "reference_characters": len(reference), "exact_match": reference == hypothesis}
