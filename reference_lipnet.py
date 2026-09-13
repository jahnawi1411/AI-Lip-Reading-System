"""Inference adapter for rizkiarm/LipNet's Keras 2.0.2 model2 checkpoint.

Architecture source (MIT): https://github.com/rizkiarm/LipNet/blob/
bec592aa1e66378bc7ab33f6110f1c8a431b3b9a/lipnet/model2.py
Uses RGB, width-before-height input, legacy reset-before GRUs and hard sigmoid.
This adapter is experimental until evaluated with matched mouth crops.
"""
from pathlib import Path
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / 'models/lipnet-reference/unseen-weights178.h5'
_CACHE = {}


def normalize_reference(crops):
    values = np.asarray(crops)
    if values.shape != (75, 50, 100, 3) or values.dtype != np.uint8:
        raise ValueError('Expected 75 RGB uint8 crops of height 50 and width 100.')
    return values.transpose(0, 2, 1, 3).astype(np.float32) / 255.0


class ReferenceCropper:
    """MediaPipe adaptation of upstream's fixed first-frame scale RGB crop.

    Landmark estimates differ from upstream dlib; this is a separately measured
    preprocessing mode, not an assertion of identical input distributions.
    """
    def __init__(self):
        self.ratio = None

    def __call__(self, frame, landmarks):
        import cv2
        from PIL import Image
        ids = (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
               78, 81, 13, 311, 308, 402, 14, 178)
        h, w = frame.shape[:2]
        points = np.asarray([(landmarks[i].x * w, landmarks[i].y * h) for i in ids])
        if not np.isfinite(points).all():
            raise ValueError('Invalid lip landmarks.')
        if self.ratio is None:
            span = points[:, 0].max() * 1.19 - points[:, 0].min() * .81
            if span < 4:
                raise ValueError('Mouth is too small.')
            self.ratio = 100.0 / span
        center = points.mean(axis=0) * self.ratio
        x, y = int(center[0] - 50), int(center[1] - 25)
        width, height = int(w * self.ratio), int(h * self.ratio)
        if width * height > 3840 * 2160:
            raise ValueError('The mouth is too small for bounded resizing. Move closer.')
        if x < 0 or y < 0 or x + 100 > width or y + 50 > height:
            raise ValueError('Lips are too close to the image edge; center the speaker.')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))
        box = tuple(int(v / self.ratio) for v in (x, y, x + 100, y + 50))
        return resized[y:y + 50, x:x + 100].copy(), box


def prepare_reference_video(path, start=0.0, duration=None):
    from preprocessing import LipDetector, prepare_video
    frames, crops, report = prepare_video(path, start=start, duration=duration,
        detector_factory=lambda model: LipDetector(model, cropper=ReferenceCropper()),
        normalizer=normalize_reference)
    report['normalization'] = 'RGB / 255; width-before-height; first-frame fixed scale'
    return frames, crops, report


def legacy_hard_sigmoid(x):
    import tensorflow as tf
    return tf.clip_by_value(0.2 * x + 0.5, 0.0, 1.0)


def get_reference_model(weights=DEFAULT_WEIGHTS):
    import h5py
    import tensorflow as tf
    tf.keras.utils.register_keras_serializable(package='VisionSpeak')(legacy_hard_sigmoid)
    key = str(Path(weights).resolve())
    if key in _CACHE:
        return _CACHE[key]
    L = tf.keras.layers
    inputs = L.Input((75, 100, 50, 3))
    x = inputs
    weighted = []
    for i, filters in enumerate((32, 64, 96), 1):
        spatial = 5 if i < 3 else 3
        x = L.ZeroPadding3D((1, spatial // 2, spatial // 2))(x)
        conv = L.Conv3D(filters, (3, spatial, spatial), strides=(1, 2, 2) if i == 1 else (1, 1, 1), name=f'conv{i}')
        bn = L.BatchNormalization(epsilon=1e-3, name=f'batc{i}')
        x = L.Activation('relu')(bn(conv(x)))
        x = L.MaxPooling3D((1, 2, 2))(x)
        weighted.extend([(conv, f'conv{i}'), (bn, f'batc{i}')])
    x = L.TimeDistributed(L.Flatten())(x)
    for i in (1, 2):
        recurrent = L.Bidirectional(L.GRU(256, return_sequences=True, reset_after=False,
                                         recurrent_activation=legacy_hard_sigmoid), name=f'bidirectional_{i}')
        x = recurrent(x)
        weighted.append((recurrent, f'bidirectional_{i}'))
    dense = L.Dense(28, activation='softmax', name='dense1')
    model = tf.keras.Model(inputs, dense(x))
    weighted.append((dense, 'dense1'))
    with h5py.File(weights, 'r') as archive:
        for layer, name in weighted:
            group = archive[name]
            names = [n.decode() if isinstance(n, bytes) else n for n in group.attrs['weight_names']]
            layer.set_weights([np.asarray(group[n]) for n in names])
    _CACHE[key] = model
    return model


def predict_reference(frames, weights=DEFAULT_WEIGHTS, decoder='greedy'):
    from recognition import decode_probabilities, decode_beam
    frames = np.asarray(frames, dtype=np.float32)
    if frames.shape != (75, 100, 50, 3) or not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError('Expected 75 RGB mouth crops, W=100 H=50, scaled to [0, 1].')
    p = get_reference_model(weights)(frames[None], training=False).numpy()[0]
    # Map the reference's 28 classes into the application's vocabulary, preserving
    # probabilities. No dictionary, spelling correction or transcript is used.
    mapped = np.zeros((75, 41), np.float32)
    mapped[:, 1:27] = p[:, :26]
    mapped[:, 39] = p[:, 26]
    mapped[:, 40] = p[:, 27]
    if decoder not in {'beam', 'greedy'}:
        raise ValueError('Decoder must be greedy or beam.')
    return decode_beam(mapped) if decoder == 'beam' else decode_probabilities(mapped)


def recognize_reference_frames(frames, weights=None, decoder='greedy'):
    from recognition import weight_fingerprint
    weights = Path(weights) if weights is not None else DEFAULT_WEIGHTS
    result = predict_reference(frames, weights, decoder)
    result.update(engine='lipnet-reference', decoder=decoder, review_required=True,
                  usable_text=result['text'] if result['status'] == 'decoded' else None,
                  message='GRID vocabulary candidate only. This model cannot reliably transcribe ordinary conversation.',
                  review_reasons=['Public regression samples are not a held-out accuracy evaluation.',
                                  'A readable output can still be wrong, especially outside GRID phrases.'],
                  domain='GRID six-word commands', preprocessing='mediapipe-rgb-reference-v1',
                  weights=str(weights.resolve()), weights_sha256=weight_fingerprint(weights),
                  audio_used=False,
                  warnings=['Experimental public LipNet model trained on GRID commands; general speech is unsupported.',
                            'MediaPipe landmarks replace upstream dlib; confidence is not calibrated.'])
    return result


def recognize_reference_file(path, weights=None, decoder='greedy', start=0.0, duration=None):
    frames, _, report = prepare_reference_video(path, start, duration)
    result = recognize_reference_frames(frames, weights, decoder)
    result.update(source=str(Path(path).resolve()), input_report=report, input_quality=report['input_quality'])
    result['warnings'].extend(report['input_quality']['warnings'])
    if not 2.5 <= report['duration_seconds'] <= 3.5:
        result['warnings'].append('Use a roughly three-second GRID phrase; resampling other durations changes speech timing.')
    return result
