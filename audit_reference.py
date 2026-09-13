"""Compare explicit preprocessing hypotheses without changing application defaults."""
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from lip_reader import get_model, load_video
from recognition import decode_probabilities, error_counts


def main():
    path = Path('data/grid_check/id2_vcd_swwp2s.mpg')
    model, _ = get_model(Path('models/reference-checkpoint-96/checkpoint'))
    cap = cv2.VideoCapture(str(path))
    crops = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crops.append(tf.image.rgb_to_grayscale(frame)[190:236, 80:220, :])
    cap.release()
    raw = tf.stack(crops)
    unsigned = tf.cast(raw - tf.reduce_mean(raw), tf.float32) / tf.math.reduce_std(tf.cast(raw, tf.float32))
    results = []
    for name, frames in [('float', load_video(path)), ('upstream-uint8', unsigned)]:
        result = decode_probabilities(model(frames[None], training=False).numpy()[0])
        result.update(preprocessing=name, **error_counts('set white with p two soon', result['text']))
        results.append(result)
        print(json.dumps(result), flush=True)
    Path('outputs/reference-preprocessing-audit.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
