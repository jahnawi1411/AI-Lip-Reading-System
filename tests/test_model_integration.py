"""Inference-only checks: never train or save model weights."""
import unittest
from pathlib import Path
import numpy as np


class ModelIntegrationTests(unittest.TestCase):
    def test_existing_checkpoint_inference_preserves_weights(self):
        from lip_reader import get_model, MODEL_DIR, build_model_with_weights
        from recognition import recognize_frames, weight_fingerprint
        weights = MODEL_DIR / 'checkpoint.weights.h5'
        before = weight_fingerprint(weights)
        model, _ = get_model(weights)
        parameters = [p.copy() for p in model.get_weights()]
        frames = np.random.default_rng(12).normal(size=(75, 46, 140, 1)).astype(np.float32)
        result = recognize_frames(frames, weights, decoder='beam')
        self.assertEqual(result['decoder'], 'beam')
        self.assertTrue(result['review_required'])
        self.assertIsNone(result['confidence'])
        self.assertEqual(len(result['alternatives']), 3)
        for original, current in zip(parameters, model.get_weights()):
            np.testing.assert_array_equal(original, current)
        self.assertEqual(before, weight_fingerprint(weights))
        with self.assertRaises(FileNotFoundError):
            build_model_with_weights(Path('does-not-exist'))

    def test_beam_combines_paths_and_preserves_repeated_letters(self):
        from recognition import BLANK, decode_beam, decode_probabilities
        p = np.full((2, BLANK + 1), 1e-9, dtype=np.float32)
        p[:, 1], p[:, BLANK] = .4, .6
        p /= p.sum(axis=1, keepdims=True)
        self.assertEqual(decode_probabilities(p)['text'], '')
        self.assertEqual(decode_beam(p)['text'], 'a')
        p = np.full((4, BLANK + 1), 1e-9, dtype=np.float32)
        p[np.arange(4), [1, BLANK, 1, 2]] = 1
        p /= p.sum(axis=1, keepdims=True)
        self.assertEqual(decode_beam(p)['text'], 'aab')
        scores = [r['ctc_log_probability'] for r in decode_beam(p)['alternatives']]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == '__main__':
    unittest.main()
