import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset import load_manifest
from recognition import BLANK, decode_probabilities, error_counts, load_sequence
from recognition import recognize_file


class RecognitionTests(unittest.TestCase):
    def test_auto_routes_archives_and_rejects_time_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.npz"
            frames = np.random.default_rng(1).normal(size=(75, 46, 140, 1)).astype(np.float32)
            np.savez(path, frames=frames)
            with patch("recognition.recognize_frames", return_value={}) as predict:
                result = recognize_file(path)
                self.assertEqual(predict.call_args.args[2], "prepared")
                self.assertEqual(result["source"], str(path.resolve()))
            with self.assertRaisesRegex(ValueError, "Time selection"):
                recognize_file(path, start=1)

    def probabilities(self, labels):
        p = np.full((len(labels), BLANK + 1), .001)
        p[np.arange(len(labels)), labels] = .96
        return p / p.sum(axis=1, keepdims=True)

    def test_ctc_repeats_and_blank_separation(self):
        result = decode_probabilities(self.probabilities([1, 1, BLANK, 1, 2, 2, BLANK]))
        self.assertEqual(result["text"], "aab")
        self.assertIsNone(result["confidence"])
        self.assertGreater(result["model_score"], .9)

    def test_blank_only_has_no_confidence_score(self):
        result = decode_probabilities(self.probabilities([BLANK] * 3))
        self.assertEqual(result["status"], "no_speech_decoded")
        self.assertIsNone(result["model_score"])

    def test_unknown_is_not_silently_removed(self):
        result = decode_probabilities(self.probabilities([0, BLANK, 1]))
        self.assertEqual(result["text"], "\ufffda")
        self.assertEqual(result["status"], "unknown_tokens")

    def test_invalid_probabilities_rejected(self):
        for p in [np.zeros((2, 41)), np.full((2, 41), np.nan), np.ones((2, 5))]:
            with self.assertRaises(ValueError):
                decode_probabilities(p)

    def test_error_counts(self):
        self.assertEqual(error_counts("Hii, I am Jahnawi", "hii i am jahnawi")["word_errors"], 0)
        counts = error_counts("a b", "a c d")
        self.assertEqual(counts["word_errors"], 2)
        self.assertEqual(counts["reference_words"], 2)
        self.assertEqual(error_counts("abc", "")["character_errors"], 3)
        with self.assertRaises(ValueError):
            error_counts("", "a")

    def test_bad_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sequence.npz"
            np.savez(path, frames=np.zeros((75, 46, 140, 1)))
            with self.assertRaisesRegex(ValueError, "variation"):
                load_sequence(path)
            np.savez(path, frames=np.ones((2, 2)))
            with self.assertRaises(ValueError):
                load_sequence(path)

    def test_manifest_rejects_speaker_leakage_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            frames = np.random.default_rng(42).normal(size=(75, 46, 140, 1)).astype(np.float32)
            np.savez(folder / "a.npz", frames=frames)
            rows = [{"sequence": "a.npz", "text": "hello", "speaker": "S1", "split": "train"},
                    {"sequence": "b.npz", "text": "hi", "speaker": "s1", "split": "test"}]
            path = folder / "manifest.json"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Speaker leakage"):
                load_manifest(path)
            rows[1].update(sequence="a.npz", speaker="s2")
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
