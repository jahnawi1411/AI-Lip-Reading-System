import unittest
import numpy as np
from input_quality import assess_crops, assess_timing


class InputQualityTests(unittest.TestCase):
    def test_dark_static_capture_gives_actionable_advice(self):
        result = assess_crops(np.full((75, 46, 140), 10, np.uint8))
        self.assertTrue(any('dark' in w for w in result['warnings']))
        self.assertTrue(any('movement' in w for w in result['warnings']))
        self.assertFalse(result['thresholds_calibrated'])

    def test_regular_capture_and_stalled_capture(self):
        self.assertEqual(assess_timing(np.arange(75) * 40)['warnings'], [])
        ticks = np.arange(75) * 40
        ticks[30:] += 300
        result = assess_timing(ticks)
        self.assertTrue(any('paused' in w for w in result['warnings']))
        self.assertEqual(result['max_frame_gap_ms'], 340)

    def test_invalid_timing_rejected(self):
        for ticks in [np.zeros(75), np.arange(74), np.arange(75) * 200,
                      np.full(75, np.nan), -np.arange(75)]:
            with self.assertRaises(ValueError):
                assess_timing(ticks)


if __name__ == '__main__':
    unittest.main()
