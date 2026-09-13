import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from preprocessing import crop_lips, normalize_sequence, prepare_video, LipDetector


class FakeDetector:
    def __init__(self, _):
        self.last_timestamp = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def detect(self, frame, timestamp):
        assert timestamp > self.last_timestamp
        self.last_timestamp = timestamp
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (140, 46)), (0, 0, 160, 120)


class PreprocessingTests(unittest.TestCase):
    def test_interval_selection_keeps_absolute_frame_order(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "interval.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
            self.assertTrue(writer.isOpened())
            for i in range(25):
                writer.write(np.full((120, 160, 3), i * 8, np.uint8))
            writer.release()
            _, crops, report = prepare_video(video, detector_factory=FakeDetector, start=.2, duration=.4)
            self.assertEqual(report["source_indices"][0], 5)
            self.assertEqual(report["source_indices"][-1], 14)
            self.assertEqual(report["selected_source_frames"], 10)
            self.assertTrue(np.all(np.diff(crops.mean(axis=(1, 2))) >= 0))
            with self.assertRaisesRegex(ValueError, "beyond"):
                prepare_video(video, detector_factory=FakeDetector, start=.5, duration=1)
            for options in ({"start": -1}, {"start": float("nan")}, {"duration": 0}):
                with self.assertRaises(ValueError):
                    prepare_video(video, detector_factory=FakeDetector, **options)

    def test_constant_sequence_is_finite(self):
        result = normalize_sequence(np.full((75, 46, 140), 128, dtype=np.uint8))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result == 0).all())

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            normalize_sequence(np.zeros((2, 46, 140)))

    def test_crop_and_bgr_conversion(self):
        landmarks = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
        landmarks[61].x, landmarks[291].x = .4, .6
        frame = np.zeros((200, 300, 3), np.uint8)
        frame[:, :, 2] = 255
        crop, box = crop_lips(frame, landmarks)
        self.assertEqual(crop.shape, (46, 140))
        self.assertTrue((crop == 76).all())
        self.assertLess(box[0], 120)
        self.assertGreater(box[2], 180)

    def test_edge_crop_rejected(self):
        landmarks = [SimpleNamespace(x=0, y=0) for _ in range(478)]
        with self.assertRaisesRegex(ValueError, "edge"):
            crop_lips(np.zeros((100, 100, 3), np.uint8), landmarks)

    def test_missing_model(self):
        with self.assertRaises(FileNotFoundError):
            LipDetector("missing-model.task")

    def test_video_sampling_and_export_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "synthetic.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
            self.assertTrue(writer.isOpened())
            for i in range(10):
                writer.write(np.full((120, 160, 3), i * 20, np.uint8))
            writer.release()
            frames, crops, report = prepare_video(video, detector_factory=FakeDetector)
            self.assertEqual(frames.shape, (75, 46, 140, 1))
            self.assertEqual(report["source_frames"], 10)
            self.assertEqual(report["unique_sampled_frames"], 10)
            self.assertEqual(report["source_indices"][0], 0)
            self.assertEqual(report["source_indices"][-1], 9)
            self.assertTrue(np.all(np.diff(crops.mean(axis=(1, 2))) >= 0))
            self.assertAlmostEqual(float(frames.mean()), 0, places=5)
            self.assertAlmostEqual(float(frames.std()), 1, places=5)

    def test_unreadable_video(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "broken.mp4"
            video.write_bytes(b"not a video")
            with self.assertRaises(ValueError):
                prepare_video(video, detector_factory=FakeDetector)

    def test_detection_failure_is_not_silently_dropped(self):
        class NoFace(FakeDetector):
            def detect(self, frame, timestamp):
                raise ValueError("No face")
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "blank.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(2):
                writer.write(np.zeros((120, 160, 3), np.uint8))
            writer.release()
            with self.assertRaisesRegex(ValueError, "Frame 0: No face"):
                prepare_video(video, detector_factory=NoFace)


if __name__ == "__main__":
    unittest.main()
