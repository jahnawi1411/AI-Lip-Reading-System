"""Explain observable capture problems; these heuristics are not confidence."""
import cv2
import numpy as np


def assess_crops(crops):
    images = np.asarray(crops)
    if images.ndim != 3 or len(images) < 2 or images.dtype != np.uint8:
        raise ValueError('Quality assessment requires unnormalized grayscale uint8 crops.')
    values = images.astype(np.float32)
    brightness = float(values.mean())
    contrast = float(values.std(axis=(1, 2)).mean())
    sharpness = float(np.median([cv2.Laplacian(frame, cv2.CV_32F).var() for frame in images]))
    change = float(np.abs(np.diff(values, axis=0)).mean())
    warnings = []
    if brightness < 40:
        warnings.append('The mouth region is dark. Add light in front of your face.')
    elif brightness > 215:
        warnings.append('The mouth region is very bright. Reduce direct light or exposure.')
    if contrast < 12:
        warnings.append('Little detail is visible around the lips. Improve lighting and move closer.')
    if sharpness < 20:
        warnings.append('The mouth region may be blurred. Keep the camera steady and check focus.')
    if change < 1:
        warnings.append('Very little visual movement was captured. Check the speaking interval.')
    return {'assessment': 'advisory', 'thresholds_calibrated': False,
            'mean_brightness': brightness, 'mean_contrast': contrast,
            'median_laplacian_variance': sharpness, 'mean_frame_change': change,
            'warnings': warnings,
            'note': 'Image heuristics only. Motion may come from the camera; sharpness varies with resolution. These values do not measure speech accuracy.'}


def assess_timing(timestamps_ms):
    ticks = np.asarray(timestamps_ms, dtype=np.float64)
    if ticks.shape != (75,) or not np.isfinite(ticks).all() or ticks[0] < 0 or np.any(np.diff(ticks) <= 0):
        raise ValueError('Expected 75 finite, strictly increasing capture timestamps.')
    duration = float(ticks[-1] - ticks[0])
    if duration <= 0 or duration > 10000:
        raise ValueError('Capture must finish within 10 seconds.')
    intervals = np.diff(ticks)
    warnings = []
    if not 2500 <= duration <= 3500:
        warnings.append('Capture timing differs from the expected three seconds. Close busy tabs and retry.')
    if float(intervals.max()) > 120:
        warnings.append('Capture paused between frames. Keep the tab visible and reduce device load.')
    return {'duration_seconds': duration / 1000, 'effective_fps': 74000 / duration,
            'max_frame_gap_ms': float(intervals.max()),
            'frame_interval_std_ms': float(intervals.std()), 'warnings': warnings}
