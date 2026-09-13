# VisionSpeak: three-part implementation plan

The user subsequently authorized model improvements using public data. The active
progress record is [MODEL_PROGRESS.md](MODEL_PROGRESS.md); the original-model-only
scope and historical milestone statements below are retained as project history.

## Part 1 — Video input, lip detection and preprocessing (implemented)

Purpose: turn a short silent video into a repeatable visual input for model development.

- Read recorded videos using OpenCV; no audio decoding.
- Locate facial landmarks with MediaPipe Face Landmarker and extract the lip region.
- Require one visible speaker per sampled frame; report failures with frame numbers.
- Convert BGR frames correctly to grayscale, resize crops to 46 x 140, sample 75
  frames in order, and normalize over the sequence.
- Export a preview image, compressed NumPy sequence and JSON processing report.
- Keep the existing Flask app, notebook, model weights and inference pipeline intact.

Entry point: `prepare_video.py`. Reusable implementation: `preprocessing.py`.

## Part 2 — Visual speech recognition and evaluation

Current scope: use the existing checkpoint without training or changing weights.
Recognition includes automatic input selection, selectable speaking intervals,
greedy decoding, optional beam-search alternatives and explicit unusable-result handling.
See [PART2.md](PART2.md). The supplied sample still fails recognition; no accuracy
improvement or calibrated confidence is claimed.

- Document the existing model vocabulary and evaluate supplied labelled clips.
- Evaluate the existing TensorFlow checkpoint against its original GRID crop pipeline.
- Keep the existing weights unchanged; expose lip-crop inference as experimental.
- Decode text, measure word/character error rates, and assess lighting and pose failures.
- Design and calibrate a confidence indicator against held-out results; detection confidence
  must not be presented as speech recognition confidence.

Completion: reproducible evaluation and a tested inference interface, with measured limitations.

## Part 3 — Flask application and live experience (implemented)

The complete application is available through `python serve.py`. Upload, webcam
frame capture, repeating recognition, live lip boxes, user accounts, persistent
private history and export/delete are implemented. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md).
Model accuracy remains unresolved; no new model was trained.

- Integrate video upload and validated recognition results into the web interface.
- Add webcam capture, live lip-region overlays, and bounded recognition windows.
- Add prediction history, timestamps, deletion, and calibrated confidence display.
- Test browser permissions, upload limits, cleanup, responsive layout and error states.

Completion: demonstrate recorded and webcam workflows from input through saved results.

## Run Part 1 on Windows

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-part1.txt
.\.venv\Scripts\python.exe prepare_video.py --download-model
.\.venv\Scripts\python.exe prepare_video.py "C:\path\to\speech.mp4" --output outputs\first-clip
```

Use a short (up to 10 seconds), constant-frame-rate video with exactly one centered,
front-facing speaker and well-lit visible lips. Videos should have 1–120 FPS and
resolution no greater than 4K. Codec availability depends on OpenCV. Use a fresh
output directory per clip. Webcam input is scheduled for Part 3.

The separate environment needs only `requirements-part1.txt`; the original full-app
requirements still apply to `app.py` and `lip_reader.py`.

Generated files:

| File | Contents |
| --- | --- |
| `lip_preview.png` | Fifteen chronological mouth crops, read left to right, top to bottom |
| `sequence.npz` | `frames`: float32 array shaped (75, 46, 140, 1); `source_indices`: sampled frame indices |
| `report.json` | Source FPS, decoded count, duration, crop boxes, sampling and normalization details |

For Part 2, load the saved array using `np.load(path)["frames"]`, then add a batch
dimension to obtain `(1, 75, 46, 140, 1)`.

The shape matches the existing network, but the visual distribution differs from
its fixed `[190:236, 80:220]` crop. Shape compatibility does not establish recognition
accuracy. This milestone produces no predicted words or confidence scores.
Uniform sampling changes temporal speed for varying clip durations; Part 2 must
choose and validate a training-consistent duration/FPS policy. Short clips repeat
frames; the report records that explicitly. Detection failures are rejected rather
than removing frames and changing the apparent speech timing.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover color conversion, normalization, crop bounds, invalid inputs, temporal
sampling through an actual encoded video, and detection failure propagation.
Synthetic-video tests substitute a detector to isolate preprocessing behavior.
Implementation verification: all eight tests passed locally on Python 3.13;
the downloaded MediaPipe model also loaded and correctly rejected a blank frame.
Verify actual landmark quality separately by processing a representative speaking
clip and inspecting the preview. A speaking clip was subsequently supplied and
processed successfully by the user; Part 2's recognition result is documented separately.

MediaPipe implementation reference and model source:
[official Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python),
[official model overview](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
