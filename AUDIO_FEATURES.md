# Video, audio and lip expressions

Select **Lip reading — with or without sound**, then **With sound — audio transcription + lip shapes** in the studio. Upload a
video with sound, or enable the camera and allow both camera and microphone
permissions. Camera recordings last three seconds. Uploaded selections process
up to 16 seconds, starting at the selected start time (0–60 seconds); with an
empty duration the first 16 seconds of that selection are used.

The result shows the audio transcript, word timestamps relative to the original
video, and a lip-shape timeline sampled about five times per second. Results are
included in private history and JSON downloads. Live camera preview also displays
the current lip shape. Changing the model turns off the camera; enable it again
to request the permissions needed for that model.

## Setup

Run in the same environment used for the application:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-audio.txt
.venv\Scripts\python.exe setup_audio.py
.venv\Scripts\python.exe app.py
```

The setup downloads the multilingual Whisper base model to `models/whisper`.
An internet connection is required for installation and the initial model
download. Subsequent transcription runs locally on CPU. No speech API key or
separate FFmpeg installation is required. To use another Whisper model, set
`VISIONSPEAK_AUDIO_MODEL` to its model name or a local CTranslate2 model directory
before setup and startup. The face model from the existing project is needed for
lip analysis (`python prepare_video.py --download-model`).

## Interpretation

Spoken words come from audio transcription. This mode does not combine the visual
speech models with Whisper or confirm that a visible person produced the sound.
Silent clips return a no-audio/no-speech result; choose an existing visual model
to attempt lip reading without sound. Speech transcription can make mistakes.

Lip labels describe smiling, pouting (rounded/projected), pursed, closed, slightly
open or open lips. MediaPipe mouth blendshapes and geometric opening measurements
are mapped to these labels using conservative rules. They are not emotion
predictions, and coefficient values are not calibrated confidence probabilities.
Multiple faces or missing faces produce unavailable samples. If face analysis
cannot run, the audio transcript remains available with an explanatory message.

Videos and microphone recordings are temporary and removed after processing;
only the result is stored in account history. Browser camera recording requires
HTTPS or localhost and MediaRecorder support. WebM uploads are supported.

Audio implementation reference: [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Verification

The local base model was installed and exercised on the existing
`data/grid_check/id2_vcd_swwp2s.mpg` sample. It returned "Set white with P2 soon."
with word timestamps and 13 lip-shape samples. This is a functional smoke test,
not an accuracy benchmark. Automated tests cover silence, missing audio, interval
validation, audio decoding, timestamp offsets, missing faces, temporary-file
cleanup, and the microphone permissions policy. Browser microphone capture still
requires a manual device/permission check on the target browser.

## Standalone lip expression recognition

Select **Lip expression recognition** in the Feature selector. Upload a video
(with or without sound), or enable the camera and record a silent three-second
clip. The result shows the most observed lip shape, timestamped samples and
sample counts in the downloaded JSON. Missing-face samples are excluded from
the summary. No words are inferred in this mode. Saved history and live cycles
are supported. It needs the face model but does not load Whisper or lip-reading
weights. The supported labels are smiling lips, pouting lips, pursed lips, closed lips,
slightly open lips and open mouth. Pout means a rounded/projected lip gesture,
not sadness. Speech movements may resemble these gestures. If mouth blendshape
outputs are unavailable, only the basic opening labels are returned. Try holding
each gesture briefly with one well-lit, front-facing face. No new model download
is needed when the existing face_landmarker.task bundle is installed.

## Gesture implementation and limits

The face model's mouthSmileLeft, mouthSmileRight, mouthPucker and mouthFunnel
outputs are used; no eye/eyebrow signals or emotion labels are classified.
Both smile sides must reach 0.5. Pucker/funnel must reach 0.55, with a 0.15
margin over a conflicting gesture. Strong pucker with an opening ratio below
0.08 maps to pursed lips; other strong rounded gestures map to pouting lips.
Ambiguous signals retain the geometric opening label. These application rules
are experimental and have not been validated on a labelled gesture dataset.
The opening label is also kept separately so an open smile is not lost.

Reference: [MediaPipe FaceLandmarker options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions).
