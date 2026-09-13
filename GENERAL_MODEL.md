# Local English sentence recognition

The optional visual-only Auto-AVSR model is installed and tested on this Windows
CPU machine. No training, GPU, microphone, remote inference or video upload to an
external service is involved. Restart the application and choose **English
sentences (experimental)**. This option is selected by default when installed.

## Observed results — 13 September 2026

The supplied `Sample.mp4` was decoded without using its reference transcript:

- Raw output: **HI I AM JENNIFER**.
- Supplied reference, normalized for evaluation: **Hi I am Jahnawi**.
- The name is incorrect. A successful decode does not imply a correct transcript.
- About 11 seconds of model-worker processing for 5.49 seconds of video on this
  machine; application startup and process-launch overhead are additional.

The public GRID clip `sbia1a.mpg`, whose filename-derived reference is “set blue in
a one again,” also passed through both the actual upload and JPEG camera-frame
routes. Raw results are in `reports/general-integration.json`. The uploaded video
decoded **SET BLUE IN A1 AGAIN**, showing a text-formatting difference from the
reference. These are smoke checks, not a representative or held-out benchmark.
Camera-route testing used public video frames, not a physical webcam.

All 45 software tests passed, including real model inference, both routes,
authentication, private history, cleanup, and original/reference-model regressions.

## Reproduce setup

Run from the project directory after installing the main application requirements:

```powershell
.\.venv\Scripts\python.exe setup_general_runtime.py
.\.venv\Scripts\python.exe setup_general_model.py
.\.venv\Scripts\python.exe recognize.py Sample.mp4 --engine auto-avsr --reference "Hi I am Jahnawi"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe serve.py
```

The first two commands access the official PyTorch package index, GitHub sources
and the author's Google Drive checkpoint. The checkpoint is about 1 GB; allow
additional disk space for the CPU runtime and resumable download parts. Setup
creates `.venv-general`, which shares the main environment's installed OpenCV,
NumPy and MediaPipe packages through a `.pth` file. Keep both environments in the
project directory. CPU PyTorch is pinned to `2.14.0+cpu`; the current runtime is
Python 3.13. Setup fails explicitly if the package is unavailable for a platform.

API and CLI defaults remain unchanged for compatibility. Select `auto-avsr`
explicitly outside the browser. The worker has a 180-second timeout and uses four
CPU threads. Results are always review-required with no confidence percentage.

## Model and adaptation

Source: [official Auto-AVSR repository](https://github.com/mpc001/auto_avsr), pinned
revision `182b62837773ab01052d4ac21ef1d2203ea7d267`. The 28 vendored source/data files
are verified against that Git tree and listed in `vendor/auto_avsr/PROVENANCE.json`.
The visual checkpoint is `vsr_trlrs2lrs3vox2avsp_base.pth` with SHA-256
`fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`.
This full hash was recorded from the downloaded artifact; its MD5 also matches
the author's published prefix. Setup pins the full hash for subsequent downloads.
Inference uses restricted tensor-only loading and strict parameter matching.

The adapter preserves short-video duration by sampling at 25 FPS, aligns faces,
crops the mouth to 96 pixels, then uses an 88-pixel center crop and the upstream
grayscale normalization. Current MediaPipe Face Landmarker anchors replace the
upstream detector. Beam size is 10 for CPU latency, compared with the author's 40.
These changes require local evaluation; upstream published accuracy is not the
accuracy of this application. Reference text is used only after inference.

The Apache-2.0 source license is retained in `vendor/auto_avsr/LICENSE`. The model
and training datasets carry their own usage conditions described upstream;
source licensing alone does not establish unrestricted commercial model rights.

## Remaining limitations

- General-speech accuracy, names, accents, different speakers, lighting and pose
  need representative labelled evaluation with documented training overlap.
- Quality advice is heuristic; confidence is not calibrated.
- Uploads accept a selected interval of up to 16 seconds, with start up to 60
  seconds. Long-video segmentation and reliable variable-frame-rate timing are
  not implemented.
- Webcam capture remains 75 frames per cycle and waits for inference. It is not
  continuous, gapless transcription. Uneven capture timing is reported.
- A physical camera session and hosted HTTPS deployment still need verification.
- The default Docker recipe includes the bridge so the page works without the
  optional model, but does not package this separate CPU runtime or checkpoint.
  Container/hosted deployment of the sentence engine is not yet validated.

See `REQUIREMENTS_STATUS.md` for the complete acceptance record.
