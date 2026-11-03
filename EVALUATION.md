# Evaluate the actual English lip-reading model

Run from the project root with the main Python environment:

```powershell
.venv/Scripts/python.exe benchmark_public.py --engine auto-avsr --output reports/english-evaluation.json
```

This evaluates all eleven existing labelled public GRID clips. It is a regression
check, not evidence of general conversational accuracy. Speaker identities and
training overlap are unknown. References are used only after prediction, never
sent to the model. Rejected clips count as empty predictions rather than being
removed. WER and CER can exceed 100% when extra words or characters are inserted.
The report includes checkpoint and manifest hashes, raw predictions, reference
labels, exact sentence rate, failures, and first/subsequent request timings.
Timing measures recognition, not browser upload or expression analysis.

## General English evaluation still required

Collect at least 30 consented, short recordings from at least five distinct
speakers as an initial pilot, including greetings, questions, names and numbers.
This is a practical starting sample, not a statistical accuracy guarantee.
Include ordinary variations in lighting and camera position. Keep a separate
development set with different speakers; do not tune settings on the test set.
Have a person verify each transcript before running inference. Record failures,
not just successful examples. Do not publish personal videos without consent.

Use the same manifest format as `data/public_regression/manifest.json`, with:

- `purpose`: describe the actual collection and evaluation scope.
- `training_overlap`: use `unknown` unless independently established.
- `speaker_identity_verified`: true only for verified distinct speakers.
- `samples`: each needs `video` (relative path), `text`, `speaker`,
  `label_source`, `mode` (`mediapipe`), and the video's `sha256`.

Compute each hash using `Get-FileHash -Algorithm SHA256 path/to/video.mp4`.
Use lowercase hash strings. Run with `--manifest path/to/manifest.json` and a
new output path. Preserve the report, including unsuccessful examples.
The English evaluator always selects automatic lip detection and ignores audio.

Report visual-only results separately from Whisper audio transcription. A correct
audio transcript is not evidence of lip-reading accuracy. Avoid turning WER into
a confidence percentage for individual predictions.

## Recorded English-model regression — 19 October 2026

Report: `reports/english-regression-2026-10-19.json`.
All 11 public GRID clips completed without input failures. WER was 66.7%, CER
43.7%, with zero exact sentence matches under the existing case/punctuation
normalization. Number formatting (for example `A1` versus `a one`) contributes
errors, but there are substantial word substitutions too. These results must not
be described as validated conversational accuracy or a 9/10 recognition system.
The first request took 28.72 seconds; subsequent requests averaged 4.60 seconds.
The test uses a restricted corpus with unknown training overlap and speakers.
