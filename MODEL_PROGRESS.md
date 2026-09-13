# Public model integration — 12–13 September 2026

Update: the optional Auto-AVSR sentence engine is now installed locally and is the
browser default when available. See [GENERAL_MODEL.md](GENERAL_MODEL.md). The
following results describe the separate public GRID integration.

Without the sentence engine, the browser defaults to **Public GRID model**.
The original H5 model is preserved and selectable for comparison. API requests
and the CLI retain their original engine default for backwards compatibility;
pass `engine=lipnet-reference` or `--engine lipnet-reference` to select the new model.

## Measured regression results

Eleven public GRID video clips, standard greedy decoding:

| Engine / preparation | Word error rate | Character error rate | Exact sentences |
| --- | --- | --- | --- |
| Original H5 / original GRID fixed crop | 100.0% | 96.2% | 0 / 11 |
| Public LipNet / automatic RGB mouth detection | 10.6% | 4.9% | 4 / 11 |

Reference `set white with p two soon` now produces `set white with b two soon`.
The personal greeting video still fails. Neither model is a general conversation
transcriber. Readable words and high model scores can be wrong.

Reports: `outputs/public-original-baseline.json`,
`outputs/public-lipnet-reference.json`. This compares complete pipelines; it does
not isolate the effect of weights from architecture or preprocessing.

Ten labels are derived from GRID sentence codes in the published filenames; the
eleventh is supported by the provided alignment. Speaker identity and overlap
with upstream model training data are unverified. These are regression samples,
not a speaker-disjoint held-out test set. No confidence calibration is claimed.
The reference model averaged about 7.3 seconds per clip in this CPU run, including
the first load. This is not a concurrent-user load test.

Beam search was also evaluated and had the same aggregate WER, CER and exact-match
count; see `outputs/public-lipnet-reference-beam.json`. Greedy remains the default.

## Model provenance and compatibility

- Source: https://github.com/rizkiarm/LipNet
- Pinned revision: `bec592aa1e66378bc7ab33f6110f1c8a431b3b9a`
- Weights: `evaluation/models/unseen-weights178.h5`
- SHA-256: `b4ea5202691223ff8a91141a014f7477df53e945bd5c4fcae8413691db13d181`
- License: MIT; included in `THIRD_PARTY_LICENSE_LIPNET.txt`.
- Adapter: `reference_lipnet.py`, based on upstream `model2.py`.
- Input: RGB, 75 × 100 × 50 × 3, width before height, values divided by 255.
- Network: three 3D convolutions with batch normalization, two bidirectional GRUs,
  28 original output classes (letters, space, CTC blank).
- Compatibility: reset-before GRUs, legacy `clip(0.2*x+0.5,0,1)` recurrent activation,
  explicit H5 array loading. No external model code is executed or deserialized.
- Preprocessing: MediaPipe landmarks replace upstream dlib; first-frame crop scale
  stays fixed through a clip. This adaptation is measured separately from supplied crops.
- No spelling dictionary, reference transcript, or language model influences decoding.

The separate nicknochnack checkpoint96 was also investigated and did not recover
the checked sentences. It is retained under `models/reference-checkpoint-96/`
for audit only and is not exposed as the new browser model. Neither downloaded
checkpoint nor the original checkpoint was trained or overwritten.

## Reproduce

Use the project Python environment:

```powershell
.\.venv\Scripts\python.exe fetch_reference_model.py
.\.venv\Scripts\python.exe fetch_public_samples.py
.\.venv\Scripts\python.exe benchmark_public.py --weights models/checkpoint.weights.h5 --output outputs/baseline-new.json
.\.venv\Scripts\python.exe benchmark_public.py --weights models/lipnet-reference/unseen-weights178.h5 --engine lipnet-reference --mode mediapipe --output outputs/reference-new.json
.\.venv\Scripts\python.exe recognize.py data/public_regression/sbia1a.mpg --engine lipnet-reference
.\.venv\Scripts\python.exe serve.py
```

The download scripts verify pinned hashes and refuse to overwrite differing files.
Choose fresh report names. Public clips remain in ignored `data/`; hashes and
source URLs are in `examples/public-regression-sources.json`.

## Other improvements

- Advisory mouth-image checks for dark/bright, low-detail, blurred and nearly static
  inputs. Thresholds are explicitly uncalibrated and do not gate recognition.
- Webcam timing reports expose effective FPS and long inter-frame gaps.
- Advice appears in the result panel and is saved with history/exported JSON.
- Public RGB recognition is available for uploads and 75-frame camera captures.
- The original inference-only routes and model remain available.

Verification: 40 tests passed, including actual public-model inference through
authenticated upload and JPEG camera-frame endpoints with private history and
unchanged weight hashes. JavaScript syntax validation passed. Reference-asset
integration checks skip explicitly on installations without the downloaded assets.
An actual browser upload of `sbia1a.mpg` displayed `set blue in a one again` and
saved it to the local verification account's history. A physical webcam was not used.

## Remaining work

General speech requires a suitable model and a labelled, speaker-disjoint dataset;
this GRID model cannot fulfill that requirement. Continuous gapless capture,
long-video segmentation, confidence calibration, physical-camera testing, Docker/
PostgreSQL runtime verification, cloud HTTPS deployment and load testing remain
incomplete. No GPU is visible to this local TensorFlow installation. Training on
these public evaluation examples would invalidate their use as regression evidence.
