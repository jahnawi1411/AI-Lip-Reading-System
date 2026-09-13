# VisionSpeak Part 2 — existing-model recognition

Part 2 uses the original TensorFlow checkpoint with no training, fine-tuning or
architecture changes. Python, TensorFlow, OpenCV and MediaPipe remain the stack.

## Run it

In your activated environment, from the project folder:

```cmd
python recognize.py Sample.mp4
```

Video input automatically uses MediaPipe lip detection. A saved Part 1 sequence
is detected automatically too:

```cmd
python recognize.py outputs\first-clip\sequence.npz
```

Each run creates a fresh timestamped JSON file under `outputs`. An explicit
`--output` filename is supported and existing output files are never overwritten.
To reproduce the environment: `python -m pip install -r requirements-part2.txt`.

## Select the speaking interval

Use `--start` and `--duration` in seconds to exclude unrelated content or silence.
For example, to select seconds 1 through 4:

```cmd
python recognize.py Sample.mp4 --start 1 --duration 3
```

This is a selection example, not an asserted location of speech in your sample.
The interval must exist, contain at least two frames, start within 60 seconds,
and last at most 10 seconds. Source frame indices and actual selected boundaries
are recorded. Uniform sampling produces 75 frames, with repeated frames for short
intervals. A duration warning explains differences from approximately 3-second
GRID input. Constant-frame-rate source video is expected; timestamps are derived
from the reported FPS. No automatic silence detection or transcript-guided trimming
is performed. Saved NPZ inputs cannot be trimmed; use the original video instead.

## Decoding and results

Greedy CTC remains the default. Optional beam search explores alternative sequences
using only the original network probabilities; it does not use a new model, a
language model, a dictionary, or your reference transcript:

```cmd
python recognize.py Sample.mp4 --decoder beam --beam-width 50
```

Three ranked candidates and their CTC sequence log probabilities are saved.
Beam width accepts 3 through 100. Higher sequence log probability ranks first,
but it is not a calibrated correctness probability. Greedy and beam disagreement
is flagged. Beam search was worse on the supplied sample and remains optional.

The result contract for future Flask integration is:

- `text`: raw decoded candidate, including unknown replacement characters.
- `usable_text`: null for unknown-token or empty output; otherwise a candidate
  that still requires review, not a certified transcript.
- `status`, `message`, `review_required`, `review_reasons`: explain the result.
- `confidence`: null; `confidence_status`: uncalibrated.
- `model_score`: the earlier emitted-run heuristic for greedy decoding only;
  not a correctness probability. Beam results use `ctc_log_probability` instead.
- `decoder`, `weights_sha256`, `preprocessing`, `source`, `warnings`: provenance.
- `input_report`: selected interval, sample indices and lip boxes for video input.

Unknown-token output is displayed as 'Could not reliably recognize speech from
this clip.' Raw output remains in JSON for inspection; alternatives never replace
it with a guessed familiar phrase. All outputs require review because this model's
accuracy has not been validated for these inputs.

## Compatibility and evaluation

`--mode grid` explicitly requests the original 360x288, 75-frame fixed GRID crop
and historical grayscale convention. The application does not assume that an
arbitrary video of those dimensions has correct GRID framing. The default video
mode locates lips instead. Shape compatibility is not evidence of accuracy.

A known transcript may be supplied only for scoring after inference:

```cmd
python recognize.py Sample.mp4 --reference "Hii, I am Jahnawi"
```

Evaluation lowercases text, removes commas and periods, and collapses whitespace.
WER counts word edits; CER counts character edits including spaces. Both rates
can exceed 100% when the candidate contains insertions. `evaluate_model.py` and
its JSON manifest format remain available for multi-clip evaluation with existing
weights. Labelled clips can measure the current model without training it.

On the supplied sample, greedy decoding produced `les` plus one unknown token:
WER 100%, CER 100%. Optional beam decoding produced more unknown tokens:
WER 100%, CER 125%. Neither recognizes 'Hii, I am Jahnawi' correctly.
These are one-clip measurements, not overall model accuracy. The sample transcript
was used for scoring only. The interface improvements do not remove the existing
checkpoint's recognition limitations.

A subsequent [matched GRID check](GRID_CHECK.md) also failed: the reference was
`set white with p two soon`, but the output was `l` plus an unknown token (100% WER).
The input was verified as 360x288, 75 frames at 25 FPS with a visible mouth crop.
TensorFlow's own greedy decoder agreed with the application decoder. This rules
out attributing the failure solely to the user's non-GRID video, but does not by
itself establish whether the cause is checkpoint quality or training-configuration
compatibility.

The original checkpoint SHA-256 is:
`24f5b985a11b0d0b2fb102950a961ae805f3c547dd49b675239208bad0874f46`.

## Verification and scope

```cmd
python -m unittest discover -s tests -v
```

Tests are inference-only: preprocessing, interval boundaries, normalization, input
routing, CTC repeats, beam path aggregation, failure handling and unchanged model
parameters/file hash. No tests train or save new weights. The earlier training
script remains an unused development file and is outside the current workflow.
The Flask/webcam/history work remains Part 3.

Decoder reference: [TensorFlow beam-search API](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder).
