# VisionSpeak requirements and acceptance record

The user asked to complete the project requirements using public models and data.
No separate approved abstract or requirements document was found in the workspace.
The working scope therefore includes visual-only recognition of English sentences,
not only GRID commands. This is an implementation assumption, not a claim that the
user supplied a new written specification.

| Requirement | Acceptance evidence | Current state |
| --- | --- | --- |
| Visual-only input | No microphone request or audio path used during recognition | Implemented |
| Uploaded short videos | Valid input runs through model and result display; temporary input removed | Implemented and tested for GRID and Auto-AVSR |
| Ordinary English sentences | Public sentence model runs locally and is evaluated against supplied references | Implemented experimentally; greeting recognized with incorrect name; representative evaluation remains |
| Camera capture | Real browser permission, physical device capture, and prediction tested together | API tested; physical user/device test remains |
| Input feedback | Detect missing/multiple faces; display quality and timing advice | Implemented; heuristic thresholds uncalibrated |
| Private accounts/history | Authentication, CSRF, account isolation, export and deletion checks | Implemented and tested |
| Accuracy claims | Labelled dataset, verified speaker identities, known training overlap, reproducible WER/CER | Small public GRID regression only; broader validation incomplete |
| Confidence | Calibrated scores on separate validation data | Unavailable; UI must not fabricate percentages |
| Continuous recognition | Measure capture gaps and avoid losing speech during inference | Short cycles implemented; gapless transcription incomplete |
| Long videos | Time-preserving segmentation, boundary handling, timestamps | Not complete |
| Online deployment | HTTPS, persistent database, backups, image build and load test | Recipes exist; external infrastructure not supplied or validated |

Acceptance is based on observed results. A functioning page, successful request,
or high model score is not evidence that the transcript is correct. No transcript
reference may be fed into inference or used to replace the decoded output.

## Demonstration procedure

1. Start the server and sign in.
2. Select the intended recognition model and read its supported scope.
3. Upload a public labelled video and compare the result with its reference.
4. Show the raw output, review requirement and any input-quality advice.
5. Open history, reload the page and verify the result persists.
6. For physical camera verification, the presenter must allow camera access and
   speak a phrase; inspect the actual output rather than substituting a sample.

## What needs human or external input

Physical camera speech cannot be supplied by the coding agent. Hosted deployment
requires an actual hosting destination, domain/HTTPS setup and production secrets.
No cloud account, public endpoint, subscription or paid compute has been created.
An accuracy target should be agreed before a release decision; none is invented here.
