# VisionSpeak — complete application guide

**September 12 model update:** read [MODEL_PROGRESS.md](MODEL_PROGRESS.md) for the
new public GRID model. It is selected by default in the browser; the original
model remains selectable. Use automatic lip detection with the public model.
The historical original-checkpoint findings below still apply to that checkpoint,
but no longer describe the only model available. General speech remains unsupported.
Run `python fetch_reference_model.py` on a fresh installation before using it.
Image-quality and capture-timing advice now appears with recognition results.

## Start the application

From the project folder in Command Prompt:

```cmd
.venv\Scripts\activate.bat
python -m pip install -r requirements-server.txt
python serve.py
```

Open **http://127.0.0.1:8080/**. If an older server is still running there, press
Ctrl+C in its terminal first, then start the updated server. The finished website
is at `/`, not `/health/ready` or `/api/v1/recognize`.

Choose **Create an account**, enter a username and a password of at least ten
characters, then sign in. Local accounts and history are stored in
`instance/visionspeak.db`. The local signing key is in `instance/session.key`.
Both are excluded from Git and the Docker image. Keep these files between local
runs if you want your accounts and history to persist.

## Use the studio

### Recorded video

1. Choose **Upload video**, select a file or drag it onto the upload area.
2. Use a short, well-lit clip with one visible speaker. Maximum upload is 30 MB
   in the interface (32 MiB total HTTP request on the server).
3. Open **Recognition options** only if needed. Automatic lip detection is the
   default. For the downloaded reference MPG, select **Original GRID sample**.
4. Press **Recognize video**. Review the result, then inspect raw output if useful.
5. Results are saved to the signed-in account's history. Uploaded video is removed
   after processing. Some browsers cannot play MPG/AVI previews, but server-side
   processing is still available.

### Webcam and live lip preview

1. Choose **Use webcam**, then **Enable camera**. Allow camera access in the browser.
2. The preview detects a single face and draws a green box around the lip region.
3. Press **Recognize capture**. It captures 75 JPEG frames over approximately
   three seconds, then submits them for visual-only recognition.
4. Enable **Keep recognizing in short cycles** for repeated capture/process cycles.
   Each cycle waits for inference and then pauses four seconds. This is windowed
   recognition, not frame-by-frame streaming transcription.
5. Use **Stop live recognition** to stop future cycles or **Turn off** to release
   the camera. Leaving the studio, hiding the tab, signing out, or closing the page
   also stops the camera. An in-flight server request may still finish and be saved.

No microphone permission is requested. Frames avoid differences in browser video
recording codecs. Camera access requires HTTPS when hosted; localhost is supported
for development. Keep the tab visible during capture. Physical camera permissions,
hardware and capture speed need checking on each target device/browser.

### History

The latest 200 predictions per account are stored, including unsuccessful attempts.
Search by candidate or filename, view a result, download/export JSON, or delete
individual/all results after confirmation. Other users cannot read or delete your
history. Deletion is permanent. The application does not store the original video
or webcam images in the history database.

## What is complete and what remains limited

| Part | Implementation |
| --- | --- |
| 1 | Video validation, lip detection, grayscale crops, normalization and sequence export |
| 2 | Existing-checkpoint inference, greedy/beam decoding, evaluation and clear failure reporting |
| 3 | Responsive website, account sessions, video upload, webcam frame capture, repeating recognition, live lip boxes, private persistent history, export and deletion |
| Deployment | Production WSGI entry point, authenticated API, SQLAlchemy database layer, PostgreSQL configuration, Docker/Compose recipes, health endpoints and bounded inference |

**Recognition accuracy is still not solved.** The unchanged checkpoint fails the
verified reference phrase `set white with p two soon` and the supplied personal
video. The application correctly displays this failure rather than guessing words.
Confidence percentages remain unavailable because they have not been calibrated.
The software workflows are implemented; this is not a claim that the abstract's
high-accuracy recognition objective has been achieved.

No training or fine-tuning was performed for this completion. Original weights
remain unchanged. A compatibility audit checked vocabulary, tensor dimensions,
the notebook's third-convolution activation and its unsigned grayscale-normalization
behavior. Neither possible notebook variation recovered the reference sentence.
These alternatives were evaluated in memory only and were not adopted. See
`outputs/compatibility-audit.json` and [MODEL_AUDIT.md](MODEL_AUDIT.md).

## Verification

```cmd
python -m unittest discover -s tests -v
node --check static/app.js
```

The suite includes model-weight immutability, multipart uploads, API authentication,
CSRF protection, login/logout, history isolation, deletion, database reopening,
request limits, frame validation, and webcam-result persistence. No test trains
or saves model weights.

Browser checks covered sign-in, file selection, recognition, saved history,
reload persistence, camera controls, and desktop/mobile layouts. Real GRID JPEG
frames also passed through the lip-preview and 75-frame webcam API endpoints;
`outputs/webcam-api-check.json` records that test. It used recorded frames, not
a physical webcam. The target device's live camera test remains necessary.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md). Local development uses SQLite. For multiple
instances, all replicas must use the same PostgreSQL database and signing key.
The core abstract stack is unchanged; SQLAlchemy/PostgreSQL are supporting
infrastructure for persistent multiuser data. No API token is exposed in browser
JavaScript; the website uses signed HttpOnly sessions and CSRF tokens.

Docker and PostgreSQL runtime testing could not be performed because Docker is
not installed here. No cloud endpoint has been published. An HTTPS host/domain,
production secrets, persistent database, backups and target-load testing are needed
before hosted release. The unresolved model accuracy is also a release limitation.
