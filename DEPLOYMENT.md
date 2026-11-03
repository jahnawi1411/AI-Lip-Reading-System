# Deployment release status — 19 October 2026

The container recipe now packages the English sentence worker, its isolated CPU
PyTorch environment, vendored inference code and local checkpoint. It also
includes the audio/expression module and dependencies. Whisper weights download
during image build, so a normal audio request does not have to install them.
Only explicitly allowed source/model assets enter the Docker build context;
accounts, personal videos, local databases and secrets remain excluded.

**Not yet release-verified:** Docker is unavailable on the development machine.
The Linux dependency installation, image startup, PostgreSQL integration and
public HTTPS camera flow must pass before calling this production-ready.
Local Windows tests do not establish Linux compatibility. All 70 test cases
report OK, but the full Windows runner exits nonzero during final teardown;
resolve this before treating the suite as a release gate.

## Build and release procedure

1. Run `python deployment_preflight.py` to verify required files, the pinned
   English model hash and all vendored source hashes. Ensure the local models exist: `models/auto-avsr/` (checkpoint and
   provenance), `models/face_landmarker.task`, original weights and LipNet weights.
   Run the documented setup scripts if missing. The image build needs network
   access to package indexes and the audio model download service.
2. On a Docker host, copy `.env.example` to a private `.env` and supply distinct,
   randomly generated API/session secrets and a database password. Never commit it.
3. Run `docker compose config --quiet`, then `docker compose build --pull`.
   Resolve any Linux dependency errors before continuing; do not silently loosen pins.
4. Run `docker compose up -d`; inspect `docker compose logs web`. Check
   `http://127.0.0.1:8080/health/ready` on the host.
5. Put an HTTPS reverse proxy in front of localhost port 8080. Keep the database
   private. Allow at least 240 seconds for inference responses at the proxy.
6. Test registration, login, an English upload twice, audio transcription,
   expression-only mode and webcam permission using the public HTTPS URL.
   Check that results state whether words came from video or sound.
7. Restart the containers and verify account/history persistence. Test a database
   backup and restore. Measure peak RAM and cold/warm latency on the actual host.
8. Send overlapping requests: one should run while the other gets HTTP 429 with
   Retry-After. The current app has bounded concurrency, not a background queue.

Compose defaults to a configurable 8 GiB memory cap and four CPUs for the web
service. These are provisional limits, not measured minimum hardware requirements.
Set `WEB_MEMORY_LIMIT` and `WEB_CPU_LIMIT` after measurement. Persistent model
reuse trades additional resident RAM for lower repeat-request latency.
An online endpoint still needs a hosting account; none has been created.

---

# VisionSpeak deployment and scaling

Historical reference-model deployment notes (superseded by the release procedure above):
run `python fetch_reference_model.py` before building the image.
The image now includes the public reference weights and their MIT license.
`VISIONSPEAK_REFERENCE_WEIGHTS` can override their runtime path. Original API
requests retain the original engine unless `engine=lipnet-reference` is supplied.
The public engine uses `mode=mediapipe` and is limited to GRID commands. See
[MODEL_PROGRESS.md](MODEL_PROGRESS.md). Docker/PostgreSQL remain untested here.

The website and API run together using `python serve.py` (or `python app.py`).
The entry point uses Waitress, not Flask's development server. No training occurs.

## Local development

SQLite under `instance/visionspeak.db` stores accounts, results and rate-limit
counters. A generated `instance/session.key` preserves sessions across local
restarts. These files must remain private and are ignored by Git/Docker.
Local default listens on 127.0.0.1:8080. Register in the browser to use private
history and camera features. Existing localhost curl uploads still work without
logging in, but do not create a shared history entry.

## Production environment

| Variable | Purpose |
| --- | --- |
| VISIONSPEAK_ENV=production | Binds all interfaces and enables Secure session cookies |
| VISIONSPEAK_API_TOKEN | Random secret of at least 32 characters for backend API clients |
| VISIONSPEAK_SECRET_KEY | Shared random signing key of at least 32 characters for all replicas |
| DATABASE_URL | PostgreSQL SQLAlchemy URL, using postgresql+psycopg:// |
| PORT | Server port, default 8080 |
| VISIONSPEAK_WEIGHTS | Optional mounted existing checkpoint path |
| VISIONSPEAK_UPLOAD_ROOT | Writable temporary upload directory |

Provide secrets through a secret manager. Do not embed the API token in browser
JavaScript. Browser users use password accounts and HttpOnly, SameSite=Lax,
signed one-day sessions with CSRF protection. Passwords are stored as Werkzeug
scrypt hashes. Production requires HTTPS for secure cookies and webcam access.
The app does not trust forwarded client-IP headers by default.

If no separate signing key is supplied, a purpose-specific key is derived from
the API token. Prefer an explicit stable key so API-token rotation does not also
log out all users. All replicas must use identical signing keys and database URLs.

## Docker and PostgreSQL recipe

Both existing model files must be present under `models` before building. The
Docker image excludes personal clips, notebooks, outputs, test accounts, secrets
and local databases. It runs as a non-root user.

Fill the three secrets in `.env.example` and save a private `.env` file. Use a
random alphanumeric PostgreSQL password in Compose, or URL-encode special
characters in a manually configured database URL. Then:

```cmd
docker compose up --build
```

The supplied Compose recipe starts one web instance and PostgreSQL with a durable
volume. Its web port is bound only to localhost. Put an HTTPS reverse proxy in
front of it for browser access; production Secure cookies intentionally do not
work over plain HTTP. The readiness endpoint can still be checked locally over
HTTP. Do not disable cookie protection to make a public HTTP deployment work.

Database tables are initialized automatically on first startup. Start one instance
to initialize the schema before scaling replicas. Future schema changes require
explicit migrations; this initial schema is not a migration framework. Back up
the PostgreSQL volume and test restores. Never run `docker compose down -v` unless
you intend to delete the database.

Docker is unavailable on this PC. Linux image builds, Compose and real PostgreSQL
runtime remain unverified. Direct dependencies are pinned; platform-specific and
transitive dependency validation is still required before publishing an image.
No registry image or hosted endpoint has been published.

## API

- GET `/`: complete browser application.
- GET `/health/live`: process health.
- GET `/health/ready`: startup model availability, not recognition accuracy.
- GET `/api/auth/session`: current browser user and CSRF token.
- POST `/api/auth/register`, `/login`, `/logout`: browser authentication.
- POST `/api/v1/recognize`: multipart uploaded video.
- POST `/api/v1/lips`: JPEG lip-preview detection for signed-in users.
- POST `/api/v1/recognize-frames`: 75 JPEG frames plus timing, for signed-in users.
- GET/DELETE `/api/v1/history`: list or clear this user's records.
- DELETE `/api/v1/history/<id>`: delete one owned record.

Machine clients can use `Authorization: Bearer <token>` for video recognition;
they do not inherit any browser user's history. Browser writes use X-CSRF-Token.
Each recognition result carries a request ID and the checkpoint fingerprint.
Sensitive filesystem paths and captured images are not returned in history.

## Limits and scaling

The total upload cap is 32 MiB. Form parsing allows 128 KiB of memory, accounting
for Werkzeug multipart read boundaries. One inference/detection operation is
active per process. Additional work returns 429 rather than allocating another
model batch. Waitress uses four threads and bounded connections. A user may make
10 recognition requests per minute and 120 preview requests per minute; login
and registration are limited to 12 attempts per 15-minute window per direct IP.
These counters are stored in the database and shared by replicas. A reverse proxy
must apply complementary per-client limits because direct IP may be the proxy.

For multiple replicas, use PostgreSQL and an HTTPS load balancer; no sticky
sessions are necessary. The supplied Compose file has one fixed host port and is
not itself a multi-replica load balancer. Each replica loads its own model, so
measure memory before selecting replica counts. The 4 GiB / 2 CPU Compose limits
are starting settings, not established capacity targets. Measure p95 latency,
429/5xx rates, CPU, peak memory and database connections under realistic traffic.

Recognition is synchronous and bounded. Webcam mode repeats short capture-and-
recognize requests, with a pause after each result. It is not continuous video
streaming. At higher volumes, a durable queue and separate inference workers are
a future extension; no queue, autoscaler or GPU deployment is claimed here.
The database stores the latest 200 result records per user. Uploaded video files
are temporary; crashes may require host-level orphan-temp cleanup. Requests
already running may finish after the client disconnects or stops live mode.

Open registration is implemented. Email verification, self-service password
recovery, administrative account management and independent security/load audits
are future production operations, not features claimed in this mini-project.

## Release status

Windows verification covers inference-only tests, account/CSRF behavior,
per-user history isolation/deletion/persistence, upload cleanup, real HTTP
inference, and recorded frames through the webcam API. Browser checks cover
sign-in, upload/result/history, camera controls and mobile/desktop layouts.
Physical webcam hardware and deployed HTTPS behavior require target-device tests.

The original checkpoint still fails the verified GRID reference clip, including
notebook-compatibility variants. The UI flags results and leaves confidence
uncalibrated. Completing the application is not evidence that speech recognition
is accurate. Resolve model quality and agree an accuracy benchmark before a
public recognition release.
