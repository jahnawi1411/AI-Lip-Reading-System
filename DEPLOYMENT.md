# VisionSpeak deployment and scaling

Model update: run `python fetch_reference_model.py` before building the image.
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
