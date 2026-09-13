# What to test now

Use Command Prompt in `C:\MiniProjectS7\AI Lip Reading System`. Commands below
assume the existing environment is activated; do not paste the command prompt
prefix `(.venv) C:\...>` itself.

## 1. Activate the environment

```cmd
.venv\Scripts\activate.bat
```

## 2. Test recognition separately from the server

```cmd
python recognize.py data\grid_check\id2_vcd_swwp2s.mpg --mode grid --reference "set white with p two soon"
```

The correct transcript is **set white with p two soon**. The current checkpoint
returns `l` plus an unknown token, with 100% WER on this clip. This is a failed
recognition test even though the program exits normally and saves a JSON report.
Keep the generated report; do not treat a raw model score as recognition accuracy.
You do not need to record more personal videos to reproduce this problem.

For your own video, independently:

```cmd
python recognize.py Sample.mp4 --reference "Hii, I am Jahnawi"
```

If you see startup warnings followed by a saved result, processing completed.
If there is a traceback or no saved result, preserve the full output for diagnosis.

## 3. Test the software

```cmd
python -m unittest discover -s tests -v
```

Expected with downloaded public assets: **40 tests, OK**. These are software checks, not evidence of model accuracy.
They do not train or modify weights.

## 4. Start the deployment-style API locally

Installed in this workspace already. On a fresh setup, first run:

```cmd
python -m pip install -r requirements-server.txt
python prepare_video.py --download-model
```

Start the server and leave this terminal open:

```cmd
python serve.py
```

Open a second Command Prompt in the project folder and run:

```cmd
curl.exe http://127.0.0.1:8080/health/ready
curl.exe -F "video=@data/grid_check/id2_vcd_swwp2s.mpg" -F "mode=grid" http://127.0.0.1:8080/api/v1/recognize
```

The first response should contain `"status":"ready"`. The second should return
JSON with `"usable_text":null`, `"review_required":true` and the recognition
failure message for this checkpoint. An HTTP 200 means the request was processed,
not that the recognized words were correct. Press Ctrl+C in the server terminal
to stop it.

The local default listens only on your PC's loopback address and needs no token.
Production mode requires authentication; see [DEPLOYMENT.md](DEPLOYMENT.md).
The root URL now displays the complete website. Create an account to use uploads,
webcam capture and private history. `app.py` and `serve.py` launch the same app.
See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for browser testing and account features.

## What has already been tested here

- All 32 software tests passed, including multipart upload boundary regression checks.
- A real Waitress process accepted an authenticated GRID upload over HTTP;
  readiness and inference both returned 200, with the failed prediction correctly
  flagged. Result: `outputs/server-smoke-result.json`.
- The model's parameters and weight file remained unchanged.
- Docker is not available on this machine; the Linux image has not been built
  or tested. No cloud deployment or load test has been performed.
