"""Persistent, bounded worker isolated from the TensorFlow environment."""
import atexit
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv-general' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
WEIGHTS = ROOT / 'models/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth'
TIMEOUT = 180
_lock = threading.Lock()
_worker = None
_responses = None


def available():
    return PYTHON.is_file() and WEIGHTS.is_file() and (WEIGHTS.parent/'PROVENANCE.json').is_file()


def _stop_worker(graceful=False):
    global _worker, _responses
    process, _worker = _worker, None
    _responses = None
    if process is not None:
        if graceful and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()


atexit.register(_stop_worker, True)


def _read_responses(process, responses):
    try:
        for line in process.stdout:
            responses.put(line)
    finally:
        responses.put(None)


def _start_worker():
    global _worker, _responses
    _worker = subprocess.Popen(
        [str(PYTHON), str(ROOT/'general_lipreading.py'), '--worker'],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    _responses = queue.Queue()
    threading.Thread(target=_read_responses, args=(_worker, _responses), daemon=True).start()


def recognize_general(path, start=0., duration=None):
    if not available():
        raise FileNotFoundError('The sentence model and its separate runtime are not installed.')
    # One request at a time keeps responses paired and bounds memory use.
    with _lock:
        try:
            if _worker is None or _worker.poll() is not None:
                _stop_worker()
                _start_worker()
            _worker.stdin.write(json.dumps({'path': str(Path(path).resolve()),
                                           'start': start, 'duration': duration}) + '\n')
            _worker.stdin.flush()
            line = _responses.get(timeout=TIMEOUT)
            if line is None:
                raise RuntimeError('Worker exited')
            response = json.loads(line)
            if 'error' in response:
                raise ValueError('Sentence recognition failed. Use a short clip with one clear, centered face.')
            result = response['result']
            if not isinstance(result, dict) or result.get('engine') != 'auto-avsr' or result.get('audio_used') is not False:
                raise RuntimeError('Unexpected sentence-worker response')
            return result
        except queue.Empty as exc:
            _stop_worker()
            raise ValueError('Sentence recognition timed out. Try a shorter clip.') from exc
        except ValueError:
            _stop_worker()
            raise ValueError('Sentence recognition failed. Use a short clip with one clear, centered face.') from None
        except Exception as exc:
            _stop_worker()
            raise ValueError('Sentence recognition failed. Use a short clip with one clear, centered face.') from exc
