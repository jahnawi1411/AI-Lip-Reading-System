"""Bounded subprocess bridge so the existing TensorFlow environment stays intact."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv-general' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
WEIGHTS = ROOT / 'models/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth'


def available():
    return PYTHON.is_file() and WEIGHTS.is_file() and (WEIGHTS.parent/'PROVENANCE.json').is_file()


def recognize_general(path, start=0., duration=None):
    if not available():
        raise FileNotFoundError('The sentence model and its separate runtime are not installed.')
    with tempfile.TemporaryDirectory(prefix='visionspeak-sentence-') as directory:
        output = Path(directory)/'result.json'
        command = [str(PYTHON), str(ROOT/'general_lipreading.py'), str(Path(path).resolve()),
                   '--start',str(start),'--output',str(output)]
        if duration is not None:
            command.extend(['--duration',str(duration)])
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=180,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('Sentence recognition timed out. Try a shorter clip.') from exc
        if process.returncode != 0 or not output.is_file():
            # Native-library logs may contain local paths. Do not return them to clients.
            raise ValueError('Sentence recognition failed. Use a short clip with one clear, centered face.')
        result = json.loads(output.read_text(encoding='utf-8'))
        if not isinstance(result,dict) or result.get('engine')!='auto-avsr' or result.get('audio_used') is not False:
            raise RuntimeError('Unexpected sentence-worker response')
        return result
