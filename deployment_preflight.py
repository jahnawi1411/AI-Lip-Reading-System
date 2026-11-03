"""Check installed inference assets before attempting a container build."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_assets(root=ROOT):
    required = [
        'models/checkpoint.weights.h5', 'models/face_landmarker.task',
        'models/lipnet-reference/unseen-weights178.h5',
        'models/auto-avsr/PROVENANCE.json',
        'models/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth',
        'vendor/auto_avsr/PROVENANCE.json', 'vendor/auto_avsr/LICENSE',
        'audiovisual.py', 'general_lipreading.py', 'general_bridge.py',
    ]
    errors = [f'Missing: {name}' for name in required if not (root / name).is_file()]
    if errors:
        return errors
    from setup_general_model import MODEL_SHA256
    checkpoint = root / 'models/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth'
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != MODEL_SHA256:
        errors.append('English checkpoint does not match the pinned SHA-256.')
    metadata = json.loads((root / 'models/auto-avsr/PROVENANCE.json').read_text())
    if metadata.get('sha256') != digest:
        errors.append('English checkpoint provenance hash does not match the file.')
    vendor = root / 'vendor/auto_avsr'
    metadata = json.loads((vendor / 'PROVENANCE.json').read_text())
    for entry in metadata['files']:
        source = vendor / entry['path']
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            errors.append(f'Vendored source missing or changed: {entry["path"]}')
    return errors


if __name__ == '__main__':
    problems = check_assets()
    for problem in problems:
        print(problem)
    if problems:
        raise SystemExit(1)
    print('Deployment assets and English model/source hashes verified.')
    print('This does not verify Linux dependencies, Docker, HTTPS or hosting.')
