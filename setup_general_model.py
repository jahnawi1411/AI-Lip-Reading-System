"""Fetch the pinned official Auto-AVSR inference sources and visual-only weights."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen, Request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
REVISION = '182b62837773ab01052d4ac21ef1d2203ea7d267'
SOURCE = f'https://raw.githubusercontent.com/mpc001/auto_avsr/{REVISION}/'
MODEL_ID = '1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_'
MODEL_SHA256 = 'fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb'


def fetch_sources():
    with urlopen(f'https://api.github.com/repos/mpc001/auto_avsr/git/trees/{REVISION}?recursive=1', timeout=60) as response:
        tree = json.load(response)['tree']
    destination = ROOT / 'vendor/auto_avsr'
    provenance = []
    for entry in tree:
        name = entry['path']
        if entry['type'] != 'blob' or not (
            (name.startswith('espnet/') and name.endswith('.py')) or
            name in {'LICENSE', 'README.md', 'spm/unigram/unigram5000_units.txt',
                     'preparation/detectors/mediapipe/20words_mean_face.npy'}):
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with urlopen(SOURCE + name, timeout=60) as response:
                content = response.read(1024 * 1024)
            # Verify the downloaded blob against the pinned Git tree.
            blob = b'blob ' + str(len(content)).encode() + b'\0' + content
            if hashlib.sha1(blob).hexdigest() != entry['sha']:
                raise ValueError(f'Source verification failed: {name}')
            target.write_bytes(content)
        content = target.read_bytes()
        blob = b'blob ' + str(len(content)).encode() + b'\0' + content
        if hashlib.sha1(blob).hexdigest() != entry['sha']:
            raise ValueError(f'Existing source differs: {name}')
        provenance.append({'path': name, 'git_blob': entry['sha'],
                           'sha256': hashlib.sha256(content).hexdigest()})
    (destination / 'PROVENANCE.json').write_text(json.dumps({'revision': REVISION, 'files': provenance}, indent=2))
    print(f'Verified {len(provenance)} upstream files', flush=True)


def fetch_weights():
    destination = ROOT / 'models/auto-avsr'
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'vsr_trlrs2lrs3vox2avsp_base.pth'
    if target.exists():
        with target.open('rb') as existing:
            if hashlib.file_digest(existing, 'sha256').hexdigest() != MODEL_SHA256:
                raise ValueError('Existing sentence model differs from the verified checkpoint.')
        print('Existing model retained:', target, flush=True)
        return
    url = f'https://drive.usercontent.google.com/download?id={MODEL_ID}&export=download&confirm=t'
    temporary = target.with_suffix('.assembled')
    digest = hashlib.sha256()
    md5 = hashlib.md5()
    count = 0
    total = 1001892616  # Content-Range size returned by the official model host.
    chunk_size = 8 * 1024 * 1024
    parts = destination / 'download-parts'
    parts.mkdir(exist_ok=True)
    def part(start):
        end = min(start + chunk_size, total)-1
        path = parts / f'{start:012d}.part'
        if path.exists() and path.stat().st_size == end-start+1:
            return
        request = Request(url, headers={'Range':f'bytes={start}-{end}'})
        with urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                raise ValueError('Server returned an unexpected model range')
            content = response.read(chunk_size+1)
        if len(content) != end-start+1:
            raise ValueError('Incomplete model range')
        path.write_bytes(content)
    offsets = list(range(0,total,chunk_size))
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(part,start) for start in offsets]
        for finished,_ in enumerate(as_completed(jobs),1):
            _.result()
            if finished % 10 == 0:
                print(f'Downloaded {finished}/{len(jobs)} verified ranges', flush=True)
    with temporary.open('wb') as output:
        for start in offsets:
            chunk = (parts/f'{start:012d}.part').read_bytes()
            count += len(chunk); output.write(chunk); digest.update(chunk); md5.update(chunk)
    if digest.hexdigest() != MODEL_SHA256 or not md5.hexdigest().startswith('49f77'):
        raise ValueError('Checkpoint does not match the upstream published MD5 prefix.')
    temporary.replace(target)
    (destination / 'PROVENANCE.json').write_text(json.dumps({
        'source': 'https://github.com/mpc001/auto_avsr', 'revision': REVISION,
        'google_drive_id': MODEL_ID, 'bytes': count, 'sha256': digest.hexdigest(), 'md5': md5.hexdigest(),
        'modality': 'video only', 'training_performed': False}, indent=2))
    print('Saved visual-only checkpoint', flush=True)


if __name__ == '__main__':
    fetch_sources()
    fetch_weights()
