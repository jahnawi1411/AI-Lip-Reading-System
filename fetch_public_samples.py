"""Download a pinned public GRID regression set; never use it for training.

Ten transcripts are derived from GRID sentence codes in filenames, not inferred
from predictions. The separate id2 clip has a supplied alignment transcript.
Speaker identities and overlap with upstream training data are not established.
"""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = 'bec592aa1e66378bc7ab33f6110f1c8a431b3b9a'
BASE = f'https://raw.githubusercontent.com/rizkiarm/LipNet/{REVISION}/evaluation/samples/'
SAMPLES = {
    'bbaf2n': 'bin blue at f two now',
    'brbk7n': 'bin red by k seven now',
    'lbax4n': 'lay blue at x four now',
    'lbbc2a': 'lay blue by c two again',
    'lrwp9a': 'lay red with p nine again',
    'lwbsza': 'lay white by s zero again',
    'pwij3p': 'place white in j three please',
    'sbia1a': 'set blue in a one again',
    'sbwe5n': 'set blue with e five now',
    'swiz3n': 'set white in z three now',
}


def fetch(destination=Path('data/public_regression')):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    registry = json.loads((Path(__file__).resolve().parent / 'examples/public-regression-sources.json').read_text(encoding='utf-8'))
    expected = {row['video']: row['sha256'] for row in registry['samples']}
    rows = []
    sources = [(f'GRID/{name}.mpg', text, 'GRID filename sentence code')
               for name, text in SAMPLES.items()]
    sources.append(('id2_vcd_swwp2s.mpg', 'set white with p two soon', 'swwp2s.align'))
    for relative, transcript, label_source in sources:
        path = destination / Path(relative).name
        url = BASE + relative
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                data = response.read(5 * 1024 * 1024 + 1)
            if len(data) > 5 * 1024 * 1024:
                raise ValueError('Unexpected sample size')
            if hashlib.sha256(data).hexdigest() != expected[path.name]:
                raise ValueError(f'Download hash mismatch: {path.name}')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[path.name]:
            raise ValueError(f'Existing sample differs: {path.name}; refusing to overwrite it.')
        rows.append({'video': path.name, 'text': transcript, 'mode': 'grid',
                     'speaker': 'unverified', 'label_source': label_source,
                     'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(path, flush=True)
    manifest = {'purpose': 'public regression only; not held-out accuracy or training data',
                'source_revision': REVISION, 'speaker_identity_verified': False,
                'training_overlap': 'unknown', 'samples': rows}
    target = destination / 'manifest.json'
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return target


if __name__ == '__main__':
    print(fetch())
