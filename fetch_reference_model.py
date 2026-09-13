"""Install the pinned public GRID checkpoint, verifying its SHA-256."""
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

REVISION = 'bec592aa1e66378bc7ab33f6110f1c8a431b3b9a'
URL = f'https://raw.githubusercontent.com/rizkiarm/LipNet/{REVISION}/evaluation/models/unseen-weights178.h5'
SHA256 = 'b4ea5202691223ff8a91141a014f7477df53e945bd5c4fcae8413691db13d181'
DESTINATION = Path(__file__).resolve().parent / 'models/lipnet-reference/unseen-weights178.h5'


def fetch(destination=DESTINATION):
    destination = Path(destination)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Existing reference weights differ; refusing to overwrite them.')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix='lipnet-', suffix='.download', dir=destination.parent, delete=False)
    temporary = Path(handle.name)
    try:
        digest = hashlib.sha256()
        total = 0
        with handle as output, urlopen(URL, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 20 * 1024 * 1024:
                    raise ValueError('Unexpected model size')
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError('Downloaded checkpoint hash mismatch')
        temporary.replace(destination)
    finally:
        handle.close()
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == '__main__':
    print(fetch())
