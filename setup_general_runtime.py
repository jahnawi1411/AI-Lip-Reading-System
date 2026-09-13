"""Install the optional CPU sentence runtime, sharing the active main environment."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import venv

ROOT = Path(__file__).resolve().parent


def main():
    if Path(sys.prefix).resolve() != (ROOT/'.venv').resolve():
        raise SystemExit('Run this script with the project .venv Python after installing requirements.txt.')
    destination = ROOT/'.venv-general'
    venv.EnvBuilder(with_pip=True).create(destination)
    python = destination/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    library = subprocess.check_output([str(python),'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],
                                     text=True,creationflags=flags).strip()
    (Path(library)/'visionspeak_shared.pth').write_text(sysconfig.get_path('purelib')+'\n',encoding='utf-8')
    subprocess.run([str(python),'-m','pip','install','torch==2.14.0+cpu',
                    '--index-url','https://download.pytorch.org/whl/cpu'],check=True,creationflags=flags)
    subprocess.run([str(python),'-c','import torch, cv2, numpy, mediapipe; print("CPU sentence runtime ready:", torch.__version__)'],
                   check=True,creationflags=flags)


if __name__ == '__main__':
    main()
