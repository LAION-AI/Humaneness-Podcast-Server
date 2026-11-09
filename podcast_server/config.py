"""Operator configuration is separate from untrusted public script input."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DATA=Path(os.getenv('PODCAST_DATA',Path.home()/'.local/share/humaneness-podcast-server')).resolve()
PREFIX=os.getenv('PODCAST_ROOT_PATH','').rstrip('/')
DEVICE=os.getenv('PODCAST_DEVICE','cuda:0')
MAX_QUEUE=int(os.getenv('PODCAST_MAX_QUEUE','4'))
TTL_HOURS=int(os.getenv('PODCAST_TTL_HOURS','72'))
MAX_STORE_GB=float(os.getenv('PODCAST_MAX_STORE_GB','8'))

def initialize():
    DATA.mkdir(parents=True,exist_ok=True,mode=0o700)
    (DATA/'jobs').mkdir(exist_ok=True,mode=0o700)

def assets():
    import json
    path=Path(os.getenv('PODCAST_ASSETS',DATA/'assets.json'))
    if not path.is_file():raise RuntimeError('Assets not prepared. Run python -m podcast_server.prepare')
    return json.loads(path.read_text())
