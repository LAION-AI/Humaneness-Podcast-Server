#!/usr/bin/env python3
"""Manage only this API. Configure env overrides in DATA/deployment.local.json."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.getenv('PODCAST_DATA',Path.home()/'.local/share/humaneness-podcast-server'));DATA.mkdir(parents=True,exist_ok=True)
PID=DATA/'api.pid'
def owned():
    try:
        pid=int(PID.read_text());return pid if b'podcast_server.app:app' in Path(f'/proc/{pid}/cmdline').read_bytes() else None
    except (FileNotFoundError,ValueError):return None
command=sys.argv[1] if len(sys.argv)>1 else 'status'
if command in ('stop','restart'):
    pid=owned()
    if pid:
        os.kill(pid,signal.SIGTERM)
        for _ in range(300):
            if not owned():break
            time.sleep(.1)
        else:raise SystemExit('Still shutting down; refusing to start a duplicate')
    if command=='stop':print('Stopped');sys.exit()
    command='start'
if command=='start':
    if owned():raise SystemExit('Already running')
    env=os.environ.copy();deploy=DATA/'deployment.local.json'
    if deploy.exists():env.update(json.loads(deploy.read_text()))
    with (DATA/'api.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-m','uvicorn','podcast_server.app:app','--host','127.0.0.1','--port',env.get('PODCAST_PORT','8816'),'--workers','1','--no-access-log'],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
    PID.write_text(str(p.pid));print('Started',p.pid)
elif command=='status':print('Running PID '+str(owned()) if owned() else 'Stopped')
else:raise SystemExit('Use start, stop, restart, status')
