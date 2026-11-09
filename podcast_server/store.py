"""Durable single-worker queue, opaque capabilities, atomic checkpoints, restart recovery."""
import hashlib,json,secrets,sqlite3,time,threading,os,shutil
from pathlib import Path
from . import config

TERMINAL={'completed','failed','cancelled'}

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def folder(job):return config.DATA/'jobs'/job

def connect():
    c=sqlite3.connect(config.DATA/'jobs.sqlite',timeout=30);c.row_factory=sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL');return c

def init():
    config.initialize()
    with connect() as c:
        c.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, key_hash TEXT, client TEXT, state TEXT, created REAL, updated REAL, attempts INTEGER DEFAULT 0, idempotency TEXT, request_hash TEXT)')
        c.execute("UPDATE jobs SET state='queued',updated=? WHERE state IN ('running','retrying')",(time.time(),))

def disk_ok():
    used=sum(p.stat().st_size for p in (config.DATA/'jobs').rglob('*') if p.is_file())
    return used<config.MAX_STORE_GB*1024**3 and shutil.disk_usage(config.DATA).free>2*1024**3

def create(episode,client,idempotency=None):
    body=episode.model_dump();digest=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        if idempotency:
            row=c.execute('SELECT * FROM jobs WHERE client=? AND idempotency=?',(client,idempotency)).fetchone()
            if row:
                if row['request_hash']!=digest:raise ValueError('Idempotency-Key reused with a different script')
                receipt=json.loads((folder(row['id'])/'receipt.json').read_text());return receipt,True
        if c.execute("SELECT count(*) FROM jobs WHERE state NOT IN ('completed','failed','cancelled')").fetchone()[0]>=config.MAX_QUEUE:raise OverflowError('Queue is full; retry later')
        if c.execute('SELECT count(*) FROM jobs WHERE client=? AND created>?',(client,time.time()-3600)).fetchone()[0]>=6:raise OverflowError('Limit: six jobs per hour per client')
        if not disk_ok():raise OverflowError('Storage budget reached; retry after cleanup')
        job=secrets.token_hex(16);key=secrets.token_urlsafe(32);now=time.time()
        path=folder(job);path.mkdir(mode=0o700)
        atomic(path/'script.json',body)
        receipt={'job_id':job,'job_token':key,'status_url':f'{config.PREFIX}/v1/jobs/{job}','poll_after_seconds':15}
        atomic(path/'receipt.json',receipt)
        atomic(path/'progress.json',{'phase':'queued','message':'Waiting for the single GPU worker','speech_turns':sum(t['type']=='speech' for t in body['turns']),'completed_candidates':0,'selected_turns':0})
        c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',(job,hashlib.sha256(key.encode()).hexdigest(),client,'queued',now,now,0,idempotency,digest))
        return receipt,False

def auth(job,key):
    if len(job)!=32 or any(x not in '0123456789abcdef' for x in job):return None
    with connect() as c:row=c.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone()
    if row and key and secrets.compare_digest(row['key_hash'],hashlib.sha256(key.encode()).hexdigest()):return dict(row)
    return None

def update(job,state):
    with connect() as c:c.execute('UPDATE jobs SET state=?, updated=? WHERE id=?',(state,time.time(),job))

def claim():
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
        if row:
            c.execute("UPDATE jobs SET state='running',updated=?,attempts=attempts+1 WHERE id=?",(time.time(),row['id']))
            return dict(row)

def status(row):
    path=folder(row['id'])
    try:progress=json.loads((path/'progress.json').read_text())
    except FileNotFoundError:progress={}
    with connect() as c:position=c.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running','retrying') AND created<?",(row['created'],)).fetchone()[0]
    return {'job_id':row['id'],'state':row['state'],'created_at':row['created'],'updated_at':row['updated'],'process_attempts':row['attempts'],'queue_ahead':position,'elapsed_seconds':round((row['updated'] if row['state'] in TERMINAL else time.time())-row['created'],1),**progress,'download_url':f"{config.PREFIX}/v1/jobs/{row['id']}/audio" if row['state']=='completed' else None}

def cleanup():
    with connect() as c:
        rows=c.execute("SELECT id FROM jobs WHERE state IN ('completed','failed','cancelled') AND updated<?",(time.time()-config.TTL_HOURS*3600,)).fetchall()
        for row in rows:
            shutil.rmtree(folder(row['id']),ignore_errors=True);c.execute('DELETE FROM jobs WHERE id=?',(row['id'],))
