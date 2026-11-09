"""Public asynchronous FastAPI service. No long generation is held in an HTTP request."""
import asyncio,json,secrets,hashlib,time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI,Request,HTTPException,Response,Header
from fastapi.responses import FileResponse,HTMLResponse,JSONResponse
from . import config,store
from .schema import Episode
from .supervisor import Supervisor

@asynccontextmanager
async def lifespan(app):
    supervisor=Supervisor();supervisor.start();app.state.supervisor=supervisor
    yield
    await asyncio.to_thread(supervisor.close)

app=FastAPI(title='Humaneness Podcast Server',version='1.0.0',description='Submit a structured episode; poll its durable job and download a 128 kbit/s mono MP3. English directions, multilingual spoken text.',root_path=config.PREFIX,lifespan=lifespan)

@app.middleware('http')
async def limits(request,call_next):
    # Check actual bytes as well as Content-Length; reject chunked oversized JSON.
    if request.method in ('POST','PUT','PATCH'):
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>512*1024:return JSONResponse({'detail':'Maximum request size: 512 KiB'},status_code=413)
        request._body=bytes(body)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    if request.url.path.startswith('/v1/jobs'):response.headers['Cache-Control']='no-store'
    return response

@app.get('/health')
def health():
    try:a=config.assets();ready=all(Path(a[k]).exists() for k in ('model','codec','asr','aligner'))
    except (RuntimeError,KeyError):ready=False
    with store.connect() as c:counts={s:n for s,n in c.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall()}
    return {'status':'ok','assets_ready':ready,'jobs':counts,'defaults':{'best_of':4,'temperature':1,'cfg':2.5,'mp3_bitrate_bps':128000,'channels':1},'max_queue':config.MAX_QUEUE,'retention_hours':config.TTL_HOURS,'device':config.DEVICE}

@app.get('/v1/voices')
def voices():
    try:a=config.assets()
    except RuntimeError as e:raise HTTPException(503,str(e))
    return {'voices':[{'id':key,'label':v['label'],'reference_sha256':v['sha256'],'source':v.get('source','https://huggingface.co/laion/Humanness-Voice-Podcast/tree/main/examples/references')} for key,v in a['voices'].items()]}

@app.post('/v1/validate')
def validate(body:Episode):
    available={v['id'] for v in voices()['voices']}
    if any(v not in available for v in body.speakers.values()):raise HTTPException(422,'Unknown reference voice ID; see GET /v1/voices')
    return {'valid':True,'speech_turns':sum(t.type=='speech' for t in body.turns),'best_of':body.options.best_of,'script':body.model_dump()}

@app.post('/v1/jobs',status_code=202)
def submit(body:Episode,request:Request,response:Response,idempotency_key:str|None=Header(default=None,max_length=128)):
    validate(body)
    if not health()['assets_ready']:raise HTTPException(503,'Model assets not prepared')
    client=request.cookies.get('podcast_client')
    if not client or len(client)!=43:client=secrets.token_urlsafe(32)
    # Server-controlled ingress should supply the original client address. Loopback-only defaults deliberately share a rate limit.
    addr=request.headers.get('cf-connecting-ip') if request.client.host in ('127.0.0.1','::1') else request.client.host
    rate_client=hashlib.sha256((addr or request.client.host).encode()).hexdigest()
    try:receipt,reused=store.create(body,rate_client,hashlib.sha256((client+':'+idempotency_key).encode()).hexdigest() if idempotency_key else None)
    except OverflowError as e:raise HTTPException(429,str(e),headers={'Retry-After':'60'})
    except ValueError as e:raise HTTPException(409,str(e))
    response.set_cookie('podcast_client',client,httponly=True,samesite='strict',secure=request.headers.get('x-forwarded-proto',request.url.scheme)=='https',max_age=86400,path=config.PREFIX or '/')
    response.headers['Location']=receipt['status_url'];response.headers['Retry-After']='15'
    return {**receipt,'reused':reused,'note':'Keep job_token private. Send Authorization: Bearer <job_token> to poll or download.'}

def authorized(job,request):
    auth=request.headers.get('authorization','')
    # Existing research proxy's allowed custom header provides the same capability when nested there.
    token=auth[7:] if auth.startswith('Bearer ') else request.headers.get('x-audio-decision','')
    row=store.auth(job,token)
    if row is None:raise HTTPException(404,'Job not found')
    return row

@app.get('/v1/jobs/{job}')
def status(job:str,request:Request):return store.status(authorized(job,request))

@app.post('/v1/jobs/{job}/cancel')
def cancel(job:str,request:Request):
    row=authorized(job,request)
    if row['state'] in store.TERMINAL:return {'state':row['state']}
    (store.folder(job)/'cancel').touch()
    if row['state']=='queued':store.update(job,'cancelled')
    return {'state':'cancellation_requested'}

@app.post('/v1/jobs/{job}/retry',status_code=202)
def retry(job:str,request:Request):
    row=authorized(job,request)
    if row['state']!='failed':raise HTTPException(409,'Only failed jobs may be retried')
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        active=c.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running','retrying')").fetchone()[0]
        if active>=config.MAX_QUEUE:raise HTTPException(429,'Queue full')
        c.execute("UPDATE jobs SET state='queued',attempts=0,updated=? WHERE id=?",(time.time(),job))
    return {'job_id':job,'state':'queued','resumes_completed_candidates':True}

@app.get('/v1/jobs/{job}/audio')
def audio(job:str,request:Request):
    row=authorized(job,request)
    if row['state']!='completed':raise HTTPException(409,'Episode is not completed')
    return FileResponse(store.folder(job)/'episode.mp3',media_type='audio/mpeg',filename='podcast-'+job[:8]+'.mp3')

@app.get('/v1/jobs/{job}/report')
def report(job:str,request:Request):
    row=authorized(job,request);p=store.folder(job)
    def read(name,default):return json.loads((p/name).read_text()) if (p/name).exists() else default
    return {'job':store.status(row),'script':read('script.json',{}),'provenance':read('provenance.json',{}),'timeline':read('timeline.json',[]),'turns':[json.loads(x.read_text()) for x in sorted(p.glob('turn-*.json'))],'events':[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()] if (p/'events.jsonl').exists() else []}

@app.delete('/v1/jobs/{job}')
def delete(job:str,request:Request):
    row=authorized(job,request)
    if row['state'] not in store.TERMINAL:raise HTTPException(409,'Cancel first; wait for cancelled state')
    import shutil
    with store.connect() as c:
        shutil.rmtree(store.folder(job));c.execute('DELETE FROM jobs WHERE id=?',(job,))
    return {'deleted':True}

@app.get('/',response_class=HTMLResponse)
def index():return (config.ROOT/'podcast_server/index.html').read_text().replace('__PREFIX__',config.PREFIX)

@app.get('/guide',response_class=Response)
def guide():return Response((config.ROOT/'docs/PODCAST_SCRIPT_WRITING_GUIDE.md').read_text(),media_type='text/plain; charset=utf-8')

@app.get('/example')
def example():return json.loads((config.ROOT/'examples/short-en.json').read_text())
