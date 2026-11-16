"""One queue supervisor, with optional one-job-per-idle-GPU dispatch."""
import fcntl,json,os,signal,subprocess,sys,threading,time
from . import config,store


def gpu_inventory():
    """Fail closed if NVIDIA cannot report both devices and compute processes."""
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    busy={line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    return [(int(index.strip()), uuid.strip(), int(free.strip()), uuid.strip() in busy)
            for index,uuid,free in (line.split(',') for line in devices.splitlines() if line.strip())]

class Supervisor:
    def __init__(self):
        self.stop=threading.Event();self.threads=[];self.processes={};self.process_lock=threading.Lock();self.lock=None
    def start(self):
        config.initialize();self.lock=(config.DATA/'worker.lock').open('w')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store.init()
        devices=[f'cuda:{i}' for i,_,_,_ in gpu_inventory()] if os.getenv('PODCAST_GPU_POOL')=='auto' else [config.DEVICE]
        if not devices:raise RuntimeError('No NVIDIA GPUs found')
        for device in devices:
            thread=threading.Thread(target=self.run,args=(device,),daemon=True);self.threads.append(thread);thread.start()
    def terminate(self,device):
        with self.process_lock:p=self.processes.get(device)
        if p and p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:return
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
    def close(self):
        self.stop.set()
        for device in list(self.processes):self.terminate(device)
        for thread in self.threads:thread.join(timeout=15)
        if self.lock:self.lock.close()
    def available(self,device):
        if os.getenv('PODCAST_GPU_POOL')!='auto':return True
        index=int(device.split(':')[1]);minimum=float(os.getenv('PODCAST_MIN_FREE_GB','10'))*1024
        return any(i==index and not busy and free>=minimum for i,_,free,busy in gpu_inventory())
    def run(self,device):
        while not self.stop.is_set():
            try:
                if not self.available(device):self.stop.wait(5);continue
                store.cleanup();row=store.claim()
                if not row:self.stop.wait(2);continue
                job=row['id'];path=store.folder(job)
                if (path/'cancel').exists():store.update(job,'cancelled');continue
                with (path/'worker.log').open('ab') as log:
                    env=os.environ.copy();env['PODCAST_DEVICE']=device
                    p=subprocess.Popen([sys.executable,'-m','podcast_server.worker',job],cwd=config.ROOT,stdout=log,stderr=log,start_new_session=True,env=env)
                    with self.process_lock:self.processes[device]=p
                    start=time.monotonic()
                    while p.poll() is None and not self.stop.wait(1):
                        if (path/'cancel').exists():self.terminate(device);break
                        heartbeat=(path/'progress.json').stat().st_mtime
                        if time.time()-heartbeat>1800 or time.monotonic()-start>6*3600:
                            self.terminate(device);break
                    if self.stop.is_set():self.terminate(device)
                with self.process_lock:self.processes.pop(device,None)
                with (path/'events.jsonl').open('a') as event_log:
                    event_log.write(json.dumps({'time':time.time(),'kind':'worker_process_exit','device':device,'returncode':p.returncode,'attempt':row['attempts']+1})+'\n')
                if self.stop.is_set():store.update(job,'queued');break
                if (path/'cancel').exists():store.update(job,'cancelled');continue
                if p.returncode==0 and (path/'episode.mp3').exists():store.update(job,'completed');continue
                if p.returncode==75:
                    self.stop.wait(30)
                    if (path/'cancel').exists():store.update(job,'cancelled');continue
                    with store.connect() as c:c.execute("UPDATE jobs SET state='queued',attempts=attempts-1,updated=? WHERE id=?",(time.time(),job))
                    continue
                if row['attempts']+1<3:
                    progress=json.loads((path/'progress.json').read_text())
                    progress.update(phase='retrying',message='Worker stopped; retrying from durable checkpoints',last_process_exit=p.returncode)
                    store.atomic(path/'progress.json',progress)
                    store.update(job,'retrying');self.stop.wait(min(30,5*(row['attempts']+1)))
                    store.update(job,'cancelled' if (path/'cancel').exists() else 'queued')
                else:
                    progress=json.loads((path/'progress.json').read_text());progress.update(phase='failed',message='Three process attempts failed. Completed checkpoints are preserved; fix the reported cause and use Retry.')
                    store.atomic(path/'progress.json',progress);store.update(job,'failed')
            except Exception:
                import logging;logging.exception('Queue supervisor error on %s',device);self.stop.wait(5)
