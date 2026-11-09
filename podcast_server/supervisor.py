"""One worker process with crash/OOM backoff and checkpoint-based resumption."""
import fcntl,json,os,signal,subprocess,sys,threading,time
from . import config,store

class Supervisor:
    def __init__(self):self.stop=threading.Event();self.process=None;self.thread=None;self.lock=None
    def start(self):
        config.initialize();self.lock=(config.DATA/'worker.lock').open('w')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store.init();self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def terminate(self):
        p=self.process
        if p and p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    def close(self):
        self.stop.set();self.terminate()
        if self.thread:self.thread.join(timeout=15)
        if self.lock:self.lock.close()
    def run(self):
        while not self.stop.is_set():
            try:
                store.cleanup();row=store.claim()
                if not row:self.stop.wait(2);continue
                job=row['id'];path=store.folder(job)
                with (path/'worker.log').open('ab') as log:
                    p=subprocess.Popen([sys.executable,'-m','podcast_server.worker',job],cwd=config.ROOT,stdout=log,stderr=log,start_new_session=True)
                    self.process=p;start=time.monotonic()
                    while p.poll() is None and not self.stop.wait(1):
                        if (path/'cancel').exists():self.terminate();break
                        heartbeat=(path/'progress.json').stat().st_mtime
                        if time.time()-heartbeat>1800 or time.monotonic()-start>6*3600:
                            self.terminate();break
                self.process=None
                with (path/'events.jsonl').open('a') as event_log:
                    event_log.write(json.dumps({'time':time.time(),'kind':'worker_process_exit','returncode':p.returncode,'attempt':row['attempts']+1})+'\n')
                if self.stop.is_set():store.update(job,'queued');break
                if (path/'cancel').exists():store.update(job,'cancelled');continue
                if p.returncode==0 and (path/'episode.mp3').exists():store.update(job,'completed');continue
                if p.returncode==75:
                    self.stop.wait(30)
                    with store.connect() as c:c.execute("UPDATE jobs SET state='queued',attempts=attempts-1,updated=? WHERE id=?",(time.time(),job))
                    continue
                if row['attempts']+1<3:
                    progress=json.loads((path/'progress.json').read_text())
                    progress.update(phase='retrying',message='Worker stopped; retrying from durable checkpoints',last_process_exit=p.returncode)
                    store.atomic(path/'progress.json',progress)
                    store.update(job,'retrying');self.stop.wait(min(30,5*(row['attempts']+1)))
                    store.update(job,'queued')
                else:
                    progress=json.loads((path/'progress.json').read_text());progress.update(phase='failed',message='Three process attempts failed. Completed checkpoints are preserved; fix the reported cause and use Retry.')
                    store.atomic(path/'progress.json',progress);store.update(job,'failed')
            except Exception:
                import logging;logging.exception('Queue supervisor error');self.stop.wait(5)
