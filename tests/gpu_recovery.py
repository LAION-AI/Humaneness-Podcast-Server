"""Destructive ONLY to this test's own GPU job, never other jobs/services.

Operator-local integration test. Creates a real N=1 job, kills its own worker
once after a candidate is durably committed, then verifies completion and hashes.
Requires the live API and its data directory on the same machine.

python -m tests.gpu_recovery --base http://127.0.0.1:8816 --data /path/to/job-store
"""
import argparse,json,time,os,signal,hashlib
from pathlib import Path
import httpx

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8816');p.add_argument('--data',required=True);p.add_argument('--output',default='/tmp/podcast-gpu-recovery.json');args=p.parse_args()
    data=Path(args.data);script=json.loads((Path(__file__).resolve().parents[1]/'examples/short-en.json').read_text());script['options']['best_of']=1;script['turns']*=5
    with httpx.Client(timeout=30) as client:
        response=client.post(args.base+'/v1/jobs',json=script);response.raise_for_status();receipt=response.json();job=receipt['job_id'];headers={'Authorization':'Bearer '+receipt['job_token']};path=data/'jobs'/job
        started=time.monotonic();before={};killed=False
        while time.monotonic()-started<1800:
            state=client.get(args.base+'/v1/jobs/'+job,headers=headers).json()
            if not killed:
                codes=list((path/'codes').glob('*.safetensors'))
                if codes:
                    api=int((data/'api.pid').read_text());children=set()
                    for f in Path('/proc',str(api),'task').glob('*/children'):children.update(f.read_text().split())
                    for pid in children:
                        command=Path('/proc',pid,'cmdline').read_bytes()
                        if b'podcast_server.worker' in command and job.encode() in command:
                            before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in codes};os.kill(int(pid),signal.SIGKILL);killed=True;break
            if state['state']=='completed':
                assert killed,'Test missed the interruption window'
                assert state['process_attempts']>=2
                assert all(hashlib.sha256((path/'codes'/name).read_bytes()).hexdigest()==sha for name,sha in before.items())
                result={'actual_gpu_inference':True,'interruption':'SIGKILL of this test job worker only','completed':True,'saved_candidates_preserved':len(before),'process_attempts':state['process_attempts'],'seconds':time.monotonic()-started}
                Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result));return
            if state['state']=='failed':raise RuntimeError(json.dumps(state))
            time.sleep(.25 if not killed else 2)
        raise TimeoutError('Recovery test did not complete in 30 minutes')
if __name__=='__main__':main()
