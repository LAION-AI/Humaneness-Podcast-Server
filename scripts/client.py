#!/usr/bin/env python3
"""Submit -> durable receipt -> poll -> authenticated MP3 download.

No LLM provider/key required. Keep the receipt private: it includes the job token.
"""
import argparse,json,os,time
from pathlib import Path
import httpx
p=argparse.ArgumentParser();p.add_argument('script',nargs='?');p.add_argument('--base',default='http://127.0.0.1:8816');p.add_argument('--receipt',default='job.local.json');p.add_argument('--output',default='episode.mp3');p.add_argument('--poll',type=float,default=15);a=p.parse_args()
base=a.base.rstrip('/');receipt=Path(a.receipt)
with httpx.Client(timeout=60) as client:
    if receipt.exists():job=json.loads(receipt.read_text())
    else:
        if not a.script:p.error('A script is required for a new job')
        response=client.post(base+'/v1/jobs',json=json.loads(Path(a.script).read_text()));response.raise_for_status();job=response.json()
        fd=os.open(receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump(job,f,indent=2)
    headers={'Authorization':'Bearer '+job['job_token']};url=base+'/v1/jobs/'+job['job_id']
    while True:
        try:r=client.get(url,headers=headers);r.raise_for_status();status=r.json()
        except httpx.TransportError as exc:print(type(exc).__name__,'— retrying status request');time.sleep(a.poll);continue
        print(status.get('state'),status.get('phase'),status.get('message'),'candidates',status.get('completed_candidates'),status.get('total_candidates'),'elapsed',status['elapsed_seconds'],flush=True)
        if status['state']=='completed':break
        if status['state'] in ('failed','cancelled'):raise SystemExit('Job stopped. Inspect its report; completed checkpoints remain available for failed-job retry.')
        time.sleep(a.poll)
    out=Path(a.output);temporary=out.with_suffix('.pending.mp3')
    with client.stream('GET',url+'/audio',headers=headers) as r:
        r.raise_for_status()
        with temporary.open('wb') as f:
            for block in r.iter_bytes():f.write(block)
    temporary.replace(out)
    r=client.get(url+'/report',headers=headers);r.raise_for_status();out.with_suffix('.report.json').write_text(json.dumps(r.json(),indent=2,ensure_ascii=False))
    print('Saved',out,'and its detailed report.')
