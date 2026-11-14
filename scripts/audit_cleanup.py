"""Operator-only replay of saved selected clips. Never modifies existing jobs.

Outputs contain private episode text/audio: keep --output outside the repository
and public static folders. This is an audit, not a public HTTP endpoint.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path.home()/'.local/share/humaneness-podcast-server')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--limit',type=int,default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True,mode=0o700)
    assets=json.loads((args.data/'assets.json').read_text())
    os.environ.setdefault('HF_HOME',assets['hf_home']);os.environ.setdefault('PODCAST_CACHE',assets['ears_cache'])
    for key in ('codec','asr','aligner'):os.environ['PODCAST_'+key.upper()]=assets[key]
    sys.path.insert(0,str(ROOT/'vendor/podcast_pipeline'))
    from run import setup
    setup(args.device)
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    from transformers import ParakeetProcessor,ParakeetForTDT
    from align_engine import QwenAligner
    from recipe import errors
    from languages import QWEN_ALIGNMENT_LANGUAGES,language_name
    from podcast_server.speech_cleanup import timed_transcription,cleanup
    torch.set_num_threads(4)
    rows=[]
    for path in (args.data/'jobs').glob('*/turn-*.json'):
        try:
            record=json.loads(path.read_text());script=json.loads((path.parent/'script.json').read_text())
            wave_path=path.with_suffix('.flac')
            if wave_path.exists() and .15<record['wer']<.8 and record['seconds']>4 and record['edit']['timing_basis']=='unverified_clip_edges':
                rows.append((path,record,script))
        except (KeyError,OSError,ValueError):continue
    rows.sort(key=lambda x:x[0].stat().st_mtime,reverse=True)
    # Limit to two turns per episode so the replay is not one outlier episode.
    selected=[];counts={}
    for row in rows:
        key=row[0].parent
        if counts.get(key,0)>=2:continue
        selected.append(row);counts[key]=counts.get(key,0)+1
        if len(selected)>=args.limit:break
    tick=time.monotonic()
    proc=ParakeetProcessor.from_pretrained(assets['asr'],local_files_only=True)
    model=ParakeetForTDT.from_pretrained(assets['asr'],dtype=torch.bfloat16,local_files_only=True).to(args.device).eval()
    aligner=QwenAligner(device=args.device);load_s=time.monotonic()-tick
    results=[]
    for number,(path,record,script) in enumerate(selected):
        wave,sr=sf.read(path.with_suffix('.flac'),dtype='float32');assert sr==48000
        w16=resample_poly(wave,1,3).astype('float32');tick=time.monotonic()
        before=timed_transcription(model,proc,w16,args.device)
        alignment=aligner.align(w16,record['text'].split(),language=language_name(script['language'])) if script['language'] in QWEN_ALIGNMENT_LANGUAGES else None
        turn=script['turns'][record['index']]
        direction=' '.join([turn.get('direction','')]+[d['cue'] for d in turn.get('delivery_chunks') or []])
        clip,edit=cleanup(wave,record['text'],before['words'],alignment,direction=direction)
        cleanup_s=time.monotonic()-tick
        sf.write(args.output/f'{number:02d}-before.flac',wave,sr,subtype='PCM_24')
        sf.write(args.output/f'{number:02d}-after.flac',clip,sr,subtype='PCM_24')
        after=timed_transcription(model,proc,resample_poly(clip,1,3).astype('float32'),args.device)
        row=dict(source=str(path),language=script['language'],text=record['text'],before=before,after=after,
                 before_errors=errors(record['text'],before['text']),after_errors=errors(record['text'],after['text']),
                 before_s=len(wave)/sr,after_s=len(clip)/sr,edit=edit,cleanup_with_asr_alignment_s=cleanup_s)
        results.append(row)
        (args.output/'results.json').write_text(json.dumps({'load_s':load_s,'rows':results},indent=2,ensure_ascii=False))
        print(json.dumps({'clip':number,'language':script['language'],'timestamp_status':before['timestamp_status'],
            'words':len(before['words']),'edge_verified':[edit['first_word_verified'],edit['last_word_verified']],
            'removed_tail_s':round(edit['removed_tail_s'],2),'muted_gap_s':round(edit['muted_gap_seconds'],2),
            'wer_before':row['before_errors']['wer'],'wer_after':row['after_errors']['wer'],
            'audit_s':round(cleanup_s,2)}),flush=True)


if __name__=='__main__':main()
