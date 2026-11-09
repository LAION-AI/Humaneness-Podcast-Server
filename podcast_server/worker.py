"""A resumable GPU job process. Token policy and codec/judges never coexist in VRAM."""
import gc,json,os,sys,time,traceback,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
from . import config
from .schema import Episode,Speech
from .store import atomic,folder,disk_ok

class Reporter:
    def __init__(self,path):
        self.path=path;self.value=json.loads((path/'progress.json').read_text());self.start=time.monotonic()
    def __call__(self,phase,message,**extra):
        self.value.update(phase=phase,message=message,heartbeat_at=time.time(),worker_seconds=round(time.monotonic()-self.start,2),**extra)
        atomic(self.path/'progress.json',self.value)
    def event(self,kind,**detail):
        with (self.path/'events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'kind':kind,**detail},ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def invalidate_candidate(path,r,turn,candidate):
    """Retry only technically unusable outputs; never rerank for content quality."""
    counts=path.parent/'regenerations.json'
    values=json.loads(counts.read_text()) if counts.exists() else {}
    key=f'{turn}-{candidate}';values[key]=values.get(key,0)+1
    if values[key]>2:raise RuntimeError('Repeated unusable candidate; operator review needed')
    atomic(counts,values);path.unlink(missing_ok=True);path.with_suffix('.json').unlink(missing_ok=True)
    r.event('invalid_audio_regeneration',turn=turn,candidate=candidate,technical_retry=values[key])

def valid_codes(path):
    if not path.exists():return False
    try:
        from safetensors.torch import load_file
        x=load_file(str(path))['codes']
        return 1<=len(x)<=320 and x.ndim==2 and x.shape[1]==12 and bool(((x>=0)&(x<1024)).all())
    except Exception:return False

def main(job):
    path=folder(job);r=Reporter(path);a=config.assets();episode=Episode.model_validate_json((path/'script.json').read_text());opts=episode.options
    runtime=config.ROOT/'vendor/podcast_pipeline';sys.path.insert(0,str(runtime))
    os.environ.setdefault('HF_HOME',a['hf_home']);os.environ.setdefault('PODCAST_CACHE',a['ears_cache'])
    for k in ('codec','asr','aligner'):os.environ['PODCAST_'+k.upper()]=a[k]
    from run import setup
    setup(config.DEVICE)
    import torch,numpy as np,soundfile as sf
    from safetensors.torch import save_file,load_file
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from phased_generate import TokenEngine
    from offline_batch import OfflineBatch
    from recipe import MODEL_REPO,MODEL_REVISION,GENERAL,sha,errors
    from speaker_scoring import FORMULA
    torch.set_num_threads(4);torch.cuda.set_device(config.DEVICE)
    torch.cuda.set_per_process_memory_fraction(float(os.getenv('PODCAST_GPU_MEMORY_FRACTION','0.9')),config.DEVICE)
    torch._dynamo.config.cache_size_limit=64
    refs={s:dict(a['voices'][voice]) for s,voice in episode.speakers.items()}
    for ref in refs.values():
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Reference integrity check failed')
    turns=[(i,t) for i,t in enumerate(episode.turns) if isinstance(t,Speech)]
    total=len(turns)*opts.best_of;codesdir=path/'codes';codesdir.mkdir(exist_ok=True)
    def codepath(i,c):return codesdir/f'{i:04d}-{c:02d}.safetensors'
    done=sum(valid_codes(codepath(i,c)) for i,t in turns for c in range(opts.best_of))
    fingerprint={'script_sha256':sha(path/'script.json'),'model':MODEL_REPO,'revision':MODEL_REVISION,'reward':FORMULA,'general':GENERAL,
        'references':{s:{**{k:v for k,v in ref.items() if k in ('sha256','code_key','label')},'codes_sha256':sha(ref['codes'])} for s,ref in refs.items()},
        'source_sha256':{str(f.relative_to(config.ROOT)):sha(f) for directory in ('podcast_server','vendor/podcast_pipeline') for f in sorted((config.ROOT/directory).rglob('*.py'))},
        'codec_revision':'f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169','ears_revision':'818506970d93c809a3295a0a02c40dee8ff2bfc3','ecapa_revision':'0f99f2d0ebe89ac095bcc5903c4dd8f72b367286',
        'execution':'BF16 phased token generation, then decoding/ranking/audit; no voice conversion','python':sys.version.split()[0],'torch':torch.__version__,'gpu':torch.cuda.get_device_name(config.DEVICE)}
    provenance=path/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())!=fingerprint:raise RuntimeError('Job provenance changed; submit a new job')
    atomic(provenance,fingerprint)
    proc=None
    if done<total:
        free,_=torch.cuda.mem_get_info(config.DEVICE)
        if free<int(float(os.getenv('PODCAST_MIN_FREE_GB','10'))*1024**3):
            r('waiting_for_gpu','Not enough free VRAM. Existing services are left running.',free_vram_bytes=free)
            return 75
        r('loading_policy','Loading the pinned Podcast checkpoint',completed_candidates=done,total_candidates=total)
        tick=time.monotonic()
        Processor=get_class_from_dynamic_module('processing_moss_tts.MossTTSLocalProcessor',a['model'])
        proc=Processor.from_pretrained(a['model'],codec_path=a['codec'],codec_weight_dtype='bf16',codec_compute_dtype='bf16')
        anchors={}
        for s,ref in refs.items():
            with np.load(ref['codes']) as data:anchors[s]=torch.from_numpy(data[ref['code_key']].astype(np.int64))
            if anchors[s].ndim!=2 or anchors[s].shape[1]!=12 or not 63<=len(anchors[s])<=187:raise ValueError('References must contain 63..187 MOSS frames, about 5..15 seconds')
        old=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16);engine=TokenEngine(Path(a['model']),proc,anchors)
        finally:torch.set_default_dtype(old)
        engine.fast.semantic.limit=1;engine.fast.graph_limit=1
        runner=OfflineBatch(SimpleNamespace(device=config.DEVICE,engine=engine,proc=proc,codec_streams={}))
        r.event('policy_loaded',seconds=time.monotonic()-tick)
        budget=min(opts.best_of,int(os.getenv('PODCAST_CANDIDATE_BATCH','4')))
        # Verify shared prefill with the actual script before reusing candidate KV.
        # This check temporarily duplicates a CPU snapshot, never GPU weights.
        first=turns[0];turn=first[1].model_dump(exclude={'type','pause_after_ms'});turn['index']=first[0]
        if not turn.get('delivery_chunks'):turn['delivery_chunks']=[{'text':turn['text'],'cue':turn['direction']}]
        try:runner.check_prefill([(episode.language,turn)],min(budget,2))
        except torch.cuda.OutOfMemoryError:
            engine.fast.clear_graphs();gc.collect();torch.cuda.empty_cache();runner.shared=False;r.event('prefill_check_oom',shared_prefill=False)
        for i,speech in turns:
            if (path/'cancel').exists():return 130
            turn=speech.model_dump(exclude={'type','pause_after_ms'});turn['index']=i
            if not turn.get('delivery_chunks'):turn['delivery_chunks']=[{'text':turn['text'],'cue':turn['direction']}]
            c=0;failures=0
            while c<opts.best_of:
                if valid_codes(codepath(i,c)):c+=1;continue
                if not disk_ok():raise RuntimeError('Storage budget/free-space limit reached; completed candidates retained')
                n=min(budget,opts.best_of-c)
                # Do not overwrite previously committed candidates after restart.
                while n>1 and any(valid_codes(codepath(i,j)) for j in range(c,c+n)):n-=1
                counts=json.loads((codesdir/'regenerations.json').read_text()) if (codesdir/'regenerations.json').exists() else {}
                seed=(opts.seed+i*1009+c*37+1000003*counts.get(f'{i}-{c}',0))%(2**31)
                r('generating',f'Turn {i+1}: generating candidates {c+1}–{c+n} of {opts.best_of}',current_turn=i,completed_candidates=done,total_candidates=total,candidate_microbatch=n)
                try:
                    _,codes,perf=runner.synthesize([(episode.language,turn)],n=n,max_frames=320,chunk_frames=26,seed=seed,temperature=opts.temperature,guidance=opts.cfg,decode_audio=False)
                    if len(codes)!=n or any(len(x)==0 for x in codes):
                        invalidate_candidate(codepath(i,c),r,i,c);return 76
                    for j,x in enumerate(codes):
                        f=codepath(i,c+j);pending=f.with_suffix('.pending')
                        save_file({'codes':x},str(pending));pending.replace(f)
                        atomic(f.with_suffix('.json'),dict(turn=i,candidate=c+j,**perf,frames=len(x),capped=len(x)>=320))
                    done+=n;c+=n;failures=0
                except torch.cuda.OutOfMemoryError:
                    failures+=1;engine.fast.clear_graphs();gc.collect();torch.cuda.empty_cache()
                    if budget>1:budget=max(1,budget//2)
                    else:
                        r.event('oom',turn=i,candidate=c,microbatch=1);return 76
                    r.event('oom_reduce_microbatch',turn=i,next_microbatch=budget,best_of_unchanged=opts.best_of)
                    if failures>=4:raise
        engine.close();del runner,engine;gc.collect();torch.cuda.empty_cache()
    # Decode and score only unfinished turns, keeping at most N short waves in RAM.
    def selected_ok(i):
        record=path/f'turn-{i:04d}.json';wave=path/f'turn-{i:04d}.flac'
        try:
            if record.exists() and wave.exists() and json.loads(record.read_text())['sha256']==sha(wave):return True
        except (ValueError,KeyError,OSError):pass
        if record.exists() or wave.exists():
            r.event('selected_clip_recovery',turn=i);record.unlink(missing_ok=True);wave.unlink(missing_ok=True)
        return False
    pending=[(i,t) for i,t in turns if not selected_ok(i)]
    if pending:
        r('loading_judges','Policy released; loading codec, Humaneness Ears, ECAPA and alignment models',completed_candidates=total)
        if proc is None:
            Processor=get_class_from_dynamic_module('processing_moss_tts.MossTTSLocalProcessor',a['model'])
            proc=Processor.from_pretrained(a['model'],codec_path=a['codec'],codec_weight_dtype='bf16',codec_compute_dtype='bf16')
        proc.audio_tokenizer.decoder.to(config.DEVICE);proc.audio_tokenizer.quantizer.to(config.DEVICE)
        from generation import DecoderOnly,DecoderGraph
        from speaker_scoring import SpeakerScorer
        from audit_alignment import BoundaryAligner
        from scoring import audio16
        from generate import crop
        codec_graphs={}
        graph_enabled=os.getenv('PODCAST_CODEC_GRAPHS','1')=='1'
        scorer=SpeakerScorer(config.DEVICE,refs,load_asr=True)
        aligner=BoundaryAligner(config.DEVICE)
        # Reuse the frozen final-audit Parakeet for fallback boundaries (Dutch, uncertain Qwen).
        aligner.model=scorer.asr;aligner.proc=scorer.proc
        for i,t in pending:
            if not disk_ok():raise RuntimeError('Storage limit reached during decoding; completed work retained')
            if (path/'cancel').exists():return 130
            r('decoding_and_ranking',f'Decoding and selecting turn {i+1}',current_turn=i,selected_turns=sum((path/f'turn-{j:04d}.json').exists() for j,_ in turns))
            tick=time.monotonic();waves=[]
            codes=[load_file(str(codepath(i,c)))['codes'] for c in range(opts.best_of)]
            decode_batch=min(opts.best_of,4);offset=0
            while offset<len(codes):
                group=codes[offset:offset+decode_batch];length=max(map(len,group))
                x=torch.full((len(group),length,12),1024,dtype=torch.long,device=config.DEVICE)
                for j,item in enumerate(group):x[j,:len(item)]=item.to(config.DEVICE)
                parts=[[] for _ in group]
                try:
                    if graph_enabled and len(group) not in codec_graphs:
                        # Validate the exact graph path against eager decoder on real generated codes.
                        probe=x[:,:26]
                        if probe.shape[1]==26:
                            graph=DecoderGraph(proc.audio_tokenizer,len(group),1024,26)
                            with torch.inference_mode(),torch.cuda.device(config.DEVICE):
                                with DecoderOnly(proc.audio_tokenizer,len(group),1024) as eager:
                                    expected,expected_sizes=eager.decode(probe);expected=expected.clone();expected_sizes=expected_sizes.clone()
                                with graph as fast:actual,actual_sizes=fast.decode(probe)
                                exact=torch.equal(expected,actual) and torch.equal(expected_sizes,actual_sizes)
                            r.event('codec_graph_validation',batch=len(group),bit_identical=exact)
                            if exact:codec_graphs[len(group)]=graph
                            else:graph_enabled=False
                            del expected,actual,expected_sizes,actual_sizes
                    factory=codec_graphs.get(len(group)) if graph_enabled else None
                    with torch.inference_mode(),torch.cuda.device(config.DEVICE),(factory or DecoderOnly(proc.audio_tokenizer,len(group),1024)) as decoder:
                        for block in x.split(26,dim=1):
                            wave,sizes=decoder.decode(block);wave=wave.float().cpu().numpy()
                            for j,size in enumerate(sizes.cpu().tolist()):
                                if size:parts[j].append(wave[j,:size].copy())
                    waves.extend(np.concatenate(p) for p in parts);offset+=len(group)
                except torch.cuda.OutOfMemoryError:
                    del x,parts;codec_graphs.clear()
                    for module in proc.audio_tokenizer.decoder.modules():
                        if hasattr(module,'_streaming_state'):module._streaming_state=None
                        module.__dict__.pop('_update_streaming_cache',None)
                    gc.collect();torch.cuda.empty_cache()
                    if decode_batch==1:raise
                    decode_batch=max(1,decode_batch//2);r.event('codec_reduce_microbatch',next_microbatch=decode_batch)
            invalid=[c for c,w in enumerate(waves) if not len(w) or not np.isfinite(w).all() or float(np.max(np.abs(w)))<1e-5]
            if invalid:
                for c in invalid:invalidate_candidate(codepath(i,c),r,i,c)
                return 76
            decode_s=time.monotonic()-tick
            if opts.best_of>1:
                # Scoring one candidate at a time bounds Ears activations, independently of N.
                scores=[];score_s=0.
                for w in waves:
                    rows,timing=scorer.score_speakers([w],[t.speaker]);scores.extend(rows);score_s+=timing['scoring_seconds']
                winner=max(range(len(scores)),key=lambda j:(scores[j]['reward'],-j))
            else:scores=[{'reward':None,'ranking_skipped':True}];winner=0;score_s=0.
            wave=waves[winner];audit_tick=time.monotonic()
            hypotheses,asr_s=scorer.transcribe([audio16(wave)])
            err=errors(t.text,hypotheses[0]);warnings=[]
            if err['wer']>.15:warnings.append('ASR transcript disagreement; listen before publication')
            if len(codes[winner])>=320:warnings.append('Generation reached 25.6-second ceiling; possible truncation')
            alignment,meta=aligner.align(audio16(wave),t.text.split(),episode.language)
            good=bool(alignment) and all(w[3]>.5 for w in alignment)
            first,last=(float(alignment[0][1]),float(alignment[-1][2])) if good else (0.,len(wave)/48000)
            good=good and 0<=first<last<=len(wave)/48000+.03 and err['wer']<=.15
            if good:
                clip,edit=crop(wave,dict(alignment_ok=True,first_word_s=first,last_word_s=min(last,len(wave)/48000)))
                edit['timing_basis']=meta['timing_basis']
            else:
                clip=wave;edit=dict(first_word_in_clip_s=0.,last_word_in_clip_s=len(wave)/48000,timing_basis='unverified_clip_edges');warnings.append('Uncertain alignment: retained untrimmed audio')
            out=path/f'turn-{i:04d}.flac';temp=out.with_suffix('.pending.flac');sf.write(temp,clip,48000,subtype='PCM_24');temp.replace(out)
            atomic(path/f'turn-{i:04d}.json',dict(index=i,speaker=t.speaker,text=t.text,winner=winner,candidates=scores,warnings=warnings,edit=edit,asr=hypotheses[0],**err,
                token_batches=[json.loads(codepath(i,c).with_suffix('.json').read_text()) for c in range(opts.best_of) if codepath(i,c).with_suffix('.json').exists()],
                seconds=len(clip)/48000,decode_seconds=decode_s,ranking_seconds=score_s,audit_seconds=time.monotonic()-audit_tick,sha256=sha(out),pause_after_ms=t.pause_after_ms))
            r('auditing',f'Turn {i+1} selected and checked',selected_turns=sum((path/f'turn-{j:04d}.json').exists() for j,_ in turns))
        codec_graphs.clear();del aligner,scorer,proc;gc.collect();torch.cuda.empty_cache()
    r('assembling','Assembling speech and scripted pauses; encoding 128 kbit/s mono MP3')
    from .assembly import assemble
    result=assemble(path,episode)
    r('completed','Episode ready to download',**result,completed_candidates=total,selected_turns=len(turns))
    return 0

if __name__=='__main__':
    # A killed API must not leave an orphan GPU worker competing with its restart.
    import ctypes,signal
    parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:sys.exit(130)
    try:sys.exit(main(sys.argv[1]))
    except Exception as exc:
        p=folder(sys.argv[1]);r=Reporter(p);r.event('worker_error',error_type=type(exc).__name__,message=str(exc)[:1000]);r('retrying','Worker error; completed checkpoints retained',last_error=type(exc).__name__+': '+str(exc)[:500]);traceback.print_exc();sys.exit(1)
