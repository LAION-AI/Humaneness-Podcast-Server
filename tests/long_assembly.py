"""Optional 20-minute-plus assembly stress test, using clearly labelled test tones.

This validates bounded output writing and MP3 duration, NOT long-form TTS quality.
Run from repository root: python -m tests.long_assembly
"""
import tempfile,json,hashlib,time,resource,os
from pathlib import Path
import numpy as np,soundfile as sf
from podcast_server.schema import Episode
from podcast_server.assembly import assemble

def main():
    with tempfile.TemporaryDirectory(prefix='podcast-assembly-') as temporary:
        path=Path(temporary);sr=48000;tone=np.zeros(sr,np.float32);tone[4800:38400]=.02*np.sin(2*np.pi*440*np.arange(33600)/sr)
        original=path/'fixture.flac';sf.write(original,tone,sr,subtype='PCM_24');digest=hashlib.sha256(original.read_bytes()).hexdigest()
        turns=[]
        for i in range(600):
            os.link(original,path/f'turn-{i:04d}.flac')
            (path/f'turn-{i:04d}.json').write_text(json.dumps({'sha256':digest,'edit':{'first_word_in_clip_s':.1,'last_word_in_clip_s':.8,'timing_basis':'test_fixture_not_model_output'},'warnings':[]}))
            turns.append({'type':'speech','speaker':'host','text':'Test fixture.','pause_after_ms':1400})
        episode=Episode(language='en',speakers={'host':'reference_s1'},turns=turns);tick=time.monotonic();result=assemble(path,episode)
        assert result['duration_seconds']>1200
        report={'fixture':'600 authored tone clips; NOT TTS inference','seconds':result['duration_seconds'],'wall_seconds':time.monotonic()-tick,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'bitrate_bps':result['bitrate_bps'],'channels':result['channels'],'output_bytes':result['bytes']}
        Path('/tmp/podcast-long-assembly-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
