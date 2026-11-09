"""Bounded-memory word-clock assembly, explicit silence, atomic verified MP3 export."""
import json,subprocess,time,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from .schema import Speech,Pause
from .store import atomic
SR=48000

def silence(writer,seconds):
    left=max(0,round(seconds*SR));block=np.zeros(SR,np.float32)
    while left: n=min(left,SR);writer.write(block[:n]);left-=n

def assemble(path,episode):
    tick=time.monotonic();cursor=0.;last_word=None;gap=0.;timeline=[];reviews=[]
    source=path/'assembled.pending.flac'
    with sf.SoundFile(source,'w',samplerate=SR,channels=1,subtype='PCM_24') as f:
        for i,t in enumerate(episode.turns):
            if isinstance(t,Pause):gap+=t.duration_ms/1000;continue
            row=json.loads((path/f'turn-{i:04d}.json').read_text());clip=path/f'turn-{i:04d}.flac'
            if hashlib.sha256(clip.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Selected audio checksum mismatch')
            first=row['edit']['first_word_in_clip_s'];last=row['edit']['last_word_in_clip_s']
            needed=gap if last_word is None else max(0.,gap-(cursor-last_word)-first)
            silence(f,needed);cursor+=needed
            actual_gap=None if last_word is None else cursor+first-last_word
            entry={'turn':i,'speaker':t.speaker,'text':t.text,'start_s':cursor,'speech_start_s':cursor+first,'speech_end_s':cursor+last,'requested_gap_s':gap,'actual_gap_s':actual_gap,'timing_basis':row['edit']['timing_basis']}
            for block in sf.blocks(clip,blocksize=SR,dtype='float32'):f.write(block);cursor+=len(block)/SR
            last_word=entry['speech_end_s'];entry['end_s']=cursor;timeline.append(entry);gap=t.pause_after_ms/1000
            if row['warnings']:reviews.append({'turn':i,'warnings':row['warnings']})
        # Explicit pause elements at the end are retained; default after-turn gap is not appended.
        trailing=0.
        for t in reversed(episode.turns):
            if isinstance(t,Pause):trailing+=t.duration_ms/1000
            else:break
        silence(f,trailing);cursor+=trailing
    target=path/'episode.pending.mp3'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-af','loudnorm=I=-18:TP=-1.5:LRA=11','-ar','48000','-ac','1','-c:a','libmp3lame','-b:a','128k','-metadata','title='+episode.title,str(target)],check=True,timeout=900,capture_output=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)]))
    audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
    if audio['channels']!=1 or audio['codec_name']!='mp3' or int(audio['bit_rate'])!=128000:raise ValueError('MP3 format verification failed')
    if abs(float(probe['format']['duration'])-cursor)>.2:raise ValueError('MP3 duration verification failed')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(target),'-f','null','-'],check=True,timeout=600,capture_output=True)
    target.replace(path/'episode.mp3');source.unlink()
    atomic(path/'timeline.json',timeline)
    result={'duration_seconds':cursor,'format':'mp3','channels':1,'bitrate_bps':128000,'sample_rate':48000,'bytes':(path/'episode.mp3').stat().st_size,'review_flags':reviews,'assembly_seconds':time.monotonic()-tick}
    atomic(path/'result.json',result)
    return result
