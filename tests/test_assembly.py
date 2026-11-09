"""Signal fixtures test timing and encoding only; they are not TTS examples."""
import json,hashlib,subprocess
import numpy as np
import soundfile as sf
from podcast_server.assembly import assemble
from podcast_server.schema import Episode
from .test_contract import example

def test_mp3_and_exact_pause_with_bounded_assembly(tmp_path):
    script=example();episode=Episode.model_validate(script);sr=48000
    for i in (0,2):
        wave=np.zeros(sr,np.float32);wave[4800:38400]=.1*np.sin(2*np.pi*440*np.arange(33600)/sr)
        p=tmp_path/f'turn-{i:04d}.flac';sf.write(p,wave,sr,subtype='PCM_24')
        (tmp_path/f'turn-{i:04d}.json').write_text(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'edit':{'first_word_in_clip_s':.1,'last_word_in_clip_s':.8,'timing_basis':'synthetic_fixture'},'warnings':[]}))
    result=assemble(tmp_path,episode);timeline=json.loads((tmp_path/'timeline.json').read_text())
    assert abs(timeline[1]['speech_start_s']-timeline[0]['speech_end_s']-.9)<1/sr
    assert result['bitrate_bps']==128000 and result['channels']==1
    assert (tmp_path/'episode.mp3').stat().st_size>10000
    assert not (tmp_path/'assembled.pending.flac').exists()
