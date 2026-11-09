"""Pinned public assets. Existing Hugging Face cache objects are reused."""
import os,json,sys
from pathlib import Path
from huggingface_hub import snapshot_download,hf_hub_download
from . import config
from .store import atomic

def main():
    config.initialize();runtime=config.ROOT/'vendor/podcast_pipeline';sys.path.insert(0,str(runtime))
    from recipe import MODEL_REPO,MODEL_REVISION,sha
    from run import setup
    setup('cpu');import settings
    model=snapshot_download(MODEL_REPO,revision=MODEL_REVISION,allow_patterns=['*.py','*.json','*.safetensors','*.txt','*.model','LICENSE','NOTICE.md'])
    assets={'model':model,'hf_home':str(settings.HF_CACHE),'ears_cache':str(settings.HERE)}
    for key,repo,rev in [('codec','OpenMOSS-Team/MOSS-Audio-Tokenizer-v2','f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169'),('asr','nvidia/parakeet-tdt-0.6b-v3','541d1f99c6b0c3cd0b11a95167540bb8edefd82b'),('aligner','Qwen/Qwen3-ForcedAligner-0.6B-hf','c07281df297b9905d24a508279258cccf987a064')]:
        assets[key]=snapshot_download(repo,revision=rev,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py'])
    snapshot_download('speechbrain/spkrec-ecapa-voxceleb',revision='0f99f2d0ebe89ac095bcc5903c4dd8f72b367286',cache_dir=str(Path(settings.HF_CACHE)/'hub'),allow_patterns=['hyperparams.yaml','*.ckpt','label_encoder.txt'])
    from ears_reward import EarsModels
    EarsModels('cpu').load('medium')
    release=json.loads((runtime/'references/source.json').read_text());voices={}
    source=json.loads((runtime/'references.json').read_text())
    for speaker,ref in source.items():
        ref=dict(ref)
        for key in ('path','codes'):
            ref[key]=hf_hub_download(release['repo'],release['directory']+'/'+Path(ref[key]).name,revision=release['revision'])
        if sha(ref['path'])!=ref['sha256']:raise RuntimeError('Reference checksum mismatch')
        voices['reference_'+speaker.lower()]=ref
    assets['voices']=voices;atomic(config.DATA/'assets.json',assets)
    print('Assets ready. Start: python -m uvicorn podcast_server.app:app --host 127.0.0.1 --port 8816 --workers 1')
if __name__=='__main__':main()
