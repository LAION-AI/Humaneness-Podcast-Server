"""Download pinned post-processing assets and register the isolated converter runtime."""
import argparse,os
from pathlib import Path
from huggingface_hub import snapshot_download
from . import config
from .store import atomic

VC_REV='952ef56c945763be50432738e31a32c30b359619'
SIDON_REV='b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52'
PRE_REV='da985ba0987f70aaeb84a80f2851cfac8c697a7b'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vc-python',type=Path,required=True,help='Python executable in the isolated converter environment')
    a=p.parse_args();python=a.vc_python.absolute()
    if not python.is_file():raise SystemExit('Converter Python not found; install the converter environment first')
    config.initialize();assets=config.assets()
    vc=Path(snapshot_download('laion/Humaneness-Voice-Converter',revision=VC_REV,allow_patterns=['infer.py','model.safetensors','conditioning/*','conditioning/**/*','NOTICE.md','README.md']))
    sidon=Path(snapshot_download('sarulab-speech/sidon-v0.1',revision=SIDON_REV,allow_patterns=['feature_extractor_cuda.pt','decoder_cuda.pt','README.md']))
    pre=Path(snapshot_download('facebook/w2v-bert-2.0',revision=PRE_REV,allow_patterns=['preprocessor_config.json']))
    assets['postprocess']={'vc_python':str(python),'vc_infer':str(vc/'infer.py'),'sidon_model':str(sidon),'sidon_preprocessor':str(pre)}
    atomic(Path(os.environ.get('PODCAST_ASSETS',config.DATA/'assets.json')),assets)
    print('Pinned converter and SIDON assets are ready.')
if __name__=='__main__':main()
