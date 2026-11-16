#!/usr/bin/env python3
"""Restore a complete mono podcast using SaruLab's published SIDON inference.

Adapted from https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta/blob/cc6ac8f4b41c16a81d2a6984f8de1c4919a6c150/app.py
Model: sarulab-speech/sidon-v0.1, MIT; original code and model retain their licenses.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import SeamlessM4TFeatureExtractor

MODEL_REV='b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52'
PREPROCESSOR_REV='da985ba0987f70aaeb84a80f2851cfac8c697a7b'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--preprocessor-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    if args.model_dir.resolve().name != MODEL_REV or args.preprocessor_dir.resolve().name != PREPROCESSOR_REV:
        raise ValueError('SIDON model and preprocessor snapshots must match pinned revisions')
    if not torch.cuda.is_available():raise RuntimeError('SIDON requires a CUDA GPU')
    torch.set_num_threads(4);started=time.perf_counter()
    wave,sr=sf.read(args.input,dtype='float32',always_2d=True)
    wave=wave.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all():raise ValueError('Invalid audio')
    peak=float(np.max(np.abs(wave)))
    if peak<1e-8:raise ValueError('Input is silent')
    target_n=round(len(wave)*48000/sr)
    x=torch.from_numpy(wave).view(1,-1)*(.9/peak)
    x=torchaudio.functional.highpass_biquad(x,sr,50)
    x=torchaudio.functional.resample(x,sr,16000)
    x=torch.nn.functional.pad(x,(0,24000)).flatten()
    fe=torch.jit.load(str(args.model_dir/'feature_extractor_cuda.pt'),map_location='cuda').eval()
    dec=torch.jit.load(str(args.model_dir/'decoder_cuda.pt'),map_location='cuda').eval()
    pre=SeamlessM4TFeatureExtractor.from_pretrained(str(args.preprocessor_dir),local_files_only=True)
    chunks=[];cache=None
    with torch.inference_mode():
        for chunk in x.split(16000*96):
            inp=pre(torch.nn.functional.pad(chunk,(160,160)),sampling_rate=16000,return_tensors='pt')
            features=fe(inp['input_features'].cuda())['last_hidden_state']
            if cache is not None:features=torch.cat([cache,features],dim=1)
            chunks.append(dec(features.transpose(1,2)).flatten()[:-960].float().cpu())
            cache=features[:,-1:]
    restored=torch.cat(chunks)[:target_n].numpy()
    if len(restored)!=target_n or not np.isfinite(restored).all() or np.max(np.abs(restored))<1e-8:
        raise ValueError('SIDON produced invalid or truncated output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(args.output,restored,48000,subtype='PCM_24')
    report={'model':'sarulab-speech/sidon-v0.1','revision':MODEL_REV,'preprocessor_revision':PREPROCESSOR_REV,
            'input_seconds':len(wave)/sr,'output_seconds':len(restored)/48000,'chunks':len(chunks),
            'sample_rate':48000,'processing_seconds':round(time.perf_counter()-started,3),
            'note':'Full podcast processed with the published demo preprocessing and 96-second chunking.'}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
