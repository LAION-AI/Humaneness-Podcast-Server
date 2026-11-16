#!/usr/bin/env python3
"""Per-turn post-processing: convert each selected turn to its speaker reference.

Run with the separate VC Python environment after the original job completed.
The original clips and MP3 stay intact. This script is called by the server worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

REVISION = '952ef56c945763be50432738e31a32c30b359619'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    job = args.job_dir.resolve()
    model = args.model_dir.resolve()
    if model.name != REVISION or not (model / 'infer.py').is_file():
        raise ValueError('Expected pinned Hugging Face converter snapshot ' + REVISION)
    if not (job/'episode.mp3').is_file():
        raise ValueError('Job must have completed successfully')
    body = json.loads((job/'script.json').read_text())
    assets = json.loads(args.assets.read_text())
    out = args.output_dir.resolve();out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job/'episode.mp3', out/'original.mp3')
    env = os.environ.copy();env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    clips = []
    for i, turn in enumerate(body['turns']):
        if turn['type'] != 'speech':continue
        source = job/f'turn-{i:04d}.flac'
        info = json.loads((job/f'turn-{i:04d}.json').read_text())
        if digest(source) != info['sha256']:raise ValueError(f'Corrupt source turn {i}')
        voice = body['speakers'][turn['speaker']]
        target = Path(assets['voices'][voice]['path'])
        if digest(target) != assets['voices'][voice]['sha256']:raise ValueError(f'Reference checksum mismatch: {voice}')
        converted = out/f'turn-{i:04d}-vc.wav'; metrics = out/f'turn-{i:04d}-vc.json'
        # The official converter accepts 1–30 seconds. Preserve short, valid TTS
        # utterances by adding trailing silence; never mutate the original clip.
        source_wave, source_sr = sf.read(source, dtype='float32')
        if source_wave.ndim != 1 or not len(source_wave) or not np.isfinite(source_wave).all():
            raise ValueError(f'Invalid source audio for turn {i}')
        padding = max(0, source_sr - len(source_wave))
        source_for_vc = source
        if padding:
            source_for_vc = out/f'turn-{i:04d}-vc-input.wav'
            sf.write(source_for_vc, np.pad(source_wave, (0, padding)), source_sr, subtype='PCM_24')
        # Redo each turn on invocation; no stale conversion is silently reused.
        cmd = [sys.executable, str(model/'infer.py'), '--model-dir', str(model),
               '--source', str(source_for_vc), '--target', str(target), '--output', str(converted),
               '--seed', str(20261011+i), '--flow-steps', '10', '--candidates', '1',
               '--precision', 'bf16', '--metrics', str(metrics)]
        print(f'Converting turn {i+1}: {turn["speaker"]} -> {voice}', flush=True)
        subprocess.run(cmd, env=env, check=True)
        wave, sr = sf.read(converted, dtype='float32')
        if sr != 24000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid conversion for turn {i}')
        clips.append({'turn':i,'speaker':turn['speaker'],'reference':voice,'source_sha256':digest(source),
                      'target_sha256':digest(target),'converted_sha256':digest(converted),'seconds':len(wave)/sr,
                      'source_padding_seconds':padding/source_sr})
    pending = out/'converted.pending.wav'
    with sf.SoundFile(pending, 'w', samplerate=24000, channels=1, subtype='PCM_24') as f:
        trailing = 0
        for i, turn in enumerate(body['turns']):
            if turn['type']=='pause':
                f.write(np.zeros(round(turn['duration_ms']*24), dtype='float32'));trailing=0;continue
            wave, _ = sf.read(out/f'turn-{i:04d}-vc.wav', dtype='float32');f.write(wave)
            if any(t['type']=='speech' for t in body['turns'][i+1:]):
                f.write(np.zeros(round(turn.get('pause_after_ms',400)*24), dtype='float32'))
    tmp = out/'podcast-vc.pending.mp3'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(pending),'-af','loudnorm=I=-18:TP=-1.5:LRA=11','-ar','48000','-ac','1','-c:a','libmp3lame','-b:a','128k',str(tmp)],check=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(tmp),'-f','null','-'],check=True)
    tmp.replace(out/'podcast-vc.mp3');pending.unlink()
    report={'experimental':False,'converter':'laion/Humaneness-Voice-Converter','revision':REVISION,
            'source_job':job.name,'settings':{'flow_steps':10,'precision':'bf16','candidates':1},
            'note':'Original selected and cleaned speech is converted per turn. Original ASR and word timestamps do not verify converted audio.', 'turns':clips}
    (out/'conversion-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Completed:',out/'podcast-vc.mp3',flush=True)

if __name__ == '__main__':main()
