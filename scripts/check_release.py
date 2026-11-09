"""Check committed-source scope, local Markdown links and common credential forms."""
from pathlib import Path
import re,subprocess,hashlib,json
root=Path(__file__).resolve().parents[1]
files=[root/p for p in subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()]
errors=[]
for p in files:
    raw=p.read_bytes();rel=p.relative_to(root)
    if p.suffix.lower() in ('.mp3','.wav','.flac','.npz','.pt','.safetensors','.sqlite','.png'):errors.append(f'Unexpected asset: {rel}')
    if re.search(rb'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-or-v1-[A-Za-z0-9]{20,}',raw):errors.append(f'Credential-like value: {rel}')
    if b'/mnt/'+b'nvme/' in raw or b'/home/'+b'c4r33u19/' in raw:errors.append(f'Machine-specific path: {rel}')
    if p.suffix=='.md':
        for target in re.findall(r'\]\(([^\s)]+)\)',raw.decode()):
            if '://' in target or target.startswith('#'):continue
            if not (p.parent/target.split('#')[0]).exists():errors.append(f'Broken local link: {rel} -> {target}')
manifest=json.loads((root/'vendor/podcast_pipeline/SOURCE.json').read_text())
for name,digest in manifest['sha256'].items():
    if hashlib.sha256((root/'vendor/podcast_pipeline'/name).read_bytes()).hexdigest()!=digest:errors.append('Vendored source changed without notice: '+name)
if errors:raise SystemExit('\n'.join(errors))
print(f'Checked {len(files)} tracked files; no matching credentials, private paths, binary audio/weights or broken local Markdown links. Vendored hashes match.')
