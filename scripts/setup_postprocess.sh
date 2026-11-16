#!/usr/bin/env bash
# Run from the repository with the main podcast venv activated.
set -euo pipefail
vc_dir="${PODCAST_VC_VENV:-${HOME}/.local/share/humaneness-podcast-vc-venv}"
python -m venv "$vc_dir"
"$vc_dir/bin/python" -m pip install 'torch==2.6.0' 'torchaudio==2.6.0' --index-url https://download.pytorch.org/whl/cu124
"$vc_dir/bin/python" -m pip install -r requirements-voice-converter.txt
"$vc_dir/bin/python" -m pip install --no-deps 'chatterbox-tts==0.1.7'
python -m podcast_server.prepare_postprocess --vc-python "$vc_dir/bin/python"
