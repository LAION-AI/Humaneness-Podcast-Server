# Deployment and operations

## Standalone service

Install the pinned environment and assets as described in README. The API process stays lightweight; it launches one subprocess per job. That subprocess loads the policy once, generates all unfinished candidates, releases it, then loads the decoder/judges and assembles the episode. Process exit releases CUDA memory. This amortizes loads over a long episode, but short jobs have substantial cold overhead.

Use **one Uvicorn worker**. A filesystem flock prevents two queue supervisors from running against the same data store. Multiple independent stores on the same GPU are not coordinated; operators must budget those separately.

```bash
export PODCAST_DEVICE=cuda:0
export PODCAST_DATA=/data/podcast-jobs
python scripts/manage.py start
python scripts/manage.py status
python scripts/manage.py restart
python scripts/manage.py stop
```

`manage.py` manages only a process whose command matches `podcast_server.app:app`. It never restarts a tunnel, Robby, ASR or someone else's GPU job. Runtime configuration can also be placed in `$PODCAST_DATA/deployment.local.json` as a JSON object of environment-variable strings. It is outside Git. Use the same Python environment for `manage.py` as for the worker.

For automatic reboot recovery, run the Uvicorn command under your normal systemd/container supervisor with `Restart=on-failure`. Do not run `manage.py start` from inside a systemd service that expects a foreground process. Mount the data directory on persistent storage; `/dev/shm` is not durable. The current research host reuses some cached weights from a pre-existing RAM filesystem; rerun preparation/repoint assets if those disappear after reboot. Job state itself is on persistent disk.

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `PODCAST_DATA` | `~/.local/share/humaneness-podcast-server` | Private SQLite queue, assets config, jobs and logs |
| `PODCAST_ASSETS` | `$PODCAST_DATA/assets.json` | Operator-only model/reference paths |
| `PODCAST_DEVICE` | `cuda:0` | Selected GPU |
| `PODCAST_CODEC_GRAPHS` | `1` | Reuse a decoder CUDA graph after a bit-identity check against eager decoding |
| `PODCAST_CANDIDATE_BATCH` | `4` | Maximum simultaneous candidates; independent of N |
| `PODCAST_GPU_MEMORY_FRACTION` | `0.9` | PyTorch allocator ceiling, fraction of total GPU VRAM; CUDA context is extra |
| `PODCAST_MIN_FREE_GB` | `10` | Free VRAM required before loading policy; measured in GiB |
| `PODCAST_MAX_QUEUE` | `4` | Total unfinished jobs admitted |
| `PODCAST_TTL_HOURS` | `72` | Terminal-job retention |
| `PODCAST_MAX_STORE_GB` | `8` | Job-store budget; also require >2 GiB free disk |
| `PODCAST_ROOT_PATH` | empty | Public reverse-proxy prefix, e.g. `/podcast` |
| `PODCAST_PORT` | `8816` | Port used by `scripts/manage.py` |
| `HF_HOME` | Hugging Face default | Persistent model cache |
| `PODCAST_CACHE` | `~/.cache/humaneness-podcast` | Pinned Ears model cache |
| `TORCHINDUCTOR_CACHE_DIR` | Torch default | Reusable compiled kernels; keep outside repository |

On the shared test host: GPU 1, candidate microbatch 4, allocator fraction 0.44, minimum 10 GiB free, and existing compile/model caches. Robby, TTS, ASR and the Audio Decision lab were left running. This is a measured hosting configuration, not a universal recommendation for every GPU. See TESTING.md for actual timings.

Storage guards check before admission and new candidate work. A concurrent external process can still consume disk between checks; write failures are retained as errors, not falsely reported as success. Cleanup removes expired terminal jobs only. It does not delete model weights or someone else's data. For space pressure, delete completed jobs through the API or archive them before removal. Do not delete the directory of a running worker.

## Existing HTTPS server

`integrations/fastapi_proxy.py` registers a dedicated `/podcast` prefix before the host application's catch-all. The podcast process listens only on loopback; the existing tunnel points to the host server as before. Submission/polling/download requests are short, so there is no 20-minute HTTP response for Cloudflare to time out.

The current public deployment reuses the already-running FastAPI facade/tunnel at:

`https://arrive-locations-parade-research.trycloudflare.com/podcast/`

The original Robby URL remains `https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/`. No new Cloudflare tunnel is required for this service. If your public prefix differs, set `PODCAST_ROOT_PATH` consistently so Swagger and response URLs resolve correctly.

Only trust the `CF-Connecting-IP` client-address header from a trusted loopback ingress which receives it from Cloudflare. If exposing a different proxy or binding directly to the Internet, configure/strip forwarded headers and add authentication/rate limiting there. Job capability tokens are needed even for the file download. Do not log Authorization headers, cookies, request bodies or job-token receipts. TLS is supplied by the ingress. Long-term uptime requires your own stable domain/tunnel and process supervision.

## Custom reference voices

The public API deliberately accepts a voice catalog ID, not arbitrary paths or URLs. To add a voice, prepare a clean 5–15-second single-speaker recording with appropriate permission, encode it with the pinned MOSS-v2 processor, and store an NPZ code matrix shaped `[frames, 12]` alongside it. Then add a record to `assets.json` → `voices` with `label`, absolute `path`, audio `sha256`, absolute `codes`, and `code_key`. Validate 63–187 frames for the tested 5–15-second range. Replacing the MP3 alone is wrong: cached codes are the actual conditioning and must correspond to it. Restart between jobs after changing the catalog. Existing jobs whose provenance changes will reject resumption.

Reference recordings are not covered by this server's CC BY 4.0 code license. This repository downloads only the published example references, preserving their provenance notices. It never saves browser microphone data because there is no microphone-capture endpoint here.

## Troubleshooting

- **Waiting for GPU:** inspect `nvidia-smi`; free unused models only if you own them. Reduce the operator microbatch for transient activation OOM. Do not lower N secretly.
- **First request is slow:** imports, CPU checkpoint construction, CUDA compilation/graph creation, and several frozen judge models load before the final export. Warm kernels persist on disk; GPU weights do not stay resident between jobs in this shared-host implementation.
- **Incomplete words:** inspect the report's ASR and cap flags. Split turns shorter and submit a new job; the maximum-reward take can omit content.
- **Failed/retrying:** read private `jobs/<id>/events.jsonl` and `worker.log`. Use `/retry` after fixing a persistent cause. Saved candidate checksums should remain unchanged.
- **Bad download:** the endpoint is available only after full MP3 verification. Check proxy Range/header handling and the job token.
- **Model import mismatch:** use the pinned environment; custom remote model code and Transformer internals are version sensitive.
