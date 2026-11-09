# API contract and job recovery

All paths below are relative to the server base URL, such as `https://HOST/podcast` or `http://127.0.0.1:8816`. The interactive reference is `/docs`, machine-readable schema `/openapi.json`.

| Method / path | Purpose |
|---|---|
| `GET /health` | Service, asset availability, queue counts, defaults and retention |
| `GET /v1/voices` | Configured reference voice IDs and hashes |
| `POST /v1/validate` | Validate episode JSON without GPU work |
| `POST /v1/jobs` | Submit an episode; HTTP 202 immediately, before loading models |
| `GET /v1/jobs/{job_id}` | Detailed progress, stage, counts, queue position and result URL |
| `POST /v1/jobs/{job_id}/cancel` | Stop this job; retained checkpoints/results expire normally |
| `POST /v1/jobs/{job_id}/retry` | Requeue a failed job using its completed checkpoints |
| `GET /v1/jobs/{job_id}/audio` | Completed MP3; supports HTTP Range through FileResponse |
| `GET /v1/jobs/{job_id}/report` | Script, model provenance, candidate scores, selected turns, diagnostics, timeline and recovery events |
| `DELETE /v1/jobs/{job_id}` | Delete a terminal job and all its private data |

## Submit and poll

The request body is the episode JSON itself. It is not wrapped in `script` or sent as an arbitrary filename. Omitted `options` use Best-of-4, temperature 1.0, CFG 2.5 and a recorded deterministic base seed. Seeds do not guarantee bit identity across GPU/driver/runtime versions or microbatch changes.

```bash
curl -fsS -X POST "$BASE/v1/jobs" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/short-en.json > job.local.json
```

The response contains `job_id`, **`job_token`**, `status_url` and `poll_after_seconds`. Keep the token secret. Every job-specific route requires:

```text
Authorization: Bearer <job_token>
```

A job ID alone never grants access. Missing/wrong tokens both return 404. The browser fetches the authenticated MP3 as a Blob before attaching it to its player; capabilities are not put in query strings. Use `scripts/client.py` for a complete submit/poll/download client that saves its receipt with mode 0600.

`Idempotency-Key` is optional and limited to 128 characters. It is scoped to the server's HttpOnly `podcast_client` cookie and the ingress rate-limit identity. Reuse a cookie jar/client session when retrying a submission; the same key and same validated JSON return the original receipt, while different JSON returns 409. An unknown submission outcome without the original session should be resolved by checking the saved receipt, not blindly posting the episode again. Never use a human-readable idempotency key as an access credential.

Poll every 15–60 seconds. `Location` and `Retry-After` are also returned on submission. Busy capacity returns 429 with `Retry-After: 60`, not a silently lost job. HTTP requests finish quickly; generation itself runs in a separate process. Network disconnection does not cancel the job.

## Progress and recovery

Job states are `queued`, `running`, `retrying`, `completed`, `failed`, `cancelled`. Detailed `phase` values include `waiting_for_gpu`, `loading_policy`, `generating`, `loading_judges`, `decoding_and_ranking`, `auditing`, `assembling`, `completed` and `failed`.

Progress includes speech-turn count, completed/total candidates, selected turns, current turn, microbatch size, heartbeat, elapsed wall time, process attempts and the latest recoverable error. GPU/compile/model-load phases may have a stationary candidate count: that is not evidence of a hung request. No fabricated ETA is returned before enough evidence exists. The report exposes genuine selection/audit data; it does not fill missing model outputs with zero.

The durable queue uses SQLite WAL. Token files and each selected-turn record are written to temporary paths, then renamed. On restart, running jobs are requeued. Valid completed acoustic-code files are reused; corrupt/incomplete token files are regenerated. Already selected turns with intact files are reused. A final checksum mismatch stops/retries rather than serving a corrupted MP3. The script, model revision, reference hashes and runtime fingerprint must match the stored job. Changing those while resuming raises an explicit error.

For an allocation failure, token microbatch size is halved; N is preserved. Decoder microbatching is independent. At microbatch 1, an OOM exits the worker so the supervisor can release its process/context and retry. Up to three worker process attempts are allowed. A worker with no progress heartbeat for 30 minutes, or a six-hour process runtime, is terminated and retried within that limit. GPU-capacity waiting is visible, retries periodically and does not consume the error budget. An API shutdown stops its worker; abrupt parent death also terminates the worker on Linux.

After persistent failure, inspect `last_error` and the private report/log, fix the cause, then call `/retry`. It resets the bounded process-attempt budget and keeps completed work. It never changes the model, reduces requested N, introduces voice conversion, or switches ranking criteria. Reference/model download errors need operator intervention if the pinned assets are missing. Cancellations are explicit and do not become automatic retries.

## Result and review

The audio endpoint returns `audio/mpeg`, mono, 48 kHz, constant 128,000 bit/s MP3. The complete file is encoded to a temporary path, probed for codec/channel/bitrate/duration, fully decoded once to detect corruption, then atomically published. Encoding includes moderate loudness normalization to -18 LUFS / -1.5 dBTP. The selected source clips and exact pause timeline remain in the private report/data directory.

All N scores are retained for selection transparency. N=1 has no fabricated ranking score. ASR WER > 0.15, a selected take reaching the generation ceiling, or uncertain alignment creates a review flag. The selected take is retained; flags are not secretly used as a second ranking gate. If a person later replaces a flagged take, that is a separate editorial decision.

The job token protects access, not encryption at rest. Default expiration is 72 hours after the terminal state; cleanup runs in the supervisor. This server has no persistent public audio listing. External source references, scripts, episode rights and any personal information are the operator's responsibility.
