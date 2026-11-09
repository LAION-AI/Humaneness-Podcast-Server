# Reproducible tests and measured limits

Validation date: **9 November 2026**. Linux / Python 3.12.13, PyTorch 2.6.0+cu124, Transformers 5.15.0, two RTX 3090 24 GB cards on a shared host. The podcast worker used **GPU 1 only**, while the original Robby, TTS, ASR and other research services remained running. At the first probe, GPU 1 had about 11.6 GiB free. Allocator cap: 0.44 of the card, candidate microbatch up to four. These are shared-host measurements, not isolated benchmarks.

## CPU and API tests

```bash
python -m pytest -q tests
```

The suite covers schema defaults, all five language routes, literal-text/cue separation, exact chunk reconstruction, unsupported N values, non-finite settings, turn-size limits, missing voices, unknown request fields, private job access, premature downloads, queue admission, cancellation/deletion, idempotency conflicts, durable restart requeue, preservation of completed checkpoints and bounded invalid-output retries. An authored signal fixture checks word-clock pauses, full MP3 decoding, exactly one channel and 128,000 bit/s encoding. These fixture tests are not presented as TTS model quality results.

```bash
python -m tests.long_assembly
```

This optional stress test creates 600 short authored tone fixtures and assembles more than 20 minutes of output with deliberate gaps. It records actual duration, encoding time and peak RSS in `/tmp/podcast-long-assembly-result.json`, then deletes temporary audio. It checks bounded-memory assembly and container/duration integrity; it does not establish 20-minute speech coherence or run the TTS model 600 times. [Measured result](validation/long-assembly.json).

## Real GPU generation

Use the published example scripts through the real API, not a fake engine:

```bash
python scripts/client.py examples/short-en.json --receipt en.local.json --output en.mp3
python scripts/client.py examples/short-de.json --receipt de.local.json --output de.mp3
```

Completed checks included:

- An English two-speaker Best-of-4 job: eight actual candidates, genuineness/reference selection, a scripted 900 ms word-to-word gap, alignment and verified MP3 export. The result was 7.93 seconds; no transcript/alignment review flags on the selected turns.
- A German six-turn Best-of-2 job: twelve actual candidates and a 48.96-second output. Two turns were flagged; one had possible truncation. The original strict winners remained in the output. This explicitly demonstrates that infrastructure success is not a guarantee of model/content correctness.
- A ten-turn English Single Take job interrupted by a **SIGKILL of only its own worker**, after all ten acoustic-code files had committed. The supervisor automatically restarted the job, reused those code files byte-for-byte and completed on process attempt two. No other GPU process was stopped. [Recovery evidence](validation/worker-recovery.json).
- A fresh English Best-of-4 job using the longer published approved prompts and the decoder graph optimization. The actual generated-code graph check was bit-identical to eager decoding at batch four. Its second turn's decode stage took about 1.56 seconds; the first included the 5.99-second validation/capture/decode cost. These different clips and stages are not a controlled quality or speed ablation against the initial smoke test. Review flags remained visible.

[Machine-readable GPU job measurements](validation/gpu-jobs.json) distinguish worker time, queue time, model loading, decoding, ranking, audit, output duration and review flags. Short-job latency includes substantial imports/loading/compilation and is not an estimate of steady-state minutes per episode. A full new 20-minute *speech-generation* run was not conducted for this release. Do not infer that a 20-minute Best-of-4 episode will always finish in 20 minutes.

## Reproduce worker recovery

The local operator can run this integration test while the service is up:

```bash
python -m tests.gpu_recovery \
  --base http://127.0.0.1:8816 \
  --data /path/to/podcast-jobs
```

It submits its own real ten-turn Single Take job, finds only that job's owned worker under the API's task children, kills that worker once after a candidate checkpoint exists, waits for automatic completion, and verifies the saved hashes. Never substitute an arbitrary GPU PID. The test has a 30-minute timeout and needs queue capacity. It generates a real episode, so it costs inference time.

A process crash and CPU queue restart were actually tested. The microbatch-reduction and invalid-output paths are implemented, but no claim is made that every CUDA failure/driver crash or physical power loss has been reproduced. Persistent bad assets, insufficient disk, incompatible packages or repeated OOM can exhaust the bounded retry budget and require operator action. The original service's usable GPU capacity was sufficient for the tested batch sizes.

## Public browser and HTTPS

With Node, Playwright and Chrome installed:

```bash
PODCAST_TEST_URL=https://YOUR-HOST/podcast/ \
PODCAST_TEST_RECEIPT=/private/path/to/completed-job.local.json \
node tests/browser.cjs
```

Optional variables: `PLAYWRIGHT_MODULE` points to an existing Playwright module directory, `CHROME_BIN` to Chrome, `PODCAST_TEST_RESULT` to a report path. The test validates the live form, restores a real private completed job from browser storage, fetches and plays its actual MP3, downloads the JSON report, and checks 390-pixel mobile layout and JavaScript errors. Headless Chrome uses a muted sink to avoid an unavailable server audio device; playback time must actually advance. It is not a listening test through physical speakers.

The public `/`, `/health`, `/docs`, `/openapi.json`, authenticated status/download and report endpoints were checked over the reused HTTPS tunnel. Existing Robby readiness and its public page were checked separately. [Browser evidence](validation/browser.json).

## Quality and scientific limits

Speaker cosine and genuineness are model estimates. Their combination can select an incomplete or linguistically imperfect take. ASR can also disagree with correct speech. Alignment timestamps are estimates, and uncertain cases retain untrimmed clips. The final report shows these distinctions and does not replace missing scores with invented values. Natural within-turn pauses, exact emotional intensity and language fidelity remain properties of the checkpoint rather than guarantees of this server.

The release tests cover English/German and N=1/2/4, with an additional N=8 capacity check recorded in the measurements. Dutch/French/Spanish routing and source recipes are included, but this server's new live integration did not regenerate a complete episode in those three languages. N=16 is accepted and is microbatched; it has not received a separate end-to-end quality study here.
