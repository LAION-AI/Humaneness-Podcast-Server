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

Speaker cosine and genuineness are model estimates. Their combination can select an incomplete or linguistically imperfect take. ASR can also disagree with correct speech. Alignment timestamps are estimates. The November 14 editor assesses each edge/gap independently and retains uncertain regions; earlier validation used all-or-nothing clip trimming. The final report shows these distinctions and does not replace missing scores with invented values. Natural within-turn pauses, exact emotional intensity and language fidelity remain properties of the checkpoint rather than guarantees of this server.

The release tests cover English/German and N=1/2/4, with an additional N=8 capacity check recorded in the measurements. Dutch/French/Spanish routing and source recipes are included, but this server's new live integration did not regenerate a complete episode in those three languages. N=16 is accepted and is microbatched; it has not received a separate end-to-end quality study here.

## November 14: script-anchored cleanup regression

The new default editor and reproduction commands are in
[SPEECH_CLEANUP.md](SPEECH_CLEANUP.md). All **28 CPU/API tests** passed, including
new signal tests for word-sample preservation, fades, internal-gap timing,
missing negations, conflicting timestamps and explicit nonverbal delivery cues.
These signal fixtures do not measure speech quality.

A local replay selected eight previously untrimmed German TTS clips with WER
between 15% and 80%, at most two per episode. Actual Parakeet v3 and Qwen inference
supported tail cuts in **7/8**, removing **56.92 seconds** of trailing audio. One
uncertain ending remained untouched and flagged. Mean ASR WER changed from
**20.58% to 4.34%**; no clip's measured WER increased. This is a deliberately
selected regression cohort, not a random population estimate. The same ASR
family both authorizes edits and evaluates the result, so this is not independent
human listening evidence. The edited tails can include both silence and unwanted
vocalizations; the removed duration is not a measured duration of mumbling alone.

Median ASR + alignment + cleanup time was **1.26 s per clip**, including the
existing audit models. Production uses one Parakeet pass and one supported-language
alignment pass; the extra after-edit transcription is used only by the audit
script. None of these eight clips exercised an internal gap edit; that part was
verified by signal fixtures. Private before/after clips and detailed scripts stay
in the operator audit folder. The checked-in
[aggregate replay receipt](validation/speech-cleanup-replay.json) contains no
private episode text, audio or job capabilities.

A fresh public HTTPS Best-of-4 request generated eight candidates for the two-host
[German fixture](../tests/fixtures/cleanup-de.json), keeping both strict reward
winners. The resulting **15.26-second** episode completed in **175.5 s** including
cold worker/model setup (worker 169.54 s). **19.84 s** of trailing audio was removed.
Both edges of both turns were verified; the explicit **900-ms** gap was preserved
to sample rounding. CPU editing took **3.64 / 3.75 ms** for the two turns. This
short cold job is not a steady-state throughput benchmark.

One selected original clip reached the generation ceiling and had 110% diagnostic
WER due to extra recognized output. Those original review flags remain visible;
they did not change the winner. A separate final evaluation of the edited clips
through the host's existing Parakeet Ultra/Photon service recognized all **21 / 20
script words with zero edit errors**. It is a different checkpoint in the same
ASR family, not independent human listening or a new production dependency.
[Fresh-job measurements](validation/speech-cleanup-fresh-job.json).

Actual HTTPS Chromium playback advanced, the authenticated JSON report downloaded,
390-pixel mobile layout had no overflow, and there were no page errors.
[Browser receipt](validation/speech-cleanup-browser.json). Reproduce against your
own completed job with `python tests/browser_cleanup.py --receipt cleanup.local.json
--result browser-result.json --base https://YOUR-HOST/podcast/`; install Playwright
in the test environment. Receipts remain private. The browser used a muted audio
sink, so this is not a physical speaker/headphone listening test.

## Default Voice Converter + SIDON integration (16 November 2026)

On an A100 80 GB / CUDA 12.4 host, a fresh public-API job submitted without `options` used Best-of-4, per-turn Humaneness Voice Converter and full-episode SIDON. Both selected turns were converted, the SIDON report identified the pinned model, and the authenticated MP3 download succeeded. End-to-end elapsed time was 97.1 seconds, with SIDON reporting 3.555 seconds. The very short first utterance triggered an ASR disagreement and unverified-edge warning; the TTS selected clip was 23.36 seconds long. This is a real quality limitation, not evidence that short text always produces a short or correct take. A separate 0.6-second source clip tested the converter's trailing-silence padding to one second successfully. These checks are not a listening evaluation.

For a separate ten-turn German webcoding episode, SIDON's full-episode GPU step took 3.164 seconds including model loading, excluding MP3 encoding. On automatic per-turn measurements, Parakeet WER versus the 162-word script was 6/162 both before and after SIDON; mean ECAPA cosine to the respective references was 0.790 before and 0.769 after; mean Ears genuineness was 4.013/6 and 4.058/6 respectively. These are model estimates for one example, not a demonstration that SIDON improves subjective quality or identity. A 137.76-second duplicate-audio test exercised the two-chunk path and preserved duration; it was not a speech-quality test.
