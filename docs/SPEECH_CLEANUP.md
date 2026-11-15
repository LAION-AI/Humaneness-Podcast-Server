# Script-anchored cleanup of podcast speech

New jobs apply this post-selection editor automatically. It removes unrequested
audio before/after supported script boundaries and silences long, verified gaps
inside a turn. It preserves the selected candidate, ranking scores, script,
temperature, CFG and voice references. No quality-based re-selection is added.

## Why an existing trim could leave mumbling

The original worker required whole-clip WER ≤ 15% before making **any** trim.
Extra words at the end could exceed that threshold, which retained the entire
untrimmed tail. It also preserved all internal audio, even in long gaps between
expected words. In 1,024 saved selected-turn records inspected on November 14,
181 used `unverified_clip_edges`; this is a diagnostic count, not a listening
assessment or a claim that all 181 contain unwanted sounds.

## Evidence and editing policy

1. The existing Parakeet TDT v3 audit runs once on the selected waveform. In
   addition to its transcript, the editor decodes the real token durations into
   word spans. Timestamp resolution is approximately 80 ms. DecodeStream retains
   token whitespace; disagreement between reconstructed tokens and the transcript
   invalidates the timestamps rather than silently fabricating positions.
2. Recognized words are matched monotonically to normalized script words. Each
   usable anchor needs a run of at least three consecutive matching words (or all
   words for a shorter script) and at least 65% target-word coverage overall.
   Whole-clip WER remains a diagnostic, not an editing or ranking eligibility
   gate. First and last edges are evaluated separately.
3. Qwen forced alignment remains an independent timing audit for supported
   languages. When estimates agree within 320 ms, the editor protects their
   union. A forced alignment alone cannot authorize deletion: it can place words
   that were never actually spoken. Disagreement disables internal edits at the
   affected anchors and is recorded. Strong ASR anchors can still trim outer
   edges. Dutch uses recognized timing because this Qwen aligner does not support
   Dutch.
4. A supported first/last word keeps a 100-ms outside guard plus up to 100 ms of
   fade. Everything farther outside the retained clip is removed. The fade is
   outside the protected word, not over its phonemes.
5. A gap of at least 650 ms between **consecutive script words**, each supported
   by its matching run, may be cleaned. The editor preserves 100-ms word guards,
   fades down/up over 100 ms and sets only the central gap to silence. It does
   not shorten the pause. A missing or unrecognized scripted word cannot be
   bridged by this rule; contradictory timing also prevents that edit.
6. Explicit nonverbal delivery cues such as laughter, a chuckle, sobbing or a
   sigh skip cleanup for that turn. These sounds may be the intended performance.
   Uncertain edges remain unchanged and flagged, rather than cutting by volume
   alone or assuming quiet speech is unwanted.

The assembler then inserts the requested word-to-word pauses and exports the
existing mono 48-kHz, 128-kbit/s MP3. Minimum gaps can exceed the requested value
when retained safety guards already occupy more time. The report records that.

## Runtime and storage

`podcast_server/speech_cleanup.py` implements the editor without changing the
pinned vendored inference/scoring code. The worker reuses the loaded Parakeet and
Qwen models. It does not introduce another model, per-candidate transcription,
or an additional full ASR pass in production. Local PCM editing is linear in the
short selected turn. Existing two-phase model loading and GPU assignment remain.

The model/provenance report records the exact cleanup policy and source hashes.
Every turn records cut ranges, mute ranges, guard/fade lengths, coverage and
warnings. Job metadata reports aggregate cuts/mutes. Selected original acoustic
codes remain available privately for reproduction. Completed old jobs are not
silently rewritten, and a resumed job cannot mix old/new policy fingerprints.

## Reproduction and interpretation

```bash
python -m pytest -q tests
python scripts/client.py tests/fixtures/cleanup-de.json \
  --receipt cleanup.local.json --output cleanup.mp3

# Optional local operator audit; never publish this private output directory.
python scripts/audit_cleanup.py --data /data/podcast-jobs \
  --output /data/private-cleanup-audit --device cuda:0 --limit 8
```

The signal tests verify fades, exact preservation of anchored word samples,
unchanged internal pause lengths, retained uncertain boundaries, missing negation
protection and explicit nonverbal cues. They are not listening tests. The replay
script uses actual saved TTS audio and actual Parakeet/Qwen inference, with a
separate after-edit ASR pass **for evaluation only**. Its report includes private
script text and audio, so it must stay outside Git/public static directories.

ASR and forced alignment are estimates. An unrecognized ending can still retain
an unwanted tail; the system explicitly flags that case. Short inter-word sounds
and requested vocalizations can remain. No promise of perfect word retention or
complete removal of every artifact is made. See [test results](TESTING.md).
