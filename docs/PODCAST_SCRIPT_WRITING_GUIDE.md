# Podcast Script Writing Guide

This guide defines the **version 1.0 JSON episode format** accepted by Humaneness Podcast Server. An API or local language model can write the script. The server needs no LLM or API key: it speaks the supplied words using the pinned [Humanness Voice Podcast checkpoint](https://huggingface.co/laion/Humanness-Voice-Podcast), selects candidates, assembles the episode and returns a downloadable MP3.

The approved recipe is a spontaneous, relaxed, gently spoken conversation. It is a listener preference established on the published samples, not a claim that these settings are optimal for every script, voice or language. [Listen to the five-language examples](https://huggingface.co/spaces/laion/Humanness-Voice-Podcast).

## Start with a complete episode

```json
{
  "version": "1.0",
  "title": "An ordinary day",
  "language": "de",
  "speakers": {
    "host": "reference_s1",
    "guest": "reference_s2"
  },
  "options": {"best_of": 4, "seed": 42, "temperature": 1.0, "cfg": 2.5},
  "turns": [
    {
      "type": "speech",
      "speaker": "host",
      "text": "Was macht einen ganz normalen Tag eigentlich wertvoll?",
      "direction": "warm, gently curious; relaxed, spontaneous, gently spoken",
      "pause_after_ms": 400
    },
    {"type": "pause", "duration_ms": 500},
    {
      "type": "speech",
      "speaker": "guest",
      "text": "Vielleicht beginnt es damit, dass wir füreinander da sind.",
      "direction": "thoughtful, sincere; easy, spontaneous, gently spoken"
    }
  ]
}
```

Get the configured reference IDs from `GET /v1/voices`. Speaker keys such as `host` and `guest` are your stable script labels, not words to read aloud. Each key maps to one fixed reference for the whole episode. The same speaker may have consecutive turns; that is how to split a long monologue or insert an exact pause inside it.

`language` is one of `en`, `de`, `nl`, `fr`, `es`. Use one language per episode. The **spoken text** can be in that language; write **all directions and cues in English**. No automatic translation or fact checking is performed. The parser checks structure, not the actual language of a sentence or cue.

## Say what to speak separately from how to speak

- `text` contains only the exact words to be spoken. Do not include speaker labels, stage directions, timestamps or parentheses. Spell numbers naturally for the target language when pronunciation matters.
- `direction` is a short English acting cue. Parentheses are added by the compiler; do not write them yourself.
- `delivery_chunks` is optional: one or two text/cue pairs. Joining their texts with **one space must reproduce `text` exactly**. Use a second cue at a meaningful sentence boundary, not on every sentence.
- English cue suggestions: `easy, spontaneous, gently spoken`; `relaxed, unforced, like talking to a friend`; `soft natural delivery, thinking aloud`; `casual and sincere, unhurried`.
- Contextual emotions may be `gently curious`, `mildly amused`, `quietly reflective`, `understated wonder` or `tender, sincere`. Strong emotions are for an explicit narrative reason. Do not stack intense emotions onto every sentence.
- Do not redesign the reference voice with age, gender or timbre descriptions. Do not add forced laughter, extreme breathiness, an announcer voice, or “very pleasant/nice to listen to.”

Example of a change inside a turn:

```json
{
  "type": "speech",
  "speaker": "host",
  "text": "What if meaning is something we practise? Maybe it begins with noticing who needs us today.",
  "direction": "warm, gently curious",
  "delivery_chunks": [
    {"text": "What if meaning is something we practise?", "cue": "warm, gently curious; relaxed, spontaneous, gently spoken"},
    {"text": "Maybe it begins with noticing who needs us today.", "cue": "gently, as a spontaneous thought"}
  ]
}
```

## Exact model prompt

The server always compiles the constant GENERAL below and the requested inline cues into `instruction`. It separately supplies the **literal transcript**, the requested language and only the current speaker's reference acoustic codes. `tokens=None` leaves natural timing free. The default temperature is 1.0, CFG 2.5, top-p 0.95 and top-k 50.

```text
GENERAL: A warm, emotionally thoughtful two-host podcast conversation. Very authentic, very genuine. Natural, intimate, spontaneous spoken delivery with clear words, subtle emotion, relaxed conversational pacing and brief meaningful pauses. Be sincere, without an exaggerated announcer voice or forced laughter. Perform only the current speaker turn and speak exactly the SCRIPT. Keep the same speaker identity as the reference audio.
SCRIPT:
(warm, gently curious; relaxed, spontaneous, gently spoken) What if meaning is something we practise?
(gently, as a spontaneous thought) Maybe it begins with noticing who needs us today.
```

The episode may contain one or more speaker labels; each generation is one referenced speaker turn. The proven GENERAL mentions a two-host podcast and is kept unchanged even for an individual host's turn. The service does not use native simultaneous two-speaker generation or overlapping voices. The upstream alternative is documented in the [native multi-speaker guide](https://huggingface.co/laion/Humanness-Voice-Podcast/blob/main/guides/MULTI_SPEAKER.md).

## Length and pauses

Each speech turn has a **hard generation ceiling of 320 frames × 80 ms = 25.6 seconds**. Keep turns comfortably below it, preferably 10–45 words. Validation rejects more than 60 whitespace-separated words or 600 characters per turn. These limits reduce risk but cannot guarantee completion for slow delivery. A reached ceiling is reported as a possible truncation, not silently accepted as perfect speech. Split longer sentences at natural boundaries before submission.

A 20-minute episode is an ordered list of many short turns, not one 20-minute model call. The service accepts up to 600 turn/pause entries, 60,000 characters of speech and an approximate one-hour script budget. Duration is an estimate; generated pacing varies.

`pause_after_ms` defaults to **400 ms** and describes the minimum word-to-word gap before the next spoken turn. An intervening `{"type":"pause","duration_ms":500}` adds 500 ms to that gap, so the example requests 900 ms. Leading and trailing explicit pause entries are retained. The final speech turn's default gap is not appended to the end of the episode. A pause may be 0–10,000 ms; use several pause entries for a longer deliberate silence.

The assembler keeps short guards outside aligned speech and applies fades only outside those boundaries. It never cuts a word to force a shorter gap. If the requested gap is shorter than retained lead/tail guards, the actual gap is larger and is recorded in the timeline. If word boundaries are uncertain, it retains the entire clip, reports `unverified_clip_edges`, and the gap is based on clip edges. Natural pauses *inside* a turn are model controlled; for an exact scripted pause, split it into two speech turns with the same speaker.

## Best-of-N and quality

`options.best_of` accepts 1, 2, 4, 8 or 16; omission means **4**. All N candidates are complete independent takes of a turn. The worker may compute them in smaller GPU microbatches; this never reduces N. Sampling seeds and execution shapes are recorded. Changing batch shape can change sampled audio even with the same base seed.

For N > 1, the winner maximizes:

```text
0.5 × clip(Humaneness Ears native genuineness / 6, 0, 1)
+ 0.5 × clip(ECAPA cosine(candidate, reference), 0, 1)
```

This is fixed normalization, not per-candidate min/max. No Content Enjoyment term, WER multiplier, voice conversion, DPO or added LoRA is used. For N=1 the sole take is retained without ranking. Parakeet and forced alignment are diagnostics after selection. Uncertain transcript/alignment results remain visible as review flags; there is no hidden re-selection or “quality retry” that changes N or the reward. Runtime errors (OOM, invalid/empty output, interrupted process) trigger bounded retries and are logged separately.

## Ask an LLM to write the script

Use [SCRIPT_WRITER_SYSTEM_PROMPT.txt](SCRIPT_WRITER_SYSTEM_PROMPT.txt) with your preferred local model or API. Supply the topic, audience, language, reference voice IDs, host roles, approximate duration and any factual sources. Ask for JSON only, then submit it to `/v1/validate` before starting a GPU job. An agent can read `/openapi.json` for the exact schema.

Writing directions for an entire episode manually is tedious. An LLM is useful for drafting natural dialogue and sparse cues; it does not need to run on the inference server. Review facts, quotations and voice permissions before publication.
