"""Script-anchored editing after reward selection; never re-rank or invent words.

Independent ASR anchors, rather than whole-clip WER, authorize individual edits.
Internal gaps become silence of the same length: the performer's pacing and the
assembler's explicit pause clock do not change. No model is loaded here.
"""
import difflib
import math
import re
import time
import unicodedata

import numpy as np

VERSION = 'script-anchors-v1'
POLICY = dict(version=VERSION, lead_guard_s=.10, tail_guard_s=.10,
              fade_s=.10, min_internal_gap_s=.65, minimum_anchor_coverage=.65,
              timestamp_disagreement_s=.32, internal_gap_mode='silence_preserve_duration')


def tokens(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def timed_transcription(model, processor, wave16, device):
    """One existing Parakeet TDT pass; accumulate its real duration predictions.

    DecodeStream preserves SentencePiece word boundaries across tokens. Timestamp
    durations are model estimates at 80-ms resolution, not alignment confidence.
    """
    import torch
    from tokenizers.decoders import DecodeStream
    tick = time.monotonic()
    with torch.inference_mode():
        inp = processor([wave16], sampling_rate=16000, return_tensors='pt', padding=True)
        inp = {k: v.to(device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
               for k, v in inp.items()}
        out = model.generate(**inp, max_new_tokens=2048)
    text = re.sub(r'<[^>]+>', '', processor.batch_decode(out.sequences)[0]).strip()
    ids = out.sequences[0].cpu().tolist()
    durations = out.durations[0].cpu().tolist()
    frame_s = processor.feature_extractor.hop_length / processor.feature_extractor.sampling_rate * 8
    skip = {processor.tokenizer.pad_token_id, processor.blank_token_id}
    stream = DecodeStream(skip_special_tokens=True)
    pieces = []; joined = ''; frame = 0; limit = len(wave16)/16000
    for token, duration in zip(ids, durations):
        start = frame*frame_s; frame += duration
        if token in skip:
            continue
        chunk = stream.step(processor.tokenizer._tokenizer, int(token))
        if chunk is None:
            continue
        end = min(limit, start+max(1, duration)*frame_s)
        pieces.append((len(joined), len(joined)+len(chunk), max(0., start), end))
        joined += chunk
    words = []
    for match in re.finditer(r'\w+', joined):
        spans = [p for p in pieces if p[0] < match.end() and p[1] > match.start()]
        if spans:
            words.append(dict(word=match.group(), start=min(p[2] for p in spans), end=max(p[3] for p in spans)))
    valid = tokens(joined) == tokens(text) and all(0 <= w['start'] < w['end'] <= limit for w in words)
    return dict(text=text, words=words if valid else [], seconds=time.monotonic()-tick,
                timestamp_status='tdt_estimates' if valid else 'token_timestamp_mismatch', frame_seconds=frame_s)


def _words(rows, duration):
    """Reject malformed/non-monotonic spans; never fix them into fake evidence."""
    result = []; previous = -1.
    for row in rows or []:
        if isinstance(row, dict):
            word, a, b = row['word'], row['start'], row['end']
        else:
            word, a, b, score = row
            if score <= .5:
                return []
        if not all(math.isfinite(float(v)) for v in (a, b)) or not 0 <= a < b <= duration+.03 or a < previous:
            return []
        previous = a
        for token in tokens(word):
            result.append(dict(word=token, start=float(a), end=min(duration, float(b))))
    return result


def cleanup(wave, text, asr_words, alignment=None, *, sr=48000, direction=''):
    """Return edited PCM and a fully recorded, conservative editing decision."""
    source = np.asarray(wave, dtype=np.float32).reshape(-1)
    duration = len(source)/sr
    if not len(source) or not np.isfinite(source).all():
        raise ValueError('Cannot clean invalid audio')
    ref = tokens(text); recognized = _words(asr_words, duration)
    forced = _words(alignment, duration)
    if [w['word'] for w in forced] != ref:
        forced = []
    blocks = difflib.SequenceMatcher(None, ref, [w['word'] for w in recognized], autojunk=False).get_matching_blocks()
    mapping = {}; block_size = {}
    for block in blocks:
        for n in range(block.size):
            mapping[block.a+n] = block.b+n; block_size[block.a+n] = block.size
    coverage = len(mapping)/max(1, len(ref)); required = min(3, len(ref))
    nonverbal = bool(re.search(r'\b(laugh\w*|chuckl\w*|giggl\w*|sigh\w*|sobb?\w*|crying|gasp\w*|humming|singing|vocal burst)\b', direction, re.I))
    warnings = []; differences = []

    def anchor(i):
        if i not in mapping or block_size.get(i, 0) < required or coverage < POLICY['minimum_anchor_coverage'] or nonverbal:
            return None
        word = recognized[mapping[i]]
        a, b = word['start'], word['end']
        if forced:
            other = forced[i]
            if max(abs(other['start']-a), abs(other['end']-b)) <= POLICY['timestamp_disagreement_s']:
                # Preserve the union when both independent estimates agree.
                a, b = min(a, other['start']), max(b, other['end'])
            else:
                differences.append(i)
        return dict(start=a, end=b)

    anchors = {i: anchor(i) for i in range(len(ref))}
    first, last = anchors.get(0), anchors.get(len(ref)-1)
    lead_guard, tail_guard, fade = POLICY['lead_guard_s'], POLICY['tail_guard_s'], POLICY['fade_s']
    start = max(0., first['start']-lead_guard-fade) if first else 0.
    end = min(duration, last['end']+tail_guard+fade) if last else duration
    if start >= end:
        start, end = 0., duration; first = last = None
        warnings.append('Inconsistent edge anchors: retained clip edges')
    output = source.copy(); gap_edits = []
    # Pauses are preserved in time; only their unrequested acoustic content is removed.
    # Do not bridge a missing script word, or words sharing one token timestamp.
    for i in range(max(0, len(ref)-1)):
        left, right = anchors[i], anchors[i+1]
        if not left or not right or right['start']-left['end'] < POLICY['min_internal_gap_s']:
            continue
        if i in differences or i+1 in differences:
            continue  # Conflicting aligners cannot authorize an internal deletion.
        a = left['end']+tail_guard; b = right['start']-lead_guard
        if b-a < 2*fade+.05:
            continue
        lo, hi = round(a*sr), round(b*sr); ramp = min(round(fade*sr), (hi-lo)//2)
        output[lo:lo+ramp] *= np.linspace(1, 0, ramp, dtype=np.float32)
        output[lo+ramp:hi-ramp] = 0
        output[hi-ramp:hi] *= np.linspace(0, 1, ramp, dtype=np.float32)
        gap_edits.append(dict(after_target_word=i, start_s=a, end_s=b,
                              silent_start_s=(lo+ramp)/sr, silent_end_s=(hi-ramp)/sr,
                              fade_s=ramp/sr, duration_preserved=True))
    lo, hi = round(start*sr), round(end*sr)
    clip = output[lo:hi].copy(); actual_start = lo/sr; actual_end = hi/sr
    fade_in = min(fade, max(0., first['start']-actual_start)) if first else 0.
    fade_out = min(fade, max(0., actual_end-last['end'])) if last else 0.
    for seconds, reverse in ((fade_in, False), (fade_out, True)):
        count = min(len(clip), round(seconds*sr))
        if count > 1:
            if reverse:clip[-count:] *= np.linspace(1, 0, count, dtype=np.float32)
            else:clip[:count] *= np.linspace(0, 1, count, dtype=np.float32)
    if not first or not last:
        warnings.append('Unverified script edge(s): retained uncertain audio; review required')
    if nonverbal:
        warnings.append('Explicit nonverbal performance cue: automatic cleanup skipped')
    if differences:
        warnings.append('ASR/forced alignment disagreement: internal edits at those words skipped')
    edit = dict(policy=POLICY, start_s=actual_start, end_s=actual_end,
                removed_head_s=actual_start, removed_tail_s=duration-actual_end,
                first_word_in_clip_s=max(0., first['start']-actual_start) if first else 0.,
                last_word_in_clip_s=min(len(clip)/sr, last['end']-actual_start) if last else len(clip)/sr,
                timing_basis='script_matched_asr_with_alignment_audit' if first and last else 'partial_script_anchors' if first or last else 'unverified_clip_edges',
                first_word_verified=bool(first), last_word_verified=bool(last),
                matched_target_fraction=coverage, matched_target_words=len(mapping),
                target_words=len(ref), fade_in_s=fade_in, fade_out_s=fade_out,
                gap_edits=gap_edits, muted_gap_seconds=sum(g['silent_end_s']-g['silent_start_s'] for g in gap_edits),
                conflicting_word_indices=sorted(set(differences)), warnings=warnings)
    return clip, edit
