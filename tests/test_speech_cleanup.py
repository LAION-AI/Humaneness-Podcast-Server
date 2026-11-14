"""Authored signals check editing invariants, not perceptual TTS quality."""
import numpy as np
from podcast_server.speech_cleanup import cleanup

SR=48000
TEXT='Please keep these words and finish this sentence.'


def words(text=TEXT, start=.8):
    return [dict(word=w.strip('.'),start=start+i*.35,end=start+i*.35+.2) for i,w in enumerate(text.split())]


def wave(seconds=6):
    return (.1*np.sin(np.arange(int(seconds*SR))*2*np.pi*171/SR)).astype(np.float32)


def test_extra_tail_words_do_not_disable_edge_cleanup():
    spans=words();spans.extend(words('unexpected unrelated additional words',start=4.))
    source=wave();clip,edit=cleanup(source,TEXT,spans)
    assert edit['first_word_verified'] and edit['last_word_verified']
    assert edit['removed_tail_s']>2
    assert edit['fade_out_s'] == .1
    assert clip[-1]==0 and clip[0]==0
    for w in words():
        a,b=round(w['start']*SR),round(w['end']*SR);offset=round(edit['start_s']*SR)
        np.testing.assert_array_equal(clip[a-offset:b-offset],source[a:b])


def test_internal_mumble_is_muted_without_shortening_pause_or_words():
    spans=words()
    for w in spans[4:]:w['start']+=1.;w['end']+=1.
    source=wave(7);clip,edit=cleanup(source,TEXT,spans)
    assert len(edit['gap_edits'])==1
    g=edit['gap_edits'][0];offset=round(edit['start_s']*SR)
    assert np.max(np.abs(clip[round(g['silent_start_s']*SR)-offset:round(g['silent_end_s']*SR)-offset]))==0
    assert g['duration_preserved']
    for w in spans:
        a,b=round(w['start']*SR),round(w['end']*SR)
        np.testing.assert_array_equal(clip[a-offset:b-offset],source[a:b])
    assert len(clip)==round(edit['end_s']*SR)-offset


def test_missing_negation_cannot_authorize_an_internal_cut():
    text='Please do not open the door today.'
    spans=words(text);spans.pop(2)
    source=wave();clip,edit=cleanup(source,text,spans)
    assert not edit['gap_edits']
    assert not edit['first_word_verified']  # No three-word opening anchor.


def test_missing_last_word_is_not_claimed_complete():
    source=wave();clip,edit=cleanup(source,TEXT,words()[:-1])
    assert not edit['last_word_verified'] and edit['removed_tail_s']==0
    assert edit['warnings']


def test_forced_alignment_alone_cannot_authorize_deletion():
    source=wave();alignment=[(w['word'],w['start'],w['end'],1.) for w in words()]
    clip,edit=cleanup(source,TEXT,[],alignment)
    np.testing.assert_array_equal(source,clip)
    assert edit['timing_basis']=='unverified_clip_edges'


def test_conflicting_timestamps_preserve_internal_audio():
    spans=words()
    for w in spans[4:]:w['start']+=1.;w['end']+=1.
    alignment=[(w['word'],w['start'],w['end'],1.) for w in spans]
    w=spans[3];alignment[3]=(w['word'],w['start'],w['end']+.6,1.)
    clip,edit=cleanup(wave(7),TEXT,spans,alignment)
    assert not edit['gap_edits'] and 3 in edit['conflicting_word_indices']


def test_explicit_laugh_is_not_mistaken_for_unrequested_mumble():
    source=wave();clip,edit=cleanup(source,TEXT,words(),direction='warm, with a gentle chuckle')
    np.testing.assert_array_equal(clip,source)
    assert any('nonverbal' in w for w in edit['warnings'])


def test_invalid_timestamps_do_not_cut_and_low_coverage_is_flagged():
    for spans in ([dict(word='Please',start=float('nan'),end=1.)],words('other words unrelated')):
        source=wave();clip,edit=cleanup(source,TEXT,spans)
        np.testing.assert_array_equal(clip,source)
        assert edit['warnings']
