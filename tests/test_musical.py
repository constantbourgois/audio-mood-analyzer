import numpy as np
import pytest

from musical import Musical, analyse_musical, estimate_tempo, onset_strength

SR = 22050


def _drum_loop(bpm, seconds=20.0, sr=SR):
    """Kick on every beat, a quieter hat on the off-beats."""
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    beat = 60.0 / bpm
    rng = np.random.default_rng(0)
    kick_t = np.arange(int(0.12 * sr)) / sr
    kick = (np.sin(2 * np.pi * 60 * kick_t) * np.exp(-kick_t / 0.04)).astype(np.float32)
    hat = (rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr))).astype(np.float32)
    for k, t in enumerate(np.arange(0, seconds - 0.2, beat / 2)):
        i = int(t * sr)
        sound = kick if k % 2 == 0 else 0.3 * hat
        y[i:i + len(sound)] += sound
    return y


def _chords(roots_and_qualities, seconds_each=2.0, sr=SR):
    """Sustained triads with a few harmonics, one after another."""
    out = []
    t = np.arange(int(seconds_each * sr)) / sr
    for root_midi, minor in roots_and_qualities:
        notes = [root_midi, root_midi + (3 if minor else 4), root_midi + 7]
        chord = np.zeros_like(t)
        for n in notes:
            f = 440.0 * 2 ** ((n - 69) / 12)
            for h, a in ((1, 1.0), (2, 0.4), (3, 0.2)):
                chord += a * np.sin(2 * np.pi * f * h * t)
        out.append(chord * np.minimum(1.0, t / 0.05))
    y = np.concatenate(out)
    return (0.2 * y / np.abs(y).max()).astype(np.float32)


@pytest.mark.parametrize("bpm", [90, 120, 136])
def test_tempo_of_a_drum_loop(bpm):
    result = analyse_musical(_drum_loop(bpm), SR)
    assert result.tempo_bpm is not None
    # Half/double-time is the known ambiguity; within that, the number must be right.
    candidates = [result.tempo_bpm] + [result.tempo_bpm / 2, result.tempo_bpm * 2]
    assert min(abs(c - bpm) / bpm for c in candidates) < 0.015


def test_tempo_does_not_depend_on_level_or_sample_rate():
    y = _drum_loop(120)
    base = analyse_musical(y, SR).tempo_bpm
    assert analyse_musical(0.02 * y, SR).tempo_bpm == pytest.approx(base, abs=0.5)
    assert analyse_musical(np.repeat(y, 2), 2 * SR).tempo_bpm == pytest.approx(base, rel=0.01)


def test_no_tempo_for_a_held_chord_or_noise():
    # A held chord still autocorrelates at some lag (its partials beat against
    # the hop) and noise is all onsets; neither has a beat.
    pad = _chords([(60, False)], seconds_each=20.0)
    noise = (0.1 * np.random.default_rng(1).standard_normal(20 * SR)).astype(np.float32)
    assert analyse_musical(pad, SR).tempo_bpm is None
    assert analyse_musical(noise, SR).tempo_bpm is None


def test_too_short_for_a_tempo():
    assert analyse_musical(_drum_loop(120, seconds=3.0), SR).tempo_bpm is None


def test_key_of_a_major_progression():
    # I - IV - V - I in D major
    y = _chords([(62, False), (67, False), (69, False), (62, False)] * 2)
    result = analyse_musical(y, SR)
    assert result.key == "D major"
    assert result.modes and result.modes[0].tonic == "D"


def test_key_of_a_minor_progression():
    # i - iv - v - i in A minor
    y = _chords([(57, True), (62, True), (64, True), (57, True)] * 2)
    assert analyse_musical(y, SR).key == "A minor"


def test_key_is_the_same_at_another_sample_rate():
    y = _chords([(62, False), (67, False), (69, False), (62, False)] * 2)
    assert analyse_musical(np.repeat(y, 2), 2 * SR).key == "D major"


def test_no_key_claimed_for_noise():
    noise = (0.1 * np.random.default_rng(1).standard_normal(20 * SR)).astype(np.float32)
    result = analyse_musical(noise, SR)
    assert result.key is None
    assert result.modes == []


def test_silence_and_empty_input():
    assert analyse_musical(np.zeros(10 * SR, dtype=np.float32), SR) == Musical()
    assert analyse_musical(np.zeros(0, dtype=np.float32), SR) == Musical()


def test_onset_envelope_has_no_seam_between_chunks():
    # Steady loop long enough to span several 512-frame chunks: a missing
    # carry-over frame would put a spike at each chunk boundary.
    y = _drum_loop(120, seconds=40.0)
    env, hop_sec = onset_strength(y, SR)
    assert len(env) > 3 * 512
    assert env[1:].max() <= 1.5 * np.percentile(env, 99.5)


def test_flat_onset_curve_has_no_tempo():
    assert estimate_tempo(np.zeros(2000, dtype=np.float32), 0.023) == (None, 0.0)


def test_tempo_alternatives_stay_plausible():
    assert Musical(tempo_bpm=136.0).tempo_alternatives == [68]
    assert Musical(tempo_bpm=118.0).tempo_alternatives == [59]
    assert Musical(tempo_bpm=90.0).tempo_alternatives == [180]
    assert Musical().tempo_alternatives == []
