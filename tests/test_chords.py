import numpy as np
import pytest

from chords import ChordSegment, estimate_chords, summarize_chords

SR = 22050


def _triad(freqs, duration, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return (0.3 * tone / len(freqs)).astype(np.float32)


@pytest.fixture
def two_chord_audio():
    """Do majeur (Do-Mi-Sol) puis La mineur (La-Do-Mi), 2 s chacun."""
    c_major = _triad([261.63, 329.63, 392.00], 2.0)
    a_minor = _triad([220.00, 261.63, 329.63], 2.0)
    return np.concatenate([c_major, a_minor]), SR


def test_estimate_chords_recognizes_major_and_minor_triads(two_chord_audio):
    y, sr = two_chord_audio
    beat_times = np.arange(0, 4.0, 0.5)  # grille régulière, indépendante du suivi de tempo
    segments = estimate_chords(y, sr, beat_times, measure_times=[0.0, 2.0])

    labels = [s.label for s in segments]
    assert "Do" in labels, f"accord majeur non reconnu, obtenu : {labels}"
    assert "Lam" in labels, f"accord mineur non reconnu, obtenu : {labels}"
    # l'ordre doit suivre le signal : Do avant Lam
    assert labels.index("Do") < labels.index("Lam")


def test_estimate_chords_merges_consecutive_identical_segments(two_chord_audio):
    y, sr = two_chord_audio
    beat_times = np.arange(0, 4.0, 0.5)
    segments = estimate_chords(y, sr, beat_times, measure_times=[0.0, 2.0])

    # Deux accords tenus sur 8 intervalles de temps : la fusion doit produire
    # un segment par accord, pas un par temps. Un seuil lâche ici laisserait
    # passer une fusion totalement inopérante.
    assert len(segments) <= 3, f"segments non fusionnés : {[s.label for s in segments]}"
    for previous, current in zip(segments, segments[1:]):
        assert previous.label != current.label or previous.end_sec < current.start_sec


def test_estimate_chords_needs_at_least_two_beats():
    y = _triad([261.63, 329.63, 392.00], 1.0)
    assert estimate_chords(y, SR, [], measure_times=[]) == []
    assert estimate_chords(y, SR, [0.5], measure_times=[]) == []


def test_estimate_chords_on_silence_returns_nothing():
    y = np.zeros(SR * 2, dtype=np.float32)
    segments = estimate_chords(y, SR, np.arange(0, 2.0, 0.5), measure_times=[0.0])
    assert segments == []


def test_segments_carry_measure_and_ordered_times(two_chord_audio):
    y, sr = two_chord_audio
    segments = estimate_chords(y, sr, np.arange(0, 4.0, 0.5), measure_times=[0.0, 2.0])
    assert segments
    for segment in segments:
        assert segment.end_sec > segment.start_sec
        assert segment.duration_sec > 0
        assert segment.measure is None or segment.measure >= 1


def test_summarize_chords_ranks_by_total_duration():
    segments = [
        ChordSegment(label="Do", start_sec=0.0, end_sec=1.0, measure=1),
        ChordSegment(label="Sol", start_sec=1.0, end_sec=5.0, measure=1),
        ChordSegment(label="Do", start_sec=5.0, end_sec=7.0, measure=2),
    ]
    summary = summarize_chords(segments)
    # Sol : 4 s d'un bloc ; Do : 1 s + 2 s = 3 s cumulées.
    assert [item["label"] for item in summary] == ["Sol", "Do"]
    assert summary[0]["seconds"] == 4.0
    assert summary[1]["seconds"] == 3.0
