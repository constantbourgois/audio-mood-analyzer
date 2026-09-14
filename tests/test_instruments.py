import numpy as np
import pytest

import instruments
from instruments import (
    INSTRUMENT_GROUPS,
    InstrumentScore,
    detect_instruments,
    instruments_between,
    summarize,
)

needs_model = pytest.mark.skipif(
    not instruments.is_available(),
    reason="YAMNet weights not fetched (run python fetch_model.py)",
)


def test_missing_model_degrades_instead_of_raising(monkeypatch):
    """The app must keep working when the weights are absent — that is the
    whole reason the tagger is optional."""
    monkeypatch.setattr(instruments, "_tagger", None)
    monkeypatch.setattr(instruments, "MODEL_PATH", "/nonexistent/yamnet.tflite")
    tagger = instruments.get_tagger()
    assert tagger.available is False

    scores, groups, frame_sec = detect_instruments(np.zeros(32000, dtype=np.float32), 16000)
    assert scores.size == 0 and groups == []
    assert summarize(scores, groups) == []
    assert instruments_between(scores, groups, frame_sec, 0.0, 5.0) == []
    monkeypatch.setattr(instruments, "_tagger", None)


def test_summarize_ranks_by_score_and_applies_the_threshold():
    groups = ["guitar", "piano", "drum kit"]
    scores = np.array([
        [0.40, 0.02, 0.30],
        [0.60, 0.02, 0.20],
    ])
    found = summarize(scores, groups)
    assert [f.name for f in found] == ["guitar", "drum kit"]   # piano is below MIN_SCORE
    assert found[0].score > found[1].score


def test_summarize_respects_the_limit():
    groups = [f"g{i}" for i in range(8)]
    scores = np.full((3, 8), 0.9)
    assert len(summarize(scores, groups, limit=3)) == 3


def test_summarize_on_no_frames_is_empty():
    assert summarize(np.zeros((0, 0)), []) == []
    assert summarize(np.zeros((0, 3)), ["a", "b", "c"]) == []


def test_instruments_between_slices_the_right_window():
    groups = ["guitar", "piano"]
    # First two frames are guitar, last two are piano.
    scores = np.array([[0.9, 0.0], [0.9, 0.0], [0.0, 0.9], [0.0, 0.9]])
    frame_sec = 1.0
    early = instruments_between(scores, groups, frame_sec, 0.0, 2.0)
    late = instruments_between(scores, groups, frame_sec, 2.0, 4.0)
    assert [i.name for i in early] == ["guitar"]
    assert [i.name for i in late] == ["piano"]


def test_instrument_groups_are_non_empty_and_named_readably():
    for name, members in INSTRUMENT_GROUPS.items():
        assert members, f"{name} has no AudioSet labels"
        assert name == name.lower()


@needs_model
def test_tagger_claims_no_instrument_for_a_bare_sine():
    """A naked sine is not an instrument, and the tagger should say so by
    naming nothing — inventing "acoustic guitar" here would be the failure."""
    sr = 16000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    scores, groups, _ = detect_instruments(y, sr)
    assert scores.size and groups, "the model did not run at all"
    assert summarize(scores, groups) == []


@needs_model
def test_tagger_names_a_full_band_synthetic_mix():
    """Full-band material — chord, bass and percussive hits — is recognised,
    and named as something electronic rather than an acoustic instrument.

    A bare chord is deliberately NOT enough: the model only commits above its
    confidence threshold, which is the behaviour we want on real uploads.
    """
    sr = 16000
    rng = np.random.default_rng(1)
    t = np.linspace(0, 8.0, sr * 8, endpoint=False)
    mix = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.00)) / 3
    mix = mix * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t) ** 2)
    mix = mix + 0.8 * np.sin(2 * np.pi * 130.81 * t)
    hits = (np.sin(2 * np.pi * 2.0 * t) > 0.95).astype(float)
    mix = mix + 0.6 * rng.normal(size=t.shape) * hits
    y = (0.45 * mix / 2.2).astype(np.float32)

    found = [i.name for i in summarize(*detect_instruments(y, sr)[:2])]
    assert found, "nothing recognised in a full-band synthetic mix"
    assert "synthesizer" in found
    assert "acoustic guitar" not in found


@needs_model
def test_detect_instruments_resamples_from_the_app_rate():
    """The pipeline runs at 22050 Hz; YAMNet only accepts 16 kHz."""
    sr = 22050
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    scores, groups, frame_sec = detect_instruments(y, sr)
    assert scores.shape[0] >= 2          # ~3 s at 0.975 s per frame
    assert scores.shape[1] == len(groups)
    assert frame_sec == pytest.approx(0.975, abs=0.01)


@needs_model
def test_scores_stay_within_range():
    sr = 16000
    rng = np.random.default_rng(0)
    y = (0.1 * rng.normal(size=sr * 3)).astype(np.float32)
    scores, groups, _ = detect_instruments(y, sr)
    assert scores.size
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    for item in summarize(scores, groups):
        assert isinstance(item, InstrumentScore)
        assert 0.0 <= item.score <= 1.0
