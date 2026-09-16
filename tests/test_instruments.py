import sys

import numpy as np
import pytest
import soundfile as sf

import instruments
from instruments import (
    INSTRUMENT_GROUPS,
    InstrumentScore,
    build_timeline,
    detect_instruments,
    instruments_between,
    load_audio_lite,
    summarize,
    to_model_rate,
)

needs_model = pytest.mark.skipif(
    not instruments.is_available(),
    reason="YAMNet weights not fetched (run python fetch_model.py)",
)


def test_missing_model_degrades_instead_of_raising(monkeypatch, tmp_path):
    """The app must keep working when the weights are absent and the
    self-download fails (offline, blocked host, ...) — that is the whole
    reason the tagger is optional."""
    monkeypatch.setattr(instruments, "_tagger", None)
    monkeypatch.setattr(instruments, "MODEL_PATH", str(tmp_path / "yamnet.tflite"))
    monkeypatch.setattr(instruments, "_download_model", lambda *a, **k: False)
    tagger = instruments.get_tagger()
    assert tagger.available is False


@needs_model
def test_tagger_self_downloads_when_model_file_is_missing(monkeypatch, tmp_path):
    """The whole point of the lazy download: the feature must not depend on a
    separate build step ever having run — two different attempts at wiring
    that build step in (render.yaml, then the dashboard build command) each
    silently failed to take effect on the actual deployment."""
    import shutil

    original_path = instruments.MODEL_PATH
    monkeypatch.setattr(instruments, "_tagger", None)
    target = tmp_path / "yamnet.tflite"
    monkeypatch.setattr(instruments, "MODEL_PATH", str(target))

    def fake_download(*a, **k):
        # Stand in for the network call: copy the already-fetched weights, so
        # the rest of _Tagger.__init__ (parsing labels, allocating tensors)
        # runs for real against a genuine file.
        shutil.copyfile(original_path, target)
        return True

    monkeypatch.setattr(instruments, "_download_model", fake_download)
    tagger = instruments.get_tagger()
    assert tagger.available is True
    assert target.exists()


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


def test_to_model_rate_resamples_correctly():
    sr = 22050
    y = np.zeros(sr * 2, dtype=np.float32)
    resampled = to_model_rate(y, sr)
    assert resampled.shape[0] == pytest.approx(instruments.SAMPLE_RATE * 2, abs=2)


def test_app_import_does_not_pull_in_librosa_or_scipy():
    """The whole point of the instruments-only rewrite: app.py's import chain
    must never load librosa/scipy/numba/scikit-learn, or the ~450 MB this
    measured saving disappears. Runs in a fresh subprocess: other test
    modules in this suite legitimately import librosa for the dormant
    heavy-analysis code (analysis.py, chords.py, song_map.py, structure.py),
    which would poison sys.modules for an in-process check."""
    import subprocess
    from pathlib import Path

    code = (
        "import sys\n"
        "import app\n"
        "heavy = [m for m in ('librosa', 'scipy', 'numba', 'sklearn') if m in sys.modules]\n"
        "assert not heavy, f'app.py pulled in: {heavy}'\n"
    )
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(repo_root), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_to_model_rate_is_a_passthrough_at_the_model_rate():
    y = np.arange(10, dtype=np.float32)
    assert np.array_equal(to_model_rate(y, instruments.SAMPLE_RATE), y)


def test_load_audio_lite_reads_a_wav_file(tmp_path):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(str(path), y, sr)

    loaded_y, loaded_sr = load_audio_lite(str(path))
    assert loaded_sr == sr
    assert loaded_y.shape[0] == sr
    assert loaded_y.dtype == np.float32


def test_load_audio_lite_downmixes_stereo_to_mono(tmp_path):
    sr = 22050
    stereo = np.zeros((sr, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, sr)

    y, loaded_sr = load_audio_lite(str(path))
    assert loaded_sr == sr
    assert y.ndim == 1
    assert y.shape[0] == sr
    assert np.allclose(y, 0.0, atol=1e-6)   # +0.5 and -0.5 average out


def test_runs_to_intervals_bridges_short_gaps_and_drops_blips():
    frame_sec = 1.0
    # 5 True frames, a 1-frame dropout, then 3 more True frames: should merge
    # into a single interval since the gap is well under GAP_MERGE_SEC.
    active = np.array([True] * 5 + [False] + [True] * 3)
    intervals = instruments._runs_to_intervals(active, frame_sec, duration=9.0)
    assert intervals == [(0.0, 9.0)]

    # A single frame shorter than MIN_RUN_SEC is noise, not a real entrance.
    short_frame_sec = 0.5
    blip = np.array([False, False, True, False, False])
    assert instruments._runs_to_intervals(blip, short_frame_sec, duration=2.5) == []


def test_runs_to_intervals_does_not_bridge_leading_or_trailing_silence():
    frame_sec = 1.0
    # Leading silence before the only run must not become part of the interval.
    active = np.array([False, False, True, True, True])
    intervals = instruments._runs_to_intervals(active, frame_sec, duration=5.0)
    assert intervals == [(2.0, 5.0)]


@needs_model
def test_build_timeline_orders_tracks_by_first_entrance():
    sr = 16000
    rng = np.random.default_rng(2)
    t = np.linspace(0, 6.0, sr * 6, endpoint=False)
    mix = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.00)) / 3
    mix = mix * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t) ** 2)
    mix = mix + 0.8 * np.sin(2 * np.pi * 130.81 * t)
    hits = (np.sin(2 * np.pi * 2.0 * t) > 0.95).astype(float)
    mix = mix + 0.6 * rng.normal(size=t.shape) * hits
    y = (0.45 * mix / 2.2).astype(np.float32)

    timeline = build_timeline(y, sr)
    assert timeline.duration_sec == pytest.approx(6.0, abs=0.1)
    starts = [track.intervals[0][0] for track in timeline.tracks]
    assert starts == sorted(starts)


def test_build_timeline_on_silence_has_no_tracks():
    sr = 16000
    y = np.zeros(sr * 3, dtype=np.float32)
    timeline = build_timeline(y, sr)
    assert timeline.tracks == []
