import librosa
import numpy as np
import pytest

from analysis import (
    NOTE_NAMES,
    analyze_loaded,
    estimate_key,
    estimate_mood,
    feature_vector,
    find_similar,
)


def test_estimate_key_returns_valid_note_and_mode(sample_audio):
    y, sr = sample_audio
    chroma_mean = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    key, mode, confidence = estimate_key(chroma_mean)
    assert key in NOTE_NAMES
    assert mode in ("major", "minor")
    assert -1.0 <= confidence <= 1.0


def test_estimate_key_on_flat_chroma_reports_finite_confidence():
    """Silence makes every correlation NaN; the confidence must still be a
    number the result template can format."""
    key, mode, confidence = estimate_key(np.zeros(12))
    assert key in NOTE_NAMES
    assert mode in ("major", "minor")
    assert np.isfinite(confidence)
    assert int(confidence * 100) == 0


@pytest.mark.parametrize(
    "mode,tempo,rms,centroid,expect_high_valence",
    [
        ("major", 140, 0.2, 3500, True),
        ("minor", 60, 0.02, 1000, False),
    ],
)
def test_estimate_mood_valence_follows_mode_and_brightness(mode, tempo, rms, centroid, expect_high_valence):
    label, valence, arousal = estimate_mood(tempo, mode, rms, centroid, 22050)
    assert label in {
        "Happy / Energetic",
        "Peaceful / Gentle",
        "Tense / Dark",
        "Melancholic / Calm",
    }
    assert (valence >= 0.5) == expect_high_valence
    assert 0.0 <= arousal <= 1.0


def test_feature_vector_is_unit_normalized(sample_audio):
    y, sr = sample_audio
    vec = feature_vector(y, sr)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)


def test_analyze_loaded_returns_expected_fields(sample_audio):
    y, sr = sample_audio
    result, vec = analyze_loaded(y, sr, "test.wav")
    assert result.filename == "test.wav"
    assert result.duration_sec == pytest.approx(len(y) / sr, abs=0.1)
    assert result.key in NOTE_NAMES
    assert result.mode in ("major", "minor")
    assert result.mood_label
    assert 0.0 <= result.valence <= 1.0
    assert 0.0 <= result.arousal <= 1.0
    assert vec.shape[0] > 0


def test_find_similar_empty_library_returns_empty(tmp_path, sample_audio):
    y, sr = sample_audio
    vec = feature_vector(y, sr)
    assert find_similar(vec, str(tmp_path)) == []


def test_find_similar_ranks_identical_file_first(tmp_path, sample_audio):
    y, sr = sample_audio
    sf_path = tmp_path / "reference.wav"
    import soundfile as sf

    sf.write(str(sf_path), y, sr)

    vec = feature_vector(y, sr)
    matches = find_similar(vec, str(tmp_path))
    assert len(matches) == 1
    assert matches[0]["filename"] == "reference.wav"
    assert matches[0]["similarity"] > 0.99


def test_find_similar_excludes_named_file(tmp_path, sample_audio):
    y, sr = sample_audio
    sf_path = tmp_path / "self.wav"
    import soundfile as sf

    sf.write(str(sf_path), y, sr)

    vec = feature_vector(y, sr)
    matches = find_similar(vec, str(tmp_path), exclude_filename="self.wav")
    assert matches == []
