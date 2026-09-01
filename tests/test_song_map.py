import pytest

from song_map import SongMap, build_song_map

EXPECTED_TRACK_KEYS = {"percussions", "basses", "accords", "voix"}


def test_build_song_map_structure(sample_audio):
    y, sr = sample_audio
    song_map = build_song_map(y, sr)

    assert isinstance(song_map, SongMap)
    assert song_map.duration_sec == pytest.approx(len(y) / sr, abs=0.1)
    assert song_map.beats_per_measure in (3, 4)
    assert {t.key for t in song_map.tracks} == EXPECTED_TRACK_KEYS


def test_track_values_are_normalized_and_aligned(sample_audio):
    y, sr = sample_audio
    song_map = build_song_map(y, sr)

    for track in song_map.tracks:
        assert len(track.times) == len(track.values)
        assert all(0.0 <= v <= 1.0 for v in track.values)
        if track.entry_measure is not None:
            assert 1 <= track.entry_measure <= max(song_map.measure_count, 1)
