from song_map import Track
from viz import measure_grid_svg, track_svg


def test_measure_grid_svg_draws_one_line_per_measure():
    measure_times = [0.0, 1.0, 2.0, 3.0]
    svg = measure_grid_svg(measure_times, duration=4.0, beats_per_measure=4)
    assert svg.startswith("<svg")
    assert svg.count("<line") == len(measure_times)


def test_measure_grid_svg_handles_no_measures():
    svg = measure_grid_svg([], duration=0.0, beats_per_measure=4)
    assert "<svg" in svg
    assert "<line" not in svg


def test_track_svg_renders_polygon_and_entry_marker_when_data_present():
    track = Track(
        key="voix",
        label="Voix",
        entry_time=1.0,
        entry_measure=2,
        times=[0.0, 1.0, 2.0],
        values=[0.1, 0.8, 0.3],
    )
    svg = track_svg(track, duration=2.0, measure_times=[0.0, 1.0])
    assert "<polygon" in svg
    assert "stroke-dasharray" in svg  # the entry-time marker


def test_track_svg_handles_no_data_without_marker():
    track = Track(key="basses", label="Basses", entry_time=None, entry_measure=None, times=[], values=[])
    svg = track_svg(track, duration=10.0, measure_times=[])
    assert "<svg" in svg
    assert "<polygon" not in svg
    assert "stroke-dasharray" not in svg
