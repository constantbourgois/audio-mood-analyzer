import numpy as np
import pytest

import layers
from layers import analyse_layers

SR = 22050


def _mix(duration=20.0, bass_from=None, drums_from=None, pad_from=None, sr=SR):
    """Synthetic track with parts entering at known times, so entry detection
    is checked against times we actually control."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.zeros_like(t)
    rng = np.random.default_rng(0)

    if bass_from is not None:
        gate = (t >= bass_from).astype(float)
        y += 0.5 * np.sin(2 * np.pi * 70 * t) * gate          # low, sustained

    if drums_from is not None:
        # sharp broadband hits twice a second: attacks, not sustained tone
        hit = ((t * 2) % 1.0 < 0.02) & (t >= drums_from)
        y += 0.6 * rng.normal(size=t.shape) * hit

    if pad_from is not None:
        gate = (t >= pad_from).astype(float)
        y += 0.25 * sum(np.sin(2 * np.pi * f * t) for f in (1200, 1600, 2000)) / 3 * gate

    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32), sr


def _entry(result, key):
    for layer in result.layers:
        if layer.key == key:
            return layer.entry_sec
    return None


def test_bass_present_from_the_start_is_found_at_the_start():
    y, sr = _mix(bass_from=0.0)
    assert _entry(analyse_layers(y, sr), "bass") == pytest.approx(0.0, abs=1.0)


def test_layer_entering_midway_is_dated_correctly():
    y, sr = _mix(duration=24.0, bass_from=0.0, pad_from=12.0)
    result = analyse_layers(y, sr)
    assert _entry(result, "bass") == pytest.approx(0.0, abs=1.5)
    assert _entry(result, "harmony") == pytest.approx(12.0, abs=2.0)


def test_drums_are_detected_from_attacks_not_sustained_energy():
    y, sr = _mix(duration=24.0, bass_from=0.0, drums_from=8.0)
    assert _entry(analyse_layers(y, sr), "percussion") == pytest.approx(8.0, abs=2.0)


def test_sustained_pad_is_not_confused_with_percussion():
    """The rolling-minimum step exists for exactly this: drums and a synth pad
    share the 800 Hz-6 kHz band, and without it a pad entering later is dated
    to whenever the drums came in."""
    y, sr = _mix(duration=24.0, drums_from=2.0, pad_from=14.0)
    result = analyse_layers(y, sr)
    harmony = _entry(result, "harmony")
    assert harmony is not None
    assert harmony > 8.0, "the pad was dated to the drum entrance, not its own"


def test_silence_yields_no_layers():
    y = np.zeros(SR * 5, dtype=np.float32)
    assert analyse_layers(y, SR).layers == []


def test_clip_shorter_than_one_frame_does_not_raise():
    y = np.zeros(100, dtype=np.float32)
    result = analyse_layers(y, SR)
    assert result.layers == []


def test_duration_is_reported_from_the_original_rate():
    y, sr = _mix(duration=10.0, bass_from=0.0, sr=44100)
    assert analyse_layers(y, sr).duration_sec == pytest.approx(10.0, abs=0.2)


def test_layers_are_ordered_by_entrance():
    y, sr = _mix(duration=30.0, bass_from=0.0, drums_from=6.0, pad_from=16.0)
    entries = [layer.entry_sec for layer in analyse_layers(y, sr).layers]
    assert entries == sorted(entries)


def test_intervals_never_run_past_the_recording():
    y, sr = _mix(duration=15.0, bass_from=0.0)
    result = analyse_layers(y, sr)
    for layer in result.layers:
        for start, end in layer.intervals:
            assert 0.0 <= start < end <= result.duration_sec + 0.5


def test_rolling_min_keeps_sustained_and_removes_spikes():
    hop_sec = layers.HOP / SR
    steady = np.ones(400, dtype=np.float32)
    spiky = np.zeros(400, dtype=np.float32)
    spiky[::20] = 1.0                      # isolated hits, silence between
    assert layers._rolling_min(steady, hop_sec, 0.4).min() == pytest.approx(1.0)
    assert layers._rolling_min(spiky, hop_sec, 0.4).max() == pytest.approx(0.0)


def test_envelope_holds_between_hits_where_an_average_would_dip():
    hop_sec = layers.HOP / SR
    pulsed = np.zeros(400, dtype=np.float32)
    pulsed[::10] = 1.0
    enveloped = layers._envelope(pulsed, hop_sec, release=1.0)
    assert enveloped.min() > 0.3, "the envelope dropped out between hits"
