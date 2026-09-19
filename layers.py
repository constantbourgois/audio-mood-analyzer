"""
Which layers are playing, and when they enter and leave.

This replaces instrument-name tagging (see instruments.py) for the parts of a
mix a general audio tagger cannot hear. YAMNet's input front-end spans only
125 Hz - 7500 Hz, so an electric bass (fundamentals ~41-200 Hz) and a kick
drum sit largely *below its lowest input bin*: measured on a real track, it
scored "Bass guitar" at 0.0002 and every drum label at ~0.000 while a bass
and a drum machine were plainly audible, and it invented an acoustic guitar
that was not there. Bigger AudioSet models (PANNs, AST) share that same mel
floor, so the fix is not a bigger model.

What separates a bass line from a drum beat from a synth pad is register and
whether the sound sustains - which is signal processing, not classification,
and needs no model at all:

- bass      : energy in 20-250 Hz.
- percussion: positive spectral flux in 2-8 kHz. A drum hit is a sudden
              broadband increase; a held note is not.
- harmony   : 800-6000 Hz energy that survives a rolling MINIMUM over a short
              window. Drums put a lot of energy in this same band, but only
              in bursts - between hits it falls away, and the rolling minimum
              takes it out. Sustained tonal content never drops, so it
              survives. Without this step a synth entering at 0:12 is
              detected at 0:03 along with the snare.

Presence is judged per layer against that layer's own quiet-to-loud range
rather than an absolute threshold or a share of the frame. An absolute
threshold cannot work across recordings of different loudness; a share of the
frame is zero-sum, so a bass that plays throughout appears to "leave" the
moment a synth enters and takes some of the share.

Validated against a real track whose entry times the author supplied by ear
(bass 0:00, kick+snare 0:03, synth 0:12): detected 0.3 s / 3.2 s / 12.0 s.

Named instruments are deliberately NOT claimed here. "Low end" is measured;
calling it a bass guitar rather than a synth bass would be a guess, and a
wrong instrument name is worse than an honest register.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FRAME = 2048
HOP = 1024

# Band edges are fixed in Hz rather than derived from the file's own rate, so
# "Bright / lead" means the same thing for a 44.1 kHz upload as for a 48 kHz
# one. Deriving the top edge from sr/2 instead made the layer cover 6-22 kHz
# or 6-24 kHz depending on the file.
#
# This replaced resampling everything to a fixed rate first, which was there
# on the theory that it shrank the spectrogram. Measured, it did not: the one
# whole-track allocation (the 2-8 kHz rows kept for the flux) is ~10 MB at
# every rate, because a higher rate gives proportionally more frames but
# coarser bins per Hz and the two cancel. It also cost 6x the runtime on a
# 94 s file (505 ms vs 83 ms) - the resample itself is dearer than the extra
# FFT work it avoids - while changing detected entry times by under 0.1 s.
BASS_HZ = (20, 250)
DRUM_HZ = (2000, 8000)
HARMONY_HZ = (800, 6000)
BRIGHT_HZ = (6000, 11000)

SUSTAIN_WINDOW_SEC = 0.4    # rolling minimum: longer than a drum hit, shorter than a held note
ENVELOPE_RELEASE_SEC = 1.0  # bridges the gap between notes of one part
ENTRY_FRACTION = 0.15       # of the layer's own p5..p95 range
ENTRY_SUSTAIN_SEC = 1.5     # must hold this long to count as an entrance
GAP_BRIDGE_SEC = 4.0
# A band holding less than this share of the track's own energy is empty,
# not quiet: it keeps numerical noise in a silent band from reading as a layer.
SILENCE_SHARE = 1e-4
MIN_SPAN_SEC = 2.0


@dataclass
class Layer:
    key: str
    label: str
    hint: str
    intervals: list = field(default_factory=list)   # [(start_sec, end_sec), ...]

    @property
    def entry_sec(self) -> float | None:
        return self.intervals[0][0] if self.intervals else None


@dataclass
class LayerMap:
    duration_sec: float
    layers: list = field(default_factory=list)      # present layers, by entry time


CHUNK_FRAMES = 512


def _stft_bands(y: np.ndarray, freqs: np.ndarray, bands: list, keep: tuple):
    """Per-frame band sums, plus the raw rows of one band, in one pass.

    Chunked deliberately. numpy's rfft promotes float32 to complex128, so
    transforming a whole track at once allocates 16 bytes per bin per frame -
    measured at +119 MB for a 40 MB upload, which two gunicorn workers would
    turn into an OOM on a 512 MB instance. Doing it 512 frames at a time caps
    that spike at a few MB, and the only thing kept for the whole track is
    the narrow band `keep` needs for its flux.
    """
    if len(y) < FRAME:
        empty = np.zeros(0, dtype=np.float32)
        return [empty] * len(bands), np.zeros((0, 0), dtype=np.float32)

    frames = sliding_window_view(y, FRAME)[::HOP]
    window = np.hanning(FRAME).astype(np.float32)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in bands]
    keep_mask = (freqs >= keep[0]) & (freqs < keep[1])

    sums = [[] for _ in bands]
    kept = []
    for start in range(0, len(frames), CHUNK_FRAMES):
        block = np.abs(np.fft.rfft(frames[start:start + CHUNK_FRAMES] * window, axis=1))
        block = block.astype(np.float32, copy=False)
        for i, mask in enumerate(masks):
            sums[i].append(block[:, mask].sum(axis=1))
        kept.append(block[:, keep_mask].copy())
        del block

    return ([np.concatenate(s) for s in sums],
            np.concatenate(kept).T if kept else np.zeros((0, 0), dtype=np.float32))


def _envelope(x: np.ndarray, hop_sec: float, release: float = ENVELOPE_RELEASE_SEC) -> np.ndarray:
    """Instant attack, slow release. A part played as separate hits (drums, a
    plucked bass) falls to zero between notes; a plain moving average dips
    with it and reads as the part dropping out, so the release holds it up."""
    if len(x) == 0:
        return x
    decay = 0.1 ** (hop_sec / release)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        decayed = out[i - 1] * decay
        out[i] = x[i] if x[i] > decayed else decayed
    return out


def _rolling_min(x: np.ndarray, hop_sec: float, seconds: float) -> np.ndarray:
    width = max(1, int(seconds / hop_sec))
    if width <= 1 or len(x) < width:
        return x
    padded = np.pad(x, (width // 2, width - 1 - width // 2), mode="edge")
    return sliding_window_view(padded, width).min(axis=-1)


def _intervals(curve: np.ndarray, times: np.ndarray, hop_sec: float,
               reference: np.ndarray) -> list:
    """Spans where the layer is present, judged against its own dynamic range.

    p5..p95 rather than min..max so one stray spike cannot set the scale, and
    a low fraction of it because a layer that enters quietly and gets louder
    later (every arrangement that builds) would otherwise not register until
    its loudest section.

    `reference` is the whole-spectrum energy, used only to tell an empty band
    apart from a steady one - see below, where the relative threshold alone
    cannot make that distinction.
    """
    if curve.size == 0:
        return []
    # A band carrying essentially nothing still has a p5..p95 spread made of
    # numerical noise, which the relative test below would happily treat as
    # signal. Compare against the track's own energy to rule that out first.
    # The question is whether this band EVER carries anything, so it asks at
    # the band's loudest, not its median: a layer that only plays for the last
    # third of a track has a near-zero median and would be discarded here.
    if np.percentile(curve, 95) < SILENCE_SHARE * np.median(reference):
        return []

    quiet, loud = np.percentile(curve, 5), np.percentile(curve, 95)
    if loud <= quiet:
        return []
    # A layer that simply never stops - a drone, a bass that plays throughout -
    # has p5 and p95 a hair apart, so the relative threshold lands in the
    # middle of that noise and `on` ends up alternating on/off at ~50%, never
    # meeting the sustain requirement below. Measured on a steady tone: the
    # layer was dated 3.2 s late purely from this. Falling back to just under
    # the layer's own p95 in that case reads it as on throughout, which it is.
    on = curve >= min(quiet + ENTRY_FRACTION * (loud - quiet), 0.999 * loud)

    width = max(1, int(ENTRY_SUSTAIN_SEC / hop_sec))
    held = np.convolve(on.astype(np.float32), np.ones(width) / width, mode="same") >= 0.6
    if not held.any():
        return []

    edges = np.diff(held.astype(np.int8))
    starts = list(times[1:][edges == 1])
    ends = list(times[1:][edges == -1])
    if held[0]:
        starts = [float(times[0])] + starts
    if held[-1]:
        ends = ends + [float(times[-1])]

    # Bridge brief dropouts BEFORE discarding short spans: a pulsing layer
    # crosses the threshold in many short bursts, and filtering first would
    # throw every one of them away and lose the layer entirely.
    merged: list = []
    for start, end in zip(starts, ends):
        if merged and start - merged[-1][1] < GAP_BRIDGE_SEC:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((float(start), float(end)))
    return [(round(s, 2), round(e, 2)) for s, e in merged if e - s >= MIN_SPAN_SEC]


def analyse_layers(y: np.ndarray, sr: int) -> LayerMap:
    duration = len(y) / sr if sr else 0.0
    nyquist = sr / 2 if sr else 0.0

    def clamp(band: tuple) -> tuple:
        """Keep a band inside what this file can represent. A 16 kHz file has
        nothing above 8 kHz to measure, so the top bands collapse rather than
        silently reading as empty."""
        return (min(band[0], nyquist), min(band[1], nyquist))

    freqs = np.fft.rfftfreq(FRAME, 1 / sr) if sr else np.zeros(0)
    hop_sec = HOP / sr if sr else 0.0
    (low_band, mid_band, high_band, total_band), drum_rows = _stft_bands(
        np.asarray(y, dtype=np.float32), freqs,
        bands=[clamp(BASS_HZ), clamp(HARMONY_HZ), clamp(BRIGHT_HZ), (0, nyquist)],
        keep=clamp(DRUM_HZ),
    )
    if low_band.size == 0:
        return LayerMap(duration_sec=round(duration, 1))

    times = np.arange(low_band.size) * hop_sec
    drum_flux = np.maximum(np.diff(drum_rows, axis=1, prepend=drum_rows[:, :1]), 0).sum(axis=0)

    raw = [
        ("bass", "Bass / low end", "20-250 Hz",
         low_band),
        ("percussion", "Drums / percussion", "sharp attacks, 2-8 kHz",
         drum_flux),
        ("harmony", "Harmony / synth", "sustained 800 Hz-6 kHz",
         _rolling_min(mid_band, hop_sec, SUSTAIN_WINDOW_SEC)),
        ("high", "Bright / lead", "sustained 6-11 kHz",
         _rolling_min(high_band, hop_sec, SUSTAIN_WINDOW_SEC)),
    ]

    found = []
    for key, label, hint, curve in raw:
        spans = _intervals(_envelope(curve, hop_sec), times, hop_sec, total_band)
        if spans:
            found.append(Layer(key=key, label=label, hint=hint, intervals=spans))
    found.sort(key=lambda layer: layer.intervals[0][0])
    return LayerMap(duration_sec=round(duration, 1), layers=found)
