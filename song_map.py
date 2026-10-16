"""
Song map: a bar grid plus a heuristic estimate of when vocals / bass /
percussion / chords enter, over time.

A deliberate choice: NO source separation (Demucs/PyTorch style), to stay
light to install (an explicit decision by the user). Direct consequence: no
specific instrument can be identified. What is isolated instead are
FREQUENCY BANDS and the harmonic/percussive components of the full mix (via
librosa's HPSS):

- "Percussion (likely)"           -> broadband energy of the percussive
                                     component (HPSS)
- "Bass (low end)"                -> energy below 250 Hz
- "Chords / harmony (mids)"       -> energy in 250-2500 Hz of the harmonic
                                     component
- "Vocals / lead melody (likely)" -> pYIN's voiced probability (monophonic
                                     pitch tracking) on the harmonic
                                     component

A known and documented limit: a melody played by a solo instrument (guitar,
saxophone...) looks like a voice to pYIN - there is no reliable way to tell
them apart without source separation. Treat it as a visual guide to the
track's texture, not an instrument-by-instrument transcription.

The bar grid assumes 3 or 4 beats per bar (the most common cases); the
count is estimated from accent contrast, not read from a score.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, medfilt

HOP_LENGTH = 512
ENTRY_THRESHOLD_RATIO = 0.2   # share of the track's peak at which it counts as "entered"
ENTRY_SUSTAIN_SEC = 0.5       # minimum time above the threshold, to ignore a lone transient
ENVELOPE_RELEASE_SEC = 1.5    # release time: bridges the gaps between notes of one track
DOWNSAMPLE_POINTS = 240       # points kept per track for display
MEASURE_TOLERANCE_SEC = 0.05  # rounding tolerance when placing an instant on the bar grid


@dataclass
class Track:
    key: str
    label: str
    entry_time: float | None
    entry_measure: int | None
    times: list = field(default_factory=list)     # seconds, downsampled for display
    values: list = field(default_factory=list)     # 0..1, downsampled


@dataclass
class SongMap:
    tempo_bpm: float
    beats_per_measure: int
    measure_count: int
    measure_times: list
    duration_sec: float
    tracks: list
    beat_times: list = field(default_factory=list)   # reused by chord detection


def _bandpass_rms(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = sr / 2
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    sos = butter(4, [low_n, high_n], btype="band", output="sos")
    filtered = sosfiltfilt(sos, y)
    return librosa.feature.rms(y=filtered, hop_length=HOP_LENGTH)[0]


def _envelope_follower(x: np.ndarray, hop_sec: float, release_sec: float = ENVELOPE_RELEASE_SEC) -> np.ndarray:
    """Instant attack, slow exponential release: turns a track played in
    pulses (drums, a bass in short notes...) into a continuous "presence"
    curve by bridging the gaps between close notes. A plain moving average
    fails here: on a track made of short peaks separated by silence, it
    drops below the threshold between peaks and prevents a "sustained"
    entry from ever being detected."""
    if len(x) == 0:
        return x
    decay = 0.1 ** (hop_sec / release_sec)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        decayed = out[i - 1] * decay
        out[i] = x[i] if x[i] > decayed else decayed
    return out


def measure_at(time_sec: float, measure_times) -> int | None:
    """1-indexed bar containing this instant, or None when it cannot be placed
    on the grid — either no bars were found at all, or the instant falls before
    the first one. A quiet intro carries no detectable beat, so the grid starts
    after it; reporting "bar 1" there would put the intro inside a bar that
    begins several seconds later."""
    if measure_times is None or not len(measure_times):
        return None
    # Small tolerance: a boundary snapped to a beat and then rounded can land a
    # few milliseconds before that bar line, which would otherwise read as
    # "before the first bar" for a section that starts exactly on it.
    index = int(np.searchsorted(measure_times, time_sec + MEASURE_TOLERANCE_SEC, side="right"))
    return index if index >= 1 else None


def _normalize01(x: np.ndarray) -> np.ndarray:
    m = float(x.max()) if len(x) else 0.0
    return x / m if m > 0 else x


def _find_entry(times: np.ndarray, values: np.ndarray, hop_sec: float) -> float | None:
    """First instant the track exceeds its own threshold and stays there long
    enough (a lone peak is ignored)."""
    if len(values) == 0:
        return None
    threshold = ENTRY_THRESHOLD_RATIO * values.max()
    if threshold <= 0:
        return None
    sustain_frames = max(1, int(ENTRY_SUSTAIN_SEC / hop_sec))
    above = values >= threshold
    run = 0
    for i, ok in enumerate(above):
        run = run + 1 if ok else 0
        if run >= sustain_frames:
            return float(times[i - sustain_frames + 1])
    return None


def _downsample(times: np.ndarray, values: np.ndarray, duration: float, n: int = DOWNSAMPLE_POINTS) -> tuple[list, list]:
    if len(times) == 0 or duration <= 0:
        return [], []
    grid = np.linspace(0, duration, n)
    resampled = np.interp(grid, times, values, left=values[0], right=values[-1])
    return [round(t, 2) for t in grid.tolist()], [round(v, 3) for v in resampled.tolist()]


def _estimate_beats_per_measure(onset_env: np.ndarray, beat_frames: np.ndarray) -> int:
    """4/4 by default (the most common); switch to 3/4 only when the accent
    contrast (beat 1 louder than the others) clearly supports it, otherwise a
    track without marked accents would flip the estimate at random."""
    if len(beat_frames) < 8:
        return 4

    accents = onset_env[beat_frames]

    def contrast_for(g: int) -> float | None:
        if len(accents) < g * 2:
            return None
        firsts = accents[0::g]
        mask = np.ones(len(accents), dtype=bool)
        mask[0::g] = False
        others = accents[mask]
        if len(others) == 0:
            return None
        return float(firsts.mean() - others.mean())

    contrast_3 = contrast_for(3)
    contrast_4 = contrast_for(4) or 0.0
    if contrast_3 is not None and contrast_3 > max(contrast_4, 0.0):
        return 3
    return 4


def build_song_map(y: np.ndarray, sr: int) -> SongMap:
    duration = len(y) / sr
    hop_sec = HOP_LENGTH / sr

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH, onset_envelope=onset_env)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    beats_per_measure = _estimate_beats_per_measure(onset_env, beat_frames)
    measure_times = beat_times[0::beats_per_measure] if len(beat_times) else np.array([])

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # The low band is measured on the harmonic component: on the full mix,
    # the drums' broadband noise spills into the low end and hides the bass's
    # real entry.
    raw_tracks = []

    perc_rms = librosa.feature.rms(y=y_percussive, hop_length=HOP_LENGTH)[0]
    raw_tracks.append(("percussion", "Percussion (likely)", perc_rms, False))

    bass_rms = _bandpass_rms(y_harmonic, sr, 30, 250)
    raw_tracks.append(("bass", "Bass (low end)", bass_rms, False))

    mid_rms = _bandpass_rms(y_harmonic, sr, 250, 2500)
    raw_tracks.append(("chords", "Chords / harmony (mids)", mid_rms, False))

    # fmin raised to C3: keeps a low bass note (often 40-130 Hz) from being
    # taken for a voice, even though it excludes the deepest male voices - an
    # accepted trade-off without source separation.
    _, _, voiced_prob = librosa.pyin(
        y_harmonic,
        fmin=librosa.note_to_hz("C3"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    voiced_prob = np.nan_to_num(voiced_prob)
    raw_tracks.append(("vocals", "Vocals / lead melody (likely)", voiced_prob, True))

    tracks = []
    for key, label, values, denoise in raw_tracks:
        times = librosa.frames_to_time(np.arange(len(values)), sr=sr, hop_length=HOP_LENGTH)
        # pYIN can occasionally flag a single "voiced" instant on noise; a
        # median filter removes it without touching a real sung passage. On
        # the RMS tracks the same filter would instead erase a legitimate drum
        # hit (1-2 frames wide), so it is kept to the vocals.
        source = medfilt(values, kernel_size=5) if denoise else values
        enveloped = _envelope_follower(source, hop_sec)
        display_curve = _normalize01(enveloped)
        entry = _find_entry(times, enveloped, hop_sec)
        entry_measure = None
        if entry is not None and len(measure_times):
            # number of bar starts <= entry = bar number (1-indexed); clamped
            # to 1 if the entry precedes the very first detected beat.
            entry_measure = max(1, int(np.searchsorted(measure_times, entry, side="right")))
        d_times, d_values = _downsample(times, display_curve, duration)
        tracks.append(Track(
            key=key,
            label=label,
            entry_time=round(entry, 2) if entry is not None else None,
            entry_measure=entry_measure,
            times=d_times,
            values=d_values,
        ))

    return SongMap(
        tempo_bpm=round(tempo, 1),
        beats_per_measure=beats_per_measure,
        measure_count=len(measure_times),
        measure_times=[round(t, 2) for t in measure_times.tolist()],
        duration_sec=round(duration, 1),
        tracks=tracks,
        beat_times=[round(t, 3) for t in beat_times.tolist()],
    )
