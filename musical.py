"""
Tempo and key, in plain numpy, for the memory-limited deployment.

The previous pipeline (analysis.py) got these from librosa - beat_track for
the tempo, chroma_cqt for the key - and that stack cost ~500 MB resident on a
512 MiB plan. Neither measurement needs it:

- Tempo: an onset-strength curve (log-compressed spectral flux: how much the
  spectrum suddenly gains, frame to frame), then its autocorrelation. A
  steady beat makes the curve repeat at the beat period, which shows up as a
  peak at that lag. Lags are weighted by a log-normal prior centred on
  120 BPM, one octave wide - the same prior librosa's tempo estimator uses -
  because the autocorrelation peaks just as well at half and double the beat
  and something has to pick between them. That choice is the known weak spot:
  a track felt at 70 BPM may come out at 140, so the page says so.

- Key and mode: a chroma vector (energy folded into the 12 pitch classes)
  from a long-window STFT, fed to keymode.py. The window is long (~0.37 s at
  44.1 kHz) because pitch classes are a semitone apart: at 110 Hz that is
  6.5 Hz, and a 2048-point frame's 21.5 Hz bins cannot tell them apart. The
  key is one answer for the whole track, so the coarse time resolution that
  comes with it costs nothing.

Both passes are chunked for the same reason as layers.py: numpy's rfft
returns complex128, 16 bytes per bin per frame, and an unchunked 16384-point
STFT of a 40 MB upload would run to hundreds of MB.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from keymode import estimate_key, estimate_modes

# --- tempo ---------------------------------------------------------------
# The onset curve is sampled every ~23 ms whatever the file's rate (the
# frame period librosa's tempo estimator uses), with 50% overlap. A fixed
# 512-sample hop sampled a 44.1 kHz file twice as often for no better
# answer, and was the dearest step of the whole analysis.
ONSET_HOP_SEC = 0.0232
ONSET_CHUNK = 512
ONSET_HZ = (30.0, 11000.0)
ONSET_BANDS = 48
LOG_COMPRESSION = 100.0     # log1p(C * |X|): quiet hits count, loud ones do not swamp them
MIN_BPM, MAX_BPM = 50.0, 220.0
ALT_MAX_BPM = 200.0         # "or felt at 235" helps nobody
PRIOR_BPM = 120.0
PRIOR_OCTAVES = 1.0
ACF_WINDOW_SEC = 8.0        # local autocorrelation window, as in librosa's tempogram
MIN_TEMPO_SEC = 6.0         # fewer beats than this is not enough to call a tempo
MIN_ONSET_PEAK = 0.18
MIN_ONSET_PEAKINESS = 2.5
# Peak autocorrelation, as a share of the zero-lag value, below which the
# curve is not repeating and any "tempo" would be noise (a pad, speech).
# Measured: 0.34-0.86 on six real tracks with a beat; 0.08 on an orchestral
# Brahms dance played with rubato, whose "tempo" also changed with the
# file's sample rate - better called unsteady than given a wrong number.
MIN_TEMPO_STRENGTH = 0.15

# --- key -----------------------------------------------------------------
CHROMA_FRAME = 16384
CHROMA_HOP = 8192
CHROMA_CHUNK = 32           # 32 x 8193 bins x 16 B = ~4 MB per chunk
CHROMA_HZ = (100.0, 5000.0)
SEMITONE_WIDTH = 0.5        # triangular weight: full at a semitone's centre, zero half a step off
SILENT_FRAME_SHARE = 1e-3   # of the loudest frame's chroma energy
MIN_KEY_SEC = 4.0
# Krumhansl correlation below which no key is claimed: white noise scores
# 0.33, the weakest real track (a drum & bass loop) 0.49.
MIN_KEY_CONFIDENCE = 0.4


@dataclass
class Musical:
    tempo_bpm: float | None = None
    tempo_strength: float = 0.0
    key: str | None = None             # e.g. "F# minor"
    key_confidence: float = 0.0
    modes: list = field(default_factory=list)   # keymode.ModeCandidate, best first

    @property
    def tempo_alternatives(self) -> list:
        """Half and double time, where still a plausible tempo: the reading
        the autocorrelation could equally have picked."""
        if self.tempo_bpm is None:
            return []
        return [round(b) for b in (self.tempo_bpm / 2, self.tempo_bpm * 2)
                if MIN_BPM <= b <= ALT_MAX_BPM]

    @property
    def key_strength(self) -> str:
        if self.key_confidence >= 0.75:
            return "clear"
        if self.key_confidence >= 0.6:
            return "likely"
        return "tentative"


def _frames(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(y) < frame:
        return np.zeros((0, frame), dtype=np.float32)
    return sliding_window_view(y, frame)[::hop]


def _mel_pool(freqs: np.ndarray) -> np.ndarray:
    """(bins x bands) matrix averaging STFT bins into mel-spaced bands.

    Summing raw bins hands the upper octaves most of the vote - there are
    twice as many bins in each octave as in the one below - so hi-hats set
    the rhythm and the kick barely counts. On a drum & bass loop that read
    a 136 BPM groove as 68. Mel-spaced bands give the low end a fair share,
    as librosa's mel-spectrogram onset strength does.
    """
    def mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    top = min(ONSET_HZ[1], freqs[-1])
    edges = 700.0 * (10 ** (np.linspace(mel(ONSET_HZ[0]), mel(top), ONSET_BANDS + 1) / 2595.0) - 1)
    which = np.searchsorted(edges, freqs, side="right") - 1
    pool = np.zeros((len(freqs), ONSET_BANDS), dtype=np.float32)
    inside = (which >= 0) & (which < ONSET_BANDS)
    pool[np.nonzero(inside)[0], which[inside]] = 1.0
    counts = pool.sum(axis=0)
    pool = pool[:, counts > 0] / counts[counts > 0]
    return pool


def onset_strength(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Positive log-spectral flux per frame, and the frame period in seconds."""
    hop = int(2 ** round(np.log2(sr * ONSET_HOP_SEC)))
    frame = 2 * hop
    frames = _frames(y, frame, hop)
    hop_sec = hop / sr
    if len(frames) < 2:
        return np.zeros(0, dtype=np.float32), hop_sec

    # Peak-normalised first, so the absolute gate in estimate_tempo means the
    # same thing for a quiet recording as for a loud master.
    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        return np.zeros(len(frames), dtype=np.float32), hop_sec
    gain = np.float32(1.0 / peak)

    window = np.hanning(frame).astype(np.float32) * gain
    pool = _mel_pool(np.fft.rfftfreq(frame, 1 / sr))
    out = []
    previous = None
    for start in range(0, len(frames), ONSET_CHUNK):
        block = np.abs(np.fft.rfft(frames[start:start + ONSET_CHUNK] * window, axis=1))
        block = np.log1p(LOG_COMPRESSION * (block.astype(np.float32) @ pool))
        # Carry the last frame across chunks so the flux has no seam.
        stacked = block if previous is None else np.vstack([previous, block])
        flux = np.maximum(np.diff(stacked, axis=0), 0).mean(axis=1)
        out.append(flux if previous is not None else np.concatenate([[0.0], flux]))
        previous = block[-1:]
        del block, stacked
    return np.concatenate(out).astype(np.float32), hop_sec


def _windowed_acf(x: np.ndarray, width: int) -> np.ndarray | None:
    """Autocorrelation averaged over overlapping windows, each normalised to
    its own zero lag. One autocorrelation of the whole track smears the beat
    peak whenever the tempo drifts, and lets the loudest passage set it; local
    windows keep each passage's peak sharp and give each an equal vote."""
    width = min(width, len(x))
    size = 1 << int(np.ceil(np.log2(2 * width)))
    hop = max(1, width // 2)
    taper = np.hanning(width)
    total = np.zeros(width)
    count = 0
    for start in range(0, max(1, len(x) - width + 1), hop):
        seg = x[start:start + width]
        seg = (seg - seg.mean()) * taper[:len(seg)]
        spectrum = np.fft.rfft(seg, size)
        acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[:width]
        if acf[0] > 0:
            total += acf / acf[0]
            count += 1
    return total / count if count else None


def estimate_tempo(envelope: np.ndarray, hop_sec: float) -> tuple[float | None, float]:
    """(bpm, strength); bpm is None when the curve does not repeat enough."""
    if envelope.size * hop_sec < MIN_TEMPO_SEC:
        return None, 0.0

    x = envelope.astype(np.float64)
    if not np.any(x):
        return None, 0.0
    # Beats are peaks: the flux spikes on each hit and falls back between.
    # A held chord gives a low, flat flux and noise a high, flat one, yet
    # either can still autocorrelate strongly at some lag (a pad's partials
    # beat against the hop), so both are ruled out on shape before the lag
    # search ever runs. Measured on 7 real tracks: p95 0.28-1.02 and
    # p95/median 3.3-28; a sustained triad 0.10 / 4.7, white noise 0.16 / 1.4.
    typical, high = np.median(x), np.percentile(x, 95)
    if high < MIN_ONSET_PEAK or high < MIN_ONSET_PEAKINESS * typical:
        return None, 0.0

    acf = _windowed_acf(x, max(2, int(ACF_WINDOW_SEC / hop_sec)))
    if acf is None:
        return None, 0.0
    n = len(acf)

    min_lag = max(1, int(np.floor(60.0 / (MAX_BPM * hop_sec))))
    max_lag = min(n - 2, int(np.ceil(60.0 / (MIN_BPM * hop_sec))))
    if max_lag <= min_lag:
        return None, 0.0
    lags = np.arange(min_lag, max_lag + 1)
    bpm = 60.0 / (lags * hop_sec)
    prior = np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
    best = int(lags[np.argmax(np.maximum(acf[lags], 0) * prior)])

    strength = float(acf[best])
    if strength < MIN_TEMPO_STRENGTH:
        return None, strength

    # Parabolic interpolation between frames: at a 23 ms hop, whole-frame
    # lags alone would put 128 BPM and 125 BPM in the same bin.
    a, b, c = acf[best - 1], acf[best], acf[best + 1]
    denom = a - 2 * b + c
    offset = 0.5 * (a - c) / denom if denom < 0 else 0.0
    return round(60.0 / ((best + offset) * hop_sec), 1), round(strength, 2)


def chroma_profile(y: np.ndarray, sr: int) -> np.ndarray:
    """Mean of per-frame max-normalised chroma; all zeros for silence."""
    frames = _frames(y, CHROMA_FRAME, CHROMA_HOP)
    if len(frames) == 0:
        # Shorter than one window: take what there is, zero-padded.
        if len(y) == 0:
            return np.zeros(12)
        frames = np.pad(y, (0, CHROMA_FRAME - len(y)))[None, :]

    freqs = np.fft.rfftfreq(CHROMA_FRAME, 1 / sr)
    band = (freqs >= CHROMA_HZ[0]) & (freqs < min(CHROMA_HZ[1], sr / 2))
    if not band.any():
        return np.zeros(12)
    midi = 69 + 12 * np.log2(freqs[band] / 440.0)
    nearest = np.round(midi).astype(int)
    weight = np.maximum(0.0, 1.0 - np.abs(midi - nearest) / SEMITONE_WIDTH)
    # A linear STFT gives each octave twice the bins of the one below, so a
    # plain fold lets the upper harmonics outvote the notes actually played
    # (the third harmonic is the fifth: keys came out a fifth up). Dividing by
    # each note's total weight makes every semitone count once, as a
    # log-frequency (CQT) chroma would.
    per_note = np.bincount(nearest - nearest.min(), weights=weight)
    weight = weight / per_note[nearest - nearest.min()]
    # Bins -> pitch classes as one (bins x 12) matrix, applied per chunk.
    fold = np.zeros((band.sum(), 12), dtype=np.float32)
    fold[np.arange(band.sum()), nearest % 12] = weight

    window = np.hanning(CHROMA_FRAME).astype(np.float32)
    per_frame = []
    for start in range(0, len(frames), CHROMA_CHUNK):
        block = np.abs(np.fft.rfft(frames[start:start + CHROMA_CHUNK] * window, axis=1))[:, band]
        per_frame.append(block.astype(np.float32) @ fold)
        del block
    chroma = np.concatenate(per_frame)

    energy = chroma.sum(axis=1)
    if energy.max() <= 0:
        return np.zeros(12)
    chroma = chroma[energy > SILENT_FRAME_SHARE * energy.max()]
    chroma = chroma / chroma.max(axis=1, keepdims=True)
    return chroma.mean(axis=0)


def analyse_musical(y: np.ndarray, sr: int) -> Musical:
    y = np.asarray(y, dtype=np.float32)
    if not sr or len(y) == 0:
        return Musical()

    envelope, hop_sec = onset_strength(y, sr)
    tempo, strength = estimate_tempo(envelope, hop_sec)
    result = Musical(tempo_bpm=tempo, tempo_strength=strength)

    if len(y) / sr >= MIN_KEY_SEC:
        chroma = chroma_profile(y, sr)
        if chroma.any():
            tonic, mode, confidence = estimate_key(chroma)
            if confidence >= MIN_KEY_CONFIDENCE:
                result.key = f"{tonic} {mode}"
                result.key_confidence = round(confidence, 2)
                result.modes = estimate_modes(chroma)
    return result
