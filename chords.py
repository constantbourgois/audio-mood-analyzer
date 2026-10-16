"""
Chord recognition by template matching on a beat-synchronous chromagram.

A deliberate choice, with no trained model: for each interval between two
beats, the chromagram is summarised by its median into a 12-pitch vector,
then compared by cosine similarity with the 24 triad templates (12 major +
12 minor). The best score wins; below a threshold, no chord is reported
rather than one being invented.

Limits worth knowing:
- major/minor triads only: a seventh, an extended chord or a power chord is
  reduced to the nearest triad (often the right root, but the quality may be
  wrong);
- the chromagram is computed on the full mix: loud drums spread energy over
  all 12 pitches and blur the match;
- segmentation depends entirely on the beat grid: if beat tracking is wrong,
  the chord boundaries are wrong with it.

Read it as a rough transcription to check by ear, not as a reference chord
chart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa

from analysis import NOTE_NAMES

HOP_LENGTH = 512  # same grid as song_map.py

# Below this cosine similarity the segment is too ambiguous (silence, noise,
# harmony too dense) to name a chord.
MIN_SIMILARITY = 0.62

# Intervals in semitones from the root.
QUALITIES = {
    "": (0, 4, 7),    # major triad -> "C"
    "m": (0, 3, 7),   # minor triad -> "Cm"
}


@dataclass
class ChordSegment:
    label: str
    start_sec: float
    end_sec: float
    measure: int | None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _build_templates() -> tuple[np.ndarray, list[str]]:
    """One normalised row per chord: its dot product with a normalised
    chroma is the cosine similarity directly."""
    rows = []
    labels = []
    for suffix, intervals in QUALITIES.items():
        for root in range(12):
            template = np.zeros(12)
            for interval in intervals:
                template[(root + interval) % 12] = 1.0
            rows.append(template / np.linalg.norm(template))
            labels.append(f"{NOTE_NAMES[root]}{suffix}")
    return np.array(rows), labels


TEMPLATES, TEMPLATE_LABELS = _build_templates()


def _match(chroma_vec: np.ndarray) -> str | None:
    norm = np.linalg.norm(chroma_vec)
    if norm <= 0:
        return None
    similarities = TEMPLATES @ (chroma_vec / norm)
    best = int(np.argmax(similarities))
    if similarities[best] < MIN_SIMILARITY:
        return None
    return TEMPLATE_LABELS[best]


def _deblip(labels: list[str | None]) -> list[str | None]:
    """A single chord between two identical segments is almost always a
    framing error rather than a real one-beat chord change."""
    smoothed = list(labels)
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
            smoothed[i] = labels[i - 1]
    return smoothed


def _measure_of(time_sec: float, measure_times) -> int | None:
    if not len(measure_times):
        return None
    return max(1, int(np.searchsorted(measure_times, time_sec, side="right")))


def estimate_chords(y: np.ndarray, sr: int, beat_times, measure_times=None) -> list[ChordSegment]:
    """Chord sequence between the detected beats, with consecutive identical
    segments merged."""
    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) < 2:
        return []

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP_LENGTH)
    beat_frames = np.clip(beat_frames, 0, chroma.shape[1] - 1)

    labels: list[str | None] = []
    for i in range(len(beat_frames) - 1):
        start, end = beat_frames[i], beat_frames[i + 1]
        window = chroma[:, start:end] if end > start else chroma[:, start:start + 1]
        labels.append(_match(np.median(window, axis=1)) if window.size else None)

    labels = _deblip(labels)

    grid = measure_times if measure_times is not None else []
    segments: list[ChordSegment] = []
    for i, label in enumerate(labels):
        if label is None:
            continue
        start, end = float(beat_times[i]), float(beat_times[i + 1])
        # Boundaries are compared unrounded: rounding first would prevent any
        # merge (a segment's rounded end does not land on the next beat), and
        # the result would be one segment per beat instead of one per chord.
        if segments and segments[-1].label == label and abs(segments[-1].end_sec - start) < 1e-6:
            segments[-1].end_sec = end
        else:
            segments.append(ChordSegment(
                label=label,
                start_sec=start,
                end_sec=end,
                measure=_measure_of(start, grid),
            ))

    for segment in segments:
        segment.start_sec = round(segment.start_sec, 2)
        segment.end_sec = round(segment.end_sec, 2)
    return segments


def summarize_chords(segments: list[ChordSegment], top_n: int = 6) -> list[dict]:
    """Most present chords, by total duration - useful for seeing a track's
    harmonic colour at a glance."""
    totals: dict[str, float] = {}
    for segment in segments:
        totals[segment.label] = totals.get(segment.label, 0.0) + segment.duration_sec
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [{"label": label, "seconds": round(seconds, 1)} for label, seconds in ranked[:top_n]]
