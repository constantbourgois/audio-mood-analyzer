"""
Key and mode from a 12-bin pitch-class profile (chroma), in plain numpy.

Split out of analysis.py so the deployed app can use it without importing
librosa: these functions only ever needed numpy, but living in a module that
imports librosa at the top made them cost ~500 MB to reach. analysis.py
re-exports everything here, so its callers and tests are unchanged.

- Key: correlation of the mean chroma with the Krumhansl-Schmuckler
  major/minor profiles over all 12 transpositions - the standard MIR
  approach, and still a statistical estimate: an ambiguous or very short
  piece can fool it.
- Modes: the 84 (tonic, church mode) readings ranked the same way, against
  profiles weighted relative to each mode's own tonic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler profiles (perceptual weight of each scale degree)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    best_score = -np.inf
    best_pitch_class = 0
    best_mode = "major"
    for i in range(12):
        major_rot = np.roll(MAJOR_PROFILE, i)
        minor_rot = np.roll(MINOR_PROFILE, i)
        major_corr = np.corrcoef(chroma_mean, major_rot)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rot)[0, 1]
        if major_corr > best_score:
            best_score, best_pitch_class, best_mode = major_corr, i, "major"
        if minor_corr > best_score:
            best_score, best_pitch_class, best_mode = minor_corr, i, "minor"
    if not np.isfinite(best_score):
        # Silence (or any flat chroma) makes every correlation NaN, leaving the
        # score at -inf: report no confidence rather than an unformattable one.
        best_score = 0.0
    return NOTE_NAMES[best_pitch_class], best_mode, float(best_score)


# Church modes: intervals in semitones from the tonic.
MODE_INTERVALS = {
    "Ionian": (0, 2, 4, 5, 7, 9, 11),
    "Dorian": (0, 2, 3, 5, 7, 9, 10),
    "Phrygian": (0, 1, 3, 5, 7, 8, 10),
    "Lydian": (0, 2, 4, 6, 7, 9, 11),
    "Mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "Aeolian": (0, 2, 3, 5, 7, 8, 10),
    "Locrian": (0, 1, 3, 5, 6, 8, 10),
}

# The degree that sets each mode apart from its nearest neighbour: the major
# 7th separates Ionian from Mixolydian, the major 6th Dorian from Aeolian,
# and so on. Recognition hinges on it.
CHARACTERISTIC_DEGREE = {
    "Ionian": 11,
    "Dorian": 9,
    "Phrygian": 1,
    "Lydian": 6,
    "Mixolydian": 10,
    "Aeolian": 8,
    "Locrian": 6,
}

_TONIC_WEIGHT = 6.0
_FIFTH_WEIGHT = 4.5
_THIRD_WEIGHT = 4.0
_SCALE_WEIGHT = 2.8
_NON_SCALE_WEIGHT = 1.2       # not zero: a chromatic passing note must not break everything
_CHARACTERISTIC_BONUS = 0.7


def _mode_profile(intervals: tuple, characteristic: int) -> np.ndarray:
    """Pitch profile anchored on the tonic (degree 0).

    Important: these profiles can NOT be derived by rotating the major one.
    The seven modes share the same set of notes - a profile rotated by two
    semitones would give Dorian exactly Ionian's score on its own tonic, and
    all seven modes would tie perfectly. Each profile therefore weights its
    degrees *relative to its tonic*, which makes them genuinely distinct.
    """
    profile = np.full(12, _NON_SCALE_WEIGHT)
    for degree in intervals:
        profile[degree] = _SCALE_WEIGHT
    profile[0] = _TONIC_WEIGHT
    profile[7 if 7 in intervals else 6] = _FIFTH_WEIGHT   # Locrian has a diminished fifth
    profile[4 if 4 in intervals else 3] = _THIRD_WEIGHT
    profile[characteristic] += _CHARACTERISTIC_BONUS
    return profile


MODE_PROFILES = {
    name: _mode_profile(intervals, CHARACTERISTIC_DEGREE[name])
    for name, intervals in MODE_INTERVALS.items()
}


@dataclass
class ModeCandidate:
    tonic: str
    mode: str
    confidence: float

    @property
    def label(self) -> str:
        return f"{self.tonic} {self.mode}"


def estimate_modes(chroma_mean: np.ndarray, top_n: int = 3) -> list[ModeCandidate]:
    """Rank the 84 possible modal readings (12 tonics x 7 modes).

    Several candidates are returned on purpose: telling a mode from its
    neighbour often comes down to a single degree, which may be rare or
    absent from the piece. The top two candidates are often both defensible,
    and the gap between their scores says how clear-cut the answer is.
    """
    candidates = []
    for mode, profile in MODE_PROFILES.items():
        for tonic in range(12):
            rotated = np.roll(profile, tonic)
            correlation = np.corrcoef(chroma_mean, rotated)[0, 1]
            if not np.isfinite(correlation):
                continue
            candidates.append(ModeCandidate(NOTE_NAMES[tonic], mode, round(float(correlation), 3)))
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[:top_n]
