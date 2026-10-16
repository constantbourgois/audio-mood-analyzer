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

# Profils de Krumhansl-Schmuckler (poids perceptif de chaque degré de la gamme)
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


# Modes ecclésiastiques : intervalles en demi-tons depuis la tonique.
MODE_INTERVALS = {
    "Ionian": (0, 2, 4, 5, 7, 9, 11),
    "Dorian": (0, 2, 3, 5, 7, 9, 10),
    "Phrygian": (0, 1, 3, 5, 7, 8, 10),
    "Lydian": (0, 2, 4, 6, 7, 9, 11),
    "Mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "Aeolian": (0, 2, 3, 5, 7, 8, 10),
    "Locrian": (0, 1, 3, 5, 6, 8, 10),
}

# Le degré qui distingue chaque mode de son voisin le plus proche : la 7e
# majeure sépare l'ionien du mixolydien, la 6e majeure le dorien de l'éolien,
# etc. C'est sur lui que se joue la reconnaissance.
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
_NON_SCALE_WEIGHT = 1.2       # pas zéro : une note de passage chromatique ne doit pas tout casser
_CHARACTERISTIC_BONUS = 0.7


def _mode_profile(intervals: tuple, characteristic: int) -> np.ndarray:
    """Profil de hauteurs ancré sur la tonique (degré 0).

    Important : on ne peut PAS dériver ces profils en faisant tourner celui du
    majeur. Les sept modes partagent le même ensemble de notes — un profil
    tourné de deux demi-tons donnerait au dorien exactement le score de
    l'ionien sur sa propre tonique, et les sept modes seraient à égalité
    parfaite. Chaque profil pèse donc ses degrés *relativement à sa tonique*,
    ce qui les rend réellement distincts.
    """
    profile = np.full(12, _NON_SCALE_WEIGHT)
    for degree in intervals:
        profile[degree] = _SCALE_WEIGHT
    profile[0] = _TONIC_WEIGHT
    profile[7 if 7 in intervals else 6] = _FIFTH_WEIGHT   # le locrien a une quinte diminuée
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
    """Classe les 84 interprétations modales possibles (12 toniques × 7 modes).

    Renvoie plusieurs candidats à dessein : distinguer un mode de son voisin
    revient souvent à trancher sur un seul degré, qui peut être rare ou absent
    du morceau. Les deux premiers candidats sont fréquemment tous les deux
    défendables, et l'écart entre leurs scores dit à quel point la réponse est
    tranchée.
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
