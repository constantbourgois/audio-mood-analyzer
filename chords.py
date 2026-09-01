"""
Reconnaissance d'accords par correspondance de gabarits (template matching)
sur un chromagramme synchronisé aux temps détectés.

Approche assumée, sans modèle entraîné : pour chaque intervalle entre deux
temps, on résume le chromagramme par sa médiane en un vecteur de 12 hauteurs,
puis on le compare par similarité cosinus aux 24 gabarits d'accords parfaits
(12 majeurs + 12 mineurs). Le meilleur score l'emporte ; sous un seuil, on ne
déclare aucun accord plutôt que d'en inventer un.

Limites à connaître :
- seulement des accords parfaits majeurs/mineurs : une septième, un accord
  enrichi ou un power chord sera ramené au triade le plus proche (souvent le
  bon fondamental, mais la qualité peut être fausse) ;
- le chromagramme est calculé sur le signal complet : une batterie forte
  étale de l'énergie sur les 12 hauteurs et brouille la détection ;
- le découpage dépend entièrement de la grille de temps : si le suivi de
  tempo se trompe, les frontières d'accords se trompent avec lui.

À lire comme une transcription approximative à vérifier à l'oreille, pas
comme une grille d'accords de référence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa

from analysis import NOTE_NAMES

HOP_LENGTH = 512  # même grille que song_map.py

# Sous ce score de similarité cosinus, le segment est jugé trop ambigu
# (silence, bruit, harmonie trop dense) pour nommer un accord.
MIN_SIMILARITY = 0.62

# Intervalles en demi-tons depuis la fondamentale.
QUALITIES = {
    "": (0, 4, 7),    # accord parfait majeur -> "Do"
    "m": (0, 3, 7),   # accord parfait mineur -> "Dom"
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
    """Une ligne par accord, normalisée : le produit scalaire avec un
    chroma normalisé donne directement la similarité cosinus."""
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
    """Un accord isolé entre deux segments identiques est presque toujours une
    erreur de trame plutôt qu'un vrai changement d'accord d'un seul temps."""
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
    """Suite d'accords entre les temps détectés, segments consécutifs
    identiques fusionnés."""
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
        # Comparaison des frontières en valeurs brutes : arrondir avant de
        # comparer empêcherait toute fusion (l'arrondi de la fin d'un segment
        # ne retombe pas sur le temps suivant), et on obtiendrait un segment
        # par temps au lieu d'un par accord.
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
    """Accords les plus présents, par durée cumulée — utile pour reconnaître
    la couleur harmonique d'un morceau d'un coup d'œil."""
    totals: dict[str, float] = {}
    for segment in segments:
        totals[segment.label] = totals.get(segment.label, 0.0) + segment.duration_sec
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [{"label": label, "seconds": round(seconds, 1)} for label, seconds in ranked[:top_n]]
