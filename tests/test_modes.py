import numpy as np
import pytest

from keymode import (
    MODE_INTERVALS,
    MODE_PROFILES,
    NOTE_NAMES,
    estimate_modes,
)


def _chroma_for(tonic: int, mode: str) -> np.ndarray:
    """Chroma idéalisé : les degrés du mode présents, la tonique appuyée."""
    intervals = MODE_INTERVALS[mode]
    chroma = np.full(12, 0.05)
    for degree in intervals:
        chroma[(tonic + degree) % 12] = 0.6
    chroma[tonic] = 1.0
    # La quinte du mode lui-même : le locrien est le seul sans quinte juste,
    # appuyer un degré 7 absent de sa gamme fausserait l'exemple.
    fifth = 7 if 7 in intervals else 6
    chroma[(tonic + fifth) % 12] = 0.8
    return chroma


def test_mode_profiles_are_not_mere_rotations_of_each_other():
    """Le piège de la détection modale : dériver les profils en faisant
    tourner celui du majeur les rendrait tous équivalents, et les sept modes
    seraient à égalité parfaite sur chaque tonique."""
    ionian = MODE_PROFILES["Ionian"]
    for name, profile in MODE_PROFILES.items():
        if name == "Ionian":
            continue
        for shift in range(12):
            assert not np.allclose(profile, np.roll(ionian, shift)), (
                f"{name} is a plain rotation of Ionian (décalage {shift})"
            )


@pytest.mark.parametrize("mode", list(MODE_INTERVALS))
def test_each_mode_is_recoverable_from_its_own_chroma(mode):
    tonic = 2  # Ré
    best = estimate_modes(_chroma_for(tonic, mode))[0]
    assert best.mode == mode
    assert best.tonic == NOTE_NAMES[tonic]


def test_estimate_modes_returns_ranked_candidates():
    candidates = estimate_modes(_chroma_for(0, "Mixolydian"), top_n=3)
    assert len(candidates) == 3
    scores = [c.confidence for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert candidates[0].label == "C Mixolydian"


def test_estimate_modes_on_flat_chroma_does_not_crash():
    """Toutes les corrélations valent NaN sur du silence : la fonction doit
    rendre une liste (vide ou non) plutôt que de propager un NaN."""
    assert estimate_modes(np.zeros(12)) == []


def test_mixolydian_and_ionian_differ_on_the_seventh():
    """Les deux modes ne se distinguent que par la 7e : le mode détecté doit
    suivre ce degré et rien d'autre."""
    ionian_best = estimate_modes(_chroma_for(0, "Ionian"))[0]
    mixo_best = estimate_modes(_chroma_for(0, "Mixolydian"))[0]
    assert ionian_best.mode == "Ionian"
    assert mixo_best.mode == "Mixolydian"
    assert ionian_best.tonic == mixo_best.tonic == "C"
