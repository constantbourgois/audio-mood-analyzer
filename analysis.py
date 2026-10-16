"""
Audio analysis: key, mode, tempo, mood, and similar tracks in a local
library.

Approach:
- Tempo: beat tracking (librosa.beat.beat_track).
- Key / mode: correlation of the mean pitch-class profile (chroma) with the
  Krumhansl-Schmuckler profiles (major/minor), over all 12 transpositions.
  This is the reference MIR (Music Information Retrieval) algorithm for the
  problem, but it remains a statistical estimate: on an ambiguous or very
  short piece it can be wrong.
- Mood: NOT a trained model (no ML here), but an interpretable heuristic
  inspired by Russell's circumplex model (valence / arousal): major/minor
  mode and spectral brightness give the valence, tempo and RMS energy give
  the arousal. An indication, not ground truth.
- Similar tracks: NO external music database (that would need a third-party
  API with a key, such as AcoustID/AcousticBrainz/Spotify). Instead the
  uploaded file is compared with the files in library/ (the user's own
  collection) by cosine similarity over a feature vector (chroma + MFCC +
  tempo).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import librosa

# Re-exported: key/mode code lives in keymode.py so the app can use it
# without librosa.
from keymode import (  # noqa: F401
    CHARACTERISTIC_DEGREE, MAJOR_PROFILE, MINOR_PROFILE, MODE_INTERVALS,
    MODE_PROFILES, NOTE_NAMES, ModeCandidate, _mode_profile, estimate_key,
    estimate_modes,
)

SAMPLE_RATE = 22050
CACHE_FILE = "library_features.json"


@dataclass
class AnalysisResult:
    filename: str
    duration_sec: float
    tempo_bpm: float
    key: str
    mode: str
    key_confidence: float
    mood_label: str
    valence: float
    arousal: float
    modal_candidates: list = field(default_factory=list)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def estimate_tempo(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def _normalize(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def estimate_mood(tempo_bpm: float, mode: str, rms_mean: float, centroid_mean: float, sr: int) -> tuple[str, float, float]:
    # Arousal (perceived energy): tempo + RMS energy
    tempo_component = _normalize(tempo_bpm, 50, 160)
    rms_component = _normalize(rms_mean, 0.01, 0.25)
    arousal = 0.6 * tempo_component + 0.4 * rms_component

    # Valence (positive/negative): mode + spectral brightness
    mode_component = 1.0 if mode == "major" else 0.0
    brightness_component = _normalize(centroid_mean, 800, 4000)
    valence = 0.6 * mode_component + 0.4 * brightness_component

    if valence >= 0.5 and arousal >= 0.5:
        label = "Happy / Energetic"
    elif valence >= 0.5 and arousal < 0.5:
        label = "Peaceful / Gentle"
    elif valence < 0.5 and arousal >= 0.5:
        label = "Tense / Dark"
    else:
        label = "Melancholic / Calm"

    return label, round(valence, 2), round(arousal, 2)


def feature_vector(y: np.ndarray, sr: int, chroma: np.ndarray | None = None, tempo: float | None = None) -> np.ndarray:
    """Vector used for track similarity (chroma + MFCC + tempo).

    chroma and tempo are accepted precomputed: the full analysis usually
    already has them, and recomputing them doubled the chromagram and the
    beat tracking on every request.
    """
    if chroma is None:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    if tempo is None:
        tempo = estimate_tempo(y, sr)
    vec = np.concatenate([chroma_mean, mfcc_mean, mfcc_std, [tempo / 200.0]])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def analyze_loaded(y: np.ndarray, sr: int, filename: str) -> tuple[AnalysisResult, np.ndarray]:
    duration = float(len(y) / sr)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key, mode, confidence = estimate_key(chroma_mean)
    modal_candidates = estimate_modes(chroma_mean)

    tempo_bpm = estimate_tempo(y, sr)

    rms_mean = float(librosa.feature.rms(y=y).mean())
    centroid_mean = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    mood_label, valence, arousal = estimate_mood(tempo_bpm, mode, rms_mean, centroid_mean, sr)

    result = AnalysisResult(
        filename=filename,
        duration_sec=round(duration, 1),
        tempo_bpm=round(tempo_bpm, 1),
        key=key,
        mode=mode,
        key_confidence=round(confidence, 2),
        mood_label=mood_label,
        valence=valence,
        arousal=arousal,
        modal_candidates=modal_candidates,
    )
    vec = feature_vector(y, sr, chroma=chroma, tempo=tempo_bpm)
    return result, vec


def analyze_file(path: str) -> tuple[AnalysisResult, np.ndarray]:
    y, sr = load_audio(path)
    return analyze_loaded(y, sr, os.path.basename(path))


def _cache_path(library_dir: str) -> str:
    return os.path.join(library_dir, CACHE_FILE)


def _load_cache(library_dir: str) -> dict:
    path = _cache_path(library_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(library_dir: str, cache: dict) -> None:
    with open(_cache_path(library_dir), "w") as f:
        json.dump(cache, f)


def find_similar(vec: np.ndarray, library_dir: str, exclude_filename: str | None = None, top_n: int = 3) -> list[dict]:
    """Compare vec with every audio file in library_dir (cached by mtime)."""
    if not os.path.isdir(library_dir):
        return []

    audio_exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    cache = _load_cache(library_dir)
    updated = False

    scores = []
    for fname in os.listdir(library_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in audio_exts or fname == exclude_filename:
            continue
        fpath = os.path.join(library_dir, fname)
        mtime = str(os.path.getmtime(fpath))

        entry = cache.get(fname)
        if not entry or entry.get("mtime") != mtime:
            try:
                y, sr = load_audio(fpath)
                lib_vec = feature_vector(y, sr)
            except Exception:
                continue
            cache[fname] = {"mtime": mtime, "vector": lib_vec.tolist()}
            updated = True
            lib_vec_arr = lib_vec
        else:
            lib_vec_arr = np.array(entry["vector"])

        similarity = float(np.dot(vec, lib_vec_arr))
        scores.append({"filename": fname, "similarity": round(similarity, 3)})

    if updated:
        _save_cache(library_dir, cache)

    scores.sort(key=lambda s: s["similarity"], reverse=True)
    return scores[:top_n]
