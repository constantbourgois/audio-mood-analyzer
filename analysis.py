"""
Analyse audio : tonalité, mode, tempo, ambiance (mood) et morceaux similaires
dans une bibliothèque locale.

Approche :
- Tempo : beat tracking (librosa.beat.beat_track).
- Tonalité / mode : corrélation du profil de hauteurs (chroma) moyen avec les
  profils de Krumhansl-Schmuckler (majeur/mineur), sur les 12 transpositions
  possibles. C'est l'algorithme de référence en MIR (Music Information
  Retrieval) pour ce problème, mais il reste une estimation statistique :
  sur un morceau ambigu ou très court, il peut se tromper.
- Ambiance / mood : PAS un modèle entraîné (aucun ML ici), mais une
  heuristique interprétable inspirée du modèle circumplex de Russell
  (valence / arousal) : le mode majeur/mineur et la brillance spectrale
  donnent la valence, le tempo et l'énergie RMS donnent l'arousal. À prendre
  comme une indication, pas une vérité absolue.
- Morceaux similaires : PAS de base de données musicale externe (ça
  demanderait une API tierce avec clé, type AcoustID/AcousticBrainz/Spotify).
  Ici, on compare le fichier uploadé aux fichiers présents dans library/ (la
  propre collection de l'utilisateur) via une similarité cosinus sur un
  vecteur de caractéristiques (chroma + MFCC + tempo).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import librosa

NOTE_NAMES = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

# Profils de Krumhansl-Schmuckler (poids perceptif de chaque degré de la gamme)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

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


def load_audio(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def estimate_tempo(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    best_score = -np.inf
    best_pitch_class = 0
    best_mode = "majeur"
    for i in range(12):
        major_rot = np.roll(MAJOR_PROFILE, i)
        minor_rot = np.roll(MINOR_PROFILE, i)
        major_corr = np.corrcoef(chroma_mean, major_rot)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rot)[0, 1]
        if major_corr > best_score:
            best_score, best_pitch_class, best_mode = major_corr, i, "majeur"
        if minor_corr > best_score:
            best_score, best_pitch_class, best_mode = minor_corr, i, "mineur"
    if not np.isfinite(best_score):
        # Silence (or any flat chroma) makes every correlation NaN, leaving the
        # score at -inf: report no confidence rather than an unformattable one.
        best_score = 0.0
    return NOTE_NAMES[best_pitch_class], best_mode, float(best_score)


def _normalize(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def estimate_mood(tempo_bpm: float, mode: str, rms_mean: float, centroid_mean: float, sr: int) -> tuple[str, float, float]:
    # Arousal (énergie perçue) : tempo + énergie RMS
    tempo_component = _normalize(tempo_bpm, 50, 160)
    rms_component = _normalize(rms_mean, 0.01, 0.25)
    arousal = 0.6 * tempo_component + 0.4 * rms_component

    # Valence (positif/négatif) : mode + brillance spectrale
    mode_component = 1.0 if mode == "majeur" else 0.0
    brightness_component = _normalize(centroid_mean, 800, 4000)
    valence = 0.6 * mode_component + 0.4 * brightness_component

    if valence >= 0.5 and arousal >= 0.5:
        label = "Joyeux / Énergique"
    elif valence >= 0.5 and arousal < 0.5:
        label = "Paisible / Doux"
    elif valence < 0.5 and arousal >= 0.5:
        label = "Tendu / Sombre"
    else:
        label = "Mélancolique / Calme"

    return label, round(valence, 2), round(arousal, 2)


def feature_vector(y: np.ndarray, sr: int) -> np.ndarray:
    """Vecteur utilisé pour la similarité entre morceaux (chroma + MFCC + tempo)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    tempo = estimate_tempo(y, sr)
    vec = np.concatenate([chroma_mean, mfcc_mean, mfcc_std, [tempo / 200.0]])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def analyze_loaded(y: np.ndarray, sr: int, filename: str) -> tuple[AnalysisResult, np.ndarray]:
    duration = float(len(y) / sr)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key, mode, confidence = estimate_key(chroma_mean)

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
    )
    vec = feature_vector(y, sr)
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
    """Compare vec à chaque fichier audio de library_dir (avec cache par mtime)."""
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
