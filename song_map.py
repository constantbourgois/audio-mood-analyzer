"""
Carte de la chanson : grille de mesures + estimation heuristique du moment
d'entree de la voix / basse / percussions / accords, dans le temps.

Choix assume : PAS de separation de sources (type Demucs/PyTorch), pour
rester leger a installer (decision explicite de l'utilisateur). Consequence
directe : on ne peut pas identifier un instrument precis. Ce qu'on isole a
la place, ce sont des BANDES DE FREQUENCE et les composantes
harmonique/percussive du signal complet (via HPSS de librosa) :

- "Percussions (probable)"      -> energie large bande de la composante
                                    percussive (HPSS)
- "Basses (graves)"              -> energie de la bande < 250 Hz
- "Accords / harmonie (médiums)" -> energie de la bande 250-2500 Hz sur la
                                    composante harmonique
- "Voix / mélodie (probable)"    -> probabilite "voisee" de pYIN (suivi de
                                    hauteur monophonique) sur la composante
                                    harmonique

Limite assumee et documentee : une mélodie jouée par un instrument solo
(guitare, saxophone...) ressemblera a de la voix pour pYIN — il n'y a pas de
distinction fiable entre les deux sans separation de sources. A traiter
comme un guide visuel de la texture du morceau, pas une transcription
instrument par instrument.

Le decoupage en mesures suppose une mesure a 3 ou 4 temps (les cas les plus
frequents) ; le nombre de temps par mesure est estime par contraste
d'accentuation, pas lu dans une partition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, medfilt

HOP_LENGTH = 512
ENTRY_THRESHOLD_RATIO = 0.2   # fraction du pic de la piste pour considerer qu'elle est "entree"
ENTRY_SUSTAIN_SEC = 0.5       # duree minimale au-dessus du seuil pour ignorer un simple transitoire
ENVELOPE_RELEASE_SEC = 1.5    # temps de relachement : comble les silences entre deux notes d'une meme piste
DOWNSAMPLE_POINTS = 240       # nombre de points gardes par piste pour l'affichage


@dataclass
class Track:
    key: str
    label: str
    entry_time: float | None
    entry_measure: int | None
    times: list = field(default_factory=list)     # secondes, sous-echantillonne pour l'affichage
    values: list = field(default_factory=list)     # 0..1, sous-echantillonne


@dataclass
class SongMap:
    tempo_bpm: float
    beats_per_measure: int
    measure_count: int
    measure_times: list
    duration_sec: float
    tracks: list


def _bandpass_rms(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = sr / 2
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    sos = butter(4, [low_n, high_n], btype="band", output="sos")
    filtered = sosfiltfilt(sos, y)
    return librosa.feature.rms(y=filtered, hop_length=HOP_LENGTH)[0]


def _envelope_follower(x: np.ndarray, hop_sec: float, release_sec: float = ENVELOPE_RELEASE_SEC) -> np.ndarray:
    """Attaque instantanee, relachement exponentiel lent : transforme une
    piste jouee en pulsations (batterie, basse en notes courtes...) en une
    courbe de "presence" continue, en comblant les silences entre deux
    notes rapprochees. Une simple moyenne glissante echoue ici : sur une
    piste faite de pics courts separes de silences, elle retombe sous le
    seuil entre chaque pic et empeche de detecter une entree "soutenue"."""
    if len(x) == 0:
        return x
    decay = 0.1 ** (hop_sec / release_sec)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        decayed = out[i - 1] * decay
        out[i] = x[i] if x[i] > decayed else decayed
    return out


def _normalize01(x: np.ndarray) -> np.ndarray:
    m = float(x.max()) if len(x) else 0.0
    return x / m if m > 0 else x


def _find_entry(times: np.ndarray, values: np.ndarray, hop_sec: float) -> float | None:
    """Premier instant ou la piste depasse son propre seuil et y reste assez
    longtemps (ignore un pic isole)."""
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
    """4/4 par defaut (le plus courant) ; on ne bascule sur 3/4 que si le
    contraste d'accentuation (temps 1 plus fort que les autres) le soutient
    clairement, sinon un morceau sans accent marque ferait basculer
    l'estimation au hasard."""
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

    # bande basse mesuree sur la composante harmonique : sur le signal complet,
    # le bruit large-bande des percussions deborde dans les graves et masque
    # l'entree reelle de la basse.
    raw_tracks = []

    perc_rms = librosa.feature.rms(y=y_percussive, hop_length=HOP_LENGTH)[0]
    raw_tracks.append(("percussions", "Percussions (probable)", perc_rms, False))

    bass_rms = _bandpass_rms(y_harmonic, sr, 30, 250)
    raw_tracks.append(("basses", "Basses (graves)", bass_rms, False))

    mid_rms = _bandpass_rms(y_harmonic, sr, 250, 2500)
    raw_tracks.append(("accords", "Accords / harmonie (médiums)", mid_rms, False))

    # fmin releve a C3 : evite de confondre une note de basse grave (souvent
    # 40-130 Hz) avec une voix, meme si ca exclut les voix d'homme les plus
    # graves - compromis assume, sans separation de sources.
    _, _, voiced_prob = librosa.pyin(
        y_harmonic,
        fmin=librosa.note_to_hz("C3"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    voiced_prob = np.nan_to_num(voiced_prob)
    raw_tracks.append(("voix", "Voix / mélodie principale (probable)", voiced_prob, True))

    tracks = []
    for key, label, values, denoise in raw_tracks:
        times = librosa.frames_to_time(np.arange(len(values)), sr=sr, hop_length=HOP_LENGTH)
        # pYIN peut ponctuellement signaler un unique instant "voise" sur du
        # bruit ; un filtre median l'elimine sans affecter un vrai passage
        # chante. Sur les pistes RMS, ce meme filtre effacerait au contraire
        # un coup de batterie legitime (large de 1-2 frames) : reserve a la voix.
        source = medfilt(values, kernel_size=5) if denoise else values
        enveloped = _envelope_follower(source, hop_sec)
        display_curve = _normalize01(enveloped)
        entry = _find_entry(times, enveloped, hop_sec)
        entry_measure = None
        if entry is not None and len(measure_times):
            # nombre de debuts de mesure <= entry = numero de mesure (1-indexe) ;
            # clampe a 1 si l'entree precede le tout premier temps detecte.
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
    )
