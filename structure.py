"""
Song structure: split the track into sections (intro, verse, chorus...) and
report which layers are playing in each one.

How it works, and what is actually measured:

- Boundaries are DATA. Beat-synchronous timbre features (MFCC) plus energy are
  fed to an agglomerative segmentation, which cuts where the texture of the
  recording changes. That is the classic self-similarity approach to structural
  segmentation.
- Section FAMILIES are DATA. Each section is summarised by its mean feature
  vector, and sections that resemble each other are grouped into a family
  (A, B, C...). So "family B occurs three times" is measured, not guessed.
- The section NAMES are a GUESS. Nothing in the signal says "chorus". The names
  come from heuristics that hold for most popular music: a quiet opening
  section is an intro, the loudest repeated family is the chorus, the other
  repeated family is the verse, a one-off section late in the track is a
  bridge. On an unusual arrangement these names will be wrong even when the
  boundaries are right — which is why the family letter is shown alongside.

Each section also lists what is playing in it. When the YAMNet tagger is
available (see instruments.py) those are real instrument names — "acoustic
guitar", "drum kit". Without it, the listing falls back to the song-map
layers, which are frequency bands and harmonic/percussive components and can
only say "harmony", never which instrument produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import librosa

from instruments import detect_instruments, instruments_between
from song_map import measure_at

HOP_LENGTH = 512
WINDOW_SEC = 1.0        # feature resolution for the structure analysis

TARGET_SECONDS_PER_SECTION = 12.0
MIN_SECTIONS = 2
MAX_SECTIONS = 10
MIN_BEATS_FOR_STRUCTURE = 12

# Cosine similarity above which two sections are called the same family.
FAMILY_SIMILARITY = 0.82

# Fraction of a layer's own peak that counts as "playing" during a section.
PART_ACTIVE_RATIO = 0.25

PART_NAMES = {
    "percussion": "percussion",
    "bass": "bass",
    "chords": "harmony",
    "vocals": "vocals/lead",
}


@dataclass
class Section:
    index: int
    role: str
    family: str
    start_sec: float
    end_sec: float
    start_measure: int | None
    end_measure: int | None
    parts: list = field(default_factory=list)
    instruments: list = field(default_factory=list)   # named by the model, when it is available

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def parts_label(self) -> str:
        """Named instruments when the model recognised any, otherwise the
        band-based layers — so the timeline always says something useful."""
        if self.instruments:
            return ", ".join(i.name for i in self.instruments)
        return ", ".join(self.parts) if self.parts else "—"


def _window_features(y: np.ndarray, sr: int) -> np.ndarray:
    """One timbre+energy column per fixed window, spanning the whole track.

    Deliberately NOT beat-synchronous. Beat tracking finds no beats in a quiet
    intro or a fade-out, so a beat-locked grid starts late and ends early, and
    the opening of the song falls outside the analysis entirely. Fixed windows
    cover every second of audio, which is what a structure view needs.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    per_window = max(1, int(round(WINDOW_SEC * sr / HOP_LENGTH)))

    columns = []
    for start in range(0, mfcc.shape[1], per_window):
        end = min(start + per_window, mfcc.shape[1])
        if end <= start:
            break
        # Energy is scaled up so it weighs against the MFCC coefficients
        # instead of vanishing next to them.
        columns.append(np.concatenate([mfcc[:, start:end].mean(axis=1),
                                       [rms[start:end].mean() * 50.0]]))
    features = np.array(columns).T

    # Standardise each dimension across time. Raw MFCC means are dominated by
    # their loudness-carrying first coefficient, which every section shares:
    # left alone, the cosine similarity between any two sections is near 1 and
    # the whole track collapses into a single family. Z-scoring makes the
    # comparison about how a section differs from the track's own average.
    centre = features.mean(axis=1, keepdims=True)
    spread = features.std(axis=1, keepdims=True)
    return (features - centre) / (spread + 1e-9)


def _group_families(vectors: list[np.ndarray]) -> list[str]:
    """Greedy grouping by cosine similarity: each section joins the first
    family it resembles, or starts a new one."""
    centroids: list[np.ndarray] = []
    families: list[str] = []
    for vector in vectors:
        norm = np.linalg.norm(vector)
        unit = vector / norm if norm > 0 else vector
        best_index, best_score = -1, -np.inf
        for i, centroid in enumerate(centroids):
            score = float(np.dot(unit, centroid))
            if score > best_score:
                best_index, best_score = i, score
        if best_index >= 0 and best_score >= FAMILY_SIMILARITY:
            families.append(chr(ord("A") + best_index))
            # Nudge the centroid toward the member just added.
            centroids[best_index] = centroids[best_index] + (unit - centroids[best_index]) * 0.5
            centroids[best_index] /= max(np.linalg.norm(centroids[best_index]), 1e-9)
        else:
            centroids.append(unit)
            families.append(chr(ord("A") + len(centroids) - 1))
    return families


def _assign_roles(families: list[str], energies: list[float]) -> list[str]:
    """Heuristic names. See the module docstring: the families are measured,
    these labels are an informed guess on top of them.

    Names are decided per FAMILY, not per section, so the same musical material
    never comes back under a different name. Intro and outro are the exception:
    they are positional, and override the family's name at the very start and
    the very end.
    """
    count = len(families)
    if count == 0:
        return []
    median_energy = float(np.median(energies))

    # Positional roles are decided FIRST. Otherwise the quiet intro/outro
    # family — typically the quietest in the track — wins the "verse" slot on
    # energy, even though every one of its occurrences is about to be renamed
    # Intro/Outro, and the real verse is left as a bare letter.
    positional: dict[int, str] = {}
    if count >= 3:
        if energies[0] <= median_energy:
            positional[0] = "Intro"
        if energies[-1] <= median_energy:
            positional[count - 1] = "Outro"

    occurrences: dict[str, list[int]] = {}
    for i, family in enumerate(families):
        occurrences.setdefault(family, []).append(i)

    # Only sections that keep their family name get to define what that name is.
    body: dict[str, list[int]] = {}
    for family, indices in occurrences.items():
        kept = [i for i in indices if i not in positional]
        if kept:
            body[family] = kept

    def mean_energy(family: str) -> float:
        return float(np.mean([energies[i] for i in body[family]]))

    repeated = [f for f, idx in body.items() if len(idx) >= 2]
    # The chorus is the repeated material that hits hardest; the verse is the
    # other repeated material, quietest first.
    chorus = max(repeated, key=mean_energy) if repeated else None
    verse_pool = [f for f in repeated if f != chorus]
    verse = min(verse_pool, key=mean_energy) if verse_pool else None

    family_role = {}
    for family in occurrences:
        if family == chorus:
            family_role[family] = "Chorus"
        elif family == verse:
            family_role[family] = "Verse"
        elif len(body.get(family, [])) == 1:
            family_role[family] = "Bridge"
        else:
            family_role[family] = f"Section {family}"

    roles: list[str] = []
    counters: dict[str, int] = {}
    for i, family in enumerate(families):
        if i in positional:
            roles.append(positional[i])
            continue
        base = family_role[family]
        # Number only the roles that actually recur, so a lone bridge is just
        # "Bridge" rather than "Bridge 1". A fallback "Section B" already
        # carries its letter and never takes a number on top.
        repeats = len(body.get(family, []))
        if repeats > 1 and not base.startswith("Section "):
            counters[base] = counters.get(base, 0) + 1
            roles.append(f"{base} {counters[base]}")
        else:
            roles.append(base)
    return roles


def _active_parts(tracks, start_sec: float, end_sec: float) -> list[str]:
    parts = []
    for track in tracks:
        if not track.times:
            continue
        times = np.asarray(track.times, dtype=float)
        values = np.asarray(track.values, dtype=float)
        window = (times >= start_sec) & (times < end_sec)
        if not window.any():
            continue
        if float(values[window].max()) >= PART_ACTIVE_RATIO:
            parts.append(PART_NAMES.get(track.key, track.key))
    return parts


def _snap_to_beat(time_sec: float, beat_times: np.ndarray, tolerance: float) -> float:
    """Nudge a boundary onto the nearest beat when one is close by, so sections
    start musically rather than mid-bar. Boundaries far from any beat (a quiet
    intro with no detected beats) are left exactly where they are."""
    if not len(beat_times):
        return time_sec
    nearest = float(beat_times[int(np.argmin(np.abs(beat_times - time_sec)))])
    return nearest if abs(nearest - time_sec) <= tolerance else time_sec


def detect_sections(y: np.ndarray, sr: int, beat_times, measure_times, tracks) -> list[Section]:
    beat_times = np.asarray(beat_times, dtype=float)
    duration = len(y) / sr
    if len(beat_times) < MIN_BEATS_FOR_STRUCTURE or duration <= 0:
        return []

    features = _window_features(y, sr)
    if features.size == 0 or features.shape[1] < MIN_SECTIONS:
        return []

    wanted = int(round(duration / TARGET_SECONDS_PER_SECTION))
    n_sections = int(np.clip(wanted, MIN_SECTIONS, MAX_SECTIONS))
    n_sections = min(n_sections, features.shape[1])

    bounds = sorted(set(int(b) for b in librosa.segment.agglomerative(features, n_sections)))
    if not bounds or bounds[0] != 0:
        bounds = [0] + bounds
    edges = bounds + [features.shape[1]]

    means, spans = [], []
    for i in range(len(edges) - 1):
        start_window, end_window = edges[i], edges[i + 1]
        if end_window <= start_window:
            continue
        means.append(features[:, start_window:end_window].mean(axis=1))
        spans.append((start_window, end_window))
    if not means:
        return []

    # One pass of the tagger for the whole track; sections just slice into it.
    tag_scores, tag_groups, tag_frame_sec = detect_instruments(y, sr)

    families = _group_families(means)
    energies = [float(vector[-1]) for vector in means]   # the scaled RMS column
    roles = _assign_roles(families, energies)

    sections = []
    for i, (start_window, end_window) in enumerate(spans):
        start_sec = _snap_to_beat(start_window * WINDOW_SEC, beat_times, WINDOW_SEC / 2)
        end_sec = _snap_to_beat(end_window * WINDOW_SEC, beat_times, WINDOW_SEC / 2)
        # Cover the whole track: the first section opens at 0 and the last one
        # runs to the final sample, whatever the window grid rounded to.
        if i == 0:
            start_sec = 0.0
        if i == len(spans) - 1:
            end_sec = duration
        end_sec = min(end_sec, duration)
        if end_sec <= start_sec:
            end_sec = min(duration, start_sec + 0.1)
        sections.append(Section(
            index=i + 1,
            role=roles[i],
            family=families[i],
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            start_measure=measure_at(start_sec, measure_times),
            end_measure=measure_at(max(start_sec, end_sec - 0.01), measure_times),
            parts=_active_parts(tracks, start_sec, end_sec),
            instruments=instruments_between(tag_scores, tag_groups, tag_frame_sec, start_sec, end_sec),
        ))
    return sections
