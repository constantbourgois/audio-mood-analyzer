"""
Instrument recognition with YAMNet (AudioSet tagger, MobileNet-v1 backbone).

This is the one part of the app that uses a trained model, and it is the only
way to put a real name on an instrument. Everything else here works on
frequency bands and harmonic/percussive components, which can say "something
is playing in the mid range" but never "that is an acoustic guitar".

Why this model: it is 4 MB and runs a 0.975 s window in about 2 ms on CPU, so
tagging a whole track costs a fraction of a second — small enough to sit
alongside librosa on a 512 MB instance. The heavier taggers (PANNs CNN14 at
~300 MB, AST/PaSST at ~340 MB) are more accurate but do not fit.

Everything degrades gracefully: if the runtime or the model file is missing,
every function returns empty and the app falls back to the band-based layers
rather than failing the request. The deployment fetches the model at build
time (see render.yaml); nothing is downloaded while serving a request.

Limits worth knowing:
- AudioSet labels are what the model was trained on, so it answers with that
  vocabulary: "Acoustic guitar" yes, a specific make or tuning no.
- It tags what it hears in the mix. A quiet instrument under a loud one is
  routinely missed, and a distorted guitar can read as a synthesizer.
- Confidence is a score, not a probability of being right. The threshold below
  is deliberately conservative: better to list three instruments that are
  really there than eight where five are guesses.
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass

import numpy as np

MODEL_PATH = os.environ.get(
    "YAMNET_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yamnet.tflite"),
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/"
    "yamnet/float32/1/yamnet.tflite"
)

SAMPLE_RATE = 16000          # YAMNet is trained at 16 kHz and accepts nothing else
FRAME_SAMPLES = 15600        # 0.975 s, the model's fixed input length
LABEL_FILE = "yamnet_label_list.txt"

# A score below this is treated as "not confidently present".
MIN_SCORE = 0.12
MAX_PER_SECTION = 4

# AudioSet carries 521 classes, most of them irrelevant here (Dog, Vehicle,
# Speech...). These are the musically useful ones, grouped so that the many
# near-duplicate labels collapse into one readable name.
INSTRUMENT_GROUPS = {
    "acoustic guitar": ["Acoustic guitar"],
    "electric guitar": ["Electric guitar"],
    "guitar": ["Guitar", "Strum", "Steel guitar, slide guitar", "Tapping (guitar technique)"],
    "bass guitar": ["Bass guitar"],
    "piano": ["Piano", "Electric piano", "Harpsichord"],
    "organ": ["Organ", "Electronic organ", "Hammond organ"],
    "synthesizer": ["Synthesizer", "Sampler", "Theremin"],
    "drum kit": ["Drum kit", "Drum machine", "Drum", "Snare drum", "Bass drum",
                 "Hi-hat", "Cymbal", "Rimshot", "Drum roll"],
    "percussion": ["Percussion", "Tambourine", "Maraca", "Gong", "Wood block",
                   "Marimba, xylophone", "Glockenspiel", "Vibraphone", "Steelpan"],
    "strings": ["Violin, fiddle", "Cello", "Double bass", "Bowed string instrument",
                "String section", "Pizzicato", "Orchestra"],
    "brass": ["Trumpet", "Trombone", "French horn", "Brass instrument", "Cornet"],
    "woodwind": ["Saxophone", "Clarinet", "Flute", "Oboe", "Bassoon", "Woodwind instrument"],
    "vocals": ["Singing", "Male singing", "Female singing", "Child singing",
               "Choir", "Vocal music", "Chant", "Mantra", "Humming", "Yodeling"],
    "rap": ["Rapping"],
    "harmonica": ["Harmonica"],
    "accordion": ["Accordion"],
    "banjo": ["Banjo"],
    "mandolin": ["Mandolin"],
    "ukulele": ["Ukulele"],
    "harp": ["Harp"],
    "bells": ["Bell", "Church bell", "Jingle bell", "Tubular bells", "Chime"],
}


@dataclass
class InstrumentScore:
    name: str
    score: float


class _Tagger:
    """Loads the model once and keeps it. Import-time friendly: a missing
    runtime or model file leaves `available` False instead of raising."""

    def __init__(self):
        self.available = False
        self._interpreter = None
        self._group_indices: dict[str, list[int]] = {}
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            return
        if not os.path.exists(MODEL_PATH):
            return
        try:
            with zipfile.ZipFile(MODEL_PATH) as bundle:
                labels = bundle.read(LABEL_FILE).decode().splitlines()
            self._interpreter = Interpreter(model_path=MODEL_PATH)
            self._interpreter.allocate_tensors()
            self._input = self._interpreter.get_input_details()[0]
            self._output = self._interpreter.get_output_details()[0]
        except Exception:
            return

        position = {label: i for i, label in enumerate(labels)}
        for group, members in INSTRUMENT_GROUPS.items():
            indices = [position[m] for m in members if m in position]
            if indices:
                self._group_indices[group] = indices
        self.available = bool(self._group_indices)

    def frame_scores(self, y16: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Per-frame score for each instrument group, plus the group names."""
        if not self.available or len(y16) < FRAME_SAMPLES:
            return np.zeros((0, 0)), []
        groups = list(self._group_indices)
        rows = []
        for start in range(0, len(y16) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            self._interpreter.set_tensor(
                self._input["index"], y16[start:start + FRAME_SAMPLES].astype(np.float32)
            )
            self._interpreter.invoke()
            raw = self._interpreter.get_tensor(self._output["index"])[0]
            # A group's score is its strongest member: "Electric guitar" firing
            # alone should carry the group, not be averaged away by the
            # quieter sibling labels that did not fire.
            rows.append([float(raw[self._group_indices[g]].max()) for g in groups])
        return np.array(rows), groups


_tagger: _Tagger | None = None


def get_tagger() -> _Tagger:
    global _tagger
    if _tagger is None:
        _tagger = _Tagger()
    return _tagger


def is_available() -> bool:
    return get_tagger().available


def to_model_rate(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return y
    import librosa

    return librosa.resample(y=y, orig_sr=sr, target_sr=SAMPLE_RATE)


def detect_instruments(y: np.ndarray, sr: int):
    """Returns (frame_scores, group_names, frame_seconds). Empty when the model
    is unavailable, so callers can simply fall back."""
    tagger = get_tagger()
    if not tagger.available:
        return np.zeros((0, 0)), [], FRAME_SAMPLES / SAMPLE_RATE
    scores, groups = tagger.frame_scores(to_model_rate(y, sr))
    return scores, groups, FRAME_SAMPLES / SAMPLE_RATE


def summarize(scores: np.ndarray, groups: list[str], limit: int = MAX_PER_SECTION) -> list[InstrumentScore]:
    """Strongest instruments over a set of frames, most confident first."""
    if scores.size == 0 or not groups:
        return []
    # Mean over the window, so a single stray frame cannot name an instrument.
    averaged = scores.mean(axis=0)
    order = np.argsort(averaged)[::-1]
    found = [
        InstrumentScore(groups[i], round(float(averaged[i]), 3))
        for i in order if averaged[i] >= MIN_SCORE
    ]
    return found[:limit]


def instruments_between(scores: np.ndarray, groups: list[str], frame_sec: float,
                        start_sec: float, end_sec: float,
                        limit: int = MAX_PER_SECTION) -> list[InstrumentScore]:
    if scores.size == 0 or frame_sec <= 0:
        return []
    first = max(0, int(start_sec / frame_sec))
    last = min(len(scores), max(first + 1, int(round(end_sec / frame_sec))))
    return summarize(scores[first:last], groups, limit)
