import numpy as np
import pytest

from song_map import Track
from structure import (
    MIN_BEATS_FOR_STRUCTURE,
    Section,
    _assign_roles,
    _group_families,
    detect_sections,
)

SR = 22050


def _block(freqs, duration, amplitude, sr=SR, noise=0.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = sum(np.sin(2 * np.pi * f * t) for f in freqs) / max(len(freqs), 1)
    # A pulse every half second so beat tracking has onsets to lock onto.
    tone = tone * (0.55 + 0.45 * np.sin(2 * np.pi * 2.0 * t) ** 2)
    if noise:
        tone = tone + noise * np.random.default_rng(0).normal(size=t.shape)
    return (amplitude * tone).astype(np.float32)


@pytest.fixture
def structured_audio():
    """Quiet sparse opening, then a denser louder body, then quiet again —
    the coarsest structure a real arrangement has."""
    intro = _block([261.63], 6.0, 0.10)
    body = _block([130.81, 261.63, 329.63, 392.00], 12.0, 0.45, noise=0.02)
    outro = _block([261.63], 6.0, 0.10)
    return np.concatenate([intro, body, outro]), SR


def test_detect_sections_finds_boundaries_and_orders_them(structured_audio):
    y, sr = structured_audio
    beat_times = np.arange(0, len(y) / sr, 0.5)
    sections = detect_sections(y, sr, beat_times, measure_times=list(np.arange(0, 24, 2.0)), tracks=[])

    assert sections, "no section detected on clearly structured audio"
    assert [s.index for s in sections] == list(range(1, len(sections) + 1))
    for previous, current in zip(sections, sections[1:]):
        assert previous.end_sec <= current.start_sec + 1e-6
        assert previous.duration_sec > 0


def test_detect_sections_separates_quiet_opening_from_loud_body(structured_audio):
    y, sr = structured_audio
    beat_times = np.arange(0, len(y) / sr, 0.5)
    sections = detect_sections(y, sr, beat_times, measure_times=list(np.arange(0, 24, 2.0)), tracks=[])

    assert len(sections) >= 2
    # The quiet opening must not be lumped into the loud body.
    assert sections[0].end_sec < len(y) / sr


def test_detect_sections_needs_enough_beats():
    y = _block([261.63], 2.0, 0.3)
    assert detect_sections(y, SR, np.arange(0, 2.0, 0.5), [], []) == []
    too_few = np.arange(0, 1.0, 0.5)
    assert len(too_few) < MIN_BEATS_FOR_STRUCTURE
    assert detect_sections(y, SR, too_few, [], []) == []


def test_group_families_reuses_a_letter_for_similar_sections():
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.98, 0.05, 0.0])
    b1 = np.array([0.0, 0.0, 1.0])
    assert _group_families([a1, b1, a2]) == ["A", "B", "A"]


def test_group_families_gives_distinct_letters_to_distinct_sections():
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    assert _group_families(vectors) == ["A", "B"]


def test_assign_roles_names_quiet_opening_intro_and_loudest_repeat_chorus():
    families = ["A", "B", "C", "B", "C", "D"]
    energies = [0.1, 0.5, 0.9, 0.5, 0.9, 0.1]
    roles = _assign_roles(families, energies)

    assert roles[0] == "Intro"
    assert roles[-1] == "Outro"
    # C repeats and is loudest -> chorus; B repeats and is quieter -> verse.
    assert roles[2] == "Chorus 1" and roles[4] == "Chorus 2"
    assert roles[1] == "Verse 1" and roles[3] == "Verse 2"


def test_assign_roles_falls_back_to_family_letter_without_repeats():
    roles = _assign_roles(["A", "B", "C"], [0.5, 0.5, 0.5])
    assert all(role for role in roles)
    assert any("Section" in role or "Intro" in role or "Outro" in role for role in roles)


def test_active_parts_lists_only_layers_playing_in_the_window():
    quiet = Track(key="basses", label="Bass", entry_time=None, entry_measure=None,
                  times=[0.0, 1.0, 2.0, 3.0], values=[0.0, 0.0, 0.9, 0.9])
    loud = Track(key="voix", label="Vocals", entry_time=None, entry_measure=None,
                 times=[0.0, 1.0, 2.0, 3.0], values=[0.9, 0.9, 0.0, 0.0])

    y = _block([261.63], 8.0, 0.4)
    beat_times = np.arange(0, 8.0, 0.4)
    sections = detect_sections(y, SR, beat_times, [0.0, 4.0], [quiet, loud])
    assert sections
    for section in sections:
        for part in section.parts:
            assert part in {"bass", "vocals/lead"}


def test_section_exposes_a_readable_parts_label():
    section = Section(index=1, role="Intro", family="A", start_sec=0.0, end_sec=4.0,
                      start_measure=1, end_measure=2, parts=["percussion", "bass"])
    assert section.parts_label == "percussion, bass"
    assert Section(index=2, role="Outro", family="B", start_sec=4.0, end_sec=8.0,
                   start_measure=3, end_measure=4).parts_label == "—"
