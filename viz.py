"""
Rendu SVG (sans dépendance graphique externe) de la carte de la chanson :
une piste = une aire remplie (activité 0-1 dans le temps) + un marqueur au
moment d'entrée détecté, superposée à une grille de mesures.

Le viewBox est le même (0 0 1000 H) pour la grille et toutes les pistes,
ce qui les garde alignées verticalement dans la page (chacune en width:100%).
"""
from __future__ import annotations

VIEW_WIDTH = 1000
TRACK_HEIGHT = 60
GRID_HEIGHT = 32
CHORD_HEIGHT = 40

# En dessous de cette largeur (unités du viewBox), l'étiquette ne tient pas
# dans le bloc : mieux vaut le laisser nu que d'écrire par-dessus le voisin.
CHORD_LABEL_MIN_WIDTH = 48

SECTION_HEIGHT = 46
SECTION_LABEL_MIN_WIDTH = 90   # "Chorus 1" needs noticeably more room than "Am"
SECTION_FAMILY_SLOTS = 4       # families beyond this reuse a slot; the label still disambiguates

# Les couleurs vivent dans la feuille de style (classes CSS), pas ici : le
# SVG est inline dans la page, donc une classe suffit, et la palette reste
# définie à un seul endroit avec le reste du thème.


def _x(t: float, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return max(0.0, min(VIEW_WIDTH, (t / duration) * VIEW_WIDTH))


def measure_grid_svg(measure_times: list, duration: float, beats_per_measure: int, label_every: int = 4) -> str:
    lines = []
    n = len(measure_times)
    label_step = max(1, label_every if n <= 60 else label_every * 2 if n <= 120 else label_every * 4)
    for i, t in enumerate(measure_times):
        x = _x(t, duration)
        lines.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{GRID_HEIGHT}" class="measure-line" />')
        if i % label_step == 0:
            lines.append(f'<text x="{x:.1f}" y="{GRID_HEIGHT - 6}" class="measure-label">{i + 1}</text>')
    body = "".join(lines)
    return (
        f'<svg viewBox="0 0 {VIEW_WIDTH} {GRID_HEIGHT}" preserveAspectRatio="none" '
        f'class="grid-svg" role="img" aria-label="Grille des {n} mesures">{body}</svg>'
    )


def chord_ribbon_svg(chords: list, duration: float, measure_times: list) -> str:
    """Bande d'accords sur la même échelle de temps que la grille et les
    pistes, pour lire d'un coup d'œil l'harmonie sous l'entrée des pistes."""
    h = CHORD_HEIGHT
    parts = []

    for t in measure_times:
        x = _x(t, duration)
        parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h}" class="measure-line" />')

    for chord in chords:
        x1 = _x(chord.start_sec, duration)
        x2 = _x(chord.end_sec, duration)
        width = max(0.0, x2 - x1)
        if width <= 0:
            continue
        quality = "minor" if chord.label.endswith("m") else "major"
        parts.append(
            f'<rect x="{x1:.1f}" y="2" width="{width:.1f}" height="{h - 4}" rx="2" '
            f'class="chord-block chord-block-{quality}" />'
        )
        if width >= CHORD_LABEL_MIN_WIDTH:
            parts.append(
                f'<text x="{(x1 + x2) / 2:.1f}" y="{h / 2 + 4:.1f}" text-anchor="middle" '
                f'class="chord-label">{chord.label}</text>'
            )

    body = "".join(parts)
    return (
        f'<svg viewBox="0 0 {VIEW_WIDTH} {h}" preserveAspectRatio="none" '
        f'class="chord-svg" role="img" aria-label="Suite d\'accords estimée, {len(chords)} segments">{body}</svg>'
    )


def section_ribbon_svg(sections: list, duration: float) -> str:
    """Structure bar: one block per section, coloured by family so repeats
    are obvious at a glance, and labelled so colour is never the only cue."""
    h = SECTION_HEIGHT
    parts = []
    for section in sections:
        x1 = _x(section.start_sec, duration)
        x2 = _x(section.end_sec, duration)
        width = max(0.0, x2 - x1)
        if width <= 0:
            continue
        slot = (ord(section.family[0]) - ord("A")) % SECTION_FAMILY_SLOTS
        parts.append(
            f'<rect x="{x1:.1f}" y="2" width="{width:.1f}" height="{h - 4}" rx="3" '
            f'class="section-block section-block-{slot}" />'
        )
        if width >= SECTION_LABEL_MIN_WIDTH:
            parts.append(
                f'<text x="{(x1 + x2) / 2:.1f}" y="{h / 2 + 6:.1f}" text-anchor="middle" '
                f'class="section-label">{section.role}</text>'
            )
    body = "".join(parts)
    return (
        f'<svg viewBox="0 0 {VIEW_WIDTH} {h}" preserveAspectRatio="none" '
        f'class="section-svg" role="img" '
        f'aria-label="Song structure, {len(sections)} sections">{body}</svg>'
    )


def track_svg(track, duration: float, measure_times: list) -> str:
    h = TRACK_HEIGHT
    margin = 4

    parts = []
    for t in measure_times:
        x = _x(t, duration)
        parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h}" class="measure-line" />')

    if track.times:
        points = []
        for t, v in zip(track.times, track.values):
            x = _x(t, duration)
            y = h - margin - v * (h - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        first_x = _x(track.times[0], duration)
        last_x = _x(track.times[-1], duration)
        polygon_points = f"{first_x:.1f},{h} " + " ".join(points) + f" {last_x:.1f},{h}"
        parts.append(f'<polygon points="{polygon_points}" class="track-area track-area-{track.key}" />')

    if track.entry_time is not None:
        ex = _x(track.entry_time, duration)
        parts.append(
            f'<line x1="{ex:.1f}" y1="0" x2="{ex:.1f}" y2="{h}" '
            f'class="track-entry-line track-entry-line-{track.key}" />'
        )

    body = "".join(parts)
    return (
        f'<svg viewBox="0 0 {VIEW_WIDTH} {h}" preserveAspectRatio="none" '
        f'class="track-svg" role="img" aria-label="Activité de la piste {track.label}">{body}</svg>'
    )
