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
GRID_HEIGHT = 26

TRACK_COLORS = {
    "percussions": "#c0392b",
    "basses": "#2f6f4f",
    "accords": "#3d6da8",
    "voix": "#b5652f",
}


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


def track_svg(track, duration: float, measure_times: list) -> str:
    color = TRACK_COLORS.get(track.key, "#888888")
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
        parts.append(f'<polygon points="{polygon_points}" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1.5" />')

    if track.entry_time is not None:
        ex = _x(track.entry_time, duration)
        parts.append(f'<line x1="{ex:.1f}" y1="0" x2="{ex:.1f}" y2="{h}" stroke="{color}" stroke-width="2" stroke-dasharray="4 2" />')

    body = "".join(parts)
    return (
        f'<svg viewBox="0 0 {VIEW_WIDTH} {h}" preserveAspectRatio="none" '
        f'class="track-svg" role="img" aria-label="Activité de la piste {track.label}">{body}</svg>'
    )
