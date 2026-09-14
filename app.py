import os
import tempfile

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from analysis import load_audio, analyze_loaded, find_similar
from chords import estimate_chords, summarize_chords
from instruments import is_available
from song_map import build_song_map
from structure import detect_sections
from viz import chord_ribbon_svg, measure_grid_svg, section_ribbon_svg, track_svg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.join(BASE_DIR, "library")
ALLOWED_EXTENSIONS = {"wav", "flac", "ogg", "mp3", "m4a"}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30 Mo

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-not-secret")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(LIBRARY_DIR, exist_ok=True)


def _warmup_numba():
    """Force librosa's numba-JIT'd functions (chroma, mfcc, beat tracking,
    hpss, pyin, and librosa.load's own lazy-loaded guvectorize) to compile
    now, at process boot, instead of on a live request. On a slow/throttled
    CPU that first-ever compile can itself take longer than a request
    timeout — this is what caused production WORKER TIMEOUT crashes."""
    import sys

    import numpy as np
    import soundfile as sf

    from analysis import SAMPLE_RATE

    try:
        t = np.linspace(0, 2.0, int(SAMPLE_RATE * 2.0), endpoint=False)
        y = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, y, SAMPLE_RATE)
            warm_y, warm_sr = load_audio(tmp.name)
        analyze_loaded(warm_y, warm_sr, "warmup.wav")
        warm_map = build_song_map(warm_y, warm_sr)
        estimate_chords(warm_y, warm_sr, warm_map.beat_times, warm_map.measure_times)
        detect_sections(warm_y, warm_sr, warm_map.beat_times, warm_map.measure_times, warm_map.tracks)
        # Loading the tagger here means its weights are paged in once in the
        # master and shared copy-on-write by every worker, not loaded per worker.
        is_available()
    except Exception as exc:
        # Never block startup on the warm-up, but say so loudly: silence here
        # means compilation lands on the first real request instead, which is
        # what the timeout crashes looked like.
        print(f"[warmup] skipped, JIT will compile on first request: {exc!r}", file=sys.stderr)


_warmup_numba()


@app.template_filter("mmss")
def mmss_filter(seconds):
    if seconds is None:
        return "?"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def file_too_large(_exc):
    flash(f"File too large (max {MAX_CONTENT_LENGTH // (1024 * 1024)} MB).")
    return redirect(url_for("index"))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("audio_file")
    if not file or file.filename == "":
        flash("Choose an audio file first.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"Unsupported format. Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    add_to_library = request.form.get("add_to_library") == "on"

    if add_to_library:
        save_path = os.path.join(LIBRARY_DIR, filename)
        file.save(save_path)
        tmp_path = save_path
        cleanup = False
    else:
        suffix = os.path.splitext(filename)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        file.save(tmp.name)
        tmp_path = tmp.name
        cleanup = True

    try:
        y, sr = load_audio(tmp_path)
        result, vec = analyze_loaded(y, sr, filename)
        song_map = build_song_map(y, sr)
        chords = estimate_chords(y, sr, song_map.beat_times, song_map.measure_times)
        sections = detect_sections(y, sr, song_map.beat_times, song_map.measure_times, song_map.tracks)
    except Exception as exc:
        flash(f"Could not analyse this file: {exc}")
        if cleanup and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return redirect(url_for("index"))

    similar = find_similar(vec, LIBRARY_DIR, exclude_filename=filename if add_to_library else None)

    if cleanup and os.path.exists(tmp_path):
        os.remove(tmp_path)

    grid_svg = measure_grid_svg(song_map.measure_times, song_map.duration_sec, song_map.beats_per_measure)
    track_svgs = [(t, track_svg(t, song_map.duration_sec, song_map.measure_times)) for t in song_map.tracks]
    chord_svg = chord_ribbon_svg(chords, song_map.duration_sec, song_map.measure_times)
    section_svg = section_ribbon_svg(sections, song_map.duration_sec)

    return render_template(
        "result.html",
        result=result,
        similar=similar,
        added_to_library=add_to_library,
        library_count=len([
            f for f in os.listdir(LIBRARY_DIR)
            if os.path.splitext(f)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS
        ]),
        song_map=song_map,
        grid_svg=grid_svg,
        track_svgs=track_svgs,
        chords=chords,
        chord_svg=chord_svg,
        top_chords=summarize_chords(chords),
        sections=sections,
        section_svg=section_svg,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
