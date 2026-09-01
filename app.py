import os
import tempfile

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from analysis import load_audio, analyze_loaded, find_similar
from song_map import build_song_map
from viz import measure_grid_svg, track_svg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.join(BASE_DIR, "library")
ALLOWED_EXTENSIONS = {"wav", "flac", "ogg", "mp3", "m4a"}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30 Mo

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-not-secret")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(LIBRARY_DIR, exist_ok=True)


@app.template_filter("mmss")
def mmss_filter(seconds):
    if seconds is None:
        return "?"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("audio_file")
    if not file or file.filename == "":
        flash("Choisis un fichier audio d'abord.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"Format non supporté. Formats acceptés : {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
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
    except Exception as exc:
        flash(f"Impossible d'analyser ce fichier : {exc}")
        if cleanup and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return redirect(url_for("index"))

    similar = find_similar(vec, LIBRARY_DIR, exclude_filename=filename if add_to_library else None)

    if cleanup and os.path.exists(tmp_path):
        os.remove(tmp_path)

    grid_svg = measure_grid_svg(song_map.measure_times, song_map.duration_sec, song_map.beats_per_measure)
    track_svgs = [(t, track_svg(t, song_map.duration_sec, song_map.measure_times)) for t in song_map.tracks]

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
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
