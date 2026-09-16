import os
import tempfile

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from instruments import build_timeline, load_audio_lite
from viz import instrument_track_svg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {"wav", "flac", "ogg", "mp3", "m4a"}
# Not removed entirely: this endpoint is a public URL with no login, so an
# unbounded upload would let anyone (or a bot) send an arbitrarily large
# request body and fill the disk or tie up a worker. 500 MB comfortably
# covers any realistic audio file - well over an hour of uncompressed
# stereo WAV - while still stopping that trivial abuse case.
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-not-secret")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Tempo/key/mode/chords/structure/song-map/similar-tracks (analysis.py,
# chords.py, song_map.py, structure.py) are deliberately not wired up here.
# That whole pipeline runs through librosa, and librosa's dependency chain
# (scipy, numba/LLVM, scikit-learn) alone measured at ~500 MB resident once
# warmed - almost the entire 512 MB ceiling of Render's free tier, before a
# single request is served. The code and its tests are kept in the repo for
# a host with more memory; this route only ever imports instruments.py,
# which needs nothing heavier than numpy and the YAMNet interpreter.


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
    suffix = os.path.splitext(filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)

    try:
        y, sr = load_audio_lite(tmp.name)
        timeline = build_timeline(y, sr)
    except Exception as exc:
        flash(f"Could not analyse this file: {exc}")
        os.remove(tmp.name)
        return redirect(url_for("index"))

    os.remove(tmp.name)

    track_svgs = [
        (track, instrument_track_svg(index, track.intervals, timeline.duration_sec))
        for index, track in enumerate(timeline.tracks)
    ]

    return render_template(
        "result.html",
        filename=filename,
        timeline=timeline,
        track_svgs=track_svgs,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
