import os
import tempfile

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from instruments import load_audio_lite
from layers import analyse_layers
from musical import analyse_musical
from viz import layer_track_svg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {"wav", "flac", "ogg", "mp3", "m4a"}
MAX_CONTENT_LENGTH = 40 * 1024 * 1024  # 40 MB

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-not-secret")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Two pipelines are deliberately not wired up here, for different reasons.
#
# The librosa pipeline (analysis.py, chords.py, song_map.py, structure.py)
# needs scipy, numba/LLVM and scikit-learn, which measured at ~500 MB
# resident once warmed - almost the entire 512 MB ceiling of this plan,
# before a single request is served. Tempo and key are back regardless:
# musical.py measures them in plain numpy (validated against that pipeline
# on real tracks, see its docstring), and the key/mode maths it shares with
# analysis.py now lives in keymode.py, which imports nothing heavier.
# Chords and song structure are still off.
#
# Instrument NAMING (instruments.py, YAMNet) is left out because it was
# wrong on real material, not because of its size. Its input front-end
# spans 125 Hz - 7500 Hz, so an electric bass and a kick drum fall below
# its lowest bin: on a real track it scored "Bass guitar" at 0.0002 with a
# bass plainly audible, missed the drums, and reported an acoustic guitar
# that was not in the recording. layers.py measures register and
# percussiveness instead, which is what actually distinguishes these parts.
# Both modules stay in the repo, with their tests, for a bigger host.


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
    return render_template("index.html", max_mb=MAX_CONTENT_LENGTH // (1024 * 1024))


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
        layer_map = analyse_layers(y, sr)
        musical = analyse_musical(y, sr)
    except Exception as exc:
        flash(f"Could not analyse this file: {exc}")
        os.remove(tmp.name)
        return redirect(url_for("index"))

    os.remove(tmp.name)

    layer_svgs = [
        (layer, layer_track_svg(index, layer.intervals, layer_map.duration_sec))
        for index, layer in enumerate(layer_map.layers)
    ]

    return render_template(
        "result.html",
        filename=filename,
        layer_map=layer_map,
        layer_svgs=layer_svgs,
        musical=musical,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
