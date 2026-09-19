# Audio Analyser

Upload an audio file and see **which layers of the arrangement are playing,
and when each one enters and leaves**.

Live at <https://audio-mood-analyzer.onrender.com> (free tier — the instance
sleeps after 15 minutes idle, so the first request after a pause waits on a
cold start).

| Layer | What is measured |
| --- | --- |
| Bass / low end | energy in 20–250 Hz |
| Drums / percussion | positive spectral flux in 2–8 kHz — a hit is a sudden broadband rise, a held note is not |
| Harmony / synth | 800 Hz–6 kHz that survives a rolling minimum, i.e. sustained rather than struck |
| Bright / lead | the same, 6–11 kHz |

Each layer is drawn as a bar on a shared time axis, with its entry and exit
times listed.

## Run it locally

Python 3.11 (what the deployment pins, see `.python-version`).

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

Then open <http://127.0.0.1:5050>. That is Flask's development server with
`debug=True`, so it reloads when you edit a file. Use `PORT=8000 python
app.py` if 5050 is taken.

To run it under gunicorn, as the deployment does — same worker count and
boot logging (production additionally overrides the timeout, see
Deployment):

```bash
PORT=5050 SECRET_KEY=anything gunicorn -c gunicorn.conf.py app:app
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

77 tests, about 30 seconds. They generate their audio in memory, so no
sample files are needed.

### File formats

`.wav`, `.flac` and `.ogg` work out of the box. `.mp3` and `.m4a` need
either libsndfile ≥ 1.2 (which recent `soundfile` wheels bundle) or `ffmpeg`
on your PATH as the fallback. If an mp3 fails to load, install ffmpeg.

## How the analysis works

All of it is in `layers.py`, in plain numpy — no model, no librosa, no scipy.

One STFT pass gives a grid of energy per frequency per moment
(`sliding_window_view` produces the overlapping frames as a *view*, so the
framing itself allocates nothing). That is collapsed immediately to four
per-frame band curves. Then, per layer:

1. **Rolling minimum** on the tonal bands. Drums and synths share
   800–6000 Hz, so the raw band lights up when the snare enters rather than
   when the synth does. Taking the quietest value in each 0.4 s window
   removes the drums — bursts collapse to near zero between hits — while
   sustained tone survives untouched. On the validation track this is the
   difference between dating the synth at 0:03 (wrong) and 0:12 (right).
2. **Envelope follower**, instant attack and slow release, so a part played
   as separate hits does not flicker off between notes.
3. **Presence** judged against that layer's own 5th–95th percentile range.
   Not an absolute threshold, which cannot work across recordings of
   different loudness; and not a share of the frame, which is zero-sum — a
   bass playing throughout would appear to "leave" the moment a synth enters
   and takes some of the share.
4. **Sustain, bridge, then filter**: a layer must hold for 1.5 s to count as
   an entrance, gaps under 4 s are bridged, and only then are short spans
   dropped. That order matters — a pulsing layer crosses the threshold in
   many short bursts, and filtering first throws every one of them away.

Validated against a real track whose entry times were supplied by ear
(bass 0:00, kick+snare 0:03, synth 0:12): detected 0.3 s / 3.2 s / 12.0 s.

## Why there are no instrument names

An earlier version used YAMNet (an AudioSet tagger) to name instruments. It
does not work on this material, for a structural reason rather than a
tunable one: **YAMNet's input front-end spans only 125 Hz – 7500 Hz**, so an
electric bass (fundamentals ~41–200 Hz) and a kick drum sit largely below its
lowest input bin.

Measured on a real track containing an electric bass from the start, kick and
snare from 0:03, and a synth from 0:12:

```
Bass guitar   0.0002     (bass plainly audible throughout)
Drum kit      0.0009     (drum machine plainly audible)
Drum machine  0.0061
Singing       0.0000     (correct — there are no vocals)
Synthesizer   0.106      (correct)
```

On a second cut it also reported an acoustic guitar that is not in the
recording at all. Larger AudioSet models (PANNs, AST) share the same mel
floor, so a bigger model is not the fix.

So the app reports register and sustain, which are directly measurable,
rather than guessing a name. It says "Bass / low end", not "electric bass" —
it genuinely cannot tell an electric bass from a synth bass, and a wrong
instrument name is worse than an honest register.

`instruments.py` is still in the repo, with its tests, unused by the app.

## Deployment

Render, free plan, auto-deploying from `main`.

- Build: `pip install -r requirements.txt`
- Start: `gunicorn -c gunicorn.conf.py --timeout 300 -b 0.0.0.0:$PORT app:app`
- `SECRET_KEY` set as an environment variable

Everything the deployment needs is in `gunicorn.conf.py` rather than in
start-command flags, so a platform running a bare `gunicorn app:app` still
binds correctly.

**Two failure modes worth knowing**, both of which have bitten this service:

1. **A CLI flag silently overrides the config file.** A stale `-w 2` in the
   dashboard's start command kept the service on two workers for a full
   deploy cycle after `gunicorn.conf.py` said one, and cost a real OOM kill.
   `gunicorn.conf.py` now prints the *effective* worker count at boot so a
   mismatch is never silent.
2. **The build cache can overwrite a fresh checkout.** A deploy reported the
   right commit as live while serving the previous build, because the 149 MB
   cache finished extracting after the git checkout. Symptom: the page does
   not change despite a successful deploy. Fix: redeploy with "Clear build
   cache" rather than pushing again.

## Memory, and why the rest of the app is switched off

The free plan gives 512 MB. Measured, fully warmed, with no request in
flight:

| Stack | Resident |
| --- | --- |
| flask + numpy + soundfile + soxr (what runs now) | **~65 MB** |
| the previous librosa pipeline | **~503 MB** |

The breakdown for the second: `scipy.signal` costs +68 MB on import,
numba/LLVM +55 MB before compiling anything, and the first JIT compile a
further +74 MB. That left essentially nothing for an actual request, which
is what caused repeated OOM crashes.

So these modules are present and tested but **not wired into `app.py`**:

| Module | Feature |
| --- | --- |
| `analysis.py` | key, church mode, tempo, mood, library similarity |
| `chords.py` | chord estimation by template matching |
| `song_map.py` | frequency-band track map and measure grid |
| `structure.py` | intro / verse / chorus segmentation |
| `instruments.py` | YAMNet instrument naming |

They all still pass their tests and can be re-enabled on a host with more
memory — Render Standard (2 GB, $25/mo) or equivalent.

## Limitations

- Register, not instruments. See above.
- No vocals layer. Worth noting YAMNet *is* dependable at detecting singing,
  so a narrow hybrid using it for voice alone is a reasonable addition.
- Validated against one track. Other material has not been checked against
  ground truth.
- Uploads are capped at 40 MB (`MAX_CONTENT_LENGTH` in `app.py`).
- The free instance sleeps after 15 minutes idle; the upload page shows a
  progress bar and an elapsed timer, which also covers that cold start.
