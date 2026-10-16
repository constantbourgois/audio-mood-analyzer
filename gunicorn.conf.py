"""Gunicorn config.

Gunicorn loads this file automatically when it sits in the working directory
under this name, with no -c flag needed. Everything the deployment needs is
therefore set here rather than as start-command flags: a platform that runs a
bare `gunicorn app:app` (as Render's default start command does) would
otherwise fall back to binding 127.0.0.1:8000 — unreachable, so the platform's
port scan finds nothing — and to the default 30s timeout.

app.py imports nothing heavier than numpy (see app.py's own comment): layer
detection, tempo and key are all plain numpy. That stack measured at ~65 MB
fully warmed, against ~500 MB for the previous librosa-based pipeline on
this plan's 512 MiB limit - the actual cause of this app's earlier OOM
crashes. With that much headroom, two workers is worth having back for real
concurrency, at essentially no OOM risk.
"""
import os
import sys

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

workers = 2

# A sync worker is busy for the whole request, including reading the upload
# body, so a 40 MB file over a slow mobile link counts against this as much
# as the analysis does - and on this plan's 0.15 of a core the analysis
# itself runs several times slower than on a laptop. 300 s is what the live
# start command has been passing as --timeout; it now lives here instead.
timeout = 300

preload_app = True


def on_starting(server):
    # Kept from when this mattered a lot more: a -w/--workers flag on the
    # actual start command silently overrides the setting above (gunicorn
    # always lets the CLI win over the config file), which is exactly what
    # kept a previous deploy running 2 workers when this file said 1. Cheap
    # to leave in as a sanity check even now that either count is safe.
    effective = server.cfg.workers
    print(f"[gunicorn.conf.py] effective worker count: {effective}"
          + ("" if effective == workers else f" <-- expected {workers}, check the start command for a stray -w/--workers flag"),
          file=sys.stderr, flush=True)
