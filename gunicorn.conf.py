"""Gunicorn config.

Gunicorn loads this file automatically when it sits in the working directory
under this name, with no -c flag needed. Everything the deployment needs is
therefore set here rather than as start-command flags: a platform that runs a
bare `gunicorn app:app` (as Render's default start command does) would
otherwise fall back to binding 127.0.0.1:8000 — unreachable, so the platform's
port scan finds nothing — and to the default 30s timeout.

librosa relies on numba's on-disk JIT cache, and compiling functions like
pyin's Viterbi decoder or librosa.load's own guvectorize is itself slow —
slow enough on a throttled CPU to exceed a request timeout by itself.

preload_app forces the app (and app.py's numba warm-up call) to import once,
single-threaded, in the master process before any worker forks. Forked
workers then inherit the already-compiled code via copy-on-write memory
instead of each re-compiling it on their own first request — which was both
the source of WORKER TIMEOUT crashes and, when two workers compiled
concurrently, capable of corrupting the shared on-disk cache and crashing a
worker with SIGSEGV.

Each worker still gets its own private cache directory post-fork, as a
safety net for any signature the warm-up doesn't happen to cover.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Two workers on purpose, overriding the WEB_CONCURRENCY=1 that Render sets
# from the instance's CPU count: an analysis occupies its worker for tens of
# seconds, and the second one keeps the page responsive meanwhile. preload_app
# means they share the interpreter and compiled code copy-on-write, so the
# second worker costs little beyond its own per-request arrays.
workers = 2

# A few minutes of audio takes far longer than the 30s default to analyse.
timeout = 300

preload_app = True

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache-master")


def post_fork(server, worker):
    cache_dir = f"/tmp/numba-cache-{worker.pid}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
