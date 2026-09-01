"""Gunicorn config.

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

preload_app = True

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache-master")


def post_fork(server, worker):
    cache_dir = f"/tmp/numba-cache-{worker.pid}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
