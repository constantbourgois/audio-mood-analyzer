"""Gunicorn hooks.

librosa relies on numba's on-disk JIT cache. Two worker processes racing to
write that cache concurrently (their first compile after a fresh deploy, with
no cache yet on disk) can corrupt the cache files and crash a worker with
SIGSEGV. Giving each worker its own private cache directory removes any
concurrent access to the same files.
"""
import os


def post_fork(server, worker):
    cache_dir = f"/tmp/numba-cache-{worker.pid}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
