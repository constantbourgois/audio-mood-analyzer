"""Download the YAMNet weights used for instrument recognition.

Run once after installing dependencies (the deployment does it at build time,
see render.yaml). The file is not committed: it is 4 MB of binary that the
build can fetch reproducibly from its canonical URL.

The app works without it — instrument names simply fall back to the
band-based layers — so a failure here is reported, not fatal.
"""
import os
import sys
import urllib.request

from instruments import MODEL_PATH, MODEL_URL


def main() -> int:
    if os.path.exists(MODEL_PATH):
        print(f"model already present: {MODEL_PATH}")
        return 0
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"downloading {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        print(f"could not download the model ({exc!r}); "
              "the app will fall back to band-based layers", file=sys.stderr)
        return 0
    print(f"saved {os.path.getsize(MODEL_PATH) / 1e6:.1f} MB to {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
