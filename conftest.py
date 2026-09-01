import io

import numpy as np
import pytest
import soundfile as sf

SR = 22050


def _make_tone(duration=2.0, freq=220.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t) ** 2
    y = (0.2 * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)
    return y, sr


@pytest.fixture
def sample_audio():
    """A short synthetic tone (y, sr), fast enough for the full analysis pipeline."""
    return _make_tone()


@pytest.fixture
def sample_wav_bytes():
    """The same tone, encoded as an in-memory WAV file for upload tests."""
    y, sr = _make_tone()
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf
