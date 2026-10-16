import io

import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True, SECRET_KEY="test-secret")
    with flask_app.test_client() as c:
        yield c


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Audio Analyser" in resp.data


def test_analyze_without_file_redirects_with_flash(client):
    resp = client.post("/analyze", data={}, content_type="multipart/form-data", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Choose an audio file" in resp.data


def test_analyze_rejects_disallowed_extension(client):
    data = {"audio_file": (io.BytesIO(b"not audio"), "notes.txt")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Unsupported format" in resp.data


def test_analyze_valid_wav_returns_result_page(client, sample_wav_bytes):
    data = {"audio_file": (sample_wav_bytes, "pytest_sample.wav")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert "Layers over time" in body
    assert "pytest_sample.wav" in body


def test_result_page_carries_player_hooks(client, sample_wav_bytes):
    # The player runs in the browser from the picked file; the page only has
    # to ship the hooks, hidden/disabled until the script attaches the audio.
    data = {"audio_file": (sample_wav_bytes, "pytest_sample.wav")}
    body = client.post("/analyze", data=data, content_type="multipart/form-data").data.decode("utf-8")
    assert "data-layer-map" in body
    assert 'data-duration="2.0"' in body
    assert "<div class=\"card player\" data-player hidden>" in body
    assert "window.attachPlayer" in body
