import io
import os

import pytest

from app import LIBRARY_DIR
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True, SECRET_KEY="test-secret")
    with flask_app.test_client() as c:
        yield c


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Analyseur audio" in resp.data


def test_analyze_without_file_redirects_with_flash(client):
    resp = client.post("/analyze", data={}, content_type="multipart/form-data", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Choisis un fichier audio" in resp.data


def test_analyze_rejects_disallowed_extension(client):
    data = {"audio_file": (io.BytesIO(b"not audio"), "notes.txt")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert resp.status_code == 200
    assert "Format non support".encode() in resp.data


def test_analyze_valid_wav_returns_result_page(client, sample_wav_bytes):
    data = {"audio_file": (sample_wav_bytes, "pytest_sample.wav")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert "mood-banner" in body
    assert "Carte de la chanson" in body


def test_analyze_add_to_library_persists_file(client, sample_wav_bytes):
    filename = "pytest_library_test.wav"
    saved_path = os.path.join(LIBRARY_DIR, filename)
    data = {
        "audio_file": (sample_wav_bytes, filename),
        "add_to_library": "on",
    }
    try:
        resp = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert os.path.exists(saved_path)
    finally:
        if os.path.exists(saved_path):
            os.remove(saved_path)
