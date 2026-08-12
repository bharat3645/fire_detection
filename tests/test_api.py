"""Tests for the FastAPI REST interface (api.py)."""

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _image_bytes(size=(64, 64), color=(220, 40, 10), fmt="PNG"):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tiny_model_path):
    os.environ["FIRE_MODEL_PATH"] = str(tiny_model_path)
    os.environ["FIRE_DETECTION_LOG_PATH"] = str(tiny_model_path.parent / "predictions.jsonl")

    import api  # imported here so env vars are set before the app object is used

    with TestClient(api.app) as test_client:
        yield test_client

    os.environ.pop("FIRE_MODEL_PATH", None)
    os.environ.pop("FIRE_DETECTION_LOG_PATH", None)


def test_root_lists_endpoints(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["predict"] == "/predict"


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["error"] is None


def test_predict_returns_valid_payload(client):
    resp = client.post("/predict", files={"file": ("test.png", _image_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "test.png"
    assert body["prediction"] in ("Fire", "No Fire")
    assert body["predicted_class"] in (0, 1)
    assert 0.0 <= body["fire_probability"] <= 1.0
    assert body["threshold"] == 0.5


def test_predict_respects_threshold_query_param(client):
    # threshold=0.0 forces every prediction to the positive ("Fire") class.
    resp = client.post("/predict?threshold=0.0", files={"file": ("test.png", _image_bytes(), "image/png")})
    assert resp.status_code == 200
    assert resp.json()["prediction"] == "Fire"


def test_predict_rejects_out_of_range_threshold(client):
    resp = client.post("/predict?threshold=1.5", files={"file": ("test.png", _image_bytes(), "image/png")})
    assert resp.status_code == 422  # FastAPI query validation


def test_predict_rejects_empty_file(client):
    resp = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400


def test_predict_rejects_garbage_bytes(client):
    resp = client.post("/predict", files={"file": ("bad.png", b"not an image", "image/png")})
    assert resp.status_code == 400


def test_predict_logs_to_audit_log(client, tiny_model_path):
    log_path = tiny_model_path.parent / "predictions.jsonl"
    log_path.unlink(missing_ok=True)

    resp = client.post("/predict", files={"file": ("audit.png", _image_bytes(), "image/png")})
    assert resp.status_code == 200
    assert log_path.exists()
    assert "audit.png" in log_path.read_text(encoding="utf-8")


def test_docs_are_served(client):
    # Auto-generated OpenAPI docs double as API documentation for this project.
    resp = client.get("/docs")
    assert resp.status_code == 200
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "/predict" in resp.json()["paths"]
