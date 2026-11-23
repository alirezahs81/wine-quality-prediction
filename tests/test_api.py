"""API tests — require a trained model (provided by CI train step)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from src.config import MODEL_FILE, SCALER_FILE


pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)),
    reason="Model artifacts missing; run scripts/train_ci.py first",
)


@pytest.fixture
def client():
    from src.api import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_predict_red(client):
    payload = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "wine_type": "red",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 3 <= body["predicted_quality"] <= 9
    assert body["quality_category"] in {"Excellent", "Good", "Average", "Below Average"}


def test_predict_invalid_type(client):
    payload = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "wine_type": "rose",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422  # pydantic validation
