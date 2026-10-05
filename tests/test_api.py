"""HTTP contract checks; model accuracy and temporal safety live in model tests."""

import pytest
from fastapi.testclient import TestClient

from bitcoin import forecast
from bitcoin.api import app


@pytest.fixture
def artifact():
    return {
        "metadata": {"target_month": "2026-11", "symbol": "BTC/USD"},
        "target_month": "2026-11",
        "metrics": {"model": {"mae_usd": 2500.0}},
    }


@pytest.fixture
def client(monkeypatch, artifact):
    monkeypatch.setattr(forecast, "load_artifact", lambda path: artifact)

    def predict(model, target_month=None):
        expected = model["target_month"]
        if target_month is not None and target_month not in {"2026-10", expected}:
            raise ValueError(f"Mês não suportado. Escolha 2026-10 ou {expected}.")
        return {"target_month": target_month or expected, "predicted_close_usd": 110000.0}

    monkeypatch.setattr(forecast, "predict", predict)
    monkeypatch.setenv("MODEL_PATH", "unused.joblib")
    with TestClient(app) as client:
        yield client


def test_ready_and_prediction(client, artifact):
    assert client.get("/health").json() == {
        "status": "ok", "model_loaded": True, "target_month": "2026-11"
    }
    response = client.get("/predict")
    assert response.status_code == 200
    assert response.json()["target_month"] == "2026-11"
    assert client.get("/predict", params={"target_month": "2026-11"}).json() == response.json()
    assert client.get("/metrics").json() == artifact["metrics"]
    assert client.get("/model").json() == artifact["metadata"]


def test_explicit_supported_alternative_month(client):
    response = client.get("/predict", params={"target_month": "2026-10"})
    assert response.status_code == 200
    assert response.json()["target_month"] == "2026-10"


@pytest.mark.parametrize("month", ["2026-13", "2026-00", "26-11", "2026-1", "novembro", "2026-11-01"])
def test_invalid_month_format(client, month):
    assert client.get("/predict", params={"target_month": month}).status_code == 422


def test_month_outside_model_horizon(client):
    response = client.get("/predict", params={"target_month": "2027-01"})
    assert response.status_code == 422
    assert "2026-11" in response.json()["detail"]


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), ValueError("invalid artifact")])
def test_unavailable_artifact_returns_503(monkeypatch, error):
    def fail_loading(path):
        raise error

    monkeypatch.setattr(forecast, "load_artifact", fail_loading)
    monkeypatch.setenv("MODEL_PATH", "missing.joblib")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        for endpoint in ["/predict", "/metrics", "/model"]:
            response = client.get(endpoint)
            assert response.status_code == 503
            assert response.json()["detail"]


def test_artifact_is_loaded_only_once(monkeypatch, artifact):
    loaded_paths = []

    def load(path):
        loaded_paths.append(path)
        return artifact

    monkeypatch.setattr(forecast, "load_artifact", load)
    monkeypatch.setenv("MODEL_PATH", "trusted.joblib")
    with TestClient(app) as client:
        client.get("/health")
        client.get("/model")
        client.get("/metrics")
    assert len(loaded_paths) == 1
    assert loaded_paths[0].name == "trusted.joblib"
