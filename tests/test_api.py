"""Tests for the FastAPI forecasting endpoints, with the model bundle mocked out."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import Predictor
from api.schemas import ForecastRequest, ForecastResponse


class FakePredictor:
    model_version = "1"

    def predict(self, request: ForecastRequest) -> ForecastResponse:
        return ForecastResponse(
            series_id=request.series_id,
            point_forecast=[1.0] * 14,
            conformal_forecast=[1.5] * 14,
            horizon=14,
            model_version=self.model_version,
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(Predictor, "load", classmethod(lambda cls: FakePredictor()))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "1"}


def test_forecast_success(client):
    response = client.post(
        "/forecast",
        json={"series_id": "D1", "history": list(np.arange(100, dtype=float))},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["series_id"] == "D1"
    assert len(body["point_forecast"]) == 14
    assert len(body["conformal_forecast"]) == 14


def test_forecast_history_too_short(client, monkeypatch):
    def raise_short_history(self, request):
        raise ValueError("history too short")

    monkeypatch.setattr(FakePredictor, "predict", raise_short_history)
    response = client.post("/forecast", json={"series_id": "D1", "history": [1.0, 2.0]})
    assert response.status_code == 400