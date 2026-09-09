"""Loads the champion model+calibration bundle and serves forecasts."""

from __future__ import annotations

import logging

import mlflow
import numpy as np
from mlflow import MlflowClient

from api.schemas import ForecastRequest, ForecastResponse
from forecasting.config import settings

logger = logging.getLogger(__name__)


class Predictor:
    """Holds a loaded model+calibration bundle in memory. Instantiate once at app startup."""

    def __init__(self, bundle, model_version: str) -> None:
        self._forecaster = bundle.forecaster
        self._calibrator = bundle.calibrator
        self.model_version = model_version

    @classmethod
    def load(cls) -> "Predictor":
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()

        model_uri = f"models:/{settings.mlflow_registered_model_name}@champion"
        loaded = mlflow.pyfunc.load_model(model_uri)
        bundle = loaded.unwrap_python_model()

        version = client.get_model_version_by_alias(settings.mlflow_registered_model_name, "champion").version
        logger.info("loaded champion model version %s", version)
        return cls(bundle, version)

    def predict(self, request: ForecastRequest) -> ForecastResponse:
        if len(request.history) < settings.input_chunk_length:
            raise ValueError(
                f"history too short: got {len(request.history)} points, "
                f"need at least {settings.input_chunk_length}"
            )

        context = np.asarray(request.history[-settings.context_length :], dtype=float)
        point_forecast = self._forecaster.predict(context, settings.forecast_length)
        conformal_forecast = self._calibrator.apply(point_forecast)

        return ForecastResponse(
            series_id=request.series_id,
            point_forecast=point_forecast.tolist(),
            conformal_forecast=conformal_forecast.tolist(),
            horizon=settings.forecast_length,
            model_version=self.model_version,
        )