"""Evaluate the champion model + calibration bundle on the canonical test split."""

from __future__ import annotations

import logging

import mlflow
from mlflow import MlflowClient

from forecasting.config import settings
from forecasting.data.loader import M4Dataset
from forecasting.pipeline import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    client = MlflowClient()

    dataset = M4Dataset()

    loaded = mlflow.pyfunc.load_model(f"models:/{settings.mlflow_registered_model_name}@champion")
    bundle = loaded.unwrap_python_model()
    version = client.get_model_version_by_alias(settings.mlflow_registered_model_name, "champion").version

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("evaluated_model_version", version)
        metrics = evaluate_model(dataset, bundle.forecaster, bundle.calibrator)
        mlflow.log_metrics(metrics)
        logger.info("evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()