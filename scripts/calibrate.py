"""Calibrate conformal quantiles for the latest training run and register the bundle."""

from __future__ import annotations

import logging

import mlflow
from mlflow import MlflowClient

from forecasting.config import settings
from forecasting.data.loader import M4Dataset
from forecasting.models.tide_model import TiDEForecaster
from forecasting.pipeline import calibrate_model
from forecasting.serving import ForecasterBundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def latest_train_run(client: MlflowClient) -> str:
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    runs = client.search_runs(
        experiment.experiment_id,
        filter_string="tags.mlflow.runName = 'train'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("no training run found; run scripts/train.py first")
    return runs[0].info.run_id


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    client = MlflowClient()

    dataset = M4Dataset()

    train_run_id = latest_train_run(client)
    model_dir = mlflow.artifacts.download_artifacts(f"runs:/{train_run_id}/model")
    model = TiDEForecaster.load(f"{model_dir}/tide_model.pt")
    
    with mlflow.start_run(run_name="calibrate"):
        mlflow.log_param("trained_from_run_id", train_run_id)

        calibrator = calibrate_model(dataset, model)
        calibrator.save("calibration.npz")
        
        logged_model = mlflow.pyfunc.log_model(
            name="bundle",
            python_model=ForecasterBundle(),
            artifacts={"model_dir": model_dir, "calibration_file": "calibration.npz"},
        )

        result = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=settings.mlflow_registered_model_name,
        )

    client.set_registered_model_alias(
        name=settings.mlflow_registered_model_name,
        alias="champion",
        version=result.version,
    )
    logger.info("bundle version %s registered and promoted to 'champion'", result.version)


if __name__ == "__main__":
    main()