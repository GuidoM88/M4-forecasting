"""Train the TiDE forecaster and log it as an MLflow run artifact."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import mlflow

from forecasting.config import settings
from forecasting.data.loader import M4Dataset
from forecasting.pipeline import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    dataset = M4Dataset()

    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params(
            {
                "input_chunk_length": settings.input_chunk_length,
                "output_chunk_length": settings.output_chunk_length,
                "n_epochs": settings.n_epochs,
                "batch_size": settings.batch_size,
                "random_state": settings.random_state,
                "train_series_sample": settings.train_series_sample,
                "max_samples_per_ts": settings.max_samples_per_ts,
            }
        )

        model = train_model(dataset)

        model_dir = Path("tmp_model_artifact")
        model_dir.mkdir(exist_ok=True)
        model.save(str(model_dir / "tide_model.pt"))
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        shutil.rmtree(model_dir)

        logger.info("training run %s complete", run.info.run_id)


if __name__ == "__main__":
    main()