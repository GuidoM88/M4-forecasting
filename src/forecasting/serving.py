"""MLflow pyfunc wrapper bundling a trained TiDE model with its conformal calibration."""

from __future__ import annotations

import mlflow

from forecasting.config import settings
from forecasting.conformal.calibration import ConformalCalibrator
from forecasting.models.tide_model import TiDEForecaster


class ForecasterBundle(mlflow.pyfunc.PythonModel):
    """Not served through the generic pyfunc predict(df) interface;
    callers unwrap this and use .forecaster / .calibrator directly."""

    def load_context(self, context) -> None:
        self.forecaster = TiDEForecaster.load(f"{context.artifacts['model_dir']}/tide_model.pt")
        self.calibrator = ConformalCalibrator.load(
            context.artifacts["calibration_file"], settings.coverage_level, settings.critical_ratio
        )

    def predict(self, context, model_input, params=None):
        raise NotImplementedError("use unwrap_python_model().forecaster / .calibrator instead")