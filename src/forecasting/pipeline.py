"""Training, calibration and evaluation orchestration for the TiDE forecaster."""

from __future__ import annotations

import numpy as np

from forecasting.config import settings
from forecasting.conformal.calibration import ConformalCalibrator
from forecasting.data.loader import M4Dataset
from forecasting.evaluation.metrics import mase, newsvendor_cost, smape
from forecasting.models.tide_model import TiDEForecaster


def clean_training_pool(dataset: M4Dataset) -> list[str]:
    """Series ids available for training, excluding calibration and test ids."""
    excluded = set(dataset.calib_ids) | set(dataset.test_ids)
    all_ids = dataset.train_df[dataset.train_df.columns[0]].tolist()
    return [uid for uid in all_ids if uid not in excluded]


def train_model(dataset: M4Dataset) -> TiDEForecaster:
    pool = clean_training_pool(dataset)
    rng = np.random.RandomState(settings.random_state)
    sample_ids = rng.choice(pool, size=min(settings.train_series_sample, len(pool)), replace=False)
    series = [dataset.train_values(uid) for uid in sample_ids]

    model = TiDEForecaster()
    model.fit(series, max_samples_per_ts=settings.max_samples_per_ts)
    return model


def calibrate_model(dataset: M4Dataset, model: TiDEForecaster) -> ConformalCalibrator:
    contexts, y_trues = [], []
    for uid in dataset.calib_ids:
        full = dataset.train_values(uid)
        contexts.append(full[: -settings.forecast_length][-settings.context_length :])
        y_trues.append(full[-settings.forecast_length :])

    y_preds = model.predict_batch(contexts, settings.forecast_length)
    residuals = np.array(y_trues) - np.array(y_preds)

    calibrator = ConformalCalibrator(settings.coverage_level, settings.critical_ratio)
    calibrator.fit(residuals)
    return calibrator


def evaluate_model(dataset: M4Dataset, model: TiDEForecaster, calibrator: ConformalCalibrator) -> dict[str, float]:
    contexts, y_trues, train_values_list = [], [], []
    for uid in dataset.test_ids:
        train_values = dataset.train_values(uid)
        train_values_list.append(train_values)
        contexts.append(train_values[-settings.context_length :])
        y_trues.append(dataset.test_values(uid)[: settings.forecast_length])

    y_preds = model.predict_batch(contexts, settings.forecast_length)

    smape_scores, mase_scores, coverage_flags = [], [], []
    naive_cost_total, conformal_cost_total = 0.0, 0.0

    for y_true, y_pred, train_values in zip(y_trues, y_preds, train_values_list):
        y_conformal = calibrator.apply(y_pred)
        smape_scores.append(smape(y_true, y_pred))
        mase_scores.append(mase(y_true, y_pred, train_values))
        coverage_flags.append(calibrator.is_covered(y_true, y_pred))
        naive_cost_total += newsvendor_cost(y_true, y_pred, settings.cost_underage, settings.cost_overage)
        conformal_cost_total += newsvendor_cost(y_true, y_conformal, settings.cost_underage, settings.cost_overage)

    n_obs = len(dataset.test_ids) * settings.forecast_length
    return {
        "smape": float(np.mean(smape_scores)),
        "mase": float(np.mean(mase_scores)),
        "coverage": float(np.mean(coverage_flags)),
        "cost_naive_avg": naive_cost_total / n_obs,
        "cost_conformal_avg": conformal_cost_total / n_obs,
    }