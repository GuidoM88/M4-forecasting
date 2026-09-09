"""Split conformal calibration for multi-horizon point forecasts."""

from __future__ import annotations

import numpy as np


def conformal_quantile(residuals: np.ndarray, tau: float) -> float:
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    q_level = min(np.ceil((n + 1) * tau) / n, 1.0)
    return float(np.quantile(r, q_level, method="higher"))


class ConformalCalibrator:
    """Per-horizon conformal quantiles fitted on calibration residuals.

    residuals passed to `fit` must have shape (n_calib_series, horizon)
    and be signed (y_true - y_pred).
    """

    def __init__(self, coverage_level: float, critical_ratio: float) -> None:
        self.coverage_level = coverage_level
        self.critical_ratio = critical_ratio
        self.q_abs: np.ndarray | None = None
        self.q_signed: np.ndarray | None = None

    def fit(self, residuals: np.ndarray) -> "ConformalCalibrator":
        residuals = np.asarray(residuals, dtype=float)
        horizon = residuals.shape[1]
        self.q_abs = np.array(
            [conformal_quantile(np.abs(residuals[:, h]), self.coverage_level) for h in range(horizon)]
        )
        self.q_signed = np.array(
            [conformal_quantile(residuals[:, h], self.critical_ratio) for h in range(horizon)]
        )
        return self

    def apply(self, point_forecast: np.ndarray) -> np.ndarray:
        if self.q_signed is None:
            raise RuntimeError("call fit() before apply()")
        return np.asarray(point_forecast, dtype=float) + self.q_signed

    def is_covered(self, y_true: np.ndarray, point_forecast: np.ndarray) -> np.ndarray:
        if self.q_abs is None:
            raise RuntimeError("call fit() before is_covered()")
        y_true, point_forecast = np.asarray(y_true, dtype=float), np.asarray(point_forecast, dtype=float)
        return np.abs(y_true - point_forecast) <= self.q_abs

    def save(self, path: str) -> None:
        np.savez(path, q_abs=self.q_abs, q_signed=self.q_signed)

    @classmethod
    def load(cls, path: str, coverage_level: float, critical_ratio: float) -> "ConformalCalibrator":
        data = np.load(path)
        calibrator = cls(coverage_level, critical_ratio)
        calibrator.q_abs = data["q_abs"]
        calibrator.q_signed = data["q_signed"]
        return calibrator