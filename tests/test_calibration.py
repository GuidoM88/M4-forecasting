"""Tests for forecasting.conformal.calibration."""

import numpy as np
import pytest

from forecasting.conformal.calibration import ConformalCalibrator, conformal_quantile


def test_conformal_quantile_matches_finite_sample_correction():
    residuals = np.arange(1, 10)  # [1..9]
    assert conformal_quantile(residuals, tau=0.667) == pytest.approx(8.0)


def test_conformal_quantile_caps_at_max():
    residuals = np.array([1.0, 2.0, 3.0])
    assert conformal_quantile(residuals, tau=0.99) == pytest.approx(3.0)


def test_calibrator_apply_shifts_by_signed_quantile_per_horizon():
    residuals = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    calibrator = ConformalCalibrator(coverage_level=0.9, critical_ratio=0.667).fit(residuals)

    point_forecast = np.array([100.0, 100.0])
    adjusted = calibrator.apply(point_forecast)

    assert adjusted[0] > point_forecast[0]  # positive residuals push h1 up
    assert adjusted[1] < point_forecast[1]  # negative residuals push h2 down


def test_calibrator_is_covered_uses_absolute_quantile():
    residuals = np.array([[1.0], [2.0], [3.0]])
    calibrator = ConformalCalibrator(coverage_level=0.9, critical_ratio=0.667).fit(residuals)

    y_true = np.array([100.0])
    within_margin = y_true - calibrator.q_abs + 0.01
    outside_margin = y_true + calibrator.q_abs + 10.0

    assert calibrator.is_covered(y_true, within_margin)[0]
    assert not calibrator.is_covered(y_true, outside_margin)[0]


def test_calibrator_save_and_load_roundtrip(tmp_path):
    residuals = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    calibrator = ConformalCalibrator(coverage_level=0.9, critical_ratio=0.667).fit(residuals)

    path = tmp_path / "calibration.npz"
    calibrator.save(str(path))
    loaded = ConformalCalibrator.load(str(path), coverage_level=0.9, critical_ratio=0.667)

    np.testing.assert_array_equal(loaded.q_abs, calibrator.q_abs)
    np.testing.assert_array_equal(loaded.q_signed, calibrator.q_signed)


def test_apply_before_fit_raises():
    calibrator = ConformalCalibrator(coverage_level=0.9, critical_ratio=0.667)
    with pytest.raises(RuntimeError):
        calibrator.apply(np.array([1.0]))