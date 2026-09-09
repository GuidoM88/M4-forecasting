"""Tests for forecasting.evaluation.metrics."""

import numpy as np
import pytest

from forecasting.evaluation.metrics import coverage, mase, newsvendor_cost, smape


def test_smape_perfect_forecast_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert smape(y, y) == pytest.approx(0.0)


def test_smape_symmetric():
    assert smape([10.0], [20.0]) == pytest.approx(smape([20.0], [10.0]))


def test_smape_handles_zero_denominator():
    assert smape([0.0], [0.0]) == pytest.approx(0.0)


def test_mase_naive_forecast_equals_one():
    y_train = np.arange(1.0, 21.0)
    y_true = np.array([21.0, 22.0])
    naive_pred = np.array([20.0, 21.0])
    assert mase(y_true, naive_pred, y_train, seasonality=1) == pytest.approx(1.0)


def test_mase_returns_nan_for_degenerate_train():
    assert np.isnan(mase([1.0], [1.0], [5.0], seasonality=1))


def test_newsvendor_cost_penalizes_shortage_and_surplus_separately():
    y_true = np.array([10.0])
    assert newsvendor_cost(y_true, [8.0], cost_underage=2.0, cost_overage=1.0) == pytest.approx(4.0)
    assert newsvendor_cost(y_true, [12.0], cost_underage=2.0, cost_overage=1.0) == pytest.approx(2.0)


def test_coverage_flags_within_margin():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([9.0, 15.0])
    q_abs = np.array([2.0, 2.0])
    assert coverage(y_true, y_pred, q_abs).tolist() == [True, False]