"""Forecast accuracy and business-cost metrics."""

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1e-8, denom)
    return 100.0 * np.mean(diff)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
    y_true, y_pred, y_train = (
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
        np.asarray(y_train, dtype=float),
    )
    if len(y_train) <= seasonality:
        return np.nan
    naive_mae = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if naive_mae == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / naive_mae


def newsvendor_cost(y_true: np.ndarray, order: np.ndarray, cost_underage: float, cost_overage: float) -> float:
    y_true, order = np.asarray(y_true, dtype=float), np.asarray(order, dtype=float)
    shortage = np.maximum(y_true - order, 0)
    surplus = np.maximum(order - y_true, 0)
    return float(np.sum(cost_underage * shortage + cost_overage * surplus))


def coverage(y_true: np.ndarray, y_pred: np.ndarray, q_abs: np.ndarray) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return np.abs(y_true - y_pred) <= q_abs