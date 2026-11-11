"""Evaluation metrics for next-day price prediction.

All metrics are computed on out-of-sample predictions only.
Directional accuracy requires the previous close (the reference point a
trader would compare against), passed explicitly to avoid ambiguity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float | None:
    """MAPE in %. Returns None when any actual is zero (undefined)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        return None
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def directional_accuracy(y_true, y_pred, prev_close) -> float:
    """Fraction of periods where predicted direction (vs previous close)
    matches actual direction. Flat days (zero move) count as correct only
    when the prediction is also flat-or-opposite-zero — rare in practice."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    prev = np.asarray(prev_close, dtype=float)
    actual_dir = np.sign(y_true - prev)
    pred_dir = np.sign(y_pred - prev)
    return float(np.mean(actual_dir == pred_dir))


def evaluate_predictions(y_true, y_pred, prev_close) -> dict:
    return {
        "mae": round(mae(y_true, y_pred), 6),
        "rmse": round(rmse(y_true, y_pred), 6),
        "mape_pct": None if (m := mape(y_true, y_pred)) is None else round(m, 4),
        "directional_accuracy": round(directional_accuracy(y_true, y_pred, prev_close), 4),
        "n": int(len(np.asarray(y_true))),
    }


def residual_interval_std(y_true, y_pred) -> float:
    """Std of out-of-sample residuals — used for an approximate 95%
    prediction interval (normal approximation; documented as such)."""
    return float(np.std(np.asarray(y_true) - np.asarray(y_pred), ddof=1))


def compare_frames(results: dict[str, dict]) -> pd.DataFrame:
    """Model-name -> metrics dict  ==>  comparison dataframe sorted by RMSE."""
    df = pd.DataFrame(results).T
    df.index.name = "model"
    if "rmse" in df.columns:
        df = df.sort_values("rmse")
    return df.reset_index()
