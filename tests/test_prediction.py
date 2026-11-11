import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import build_dataset
from src.prediction import predict as predict_mod
from src.prediction.evaluate import (
    compare_frames,
    directional_accuracy,
    evaluate_predictions,
    mae,
    mape,
    residual_interval_std,
    rmse,
)
from src.prediction.train import chronological_split, train_ticker, walk_forward
from tests.test_features import synth_prices


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(synth_prices(500, seed=11))


# ---------- metrics ----------

def test_mae_rmse_known_values():
    assert mae([1, 2, 3], [1.5, 2.5, 3.5]) == pytest.approx(0.5)
    assert rmse([1, 2, 3], [1.5, 2.5, 3.5]) == pytest.approx(0.5)


def test_mape_and_zero_guard():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert mape([0, 200], [1, 180]) is None


def test_directional_accuracy_known():
    prev = [100, 100, 100, 100]
    truth = [110, 90, 110, 90]
    pred = [105, 95, 95, 95]
    assert directional_accuracy(truth, pred, prev) == pytest.approx(0.75)


def test_evaluate_predictions_keys():
    out = evaluate_predictions([100, 101], [101, 100], [99, 100])
    for key in ("mae", "rmse", "mape_pct", "directional_accuracy", "n"):
        assert key in out


def test_residual_interval_std():
    # residuals: -0.1, -0.2, +0.1 -> sample std ~= 0.1528
    assert residual_interval_std([1, 2, 3], [1.1, 2.2, 2.9]) == pytest.approx(0.1528, abs=0.001)
    # constant residuals -> zero spread
    assert residual_interval_std([1, 2, 3], [1.1, 2.1, 3.1]) == pytest.approx(0.0)


def test_compare_frames_sorted_by_rmse():
    df = compare_frames({"a": {"rmse": 2.0}, "b": {"rmse": 1.0}})
    assert df["model"].tolist() == ["b", "a"]


# ---------- splitting ----------

def test_chronological_split_order_and_sizes():
    tr, va, te = chronological_split(1000)
    assert tr[-1] < va[0] < va[-1] < te[0]
    assert len(tr) == 700 and len(va) == 150 and len(te) == 150


# ---------- training ----------

def test_train_ticker_produces_artifacts_and_meta(dataset, tmp_path):
    meta = train_ticker("0700.HK", dataset, models_dir=tmp_path)
    assert meta is not None
    assert (tmp_path / "0700_HK" / "best_model.joblib").exists()
    assert (tmp_path / "0700_HK" / "meta.json").exists()
    assert (tmp_path / "0700_HK" / "model_comparison.csv").exists()
    assert meta["selected_model"] in ("linear_regression", "random_forest", "xgboost")
    assert meta["test"]["rmse"] > 0
    assert 0.0 <= meta["test"]["directional_accuracy"] <= 1.0
    # 6 candidates (3 models x 2 feature sets)
    assert len(meta["validation"]) == 6


def test_train_ticker_skips_small_data(tmp_path):
    small = build_dataset(synth_prices(60, seed=5))
    assert train_ticker("0700.HK", small, models_dir=tmp_path) is None


def test_walk_forward_runs(dataset):
    wf = walk_forward(dataset, ["close_prev", "ret_1d", "rsi_14"], "linear_regression", n_folds=4)
    assert "mae" in wf and wf.get("folds") == 4


def test_baseline_beats_or_loses_honestly_recorded(dataset, tmp_path):
    meta = train_ticker("0700.HK", dataset, models_dir=tmp_path)
    # both numbers exist and are finite — no fabrication either way
    assert np.isfinite(meta["test"]["rmse"])
    assert np.isfinite(meta["baseline_test"]["rmse"])


# ---------- prediction ----------

def test_predict_ticker_roundtrip(dataset, tmp_path):
    train_ticker("0700.HK", dataset, models_dir=tmp_path)
    result = predict_mod.predict_ticker("0700.HK", dataset=dataset, models_dir=tmp_path)
    assert result is not None
    assert result["predicted_next_close"] > 0
    expected_pct = 100 * (result["predicted_next_close"] / result["last_close"] - 1)
    assert result["predicted_pct_change"] == pytest.approx(expected_pct, abs=0.01)
    assert result["interval_95_low"] <= result["predicted_next_close"] <= result["interval_95_high"]
    assert result["as_of_date"] == str(dataset["date"].iloc[-1])


def test_predict_ticker_without_model_returns_none(dataset, tmp_path):
    assert predict_mod.predict_ticker("0700.HK", dataset=dataset, models_dir=tmp_path) is None
