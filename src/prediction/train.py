"""Model training with chronological validation.

Per ticker:
1. Chronological split: train (70%) / validation (15%) / test (15%).
2. Fit LinearRegression (scaled), RandomForest, XGBoost — with BOTH a
   price-only feature set and a price+sentiment feature set. The honest
   reason: with only ~10 days of social data, sentiment coverage is a
   handful of sessions, so its contribution must be *measured*, not
   assumed. The variant comparison is recorded in the model metadata.
3. Select the best (model, feature-set) by validation RMSE.
4. Walk-forward re-validation of the winner (4 expanding folds).
5. Report the winner on the unseen test set ONCE, against the
   previous-close baseline.
6. Refit on all labeled rows and persist model + metadata with joblib.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config.settings import MODELS_DIR, Universe, load_universe
from src.features.feature_engineering import (
    PRICE_FEATURES,
    FEATURE_COLS,
    TARGET_COL,
    build_dataset,
)
from src.prediction.evaluate import compare_frames, evaluate_predictions, residual_interval_std
from src.utils.logger import get_logger
from src.utils.storage import load_from_sqlite

log = get_logger(__name__)

FEATURE_SETS = {"price_only": PRICE_FEATURES, "with_sentiment": FEATURE_COLS}
MIN_ROWS = 120
SEED = 42


def make_models(seed: int = SEED) -> dict:
    return {
        "linear_regression": Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=seed, n_jobs=-1, verbosity=0,
        ),
    }


def chronological_split(n: int, train_frac: float = 0.70, val_frac: float = 0.15):
    n_train = int(n * train_frac)
    n_val = max(1, int(n * val_frac))
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)
    return train_idx, val_idx, test_idx


def load_datasets(universe: Universe | None = None) -> dict[str, pd.DataFrame]:
    """Build per-ticker modeling datasets from the local store."""
    universe = universe or load_universe()
    prices = load_from_sqlite("prices")
    posts = load_from_sqlite("posts_annotated")
    news = load_from_sqlite("news_annotated")
    datasets = {}
    for asset in universe.assets:
        p = prices[prices["ticker"] == asset.ticker] if not prices.empty else pd.DataFrame()
        if p.empty:
            log.warning("no stored prices for %s — run `python main.py --mode collect` first", asset.ticker)
            continue
        s = posts[posts["ticker"] == asset.ticker] if not posts.empty else None
        n = news[news["ticker"] == asset.ticker] if not news.empty else None
        datasets[asset.ticker] = build_dataset(p, s, n)
    return datasets


def walk_forward(dataset: pd.DataFrame, feature_set: list[str], model_name: str, n_folds: int = 4) -> dict:
    labeled = dataset[dataset[TARGET_COL].notna()].reset_index(drop=True)
    n = len(labeled)
    if n < MIN_ROWS or n_folds < 2:
        return {"error": "insufficient data for walk-forward"}
    block = n // (n_folds + 1)
    preds, trues, prevs = [], [], []
    for fold in range(1, n_folds + 1):
        train_end = block * fold
        val_end = min(block * (fold + 1), n)
        if val_end - train_end < 5:
            continue
        model = make_models()[model_name]
        tr = labeled.iloc[:train_end]
        va = labeled.iloc[train_end:val_end]
        model.fit(tr[feature_set], tr[TARGET_COL])
        preds.extend(model.predict(va[feature_set]))
        trues.extend(va[TARGET_COL])
        prevs.extend(va["close"])
    if not preds:
        return {"error": "no walk-forward folds produced predictions"}
    metrics = evaluate_predictions(trues, preds, prevs)
    metrics["folds"] = n_folds
    return metrics


def train_ticker(ticker: str, dataset: pd.DataFrame, models_dir: Path | None = None) -> dict | None:
    models_dir = models_dir or MODELS_DIR
    labeled = dataset[dataset[TARGET_COL].notna()].reset_index(drop=True)
    if len(labeled) < MIN_ROWS:
        log.warning("%s: only %d labeled rows (<%d) — skipping training", ticker, len(labeled), MIN_ROWS)
        return None

    train_idx, val_idx, test_idx = chronological_split(len(labeled))
    candidates: dict[str, dict] = {}
    for set_name, feature_set in FEATURE_SETS.items():
        for model_name, model in make_models().items():
            model.fit(labeled.iloc[train_idx][feature_set], labeled.iloc[train_idx][TARGET_COL])
            val_pred = model.predict(labeled.iloc[val_idx][feature_set])
            metrics = evaluate_predictions(
                labeled.iloc[val_idx][TARGET_COL], val_pred, labeled.iloc[val_idx]["close"]
            )
            candidates[f"{model_name}|{set_name}"] = {
                "model": model_name,
                "feature_set": set_name,
                "val": metrics,
            }

    # previous-close baseline on validation for context
    baseline_val = evaluate_predictions(
        labeled.iloc[val_idx][TARGET_COL],
        labeled.iloc[val_idx]["close"],
        labeled.iloc[val_idx]["close_prev"],
    )

    best_key = min(candidates, key=lambda k: candidates[k]["val"]["rmse"])
    best = candidates[best_key]
    model_name, set_name = best["model"], best["feature_set"]
    feature_set = FEATURE_SETS[set_name]

    # Refit on train+val, evaluate ONCE on the unseen test block
    refit_idx = np.concatenate([train_idx, val_idx])
    model = make_models()[model_name]
    model.fit(labeled.iloc[refit_idx][feature_set], labeled.iloc[refit_idx][TARGET_COL])
    test_pred = model.predict(labeled.iloc[test_idx][feature_set])
    test_metrics = evaluate_predictions(
        labeled.iloc[test_idx][TARGET_COL], test_pred, labeled.iloc[test_idx]["close"]
    )
    baseline_test = evaluate_predictions(
        labeled.iloc[test_idx][TARGET_COL],
        labeled.iloc[test_idx]["close"],
        labeled.iloc[test_idx]["close_prev"],
    )
    interval_std = residual_interval_std(labeled.iloc[test_idx][TARGET_COL], test_pred)

    wf_metrics = walk_forward(labeled, feature_set, model_name)

    # Final deployable model: refit on ALL labeled rows
    final_model = make_models()[model_name]
    final_model.fit(labeled[feature_set], labeled[TARGET_COL])

    artifact_dir = models_dir / ticker.replace(".", "_")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, artifact_dir / "best_model.joblib")
    meta = {
        "ticker": ticker,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": model_name,
        "selected_feature_set": set_name,
        "feature_cols": feature_set,
        "n_labeled_rows": len(labeled),
        "sentiment_rows": int(dataset["has_sentiment"].sum()),
        "val_rmse": best["val"]["rmse"],
        "validation": {k: v["val"] for k, v in candidates.items()},
        "baseline_validation": baseline_val,
        "test": test_metrics,
        "baseline_test": baseline_test,
        "walk_forward": wf_metrics,
        "residual_std_95pct_interval": round(1.96 * interval_std, 6),
        "as_of_date": str(dataset["date"].iloc[-1]),
    }
    with open(artifact_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    with open(artifact_dir / "model_comparison.csv", "w", encoding="utf-8", newline="") as fh:
        compare_frames({k: v["val"] for k, v in candidates.items()}).to_csv(fh, index=False)

    log.info(
        "%s: selected %s (%s) | test RMSE %.4f vs baseline %.4f | dir acc %.3f",
        ticker, model_name, set_name, test_metrics["rmse"], baseline_test["rmse"],
        test_metrics["directional_accuracy"],
    )
    return meta


def train_all(universe: Universe | None = None, datasets: dict[str, pd.DataFrame] | None = None,
              models_dir: Path | None = None) -> dict[str, dict]:
    universe = universe or load_universe()
    datasets = datasets if datasets is not None else load_datasets(universe)
    results = {}
    for asset in universe.assets:
        ds = datasets.get(asset.ticker)
        if ds is None or ds.empty:
            continue
        meta = train_ticker(asset.ticker, ds, models_dir=models_dir)
        if meta:
            results[asset.ticker] = meta
    if results:
        summary = compare_frames({t: m["test"] for t, m in results.items()})
        summary.insert(0, "ticker", summary.pop("model"))
        with open((models_dir or MODELS_DIR) / "test_summary.csv", "w", encoding="utf-8", newline="") as fh:
            summary.to_csv(fh, index=False)
    return results
