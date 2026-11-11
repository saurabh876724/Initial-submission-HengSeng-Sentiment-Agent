"""Next-period (n+1) price prediction using the persisted best models."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from config.settings import MODELS_DIR, Universe, load_universe
from src.utils.logger import get_logger
from src.utils.storage import save_to_sqlite

log = get_logger(__name__)


def predict_ticker(
    ticker: str,
    dataset: pd.DataFrame | None = None,
    models_dir: Path | None = None,
) -> dict | None:
    """Predict next session's close for one ticker.

    Returns None (with a warning) when no trained model exists — the
    caller must run training first. Never fabricates a prediction.
    """
    from src.prediction.train import load_datasets

    models_dir = models_dir or MODELS_DIR
    artifact_dir = models_dir / ticker.replace(".", "_")
    model_path = artifact_dir / "best_model.joblib"
    meta_path = artifact_dir / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        log.warning("no trained model for %s — run `python main.py --mode train` first", ticker)
        return None

    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    if dataset is None:
        dataset = load_datasets().get(ticker)
    if dataset is None or dataset.empty:
        log.warning("no feature data for %s — run collection first", ticker)
        return None

    row = dataset.iloc[[-1]][meta["feature_cols"]]
    last_close = float(dataset["close"].iloc[-1])
    as_of = str(dataset["date"].iloc[-1])
    predicted = float(model.predict(row)[0])
    interval = meta.get("residual_std_95pct_interval", 0.0)

    result = {
        "ticker": ticker,
        "as_of_date": as_of,
        "last_close": round(last_close, 4),
        "predicted_next_close": round(predicted, 4),
        "predicted_pct_change": round(100 * (predicted / last_close - 1), 4),
        "interval_95_low": round(predicted - interval, 4),
        "interval_95_high": round(predicted + interval, 4),
        "model": meta["selected_model"],
        "feature_set": meta["selected_feature_set"],
        "test_rmse": meta["test"]["rmse"],
        "test_directional_accuracy": meta["test"]["directional_accuracy"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    log.info(
        "%s: last close %.2f (%s) -> predicted %.2f (%+.2f%%) [%s/%s]",
        ticker, last_close, as_of, predicted, result["predicted_pct_change"],
        result["model"], result["feature_set"],
    )
    return result


def predict_all(
    universe: Universe | None = None,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    universe = universe or load_universe()
    from src.prediction.train import load_datasets

    datasets = load_datasets(universe)
    results = []
    for asset in universe.assets:
        r = predict_ticker(asset.ticker, dataset=datasets.get(asset.ticker), models_dir=models_dir)
        if r:
            results.append(r)
    df = pd.DataFrame(results)
    if not df.empty:
        save_to_sqlite(df, "predictions", dedup_keys=["ticker", "generated_at"])
        with open((models_dir or MODELS_DIR) / "predictions_latest.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    return df
