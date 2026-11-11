"""HengSeng Sentiment Price Prediction Agent — pipeline entry point.

Modes:
  collect    historical prices + social posts (Apify) + news (Exa/Tavily)
  sentiment  clean, align and FinBERT-annotate all collected data
  train      train/compare models per ticker (chronological validation)
  predict    next-period price predictions + per-asset analytics
  report     visual HTML/PNG reports per exchange
  all        run every stage in order

Examples:
  python main.py --mode collect
  python main.py --mode all --days 10
"""
from __future__ import annotations

import argparse
import sys

from src.utils.logger import get_logger

log = get_logger("main")

MODES = ["collect", "sentiment", "train", "predict", "report", "all"]


def run_mode(mode: str, days: int | None = None) -> None:
    from config.settings import load_universe

    universe = load_universe()

    if mode == "collect":
        from src.ingestion.pipeline import collect_news, collect_social
        from src.ingestion.prices import collect_prices

        prices = collect_prices(universe)
        log.info("prices collected for %d/%d assets", len(prices), len(universe.assets))
        social = collect_social(universe, days=days)
        log.info("social posts this run: %d", len(social))
        news = collect_news(universe, days=days)
        log.info("news articles this run: %d", len(news))

    elif mode == "sentiment":
        from src.sentiment.run_sentiment import annotate_all

        out = annotate_all(universe)
        log.info("annotated %d posts / %d news articles", len(out["posts"]), len(out["news"]))

    elif mode == "train":
        from src.prediction.train import train_all

        results = train_all(universe)
        for ticker, meta in results.items():
            log.info(
                "%s -> %s (%s) test RMSE %.4f (baseline %.4f), dir acc %.3f",
                ticker, meta["selected_model"], meta["selected_feature_set"],
                meta["test"]["rmse"], meta["baseline_test"]["rmse"],
                meta["test"]["directional_accuracy"],
            )
        if not results:
            log.error("no models trained — run `python main.py --mode collect` first")

    elif mode == "predict":
        from src.analytics.stock_analysis import analyze_universe
        from src.prediction.predict import predict_all

        preds = predict_all(universe)
        if preds.empty:
            log.error("no predictions produced — run `python main.py --mode train` first")
            return
        analysis = analyze_universe(universe)
        print("\n=== Next-period predictions ===")
        print(preds[["ticker", "as_of_date", "last_close", "predicted_next_close",
                     "predicted_pct_change", "model"]].to_string(index=False))
        print("\n=== Asset analytics (signals, NOT financial advice) ===")
        cols = [c for c in ["ticker", "price", "direction", "social_sentiment",
                            "social_mentions", "news_sentiment", "support_1",
                            "resistance_1"] if c in analysis.columns]
        print(analysis[cols].to_string(index=False))

    elif mode == "report":
        from src.analytics.report_builder import build_all_reports

        paths = build_all_reports(universe)
        for p in paths:
            print(f"report: {p}")
        if not paths:
            log.error("no reports generated — run earlier stages first")

    else:
        raise ValueError(f"unknown mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HengSeng Sentiment Price Prediction Agent")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--days", type=int, default=None,
                        help="social/news lookback window (default: from assets.yaml)")
    args = parser.parse_args(argv)

    modes = MODES[:-1] if args.mode == "all" else [args.mode]
    for mode in modes:
        log.info("=== stage: %s ===", mode)
        try:
            run_mode(mode, days=args.days)
        except Exception:
            log.exception("stage %s failed", mode)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
