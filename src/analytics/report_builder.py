"""Visual report generation per exchange (spec §12).

Outputs to reports/:
- {EXCHANGE}_report.html : interactive Plotly report (self-contained)
- {EXCHANGE}_overview.png : static matplotlib summary panel

Sorted views: by social mentions, by sentiment score, by direction.
Includes model evaluation, prediction, data-quality limitations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, Universe, load_universe
from src.analytics.stock_analysis import DISCLAIMER, analyze_universe
from src.utils.logger import get_logger
from src.utils.storage import load_from_sqlite, load_parquet

log = get_logger(__name__)


def _model_table(universe: Universe, tickers: list[str]) -> pd.DataFrame:
    import json

    rows = []
    for t in tickers:
        meta_path = MODELS_DIR / t.replace(".", "_") / "meta.json"
        if not meta_path.exists():
            rows.append({"ticker": t, "model": "not trained"})
            continue
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        rows.append(
            {
                "ticker": t,
                "model": meta["selected_model"],
                "feature_set": meta["selected_feature_set"],
                "test_MAE": meta["test"]["mae"],
                "test_RMSE": meta["test"]["rmse"],
                "test_dir_acc": meta["test"]["directional_accuracy"],
                "baseline_RMSE": meta["baseline_test"]["rmse"],
                "baseline_dir_acc": meta["baseline_test"]["directional_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def _status_table(tickers: list[str]) -> pd.DataFrame:
    status = load_from_sqlite("collection_status")
    if status.empty:
        return pd.DataFrame(columns=["source", "runs", "items", "new_items", "errors", "last_run"])
    status = status[status["ticker"].isin(tickers)]
    g = status.groupby("source").agg(
        runs=("ok", "size"),
        items=("items", "sum"),
        new_items=("new_items", "sum"),
        errors=("ok", lambda s: int((~s.astype(bool)).sum())),
        last_run=("finished_at", "max"),
    ).reset_index()
    return g


def _fig_overview(ex_df: pd.DataFrame, exchange: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Social mentions (10-day window)",
            "Engagement-weighted sentiment",
            "Direction distribution",
            "Predicted next-day change %",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "domain"}, {"type": "bar"}],
        ],
    )
    d = ex_df.sort_values("social_mentions", ascending=False)
    fig.add_trace(go.Bar(x=d["ticker"], y=d["social_mentions"], marker_color="#4f8ef7"), 1, 1)
    colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in d["social_sentiment"]]
    fig.add_trace(go.Bar(x=d["ticker"], y=d["social_sentiment"], marker_color=colors), 1, 2)
    dir_counts = d["direction"].value_counts()
    fig.add_trace(
        go.Pie(labels=dir_counts.index.tolist(), values=dir_counts.tolist(), hole=0.5), 2, 1
    )
    if d["predicted_pct_change"].notna().any():
        pcolors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in d["predicted_pct_change"].fillna(0)]
        fig.add_trace(go.Bar(x=d["ticker"], y=d["predicted_pct_change"], marker_color=pcolors), 2, 2)
    fig.update_layout(height=700, title_text=f"{exchange} market overview", showlegend=False,
                      template="plotly_dark")
    return fig


def _fig_price_prediction(ticker: str, exchange_name: str) -> go.Figure | None:
    prices = load_from_sqlite("prices", where="ticker = ?", params=(ticker,))
    if prices.empty:
        return None
    prices = prices.sort_values("date").tail(120)
    pred_path = MODELS_DIR / "predictions_latest.json"
    pred = None
    if pred_path.exists():
        import json

        with open(pred_path, encoding="utf-8") as fh:
            pred = next((r for r in json.load(fh) if r["ticker"] == ticker), None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices["date"], y=prices["close"], name="Close (last 120 sessions)",
                             line=dict(color="#4f8ef7")))
    if pred:
        next_x = pred.get("as_of_date")
        fig.add_trace(go.Scatter(
            x=[next_x], y=[pred["predicted_next_close"]], mode="markers+text",
            name="Predicted n+1", marker=dict(color="#f1c40f", size=12, symbol="diamond"),
            text=[f"{pred['predicted_next_close']:.2f}"], textposition="top center",
        ))
    fig.update_layout(title=f"{ticker} — actual close vs predicted next-period", height=420,
                      template="plotly_dark")
    return fig


def _png_overview(ex_df: pd.DataFrame, exchange: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{exchange} sentiment & prediction overview", fontsize=14)
    d = ex_df.sort_values("social_mentions", ascending=False)

    axes[0][0].bar(d["ticker"], d["social_mentions"], color="#4f8ef7")
    axes[0][0].set_title("Social mentions")
    axes[0][0].tick_params(axis="x", rotation=30)

    colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in d["social_sentiment"]]
    axes[0][1].bar(d["ticker"], d["social_sentiment"], color=colors)
    axes[0][1].axhline(0, color="black", linewidth=0.8)
    axes[0][1].set_title("Sentiment score")
    axes[0][1].tick_params(axis="x", rotation=30)

    axes[1][0].axis("off")
    top = d.nlargest(3, "social_mentions")[["ticker", "direction"]]
    bull = d[d["direction"] == "bullish"][["ticker", "social_sentiment"]]
    bear = d[d["direction"] == "bearish"][["ticker", "social_sentiment"]]
    txt = "Top discussed:\n" + "\n".join(f"  {r.ticker} ({r.direction})" for r in top.itertuples())
    txt += "\n\nMost bullish:\n" + ("\n".join(f"  {r.ticker} ({r.social_sentiment:+.2f})" for r in bull.itertuples()) or "  none")
    txt += "\n\nMost bearish:\n" + ("\n".join(f"  {r.ticker} ({r.social_sentiment:+.2f})" for r in bear.itertuples()) or "  none")
    axes[1][0].text(0.05, 0.95, txt, va="top", family="monospace", fontsize=10)

    axes[1][1].axis("off")
    axes[1][1].text(0.05, 0.95, DISCLAIMER + "\n\nSentiment window: last 10 days only.\nHistorical model training uses price features;\nsentiment contribution is reported, not assumed.",
                    va="top", fontsize=9, wrap=True)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_exchange_report(exchange: str, universe: Universe | None = None,
                          analysis: pd.DataFrame | None = None) -> Path | None:
    universe = universe or load_universe()
    analysis = analyze_universe(universe) if analysis is None else analysis
    if analysis.empty:
        log.warning("report: no analysis rows for %s", exchange)
        return None
    ex_df = analysis[analysis["exchange"] == exchange].copy()
    if ex_df.empty:
        return None
    tickers = ex_df["ticker"].tolist()
    exchange_name = universe.exchanges.get(exchange, {}).get("name", exchange)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    by_mentions = ex_df.sort_values("social_mentions", ascending=False)
    by_sentiment = ex_df.sort_values("social_sentiment", ascending=False)
    by_direction = ex_df.sort_values(["direction", "composite_signal"], ascending=[True, False])
    models = _model_table(universe, tickers)
    status = _status_table(tickers)
    trend = load_parquet(PROCESSED_DIR / "sentiment_trend.parquet")

    parts = [
        f"<html><head><meta charset='utf-8'><title>{exchange} sentiment report</title></head><body style='background:#111;color:#eee;font-family:Segoe UI,Arial'>",
        f"<h1>{exchange_name} — Sentiment & Prediction Report</h1>",
        f"<p>Generated: {generated} &nbsp;|&nbsp; <b>{DISCLAIMER}</b></p>",
        "<h2>Market overview</h2>",
        f"<p>Total social mentions: <b>{int(ex_df['social_mentions'].sum())}</b> | "
        f"Bullish: <b>{int((ex_df['direction']=='bullish').sum())}</b> | "
        f"Sideways: <b>{int((ex_df['direction']=='sideways').sum())}</b> | "
        f"Bearish: <b>{int((ex_df['direction']=='bearish').sum())}</b></p>",
        _fig_overview(ex_df, exchange).to_html(full_html=False, include_plotlyjs="cdn"),
        "<h2>Ranking — by social mentions</h2>",
        by_mentions[["ticker", "company_name", "social_mentions", "social_sentiment", "direction"]].to_html(index=False, border=0),
        "<h2>Ranking — by sentiment score</h2>",
        by_sentiment[["ticker", "company_name", "social_sentiment", "news_sentiment", "social_mentions", "direction"]].to_html(index=False, border=0),
        "<h2>Ranking — by sentiment direction</h2>",
        by_direction[["ticker", "direction", "composite_signal", "price", "predicted_next_close", "predicted_pct_change"]].to_html(index=False, border=0),
        "<h2>Price vs prediction (per asset)</h2>",
    ]
    for t in tickers:
        fig = _fig_price_prediction(t, exchange_name)
        if fig:
            parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    parts += [
        "<h2>Model performance (out-of-sample test set)</h2>",
        models.to_html(index=False, border=0),
        "<h2>Data collection status</h2>",
        status.to_html(index=False, border=0),
        "<h2>Data-quality limitations</h2>",
        "<ul>"
        "<li>Social sentiment covers only the last 10 days — far too short to validate sentiment-driven prediction statistically. Sentiment features are included and their measured effect is reported, not assumed.</li>"
        "<li>Models are trained primarily on price-derived features; the previous-close baseline is shown for honest comparison.</li>"
        "<li>Directional signals are analytical research outputs, not financial advice.</li>"
        "</ul>",
        "</body></html>",
    ]

    out_html = REPORTS_DIR / f"{exchange}_report.html"
    out_html.write_text("".join(parts), encoding="utf-8")
    out_png = REPORTS_DIR / f"{exchange}_overview.png"
    _png_overview(ex_df, exchange_name, out_png)
    log.info("report written: %s (+ %s)", out_html, out_png.name)
    return out_html


def build_all_reports(universe: Universe | None = None, analysis: pd.DataFrame | None = None) -> list[Path]:
    universe = universe or load_universe()
    paths = []
    for exchange in universe.exchanges:
        p = build_exchange_report(exchange, universe, analysis)
        if p:
            paths.append(p)
    return paths
