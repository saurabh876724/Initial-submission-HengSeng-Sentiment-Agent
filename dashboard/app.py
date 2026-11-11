"""HengSeng Sentiment Agent — Streamlit dashboard.

Sections (spec §11):
  A Market Overview | B Stock Ranking | C Individual Stock Analysis
  D Price Prediction | E Model Performance | F Data Collection Status

All data is read from the local store produced by the pipeline stages.
Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS_DIR, PROCESSED_DIR  # noqa: E402
from src.analytics.stock_analysis import DISCLAIMER  # noqa: E402
from src.utils.storage import load_from_sqlite, load_parquet  # noqa: E402

st.set_page_config(page_title="HengSeng Sentiment Agent", page_icon="📈", layout="wide")


# ---------------- data loading (cached) ----------------

@st.cache_data(ttl=300, show_spinner=False)
def load_analysis() -> pd.DataFrame:
    csv = PROCESSED_DIR / "stock_analysis.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_prices(ticker: str) -> pd.DataFrame:
    return load_from_sqlite("prices", where="ticker = ?", params=(ticker,)).sort_values("date")


@st.cache_data(ttl=300, show_spinner=False)
def load_posts() -> pd.DataFrame:
    return load_from_sqlite("posts_annotated")


@st.cache_data(ttl=300, show_spinner=False)
def load_news() -> pd.DataFrame:
    return load_from_sqlite("news_annotated")


@st.cache_data(ttl=300, show_spinner=False)
def load_trend() -> pd.DataFrame:
    df = load_parquet(PROCESSED_DIR / "sentiment_trend.parquet")
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_by_source() -> pd.DataFrame:
    df = load_parquet(PROCESSED_DIR / "sentiment_by_source.parquet")
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = MODELS_DIR / "predictions_latest.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_model_metas(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        meta_path = MODELS_DIR / t.replace(".", "_") / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append({
                "ticker": t, "model": meta["selected_model"],
                "feature_set": meta["selected_feature_set"],
                "test_MAE": meta["test"]["mae"], "test_RMSE": meta["test"]["rmse"],
                "test_MAPE_%": meta["test"]["mape_pct"],
                "test_dir_acc": meta["test"]["directional_accuracy"],
                "baseline_RMSE": meta["baseline_test"]["rmse"],
                "baseline_dir_acc": meta["baseline_test"]["directional_accuracy"],
                "walk_forward_RMSE": meta.get("walk_forward", {}).get("rmse"),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def load_status() -> pd.DataFrame:
    return load_from_sqlite("collection_status")


def csv_button(df: pd.DataFrame, name: str):
    if not df.empty:
        st.download_button(f"⬇ Export {name} (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{name}.csv", mime="text/csv")


# ---------------- layout ----------------

st.title("📈 HengSeng Sentiment Price Prediction Agent")
st.caption(f"HKEX + KRX social sentiment, news sentiment and next-period price prediction. {DISCLAIMER}")

analysis = load_analysis()
if analysis.empty:
    st.warning(
        "No analysis data yet. Run the pipeline first:\n\n"
        "```\npython main.py --mode collect\npython main.py --mode sentiment\n"
        "python main.py --mode train\npython main.py --mode predict\n```"
    )
    st.stop()

# ---- sidebar filters ----
st.sidebar.header("Filters")
exchanges = sorted(analysis["exchange"].unique())
sel_exchanges = st.sidebar.multiselect("Exchange", exchanges, default=exchanges)
view = analysis[analysis["exchange"].isin(sel_exchanges)] if sel_exchanges else analysis
sel_tickers = st.sidebar.multiselect("Assets", sorted(view["ticker"].unique()),
                                     default=sorted(view["ticker"].unique()))
view = view[view["ticker"].isin(sel_tickers)]

posts = load_posts()
date_from = date_to = None
if not posts.empty and "posted_at_utc" in posts.columns:
    ts = pd.to_datetime(posts["posted_at_utc"], errors="coerce", utc=True).dropna()
    if len(ts):
        date_from, date_to = st.sidebar.date_input(
            "Post date range",
            value=(ts.min().date(), ts.max().date()),
            min_value=ts.min().date(),
            max_value=ts.max().date(),
        )

st.sidebar.markdown("---")
st.sidebar.metric("Assets shown", len(view))
st.sidebar.metric("Total mentions", int(view["social_mentions"].sum()))

if view.empty:
    st.info("No assets match the current filters.")
    st.stop()

tabs = st.tabs(["Market Overview", "Stock Ranking", "Stock Analysis",
                "Price Prediction", "Model Performance", "Data Status"])

# ============ A. Market Overview ============
with tabs[0]:
    for ex in sorted(view["exchange"].unique()):
        ex_df = view[view["exchange"] == ex]
        st.subheader(f"{ex} sentiment overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Social mentions", int(ex_df["social_mentions"].sum()))
        avg_sent = ex_df["social_sentiment"].mean()
        c2.metric("Avg social sentiment", f"{avg_sent:+.3f}",
                  delta="bullish" if avg_sent > 0.05 else "bearish" if avg_sent < -0.05 else "flat")
        c3.metric("Avg news sentiment", f"{ex_df['news_sentiment'].mean():+.3f}")
        bull, bear = (ex_df["direction"] == "bullish").sum(), (ex_df["direction"] == "bearish").sum()
        c4.metric("Bullish / Bearish", f"{bull} / {bear}")
        fig = go.Figure(go.Bar(
            x=ex_df["ticker"], y=ex_df["social_sentiment"],
            marker_color=["#2ecc71" if s >= 0 else "#e74c3c" for s in ex_df["social_sentiment"]],
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=10), template="plotly_dark",
                          yaxis_title="Engagement-weighted sentiment")
        st.plotly_chart(fig, use_container_width=True)

    trend = load_trend()
    if not trend.empty:
        st.subheader("Sentiment trend")
        t = trend[trend["ticker"].isin(view["ticker"])]
        fig = go.Figure()
        for ticker, g in t.groupby("ticker"):
            fig.add_trace(go.Scatter(x=g["date"], y=g["avg_sentiment"], mode="lines+markers", name=ticker))
        fig.update_layout(height=380, template="plotly_dark", yaxis_title="Avg sentiment")
        st.plotly_chart(fig, use_container_width=True)

# ============ B. Stock Ranking ============
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By social mentions")
        rank_m = view.sort_values("social_mentions", ascending=False)[
            ["ticker", "exchange", "company_name", "social_mentions", "direction"]]
        st.dataframe(rank_m, use_container_width=True, hide_index=True)
        csv_button(rank_m, "ranking_mentions")
    with c2:
        st.subheader("By sentiment score")
        rank_s = view.sort_values("social_sentiment", ascending=False)[
            ["ticker", "exchange", "social_sentiment", "news_sentiment", "direction"]]
        st.dataframe(rank_s, use_container_width=True, hide_index=True)
        csv_button(rank_s, "ranking_sentiment")

# ============ C. Individual Stock Analysis ============
with tabs[2]:
    ticker = st.selectbox("Asset", sorted(view["ticker"].unique()))
    row = view[view["ticker"] == ticker].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"{row['price']:,.2f}", delta=str(row.get("price_date", "")))
    c2.metric("Social sentiment", f"{row['social_sentiment']:+.3f}",
              delta=f"{int(row['social_mentions'])} mentions")
    c3.metric("News sentiment", f"{row['news_sentiment']:+.3f}")
    c4.metric("Direction", row["direction"].upper())
    c5.metric("RSI(14)", f"{row['rsi_14']:.1f}")

    st.markdown(f"**{row['company_name']}** — support/resistance (pivot method): "
                f"S2 {row['support_2']:,.2f} | S1 {row['support_1']:,.2f} | P {row['pivot']:,.2f} | "
                f"R1 {row['resistance_1']:,.2f} | R2 {row['resistance_2']:,.2f}  ")
    st.markdown(f"Technicals: SMA20 {row['sma_20']:,.2f} · SMA50 {row['sma_50']:,.2f} · "
                f"MACD hist {row['macd_hist']:+.3f} · ATR14 {row['atr_14']:,.2f} · "
                f"Ann. vol(20d) {row['volatility_20d_annualized'] * 100:.1f}%")
    st.caption(DISCLAIMER)

    posts_all = load_posts()
    if not posts_all.empty:
        p_t = posts_all[posts_all["ticker"] == ticker]
        if date_from and date_to and "posted_at_utc" in p_t.columns:
            pt_ts = pd.to_datetime(p_t["posted_at_utc"], errors="coerce", utc=True)
            p_t = p_t[pt_ts.dt.date.between(date_from, date_to)]
        c1, c2 = st.columns([1, 2])
        with c1:
            if not p_t.empty:
                counts = p_t["sentiment_label"].value_counts()
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.5))
                fig.update_layout(title="Post sentiment breakdown", height=300, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                eng = p_t[["likes", "replies", "reposts", "views"]].sum()
                st.markdown("**Engagement totals**  ")
                st.write(" · ".join(f"{k}: {int(v)}" for k, v in eng.items() if pd.notna(v)))
        with c2:
            st.markdown("**Recent posts**")
            cols = [c for c in ["source", "text", "sentiment_label", "sentiment_score", "posted_at_utc"] if c in p_t.columns]
            st.dataframe(p_t.sort_values("posted_at_utc", ascending=False)[cols].head(30),
                         use_container_width=True, hide_index=True)

    news_all = load_news()
    if not news_all.empty:
        n_t = news_all[news_all["ticker"] == ticker]
        if not n_t.empty:
            st.markdown("**News headlines**")
            time_col = next((c for c in ("posted_at_utc", "published_at_utc") if c in n_t.columns), None)
            show = n_t.sort_values(time_col, ascending=False) if time_col else n_t
            st.dataframe(show[["source", "title", "sentiment_label", "sentiment_score", "url"]].head(20),
                         use_container_width=True, hide_index=True)

# ============ D. Price Prediction ============
with tabs[3]:
    preds = load_predictions()
    if preds.empty:
        st.info("No predictions yet — run `python main.py --mode predict`.")
    else:
        p_view = preds[preds["ticker"].isin(view["ticker"])]
        st.dataframe(p_view[["ticker", "as_of_date", "last_close", "predicted_next_close",
                             "predicted_pct_change", "interval_95_low", "interval_95_high", "model"]],
                     use_container_width=True, hide_index=True)
        csv_button(p_view, "predictions")
        sel = st.selectbox("Chart asset", sorted(p_view["ticker"].unique())) if not p_view.empty else None
        if sel:
            prices = load_prices(sel)
            prow = p_view[p_view["ticker"] == sel].iloc[0]
            fig = go.Figure()
            hist = prices.tail(120)
            fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], name="Actual close",
                                     line=dict(color="#4f8ef7")))
            fig.add_trace(go.Scatter(x=[prow["as_of_date"]], y=[prow["predicted_next_close"]],
                                     mode="markers+text", name="Predicted n+1",
                                     marker=dict(color="#f1c40f", size=13, symbol="diamond"),
                                     text=[f"{prow['predicted_next_close']:,.2f}"], textposition="top center"))
            fig.update_layout(title=f"{sel} — actual vs predicted next-period close "
                                    f"({prow['predicted_pct_change']:+.2f}%)",
                              height=450, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("95% interval from out-of-sample residual std (normal approximation): "
                       f"[{prow['interval_95_low']:,.2f}, {prow['interval_95_high']:,.2f}]")

# ============ E. Model Performance ============
with tabs[4]:
    metas = load_model_metas(tuple(view["ticker"]))
    if metas.empty:
        st.info("No trained models yet — run `python main.py --mode train`.")
    else:
        st.dataframe(metas, use_container_width=True, hide_index=True)
        csv_button(metas, "model_performance")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=metas["ticker"], y=metas["test_RMSE"], name="Model RMSE"))
        fig.add_trace(go.Bar(x=metas["ticker"], y=metas["baseline_RMSE"], name="Prev-close baseline RMSE"))
        fig.update_layout(barmode="group", title="Out-of-sample RMSE vs baseline (lower is better)",
                          height=380, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure(go.Bar(x=metas["ticker"], y=metas["test_dir_acc"],
                                marker_color="#9b59b6"))
        fig2.add_hline(y=0.5, line_dash="dash", line_color="gray",
                       annotation_text="coin flip")
        fig2.update_layout(title="Directional accuracy (test set)", height=340, template="plotly_dark",
                           yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

# ============ F. Data Collection Status ============
with tabs[5]:
    status = load_status()
    if status.empty:
        st.info("No collection runs recorded yet.")
    else:
        g = status.groupby("source").agg(
            runs=("ok", "size"),
            items=("items", "sum"),
            new_items=("new_items", "sum"),
            failed_runs=("ok", lambda s: int((~s.astype(bool)).sum())),
            last_run=("finished_at", "max"),
        ).reset_index()
        st.dataframe(g, use_container_width=True, hide_index=True)
        csv_button(g, "collection_status")
        errors = status[~status["ok"].astype(bool)]
        if not errors.empty:
            st.warning(f"{len(errors)} failed collection runs — details below.")
            st.dataframe(errors[["source", "ticker", "error", "finished_at"]],
                         use_container_width=True, hide_index=True)
        missing = view[view["social_mentions"] == 0]["ticker"].tolist()
        if missing:
            st.warning("Assets with zero social mentions in window: " + ", ".join(missing))

st.markdown("---")
st.caption("Data: Apify (LIHKG, Telegram, Naver 토론실, Daum search) · Exa/Tavily news · Yahoo Finance prices · "
           "FinBERT sentiment · scikit-learn/XGBoost models.")
