"""Streamlit demo dashboard - the single screen you show in an interview.

Reads everything from Redis (populated by the ML pipeline and the C++ engine)
and refreshes every few seconds so the arbitrage feed and live edge update in
real time. Purely a viewer: it performs no computation of its own.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from shared import config
from shared.schemas import Keys
from shared.store import get_json, get_json_list, get_redis

st.set_page_config(page_title="Kalshi Trading System", layout="wide")

# Auto-refresh (optional dependency; degrade gracefully if missing).
try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=3000, key="refresh")
except Exception:
    pass


@st.cache_resource
def _redis():
    return get_redis()


r = _redis()


def pct(x, digits=2):
    try:
        return f"{x * 100:.{digits}f}%"
    except Exception:
        return "-"


# --- Header ----------------------------------------------------------------
st.title("Kalshi Trading System")
st.caption(
    "LSTM (TensorFlow) + Transformer (PyTorch) ensemble for Kalshi weather "
    "markets, Bayesian-optimized Kelly sizing, and a C++17 Boost.Asio / ZeroMQ "
    "/ Redis cross-exchange arbitrage engine. Demo / no real money."
)

status = get_json(r, Keys.PIPELINE_STATUS, {})
if status:
    st.info(f"Pipeline stage: **{status.get('stage', '?')}** - {status.get('detail', '')}")

forecast = get_json(r, Keys.FORECAST_LATEST, {})
odds = get_json(r, Keys.ODDS_LATEST, {})
kelly = get_json(r, Keys.KELLY_LATEST, {})
backtest = get_json(r, Keys.BACKTEST_METRICS, {})
equity = get_json(r, Keys.BACKTEST_EQUITY, [])
engine = get_json(r, Keys.ENGINE_STATS, {})
arbs = get_json_list(r, Keys.ARB_SIGNALS, 50)

# --- Section 1: ML forecast + edge ----------------------------------------
st.header("1 - Weather forecast & pricing")
if forecast:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contract", forecast.get("market", config.MARKET_ID))
    c2.metric(
        f"Ensemble P(high > {forecast.get('threshold_f','?')}F)",
        pct(forecast.get("ensemble_prob", 0)),
    )
    c3.metric("Predicted high (F)", forecast.get("predicted_high_f", "-"))
    c4.metric("Resolves", forecast.get("target_date", "-"))

    d1, d2 = st.columns(2)
    d1.metric("LSTM (TensorFlow)", pct(forecast.get("lstm_prob", 0)))
    d2.metric("Transformer (PyTorch)", pct(forecast.get("transformer_prob", 0)))

if odds and kelly:
    st.subheader("Edge & Kelly sizing")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Model prob", pct(odds.get("model_prob", 0)))
    e2.metric("Market prob", pct(odds.get("market_prob", 0)))
    e3.metric("Edge", pct(odds.get("edge", 0)), delta=pct(odds.get("edge", 0)))
    e4.metric("Signal", kelly.get("side", "-"))
    e5.metric("Kelly stake", f"${kelly.get('stake_usd', 0):,.0f}")
    st.caption(
        f"Full Kelly f*={kelly.get('kelly_fraction_full', 0):.3f}, "
        f"used (half-Kelly, capped)={kelly.get('kelly_fraction_used', 0):.3f}, "
        f"bankroll=${kelly.get('bankroll_usd', 0):,.0f}"
    )
else:
    st.write("Waiting for the ML pipeline to publish a forecast...")

# --- Section 2: Backtest ---------------------------------------------------
st.header("2 - Strategy backtest (out-of-sample)")
if backtest:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Avg edge", pct(backtest.get("avg_edge", 0)))
    b2.metric("Monthly ROI", pct(backtest.get("monthly_roi", 0)))
    b3.metric("Max drawdown", pct(backtest.get("max_drawdown", 0)))
    b4.metric("Sharpe", f"{backtest.get('sharpe', 0):.2f}")

    b5, b6, b7, b8 = st.columns(4)
    b5.metric("Trades", backtest.get("n_trades", 0))
    b6.metric("Hit rate", pct(backtest.get("hit_rate", 0), 1))
    b7.metric("Total return", pct(backtest.get("total_return", 0)))
    b8.metric("Final bankroll", f"${backtest.get('final_bankroll', 0):,.0f}")

    st.caption(
        f"Window {backtest.get('window_start','?')} to {backtest.get('window_end','?')} | "
        f"ensemble weight_lstm={backtest.get('weight_lstm', 0):.2f}, "
        f"edge_threshold={backtest.get('edge_threshold', 0):.3f}, "
        f"tuned via {backtest.get('opt_method','?')} | "
        f"backends={backtest.get('backends', {})}"
    )

    if equity:
        eq_df = pd.DataFrame(equity, columns=["date", "equity"])
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df = eq_df.set_index("date")
        st.line_chart(eq_df, height=280)
    st.caption(
        "Note: metrics are computed from data at runtime (not hardcoded). "
        "Exact numbers depend on the data window and random seeds."
    )
else:
    st.write("Waiting for the backtest to complete...")

# --- Section 3: Live arbitrage engine -------------------------------------
st.header("3 - Live C++ arbitrage engine")
if engine:
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Latency p50", f"{engine.get('latency_us_p50', 0):.0f} us")
    g2.metric("Latency p99", f"{engine.get('latency_us_p99', 0):.0f} us")
    g3.metric("Msgs/sec", f"{engine.get('msgs_per_sec', 0):.0f}")
    g4.metric("Uptime", f"{engine.get('uptime_pct', 0):.2f}%")

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Messages processed", f"{engine.get('messages_processed', 0):,}")
    h2.metric("Arb opportunities", f"{engine.get('arb_opportunities', 0):,}")
    h3.metric("Trades executed", f"{engine.get('trades_executed', 0):,}")
    h4.metric("Uptime (s)", f"{engine.get('uptime_seconds', 0):.0f}")
else:
    st.write("Waiting for the C++ engine to report stats...")

st.subheader("Recent arbitrage signals")
if arbs:
    adf = pd.DataFrame(arbs)
    if "spread_pct" in adf:
        adf["spread_%"] = (adf["spread_pct"] * 100).round(3)
    cols = [c for c in ["market", "buy_venue", "sell_venue", "buy_price",
                        "sell_price", "spread_%", "latency_us"] if c in adf]
    st.dataframe(adf[cols].head(25), use_container_width=True, hide_index=True)
else:
    st.write("No arbitrage detected yet - the simulator injects spreads periodically.")

# --- Live venue quotes -----------------------------------------------------
with st.expander("Live venue quotes (top of book)"):
    rows = []
    for v in config.VENUES:
        q = get_json(r, Keys.market_quote(v), {})
        if q:
            rows.append(q)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
