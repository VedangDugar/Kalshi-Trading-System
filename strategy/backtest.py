"""Event-driven backtester over the out-of-sample window.

Walks day by day: forms a decision from (model_prob vs market_prob), sizes it
with fractional Kelly, then compounds the bankroll by the realized outcome. All
headline numbers (edge, monthly ROI, drawdown, Sharpe) are computed here from
data - never hardcoded.

Return mechanics for a fraction ``f`` of bankroll on a binary contract:
  YES at price q:  win -> B *= 1 + f*((1-q)/q);  lose -> B *= 1 - f
  NO  at price 1-q: win -> B *= 1 + f*(q/(1-q)); lose -> B *= 1 - f
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared import config
from shared.schemas import BacktestMetrics
from strategy import kelly, odds


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    equity_curve: list[list]   # [[iso_date, equity_usd], ...]
    daily_returns: list[float]


def _iso(d) -> str:
    return pd.Timestamp(d).date().isoformat()


def run_backtest(
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    outcome: np.ndarray,
    dates: np.ndarray,
    edge_threshold: float = 0.02,
    safety_fraction: float | None = None,
    bankroll_usd: float | None = None,
    cap: float | None = None,
    fee_rate: float | None = None,
) -> BacktestResult:
    bankroll = float(bankroll_usd if bankroll_usd is not None else config.BANKROLL_USD)
    start_bankroll = bankroll
    safety = config.KELLY_FRACTION if safety_fraction is None else safety_fraction
    cap = config.KELLY_CAP if cap is None else cap
    fee_rate = config.FEE_RATE if fee_rate is None else fee_rate

    equity_curve: list[list] = []
    daily_rets: list[float] = []
    taken_edges: list[float] = []
    wins = 0
    n_trades = 0

    for p, q, y, dt in zip(model_prob, market_prob, outcome, dates):
        decision = odds.decide(float(p), float(q), edge_threshold)
        prev = bankroll

        if decision.side != "PASS":
            f = kelly.size_position(
                decision.side, float(p), float(q), bankroll, safety, cap
            ).kelly_used
            if f > 0:
                n_trades += 1
                taken_edges.append(decision.edge)
                q_ = float(min(max(q, 1e-4), 1 - 1e-4))
                if decision.side == "YES":
                    b = (1 - q_) / q_
                    won = y == 1
                else:  # NO
                    b = q_ / (1 - q_)
                    won = y == 0
                # Constant position sizing: stake a Kelly fraction of the START
                # bankroll (not the compounding balance). This is the standard
                # conservative choice - it keeps a single daily binary market
                # from compounding into absurd numbers and makes returns
                # interpretable. Transaction cost/slippage is proportional to
                # position size (Kalshi taker fee + crossing the spread).
                stake = f * start_bankroll
                pnl = (stake * b) if won else (-stake)
                pnl -= stake * fee_rate
                bankroll += pnl
                if won:
                    wins += 1

        equity_curve.append([_iso(dt), round(bankroll, 2)])
        daily_rets.append((bankroll - prev) / start_bankroll)

    # --- metrics ----------------------------------------------------------
    rets = np.asarray(daily_rets, dtype="float64")
    eq = np.asarray([e[1] for e in equity_curve], dtype="float64")

    total_return = bankroll / start_bankroll - 1.0
    n_days = max(1, len(dates))
    n_months = max(1e-9, n_days / 30.0)
    monthly_roi = (bankroll / start_bankroll) ** (1.0 / n_months) - 1.0

    if len(eq):
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / running_max
        max_dd = float(-drawdowns.min())
    else:
        max_dd = 0.0

    if rets.std() > 1e-12:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    metrics = BacktestMetrics(
        n_trades=int(n_trades),
        avg_edge=float(np.mean(taken_edges)) if taken_edges else 0.0,
        hit_rate=float(wins / n_trades) if n_trades else 0.0,
        monthly_roi=float(monthly_roi),
        total_return=float(total_return),
        max_drawdown=float(max_dd),
        sharpe=sharpe,
        final_bankroll=round(float(bankroll), 2),
        window_start=_iso(dates[0]) if len(dates) else "",
        window_end=_iso(dates[-1]) if len(dates) else "",
    )
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, daily_returns=daily_rets)
