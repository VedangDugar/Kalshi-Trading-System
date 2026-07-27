"""Redis key names and JSON message schemas shared across all services.

Keeping these in one module means the ingestion job, ML pipeline, market-data
publisher, C++ engine (via its mirrored header) and dashboard never disagree
about where data lives or what shape it has.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Redis keys (single source of truth). Namespaced under "kalshi:".
# ---------------------------------------------------------------------------
class Keys:
    # ML + strategy outputs (written by the `ml` service, read by dashboard)
    FORECAST_LATEST = "kalshi:forecast:latest"       # JSON ForecastMessage
    ODDS_LATEST = "kalshi:odds:latest"               # JSON OddsMessage
    KELLY_LATEST = "kalshi:kelly:latest"             # JSON KellyMessage
    BACKTEST_METRICS = "kalshi:backtest:metrics"     # JSON BacktestMetrics
    BACKTEST_EQUITY = "kalshi:backtest:equity"       # JSON list[[iso_date, equity]]
    PIPELINE_STATUS = "kalshi:pipeline:status"       # JSON {stage, ts, detail}

    # Real-time arbitrage (written by the C++ engine, read by dashboard)
    ARB_SIGNALS = "kalshi:arb:signals"               # Redis LIST of JSON ArbSignal
    ENGINE_STATS = "kalshi:engine:stats"             # JSON EngineStats

    # Latest quote per venue (written by publisher, handy for the dashboard)
    @staticmethod
    def market_quote(venue: str) -> str:
        return f"kalshi:market:quote:{venue}"


# ZeroMQ topic for a venue's book updates. The engine subscribes to all of them.
def md_topic(venue: str) -> str:
    return f"book.{venue}"


ARB_SIGNALS_MAXLEN = 200  # keep the most recent N arb signals in the list


# ---------------------------------------------------------------------------
# Message schemas (dataclasses -> dict -> JSON). Every field documented so the
# contract is self-describing for anyone reading the repo.
# ---------------------------------------------------------------------------
def _now_ns() -> int:
    import time

    return time.time_ns()


@dataclass
class Quote:
    """A single venue's top-of-book for the binary "YES" contract.

    Prices are in probability units [0, 1] (i.e. Kalshi cents / 100).
    """

    venue: str
    market: str
    yes_bid: float
    yes_ask: float
    seq: int
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2.0


@dataclass
class ArbSignal:
    """A detected cross-venue arbitrage opportunity emitted by the C++ engine."""

    market: str
    buy_venue: str      # buy YES cheap here
    sell_venue: str     # sell YES rich here
    buy_price: float    # yes_ask on buy venue
    sell_price: float   # yes_bid on sell venue
    spread_pct: float   # (sell_price - buy_price) as a fraction, e.g. 0.0012
    latency_us: int     # detection latency (recv -> decision) in microseconds
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineStats:
    """Health/latency counters published by the C++ engine."""

    started_at_ns: int
    uptime_seconds: float
    messages_processed: int
    arb_opportunities: int
    trades_executed: int
    latency_us_p50: float
    latency_us_p99: float
    msgs_per_sec: float
    uptime_pct: float  # rolling availability estimate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastMessage:
    """Ensemble forecast for the binary weather contract."""

    city: str
    market: str
    target_date: str          # ISO date the contract resolves on
    threshold_f: float
    lstm_prob: float          # P(high > threshold) from TensorFlow LSTM
    transformer_prob: float   # P(high > threshold) from PyTorch Transformer
    ensemble_prob: float      # blended probability (our "proprietary odds")
    predicted_high_f: float   # point estimate of the daily high
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OddsMessage:
    """Our model probability vs. the market-implied probability -> edge."""

    market: str
    model_prob: float          # from the ensemble
    market_prob: float         # implied by the Kalshi YES mid price
    edge: float                # model_prob - market_prob
    fair_yes_price: float      # what we think YES is worth
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KellyMessage:
    """Position sizing recommendation from fractional Kelly."""

    market: str
    side: str                  # "YES" or "NO"
    edge: float
    kelly_fraction_full: float  # raw Kelly f*
    kelly_fraction_used: float  # after applying the safety fraction
    stake_usd: float
    bankroll_usd: float
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestMetrics:
    """Headline results computed by the backtester over the data window.

    NOTE: these are computed from data at runtime, never hardcoded. The values
    quoted on the resume (~3.2% edge, ~12% MoM ROI, <5% drawdown) are the kind
    of numbers this report produces on a favorable window.
    """

    n_trades: int
    avg_edge: float
    hit_rate: float
    monthly_roi: float
    total_return: float
    max_drawdown: float
    sharpe: float
    final_bankroll: float
    window_start: str
    window_end: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
