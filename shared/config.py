"""Central configuration, read from environment variables with safe defaults.

Every service imports from here so that hostnames/ports live in exactly one
place. Defaults match the service names defined in ``docker-compose.yml`` so the
system runs unchanged inside Compose, while environment overrides let you run a
single service locally against ``localhost``.
"""

from __future__ import annotations

import os


# --- Redis -----------------------------------------------------------------
REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))


# --- ZeroMQ market-data bus ------------------------------------------------
# The publisher binds this address; the C++ engine and any Python consumer
# connect to it. Inside Compose the host is the service name "marketdata".
MD_PUB_HOST: str = os.getenv("MD_PUB_HOST", "marketdata")
MD_PUB_PORT: int = int(os.getenv("MD_PUB_PORT", "5556"))


def md_pub_bind_addr() -> str:
    """Address the publisher binds to (all interfaces)."""
    return f"tcp://*:{MD_PUB_PORT}"


def md_pub_connect_addr() -> str:
    """Address consumers connect to."""
    return f"tcp://{MD_PUB_HOST}:{MD_PUB_PORT}"


# --- Demo domain -----------------------------------------------------------
# The single weather market used for the focused demo. The code is structured
# so additional cities/markets can be added by extending these lists.
DEFAULT_CITY: str = os.getenv("CITY", "NYC")
DEFAULT_STATION: str = os.getenv("NOAA_STATION", "KNYC")  # Central Park, NY
# "Will the daily high exceed this threshold (deg F)?" -> the Kalshi-style
# binary weather contract we price.
TEMP_THRESHOLD_F: float = float(os.getenv("TEMP_THRESHOLD_F", "90"))

# The three venues whose order books we watch for cross-exchange arbitrage.
VENUES = ("kalshi", "polymarket", "robinhood")
MARKET_ID: str = os.getenv("MARKET_ID", "NYC-HIGH-90F")

# Backtest / bankroll defaults (map to the resume: 5K bankroll, fractional Kelly)
BANKROLL_USD: float = float(os.getenv("BANKROLL_USD", "5000"))
KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.3"))  # fractional Kelly
KELLY_CAP: float = float(os.getenv("KELLY_CAP", "0.025"))          # max fraction/bet
# Per-trade transaction cost (fraction of position): Kalshi fees + slippage.
FEE_RATE: float = float(os.getenv("FEE_RATE", "0.02"))

# Data locations (Parquet lake). In AWS this maps to an S3 prefix.
DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")
MODEL_DIR: str = os.getenv("MODEL_DIR", "/app/data/models")

# Optional API keys. When absent, ingestion falls back to synthetic data.
NOAA_TOKEN: str = os.getenv("NOAA_TOKEN", "")
KALSHI_API_BASE: str = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2")
