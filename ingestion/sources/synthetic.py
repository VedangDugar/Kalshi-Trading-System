"""Synthetic data generators - the always-available fallback.

These produce *realistic* daily weather and a matching Kalshi-style binary
market history so the whole system runs offline and reproducibly. The design
goal is that our ML model, which conditions on recent weather, can beat the
market's roughly-climatological pricing near the contract threshold - creating
the small, real edge the strategy and backtester then exploit.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd


# Rough climatological parameters per city (deg F). Extend to add markets.
CITY_CLIMATE = {
    "NYC": {"base": 57.0, "amp": 23.0, "phase_doy": 200, "dtr": 15.0, "noise": 6.5},
    "CHI": {"base": 50.0, "amp": 27.0, "phase_doy": 200, "dtr": 16.0, "noise": 7.5},
    "MIA": {"base": 77.0, "amp": 9.0, "phase_doy": 210, "dtr": 12.0, "noise": 4.0},
    "LAX": {"base": 66.0, "amp": 9.0, "phase_doy": 225, "dtr": 14.0, "noise": 4.5},
}


def _seasonal_mean(doy: int, c: dict) -> float:
    """Climatological mean daily-high for a day-of-year."""
    return c["base"] + c["amp"] * math.sin(2 * math.pi * (doy - c["phase_doy"]) / 365.0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_weather(
    city: str = "NYC",
    years: int = 6,
    threshold_f: float = 90.0,  # kept for signature compatibility; unused here
    end: date | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a daily weather history AND the binary contract it settles.

    Columns: date, tmax_f, tmin_f, humidity, wind_mph, label_high,
             market_yes_price, threshold_f.

    Design (this is the honest core of the demo's edge):
      - A latent "warmth" state ``s`` follows an AR(1) process, so it persists
        day to day. The daily high we observe is a NOISY reflection of ``s``
        (observation noise + irreducible weather noise).
      - The contract ("will tomorrow be above normal?") settles YES with the
        TRUE probability ``q = sigmoid(beta * s)``.
      - The MARKET price is a slightly shrunk, noisy version of that true
        probability: ``0.5 + K*(q-0.5) + noise`` with K<1. Because the market
        under-reacts to the signal (K<1), a model that recovers ``q`` from the
        weather sequence earns a small, realistic residual edge (~few %).

    The model never sees ``s`` or ``q`` - it only sees the noisy weather window,
    so it must actually learn the signal, and its edge is limited by both the
    market's efficiency (K) and its own estimation error.
    """
    c = CITY_CLIMATE.get(city, CITY_CLIMATE["NYC"])
    rng = np.random.default_rng(seed)
    end = end or date.today()
    n_days = years * 365
    start = end - timedelta(days=n_days - 1)

    BETA = 1.5          # how strongly latent warmth maps to P(above normal)
    K_MARKET = 0.65     # market shrinkage (<1: market under-reacts to the signal)
    MARKET_NOISE = 0.02
    MARKET_OBS_NOISE = 1.1  # the market is a *noisier* observer than our model
    OBS_NOISE = 0.25    # weather observation noise (denoised by the model window)
    phi = 0.6           # latent persistence

    s = 0.0
    s_prev = 0.0
    rows = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        doy = d.timetuple().tm_yday
        clim = _seasonal_mean(doy, c)
        trend = 0.15 * (i / 365.0)

        # Latent AR(1) warmth state (unit stationary variance).
        s = phi * s_prev + rng.normal(0.0, math.sqrt(1 - phi**2))

        # Observed weather = seasonal normal + noisy reflection of latent state.
        a_obs = s + rng.normal(0.0, OBS_NOISE)
        tmax = clim + trend + 6.0 * a_obs + rng.normal(0.0, 1.5)
        tmin = tmax - c["dtr"] - abs(rng.normal(0.0, 2.0))
        humidity = float(np.clip(rng.normal(62, 15), 15, 100))
        wind = float(np.clip(rng.normal(8, 3), 0, 40))

        # True probability and realized outcome for TODAY's contract.
        q = _sigmoid(BETA * s)
        outcome = 1 if rng.random() < q else 0

        # The market prices this contract using ONLY yesterday's information
        # (info available before settlement), observed noisily, extrapolated via
        # persistence, and under-reacting (shrinkage K<1). Our model, using the
        # full recent weather window, estimates yesterday's state far more
        # cleanly - so it forecasts today better and earns a small, real edge.
        market_obs = s_prev + rng.normal(0.0, MARKET_OBS_NOISE)
        market_q = _sigmoid(BETA * K_MARKET * phi * market_obs)
        market = float(np.clip(market_q + rng.normal(0, MARKET_NOISE), 0.02, 0.98))

        s_prev = s

        rows.append(
            {
                "date": pd.Timestamp(d),
                "tmax_f": round(float(tmax), 2),
                "tmin_f": round(float(tmin), 2),
                "humidity": round(humidity, 1),
                "wind_mph": round(wind, 1),
                "label_high": int(outcome),
                "market_yes_price": round(market, 4),
                "threshold_f": round(float(clim + trend), 2),  # seasonal normal
            }
        )
    return pd.DataFrame(rows)


# NOTE: the order-book snapshot generator used by the market-data publisher
# lives in marketdata/orderbook.py (NumPy-only) so that lightweight service does
# not need to depend on pandas.
