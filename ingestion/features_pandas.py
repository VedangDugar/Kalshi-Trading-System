"""Pandas mirror of the Spark feature engineering.

Used as a fallback so the ML service can run standalone (e.g. local testing, or
if the Spark step is skipped). It produces the *same* column schema as
``spark_ingest.build_features`` so downstream code is identical either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_features_pandas(pdf: pd.DataFrame, threshold_f: float, seed: int = 7) -> pd.DataFrame:
    """Engineer features and define the "above trailing normal" contract.

    Contract: YES resolves if tomorrow's high exceeds the trailing 30-day normal
    (a robust, ~50/50 "above-normal temperature" market). ``threshold_f`` from
    config is ignored for the label (kept only for schema compatibility); the
    per-day normal is stored in the ``threshold_f`` column for display.

    Market pricing: a naive climatological market prices this ~0.5 (it does not
    know today's persistence), so a model that reads recent weather earns an edge.
    """
    rng = np.random.default_rng(seed)
    df = pdf.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["tmax_lag1"] = df["tmax_f"].shift(1)
    df["tmax_lag2"] = df["tmax_f"].shift(2)
    df["tmax_lag3"] = df["tmax_f"].shift(3)
    df["tmin_lag1"] = df["tmin_f"].shift(1)
    df["humidity_lag1"] = df["humidity"].shift(1)
    df["wind_lag1"] = df["wind_mph"].shift(1)
    df["tmax_roll7"] = df["tmax_f"].shift(1).rolling(7).mean()
    # Trailing 30-day "normal" (known at decision time -> no leakage).
    df["normal_30"] = df["tmax_f"].shift(1).rolling(30).mean()
    df["tmax_roll30"] = df["normal_30"]

    doy = df["date"].dt.dayofyear
    df["doy"] = doy
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

    df["target_high_f"] = df["tmax_f"]

    # If the contract columns were pre-generated (synthetic path), keep them.
    # Otherwise (real NOAA weather) derive an "above trailing-normal" contract
    # priced by a fairly-efficient market off the 7-day smoothed anomaly.
    if "label_high" not in df.columns:
        df["label_high"] = (df["tmax_f"] > df["normal_30"]).astype(int)
        ALPHA, K = 0.22, 0.92
        anom = (df["tmax_roll7"] - df["normal_30"]).to_numpy()
        p_info = 1.0 / (1.0 + np.exp(-ALPHA * anom))
        noise = rng.normal(0.0, 0.02, size=len(df))
        df["market_yes_price"] = np.clip(0.5 + K * (p_info - 0.5) + noise, 0.02, 0.98).round(4)
        df["threshold_f"] = df["normal_30"].round(2)

    df = df.dropna(
        subset=["tmax_lag1", "tmax_lag2", "tmax_lag3", "tmax_roll7", "normal_30"]
    ).reset_index(drop=True)
    return df
