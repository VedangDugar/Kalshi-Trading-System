"""Load the Parquet feature lake and turn it into supervised sequences.

The models forecast the *next day's* binary outcome (daily high > threshold)
from a rolling window of the previous ``seq_len`` days of weather. Crucially the
window only contains PAST days, so there is no same-day leakage, and the market
price is deliberately excluded from the model features - the model is a pure
weather forecaster, which is what lets it earn an edge over the market's
roughly-climatological pricing.

Time-ordered splits: train | validation | test.
  - train: fit the LSTM + Transformer
  - validation: tune ensemble weight + edge threshold via Bayesian optimization
  - test: the untouched out-of-sample window the backtest reports on
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared import config

# Weather features fed to the sequence models (NO market price -> genuine edge).
FEATURE_COLS = ["tmax_f", "tmin_f", "humidity", "wind_mph", "doy_sin", "doy_cos"]


@dataclass
class Dataset:
    X: np.ndarray            # (N, seq_len, n_features), standardized
    y: np.ndarray            # (N,) binary outcome
    dates: np.ndarray        # (N,)
    market: np.ndarray       # (N,) market YES price (implied prob)
    highs: np.ndarray        # (N,) realized daily high (deg F)
    threshold_f: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    seq_len: int
    i_train_end: int         # train = [:i_train_end]
    i_val_end: int           # val = [i_train_end:i_val_end]; test = [i_val_end:]
    X_live: np.ndarray       # (1, seq_len, n_features) most recent window
    live_market_price: float

    # Convenience slices -----------------------------------------------------
    def train(self):
        s = slice(0, self.i_train_end)
        return self.X[s], self.y[s]

    def val(self):
        s = slice(self.i_train_end, self.i_val_end)
        return self.X[s], self.y[s], self.market[s], self.dates[s], self.highs[s]

    def test(self):
        s = slice(self.i_val_end, None)
        return self.X[s], self.y[s], self.market[s], self.dates[s], self.highs[s]

    @property
    def n_features(self) -> int:
        return self.X.shape[-1]


def load_features() -> pd.DataFrame:
    """Read the partitioned Parquet feature lake into a time-sorted DataFrame."""
    path = os.path.join(config.DATA_DIR, "features")
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_dataset(
    df: pd.DataFrame,
    seq_len: int = 14,
    val_frac: float = 0.15,
    test_frac: float = 0.2,
) -> Dataset:
    feats = df[FEATURE_COLS].to_numpy(dtype="float32")
    labels = df["label_high"].to_numpy(dtype="float32")
    market = df["market_yes_price"].to_numpy(dtype="float32")
    highs = df["tmax_f"].to_numpy(dtype="float32")
    dates = df["date"].to_numpy()
    # Dynamic contract threshold = the latest trailing "normal" (for display).
    threshold = float(df["threshold_f"].iloc[-1])

    X, y, d, m, h = [], [], [], [], []
    for i in range(seq_len, len(df)):
        X.append(feats[i - seq_len : i])   # past days only
        y.append(labels[i])                # next-day outcome
        d.append(dates[i])
        m.append(market[i])
        h.append(highs[i])
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="float32")
    d = np.asarray(d)
    m = np.asarray(m, dtype="float32")
    h = np.asarray(h, dtype="float32")

    n = len(X)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    i_train_end = n - n_test - n_val
    i_val_end = n - n_test

    # Standardize using TRAIN statistics only.
    flat = X[:i_train_end].reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    Xz = ((X - mean) / std).astype("float32")

    # Live window = last seq_len days of actual weather -> forecast "tomorrow".
    live_window = (feats[-seq_len:] - mean) / std
    X_live = live_window[np.newaxis, :, :].astype("float32")

    return Dataset(
        X=Xz,
        y=y,
        dates=d,
        market=m,
        highs=h,
        threshold_f=threshold,
        feature_mean=mean,
        feature_std=std,
        seq_len=seq_len,
        i_train_end=i_train_end,
        i_val_end=i_val_end,
        X_live=X_live,
        live_market_price=float(m[-1]) if len(m) else 0.5,
    )
