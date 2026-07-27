"""Train the LSTM + Transformer ensemble on the feature lake.

Exposes ``train_ensemble`` (used by ``run_pipeline``) and a small CLI so you can
train and eyeball validation accuracy on its own:

    python -m ml.train
"""

from __future__ import annotations

import os

import numpy as np

from ml.dataset import Dataset, build_dataset, load_features
from ml.ensemble import WeatherEnsemble
from shared import config


def train_ensemble(ds: Dataset) -> WeatherEnsemble:
    X_train, y_train = ds.train()
    ens = WeatherEnsemble(seq_len=ds.seq_len, n_features=ds.n_features)
    ens.fit(X_train, y_train)
    return ens


def _accuracy(prob: np.ndarray, y: np.ndarray) -> float:
    return float(((prob >= 0.5).astype(int) == y.astype(int)).mean())


def main() -> int:
    df = load_features()
    ds = build_dataset(df)
    ens = train_ensemble(ds)

    X_val, y_val, *_ = ds.val()
    pred = ens.predict(X_val)
    print(f"[train] backends: {ens.backends}")
    print(f"[train] val accuracy  LSTM={_accuracy(pred.lstm_prob, y_val):.3f}  "
          f"Transformer={_accuracy(pred.transformer_prob, y_val):.3f}  "
          f"Ensemble={_accuracy(pred.ensemble_prob, y_val):.3f}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    try:
        ens.lstm.save(os.path.join(config.MODEL_DIR, "lstm.keras"))
        ens.transformer.save(os.path.join(config.MODEL_DIR, "transformer.pt"))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
