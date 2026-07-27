"""Ensemble of the TensorFlow LSTM and PyTorch Transformer.

Blends the two probability estimates with a convex weight ``w`` (weight on the
LSTM). The weight is a hyperparameter that ``strategy/bayesian_opt.py`` tunes.
The blended number is our "proprietary odds" - the probability we actually
trade against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.lstm_tf import LSTMForecaster
from ml.transformer_torch import TransformerForecaster


@dataclass
class EnsemblePrediction:
    lstm_prob: np.ndarray
    transformer_prob: np.ndarray
    ensemble_prob: np.ndarray


class WeatherEnsemble:
    def __init__(self, seq_len: int, n_features: int, weight_lstm: float = 0.5):
        self.weight_lstm = float(np.clip(weight_lstm, 0.0, 1.0))
        self.lstm = LSTMForecaster(seq_len, n_features)
        self.transformer = TransformerForecaster(seq_len, n_features)

    @property
    def backends(self) -> dict[str, str]:
        return {"lstm": self.lstm.backend, "transformer": self.transformer.backend}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeatherEnsemble":
        self.lstm.fit(X, y)
        self.transformer.fit(X, y)
        return self

    def set_weight(self, weight_lstm: float) -> None:
        self.weight_lstm = float(np.clip(weight_lstm, 0.0, 1.0))

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        p_lstm = np.asarray(self.lstm.predict_proba(X), dtype="float64")
        p_tfm = np.asarray(self.transformer.predict_proba(X), dtype="float64")
        w = self.weight_lstm
        blended = w * p_lstm + (1.0 - w) * p_tfm
        return EnsemblePrediction(
            lstm_prob=p_lstm,
            transformer_prob=p_tfm,
            ensemble_prob=np.clip(blended, 1e-4, 1 - 1e-4),
        )
