"""Probability calibration (Platt scaling).

Neural nets - especially small ones on noisy targets - are often overconfident:
they push probabilities toward 0/1. Trading on raw, overconfident probabilities
massively overstates the edge. So we fit a 1-D logistic map on a *held-out
validation set*,

    p_calibrated = sigmoid(a * logit(p_raw) + b),

which rescales the model's confidence to match observed frequencies. This is the
difference between a backtest that looks fake and one that is believable.
"""

from __future__ import annotations

import numpy as np


class PlattScaler:
    def __init__(self, lr: float = 0.1, epochs: int = 2000, reg: float = 10.0):
        self.a = 1.0
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        # L2 regularization pulling the map toward the identity (a=1, b=0).
        # On a modest validation set this prevents the calibrator from
        # over-widening a weak model into overconfident (and unrealistic-edge)
        # probabilities.
        self.reg = reg

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattScaler":
        z = self._logit(np.asarray(p_raw, dtype="float64"))
        y = np.asarray(y, dtype="float64")
        n = max(1, len(z))
        for _ in range(self.epochs):
            logits = self.a * z + self.b
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_a = float(((p - y) * z).mean()) + self.reg / n * (self.a - 1.0)
            grad_b = float((p - y).mean()) + self.reg / n * self.b
            self.a -= self.lr * grad_a
            self.b -= self.lr * grad_b
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        z = self._logit(np.asarray(p_raw, dtype="float64"))
        return 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))
