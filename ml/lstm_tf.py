"""TensorFlow/Keras LSTM classifier for the binary weather contract.

Predicts P(next-day high > threshold) from a window of past weather. Kept small
so it trains in seconds on CPU. If TensorFlow is unavailable for any reason, a
lightweight NumPy logistic fallback keeps the pipeline running (and the demo is
honest about which path ran).
"""

from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf

    _HAVE_TF = True
except Exception:  # pragma: no cover - import guard
    _HAVE_TF = False


class LSTMForecaster:
    name = "lstm_tf"

    def __init__(self, seq_len: int, n_features: int, units: int = 32, seed: int = 7):
        self.seq_len = seq_len
        self.n_features = n_features
        self.units = units
        self.seed = seed
        self.model = None
        self.backend = "tensorflow" if _HAVE_TF else "numpy-fallback"
        self._fallback = None

    def _build(self):
        tf.random.set_seed(self.seed)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.seq_len, self.n_features)),
                tf.keras.layers.LSTM(self.units),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 25, batch_size: int = 32):
        if not _HAVE_TF:
            self._fallback = _LogisticFallback(seed=self.seed).fit(X, y)
            return self
        self.model = self._build()
        cb = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=4, restore_best_weights=True
        )
        self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[cb]
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not _HAVE_TF:
            return self._fallback.predict_proba(X)
        return self.model.predict(X, verbose=0).ravel()

    def save(self, path: str) -> None:
        if _HAVE_TF and self.model is not None:
            self.model.save(path)


class _LogisticFallback:
    """Minimal logistic regression on flattened features (no external deps)."""

    def __init__(self, seed: int = 0, lr: float = 0.05, epochs: int = 300):
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None

    def _feat(self, X: np.ndarray) -> np.ndarray:
        # Use summary stats of the window (mean + last step) as features.
        return np.concatenate([X.mean(axis=1), X[:, -1, :]], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        F = self._feat(X)
        self.mu = F.mean(axis=0)
        self.sd = F.std(axis=0) + 1e-6
        Fz = (F - self.mu) / self.sd
        n, d = Fz.shape
        self.w = np.zeros(d)
        for _ in range(self.epochs):
            z = Fz @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = Fz.T @ (p - y) / n
            grad_b = float((p - y).mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Fz = (self._feat(X) - self.mu) / self.sd
        z = Fz @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))
