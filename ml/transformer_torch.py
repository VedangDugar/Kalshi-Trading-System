"""PyTorch Transformer-encoder classifier for the binary weather contract.

A small self-attention encoder over the weather sequence, mean-pooled into a
sigmoid head. Complements the LSTM: the two make different errors, so averaging
them (see ``ensemble.py``) is more robust than either alone. Falls back to the
same NumPy logistic model as the LSTM if PyTorch is unavailable.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - import guard
    _HAVE_TORCH = False

from ml.lstm_tf import _LogisticFallback


if _HAVE_TORCH:

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 64):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class _TransformerNet(nn.Module):
        def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4, layers: int = 2):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            self.posenc = _PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=64,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1)
            )

        def forward(self, x):
            h = self.posenc(self.proj(x))
            h = self.encoder(h)
            h = h.mean(dim=1)  # mean-pool over time
            return self.head(h).squeeze(-1)


class TransformerForecaster:
    name = "transformer_torch"

    def __init__(self, seq_len: int, n_features: int, seed: int = 13):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.model = None
        self.backend = "pytorch" if _HAVE_TORCH else "numpy-fallback"
        self._fallback = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 40, batch_size: int = 32, lr: float = 1e-3):
        if not _HAVE_TORCH:
            self._fallback = _LogisticFallback(seed=self.seed).fit(X, y)
            return self

        torch.manual_seed(self.seed)
        self.model = _TransformerNet(self.n_features)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)

        self.model.train()
        n = len(Xt)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                opt.zero_grad()
                logits = self.model(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not _HAVE_TORCH:
            return self._fallback.predict_proba(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            return torch.sigmoid(logits).numpy().ravel()

    def save(self, path: str) -> None:
        if _HAVE_TORCH and self.model is not None:
            torch.save(self.model.state_dict(), path)
