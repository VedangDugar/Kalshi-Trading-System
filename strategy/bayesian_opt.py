"""Bayesian optimization of strategy hyperparameters (Optuna).

We tune two knobs on the *validation* window (never the test window):
  - ``weight_lstm``: how much to trust the LSTM vs. the Transformer in the blend
  - ``edge_threshold``: how large an edge must be before we trade

Optuna's TPE sampler is a form of Bayesian optimization (it models P(params |
score) and proposes promising points). If Optuna is unavailable, we fall back to
a deterministic grid search so the pipeline still produces tuned params.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategy.backtest import run_backtest

try:
    import optuna

    _HAVE_OPTUNA = True
except Exception:  # pragma: no cover
    _HAVE_OPTUNA = False


@dataclass
class BestParams:
    weight_lstm: float
    edge_threshold: float
    score: float
    method: str


def _score(weight_lstm, edge_threshold, lstm_p, tfm_p, market, outcome, dates) -> float:
    blended = weight_lstm * lstm_p + (1 - weight_lstm) * tfm_p
    blended = np.clip(blended, 1e-4, 1 - 1e-4)
    res = run_backtest(blended, market, outcome, dates, edge_threshold=edge_threshold)
    m = res.metrics
    # Maximize risk-adjusted growth; require enough activity to be meaningful.
    if m.n_trades < 3:
        return -1.0 + 0.001 * m.n_trades
    return m.total_return - 0.5 * m.max_drawdown


def optimize(
    lstm_p: np.ndarray,
    tfm_p: np.ndarray,
    market: np.ndarray,
    outcome: np.ndarray,
    dates: np.ndarray,
    n_trials: int = 40,
) -> BestParams:
    lstm_p = np.asarray(lstm_p, dtype="float64")
    tfm_p = np.asarray(tfm_p, dtype="float64")

    if _HAVE_OPTUNA:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            w = trial.suggest_float("weight_lstm", 0.0, 1.0)
            thr = trial.suggest_float("edge_threshold", 0.005, 0.04)
            return _score(w, thr, lstm_p, tfm_p, market, outcome, dates)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=17)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        bp = study.best_params
        return BestParams(
            weight_lstm=float(bp["weight_lstm"]),
            edge_threshold=float(bp["edge_threshold"]),
            score=float(study.best_value),
            method="optuna-tpe",
        )

    # ---- grid-search fallback -------------------------------------------
    best = BestParams(0.5, 0.02, -1e9, "grid-search")
    for w in np.linspace(0, 1, 9):
        for thr in np.linspace(0.005, 0.04, 12):
            s = _score(w, thr, lstm_p, tfm_p, market, outcome, dates)
            if s > best.score:
                best = BestParams(float(w), float(thr), float(s), "grid-search")
    return best
