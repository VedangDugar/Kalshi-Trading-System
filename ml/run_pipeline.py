"""End-to-end ML + strategy pipeline (the `ml` service entrypoint).

Sequence:
  1. Ensure the Parquet feature lake exists (Spark normally builds it; if not,
     fall back to a local synthetic build so this service can run standalone).
  2. Build time-ordered train/val/test sequences.
  3. Train the LSTM (TensorFlow) + Transformer (PyTorch) ensemble.
  4. Bayesian-optimize the blend weight + edge threshold on validation.
  5. Backtest on the untouched test window -> edge / MoM ROI / drawdown / Sharpe.
  6. Produce a live "tomorrow" forecast, edge, and Kelly-sized position.
  7. Publish everything to Redis for the dashboard, then keep the live forecast
     refreshing on a loop so the demo feels alive.
"""

from __future__ import annotations

import json
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from ml.calibration import PlattScaler
from ml.dataset import build_dataset, load_features
from ml.train import train_ensemble
from shared import config
from shared.schemas import (
    BacktestMetrics,
    ForecastMessage,
    Keys,
    KellyMessage,
    OddsMessage,
)
from strategy import bayesian_opt, kelly, odds
from strategy.backtest import run_backtest


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def _status(r, stage: str, detail: str = "") -> None:
    try:
        r.set(Keys.PIPELINE_STATUS, json.dumps({"stage": stage, "detail": detail, "ts": time.time()}))
    except Exception:
        pass


def ensure_features():
    """Load the feature lake; if absent, build a synthetic one locally."""
    try:
        df = load_features()
        if len(df) > 100:
            _log(f"loaded feature lake: {len(df)} rows")
            return df
    except Exception:
        pass

    _log("feature lake missing -> building synthetic features locally")
    from ingestion.features_pandas import build_features_pandas
    from ingestion.sources import synthetic

    raw = synthetic.generate_weather(
        config.DEFAULT_CITY, years=6, threshold_f=config.TEMP_THRESHOLD_F
    )
    df = build_features_pandas(raw, config.TEMP_THRESHOLD_F)
    out = os.path.join(config.DATA_DIR, "features")
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, "part-local.parquet"), index=False)
    _log(f"wrote synthetic feature lake: {len(df)} rows")
    return df


def run_once():
    from shared.store import get_redis

    r = get_redis()
    _status(r, "loading")
    df = ensure_features()
    ds = build_dataset(df)

    _status(r, "training", "LSTM + Transformer")
    _log("training ensemble (LSTM/TF + Transformer/PyTorch)...")
    ens = train_ensemble(ds)
    _log(f"backends: {ens.backends}")

    # Predictions on validation + test using each sub-model.
    X_val, y_val, m_val, d_val, _ = ds.val()
    X_test, y_test, m_test, d_test, _ = ds.test()
    val_pred = ens.predict(X_val)
    test_pred = ens.predict(X_test)

    # Bayesian optimization on validation only.
    _status(r, "optimizing", "Bayesian (Optuna)")
    _log("running Bayesian optimization of blend weight + edge threshold...")
    best = bayesian_opt.optimize(
        val_pred.lstm_prob, val_pred.transformer_prob, m_val, y_val, d_val
    )
    _log(
        f"best params: weight_lstm={best.weight_lstm:.3f} "
        f"edge_threshold={best.edge_threshold:.3f} "
        f"(score={best.score:.3f}, method={best.method})"
    )
    ens.set_weight(best.weight_lstm)

    # Blended probabilities with the tuned weight.
    def _blend(pred):
        return np.clip(
            best.weight_lstm * pred.lstm_prob
            + (1 - best.weight_lstm) * pred.transformer_prob,
            1e-4, 1 - 1e-4,
        )

    val_blend = _blend(val_pred)
    test_blend = _blend(test_pred)

    # Calibrate confidence on validation, then apply to test (avoids the
    # overconfidence that would fake an unrealistically large edge).
    _log("calibrating ensemble probabilities (Platt scaling) on validation...")
    scaler = PlattScaler().fit(val_blend, y_val)
    test_cal = scaler.transform(test_blend)
    _log(f"calibration params: a={scaler.a:.3f} b={scaler.b:.3f}")

    _status(r, "backtesting")
    _log("backtesting on out-of-sample test window...")
    bt = run_backtest(
        test_cal, m_test, y_test, d_test, edge_threshold=best.edge_threshold
    )
    m = bt.metrics
    _log(
        f"BACKTEST: trades={m.n_trades} avg_edge={m.avg_edge:.3%} "
        f"hit_rate={m.hit_rate:.1%} monthly_roi={m.monthly_roi:.2%} "
        f"max_dd={m.max_drawdown:.2%} sharpe={m.sharpe:.2f} "
        f"final=${m.final_bankroll:,.0f}"
    )

    # Persist backtest results + model metadata.
    r.set(Keys.BACKTEST_METRICS, json.dumps({
        **m.to_dict(),
        "backends": ens.backends,
        "opt_method": best.method,
        "weight_lstm": best.weight_lstm,
        "edge_threshold": best.edge_threshold,
    }))
    r.set(Keys.BACKTEST_EQUITY, json.dumps(bt.equity_curve))

    return r, ens, ds, best, scaler


def publish_live(r, ens, ds, best, scaler, jitter: bool = False):
    """Compute + publish the live 'tomorrow' forecast, odds and Kelly sizing."""
    live_pred = ens.predict(ds.X_live)
    p_lstm = float(live_pred.lstm_prob[0])
    p_tfm = float(live_pred.transformer_prob[0])
    # Calibrated ensemble probability (our tradeable "proprietary odds").
    raw_blend = best.weight_lstm * p_lstm + (1 - best.weight_lstm) * p_tfm
    p_ens = float(scaler.transform(np.array([raw_blend]))[0])

    # Simulate a slowly-moving market quote around the last observed price so
    # the dashboard's edge updates live (documented as demo-only movement).
    market_price = ds.live_market_price
    if jitter:
        market_price = float(np.clip(market_price + np.random.normal(0, 0.01), 0.02, 0.98))

    last_date = pd.Timestamp(ds.dates[-1])
    target_date = (last_date + timedelta(days=1)).date().isoformat()

    fc = ForecastMessage(
        city=config.DEFAULT_CITY,
        market=config.MARKET_ID,
        target_date=target_date,
        threshold_f=ds.threshold_f,
        lstm_prob=round(p_lstm, 4),
        transformer_prob=round(p_tfm, 4),
        ensemble_prob=round(p_ens, 4),
        predicted_high_f=round(_prob_to_high(p_ens, ds.threshold_f), 1),
    )
    r.set(Keys.FORECAST_LATEST, json.dumps(fc.to_dict()))

    decision = odds.decide(p_ens, market_price, best.edge_threshold)
    om = OddsMessage(
        market=config.MARKET_ID,
        model_prob=round(p_ens, 4),
        market_prob=round(market_price, 4),
        edge=round(p_ens - market_price, 4),
        fair_yes_price=round(p_ens, 4),
    )
    r.set(Keys.ODDS_LATEST, json.dumps(om.to_dict()))

    if decision.side != "PASS":
        sizing = kelly.size_position(
            decision.side, p_ens, market_price, config.BANKROLL_USD
        )
    else:
        sizing = kelly.Sizing("PASS", 0.0, 0.0, 0.0)

    km = KellyMessage(
        market=config.MARKET_ID,
        side=decision.side,
        edge=round(decision.edge, 4),
        kelly_fraction_full=round(sizing.kelly_full, 4),
        kelly_fraction_used=round(sizing.kelly_used, 4),
        stake_usd=sizing.stake_usd,
        bankroll_usd=config.BANKROLL_USD,
    )
    r.set(Keys.KELLY_LATEST, json.dumps(km.to_dict()))


def _prob_to_high(prob: float, threshold: float) -> float:
    """Rough inverse: map P(high>threshold) back to a point estimate of the high.

    Assumes a ~7F daily forecast std; purely for a human-readable number.
    """
    from math import sqrt

    import numpy as _np

    # inverse normal via numpy (erfinv-free approximation)
    p = min(max(prob, 1e-3), 1 - 1e-3)
    # z such that P(Z> -z)=p  => z = Phi^{-1}(p)
    z = _np_invnorm(p)
    return threshold + z * 7.0


def _np_invnorm(p: float) -> float:
    # Acklam's rational approximation to the inverse normal CDF.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = (2 * __import__("math").log(p)) ** 0.5 * -1
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = (2 * __import__("math").log(1 - p)) ** 0.5 * -1
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def main() -> int:
    r, ens, ds, best, scaler = run_once()
    _status(r, "live", "publishing live forecast")
    publish_live(r, ens, ds, best, scaler, jitter=False)
    _log("initial results published to Redis. Entering live-refresh loop...")

    # Keep the live forecast/edge refreshing so the dashboard feels live.
    while True:
        try:
            publish_live(r, ens, ds, best, scaler, jitter=True)
            _status(r, "live", "refreshing")
        except Exception as e:  # keep the service alive for the demo
            _log(f"refresh error (continuing): {e}")
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
