"""Order-book snapshot generator (NumPy-only, no pandas).

Kept separate from ``ingestion/sources/synthetic.py`` (which depends on pandas)
so the lightweight market-data publisher image stays small.
"""

from __future__ import annotations

import numpy as np


def generate_orderbook_snapshot(
    venues: tuple[str, ...],
    fair_prob: float,
    rng: np.random.Generator,
    arb_bias: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Produce one top-of-book snapshot per venue around a fair probability.

    ``arb_bias`` optionally pushes one venue off fair to create a temporary
    cross-venue spread for the arbitrage engine to detect.
    """
    books = {}
    for i, v in enumerate(venues):
        skew = arb_bias if i == 0 else 0.0
        mid = float(np.clip(fair_prob + rng.normal(0, 0.004) + skew, 0.02, 0.98))
        half_spread = float(np.clip(rng.normal(0.006, 0.0015), 0.002, 0.02))
        books[v] = {
            "yes_bid": round(mid - half_spread, 4),
            "yes_ask": round(mid + half_spread, 4),
        }
    return books
