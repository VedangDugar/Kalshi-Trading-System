"""Turn a model probability into a trade decision vs. the market.

For a binary contract, YES trades at price ``q`` (the market-implied
probability) and pays 1 if it resolves YES. If our model says the true
probability is ``p``:
  - buying YES has expected edge  (p - q)
  - buying NO  has expected edge  (q - p)
We take whichever side is positive, and only if the edge clears a threshold
(tuned by Bayesian optimization) to avoid trading on noise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    side: str          # "YES", "NO", or "PASS"
    edge: float        # signed edge for the chosen side (>=0 unless PASS)
    model_prob: float  # p
    market_prob: float # q
    fair_yes_price: float


def decide(model_prob: float, market_prob: float, edge_threshold: float = 0.0) -> Decision:
    p = float(min(max(model_prob, 1e-4), 1 - 1e-4))
    q = float(min(max(market_prob, 1e-4), 1 - 1e-4))
    yes_edge = p - q
    no_edge = q - p

    if yes_edge >= no_edge and yes_edge > edge_threshold:
        return Decision("YES", yes_edge, p, q, p)
    if no_edge > edge_threshold:
        return Decision("NO", no_edge, p, q, p)
    return Decision("PASS", 0.0, p, q, p)
