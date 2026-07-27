"""Fractional Kelly position sizing for binary contracts.

For a YES bet at price ``q`` with true win probability ``p`` (contract pays 1),
the full-Kelly fraction of bankroll simplifies neatly to:

    f* = (p - q) / (1 - q)

and symmetrically for a NO bet at price ``1 - q`` with win prob ``1 - p``:

    f* = (q - p) / q

We then apply a safety multiplier (half-Kelly by default) and a hard cap,
because full Kelly is famously too aggressive in the presence of estimation
error - a point worth making in an interview.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared import config


@dataclass
class Sizing:
    side: str
    kelly_full: float   # raw f*
    kelly_used: float   # after safety fraction + cap
    stake_usd: float


def kelly_fraction(side: str, model_prob: float, market_prob: float) -> float:
    p = float(min(max(model_prob, 1e-4), 1 - 1e-4))
    q = float(min(max(market_prob, 1e-4), 1 - 1e-4))
    if side == "YES":
        f = (p - q) / (1.0 - q)
    elif side == "NO":
        f = (q - p) / q
    else:
        return 0.0
    return max(0.0, f)


def size_position(
    side: str,
    model_prob: float,
    market_prob: float,
    bankroll_usd: float,
    safety_fraction: float | None = None,
    cap: float | None = None,
) -> Sizing:
    """Return sizing for a single contract decision."""
    safety = config.KELLY_FRACTION if safety_fraction is None else safety_fraction
    cap = config.KELLY_CAP if cap is None else cap
    f_full = kelly_fraction(side, model_prob, market_prob)
    f_used = min(f_full * safety, cap)
    return Sizing(
        side=side,
        kelly_full=f_full,
        kelly_used=f_used,
        stake_usd=round(f_used * bankroll_usd, 2),
    )
