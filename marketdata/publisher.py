"""ZeroMQ market-data simulator for three venues.

Publishes top-of-book quotes for the same binary contract on Kalshi, Polymarket
and Robinhood over a ZeroMQ PUB socket. The three venues track a common "fair"
probability (a slow random walk, nudged toward the ML forecast in Redis when
available), but occasionally one venue is knocked off fair - creating a
transient cross-venue spread of ~0.05-0.15% that the C++ engine can arbitrage.

Message framing (multipart):
    frame 0: topic  = "book.<venue>"   (SUB prefix subscription)
    frame 1: payload = JSON Quote       (see shared/schemas.py)
"""

from __future__ import annotations

import json
import time

import numpy as np
import zmq

from marketdata.orderbook import generate_orderbook_snapshot
from shared import config
from shared.schemas import Keys, Quote, md_topic


def _load_fair_from_redis(r) -> float | None:
    try:
        raw = r.get(Keys.FORECAST_LATEST)
        if raw:
            return float(json.loads(raw).get("ensemble_prob"))
    except Exception:
        pass
    return None


def main() -> None:
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(config.md_pub_bind_addr())
    print(f"[marketdata] PUB bound at {config.md_pub_bind_addr()}", flush=True)

    # Optional Redis for latest-quote mirroring + reading the ML fair value.
    try:
        from shared.store import get_redis

        r = get_redis()
    except Exception:
        r = None

    rng = np.random.default_rng(123)
    fair = 0.5
    seq = 0
    tick_hz = 60            # per-venue publish rate target
    dt = 1.0 / tick_hz
    arb_active_until = 0.0
    arb_bias = 0.0
    next_fair_pull = 0.0

    while True:
        now = time.time()

        # Occasionally refresh the fair value toward the ML forecast.
        if now >= next_fair_pull and r is not None:
            model_fair = _load_fair_from_redis(r)
            if model_fair is not None:
                fair = 0.7 * fair + 0.3 * model_fair
            next_fair_pull = now + 5.0

        # Slow random walk for the fair probability.
        fair = float(np.clip(fair + rng.normal(0, 0.002), 0.05, 0.95))

        # Randomly open a short-lived arbitrage window on venue[0].
        if now > arb_active_until and rng.random() < 0.02:
            # 0.05% - 0.15% exploitable spread once the half-spreads are crossed.
            arb_bias = float(rng.choice([-1, 1])) * float(rng.uniform(0.010, 0.018))
            arb_active_until = now + rng.uniform(0.15, 0.4)
        if now > arb_active_until:
            arb_bias = 0.0

        books = generate_orderbook_snapshot(config.VENUES, fair, rng, arb_bias)
        for venue, book in books.items():
            seq += 1
            quote = Quote(
                venue=venue,
                market=config.MARKET_ID,
                yes_bid=book["yes_bid"],
                yes_ask=book["yes_ask"],
                seq=seq,
            )
            payload = json.dumps(quote.to_dict())
            pub.send_multipart([md_topic(venue).encode(), payload.encode()])
            if r is not None:
                try:
                    r.set(Keys.market_quote(venue), payload)
                except Exception:
                    pass

        time.sleep(dt)


if __name__ == "__main__":
    main()
