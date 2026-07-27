"""Best-effort Kalshi public market metadata.

Kalshi exposes a public REST API for market listings. We use it only to confirm
connectivity / list live weather markets for context; historical minute data is
not freely available, so the backtest uses the market history synthesized in
``synthetic.py`` (documented honestly in the README). Any failure -> None.
"""

from __future__ import annotations

try:
    import requests
except ImportError:
    requests = None  # type: ignore


def fetch_weather_markets(api_base: str, limit: int = 20, timeout: int = 10) -> list[dict] | None:
    """Return a list of live weather-related markets, or None on failure."""
    if requests is None:
        return None
    try:
        resp = requests.get(
            f"{api_base}/markets",
            params={"limit": limit, "status": "open"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        markets = resp.json().get("markets", [])
        weather = [
            m
            for m in markets
            if any(k in (m.get("title", "") + m.get("ticker", "")).lower()
                   for k in ("temp", "weather", "high", "rain", "snow"))
        ]
        return weather or markets
    except Exception:
        return None
