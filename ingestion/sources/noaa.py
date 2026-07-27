"""Real weather ingestion from NOAA / NCEI Climate Data Online (CDO).

Requires a free API token (env ``NOAA_TOKEN``). If the token is missing or any
request fails, this returns ``None`` and the caller falls back to synthetic
data. Kept deliberately small and defensive - the point is graceful degradation.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

try:
    import requests
except ImportError:  # requests may be absent in some minimal images
    requests = None  # type: ignore

CDO_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"


def fetch_weather(
    station: str,
    token: str,
    years: int = 6,
    end: date | None = None,
    timeout: int = 20,
) -> pd.DataFrame | None:
    """Fetch daily TMAX/TMIN for a station. Returns None on any failure.

    ``station`` should be a GHCND station id (e.g. "GHCND:USW00094728" for
    Central Park). We map friendly ids in the caller.
    """
    if not token or requests is None:
        return None

    end = end or date.today()
    start = end - timedelta(days=years * 365)
    headers = {"token": token}
    all_rows: list[dict] = []

    # CDO paginates and rate-limits; we page in yearly chunks to stay small.
    chunk_start = start
    try:
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=365), end)
            params = {
                "datasetid": "GHCND",
                "stationid": station,
                "datatypeid": ["TMAX", "TMIN"],
                "startdate": chunk_start.isoformat(),
                "enddate": chunk_end.isoformat(),
                "units": "standard",  # Fahrenheit
                "limit": 1000,
            }
            resp = requests.get(
                f"{CDO_BASE}/data", headers=headers, params=params, timeout=timeout
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            all_rows.extend(results)
            chunk_start = chunk_end + timedelta(days=1)
    except Exception:
        return None

    if not all_rows:
        return None

    raw = pd.DataFrame(all_rows)
    # Pivot TMAX/TMIN into columns keyed by date.
    raw["day"] = pd.to_datetime(raw["date"]).dt.normalize()
    wide = raw.pivot_table(index="day", columns="datatype", values="value", aggfunc="mean")
    if "TMAX" not in wide:
        return None
    out = pd.DataFrame(
        {
            "date": wide.index,
            "tmax_f": wide.get("TMAX"),
            "tmin_f": wide.get("TMIN"),
        }
    ).reset_index(drop=True)
    out = out.dropna(subset=["tmax_f"]).sort_values("date").reset_index(drop=True)
    # NOAA doesn't give us humidity/wind here; fill neutral placeholders so the
    # feature schema matches the synthetic source.
    out["humidity"] = 60.0
    out["wind_mph"] = 8.0
    return out
