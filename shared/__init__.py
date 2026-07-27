"""Shared contracts used across every service in the Kalshi Trading System.

This package is intentionally dependency-free (standard library only) so that
the Python ML/strategy services, the market-data publisher, and the dashboard
can all import the *same* Redis key names and JSON message schemas. The C++
engine mirrors these constants in ``engine/src/contracts.hpp``.
"""

from . import schemas  # noqa: F401
from . import config  # noqa: F401
