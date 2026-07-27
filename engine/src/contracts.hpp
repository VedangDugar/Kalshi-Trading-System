// Mirror of shared/schemas.py so the C++ engine agrees with the Python services
// on Redis keys and ZeroMQ topics. Keep in sync with shared/schemas.py.
#pragma once

#include <string>

namespace contracts {

// ZeroMQ SUB prefix: publisher sends topics like "book.kalshi".
inline const std::string TOPIC_PREFIX = "book.";

// Redis keys.
inline const std::string ARB_SIGNALS_KEY = "kalshi:arb:signals";   // LIST
inline const std::string ENGINE_STATS_KEY = "kalshi:engine:stats"; // JSON

inline constexpr int ARB_SIGNALS_MAXLEN = 200;

// Minimum profitable spread to act on, as a fraction (0.0005 = 0.05%).
inline constexpr double MIN_SPREAD_PCT = 0.0005;

}  // namespace contracts
