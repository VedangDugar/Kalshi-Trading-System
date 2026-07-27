// Cross-venue arbitrage detection for a single binary contract.
//
// Keeps the latest top-of-book per venue. An arbitrage exists when we can BUY
// YES on one venue at its ask and simultaneously SELL YES on another venue at a
// higher bid: profit = best_bid - best_ask (> 0). We report it as a fraction of
// the buy price (spread_pct) and only if it clears MIN_SPREAD_PCT.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "contracts.hpp"

struct Quote {
  double yes_bid = 0.0;
  double yes_ask = 0.0;
  std::int64_t seq = 0;
  std::int64_t ts_ns = 0;
  bool valid = false;
};

struct ArbSignal {
  std::string market;
  std::string buy_venue;
  std::string sell_venue;
  double buy_price = 0.0;
  double sell_price = 0.0;
  double spread_pct = 0.0;
  std::int64_t latency_us = 0;
  std::int64_t ts_ns = 0;
};

class ArbDetector {
 public:
  void update(const std::string& venue, const Quote& q) { books_[venue] = q; }

  // Returns the best available arbitrage across all known venues, if any.
  std::optional<ArbSignal> detect(const std::string& market) const {
    std::string buy_venue, sell_venue;
    double best_ask = 1e18, best_bid = -1e18;

    for (const auto& [venue, q] : books_) {
      if (!q.valid) continue;
      if (q.yes_ask < best_ask) {
        best_ask = q.yes_ask;
        buy_venue = venue;
      }
      if (q.yes_bid > best_bid) {
        best_bid = q.yes_bid;
        sell_venue = venue;
      }
    }

    if (buy_venue.empty() || sell_venue.empty() || buy_venue == sell_venue) {
      return std::nullopt;
    }
    if (best_bid <= best_ask) return std::nullopt;

    const double spread_pct = (best_bid - best_ask) / best_ask;
    if (spread_pct < contracts::MIN_SPREAD_PCT) return std::nullopt;

    ArbSignal sig;
    sig.market = market;
    sig.buy_venue = buy_venue;
    sig.sell_venue = sell_venue;
    sig.buy_price = best_ask;
    sig.sell_price = best_bid;
    sig.spread_pct = spread_pct;
    return sig;
  }

  std::size_t venue_count() const { return books_.size(); }

 private:
  std::unordered_map<std::string, Quote> books_;
};
