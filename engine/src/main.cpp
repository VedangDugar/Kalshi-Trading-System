// Kalshi C++17 low-latency arbitrage engine.
//
// Architecture:
//   - A single Boost.Asio io_context drives everything (one thread, no locks).
//   - A ZeroMQ SUB socket feeds top-of-book quotes for 3 venues; its ZMQ_FD is
//     registered with Asio so the same event loop reacts to market data.
//   - ArbDetector flags cross-venue price dislocations.
//   - Detected opportunities + rolling health/latency stats are written to Redis
//     (hiredis) for the dashboard.
//   - A periodic Asio steady_timer publishes EngineStats once per second.
//
// The container is configured with `restart: unless-stopped`, so a crash is
// auto-recovered - the basis of the "high uptime" story.

#include <boost/asio.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "arb_detector.hpp"
#include "contracts.hpp"
#include "latency.hpp"
#include "redis_client.hpp"
#include "zmq_sub.hpp"

using json = nlohmann::json;
namespace asio = boost::asio;

static std::string env_or(const char* key, const std::string& def) {
  const char* v = std::getenv(key);
  return v ? std::string(v) : def;
}

static std::int64_t now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

class Engine {
 public:
  Engine(asio::io_context& io, RedisClient& redis, const std::string& md_addr)
      : io_(io),
        redis_(redis),
        sd_(io),
        stats_timer_(io),
        started_at_ns_(now_ns()) {
    sub_.connect(md_addr, contracts::TOPIC_PREFIX);
    sd_.assign(sub_.fd());
    std::cout << "[engine] subscribed to " << md_addr << " (prefix '"
              << contracts::TOPIC_PREFIX << "')" << std::endl;
  }

  void start() {
    schedule_read();
    schedule_stats();
  }

 private:
  void schedule_read() {
    sd_.async_wait(asio::posix::stream_descriptor::wait_read,
                   [this](const boost::system::error_code& ec) { on_readable(ec); });
  }

  void on_readable(const boost::system::error_code& ec) {
    if (ec) {
      std::cerr << "[engine] wait error: " << ec.message() << std::endl;
      schedule_read();
      return;
    }
    // ZMQ_FD is edge-triggered: drain everything currently available.
    std::string topic, payload;
    while (sub_.has_pollin()) {
      if (!sub_.recv(topic, payload)) break;
      process(payload);
    }
    schedule_read();
  }

  void process(const std::string& payload) {
    std::int64_t recv_ns = now_ns();
    json j;
    try {
      j = json::parse(payload);
    } catch (...) {
      return;
    }

    Quote q;
    q.yes_bid = j.value("yes_bid", 0.0);
    q.yes_ask = j.value("yes_ask", 0.0);
    q.seq = j.value("seq", 0);
    q.ts_ns = j.value("ts_ns", recv_ns);
    q.valid = true;
    const std::string venue = j.value("venue", "");
    const std::string market = j.value("market", "");

    detector_.update(venue, q);
    ++messages_processed_;
    ++msgs_since_tick_;

    // Pipeline latency: publisher timestamp -> engine decision.
    double latency_us = static_cast<double>(recv_ns - q.ts_ns) / 1000.0;
    if (latency_us >= 0) latency_.add(latency_us);

    if (auto sig = detector_.detect(market)) {
      sig->latency_us = static_cast<std::int64_t>(latency_us);
      sig->ts_ns = recv_ns;
      ++arb_opportunities_;
      ++trades_executed_;  // demo: we "execute" every detected opportunity
      publish_arb(*sig);
    }
  }

  void publish_arb(const ArbSignal& s) {
    json j = {
        {"market", s.market},   {"buy_venue", s.buy_venue},
        {"sell_venue", s.sell_venue}, {"buy_price", s.buy_price},
        {"sell_price", s.sell_price}, {"spread_pct", s.spread_pct},
        {"latency_us", s.latency_us}, {"ts_ns", s.ts_ns},
    };
    redis_.lpush_capped(contracts::ARB_SIGNALS_KEY, j.dump(),
                        contracts::ARB_SIGNALS_MAXLEN);
  }

  void schedule_stats() {
    stats_timer_.expires_after(std::chrono::seconds(1));
    stats_timer_.async_wait([this](const boost::system::error_code& ec) {
      if (!ec) {
        publish_stats();
        schedule_stats();
      }
    });
  }

  void publish_stats() {
    // Availability is only meaningful once the feed is established: don't
    // penalize the initial warmup before the first message arrives. After that,
    // an interval is "healthy" if we processed at least one message in it, so a
    // genuine feed outage correctly lowers the number.
    if (messages_processed_ > 0) {
      ++total_intervals_;
      if (msgs_since_tick_ > 0) ++healthy_intervals_;
    }
    double uptime_pct =
        total_intervals_ ? 100.0 * healthy_intervals_ / total_intervals_ : 100.0;
    double uptime_s = static_cast<double>(now_ns() - started_at_ns_) / 1e9;

    json j = {
        {"started_at_ns", started_at_ns_},
        {"uptime_seconds", uptime_s},
        {"messages_processed", messages_processed_},
        {"arb_opportunities", arb_opportunities_},
        {"trades_executed", trades_executed_},
        {"latency_us_p50", latency_.p50()},
        {"latency_us_p99", latency_.p99()},
        {"msgs_per_sec", static_cast<double>(msgs_since_tick_)},
        {"uptime_pct", uptime_pct},
    };
    redis_.set(contracts::ENGINE_STATS_KEY, j.dump());

    std::cout << "[engine] msgs=" << messages_processed_
              << " arb=" << arb_opportunities_
              << " p50=" << latency_.p50() << "us"
              << " p99=" << latency_.p99() << "us"
              << " mps=" << msgs_since_tick_
              << " uptime=" << uptime_pct << "%" << std::endl;

    msgs_since_tick_ = 0;
  }

  asio::io_context& io_;
  RedisClient& redis_;
  asio::posix::stream_descriptor sd_;
  asio::steady_timer stats_timer_;
  ZmqSubscriber sub_;
  ArbDetector detector_;
  LatencyStats latency_;

  std::int64_t started_at_ns_;
  std::uint64_t messages_processed_ = 0;
  std::uint64_t msgs_since_tick_ = 0;
  std::uint64_t arb_opportunities_ = 0;
  std::uint64_t trades_executed_ = 0;
  std::uint64_t total_intervals_ = 0;
  std::uint64_t healthy_intervals_ = 0;
};

int main() {
  const std::string redis_host = env_or("REDIS_HOST", "redis");
  const int redis_port = std::stoi(env_or("REDIS_PORT", "6379"));
  const std::string md_host = env_or("MD_PUB_HOST", "marketdata");
  const std::string md_port = env_or("MD_PUB_PORT", "5556");
  const std::string md_addr = "tcp://" + md_host + ":" + md_port;

  std::cout << "[engine] starting C++17 arbitrage engine" << std::endl;

  RedisClient redis;
  if (!redis.connect(redis_host, redis_port)) {
    std::cerr << "[engine] FATAL: could not connect to Redis" << std::endl;
    return 1;
  }

  try {
    asio::io_context io;
    Engine engine(io, redis, md_addr);
    engine.start();
    io.run();
  } catch (const std::exception& e) {
    std::cerr << "[engine] FATAL: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
