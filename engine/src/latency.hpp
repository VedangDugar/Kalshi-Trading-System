// Rolling latency statistics (microseconds) with p50/p99 percentiles.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

class LatencyStats {
 public:
  explicit LatencyStats(std::size_t window = 8192) : window_(window) {
    samples_.reserve(window);
  }

  void add(double us) {
    if (samples_.size() < window_) {
      samples_.push_back(us);
    } else {
      samples_[cursor_] = us;  // ring overwrite
    }
    cursor_ = (cursor_ + 1) % window_;
  }

  double percentile(double p) const {
    if (samples_.empty()) return 0.0;
    std::vector<double> sorted(samples_);
    std::sort(sorted.begin(), sorted.end());
    std::size_t idx = static_cast<std::size_t>(p / 100.0 * (sorted.size() - 1));
    return sorted[idx];
  }

  double p50() const { return percentile(50.0); }
  double p99() const { return percentile(99.0); }
  std::size_t count() const { return samples_.size(); }

 private:
  std::size_t window_;
  std::vector<double> samples_;
  std::size_t cursor_ = 0;
};
