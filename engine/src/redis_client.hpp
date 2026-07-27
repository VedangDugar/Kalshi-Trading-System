// Minimal Redis client wrapper around hiredis (synchronous, connection-retrying).
#pragma once

#include <string>

struct redisContext;  // forward declare to keep hiredis out of the header

class RedisClient {
 public:
  RedisClient() = default;
  ~RedisClient();

  // Connect with retries; returns true once connected.
  bool connect(const std::string& host, int port, int max_retries = 30);

  bool ok() const { return ctx_ != nullptr; }

  // SET key value
  void set(const std::string& key, const std::string& value);

  // LPUSH key value; then LTRIM key 0 maxlen-1 (keep newest N).
  void lpush_capped(const std::string& key, const std::string& value, int maxlen);

 private:
  void reconnect();

  redisContext* ctx_ = nullptr;
  std::string host_;
  int port_ = 6379;
};
