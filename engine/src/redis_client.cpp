#include "redis_client.hpp"

#include <hiredis/hiredis.h>

#include <chrono>
#include <cstdio>
#include <thread>

RedisClient::~RedisClient() {
  if (ctx_) {
    redisFree(ctx_);
    ctx_ = nullptr;
  }
}

bool RedisClient::connect(const std::string& host, int port, int max_retries) {
  host_ = host;
  port_ = port;
  for (int attempt = 0; attempt < max_retries; ++attempt) {
    redisContext* c = redisConnect(host.c_str(), port);
    if (c != nullptr && c->err == 0) {
      ctx_ = c;
      std::printf("[engine] connected to Redis %s:%d\n", host.c_str(), port);
      return true;
    }
    if (c) redisFree(c);
    std::printf("[engine] Redis not ready (attempt %d/%d), retrying...\n",
                attempt + 1, max_retries);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return false;
}

void RedisClient::reconnect() {
  if (ctx_) {
    redisFree(ctx_);
    ctx_ = nullptr;
  }
  redisContext* c = redisConnect(host_.c_str(), port_);
  if (c != nullptr && c->err == 0) {
    ctx_ = c;
  } else if (c) {
    redisFree(c);
  }
}

void RedisClient::set(const std::string& key, const std::string& value) {
  if (!ctx_) reconnect();
  if (!ctx_) return;
  redisReply* reply = static_cast<redisReply*>(
      redisCommand(ctx_, "SET %s %s", key.c_str(), value.c_str()));
  if (reply == nullptr) {
    reconnect();
    return;
  }
  freeReplyObject(reply);
}

void RedisClient::lpush_capped(const std::string& key, const std::string& value,
                               int maxlen) {
  if (!ctx_) reconnect();
  if (!ctx_) return;
  redisReply* r1 = static_cast<redisReply*>(
      redisCommand(ctx_, "LPUSH %s %s", key.c_str(), value.c_str()));
  if (r1 == nullptr) {
    reconnect();
    return;
  }
  freeReplyObject(r1);
  redisReply* r2 = static_cast<redisReply*>(
      redisCommand(ctx_, "LTRIM %s 0 %d", key.c_str(), maxlen - 1));
  if (r2) freeReplyObject(r2);
}
