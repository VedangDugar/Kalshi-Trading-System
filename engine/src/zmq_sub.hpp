// ZeroMQ SUB socket wrapper (C API) designed to integrate with Boost.Asio.
//
// ZeroMQ exposes an OS file descriptor (ZMQ_FD) that becomes readable when the
// socket needs attention. We hand that fd to Boost.Asio, which lets a single
// Asio io_context drive both the market-data feed and periodic timers - the
// idiomatic way to marry ZeroMQ with Asio's async model.
//
// IMPORTANT: ZMQ_FD is edge-triggered and only indicates "check me"; you must
// drain with ZMQ_DONTWAIT while (ZMQ_EVENTS & ZMQ_POLLIN) is set.
#pragma once

#include <zmq.h>

#include <cstdint>
#include <stdexcept>
#include <string>

class ZmqSubscriber {
 public:
  ZmqSubscriber() {
    ctx_ = zmq_ctx_new();
    sock_ = zmq_socket(ctx_, ZMQ_SUB);
  }

  ~ZmqSubscriber() {
    if (sock_) zmq_close(sock_);
    if (ctx_) zmq_ctx_term(ctx_);
  }

  void connect(const std::string& addr, const std::string& prefix) {
    if (zmq_connect(sock_, addr.c_str()) != 0) {
      throw std::runtime_error("zmq_connect failed: " + addr);
    }
    zmq_setsockopt(sock_, ZMQ_SUBSCRIBE, prefix.data(), prefix.size());
  }

  int fd() const {
    int fd = 0;
    size_t len = sizeof(fd);
    zmq_getsockopt(sock_, ZMQ_FD, &fd, &len);
    return fd;
  }

  bool has_pollin() const {
    uint32_t events = 0;
    size_t len = sizeof(events);
    zmq_getsockopt(sock_, ZMQ_EVENTS, &events, &len);
    return (events & ZMQ_POLLIN) != 0;
  }

  // Non-blocking receive of one [topic, payload] message. Returns false when
  // there is nothing to read.
  bool recv(std::string& topic, std::string& payload) {
    zmq_msg_t tmsg;
    zmq_msg_init(&tmsg);
    int rc = zmq_msg_recv(&tmsg, sock_, ZMQ_DONTWAIT);
    if (rc < 0) {
      zmq_msg_close(&tmsg);
      return false;
    }
    topic.assign(static_cast<char*>(zmq_msg_data(&tmsg)), zmq_msg_size(&tmsg));
    zmq_msg_close(&tmsg);

    zmq_msg_t pmsg;
    zmq_msg_init(&pmsg);
    rc = zmq_msg_recv(&pmsg, sock_, 0);  // second frame is guaranteed present
    if (rc < 0) {
      zmq_msg_close(&pmsg);
      return false;
    }
    payload.assign(static_cast<char*>(zmq_msg_data(&pmsg)), zmq_msg_size(&pmsg));
    zmq_msg_close(&pmsg);
    return true;
  }

 private:
  void* ctx_ = nullptr;
  void* sock_ = nullptr;
};
