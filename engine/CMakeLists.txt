cmake_minimum_required(VERSION 3.16)
project(kalshi_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Boost.Asio (header-only usage, but link Boost::system for error_code + threads).
find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)

# nlohmann_json (installed via apt: nlohmann-json3-dev).
find_package(nlohmann_json REQUIRED)

# libzmq + hiredis are located via their headers/libs installed by apt.
find_library(ZMQ_LIB zmq REQUIRED)
find_library(HIREDIS_LIB hiredis REQUIRED)

add_executable(engine
  src/main.cpp
  src/redis_client.cpp
)

target_include_directories(engine PRIVATE src)
target_link_libraries(engine
  PRIVATE
    Boost::system
    Threads::Threads
    nlohmann_json::nlohmann_json
    ${ZMQ_LIB}
    ${HIREDIS_LIB}
)
