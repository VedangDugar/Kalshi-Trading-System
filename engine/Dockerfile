# C++17 arbitrage engine. Single-stage build for robustness (the build toolchain
# stays in the image, but the demo prioritizes "it just builds and runs" over a
# minimal image size).
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-system-dev \
    libzmq3-dev \
    libhiredis-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt .
COPY src ./src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j \
    && cp build/engine /usr/local/bin/engine

ENTRYPOINT ["/usr/local/bin/engine"]
