# PySpark ingestion image. Rather than depend on a third-party Spark base image
# (whose tags change), we use a slim Python image + a JDK and install PySpark via
# pip - the pip package bundles a full Spark distribution and runs the job in
# local mode. Source code is bind-mounted at runtime (see docker-compose.yml).
FROM python:3.11-slim

# Java runtime required by Spark's JVM (OpenJDK 17 works with Spark 3.5).
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "pyspark==3.5.3" \
    "numpy<2" \
    pandas \
    pyarrow \
    requests \
    redis

ENV PYTHONPATH=/app \
    JAVA_HOME=/usr/lib/jvm/default-java
WORKDIR /app
