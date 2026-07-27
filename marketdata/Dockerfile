# Market-data simulator: lightweight Python + ZeroMQ.
FROM python:3.11-slim

WORKDIR /app
COPY marketdata/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/app
