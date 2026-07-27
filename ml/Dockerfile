# ML service: TensorFlow (LSTM) + PyTorch (Transformer) + Optuna + strategy.
# Source is bind-mounted at runtime (see docker-compose.yml); this image only
# provides the Python environment.
FROM python:3.11-slim

# libgomp1 is required at runtime by both TensorFlow and PyTorch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.2.2

ENV PYTHONPATH=/app \
    TF_CPP_MIN_LOG_LEVEL=3 \
    OMP_NUM_THREADS=2
