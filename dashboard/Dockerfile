# Streamlit dashboard. Source is bind-mounted at runtime (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app
COPY dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/app
EXPOSE 8501
