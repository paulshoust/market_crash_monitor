# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tzdata \
 && rm -rf /var/lib/apt/lists/*

# App dir is /market_monitor (not /app)
WORKDIR /market_monitor

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code & scripts
COPY market_monitor.py ./market_monitor.py
COPY scheduler.py ./scheduler.py
COPY entrypoint.sh ./entrypoint.sh
COPY healthcheck.sh ./healthcheck.sh
RUN chmod +x entrypoint.sh healthcheck.sh

# Non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

# Persistent state/logs
VOLUME ["/data"]
ENV DATA_DIR=/data
ENV TZ=Europe/Madrid

# Scheduling & health defaults
ENV SCHEDULE_CRON="0 17 * * 1" \
    RUN_AT_START=0 \
    HEALTH_HTTP=1 \
    HEALTH_PORT=8087 \
    HEALTH_MAX_AGE_MIN=180

# Optional health port
EXPOSE 8087

# Use our script for container health
HEALTHCHECK --interval=5m --timeout=10s --start-period=1m --retries=3 \
  CMD /market_monitor/healthcheck.sh || exit 1

ENTRYPOINT ["/market_monitor/entrypoint.sh"]
