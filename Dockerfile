# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl tzdata bash coreutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /market_monitor

# If you have a requirements.txt, keep this; otherwise inline-install.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy the app
COPY . .

# Make scripts runnable
RUN chmod +x entrypoint.sh \
    && install -m 0755 healthcheck.sh /usr/local/bin/healthcheck.sh

# Healthcheck: stale state file means "unhealthy"
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
  CMD /usr/local/bin/healthcheck.sh

ENTRYPOINT ["./entrypoint.sh"]
