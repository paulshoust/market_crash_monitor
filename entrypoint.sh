#!/usr/bin/env bash
set -Eeuo pipefail

echo "[entrypoint] market-crash-monitor starting"
echo "[entrypoint] TZ=${TZ:-Europe/Madrid} DATA_DIR=${DATA_DIR:-/data}"
echo "[entrypoint] SCHEDULE_CRON=${SCHEDULE_CRON:-'0 17 * * 1'} RUN_AT_START=${RUN_AT_START:-0}"
echo "[entrypoint] HEALTH_HTTP=${HEALTH_HTTP:-1} HEALTH_PORT=${HEALTH_PORT:-8087}"

exec python /market_monitor/scheduler.py
