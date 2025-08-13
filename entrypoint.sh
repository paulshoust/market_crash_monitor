#!/usr/bin/env bash
set -Eeuo pipefail

echo "[entrypoint] market-crash-monitor starting"
echo "[entrypoint] TZ=${TZ:-Europe/Madrid} DATA_DIR=${DATA_DIR:-/data}"
echo "[entrypoint] SCHEDULE_CRON=${SCHEDULE_CRON:-0 17 * * 1} RUN_AT_START=${RUN_AT_START:-0}"
echo "[entrypoint] HEALTH_HTTP=${HEALTH_HTTP:-1} HEALTH_PORT=${HEALTH_PORT:-8087}"

# If a command is provided (from docker-compose 'command:'), run it.
if [[ $# -gt 0 ]]; then
  echo "[entrypoint] exec: $*"
  exec "$@"
fi

# Default: legacy loop mode (kept for compatibility)
exec python -u /market_monitor/market_monitor.py --loop --interval "${INTERVAL_SECONDS:-3600}"
