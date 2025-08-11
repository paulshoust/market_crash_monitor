set -Eeuo pipefail

STATE_FILE="${DATA_DIR:-/data}/dashboard_state.json"
MAX_AGE_MIN="${HEALTH_MAX_AGE_MIN:-180}"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "[health] missing $STATE_FILE"; exit 1
fi

# GNU/BSD compatibility
if date -r "$STATE_FILE" +%s >/dev/null 2>&1; then
  MTIME_EPOCH=$(date -r "$STATE_FILE" +%s)
else
  MTIME_EPOCH=$(stat -c %Y "$STATE_FILE")
fi
NOW_EPOCH=$(date +%s)
AGE_MIN=$(( (NOW_EPOCH - MTIME_EPOCH) / 60 ))

if (( AGE_MIN > MAX_AGE_MIN )); then
  echo "[health] stale state: ${AGE_MIN}min > ${MAX_AGE_MIN}min"; exit 1
fi

echo "[health] OK (${AGE_MIN} min old)"
