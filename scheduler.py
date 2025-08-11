#!/usr/bin/env python3
import os, sys, time, json, subprocess, threading, datetime as dt
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from zoneinfo import ZoneInfo
from croniter import croniter

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
STATE_PATH = DATA_DIR / "dashboard_state.json"

SCHEDULE_CRON = os.getenv("SCHEDULE_CRON", "0 17 * * 1")  # Monday 17:00
TZ = os.getenv("TZ", "Europe/Madrid")
RUN_AT_START = (os.getenv("RUN_AT_START", "0").strip().lower() in ("1","true","yes","on"))

HEALTH_HTTP = (os.getenv("HEALTH_HTTP", "1").strip().lower() in ("1","true","yes","on"))
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8087"))
HEALTH_MAX_AGE_MIN = int(os.getenv("HEALTH_MAX_AGE_MIN", "180"))
MAIN_CRON = os.getenv("SCHEDULE_CRON", "0 */6 * * *")      # every 6 hours
WEEKLY_CRON = os.getenv("WEEKLY_CRON", "0 17 * * 1")       # Monday 17:00 Madrid


def now_tz():
    return dt.datetime.now(ZoneInfo(TZ))

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/healthz"):
            self.send_response(404); self.end_headers(); return
        try:
            mtime = STATE_PATH.stat().st_mtime
            age = int((time.time() - mtime)/60)
        except Exception:
            age = None
        ok = (age is not None) and (age <= HEALTH_MAX_AGE_MIN)
        payload = {
            "ok": bool(ok),
            "state_age_min": age,
            "state_exists": STATE_PATH.exists(),
            "schedule_cron": SCHEDULE_CRON,
            "tz": TZ,
        }
        self.send_response(200 if ok else 500)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
    def log_message(self, *_):  # quiet
        return

def serve_health():
    HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler).serve_forever()

def run_once():
    cmd = [sys.executable, "/market_monitor/market_monitor.py"]
    print(f"[scheduler] Running: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    print(f"[scheduler] Run finished with rc={rc}")
    return rc

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if HEALTH_HTTP:
        threading.Thread(target=serve_health, daemon=True).start()
    if RUN_AT_START:
        try:
            run_once()
        except Exception as e:
            print("[scheduler] First run failed:", e)

    while True:
        now = now_tz()
        nxt_main   = croniter(MAIN_CRON, now).get_next(dt.datetime).replace(tzinfo=ZoneInfo(TZ))
        nxt_weekly = croniter(WEEKLY_CRON, now).get_next(dt.datetime).replace(tzinfo=ZoneInfo(TZ))
        nxt = min(nxt_main, nxt_weekly)
        reason = "weekly" if nxt == nxt_weekly else "periodic"
        sleep_s = max(1, int((nxt - now).total_seconds()))
        print(f"[scheduler] Now {now.isoformat()}  Next {nxt.isoformat()}  ({reason}, sleep {sleep_s}s)")
        try:
            time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("[scheduler] Interrupted"); sys.exit(0)
        try:
            # Pass run reason to the child process
            env = os.environ.copy()
            env["RUN_REASON"] = reason
            cmd = [sys.executable, "/market_monitor/market_monitor.py"]
            print(f"[scheduler] Running ({reason}): {' '.join(cmd)}")
            rc = subprocess.call(cmd, env=env)
            print(f"[scheduler] Run finished rc={rc}")
        except Exception as e:
            print("[scheduler] Scheduled run failed:", e)

if __name__ == "__main__":
    main()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if HEALTH_HTTP:
        threading.Thread(target=serve_health, daemon=True).start()
    if RUN_AT_START:
        try: run_once()
        except Exception as e: print("[scheduler] First run failed:", e)
    while True:
        now = now_tz()
        nxt = croniter(SCHEDULE_CRON, now).get_next(dt.datetime).replace(tzinfo=ZoneInfo(TZ))
        sleep_s = max(1, int((nxt - now).total_seconds()))
        print(f"[scheduler] Now {now.isoformat()}  Next {nxt.isoformat()}  (sleep {sleep_s}s)")
        try: time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("[scheduler] Interrupted"); sys.exit(0)
        try: run_once()
        except Exception as e: print("[scheduler] Scheduled run failed:", e)

if __name__ == "__main__":
    main()
