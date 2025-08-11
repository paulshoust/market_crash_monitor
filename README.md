# Market Crash Monitor

Headless container that fetches macro + market data, scores stress (credit-weighted), detects **stage** (Mid → Late → Capitulation → Early Recovery), and sends a clean **Telegram** brief with an optional **LLM** explainer. Europe-led stress is explicitly considered. Runs **weekly every Monday 17:00 (Europe/Madrid)** by default.

## Quick links
- GitHub: https://github.com/paulshoust/market_crash_monitor
- Docker Hub: paulshoust/market_crash_monitor

---

## Features

- **Data**: FRED (with proxies for fragile series), Yahoo Finance (with `^BDIY → BDRY` fallback).
- **Signals**: Inflation, Credit (heavy weight), Growth, Geopolitics/de-globalization.
- **Stages**: `mid_cycle`, `late_cycle`, `capitulation`, `early_recovery`, plus “imminent” logic.
- **LLM brief**: 3–5 sentences, beginner-friendly (optional via `LLM_SUMMARY=true`).
- **Europe-led stress**: EU HY vs US HY, FEZ vs ACWI, EURUSD, EZ HICP gap.
- **Delivery**: Telegram + stdout.
- **Ops**: Weekly CRON scheduling, tiny HTTP `/healthz` on `:8087`, Docker HEALTHCHECK via state freshness.

---

## Run (Docker Compose)

Create a folder on the server (e.g., `/opt/market_crash_monitor`) with:

- `docker-compose.yml` (in this repo)
- `.env` (based on `.env.sample`)
- a `data/` directory (or it will be created automatically)

Start:
```bash
docker compose pull
docker compose up -d
docker compose logs -f
