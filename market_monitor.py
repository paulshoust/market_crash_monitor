#!/usr/bin/env python3
"""
Market Monitor PRO — Global Cycle & Europe-Aware (FRED + Yahoo + Telegram HTML)
- Keeps your original behavior: robust HTML Telegram, rotating file logging, CLI flags (--loop/--interval/--dry-run/--verbose).
- Adds Europe + Global proxies, Dalio-style 4-pillar composite ("Brink" 0–100), regime stage, and rule-based allocation tilts.
- Optional mini-LLM brief if OPENAI_API_KEY and LLM_SUMMARY=1 are present (skips otherwise).
- Smart send policy: ALERT_SEND_POLICY=always (default) or smart (weekly + fast moves).

Environment (compatible with your v1):
- TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- FRED_API_KEY (recommended)
- CRASH_DASHBOARD_DATA_DIR or DATA_DIR (state, logs)
- APP_NAME (default: market_monitor_pro)
- OPENAI_API_KEY, LLM_SUMMARY=1 (optional)
- ALERT_SEND_POLICY=always|smart ; WEEKLY_DAY=4 (0=Mon…6=Sun)
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from dotenv import load_dotenv
# Load exactly the .env that sits next to this file
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
print("[env] .env loaded via python-dotenv")

# Optional Yahoo dependency: yfinance (gracefully degrades if missing)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

APP_NAME = os.getenv("APP_NAME", "market_monitor_pro")

# Data dir (compat with prior script)
DATA_DIR = Path(os.getenv("CRASH_DASHBOARD_DATA_DIR") or os.getenv("DATA_DIR") or "./data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = DATA_DIR / f"{APP_NAME}.log"
STATE_PATH = DATA_DIR / "dashboard_state.json"

# -----------------------------
# Logging (kept as before)
# -----------------------------
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

_file_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)

# -----------------------------
# Utility
# -----------------------------
def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[state] Failed to read state: {e}")
    return {}

def save_state(state: dict):
    try:
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[state] Failed to write state: {e}")

def pct_change(a: float, b: float) -> Optional[float]:
    try:
        if b == 0 or b is None or a is None:
            return None
        return (a - b) / b * 100.0
    except Exception:
        return None

def zscore(series: List[float]) -> Optional[float]:
    # returns z of last point
    xs = [x for x in series if isinstance(x, (int, float)) and not math.isnan(x)]
    if len(xs) < 8:
        return None
    mu = statistics.mean(xs[:-1])
    sd = statistics.pstdev(xs[:-1]) or 1e-9
    return (xs[-1] - mu) / sd

def moving_avg(series: List[float], n: int) -> Optional[float]:
    xs = [x for x in series if isinstance(x, (int, float)) and not math.isnan(x)]
    if len(xs) < n:
        return None
    return statistics.mean(xs[-n:])

# -----------------------------
# Data fetchers
# -----------------------------
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY")

def fred_series(series_id: str, observation_start: str = "2010-01-01") -> List[Tuple[dt.date, float]]:
    """Fetch a FRED series (last N years). Returns list of (date, value)."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY or "",
        "file_type": "json",
        "observation_start": observation_start,
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        out = []
        for o in obs:
            d = dt.date.fromisoformat(o["date"][:10])
            vraw = o.get("value", ".")
            if vraw in (".", "", None):
                continue
            try:
                v = float(vraw)
            except Exception:
                continue
            out.append((d, v))
        logger.debug(f"[fred] {series_id}: {len(out)} obs")
        return out
    except Exception as e:
        logger.warning(f"[fred] Failed {series_id}: {e}")
        return []

def yahoo_prices(symbol: str, period: str = "3y", interval: str = "1d") -> List[Tuple[dt.date, float]]:
    """Fetch Yahoo daily close prices. Returns list of (date, close)."""
    if not YF_OK:
        logger.warning(f"[yahoo] yfinance not available, skipping {symbol}")
        return []
    try:
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if data is None or data.empty:
            logger.warning(f"[yahoo] Empty for {symbol}")
            return []
        out = []
        for idx, row in data.iterrows():
            # idx is pandas Timestamp
            date = idx.date()
            close = float(row["Close"])
            if not math.isnan(close):
                out.append((date, close))
        logger.debug(f"[yahoo] {symbol}: {len(out)} obs")
        return out
    except Exception as e:
        logger.warning(f"[yahoo] Failed {symbol}: {e}")
        return []

# -----------------------------
# Indicators (US, Europe, Global)
# -----------------------------

def last_value(series: List[Tuple[dt.date, float]]) -> Optional[float]:
    if not series:
        return None
    return series[-1][1]

def yoy(series: List[Tuple[dt.date, float]]) -> Optional[float]:
    # compute YoY rate from monthly index level
    if not series or len(series) < 13:
        return None
    cur = series[-1][1]
    # find value roughly 1y ago
    target_date = series[-1][0] - dt.timedelta(days=365)
    # pick the closest date
    prev = None
    best = float("inf")
    for d, v in series[:-1]:
        delta = abs((series[-1][0] - d).days - 365)
        if delta < best:
            best = delta
            prev = v
    if prev is None or prev == 0:
        return None
    return (cur - prev) / prev * 100.0

def series_to_dict(series: List[Tuple[dt.date, float]]) -> Dict[str, float]:
    return {d.isoformat(): v for d, v in series}

# --- Core Market lines ---
def get_market_tickers() -> Dict[str, str]:
    return {
        "SPX": "^GSPC",
        "GOLD": "GC=F",
        "VIX": "^VIX",
        "EURUSD": "EURUSD=X",
        "ACWI": "ACWI",
        "FEZ": "FEZ",
        "BDIY": "^BDIY",
        "CEW": "CEW",
        "COPPER": "HG=F",
        "BRENT": "BZ=F",
        "XLU": "XLU",
        "SOXX": "SOXX",
    }

# --- FRED series dictionary (some may be missing depending on API limits) ---
FRED_SERIES = {
    # Inflation & expectations
    "US_CPI": "CPIAUCSL",          # CPI index (YoY computed)
    "US_CORE_CPI": "CPILFESL",     # Core CPI index
    "PCE": "PCEPI",                # PCE price index

    "T10Y_BREAKEVEN": "T10YIE",    # 10y inflation expectation
    "REAL_10Y": "DFII10",          # 10y TIPS real yield

    # Growth
    "CFNAI_MA3": "CFNAIMA3",       # Chicago Fed NA Activity Index, 3-mo avg
    "ISM_PMI": "NAPM",             # ISM Manufacturing PMI (index level)
    "IPMAN": "IPMAN",              # Industrial Production: Manufacturing (index; YoY computed)
    "NEWORDER": "NEWORDER",        # New Orders: Nondef Cap Goods ex-Aircraft (index; YoY)

    # Rates/curve
    "DGS10": "DGS10",              # 10y UST
    "DGS2":  "DGS2",               # 2y UST
    "DGS3MO":"DGS3MO",             # 3m T-bill
    "FEDFUNDS":"FEDFUNDS",         # Fed Funds effective

    # Credit spreads
    "HY_OAS_US": "BAMLH0A0HYM2",   # ICE BofA US HY OAS (percent)
    "TED_SPREAD": "TEDRATE",       # TED Spread (legacy, may be sparse)
    "BAA10Y_SPREAD": "BAA10YM",    # Moody's Baa - 10y Treasury (percent)

    # Globalization / USD / supply chain
    "USD_BROAD": "DTWEXBGS",       # Broad trade-weighted USD
    "GSCPI": "GSCPI",              # NY Fed global supply chain pressure index

    # Europe specifics
    "EZ_HICP": "CP0000EZ19M086NEST",   # HICP total (index); YoY computed (Eurostat via FRED proxy)
    # Core HICP might not be available consistently; include a likely OECD proxy if present
    "EZ_LTIR_DE": "IRLTLT01DEM156N",   # Long-term rates, Germany (Bund proxy, %)
    "EZ_HY_OAS": "BAMLHE00EHYIOAS",    # ICE BofA Euro HY OAS (percent) — may be restricted; skip if missing
    # ECB Deposit Facility Rate (may not be on FRED or is under different id; skip gracefully if missing)
    "ECB_DFR": "ECBDFR"
}

@dataclass
class Indicators:
    # Prices
    spx: Optional[float] = None
    gold: Optional[float] = None
    vix: Optional[float] = None
    # U.S. macro
    cpi_yoy: Optional[float] = None
    core_cpi_yoy: Optional[float] = None
    pce_yoy: Optional[float] = None
    breakeven_10y: Optional[float] = None
    real_10y: Optional[float] = None
    ismpmi: Optional[float] = None
    cfnai_ma3: Optional[float] = None
    ipman_yoy: Optional[float] = None
    neworder_yoy: Optional[float] = None
    dgs10: Optional[float] = None
    dgs2: Optional[float] = None
    dgs3mo: Optional[float] = None
    fedfunds: Optional[float] = None
    hy_oas_us: Optional[float] = None
    ted: Optional[float] = None
    baa10y_spread: Optional[float] = None
    usd_broad: Optional[float] = None
    gscpi: Optional[float] = None
    # Europe & Global
    ez_hicp_yoy: Optional[float] = None
    ez_ltir_de: Optional[float] = None
    ez_hy_oas: Optional[float] = None
    ecb_dfr: Optional[float] = None

    eurusd: Optional[float] = None
    acwi: Optional[float] = None
    fez: Optional[float] = None
    bdiy: Optional[float] = None
    cew: Optional[float] = None
    copper: Optional[float] = None
    brent: Optional[float] = None
    xlu: Optional[float] = None
    soxx: Optional[float] = None

def fetch_all_indicators(lookback_years: int = 6) -> Tuple[Indicators, dict]:
    start = (dt.date.today() - dt.timedelta(days=365*lookback_years)).isoformat()

    # FRED
    fred_cache = {}
    for key, sid in FRED_SERIES.items():
        fred_cache[key] = fred_series(sid, observation_start=start)

    # Yahoo
    y = {}
    for k, sym in get_market_tickers().items():
        y[k] = yahoo_prices(sym, period=f"{lookback_years}y")

    ind = Indicators()
    # Prices
    ind.spx = last_value(y.get("SPX", []))
    ind.gold = last_value(y.get("GOLD", []))
    ind.vix = last_value(y.get("VIX", []))

    # U.S. macro
    ind.cpi_yoy = yoy(fred_cache["US_CPI"])
    ind.core_cpi_yoy = yoy(fred_cache["US_CORE_CPI"])
    ind.pce_yoy = yoy(fred_cache["PCE"])
    ind.breakeven_10y = last_value(fred_cache["T10Y_BREAKEVEN"])
    ind.real_10y = last_value(fred_cache["REAL_10Y"])
    ind.ismpmi = last_value(fred_cache["ISM_PMI"])
    ind.cfnai_ma3 = last_value(fred_cache["CFNAI_MA3"])
    ind.ipman_yoy = yoy(fred_cache["IPMAN"])
    ind.neworder_yoy = yoy(fred_cache["NEWORDER"])
    ind.dgs10 = last_value(fred_cache["DGS10"])
    ind.dgs2 = last_value(fred_cache["DGS2"])
    ind.dgs3mo = last_value(fred_cache["DGS3MO"])
    ind.fedfunds = last_value(fred_cache["FEDFUNDS"])
    ind.hy_oas_us = last_value(fred_cache["HY_OAS_US"])
    ind.ted = last_value(fred_cache["TED_SPREAD"])
    ind.baa10y_spread = last_value(fred_cache["BAA10Y_SPREAD"])
    ind.usd_broad = last_value(fred_cache["USD_BROAD"])
    ind.gscpi = last_value(fred_cache["GSCPI"])

    # Europe & global
    ind.ez_hicp_yoy = yoy(fred_cache["EZ_HICP"])
    ind.ez_ltir_de = last_value(fred_cache["EZ_LTIR_DE"])
    ind.ez_hy_oas = last_value(fred_cache["EZ_HY_OAS"])
    ind.ecb_dfr = last_value(fred_cache["ECB_DFR"])

    ind.eurusd = last_value(y.get("EURUSD", []))
    ind.acwi = last_value(y.get("ACWI", []))
    ind.fez = last_value(y.get("FEZ", []))
    ind.bdiy = last_value(y.get("BDIY", []))
    ind.cew = last_value(y.get("CEW", []))
    ind.copper = last_value(y.get("COPPER", []))
    ind.brent = last_value(y.get("BRENT", []))
    ind.xlu = last_value(y.get("XLU", []))
    ind.soxx = last_value(y.get("SOXX", []))

    # Also return raw caches for change/ratio calculations
    raw = {"fred": fred_cache, "yahoo": y}
    return ind, raw

# -----------------------------
# Scoring model (all pillars: higher = more stress)
# -----------------------------

@dataclass
class ScoreBreakdown:
    inflation: float
    credit: float
    growth: float
    geo: float
    brink: float

def safe_z_from_series(series: List[Tuple[dt.date, float]]) -> Optional[float]:
    vals = [v for _, v in series if isinstance(v, (int, float)) and not math.isnan(v)]
    if len(vals) < 8:
        return None
    mu = statistics.mean(vals[:-1])
    sd = statistics.pstdev(vals[:-1]) or 1e-9
    return (vals[-1] - mu) / sd

def compute_pillars(ind: Indicators, raw: dict) -> ScoreBreakdown:
    # --- Inflation pressure ---
    infl_zs = []

    # US CPI/Core/PCE YoY as positive stress when high
    for key in ["cpi_yoy", "core_cpi_yoy", "pce_yoy"]:
        v = getattr(ind, key)
        if isinstance(v, (int, float)):
            # transform to z via historical series if available
            series_key = {"cpi_yoy":"US_CPI", "core_cpi_yoy":"US_CORE_CPI", "pce_yoy":"PCE"}[key]
            z = safe_z_from_series(raw["fred"][series_key])
            if z is not None:
                infl_zs.append(z)

    # 10y breakeven (elevated = stress)
    if raw["fred"]["T10Y_BREAKEVEN"]:
        z = safe_z_from_series(raw["fred"]["T10Y_BREAKEVEN"])
        if z is not None:
            infl_zs.append(z)

    # Oil uptrend => inflation pressure
    brent_series = raw["yahoo"].get("BRENT", [])
    if len(brent_series) >= 130:
        last = brent_series[-1][1]; past = brent_series[-130][1]  # ~6 months
        if past not in (None, 0) and last is not None:
            chg = (last - past) / past * 100.0
            infl_zs.append(chg/10.0)  # scale

    # Euro HICP YoY
    if raw["fred"]["EZ_HICP"]:
        z = safe_z_from_series(raw["fred"]["EZ_HICP"])
        if z is not None:
            infl_zs.append(z)

    inflation = statistics.mean(infl_zs) if infl_zs else 0.0

    # --- Credit & monetary stress ---
    cred_zs = []

    for k in ["HY_OAS_US", "BAA10Y_SPREAD"]:
        s = raw["fred"].get(k, [])
        if s:
            z = safe_z_from_series(s)
            if z is not None:
                cred_zs.append(z)

    # Euro HY OAS if available
    if raw["fred"].get("EZ_HY_OAS"):
        z = safe_z_from_series(raw["fred"]["EZ_HY_OAS"])
        if z is not None:
            cred_zs.append(z)

    # TED spread (legacy but still informative when it moves)
    if raw["fred"]["TED_SPREAD"]:
        z = safe_z_from_series(raw["fred"]["TED_SPREAD"])
        if z is not None:
            cred_zs.append(z)

    # Yield curve stress: inversion depth as stress (negative slope -> positive stress)
    if ind.dgs10 and ind.dgs3mo and ind.dgs2:
        s10_3m = ind.dgs10 - ind.dgs3mo
        s2_10 = ind.dgs2 - ind.dgs10
        # Map inversion to stress roughly in z-like units
        cred_zs.append((-s10_3m) / 0.5)  # -0.5pp inversion ~ +1 z
        cred_zs.append((s2_10) / 0.5)    # +0.5pp (2y>10y) ~ +1 z

    # Real 10y very high (restrictive) -> stress
    if raw["fred"]["REAL_10Y"]:
        z = safe_z_from_series(raw["fred"]["REAL_10Y"])
        if z is not None:
            cred_zs.append(z)

    credit = statistics.mean(cred_zs) if cred_zs else 0.0

    # --- Growth momentum (lower growth = higher stress) ---
    grow_zs = []

    # CFNAI MA3 below 0 => stress; use negative sign so more negative => higher stress
    if raw["fred"]["CFNAI_MA3"]:
        z = safe_z_from_series(raw["fred"]["CFNAI_MA3"])
        if z is not None:
            grow_zs.append(-z)

    # ISM PMI below 50 => stress
    if ind.ismpmi is not None:
        # Deviation from 50 scaled
        grow_zs.append((50.0 - ind.ismpmi) / 2.0)

    # Real economy proxies: Copper/Gold down, BDIY down, FEZ vs ACWI underperform
    def ratio_z(y_sym_a: str, y_sym_b: str) -> Optional[float]:
        A = raw["yahoo"].get(y_sym_a, []); B = raw["yahoo"].get(y_sym_b, [])
        if len(A) >= 130 and len(B) >= 130:
            ra = A[-1][1] / max(A[-130][1], 1e-9)
            rb = B[-1][1] / max(B[-130][1], 1e-9)
            r = ra / max(rb, 1e-9)
            return (1.0 - r) * 5.0  # down 10% => +0.5 stress
        return None

    cg = ratio_z("COPPER", "GOLD")
    if cg is not None:
        grow_zs.append(cg)

    # Baltic Dry Index 6m momentum
    B = raw["yahoo"].get("BDIY", [])
    if len(B) >= 130:
        m = B[-1][1] / max(B[-130][1], 1e-9)
        grow_zs.append((1.0 - m) * 5.0)

    # FEZ vs ACWI underperformance
    fez_vs_acwi = ratio_z("FEZ", "ACWI")
    if fez_vs_acwi is not None:
        grow_zs.append(fez_vs_acwi)

    growth = statistics.mean(grow_zs) if grow_zs else 0.0

    # --- Geo / De-globalization / Reserve currency pressure ---
    geo_zs = []
    # Broad USD stronger => stress for global liquidity
    if raw["fred"]["USD_BROAD"]:
        z = safe_z_from_series(raw["fred"]["USD_BROAD"])
        if z is not None:
            geo_zs.append(z)

    # EURUSD down => stress (add scaled momentum signal)
    E = raw["yahoo"].get("EURUSD", [])
    if len(E) >= 130:
        m = E[-1][1] / max(E[-130][1], 1e-9)
        geo_zs.append((1.0 - m) * 5.0)

    # Supply chain pressure (GSCPI) up => stress
    if raw["fred"]["GSCPI"]:
        z = safe_z_from_series(raw["fred"]["GSCPI"])
        if z is not None:
            geo_zs.append(z)

    geo = statistics.mean(geo_zs) if geo_zs else 0.0

    # Weights (credit-heavy as per user)
    w_credit = 0.40
    w_infl = 0.25
    w_growth = 0.20
    w_geo = 0.15

    # Normalize pillars to a 0..100-ish scale using 10*z as rough mapping and clipping
    def nz(x): return 0.0 if x is None else x
    comp = (w_credit * nz(credit) + w_infl * nz(inflation) + w_growth * nz(growth) + w_geo * nz(geo))
    # map z-like composite to 0..100
    brink = max(0.0, min(100.0, 50.0 + 10.0 * comp))

    return ScoreBreakdown(inflation=float(inflation), credit=float(credit), growth=float(growth), geo=float(geo), brink=float(brink))

# -----------------------------
# Stage determination & allocation tilts
# -----------------------------
TARGETS = {
    "mid_cycle": {
        "GOLD": 0.35, "CASH": 0.15, "TIPS": 0.10,
        "COMMODITY_PROD": 0.18, "INFRA": 0.12, "EM": 0.10
    },
    "late_cycle": {
        "GOLD": 0.55, "CASH": 0.20, "TIPS": 0.10,
        "COMMODITY_PROD": 0.07, "INFRA": 0.05, "EM": 0.03
    },
    "capitulation": {
        "GOLD": 0.50, "CASH": 0.25, "TIPS": 0.10,
        "COMMODITY_PROD": 0.06, "INFRA": 0.05, "EM": 0.04
    },
    "early_recovery": {
        "GOLD": 0.40, "CASH": 0.15, "TIPS": 0.10,
        "COMMODITY_PROD": 0.15, "INFRA": 0.12, "EM": 0.08
    },
}

def determine_stage(scores: ScoreBreakdown, ind: Indicators, raw: dict) -> str:
    # Hard crisis override: HY OAS spike, VIX surge
    vix = ind.vix or 0.0
    hy = ind.hy_oas_us or 0.0
    crisis = (vix >= 35.0) or (hy >= 7.0)  # crude crisis bars
    if crisis and scores.brink >= 60:
        return "capitulation"
    # Brink bands
    if scores.brink >= 65:
        return "late_cycle"
    if scores.brink <= 40:
        return "early_recovery"
    return "mid_cycle"

def apply_tilts(stage: str, scores: ScoreBreakdown, ind: Indicators, raw: dict) -> Dict[str, float]:
    w = dict(TARGETS[stage])  # copy

    # Inflation pressure tilt: add to TIPS/Gold/Commodities if inflation high relative to history
    if scores.inflation > 0.5:
        w["TIPS"] += 0.02
        w["GOLD"] += 0.02
        w["COMMODITY_PROD"] += 0.02
        w["EM"] -= 0.02

    # Strong USD -> trim EM, raise Cash
    usd_momentum = None
    U = raw["fred"]["USD_BROAD"]
    if len(U) >= 13:
        usd_momentum = (U[-1][1] / max(U[-13][1], 1e-9)) - 1.0
    if usd_momentum is not None and usd_momentum > 0.03:
        w["EM"] -= 0.02
        w["CASH"] += 0.02

    # High real rates -> trim Gold slightly to TIPS/Cash
    if ind.real_10y is not None and ind.real_10y > 2.0:
        w["GOLD"] -= 0.02
        w["TIPS"] += 0.01
        w["CASH"] += 0.01

    # HY OAS weekly spike -> temporarily raise Cash
    hy_series = raw["fred"]["HY_OAS_US"]
    if len(hy_series) >= 6:
        last = hy_series[-1][1]; prev = hy_series[-6][1]  # ~w/w (daily series weekdays)
        if last - prev >= 0.5:  # +50 bps
            w["CASH"] += 0.03
            w["EM"] -= 0.01
            w["COMMODITY_PROD"] -= 0.01
            w["INFRA"] -= 0.01

    # Normalize (keep >=0 and sum 1.0)
    for k in list(w.keys()):
        w[k] = max(0.0, w[k])
    s = sum(w.values()) or 1.0
    for k in list(w.keys()):
        w[k] = round(w[k] / s, 4)
    return w

# -----------------------------
# Telegram
# -----------------------------
def send_telegram(msg_html: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("[tg] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID; skipping send.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": msg_html,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code != 200:
            logger.warning(f"[tg] Non-200: {r.status_code} {r.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"[tg] Failed: {e}")
        return False

def format_money_allocations(weights: Dict[str, float], portfolio_value: Optional[float]) -> str:
    if portfolio_value is None:
        return "\n".join([f"   ‣ {k}: {round(v*100,1)}%" for k,v in weights.items()])
    lines = []
    for k, v in weights.items():
        amt = v * portfolio_value
        lines.append(f"   ‣ {k}: {round(v*100,1)}%  (≈ ${amt:,.0f})")
    return "\n".join(lines)

def make_message(now: dt.datetime, ind: Indicators, raw: dict, scores: ScoreBreakdown, stage: str, weights: Dict[str, float], portfolio_value: Optional[float], prev_brink: Optional[float]) -> str:
    # Top line
    spx = ind.spx; gold = ind.gold; vix = ind.vix
    spx_str = f"{spx:,.2f}" if spx else "—"
    gold_str = f"{gold:,.2f}" if gold else "—"
    vix_str = f"{vix:,.1f}" if vix else "—"

    # Brink and trend
    delta = None if prev_brink is None else (scores.brink - prev_brink)
    arrow = "↑" if (delta is not None and delta > 0.15) else ("↓" if (delta is not None and delta < -0.15) else "→")
    brink_line = f"{scores.brink:.0f}/100 {arrow}"
    if delta is not None:
        brink_line += f" ({delta:+.2f})"

    # Signals bullets (keep short, 4–6)
    bullets = []

    # Curve
    if ind.dgs10 and ind.dgs3mo:
        bullets.append(f"10Y–3M: {ind.dgs10 - ind.dgs3mo:+.2f} pp")
    if ind.dgs2 and ind.dgs10:
        bullets.append(f"2s10s: {ind.dgs2 - ind.dgs10:+.2f} pp")

    # HY OAS, Euro HY
    if ind.hy_oas_us is not None:
        bullets.append(f"US HY OAS: {ind.hy_oas_us:.2f}%")
    if ind.ez_hy_oas is not None:
        bullets.append(f"EU HY OAS: {ind.ez_hy_oas:.2f}%")

    # Inflation
    if ind.cpi_yoy is not None and ind.core_cpi_yoy is not None and ind.breakeven_10y is not None:
        bullets.append(f"CPI/Core: {ind.cpi_yoy:.1f}%/{ind.core_cpi_yoy:.1f}% | 10y BE: {ind.breakeven_10y:.2f}%")

    # Growth
    if ind.ismpmi is not None and ind.cfnai_ma3 is not None:
        bullets.append(f"PMI: {ind.ismpmi:.1f} | CFNAI(3m): {ind.cfnai_ma3:.2f}")

    # Europe/global stresses
    if ind.eurusd is not None:
        bullets.append(f"EURUSD: {ind.eurusd:.4f}")
    if ind.bdiy is not None:
        bullets.append(f"BDIY lvl: {ind.bdiy:,.0f}")

    # Truncate to max 6 bullets
    bullets = bullets[:6]
    bullets_text = "\n".join([f" - {html_escape(b)}" for b in bullets])

    # Stage emoji
    stage_emoji = {"mid_cycle":"🟡", "late_cycle":"🟠", "capitulation":"🔴", "early_recovery":"🟢"}.get(stage, "🟡")

    # Allocation
    alloc_text = format_money_allocations(weights, portfolio_value)

    # Short guidance
    guidance = {
        "mid_cycle": "Balanced risk: keep powder dry, add selectively on weakness.",
        "late_cycle": "Defensive bias: quality cashflows, TIPS/Gold; avoid leverage.",
        "capitulation": "Capital preservation first. Phase-in only with strict risk controls.",
        "early_recovery": "Begin risk add gradually; prefer quality & cashflow visibility."
    }[stage]

    msg = (
f"<b>Market Monitor PRO — {now.strftime('%Y-%m-%d')}</b>\n"
f"S&amp;P 500: <b>{spx_str}</b> | Gold: <b>{gold_str}</b> | VIX: <b>{vix_str}</b>\n\n"
f"<b>Stage:</b> {stage.replace('_',' ').title()} {stage_emoji}\n"
f"<b>Brink:</b> {brink_line}\n"
f"<b>Signals:</b>\n{bullets_text}\n\n"
f"<b>Allocation (tilted):</b>\n{html_escape(alloc_text)}\n\n"
f"{html_escape(guidance)}"
    )
    return msg

# -----------------------------
# Cadence policy
# -----------------------------
def should_send(now: dt.datetime, policy: str, scores: ScoreBreakdown, ind: Indicators, raw: dict, state: dict) -> bool:
    if policy == "always":
        return True
    # smart mode: weekly checkpoint + fast moves
    weekly_day = int(os.getenv("WEEKLY_DAY", "4"))  # Friday
    last_sent = state.get("last_sent_ts")
    last_sent_day = None
    if last_sent:
        try:
            last_sent_day = dt.datetime.fromisoformat(last_sent).date()
        except Exception:
            pass

    send_weekly = (now.weekday() == weekly_day) and (last_sent_day != now.date())

    # Fast-move triggers
    fast = False

    # VIX day-over-day +5 points
    V = raw["yahoo"].get("VIX", [])
    if len(V) >= 2:
        if V[-1][1] - V[-2][1] >= 5.0:
            fast = True

    # HY OAS +30 bps w/w
    H = raw["fred"].get("HY_OAS_US", [])
    if len(H) >= 6:
        if H[-1][1] - H[-6][1] >= 0.30:
            fast = True

    # 2y yield +15 bps d/d
    U2 = raw["fred"].get("DGS2", [])
    if len(U2) >= 2:
        if U2[-1][1] - U2[-2][1] >= 0.15:
            fast = True

    # Brink change >= 3 points day-over-day
    prev_brink = state.get("brink")
    if prev_brink is not None and abs(scores.brink - float(prev_brink)) >= 3.0:
        fast = True

    return send_weekly or fast

# -----------------------------
# LLM Mini-brief (optional)
# -----------------------------
def make_llm_brief(scores: ScoreBreakdown, stage: str) -> Optional[str]:
    if os.getenv("LLM_SUMMARY", "0") != "1":
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.warning("[llm] LLM_SUMMARY=1 but OPENAI_API_KEY missing; skipping.")
        return None
    try:
        # Keep it super simple to avoid adding nonstandard deps
        import requests as _r
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        prompt = (
            "You are a macro strategist. In 2-3 tight sentences, explain the current market regime and key risks "
            "using these inputs:\n"
            f"Pillars (higher = more stress): Inflation={scores.inflation:.2f}, Credit={scores.credit:.2f}, "
            f"Growth={scores.growth:.2f}, Geo={scores.geo:.2f}. Stage={stage}. "
            "Avoid tables or emojis."
        )
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role":"user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 120
        }
        resp = _r.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"[llm] Non-200: {resp.status_code} {resp.text[:200]}")
            return None
        js = resp.json()
        txt = js["choices"][0]["message"]["content"].strip()
        return txt
    except Exception as e:
        logger.warning(f"[llm] Failed to get brief: {e}")
        return None

# -----------------------------
# Main run
# -----------------------------
def run_once(cfg: dict, dry_run: bool=False) -> int:
    now = dt.datetime.now()
    logger.info("[run] Start")

    try:
        ind, raw = fetch_all_indicators(lookback_years=cfg.get("lookback_years", 6))
        logger.info("[run] Data fetched.")
    except Exception as e:
        logger.error(f"[run] Data fetch failed: {e}")
        return 1

    scores = compute_pillars(ind, raw)
    logger.info(f"[run] Scores: infl={scores.inflation:.2f} credit={scores.credit:.2f} growth={scores.growth:.2f} geo={scores.geo:.2f} brink={scores.brink:.2f}")

    stage = determine_stage(scores, ind, raw)
    logger.info(f"[run] Stage: {stage}")

    weights = apply_tilts(stage, scores, ind, raw)

    # State
    state = load_state()
    prev_brink = state.get("brink")

    # Portfolio value (optional for pretty $ lines)
    portfolio_value = None
    try:
        pv = os.getenv("PORTFOLIO_VALUE_USD")
        if pv:
            portfolio_value = float(pv)
    except Exception:
        portfolio_value = None

    msg = make_message(now, ind, raw, scores, stage, weights, portfolio_value, prev_brink)

    # Optional LLM brief
    llm = make_llm_brief(scores, stage)
    if llm:
        msg += "\n\n" + html_escape(llm)

    policy = os.getenv("ALERT_SEND_POLICY", "always").lower()
    do_send = (policy == "always") or should_send(now, policy, scores, ind, raw, state)
    if dry_run:
        do_send = False

    if do_send:
        ok = send_telegram(msg)
        if ok:
            logger.info("[tg] Sent.")
            state["last_sent_ts"] = now.isoformat(timespec="seconds")
        else:
            logger.warning("[tg] Send failed.")
    else:
        logger.info("[tg] Skipped by policy/dry-run.")

    # Save state
    state["brink"] = scores.brink
    save_state(state)

    logger.info("[run] Done.")
    return 0

def build_arg_parser():
    ap = argparse.ArgumentParser(description="Market Monitor PRO — Global Cycle & Europe-Aware")
    ap.add_argument("--dry-run", action="store_true", help="Run without sending Telegram")
    ap.add_argument("--loop", action="store_true", help="Loop forever")
    ap.add_argument("--interval", type=int, default=3600, help="Loop sleep seconds")
    ap.add_argument("--verbose", action="store_true", help="DEBUG logging to console")
    ap.add_argument("--lookback-years", type=int, default=int(os.getenv("LOOKBACK_YEARS", "6")), help="Historical window (years)")
    return ap

def main():
    args = build_arg_parser().parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        _console_handler.setLevel(logging.DEBUG)

    cfg = {
        "lookback_years": args.lookback_years
    }

    if args.loop:
        print("[main] Entering loop mode…")
        while True:
            rc = run_once(cfg, dry_run=args.dry_run)
            sleep_for = args.interval if rc == 0 else min(args.interval * 2, 7200)
            print(f"[main] Sleeping {sleep_for}s…")
            time.sleep(sleep_for)
    else:
        rc = run_once(cfg, dry_run=args.dry_run)
        sys.exit(rc)

if __name__ == "__main__":
    main()
