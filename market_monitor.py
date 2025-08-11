#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Market Monitor — Europe & Global aware, resilient to missing FRED series.
v1.9.3

- Removes fragile FRED series (ISM PMI=NAMP, GSCPI) and uses robust proxies.
- BDIY→BDRY fallback (Yahoo) to avoid empty Baltic Dry.
- Credit-heavy regime score across Inflation, Credit, Growth, Geo/deglobalization.
- Stage map: mid_cycle → late_cycle → capitulation → early_recovery (with crisis override).
- Only propose allocation when NEXT STAGE is imminent or started (no monthly churn).
- LLM mini-brief: beginner-friendly; uses OpenAI SDK first, with clear logs and HTTP fallback.
- New: --llm-test CLI to sanity-check your OpenAI setup.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

VERSION = "1.9.3"

# --- Load .env in the same folder (dotenv if available; fallback manual) ---
def _manual_load_env(env_path: Path):
    try:
        txt = env_path.read_text(encoding="utf-8")
    except Exception:
        return False
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
    return True

def load_env():
    try:
        from dotenv import load_dotenv
        if load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False):
            print("[env] .env loaded via python-dotenv")
            return
    except Exception:
        pass
    p = Path(__file__).with_name(".env")
    if p.exists() and _manual_load_env(p):
        print("[env] .env loaded via manual parser")
    else:
        print("[env] No .env found; relying on process env.")

load_env()

# Quiet yfinance logs
try:
    import logging as _logging
    _logging.getLogger("yfinance").setLevel(_logging.ERROR)
    _logging.getLogger("yfinance.yf_logger").setLevel(_logging.ERROR)
except Exception:
    pass

# Optional Yahoo dependency
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

APP_NAME = os.getenv("APP_NAME", "market_monitor")

# -----------------------------
# Logging
# -----------------------------
DATA_DIR = Path(os.getenv("CRASH_DASHBOARD_DATA_DIR") or os.getenv("DATA_DIR") or "./data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = DATA_DIR / f"{APP_NAME}.log"
STATE_PATH = DATA_DIR / "dashboard_state.json"

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
# Utils
# -----------------------------
def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def load_state() -> dict:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[state] read failed: {e}")
    return {}

def save_state(state: dict):
    try:
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[state] write failed: {e}")

def zscore_last(series: List[float]) -> Optional[float]:
    xs = [x for x in series if isinstance(x, (int, float)) and not math.isnan(x)]
    if len(xs) < 8:
        return None
    mu = statistics.mean(xs[:-1])
    sd = statistics.pstdev(xs[:-1]) or 1e-9
    return (xs[-1] - mu) / sd

def safe_z_from_series(series: List[Tuple[dt.date, float]]) -> Optional[float]:
    vals = [v for _, v in series if isinstance(v, (int, float)) and not math.isnan(v)]
    if len(vals) < 8:
        return None
    mu = statistics.mean(vals[:-1])
    sd = statistics.pstdev(vals[:-1]) or 1e-9
    return (vals[-1] - mu) / sd

def yoy(series: List[Tuple[dt.date, float]]) -> Optional[float]:
    if not series or len(series) < 13:
        return None
    cur_d, cur_v = series[-1]
    prev_v = None; best = 10**9
    for d, v in series[:-1]:
        delta = abs((cur_d - d).days - 365)
        if delta < best:
            best = delta; prev_v = v
    if prev_v in (None, 0):
        return None
    return (cur_v - prev_v) / prev_v * 100.0

def last_value(series: List[Tuple[dt.date, float]]) -> Optional[float]:
    return series[-1][1] if series else None

# -----------------------------
# Data fetchers (FRED + Yahoo)
# -----------------------------
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY")

def fred_series(series_id: str, observation_start: str = "2010-01-01") -> List[Tuple[dt.date, float]]:
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
        out = []
        for o in js.get("observations", []):
            date = o.get("date", "")[:10]
            vraw = o.get("value", ".")
            if not date or vraw in (".", "", None):
                continue
            try:
                out.append((dt.date.fromisoformat(date), float(vraw)))
            except Exception:
                pass
        logger.debug(f"[fred] {series_id}: {len(out)} obs")
        return out
    except Exception as e:
        logger.warning(f"[fred] Failed {series_id}: {e}")
        return []

def yahoo_prices(symbol: str, period: str = "3y", interval: str = "1d") -> List[Tuple[dt.date, float]]:
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
            date = idx.date()
            val = row["Close"]
            try:
                close = float(val)
            except Exception:
                try:
                    close = float(getattr(val, "item", lambda: val)())
                except Exception:
                    try:
                        close = float(getattr(val, "iloc", [val])[0])
                    except Exception:
                        continue
            if not math.isnan(close):
                out.append((date, close))
        logger.debug(f"[yahoo] {symbol}: {len(out)} obs")
        return out
    except Exception as e:
        logger.warning(f"[yahoo] Failed {symbol}: {e}")
        return []

# -----------------------------
# Series map
# -----------------------------
def get_market_tickers() -> Dict[str, str]:
    return {
        "SPX": "^GSPC",
        "GOLD": "GC=F",
        "VIX": "^VIX",
        "EURUSD": "EURUSD=X",
        "ACWI": "ACWI",
        "FEZ": "FEZ",
        "BDIY": "^BDIY",
        "BDRY": "BDRY",          # fallback if BDIY empty
        "CEW": "CEW",
        "COPPER": "HG=F",
        "BRENT": "BZ=F",
        "XLU": "XLU",
        "SOXX": "SOXX",
        "HYG": "HYG",
        "JNK": "JNK",
        "IEF": "IEF",
        "TIP": "TIP",
        "DXY": "DX-Y.NYB",
    }

# FRED series dictionary — removed NAPM & GSCPI (we proxy them)
FRED_SERIES = {
    "US_CPI": "CPIAUCSL",
    "US_CORE_CPI": "CPILFESL",
    "PCE": "PCEPI",
    "T10Y_BREAKEVEN": "T10YIE",
    "REAL_10Y": "DFII10",
    "CFNAI_MA3": "CFNAIMA3",
    "IPMAN": "IPMAN",
    "NEWORDER": "NEWORDER",
    "DGS10": "DGS10",
    "DGS2": "DGS2",
    "DGS3MO": "DGS3MO",
    "FEDFUNDS": "FEDFUNDS",
    "HY_OAS_US": "BAMLH0A0HYM2",
    "TED_SPREAD": "TEDRATE",
    "BAA10Y_SPREAD": "BAA10YM",
    "USD_BROAD": "DTWEXBGS",
    "EZ_HICP": "CP0000EZ19M086NEST",
    "EZ_LTIR_DE": "IRLTLT01DEM156N",
    "EZ_HY_OAS": "BAMLHE00EHYIOAS",
    "ECB_DFR": "ECBDFR",
}

@dataclass
class Indicators:
    spx: Optional[float] = None
    gold: Optional[float] = None
    vix: Optional[float] = None
    cpi_yoy: Optional[float] = None
    core_cpi_yoy: Optional[float] = None
    pce_yoy: Optional[float] = None
    breakeven_10y: Optional[float] = None
    real_10y: Optional[float] = None
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
    fred_cache = {k: fred_series(sid, observation_start=start) for k, sid in FRED_SERIES.items()}
    y = {k: yahoo_prices(sym, period=f"{lookback_years}y") for k, sym in get_market_tickers().items()}
    if not y.get("BDIY") or len(y["BDIY"]) == 0:
        logger.info("[yahoo] ^BDIY empty; using BDRY fallback")
        y["BDIY"] = y.get("BDRY", [])
    ind = Indicators()
    ind.spx   = last_value(y.get("SPX", []))
    ind.gold  = last_value(y.get("GOLD", []))
    ind.vix   = last_value(y.get("VIX", []))
    ind.cpi_yoy       = yoy(fred_cache["US_CPI"])
    ind.core_cpi_yoy  = yoy(fred_cache["US_CORE_CPI"])
    ind.pce_yoy       = yoy(fred_cache["PCE"])
    ind.breakeven_10y = last_value(fred_cache["T10Y_BREAKEVEN"])
    ind.real_10y      = last_value(fred_cache["REAL_10Y"])
    ind.cfnai_ma3     = last_value(fred_cache["CFNAI_MA3"])
    ind.ipman_yoy     = yoy(fred_cache["IPMAN"])
    ind.neworder_yoy  = yoy(fred_cache["NEWORDER"])
    ind.dgs10         = last_value(fred_cache["DGS10"])
    ind.dgs2          = last_value(fred_cache["DGS2"])
    ind.dgs3mo        = last_value(fred_cache["DGS3MO"])
    ind.fedfunds      = last_value(fred_cache["FEDFUNDS"])
    ind.hy_oas_us     = last_value(fred_cache["HY_OAS_US"])
    ind.ted           = last_value(fred_cache["TED_SPREAD"])
    ind.baa10y_spread = last_value(fred_cache["BAA10Y_SPREAD"])
    ind.usd_broad     = last_value(fred_cache["USD_BROAD"])
    ind.ez_hicp_yoy   = yoy(fred_cache["EZ_HICP"])
    ind.ez_ltir_de    = last_value(fred_cache["EZ_LTIR_DE"])
    ind.ez_hy_oas     = last_value(fred_cache["EZ_HY_OAS"])
    ind.ecb_dfr       = last_value(fred_cache["ECB_DFR"])
    ind.eurusd = last_value(y.get("EURUSD", []))
    ind.acwi   = last_value(y.get("ACWI", []))
    ind.fez    = last_value(y.get("FEZ", []))
    ind.bdiy   = last_value(y.get("BDIY", []))
    ind.cew    = last_value(y.get("CEW", []))
    ind.copper = last_value(y.get("COPPER", []))
    ind.brent  = last_value(y.get("BRENT", []))
    ind.xlu    = last_value(y.get("XLU", []))
    ind.soxx   = last_value(y.get("SOXX", []))
    raw = {"fred": fred_cache, "yahoo": y}
    return ind, raw

# -----------------------------
# Scoring (higher = more stress)
# -----------------------------
@dataclass
class ScoreBreakdown:
    inflation: float
    credit: float
    growth: float
    geo: float
    brink: float

def compute_pillars(ind: Indicators, raw: dict) -> ScoreBreakdown:
    infl_zs, cred_zs, grow_zs, geo_zs = [], [], [], []
    for key, s_key in [("cpi_yoy","US_CPI"), ("core_cpi_yoy","US_CORE_CPI"), ("pce_yoy","PCE")]:
        v = getattr(ind, key)
        if isinstance(v, (int, float)) and raw["fred"][s_key]:
            z = safe_z_from_series(raw["fred"][s_key])
            if z is not None:
                infl_zs.append(z)
    if raw["fred"]["T10Y_BREAKEVEN"]:
        z = safe_z_from_series(raw["fred"]["T10Y_BREAKEVEN"])
        if z is not None: infl_zs.append(z)
    BZ = raw["yahoo"].get("BRENT", [])
    if len(BZ) >= 130:
        last, past = BZ[-1][1], BZ[-130][1]
        infl_zs.append(((last - past)/max(past,1e-9)) * 5.0)
    if raw["fred"]["EZ_HICP"]:
        z = safe_z_from_series(raw["fred"]["EZ_HICP"])
        if z is not None: infl_zs.append(z)
    inflation = statistics.mean(infl_zs) if infl_zs else 0.0

    for k in ["HY_OAS_US", "BAA10Y_SPREAD", "TED_SPREAD"]:
        s = raw["fred"].get(k, [])
        if s:
            z = safe_z_from_series(s)
            if z is not None: cred_zs.append(z)
    if raw["fred"].get("EZ_HY_OAS"):
        z = safe_z_from_series(raw["fred"]["EZ_HY_OAS"])
        if z is not None: cred_zs.append(z)
    if ind.dgs10 and ind.dgs3mo and ind.dgs2:
        s10_3m = ind.dgs10 - ind.dgs3mo
        s2_10  = ind.dgs2 - ind.dgs10
        cred_zs += [(-s10_3m)/0.5, (s2_10)/0.5]
    if raw["fred"]["REAL_10Y"]:
        z = safe_z_from_series(raw["fred"]["REAL_10Y"])
        if z is not None: cred_zs.append(z)
    credit = statistics.mean(cred_zs) if cred_zs else 0.0

    if raw["fred"]["CFNAI_MA3"]:
        z = safe_z_from_series(raw["fred"]["CFNAI_MA3"])
        if z is not None: grow_zs.append(-z)

    def ratio_stress(a,b,scale=5.0):
        A = raw["yahoo"].get(a, []); B = raw["yahoo"].get(b, [])
        if len(A) >= 130 and len(B) >= 130:
            ra = A[-1][1]/max(A[-130][1],1e-9)
            rb = B[-1][1]/max(B[-130][1],1e-9)
            return (1.0 - (ra/rb)) * scale
        return None

    soxx_vs_spx = ratio_stress("SOXX","SPX",scale=4.0)
    xlu_vs_spx  = ratio_stress("XLU","SPX",scale=3.0)
    if soxx_vs_spx is not None: grow_zs.append(soxx_vs_spx)
    if xlu_vs_spx  is not None: grow_zs.append(xlu_vs_spx)

    def ratio_z(y_sym_a: str, y_sym_b: str) -> Optional[float]:
        A = raw["yahoo"].get(y_sym_a, []); B = raw["yahoo"].get(y_sym_b, [])
        if len(A) >= 130 and len(B) >= 130:
            ra = A[-1][1]/max(A[-130][1],1e-9)
            rb = B[-1][1]/max(B[-130][1],1e-9)
            return (1.0 - (ra/rb)) * 5.0
        return None
    cg = ratio_z("COPPER","GOLD")
    if cg is not None: grow_zs.append(cg)
    B = raw["yahoo"].get("BDIY", [])
    if len(B) >= 130:
        m = B[-1][1] / max(B[-130][1],1e-9)
        grow_zs.append((1.0 - m) * 5.0)
    if ind.ipman_yoy is not None:
        grow_zs.append((0.0 - ind.ipman_yoy) / 4.0)
    if ind.neworder_yoy is not None:
        grow_zs.append((0.0 - ind.neworder_yoy) / 6.0)
    fez_vs_acwi = ratio_z("FEZ","ACWI")
    if fez_vs_acwi is not None: grow_zs.append(fez_vs_acwi)
    growth = statistics.mean(grow_zs) if grow_zs else 0.0

    if raw["fred"]["USD_BROAD"]:
        z = safe_z_from_series(raw["fred"]["USD_BROAD"])
        if z is not None: geo_zs.append(z)
    E = raw["yahoo"].get("EURUSD", [])
    if len(E) >= 130:
        m = E[-1][1] / max(E[-130][1],1e-9)
        geo_zs.append((1.0 - m) * 5.0)
    D = raw["yahoo"].get("DXY", [])
    if len(D) >= 130:
        m = D[-1][1] / max(D[-130][1],1e-9)
        geo_zs.append((m - 1.0) * 4.0)
    if len(B) >= 130:
        m = B[-1][1] / max(B[-130][1],1e-9)
        geo_zs.append((1.0 - m) * 4.0)
    geo = statistics.mean(geo_zs) if geo_zs else 0.0

    w_credit, w_infl, w_growth, w_geo = 0.40, 0.25, 0.20, 0.15
    comp = w_credit*credit + w_infl*inflation + w_growth*growth + w_geo*geo
    brink = max(0.0, min(100.0, 50.0 + 10.0*comp))
    return ScoreBreakdown(float(inflation), float(credit), float(growth), float(geo), float(brink))

# -----------------------------
# Stage & “imminent” logic
# -----------------------------
TARGETS = {
    "mid_cycle":      {"GOLD":0.356,"CASH":0.144,"TIPS":0.115,"COMMODITY_PROD":0.192,"INFRA":0.115,"EM":0.078},
    "late_cycle":     {"GOLD":0.55,"CASH":0.20,"TIPS":0.10,"COMMODITY_PROD":0.07,"INFRA":0.05,"EM":0.03},
    "capitulation":   {"GOLD":0.50,"CASH":0.25,"TIPS":0.10,"COMMODITY_PROD":0.06,"INFRA":0.05,"EM":0.04},
    "early_recovery": {"GOLD":0.40,"CASH":0.15,"TIPS":0.10,"COMMODITY_PROD":0.15,"INFRA":0.12,"EM":0.08},
}

def determine_stage(scores: ScoreBreakdown, ind: Indicators, raw: dict) -> str:
    vix = ind.vix or 0.0
    hy  = ind.hy_oas_us or 0.0
    if (vix >= 35.0 or hy >= 7.0) and scores.brink >= 60:
        return "capitulation"
    if scores.brink >= 65: return "late_cycle"
    if scores.brink <= 40: return "early_recovery"
    return "mid_cycle"

def imminent_next_stage(current_stage: str, scores: ScoreBreakdown, ind: Indicators, raw: dict) -> Optional[str]:
    hy = ind.hy_oas_us or 0.0
    vix = ind.vix or 0.0
    H = raw["fred"].get("HY_OAS_US", [])
    hy_wk_up = len(H) >= 6 and (H[-1][1] - H[-6][1]) >= 0.30
    if current_stage == "mid_cycle":
        if scores.brink >= 62 or (scores.credit > 0.8 and (hy_wk_up or vix >= 25.0)):
            return "late_cycle"
    if current_stage == "late_cycle":
        if scores.brink >= 72 or hy >= 6.5 or vix >= 30.0 or hy_wk_up:
            return "capitulation"
    if current_stage == "capitulation":
        if scores.brink <= 42 and scores.growth < 0.4:
            return "early_recovery"
    return None

def apply_tilts(stage: str, scores: ScoreBreakdown, ind: Indicators, raw: dict) -> Dict[str, float]:
    w = dict(TARGETS[stage])
    if scores.inflation > 0.5:
        w["TIPS"] += 0.02; w["GOLD"] += 0.02; w["COMMODITY_PROD"] += 0.02; w["EM"] -= 0.02
    U = raw["fred"]["USD_BROAD"]
    usd_mom = None
    if len(U) >= 13:
        usd_mom = (U[-1][1]/max(U[-13][1],1e-9)) - 1.0
    if usd_mom is not None and usd_mom > 0.03:
        w["EM"] -= 0.02; w["CASH"] += 0.02
    if ind.real_10y is not None and ind.real_10y > 2.0:
        w["GOLD"] -= 0.02; w["TIPS"] += 0.01; w["CASH"] += 0.01
    H = raw["fred"]["HY_OAS_US"]
    if len(H) >= 6 and (H[-1][1] - H[-6][1]) >= 0.50:
        w["CASH"] += 0.03; w["EM"] -= 0.01; w["COMMODITY_PROD"] -= 0.01; w["INFRA"] -= 0.01
    for k in list(w.keys()):
        w[k] = max(0.0, w[k])
    s = sum(w.values()) or 1.0
    for k in list(w.keys()):
        w[k] = round(w[k]/s, 4)
    return w

# -----------------------------
# Europe-led stress line
# -----------------------------
def europe_led_stress(ind: Indicators, raw: dict) -> Optional[str]:
    score = 0; parts = []
    ez = raw["fred"].get("EZ_HY_OAS", []); us = raw["fred"].get("HY_OAS_US", [])
    if len(ez) >= 6 and len(us) >= 6:
        d_ez = ez[-1][1] - ez[-6][1]; d_us = us[-1][1] - us[-6][1]
        if (d_ez - d_us) >= 0.20:
            score += 1; parts.append(f"EU HY +{d_ez*100:.0f} bps vs US +{d_us*100:.0f} bps")
    FEZ = raw["yahoo"].get("FEZ", []); ACWI = raw["yahoo"].get("ACWI", [])
    if len(FEZ) >= 130 and len(ACWI) >= 130:
        fez_chg = FEZ[-1][1]/max(FEZ[-130][1],1e-9) - 1.0
        acwi_chg = ACWI[-1][1]/max(ACWI[-130][1],1e-9) - 1.0
        if (fez_chg - acwi_chg) <= -0.04:
            score += 1; parts.append(f"FEZ lagging ACWI by {abs(fez_chg - acwi_chg)*100:.1f} pp (6m)")
    E = raw["yahoo"].get("EURUSD", [])
    if len(E) >= 130:
        eur_m = E[-1][1]/max(E[-130][1],1e-9) - 1.0
        if eur_m <= -0.03:
            score += 1; parts.append(f"EURUSD {eur_m*100:.1f}% (6m)")
    if ind.ez_hicp_yoy is not None and ind.cpi_yoy is not None and (ind.ez_hicp_yoy - ind.cpi_yoy) >= 0.5:
        score += 1; parts.append(f"EZ HICP {ind.ez_hicp_yoy:.1f}% vs US CPI {ind.cpi_yoy:.1f}%")
    return ("Europe-led stress: prefer USD-quality & defense. (" + "; ".join(parts[:2]) + ")") if score >= 2 else None

# -----------------------------
# Telegram
# -----------------------------
def send_telegram(msg_html: str) -> bool:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        logger.warning("[tg] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID; skipping send.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg_html, "parse_mode": "HTML", "disable_web_page_preview": True}
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

def make_message(now: dt.datetime, ind: Indicators, raw: dict, scores: ScoreBreakdown, stage: str,
                 weights: Optional[Dict[str, float]], portfolio_value: Optional[float],
                 prev_brink: Optional[float], action_note: str) -> str:
    spx = ind.spx; gold = ind.gold; vix = ind.vix
    spx_str  = f"{spx:,.2f}" if spx else "—"
    gold_str = f"{gold:,.2f}" if gold else "—"
    vix_str  = f"{vix:,.1f}" if vix else "—"
    delta = None if prev_brink is None else (scores.brink - prev_brink)
    arrow = "↑" if (delta is not None and delta > 0.15) else ("↓" if (delta is not None and delta < -0.15) else "→")
    brink_line = f"{scores.brink:.0f}/100 {arrow}"
    if delta is not None: brink_line += f" ({delta:+.2f})"
    bullets = []
    if ind.dgs10 and ind.dgs3mo:
        bullets.append(f"10Y–3M: {ind.dgs10 - ind.dgs3mo:+.2f} pp")
    if ind.dgs2 and ind.dgs10:
        bullets.append(f"2s10s: {ind.dgs2 - ind.dgs10:+.2f} pp")
    if ind.hy_oas_us is not None:
        bullets.append(f"US HY OAS: {ind.hy_oas_us:.2f}%")
    if ind.ez_hy_oas is not None:
        bullets.append(f"EU HY OAS: {ind.ez_hy_oas:.2f}%")
    if ind.cpi_yoy is not None and ind.core_cpi_yoy is not None and ind.breakeven_10y is not None:
        bullets.append(f"CPI/Core: {ind.cpi_yoy:.1f}%/{ind.core_cpi_yoy:.1f}% | 10y BE: {ind.breakeven_10y:.2f}%")
    if ind.cfnai_ma3 is not None:
        bullets.append(f"CFNAI(3m): {ind.cfnai_ma3:.2f}")
    if ind.eurusd is not None:
        bullets.append(f"EURUSD: {ind.eurusd:.4f}")
    if ind.bdiy is not None:
        bullets.append(f"BDIY lvl: {ind.bdiy:,.0f}")
    bullets = bullets[:6]
    bullets_text = "\n".join([f" - {html_escape(b)}" for b in bullets])
    stage_emoji = {"mid_cycle":"🟡","late_cycle":"🟠","capitulation":"🔴","early_recovery":"🟢"}.get(stage,"🟡")
    msg = (
        f"<b>Market Monitor — {now.strftime('%Y-%m-%d')}</b>\n"
        f"v{VERSION}\n"
        f"S&amp;P 500: <b>{spx_str}</b> | Gold: <b>{gold_str}</b> | VIX: <b>{vix_str}</b>\n\n"
        f"<b>Stage:</b> {stage.replace('_',' ').title()} {stage_emoji}\n"
        f"<b>Brink:</b> {brink_line}\n"
        f"<b>Signals:</b>\n{bullets_text}\n\n"
    )
    eu_line = europe_led_stress(ind, raw)
    if eu_line:
        msg += html_escape(eu_line) + "\n\n"
    if weights is not None:
        alloc_text = format_money_allocations(weights, portfolio_value)
        msg += f"<b>Allocation (proposed):</b>\n{html_escape(alloc_text)}\n\n"
    msg += html_escape(action_note)
    return msg

# -----------------------------
# Cadence
# -----------------------------
def _parse_policy() -> str:
    raw = (os.getenv("ALERT_SEND_POLICY") or "always").lower()
    return "smart" if "smart" in raw else "always"

def _parse_weekday() -> int:
    s = os.getenv("WEEKLY_DAY", "4")
    for ch in s:
        if ch.isdigit():
            return max(0, min(6, int(ch)))
    return 4

def should_send(
    now: dt.datetime,
    policy: str,
    scores: ScoreBreakdown,
    ind: Indicators,
    raw: dict,
    state: dict,
) -> bool:
    """
    Decide whether to send an alert.

    Modes:
      - policy == "always": always send.
      - policy == "smart": send if either
          (A) it's the weekly summary window (Monday by default), OR
          (B) a fast/significant change is detected in key risk indicators.

    Overrides:
      - If scheduler sets RUN_REASON=weekly, force-send (guarantees Monday 17:00 summary
        even if 6h spacing doesn't land exactly on 17:00).
    """
    # 0) Explicit weekly override from the scheduler
    if (os.getenv("RUN_REASON") or "").strip().lower() == "weekly":
        return True

    # 1) Simple mode
    if policy == "always":
        return True

    # 2) Weekly summary day (defaults to Monday=0; configurable via WEEKLY_DAY)
    weekly_day = _parse_weekday()  # keeps your existing env parsing
    last_sent = state.get("last_sent_ts")
    last_sent_day = None
    if last_sent:
        try:
            last_sent_day = dt.datetime.fromisoformat(last_sent).date()
        except Exception:
            pass
    send_weekly = (now.weekday() == weekly_day) and (last_sent_day != now.date())

    # 3) Fast-change detection (configurable thresholds with safe defaults)
    #    Env vars let you tune sensitivity without code changes.
    vix_jump  = float(os.getenv("FAST_VIX_JUMP",  "5.0"))   # points d/d
    hy_6d     = float(os.getenv("FAST_HY_OAS_6D", "0.30"))  # abs % over ~1 week
    dgs2_jump = float(os.getenv("FAST_DGS2_JUMP", "0.15"))  # abs % d/d
    brink_d   = float(os.getenv("FAST_BRINK_DELTA", "3.0")) # brink points

    fast = False

    # VIX: large day-over-day volatility spikes
    V = raw.get("yahoo", {}).get("VIX", [])
    if len(V) >= 2 and (V[-1][1] - V[-2][1]) >= vix_jump:
        fast = True

    # HY OAS: credit spreads widening quickly in ~a week
    H = raw.get("fred", {}).get("HY_OAS_US", [])
    if len(H) >= 6 and (H[-1][1] - H[-6][1]) >= hy_6d:
        fast = True

    # 2Y yield: front-end rate shock day-over-day
    U2 = raw.get("fred", {}).get("DGS2", [])
    if len(U2) >= 2 and (U2[-1][1] - U2[-2][1]) >= dgs2_jump:
        fast = True

    # Composite “Brink” score: regime risk jumping materially
    prev_brink = state.get("brink")
    if prev_brink is not None and abs(scores.brink - float(prev_brink)) >= brink_d:
        fast = True

    return send_weekly or fast

# -----------------------------
# LLM mini-brief (SDK first; HTTP fallback)
# -----------------------------
def _llm_enabled() -> bool:
    raw = (os.getenv("LLM_SUMMARY", "0") or "").strip().lower()
    on = raw in ("1", "true", "yes", "on")
    logger.info(f"[llm] LLM_SUMMARY='{raw}' -> enabled={on}")
    return on

def make_llm_brief(scores: ScoreBreakdown, stage: str) -> Optional[str]:
    if not _llm_enabled():
        logger.info("[llm] Skipped: LLM_SUMMARY not enabled")
        return None
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        logger.info("[llm] Skipped: OPENAI_API_KEY missing")
        return None
    model_primary = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    model_fallbacks = [m for m in [model_primary, "gpt-4o-mini-2024-07-18", "gpt-4o", "gpt-4o-2024-08-06"] if m]
    org = (os.getenv("OPENAI_ORG_ID") or "").strip()
    project = (os.getenv("OPENAI_PROJECT") or "").strip()
    prompt = (
        "Explain today’s macro regime to a smart beginner. Keep it concise (3–5 sentences), plain English, no jargon. "
        "Briefly define any tricky term in-line (e.g., 'yield-curve inversion = short rates above long rates'). "
        "Focus on what the signals imply for risk and inflation, and what could push us into the next stage.\n\n"
        f"Pillars (higher=worse): Inflation={scores.inflation:.2f}, Credit={scores.credit:.2f}, "
        f"Growth={scores.growth:.2f}, Geo={scores.geo:.2f}. Stage={stage}."
    )
    try:
        from openai import OpenAI
        client_kwargs = {"api_key": key}
        if org: client_kwargs["organization"] = org
        if project: client_kwargs["project"] = project
        client = OpenAI(**client_kwargs)
        for i, model in enumerate(model_fallbacks, 1):
            try:
                logger.info(f"[llm] SDK chat attempt {i}/{len(model_fallbacks)} model={model}")
                chat = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=220,
                )
                text = chat.choices[0].message.content.strip()
                logger.info("[llm] SDK chat ok")
                return text
            except Exception as e:
                logger.warning(f"[llm] SDK chat error: {e}")
            try:
                logger.info(f"[llm] SDK responses attempt {i}/{len(model_fallbacks)} model={model}")
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=0.2,
                    max_output_tokens=220,
                )
                text = getattr(resp, "output_text", None)
                if not text:
                    try:
                        text = resp.output[0].content[0].text
                    except Exception:
                        text = None
                if text:
                    text = text.strip()
                    logger.info("[llm] SDK responses ok")
                    return text
                else:
                    logger.warning("[llm] SDK responses returned empty text")
            except Exception as e:
                logger.warning(f"[llm] SDK responses error: {e}")
    except Exception as e:
        logger.warning(f"[llm] OpenAI SDK not available or failed to init: {e}")
    base_headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if org: base_headers["OpenAI-Organization"] = org
    if project: base_headers["OpenAI-Project"] = project
    for i, model in enumerate(model_fallbacks, 1):
        try:
            logger.info(f"[llm] HTTP chat attempt {i}/{len(model_fallbacks)} model={model}")
            payload = {"model": model, "messages": [{"role":"user","content":prompt}], "temperature":0.2, "max_tokens":220}
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=base_headers, json=payload, timeout=30)
            if r.status_code == 200:
                js = r.json()
                text = js["choices"][0]["message"]["content"].strip()
                logger.info("[llm] HTTP chat ok")
                return text
            else:
                logger.warning(f"[llm] HTTP chat non-200 {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"[llm] HTTP chat exception: {e}")
        try:
            logger.info(f"[llm] HTTP responses attempt {i}/{len(model_fallbacks)} model={model}")
            payload = {"model": model, "input": prompt, "temperature":0.2, "max_output_tokens":220}
            r = requests.post("https://api.openai.com/v1/responses", headers=base_headers, json=payload, timeout=30)
            if r.status_code == 200:
                js = r.json()
                text = js.get("output_text")
                if not text:
                    try:
                        text = js["choices"][0]["message"]["content"]
                    except Exception:
                        text = None
                if text:
                    text = text.strip()
                    logger.info("[llm] HTTP responses ok")
                    return text
                else:
                    logger.warning(f"[llm] HTTP responses parsed empty: {str(js)[:200]}")
            else:
                logger.warning(f"[llm] HTTP responses non-200 {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"[llm] HTTP responses exception: {e}")
    logger.warning("[llm] All attempts failed; continuing without brief")
    return None

# -----------------------------
# Main run
# -----------------------------
def run_once(cfg: dict, dry_run: bool=False) -> int:
    now = dt.datetime.now()
    logger.info(f"[run] Start v{VERSION}")
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
    state = load_state()
    prev_brink = state.get("brink")
    prev_stage = state.get("stage")
    action_note = ""
    weights_for_message: Optional[Dict[str, float]] = None
    next_imminent = imminent_next_stage(stage, scores, ind, raw)
    if prev_stage is None:
        weights_for_message = apply_tilts(stage, scores, ind, raw)
        action_note = "Initial setup: applying stage-based allocation."
    elif stage != prev_stage:
        weights_for_message = apply_tilts(stage, scores, ind, raw)
        action_note = f"Stage changed from {prev_stage.replace('_',' ')} to {stage.replace('_',' ')} — adjust allocation accordingly."
    elif next_imminent:
        weights_for_message = apply_tilts(next_imminent, scores, ind, raw)
        action_note = f"Next stage looks imminent ({next_imminent.replace('_',' ')}). Consider pre-emptive tilt."
    else:
        action_note = "No allocation change recommended — monitoring signals for stage shift."
    portfolio_value = None
    for env_name in ("PORTFOLIO_VALUE_USD", "PORTFOLIO_VALUE"):
        try:
            pv = os.getenv(env_name)
            if pv:
                portfolio_value = float(pv); break
        except Exception:
            pass
    msg = make_message(now, ind, raw, scores, stage, weights_for_message, portfolio_value, prev_brink, action_note)
    llm = make_llm_brief(scores, stage)
    if llm:
        msg += "\n\n" + html_escape(llm)
    policy = _parse_policy()
    do_send = (policy == "always") or should_send(now, policy, scores, ind, raw, state)
    if dry_run: do_send = False
    if do_send:
        ok = send_telegram(msg)
        if ok:
            logger.info("[tg] Sent."); state["last_sent_ts"] = now.isoformat(timespec="seconds")
        else:
            logger.warning("[tg] Send failed.")
    else:
        logger.info("[tg] Skipped by policy/dry-run.")
    state["brink"] = scores.brink
    state["stage"] = stage
    save_state(state)
    logger.info("[run] Done.")
    return 0

def build_arg_parser():
    ap = argparse.ArgumentParser(description="Market Monitor — Europe & Global aware")
    ap.add_argument("--dry-run", action="store_true", help="Run without sending Telegram")
    ap.add_argument("--loop", action="store_true", help="Loop forever")
    ap.add_argument("--interval", type=int, default=3600, help="Loop sleep seconds")
    ap.add_argument("--verbose", action="store_true", help="DEBUG logging to console")
    ap.add_argument("--lookback-years", type=int, default=int(os.getenv("LOOKBACK_YEARS", "6")), help="Historical window (years)")
    ap.add_argument("--llm-test", action="store_true", help="Run a one-off LLM test and exit")
    return ap

def main():
    args = build_arg_parser().parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.DEBUG)
    if args.llm_test:
        print(f"[llm-test] VERSION v{VERSION}")
        dummy_scores = ScoreBreakdown(0.2, 0.3, 0.1, 0.2, 50.0)
        text = make_llm_brief(dummy_scores, "mid_cycle")
        if text:
            print("[llm-test] OK. Sample brief:\n" + text)
            sys.exit(0)
        else:
            print("[llm-test] FAILED. See log for details.")
            sys.exit(2)
    cfg = {"lookback_years": args.lookback_years}
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
