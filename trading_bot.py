
import asyncio
import logging
import os
import sqlite3
import time
import io
import json
from typing import Dict, Any, Optional, List, Tuple

import aiohttp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from aiohttp import web

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==========================================================
# 0. Config
# ==========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("SignalBot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", 8080))

DB_NAME = "signal_bot.db"
MODEL_PATH = "ai_model.joblib"
SCALER_PATH = "ai_scaler.joblib"

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
MAX_SL_PERCENT = 2.0
MIN_BTC_VOLUME = 1600.0
MAX_SIGNAL_AGE = 600
MAX_SLIPPAGE = 1.0

OB_MIN_BIDS = 5
OB_MIN_ASKS = 5
OB_MAX_SPREAD_PCT = 0.5
OB_MIN_IMBALANCE_CONF = 0.15

OB_QUALITY_MIN_SCORE = float(os.getenv("OB_QUALITY_MIN_SCORE", "0.45"))
OB_AUTO_FILTER_ENABLED = os.getenv("OB_AUTO_FILTER_ENABLED", "true").lower() == "true"
OB_MIN_DEPTH_USDT = float(os.getenv("OB_MIN_DEPTH_USDT", "50000"))
OB_MAX_STOP_HUNT_RISK = float(os.getenv("OB_MAX_STOP_HUNT_RISK", "0.6"))
OB_MAX_SLIPPAGE_PCT = float(os.getenv("OB_MAX_SLIPPAGE_PCT", "0.3"))

# ==========================================================
# 1. Async Database
# ==========================================================
def _sync_execute(query: str, params: tuple = ()):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.fetchall()

def _sync_fetch_df(query: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_NAME) as conn:
        return pd.read_sql_query(query, conn, params=params)

async def db_execute(query: str, params: tuple = ()):
    return await asyncio.to_thread(_sync_execute, query, params)

async def db_fetch_df(query: str, params: tuple = ()) -> pd.DataFrame:
    return await asyncio.to_thread(_sync_fetch_df, query, params)

async def init_database():
    queries = [
        """CREATE TABLE IF NOT EXISTS trade_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT,
            rsi REAL, spread_pct REAL, vol_ratio REAL, lower_wick_ratio REAL,
            upper_wick_ratio REAL, trend_code INTEGER, adx REAL,
            plus_di REAL, minus_di REAL, price_to_sma7_ratio REAL,
            atr_pct REAL, orderbook_imbalance REAL,
            outcome INTEGER DEFAULT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT UNIQUE,
            symbol TEXT, interval TEXT, direction TEXT, strategy TEXT,
            entry_price REAL, stop_loss REAL, tp1 REAL, tp2 REAL, tp3 REAL,
            sl_percent REAL, rsi REAL, adx REAL, trend TEXT, ai_prob REAL,
            ai_confidence TEXT, ob_imbalance REAL, ob_slippage REAL,
            ob_stop_hunt REAL, ob_iceberg_bids INTEGER, ob_iceberg_asks INTEGER,
            ob_quality_score REAL, ob_rejection_reason TEXT,
            feedback TEXT DEFAULT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS bot_stats (
            key TEXT PRIMARY KEY, value INTEGER DEFAULT 0)""",
        """CREATE TABLE IF NOT EXISTS gemini_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT,
            pattern_detected TEXT, signal TEXT, confidence_score REAL,
            analysis_summary TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    ]
    for q in queries:
        await db_execute(q)
    for k in ["total_signals","feedback_good","feedback_bad","ob_rejected","spread_rejected",
              "ob_quality_rejected","ob_depth_rejected","ob_stop_hunt_rejected","ob_slippage_rejected","volume_rejected"]:
        await db_execute("INSERT OR IGNORE INTO bot_stats (key, value) VALUES (?, 0)", (k,))
    LOGGER.info("Database initialized.")

# ==========================================================
# 2. Rate Limiting
# ==========================================================
class RateLimiter:
    def __init__(self, rate=10, per=1):
        self.rate = rate
        self.per = per
        self.tokens = float(rate)
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per))
            self.updated_at = now
            if self.tokens < 1:
                wait = (1 - self.tokens) * (self.per / self.rate)
                await asyncio.sleep(wait)
                self.tokens = 0
            else:
                self.tokens -= 1

binance_limiter = RateLimiter(rate=20, per=1)
bybit_limiter = RateLimiter(rate=10, per=1)
okx_limiter = RateLimiter(rate=10, per=1)
bitget_limiter = RateLimiter(rate=10, per=1)

# ==========================================================
# 3. Exchanges (Klines)
# ==========================================================
EXCHANGES = [
    {"name":"Binance","weight":10,"limiter":binance_limiter,
     "url":"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=100",
     "interval_map":{"15m":"15m","1h":"1h","4h":"4h","1d":"1d"},
     "parser": lambda d: d if isinstance(d, list) else None},
    {"name":"Bybit","weight":8,"limiter":bybit_limiter,
     "url":"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit=100",
     "interval_map":{"15m":"15","1h":"60","4h":"240","1d":"D"},
     "parser": lambda d: _parse_bybit(d)},
    {"name":"OKX","weight":8,"limiter":okx_limiter,
     "url":"https://www.okx.com/api/v5/market/history-candles?instId={symbol}-SWAP&bar={interval}&limit=100",
     "interval_map":{"15m":"15m","1h":"1H","4h":"4H","1d":"1D"},
     "parser": lambda d: _parse_okx(d)}
]

def _parse_bybit(data):
    try:
        if data.get("retCode") != 0: return None
        res = data.get("result", {}).get("list", [])
        return [[int(x[0]), x[1], x[2], x[3], x[4], x[5]] for x in reversed(res)]
    except: return None

def _parse_okx(data):
    try:
        res = data.get("data", [])
        return [[int(x[0]), x[1], x[2], x[3], x[4], x[5]] for x in reversed(res)]
    except: return None

def validate_klines(klines, symbol):
    if not klines or len(klines) < 10:
        return False, "too_few"
    try:
        last = float(klines[-1][4])
        prev = float(klines[-2][4])
    except: return False, "format"
    if prev > 0 and abs(last - prev) / prev > 0.5:
        return False, "jump"
    if last <= 0 or last > 1e6 or last < 1e-6:
        return False, "range"
    return True, "ok"

async def fetch_klines(session, symbol, interval):
    for ex in sorted(EXCHANGES, key=lambda x: x["weight"], reverse=True):
        try:
            await ex["limiter"].acquire()
            mi = ex["interval_map"].get(interval, interval)
            url = ex["url"].format(symbol=symbol, interval=mi)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status == 200:
                    data = await r.json()
                    klines = ex["parser"](data)
                    if klines and len(klines) >= 50:
                        ok, _ = validate_klines(klines, symbol)
                        if ok: return klines
        except: pass
        await asyncio.sleep(0.05)
    return None

# ==========================================================
# 3b. Multi-Exchange Order Book
# ==========================================================
OB_EXCHANGES = [
    {
        "name": "Binance Futures",
        "limiter": binance_limiter,
        "url": "https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={limit}",
        "parser": lambda d: (d.get("bids", []), d.get("asks", []))
    },
    {
        "name": "Binance Spot",
        "limiter": binance_limiter,
        "url": "https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}",
        "parser": lambda d: (d.get("bids", []), d.get("asks", []))
    },
    {
        "name": "Bybit Linear",
        "limiter": bybit_limiter,
        "url": "https://api.bybit.com/v5/market/orderbook?category=linear&symbol={symbol}&limit={limit}",
        "parser": lambda d: _parse_bybit_ob(d)
    },
    {
        "name": "Bybit Spot",
        "limiter": bybit_limiter,
        "url": "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit={limit}",
        "parser": lambda d: _parse_bybit_ob(d)
    },
    {
        "name": "OKX",
        "limiter": okx_limiter,
        "url": "https://www.okx.com/api/v5/market/books?instId={symbol}-SWAP&sz={limit}",
        "parser": lambda d: _parse_okx_ob(d)
    },
    {
        "name": "Bitget Futures",
        "limiter": bitget_limiter,
        "url": "https://api.bitget.com/api/v2/mix/market/depth?symbol={symbol}_UMCBL&limit={limit}&productType=USDT-FUTURES",
        "parser": lambda d: _parse_bitget_ob(d)
    },
    {
        "name": "Bitget Spot",
        "limiter": bitget_limiter,
        "url": "https://api.bitget.com/api/v2/spot/market/depth?symbol={symbol}&limit={limit}&type=step0",
        "parser": lambda d: _parse_bitget_spot_ob(d)
    }
]

def _parse_bybit_ob(data):
    try:
        if data.get("retCode") != 0: return [], []
        result = data.get("result", {})
        bids = [[b[0], b[1]] for b in result.get("b", [])]
        asks = [[a[0], a[1]] for a in result.get("a", [])]
        return bids, asks
    except: return [], []

def _parse_okx_ob(data):
    try:
        book = data.get("data", [{}])[0]
        bids = [[b[0], b[1]] for b in book.get("bids", [])]
        asks = [[a[0], a[1]] for a in book.get("asks", [])]
        return bids, asks
    except: return [], []

def _parse_bitget_ob(data):
    try:
        if data.get("code") != "00000": return [], []
        result = data.get("data", {})
        bids = [[b[0], b[1]] for b in result.get("bids", [])]
        asks = [[a[0], a[1]] for a in result.get("asks", [])]
        return bids, asks
    except: return [], []

def _parse_bitget_spot_ob(data):
    try:
        if data.get("code") != "00000": return [], []
        result = data.get("data", {})
        bids = [[b[0], b[1]] for b in result.get("bids", [])]
        asks = [[a[0], a[1]] for a in result.get("asks", [])]
        return bids, asks
    except: return [], []

async def fetch_order_book(session, symbol, limit=50):
    for ex in OB_EXCHANGES:
        for attempt in range(2):
            try:
                await ex["limiter"].acquire()
                url = ex["url"].format(symbol=symbol, limit=limit)
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 200:
                        data = await r.json()
                        bids, asks = ex["parser"](data)
                        if len(bids) >= OB_MIN_BIDS and len(asks) >= OB_MIN_ASKS:
                            LOGGER.info("OB from {} for {}: bids={}, asks={}".format(ex['name'], symbol, len(bids), len(asks)))
                            return bids, asks, ex["name"]
                        else:
                            LOGGER.warning("{} OB shallow for {}: bids={}, asks={}".format(ex['name'], symbol, len(bids), len(asks)))
                    else:
                        LOGGER.warning("{} OB HTTP {} for {}".format(ex['name'], r.status, symbol))
            except Exception as e:
                LOGGER.warning("{} OB error {} (attempt {}): {}".format(ex['name'], symbol, attempt+1, e))
            await asyncio.sleep(0.3)
    LOGGER.error("All OB sources failed for {}".format(symbol))
    return [], [], None

# ==========================================================
# 4. Order Book Quality Validator
# ==========================================================
def validate_order_book(ob_data: dict, direction: str, symbol: str) -> Tuple[bool, str]:
    if ob_data["spread_pct"] > OB_MAX_SPREAD_PCT:
        return False, "spread_too_high ({:.3f}%)".format(ob_data['spread_pct'])
    imbalance = ob_data["imbalance"]
    if direction == "LONG" and imbalance < -OB_MIN_IMBALANCE_CONF:
        return False, "ob_against_long (imbalance: {:.2f})".format(imbalance)
    if direction == "SHORT" and imbalance > OB_MIN_IMBALANCE_CONF:
        return False, "ob_against_short (imbalance: {:.2f})".format(imbalance)
    if ob_data["stop_hunt_risk"] > 0.7:
        return False, "stop_hunt_risk_high ({})".format(ob_data['stop_hunt_risk'])
    return True, "ok"

def ob_confidence_score(ob_data: dict, direction: str) -> float:
    score = 0.5
    imbalance = ob_data["imbalance"]
    if direction == "LONG":
        if imbalance > 0.3: score += 0.3
        elif imbalance > 0.1: score += 0.15
        elif imbalance < -0.3: score -= 0.3
    else:
        if imbalance < -0.3: score += 0.3
        elif imbalance < -0.1: score += 0.15
        elif imbalance > 0.3: score -= 0.3
    if ob_data["spread_pct"] < 0.05: score += 0.1
    elif ob_data["spread_pct"] > 0.3: score -= 0.2
    total_iceberg = ob_data["iceberg_bids"] + ob_data["iceberg_asks"]
    if total_iceberg >= 5: score += 0.1
    if ob_data["stop_hunt_risk"] > 0.5: score -= 0.2
    return max(0.0, min(1.0, score))

def ob_quality_filter(ob_data: dict, direction: str, symbol: str, entry_price: float = 0.0) -> Tuple[bool, str, float]:
    if not OB_AUTO_FILTER_ENABLED:
        return True, "auto_filter_disabled", ob_confidence_score(ob_data, direction)

    score = ob_confidence_score(ob_data, direction)
    reasons = []

    if score < OB_QUALITY_MIN_SCORE:
        reasons.append("quality_score_low ({:.2f} < {})".format(score, OB_QUALITY_MIN_SCORE))

    min_depth = OB_MIN_DEPTH_USDT
    if entry_price > 0:
        bid_depth_usdt = ob_data.get("bid_depth", 0) * entry_price
        ask_depth_usdt = ob_data.get("ask_depth", 0) * entry_price
    else:
        bid_depth_usdt = ob_data.get("bid_depth", 0)
        ask_depth_usdt = ob_data.get("ask_depth", 0)

    if direction == "LONG" and bid_depth_usdt < min_depth:
        reasons.append("bid_depth_low ({:,.0f} < {:,.0f} USDT)".format(bid_depth_usdt, min_depth))
    if direction == "SHORT" and ask_depth_usdt < min_depth:
        reasons.append("ask_depth_low ({:,.0f} < {:,.0f} USDT)".format(ask_depth_usdt, min_depth))

    if ob_data.get("stop_hunt_risk", 0) > OB_MAX_STOP_HUNT_RISK:
        reasons.append("stop_hunt_risk ({:.2f} > {})".format(ob_data['stop_hunt_risk'], OB_MAX_STOP_HUNT_RISK))

    if ob_data.get("slippage", 0) > OB_MAX_SLIPPAGE_PCT:
        reasons.append("slippage_high ({:.3f}% > {}%)".format(ob_data['slippage'], OB_MAX_SLIPPAGE_PCT))

    if ob_data["spread_pct"] > 0.3:
        reasons.append("spread_wide ({:.3f}% > 0.3%)".format(ob_data['spread_pct']))

    if reasons:
        return False, " | ".join(reasons), score
    return True, "quality_passed", score

# ==========================================================
# 5. Indicators
# ==========================================================
def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(d if d > 0 else 0.0)
        losses.append(abs(d) if d < 0 else 0.0)
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    alpha = 1.0 / period
    for i in range(period, len(gains)):
        ag = alpha * gains[i] + (1 - alpha) * ag
        al = alpha * losses[i] + (1 - alpha) * al
    if al == 0: return 100.0
    return round(100.0 - (100.0 / (1.0 + ag / al)), 2)

def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1: return 0.0
    trs = []
    for i in range(1, len(highs)):
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr

def calc_dmi(highs, lows, closes, period=14):
    if len(highs) < period * 2: return 0.0, 0.0, 0.0
    trs, pdm, mdm = [], [], []
    for i in range(1, len(highs)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        up = highs[i] - highs[i-1]
        dn = lows[i-1] - lows[i]
        pdm.append(up if up > dn and up > 0 else 0.0)
        mdm.append(dn if dn > up and dn > 0 else 0.0)
        trs.append(tr)
    str_ = sum(trs[:period]); spdm = sum(pdm[:period]); smdm = sum(mdm[:period])
    dxs = []
    for i in range(period, len(trs)):
        str_ = str_ - str_/period + trs[i]
        spdm = spdm - spdm/period + pdm[i]
        smdm = smdm - smdm/period + mdm[i]
        pdi = (spdm / str_ * 100) if str_ > 0 else 0
        mdi = (smdm / str_ * 100) if str_ > 0 else 0
        s = pdi + mdi
        dx = (abs(pdi - mdi) / s * 100) if s > 0 else 0
        dxs.append((pdi, mdi, dx))
    if not dxs: return 0.0, 0.0, 0.0
    lp, lm, _ = dxs[-1]
    adx = sum(x[2] for x in dxs[:period]) / period if len(dxs) >= period else sum(x[2] for x in dxs) / len(dxs)
    for i in range(period, len(dxs)):
        adx = (adx * (period - 1) + dxs[i][2]) / period
    return round(lp, 2), round(lm, 2), round(adx, 2)

def detect_hidden_divergence(highs, lows, closes):
    if len(closes) < 20: return False, False
    price_lows = lows[-10:]
    rsi_vals = [calc_rsi(closes[:len(closes)-10+i+1]) for i in range(10)]
    if len(rsi_vals) < 10: return False, False
    hidden_bull = (lows[-1] > min(price_lows)) and (rsi_vals[-1] < min(rsi_vals))
    hidden_bear = (highs[-1] < max(highs[-10:])) and (rsi_vals[-1] > max(rsi_vals))
    return hidden_bull, hidden_bear

def find_pivots(highs, lows, lr=3):
    ph, pl = [], []
    for i in range(lr, len(highs) - lr - 1):
        if all(highs[i] > highs[i-j] for j in range(1, lr+1)) and all(highs[i] >= highs[i+j] for j in range(1, lr+1)):
            ph.append((i, highs[i]))
        if all(lows[i] < lows[i-j] for j in range(1, lr+1)) and all(lows[i] <= lows[i+j] for j in range(1, lr+1)):
            pl.append((i, lows[i]))
    return ph, pl

def dow_trend(ph, pl):
    if len(ph) < 2 or len(pl) < 2: return "NEUTRAL"
    if ph[-1][1] > ph[-2][1] and pl[-1][1] > pl[-2][1]: return "BULLISH"
    if ph[-1][1] < ph[-2][1] and pl[-1][1] < pl[-2][1]: return "BEARISH"
    return "NEUTRAL"

def htf_sr(k4h, k1d):
    s, r = [], []
    for klines in [k4h, k1d]:
        if klines and len(klines) >= 30:
            h = [float(k[2]) for k in klines[:-1]]
            l = [float(k[3]) for k in klines[:-1]]
            ph, pl = find_pivots(h, l)
            r.extend([p[1] for p in ph[-3:]])
            s.extend([p[1] for p in pl[-3:]])
    return s, r


# ==========================================================
# 5b. Multi-Timeframe Analysis (4H + 1H + 15M Confluence)
# ==========================================================
def analyze_4h_direction(k4h):
    """Analyze 4H timeframe for overall direction and key levels"""
    if not k4h or len(k4h) < 50:
        return "NEUTRAL", [], []

    H = [float(k[2]) for k in k4h[:-1]]
    L = [float(k[3]) for k in k4h[:-1]]
    C = [float(k[4]) for k in k4h[:-1]]

    ph, pl = find_pivots(H, L)
    trend = dow_trend(ph, pl)

    # Key levels from 4H
    key_levels = []
    if ph:
        key_levels.extend([p[1] for p in ph[-5:]])
    if pl:
        key_levels.extend([p[1] for p in pl[-5:]])

    # Simple OB detection (last 3 candles before strong move)
    ob_zones = []
    for i in range(len(k4h) - 10, len(k4h) - 3):
        if i < 3: continue
        o, h, l, c = float(k4h[i][1]), float(k4h[i][2]), float(k4h[i][3]), float(k4h[i][4])
        # Bullish OB: bearish candle before strong green move
        if c < o and float(k4h[i+1][4]) > float(k4h[i+1][1]) * 1.01:
            ob_zones.append((l, o, "bullish"))
        # Bearish OB: bullish candle before strong red move
        if c > o and float(k4h[i+1][4]) < float(k4h[i+1][1]) * 0.99:
            ob_zones.append((o, h, "bearish"))

    return trend, key_levels, ob_zones


def analyze_1h_structure(k1h):
    """Analyze 1H timeframe for trend, breaks, reversal, OB, FVG, Liquidity"""
    if not k1h or len(k1h) < 50:
        return "NEUTRAL", [], [], [], []

    H = [float(k[2]) for k in k1h[:-1]]
    L = [float(k[3]) for k in k1h[:-1]]
    C = [float(k[4]) for k in k1h[:-1]]
    O = [float(k[1]) for k in k1h[:-1]]

    ph, pl = find_pivots(H, L)
    trend = dow_trend(ph, pl)

    # Breaks (recent highs/lows broken)
    breaks = []
    if len(H) >= 10:
        recent_high = max(H[-10:])
        recent_low = min(L[-10:])
        if C[-1] > recent_high:
            breaks.append(("high", recent_high))
        if C[-1] < recent_low:
            breaks.append(("low", recent_low))

    # FVG (Fair Value Gaps) - 3 candle pattern
    fvg_zones = []
    for i in range(len(k1h) - 20, len(k1h) - 2):
        if i < 2: continue
        c1, h1, l1 = float(k1h[i-2][4]), float(k1h[i-2][2]), float(k1h[i-2][3])
        c2, h2, l2 = float(k1h[i-1][4]), float(k1h[i-1][2]), float(k1h[i-1][3])
        c3, h3, l3 = float(k1h[i][4]), float(k1h[i][2]), float(k1h[i][3])

        # Bullish FVG: candle 2 low > candle 1 high
        if l2 > h1:
            fvg_zones.append((h1, l2, "bullish"))
        # Bearish FVG: candle 2 high < candle 1 low
        if h2 < l1:
            fvg_zones.append((h2, l1, "bearish"))

    # Liquidity (Equal Highs/Lows)
    liquidity = []
    for i in range(len(H) - 15, len(H) - 1):
        for j in range(i + 1, len(H)):
            if abs(H[i] - H[j]) / H[i] < 0.001:  # 0.1% tolerance
                liquidity.append(("eq_high", H[i]))
            if abs(L[i] - L[j]) / L[i] < 0.001:
                liquidity.append(("eq_low", L[i]))

    # OB zones on 1H
    ob_zones = []
    for i in range(len(k1h) - 15, len(k1h) - 3):
        if i < 3: continue
        o, h, l, c = float(k1h[i][1]), float(k1h[i][2]), float(k1h[i][3]), float(k1h[i][4])
        if c < o and float(k1h[i+1][4]) > float(k1h[i+1][1]) * 1.015:
            ob_zones.append((l, o, "bullish"))
        if c > o and float(k1h[i+1][4]) < float(k1h[i+1][1]) * 0.985:
            ob_zones.append((o, h, "bearish"))

    return trend, breaks, fvg_zones, liquidity, ob_zones


def check_mtf_confluence(direction, h4_trend, h1_trend, h1_ob, h1_fvg, h1_liq, current_price):
    """Check if 15m signal aligns with 4H and 1H"""
    score_adjustment = 0.0
    reasons = []

    # 4H Direction check
    if h4_trend == "BULLISH" and direction == "LONG":
        score_adjustment += 0.10
        reasons.append("4H_Bullish_Align")
    elif h4_trend == "BEARISH" and direction == "SHORT":
        score_adjustment += 0.10
        reasons.append("4H_Bearish_Align")
    elif h4_trend != "NEUTRAL":
        score_adjustment -= 0.10
        reasons.append("4H_Misaligned")

    # 1H Trend check
    if h1_trend == "BULLISH" and direction == "LONG":
        score_adjustment += 0.10
        reasons.append("1H_Bullish_Align")
    elif h1_trend == "BEARISH" and direction == "SHORT":
        score_adjustment += 0.10
        reasons.append("1H_Bearish_Align")
    elif h1_trend != "NEUTRAL":
        score_adjustment -= 0.10
        reasons.append("1H_Misaligned")

    # 1H OB proximity check
    for ob_low, ob_high, ob_type in h1_ob:
        if ob_low * 0.995 <= current_price <= ob_high * 1.005:
            if ob_type == "bullish" and direction == "LONG":
                score_adjustment += 0.05
                reasons.append("1H_OB_Bullish_Proximity")
            elif ob_type == "bearish" and direction == "SHORT":
                score_adjustment += 0.05
                reasons.append("1H_OB_Bearish_Proximity")

    # 1H FVG proximity check
    for fvg_low, fvg_high, fvg_type in h1_fvg:
        if fvg_low * 0.995 <= current_price <= fvg_high * 1.005:
            if fvg_type == "bullish" and direction == "LONG":
                score_adjustment += 0.05
                reasons.append("1H_FVG_Bullish_Proximity")
            elif fvg_type == "bearish" and direction == "SHORT":
                score_adjustment += 0.05
                reasons.append("1H_FVG_Bearish_Proximity")

    # Liquidity sweep check
    for liq_type, liq_price in h1_liq:
        if abs(current_price - liq_price) / liq_price < 0.005:
            if liq_type == "eq_low" and direction == "LONG":
                score_adjustment += 0.05
                reasons.append("1H_Liq_Low_Sweep")
            elif liq_type == "eq_high" and direction == "SHORT":
                score_adjustment += 0.05
                reasons.append("1H_Liq_High_Sweep")

    return score_adjustment, reasons


# ==========================================================
# 6. Order Book Microstructure Analyzer
# ==========================================================
class OrderBookAnalyzer:
    @staticmethod
    def analyze(bids, asks, entry_price=0.0, sl_price=0.0):
        if not bids or not asks:
            return {"imbalance": 0.0, "spread_pct": 0.0, "slippage": 0.0,
                    "stop_hunt_risk": 0.0, "iceberg_bids": 0, "iceberg_asks": 0,
                    "bid_depth": 0.0, "ask_depth": 0.0, "source": "none"}
        bv20 = sum(float(b[1]) for b in bids[:20])
        av20 = sum(float(a[1]) for a in asks[:20])
        bv10 = sum(float(b[1]) for b in bids[:10])
        av10 = sum(float(a[1]) for a in asks[:10])
        tv10 = bv10 + av10 + 1e-9
        imbalance = (bv10 - av10) / tv10
        bb = float(bids[0][0])
        ba = float(asks[0][0])
        spread_pct = ((ba - bb) / bb) * 100 if bb > 0 else 0.0
        slippage = spread_pct
        avg_bid_size = bv10 / 10 if bids else 0
        avg_ask_size = av10 / 10 if asks else 0
        iceberg_bids = sum(1 for b in bids[:20] if float(b[1]) > avg_bid_size * 3)
        iceberg_asks = sum(1 for a in asks[:20] if float(a[1]) > avg_ask_size * 3)
        stop_hunt_risk = 0.0
        if sl_price > 0 and entry_price > 0:
            bid_prices = [float(b[0]) for b in bids[:20]]
            ask_prices = [float(a[0]) for a in asks[:20]]
            all_prices = bid_prices + ask_prices
            sl_distances = [abs(p - sl_price) / sl_price * 100 for p in all_prices if p > 0]
            if sl_distances:
                min_dist = min(sl_distances)
                if min_dist < 0.1:
                    stop_hunt_risk = round(1.0 - (min_dist / 0.1), 2)
        return {
            "imbalance": round(float(imbalance), 4),
            "spread_pct": round(float(spread_pct), 4),
            "slippage": round(float(slippage), 4),
            "stop_hunt_risk": round(float(stop_hunt_risk), 2),
            "iceberg_bids": int(iceberg_bids),
            "iceberg_asks": int(iceberg_asks),
            "bid_depth": round(float(bv20), 2),
            "ask_depth": round(float(av20), 2),
            "source": "multi"
        }

# ==========================================================
# 7. AI Engine
# ==========================================================
class ChartGenerator:
    @staticmethod
    def generate_chart_image(df: pd.DataFrame, symbol: str, signal: dict) -> bytes:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.plot(df.index, df["close"], label="Price", color="#1f77b4", linewidth=1.5)
        entry = signal.get("entry_price", signal.get("entry", 0))
        sl = signal.get("stop_loss", signal.get("sl", 0))
        tp1 = signal.get("tp1", 0)
        if entry: ax.axhline(y=entry, color="blue", linestyle="--", linewidth=1, label="Entry: " + str(entry))
        if sl: ax.axhline(y=sl, color="red", linestyle="--", linewidth=1, label="SL: " + str(round(sl, 4)))
        if tp1: ax.axhline(y=tp1, color="green", linestyle="--", linewidth=1, label="TP1: " + str(round(tp1, 4)))
        ax.set_title(symbol + " - " + signal.get('direction', 'SIGNAL') + " Chart", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

def klines_to_df(klines):
    df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume", "_1", "_2", "_3", "_4", "_5", "_6"])
    return df[["open", "high", "low", "close", "volume"]].astype(float)

class ChartImageAnalyzer:
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            LOGGER.error("Gemini not available: google-genai not installed.")
            self.client = None
            return
        if not api_key:
            LOGGER.error("Gemini API key not set.")
            self.client = None
            return
        try:
            self.client = genai.Client(api_key=api_key)
            LOGGER.info("Gemini client initialized.")
        except Exception as e:
            LOGGER.error("Gemini client init failed: " + str(e))
            self.client = None

    async def analyze_chart_image(self, image_bytes: bytes) -> dict:
        if not self.client:
            LOGGER.error("Gemini client is None.")
            return None
        try:
            prompt_text = (
                "Analyze this trading chart image. "
                "Return ONLY valid JSON with no markdown formatting: "
                '{"pattern_detected": "pattern name or None", '
                '"signal": "LONG or SHORT or NEUTRAL", '
                '"confidence_score": 0-100, '
                '"analysis_summary": "short analysis"}'
            )
            LOGGER.info("Sending {} bytes to Gemini...".format(len(image_bytes)))
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt_text
                ]
            )
            text = ""
            if hasattr(response, "text") and response.text:
                text = response.text
            elif hasattr(response, "parts") and response.parts:
                text = "".join([p.text for p in response.parts if hasattr(p, "text")])
            else:
                LOGGER.error("Unexpected Gemini response format: " + str(type(response)))
                return None
            text = text.replace("```json", "").replace("```", "").strip()
            LOGGER.info("Gemini raw response: " + text[:200] + "...")
            result = json.loads(text)
            LOGGER.info("Gemini parsed: " + str(result))
            return result
        except json.JSONDecodeError as e:
            LOGGER.error("Gemini returned invalid JSON: " + str(e))
            return None
        except Exception as e:
            LOGGER.error("Gemini analysis error: " + type(e).__name__ + ": " + str(e))
            return None

class AIEngine:
    FEATURE_COLS = [
        "rsi","spread_pct","vol_ratio","lower_wick_ratio","upper_wick_ratio",
        "trend_code","adx","plus_di","minus_di","price_to_sma7_ratio","atr_pct","orderbook_imbalance"
    ]
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_samples = 8
        self._training = False
        self.load()

    def save(self):
        try:
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            LOGGER.info("AI model saved.")
        except Exception as e:
            LOGGER.error("Save error: " + str(e))

    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.is_trained = True
                LOGGER.info("AI model loaded.")
            except Exception as e:
                LOGGER.error("Load error: " + str(e))

    async def predict(self, features: Dict[str, float]) -> float:
        if not self.is_trained: return 0.50
        def _pred():
            df = pd.DataFrame([features])[self.FEATURE_COLS].fillna(0.0)
            Xs = self.scaler.transform(df)
            return float(self.model.predict_proba(Xs)[0][1])
        return await asyncio.to_thread(_pred)

    def confidence_label(self, prob: float) -> str:
        if prob >= 0.75: return "Very High"
        if prob >= 0.60: return "High"
        if prob >= 0.55: return "Moderate"
        return "Low"

    async def retrain(self):
        if self._training: return False
        self._training = True
        try:
            df = await db_fetch_df("SELECT * FROM trade_features WHERE outcome IS NOT NULL")
            if len(df) < self.min_samples or len(df["outcome"].unique()) < 2:
                LOGGER.info("Not enough data for training ({}).".format(len(df)))
                return False
            def _train():
                for c in self.FEATURE_COLS:
                    if c not in df.columns: df[c] = 0.0
                X = df[self.FEATURE_COLS].astype(float)
                y = df["outcome"].astype(int)
                pos = (y == 1).sum(); neg = (y == 0).sum()
                spw = float(neg / pos) if pos > 0 else 1.0
                self.model.set_params(scale_pos_weight=spw)
                Xs = self.scaler.fit_transform(X)
                self.model.fit(Xs, y)
                self.is_trained = True
                self.save()
            await asyncio.to_thread(_train)
            LOGGER.info("AI retrained on {} trades.".format(len(df)))
            return True
        except Exception as e:
            LOGGER.error("Retrain error: " + str(e)); return False
        finally:
            self._training = False

# ==========================================================
# 8. Signal Analysis
# ==========================================================
# ==========================================================
# 8. Signal Analysis (3 New Strategies)
# ==========================================================
def _swing_high(H, order=5):
    """Find swing high from High prices list"""
    for i in range(len(H) - 1, order - 1, -1):
        if i - order < 0 or i + order >= len(H):
            continue
        left = H[i - order : i]
        right = H[i + 1 : i + order + 1]
        if H[i] > max(left) and H[i] > max(right):
            return H[i]
    return None


def _swing_low(L, order=5):
    """Find swing low from Low prices list"""
    for i in range(len(L) - 1, order - 1, -1):
        if i - order < 0 or i + order >= len(L):
            continue
        left = L[i - order : i]
        right = L[i + 1 : i + order + 1]
        if L[i] < min(left) and L[i] < min(right):
            return L[i]
    return None


def analyze_signal(klines, symbol, interval, htf_s, htf_r, h4_trend="NEUTRAL", h1_trend="NEUTRAL", h1_ob=[], h1_fvg=[], h1_liq=[], max_sl=2.0):
    if len(klines) < 50: return None
    closed = klines[:-1]
    O = [float(k[1]) for k in closed]
    H = [float(k[2]) for k in closed]
    L = [float(k[3]) for k in closed]
    C = [float(k[4]) for k in closed]
    V = [float(k[5]) for k in closed]
    live = float(klines[-1][4])

    current_time_ms = int(time.time() * 1000)
    candle_start_ms = int(klines[-1][0])
    elapsed = (current_time_ms - candle_start_ms) / 1000.0
    if elapsed > MAX_SIGNAL_AGE:
        return None

    # Calculate all indicators
    rsi = calc_rsi(C)
    atr = calc_atr(H, L, C)
    pdi, mdi, adx = calc_dmi(H, L, C)
    sma7 = sum(C[-7:]) / 7

    co, ch, cl, cc, cv = O[-1], H[-1], L[-1], C[-1], V[-1]
    bb = min(co, cc); bt = max(co, cc)
    body = abs(cc - co); rng = ch - cl
    if rng == 0 or body == 0 or atr == 0: return None
    uw = ch - bt; lw = bb - cl
    spread_pct = (rng / cl) * 100
    if spread_pct > 5.0: return None

    ph, pl = find_pivots(H, L)
    trend = dow_trend(ph, pl)
    avg_v20 = sum(V[-21:-1]) / 20 if len(V) >= 21 else cv
    vol_spike = cv >= 1.2 * avg_v20

    # ==========================================================
    # Strategy 1: RSI+DMI Breakout (Enhanced)
    # ==========================================================
    rsi_len = 14
    dmi_len = 14
    adx_min = 25
    di_lookback = 5
    rsi_lookback = 20

    # Need enough data
    if len(C) >= max(rsi_len, dmi_len, rsi_lookback, di_lookback) + 1:
        # Calculate RSI for lookback window
        rsi_vals = [calc_rsi(C[:i+1]) for i in range(len(C)-rsi_lookback, len(C))]
        if len(rsi_vals) >= rsi_lookback:
            rsi_hi = max(rsi_vals)
            rsi_lo = min(rsi_vals)

            # DI crossover check
            pdi_prev = pdi
            mdi_prev = mdi
            if len(C) > di_lookback + dmi_len * 2:
                _, pdi_prev, mdi_prev = calc_dmi(H[:-di_lookback], L[:-di_lookback], C[:-di_lookback])

            s1_long = (adx >= adx_min and pdi > mdi and pdi > pdi_prev and rsi > rsi_hi)
            s1_short = (adx >= adx_min and mdi > pdi and mdi > mdi_prev and rsi < rsi_lo)
        else:
            s1_long = s1_short = False
    else:
        s1_long = s1_short = False

    # ==========================================================
    # Strategy 2: Candle Setup (Enhanced with shadow_big=4.0)
    # ==========================================================
    shadow_big = 4.0
    shadow_small = 0.25

    if body > 0:
        lower_shadow = lw
        upper_shadow = uw

        s2_long = (lower_shadow >= shadow_big * body and 
                   upper_shadow <= shadow_small * body and
                   cl < sma7 < bb and vol_spike)
        s2_short = (upper_shadow >= shadow_big * body and 
                    lower_shadow <= shadow_small * body and
                    bt < sma7 < ch and vol_spike)
    else:
        s2_long = s2_short = False

    # ==========================================================
    # Strategy 3: HH/LL Breakout with Volume (Swing High/Low)
    # ==========================================================
    swing_order = 5
    vol_period = 20
    vol_ratio = 1.5

    if len(C) >= swing_order * 2 + 1 and len(V) >= vol_period:
        hh = _swing_high(H, swing_order)
        ll = _swing_low(L, swing_order)
        avg_vol = sum(V[-vol_period:]) / vol_period
        vol_ok = cv >= vol_ratio * avg_vol

        s3_long = (hh is not None and cc > hh and vol_ok)
        s3_short = (ll is not None and cc < ll and vol_ok)
    else:
        s3_long = s3_short = False

    # ==========================================================
    # Strategy 4: Advanced Candle (Engulfing + Volume)
    # ==========================================================
    if len(C) >= 3 and body > 0:
        prev_body = abs(C[-2] - O[-2])
        prev_green = C[-2] > O[-2]
        prev_red = C[-2] < O[-2]

        # Bullish Engulfing
        s4_long = (prev_red and prev_body > 0 and 
                   body > prev_body * 1.5 and 
                   cc > O[-2] and co < C[-2] and
                   vol_spike)
        # Bearish Engulfing
        s4_short = (prev_green and prev_body > 0 and 
                    body > prev_body * 1.5 and 
                    cc < O[-2] and co > C[-2] and
                    vol_spike)
    else:
        s4_long = s4_short = False

    # ==========================================================
    # Strategy 5: ATR Breakout (ATR expansion + price move)
    # ==========================================================
    if len(C) >= 20:
        atr_20 = calc_atr(H[-20:], L[-20:], C[-20:])
        atr_current = atr
        atr_expansion = atr_current > atr_20 * 1.5 if atr_20 > 0 else False

        s5_long = (atr_expansion and cc > sma7 and 
                   (cc - C[-5]) / C[-5] * 100 > 1.0 and vol_spike)
        s5_short = (atr_expansion and cc < sma7 and 
                    (C[-5] - cc) / C[-5] * 100 > 1.0 and vol_spike)
    else:
        s5_long = s5_short = False

    # ==========================================================
    # Strategy 6: SMC EQ Sweep (Equal High/Low sweep)
    # ==========================================================
    if len(H) >= 10 and len(L) >= 10:
        eq_high = max(H[-10:-1])
        eq_low = min(L[-10:-1])

        # Equal High sweep (price goes above then back below)
        s6_short = (ch > eq_high * 1.002 and cc < eq_high and 
                    abs(ch - eq_high) / eq_high * 100 < 0.5 and vol_spike)
        # Equal Low sweep (price goes below then back above)
        s6_long = (cl < eq_low * 0.998 and cc > eq_low and 
                   abs(eq_low - cl) / eq_low * 100 < 0.5 and vol_spike)
    else:
        s6_long = s6_short = False

    # ==========================================================
    # Strategy 7: OB Imbalance (Order Block detection)
    # ==========================================================
    if len(C) >= 5:
        # Bullish OB: last 3 candles down, then strong green with volume
        ob_bull = (C[-3] < O[-3] and C[-4] < O[-4] and 
                   C[-2] < O[-2] and cc > co and 
                   body > abs(C[-2] - O[-2]) * 1.5 and vol_spike)
        # Bearish OB: last 3 candles up, then strong red with volume
        ob_bear = (C[-3] > O[-3] and C[-4] > O[-4] and 
                   C[-2] > O[-2] and cc < co and 
                   body > abs(C[-2] - O[-2]) * 1.5 and vol_spike)

        s7_long = ob_bull
        s7_short = ob_bear
    else:
        s7_long = s7_short = False

    # ==========================================================
    # Strategy 8: Hidden Divergence (kept from original)
    # ==========================================================
    hidden_long, hidden_short = detect_hidden_divergence(H, L, C)

    # ==========================================================
    # Voting System (votes_needed=1 - any strategy triggers)
    # ==========================================================
    longs, shorts = [], []

    if s1_long: longs.append("RSI+DMI Breakout")
    if s2_long: longs.append("Candle Setup")
    if s3_long: longs.append("HH/LL Breakout")
    if s4_long: longs.append("Advanced Candle")
    if s5_long: longs.append("ATR Breakout")
    if s6_long: longs.append("SMC EQ Sweep")
    if s7_long: longs.append("OB Imbalance")
    if hidden_long: longs.append("Hidden Divergence")

    if s1_short: shorts.append("RSI+DMI Breakout")
    if s2_short: shorts.append("Candle Setup")
    if s3_short: shorts.append("HH/LL Breakout")
    if s4_short: shorts.append("Advanced Candle")
    if s5_short: shorts.append("ATR Breakout")
    if s6_short: shorts.append("SMC EQ Sweep")
    if s7_short: shorts.append("OB Imbalance")
    if hidden_short: shorts.append("Hidden Divergence")

    # Build signal
    def build(direction, strategies, entry, sl, risk):
        sl_pct = (risk / entry) * 100 if entry > 0 else 999
        if sl_pct <= max_sl and risk > 0:
            if direction == "LONG":
                tp1 = round(entry + (risk * 2), 5)
                tp2 = round(entry + (risk * 5), 5)
                tp3 = round(entry + (risk * 7), 5)
            else:
                tp1 = round(entry - (risk * 2), 5)
                tp2 = round(entry - (risk * 5), 5)
                tp3 = round(entry - (risk * 7), 5)
            return {
                "strategy": " + ".join(strategies), "direction": direction,
                "entry_price": entry, "stop_loss": round(sl, 5), "sl_percent": round(sl_pct, 2),
                "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "rsi": rsi, "adx": adx, "trend": trend, "sma7": round(sma7, 5),
                "atr": atr, "live": live, "vol_spike": vol_spike,
                "lower_wick_ratio": round(lw / rng, 4), "upper_wick_ratio": round(uw / rng, 4),
                "price_to_sma7_ratio": round(cc / sma7, 4), "atr_pct": round((atr / cc) * 100, 4),
                "spread_pct": round(spread_pct, 4), "vol_ratio": round(cv / avg_v20, 4) if avg_v20 > 0 else 1.0,
                "trend_code": 1 if trend == "BULLISH" else (-1 if trend == "BEARISH" else 0),
                "plus_di": pdi, "minus_di": mdi,
                "h4_trend": h4_trend, "h1_trend": h1_trend,
                "h1_ob_count": len(h1_ob), "h1_fvg_count": len(h1_fvg), "h1_liq_count": len(h1_liq)
            }
        return None

    # votes_needed=1: any single strategy triggers
    if longs:
        sl = max(cl, cc - 1.5 * atr)
        risk = cc - sl
        res = build("LONG", longs, cc, sl, risk)
        if res and abs((live - cc) / cc) * 100 <= MAX_SLIPPAGE:
            return res
    if shorts:
        sl = min(ch, cc + 1.5 * atr)
        risk = sl - cc
        res = build("SHORT", shorts, cc, sl, risk)
        if res and abs((cc - live) / cc) * 100 <= MAX_SLIPPAGE:
            return res
    return None
# ==========================================================
# 9. Telegram Manager (Signal Only + Feedback + Chart Analysis)
# ==========================================================
class TelegramManager:
    def __init__(self, token, chat_id, chart_analyzer=None, ai_engine=None, allowed_users=None):
        self.bot = Bot(token=token)
        self.chat_id = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        self.sent_alerts = {}
        self.chart_analyzer = chart_analyzer
        self.ai_engine = ai_engine
        self.allowed_users = allowed_users or []

    def is_authorized(self, user_id):
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users

    async def send(self, text, reply_markup=None, chat_id=None, retries=3):
        target = chat_id if chat_id is not None else self.chat_id
        for i in range(retries):
            try:
                await self.bot.send_message(chat_id=target, text=text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
                return True
            except Exception as e:
                if i == retries - 1:
                    LOGGER.error("TG error: " + str(e))
                    return False
                await asyncio.sleep(2 ** i)

    async def notify_signal(self, signal, symbol, interval, ai_prob, ai_conf, ob_data, btc_trend, alert_id, gemini_result=None, ob_conf_score=None, ob_quality_status=None):
        tv = "https://www.tradingview.com/chart/?symbol=BINANCE:{}".format(symbol)
        dir_emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
        conf_emoji = "🔥" if ai_prob >= 0.75 else ("✅" if ai_prob >= 0.60 else ("⚠️" if ai_prob >= 0.55 else "❓"))

        ob_quality = ""
        if ob_conf_score is not None:
            if ob_conf_score >= 0.7: ob_quality = "🟢 Strong OB | قوی"
            elif ob_conf_score >= 0.4: ob_quality = "🟡 Neutral OB | متوسط"
            else: ob_quality = "🔴 Weak OB | ضعیف"

        ob_filter_status = ""
        if ob_quality_status:
            ob_filter_status = "\n🔍 *OB Auto Filter | فیلتر خودکار:* `{}`".format(ob_quality_status)

        gemini_text = ""
        if gemini_result:
            gemini_text = (
                "\n🧠 *Gemini Vision Analysis | تحلیل تصویری:*\n"
                "• Pattern | الگو: `{}`\n"
                "• Signal | سیگنال: `{}`\n"
                "• Confidence | اطمینان: `{}`\n"
                "• Summary | خلاصه: _{}_"
            ).format(
                str(gemini_result.get('pattern_detected', 'N/A')),
                str(gemini_result.get('signal', 'N/A')),
                str(gemini_result.get('confidence_score', 'N/A')),
                str(gemini_result.get('analysis_summary', 'N/A'))
            )

        msg = (
            "🚨 *NEW TRADING SIGNAL | سیگنال جدید ترید* 🚨\n\n"
            "🪙 *Symbol | ارز:* `#{}`\n"
            "📊 *Direction | جهت:* {} {}\n"
            "🎯 *Strategy | استراتژی:* {} ({})\n"
            "{} *AI Score | امتیاز AI:* `{}` Confidence | اطمینان\n"
            "⏱️ *Timeframe | تایم‌فریم:* {}\n\n"
            "💵 *Entry Price | قیمت ورود:* `{}`\n"
            "🛡️ *Stop Loss | استاپ لاس:* `{}` (`{}%`)\n\n"
            "🎯 *Take Profit Targets | اهداف سود:*\n"
            "🔹 *TP1:* `{}`\n"
            "🔹 *TP2:* `{}`\n"
            "🔹 *TP3:* `{}`\n\n"
            "📉 *RSI:* `{}` | *Trend | ترند:* `{}`\n"
            "🌐 *BTC Trend | ترند بیت‌کوین:* `{}`"
            "{}\n\n"
            "📖 *Order Book Microstructure | سفارشات کتاب:*\n"
            "• Imbalance Ratio | نسبت عدم تعادل: `{}`\n"
            "• Slippage | لغزش قیمت: `{}%`\n"
            "• Stop Hunt Risk | ریسک شکار استاپ: `{}`\n"
            "• Iceberg Bids/Asks | سفارشات یخی خرید/فروش: `{}` / `{}`\n"
            "• Depth (Bid/Ask) | عمق بازار (خرید/فروش): `{:,.0f}` / `{:,.0f}`\n"
            "• Source | منبع: `{}`\n"
            "• OB Quality | کیفیت سفارشات: `{}`"
            "{}"
        ).format(
            symbol,
            signal['direction'], dir_emoji,
            signal['strategy'], interval,
            conf_emoji, "{:.1%}".format(ai_prob),
            interval,
            str(signal['entry_price']),
            str(signal['stop_loss']), str(signal['sl_percent']),
            str(signal['tp1']),
            str(signal['tp2']),
            str(signal['tp3']),
            str(signal['rsi']), signal['trend'],
            btc_trend,
            gemini_text,
            "{:.2f}".format(ob_data['imbalance']),
            "{:.2f}".format(ob_data['slippage']),
            str(ob_data['stop_hunt_risk']),
            str(ob_data['iceberg_bids']), str(ob_data['iceberg_asks']),
            ob_data['bid_depth'], ob_data['ask_depth'],
            str(ob_data.get('source', 'unknown')),
            ob_quality,
            ob_filter_status
        )

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 TradingView", url=tv)],
            [
                InlineKeyboardButton("✅ Good Signal | سیگنال خوب", callback_data="good|" + alert_id),
                InlineKeyboardButton("❌ Bad Signal | سیگنال بد", callback_data="bad|" + alert_id)
            ]
        ])
        await self.send(msg, reply_markup=kb)

    async def notify_feedback(self, alert_id, feedback_type):
        label = "Good signal (AI learning) ✅ | سیگنال خوب (یادگیری AI)" if feedback_type == "good" else "Bad signal (AI learning) ❌ | سیگنال بد (یادگیری AI)"
        await self.send("Feedback recorded | بازخورد ثبت شد: " + label)

    async def analyze_user_chart(self, photo, chat_id):
        LOGGER.info("=" * 50)
        LOGGER.info("analyze_user_chart STARTED for chat_id={}".format(chat_id))
        LOGGER.info("photo type={}".format(type(photo)))
        LOGGER.info("photo.file_id={}".format(getattr(photo, 'file_id', 'NO_FILE_ID')))
        LOGGER.info("photo.file_size={}".format(getattr(photo, 'file_size', 'NO_SIZE')))
        
        try:
            LOGGER.info("Step 1: Getting file from Telegram...")
            file_obj = await self.bot.get_file(photo.file_id)
            LOGGER.info("Step 1 OK: file_path={}".format(file_obj.file_path))
            
            file_url = "https://api.telegram.org/file/bot{}/{}".format(self.bot.token, file_obj.file_path)
            LOGGER.info("Step 2: file_url built (len={})".format(len(file_url)))

            if not self.chart_analyzer:
                LOGGER.error("Step 2 FAILED: chart_analyzer is None!")
                await self.send(
                    "⚠️ *Chart analyzer not initialized | تحلیل‌گر چارت راه‌اندازی نشده*",
                    chat_id=chat_id
                )
                return

            if not self.chart_analyzer.client:
                LOGGER.error("Step 2 FAILED: chart_analyzer.client is None!")
                LOGGER.error("GEMINI_AVAILABLE={}".format(GEMINI_AVAILABLE))
                LOGGER.error("GEMINI_API_KEY set={}".format(bool(os.getenv('GEMINI_API_KEY', ''))))
                await self.send(
                    "⚠️ *Gemini analyzer not available | تحلیل‌گر جمینی در دسترس نیست*\n\n"
                    "Check GEMINI_API_KEY | کلید API رو چک کن",
                    chat_id=chat_id
                )
                return

            async with aiohttp.ClientSession() as s:
                LOGGER.info("Step 3: Downloading image from Telegram...")
                async with s.get(file_url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    LOGGER.info("Step 3: HTTP status={}".format(r.status))
                    if r.status == 200:
                        image_bytes = await r.read()
                        LOGGER.info("Step 3 OK: downloaded {} bytes".format(len(image_bytes)))
                        
                        await self.send(
                            "🧠 *Analyzing chart... | در حال تحلیل چارت...*\n"
                            "_Image size: {} bytes_".format(len(image_bytes)),
                            chat_id=chat_id
                        )

                        LOGGER.info("Step 4: Calling Gemini API...")
                        gemini_result = await self.chart_analyzer.analyze_chart_image(image_bytes)
                        LOGGER.info("Step 4: Gemini result={}".format(gemini_result))

                        if gemini_result:
                            LOGGER.info("Step 5: Saving to database...")
                            try:
                                await db_execute(
                                    "INSERT INTO gemini_analysis (symbol, pattern_detected, signal, confidence_score, analysis_summary) VALUES (?,?,?,?,?)",
                                    ("USER_UPLOAD", 
                                     gemini_result.get("pattern_detected"), 
                                     gemini_result.get("signal"),
                                     gemini_result.get("confidence_score"), 
                                     gemini_result.get("analysis_summary"))
                                )
                                LOGGER.info("Step 5 OK: saved to DB")
                            except Exception as db_e:
                                LOGGER.error("Step 5 DB error: {}".format(db_e))

                            msg = (
                                "🧠 *Gemini Chart Analysis | تحلیل تصویری چارت*\n\n"
                                "📊 *Pattern Detected | الگوی شناسایی شده:*\n"
                                "`{}`\n\n"
                                "📈 *Signal | سیگنال:* `{}`\n\n"
                                "🎯 *Confidence | اطمینان:* `{}%`\n\n"
                                "📝 *Summary | خلاصه:*\n"
                                "_{}_"
                            ).format(
                                str(gemini_result.get('pattern_detected', 'N/A')),
                                str(gemini_result.get('signal', 'N/A')),
                                str(gemini_result.get('confidence_score', 'N/A')),
                                str(gemini_result.get('analysis_summary', 'N/A'))
                            )
                        else:
                            LOGGER.error("Step 4 FAILED: Gemini returned None")
                            msg = "❌ *Analysis failed | تحلیل ناموفق*\n\nGemini could not analyze the image.\n\nPossible reasons:\n• Image too large\n• API rate limit\n• Invalid API key"

                        LOGGER.info("Step 6: Sending result to user...")
                        await self.send(msg, chat_id=chat_id)
                        LOGGER.info("Step 6 OK")
                    else:
                        LOGGER.error("Step 3 FAILED: HTTP {}".format(r.status))
                        await self.send(
                            "❌ *Download failed | دانلود ناموفق*\n\nHTTP {}".format(r.status),
                            chat_id=chat_id
                        )
                        
        except Exception as e:
            LOGGER.error("=" * 50)
            LOGGER.error("analyze_user_chart CRASHED!")
            LOGGER.error("Error type: {}".format(type(e).__name__))
            LOGGER.error("Error message: {}".format(str(e)))
            import traceback
            LOGGER.error("Traceback:\n{}".format(traceback.format_exc()))
            LOGGER.error("=" * 50)
            try:
                await self.send(
                    "❌ *Error | خطا*\n\n"
                    "Could not analyze image | نمی‌تونم عکس رو تحلیل کنم\n\n"
                    "Error: `{}`".format(str(e)[:100]),
                    chat_id=chat_id
                )
            except Exception as send_e:
                LOGGER.error("Failed to send error message: {}".format(send_e))

    async def command_listener(self):
        last_id = 0
        LOGGER.info("Command listener started. Waiting for messages...")
        while True:
            try:
                LOGGER.debug("Polling for updates... last_id={}".format(last_id))
                updates = await self.bot.get_updates(offset=last_id + 1, timeout=5)
                LOGGER.debug("Got {} updates".format(len(updates)))
                
                for u in updates:
                    last_id = u.update_id
                    LOGGER.info("Processing update_id={}".format(u.update_id))

                    # Authorization check
                    user_id = None
                    if u.message and u.message.from_user:
                        user_id = u.message.from_user.id
                        LOGGER.info("Message from user_id={}".format(user_id))
                    elif u.callback_query and u.callback_query.from_user:
                        user_id = u.callback_query.from_user.id
                        LOGGER.info("Callback from user_id={}".format(user_id))

                    if user_id and not self.is_authorized(user_id):
                        LOGGER.warning("Unauthorized access from user: {}".format(user_id))
                        if u.message:
                            try:
                                await self.bot.send_message(
                                    chat_id=u.message.chat_id, 
                                    text="⛔ Access denied. You are not authorized to use this bot."
                                )
                            except: pass
                        continue

                    cid = u.message.chat_id if u.message else (u.callback_query.message.chat_id if u.callback_query else self.chat_id)
                    LOGGER.info("Chat ID: {}".format(cid))

                    # Handle text messages
                    if u.message and u.message.text:
                        cmd = u.message.text.strip().split("@")[0].lower()
                        parts = cmd.split()
                        LOGGER.info("Text command: {}".format(parts[0]))

                        if parts[0] == "/stats":
                            rows = await db_execute("SELECT key, value FROM bot_stats")
                            stats = {r[0]: r[1] for r in rows}
                            total = stats.get("total_signals", 0)
                            good = stats.get("feedback_good", 0)
                            bad = stats.get("feedback_bad", 0)
                            total_fb = good + bad
                            accuracy = round(good / total_fb * 100, 1) if total_fb > 0 else 0
                            msg = (
                                "📊 *Bot Statistics | آمار ربات*\n\n"
                                "🔢 *Total Signals | کل سیگنال‌ها:* `{}`\n"
                                "👍 *Good Feedback | بازخورد خوب:* `{}`\n"
                                "👎 *Bad Feedback | بازخورد بد:* `{}`\n"
                                "🎯 *Accuracy | دقت:* `{}%`\n\n"
                                "🚫 *OB Rejected | رد OB:* `{}`\n"
                                "🚫 *Spread Rejected | رد اسپرد:* `{}`\n"
                                "🚫 *OB Quality Rejected | رد کیفیت:* `{}`\n"
                                "🚫 *OB Depth Rejected | رد عمق:* `{}`\n"
                                "🚫 *OB Stop Hunt Rejected | رد شکار استاپ:* `{}`\n"
                                "🚫 *OB Slippage Rejected | رد لغزش:* `{}`"
                            ).format(
                                str(total),
                                str(good),
                                str(bad),
                                str(accuracy),
                                str(stats.get('ob_rejected', 0)),
                                str(stats.get('spread_rejected', 0)),
                                str(stats.get('ob_quality_rejected', 0)),
                                str(stats.get('ob_depth_rejected', 0)),
                                str(stats.get('ob_stop_hunt_rejected', 0)),
                                str(stats.get('ob_slippage_rejected', 0))
                            )
                            await self.send(msg, chat_id=cid)

                        elif parts[0] == "/result" and len(parts) >= 3:
                            alert_id = parts[1]
                            result = parts[2].lower()
                            if result in ("win", "loss"):
                                outcome = 1 if result == "win" else 0
                                fb_type = "good" if result == "win" else "bad"

                                await db_execute("UPDATE signal_history SET feedback = ? WHERE alert_id = ?", (fb_type, alert_id))
                                await db_execute(
                                    "UPDATE trade_features SET outcome = ? WHERE id = (SELECT id FROM signal_history WHERE alert_id = ?)",
                                    (outcome, alert_id)
                                )
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'feedback_{}'".format(fb_type))

                                await self.send(
                                    "✅ *Result recorded | نتیجه ثبت شد*\n\n"
                                    "Alert ID: `{}`\n"
                                    "Result: `{}`\n\n"
                                    "AI will learn from this feedback. | AI از این بازخورد یاد می‌گیره.".format(alert_id, result.upper()),
                                    chat_id=cid
                                )

                                asyncio.create_task(self.ai_engine.retrain())
                            else:
                                await self.send(
                                    "❌ *Invalid result | نتیجه نامعتبر*\n\n"
                                    "Usage: `/result ALERT_ID win` or `/result ALERT_ID loss`",
                                    chat_id=cid
                                )

                        elif parts[0] == "/analyze":
                            msg = (
                                "🖼️ *Chart Analysis | تحلیل چارت*\n\n"
                                "▫️ عکس چارتت رو بفرست\n"
                                "▫️ ربات خودکار تحلیل می‌کنه\n"
                                "▫️ Gemini AI الگو رو تشخیص می‌ده"
                            )
                            await self.send(msg, chat_id=cid)

                        elif parts[0] == "/help":
                            msg = (
                                "🤖 *Control Menu | منوی کنترل*\n\n"
                                "▫️ `/stats` — Statistics | آمار\n"
                                "▫️ `/result ALERT_ID win/loss` — Report result | گزارش نتیجه\n"
                                "▫️ `/analyze` — Chart analysis | تحلیل چارت\n"
                                "▫️ `/help` — Help | راهنما\n\n"
                                "💡 *How to use | نحوه استفاده:*\n"
                                "1. Wait for signal | منتظر سیگنال بمون\n"
                                "2. Trade manually | خودت ترید کن\n"
                                "3. Report result with `/result` | نتیجه رو گزارش بده"
                            )
                            await self.send(msg, chat_id=cid)

                    # Handle photos - THIS IS THE CRITICAL PART
                    if u.message:
                        LOGGER.info("Message has photo={}".format(bool(u.message.photo)))
                        LOGGER.info("Message has document={}".format(bool(u.message.document)))
                        
                        if u.message.photo:
                            LOGGER.info("Photo detected! Count={}".format(len(u.message.photo)))
                            LOGGER.info("Photo sizes: {}".format([p.file_size for p in u.message.photo]))
                            # Use the largest photo (last one)
                            photo = u.message.photo[-1]
                            LOGGER.info("Selected photo: file_id={}, width={}, height={}, size={}".format(
                                photo.file_id, photo.width, photo.height, photo.file_size
                            ))
                            await self.analyze_user_chart(photo, cid)
                        elif u.message.document:
                            LOGGER.info("Document detected: mime_type={}".format(u.message.document.mime_type))
                            # Check if it's an image
                            if u.message.document.mime_type and 'image' in u.message.document.mime_type:
                                LOGGER.info("Document is an image, treating as photo")
                                # Create a simple object with file_id
                                class FakePhoto:
                                    def __init__(self, file_id):
                                        self.file_id = file_id
                                        self.file_size = 0
                                await self.analyze_user_chart(FakePhoto(u.message.document.file_id), cid)

                    # Handle callback queries
                    if u.callback_query:
                        cq = u.callback_query
                        data = cq.data or ""
                        parts = data.split("|")
                        LOGGER.info("Callback query: data={}".format(data))
                        if len(parts) == 2:
                            fb_type, alert_id = parts[0], parts[1]
                            if fb_type in ("good", "bad"):
                                outcome = 1 if fb_type == "good" else 0

                                await db_execute("UPDATE signal_history SET feedback = ? WHERE alert_id = ?", (fb_type, alert_id))
                                await db_execute(
                                    "UPDATE trade_features SET outcome = ? WHERE id = (SELECT id FROM signal_history WHERE alert_id = ?)",
                                    (outcome, alert_id)
                                )
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'feedback_{}'".format(fb_type))
                                await self.notify_feedback(alert_id, fb_type)

                                asyncio.create_task(self.ai_engine.retrain())

                                try:
                                    await self.bot.answer_callback_query(cq.id)
                                except: pass
            except Exception as e:
                LOGGER.error("Command listener error: {}".format(str(e)))
                import traceback
                LOGGER.error(traceback.format_exc())
            await asyncio.sleep(2)

# ==========================================================
# 10. Main Signal Bot
# ==========================================================
class SignalBot:
    def __init__(self):
        self.ai = AIEngine()
        self.chart_analyzer = ChartImageAnalyzer(GEMINI_API_KEY)
        self.tg = TelegramManager(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, self.chart_analyzer, self.ai)
        self.btc_trend = "NEUTRAL"
        self.btc_pause_until = 0
        self.symbol_cache = {"symbols": [], "last_update": 0, "volumes": {}}

    async def start(self):
        await init_database()
        asyncio.create_task(self.tg.command_listener())
        LOGGER.info("Signal Bot ready. Mode: SIGNAL ONLY + AI Learning from Feedback")

    async def get_symbols(self, session):
        now = time.time()
        if now - self.symbol_cache["last_update"] > 300 or not self.symbol_cache["symbols"]:
            try:
                url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                async with session.get(url, timeout=15) as r:
                    if r.status == 200:
                        data = await r.json()
                        btc_p = next((float(x["lastPrice"]) for x in data if x.get("symbol") == "BTCUSDT"), 60000)
                        min_vol_usdt = MIN_BTC_VOLUME * btc_p
                        syms = []
                        vols = {}
                        for x in data:
                            sym = x["symbol"]
                            if sym.endswith("USDT"):
                                qv = float(x.get("quoteVolume", 0))
                                if qv >= min_vol_usdt:
                                    syms.append(sym)
                                    vols[sym] = qv
                        self.symbol_cache = {"symbols": syms, "last_update": now, "volumes": vols}
                        LOGGER.info("{} symbols loaded (min vol: {:,.0f} USDT).".format(len(syms), min_vol_usdt))
                    else:
                        LOGGER.error("Failed to fetch 24hr ticker: HTTP {}".format(r.status))
            except Exception as e:
                LOGGER.error("Symbol fetch error: " + str(e))
                if not self.symbol_cache["symbols"]:
                    self.symbol_cache = {"symbols": [], "last_update": 0, "volumes": {}}
        return self.symbol_cache["symbols"]

    async def update_btc(self, session):
        try:
            k4h = await fetch_klines(session, "BTCUSDT", "4h")
            if k4h:
                H = [float(k[2]) for k in k4h[:-1]]
                L = [float(k[3]) for k in k4h[:-1]]
                ph, pl = find_pivots(H, L)
                self.btc_trend = dow_trend(ph, pl)
            k15 = await fetch_klines(session, "BTCUSDT", "15m")
            if k15 and len(k15) >= 2:
                change = abs(float(k15[-2][4]) - float(k15[-2][1])) / float(k15[-2][1]) * 100
                if change >= 1.5:
                    self.btc_pause_until = time.time() + 1800
                    LOGGER.warning("BTC Volatility Spike {:.1f}% — 30m pause.".format(change))
        except Exception as e:
            LOGGER.error("BTC update: " + str(e))

    async def process_signal(self, session, symbol, interval, signal, h4_trend="NEUTRAL", h1_trend="NEUTRAL", h1_ob=[], h1_fvg=[], h1_liq=[]):
        bids, asks, ob_source = await fetch_order_book(session, symbol)
        ob = OrderBookAnalyzer.analyze(bids, asks, signal["entry_price"], signal["stop_loss"])
        ob["source"] = ob_source or "failed"

        ob_valid, ob_reason = validate_order_book(ob, signal["direction"], symbol)
        ob_conf = ob_confidence_score(ob, signal["direction"])

        if not ob_valid:
            LOGGER.warning("OB REJECTED {}: {}".format(symbol, ob_reason))
            if "spread" in ob_reason:
                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'spread_rejected'")
            else:
                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'ob_rejected'")
            return

        LOGGER.info("OB ACCEPTED {} from {}: conf={:.2f}, imb={:.2f}".format(symbol, ob_source, ob_conf, ob['imbalance']))

        ob_quality_passed, ob_quality_reason, ob_quality_score = ob_quality_filter(
            ob, signal["direction"], symbol, signal["entry_price"]
        )

        if not ob_quality_passed:
            LOGGER.warning("OB QUALITY FILTER REJECTED {}: {}".format(symbol, ob_quality_reason))
            if "quality_score" in ob_quality_reason:
                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'ob_quality_rejected'")
            elif "depth" in ob_quality_reason:
                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'ob_depth_rejected'")
            elif "stop_hunt" in ob_quality_reason:
                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'ob_stop_hunt_rejected'")
            elif "slippage" in ob_quality_reason:
                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'ob_slippage_rejected'")
            else:
                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'ob_quality_rejected'")
            return

        LOGGER.info("OB QUALITY PASSED {}: score={:.2f}, reason={}".format(symbol, ob_quality_score, ob_quality_reason))

        features = {
            "rsi": signal["rsi"], "spread_pct": ob["spread_pct"],
            "vol_ratio": signal["vol_ratio"], "lower_wick_ratio": signal["lower_wick_ratio"],
            "upper_wick_ratio": signal["upper_wick_ratio"], "trend_code": signal["trend_code"],
            "adx": signal["adx"], "plus_di": signal["plus_di"], "minus_di": signal["minus_di"],
            "price_to_sma7_ratio": signal["price_to_sma7_ratio"], "atr_pct": signal["atr_pct"],
            "orderbook_imbalance": ob["imbalance"]
        }

        # MTF Confluence Adjustment
        mtf_adjustment, mtf_reasons = check_mtf_confluence(
            signal["direction"], h4_trend, h1_trend, h1_ob, h1_fvg, h1_liq, signal["entry_price"]
        )

        base_prob = await self.ai.predict(features)
        prob = min(1.0, max(0.0, base_prob + mtf_adjustment))
        conf_label = self.ai.confidence_label(prob)

        if mtf_reasons:
            LOGGER.info("MTF {}: {} | Adjustment: {:+.2f}".format(symbol, ", ".join(mtf_reasons), mtf_adjustment))

        if prob < 0.55 and self.ai.is_trained:
            LOGGER.info("AI rejected {} (base: {:.2f}, mtf: {:+.2f}, final: {:.2f} — {})".format(
                symbol, base_prob, mtf_adjustment, prob, conf_label))
            return

        gemini_result = None
        if self.chart_analyzer.client:
            try:
                df = klines_to_df(await fetch_klines(session, symbol, interval))
                if df is not None and not df.empty:
                    chart_bytes = ChartGenerator.generate_chart_image(df, symbol, signal)
                    gemini_result = await self.chart_analyzer.analyze_chart_image(chart_bytes)
                    if gemini_result:
                        await db_execute(
                            "INSERT INTO gemini_analysis (symbol, pattern_detected, signal, confidence_score, analysis_summary) VALUES (?,?,?,?,?)",
                            (symbol, gemini_result.get("pattern_detected"), gemini_result.get("signal"),
                             gemini_result.get("confidence_score"), gemini_result.get("analysis_summary"))
                        )
                        gemini_signal = gemini_result.get("signal", "NEUTRAL")
                        gemini_conf = gemini_result.get("confidence_score", 0)
                        if gemini_signal != "NEUTRAL" and gemini_signal != signal["direction"] and gemini_conf >= 70:
                            LOGGER.info("Gemini REJECTED {}: Gemini says {} (conf: {}), we have {}".format(symbol, gemini_signal, gemini_conf, signal['direction']))
                            return
                        LOGGER.info("Gemini APPROVED {}: {} (conf: {})".format(symbol, gemini_signal, gemini_conf))
            except Exception as e:
                LOGGER.error("Gemini chart analysis failed for {}: {}".format(symbol, str(e)))

        alert_id = "{}_{}_{}".format(symbol, interval, int(time.time()))

        # Save features first to get feature_id
        cols = "symbol, rsi, spread_pct, vol_ratio, lower_wick_ratio, upper_wick_ratio, trend_code, adx, plus_di, minus_di, price_to_sma7_ratio, atr_pct, orderbook_imbalance"
        vals = (symbol, features["rsi"], features["spread_pct"], features["vol_ratio"], features["lower_wick_ratio"],
                features["upper_wick_ratio"], features["trend_code"], features["adx"], features["plus_di"],
                features["minus_di"], features["price_to_sma7_ratio"], features["atr_pct"], features["orderbook_imbalance"])
        await db_execute("INSERT INTO trade_features ({}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)".format(cols), vals)
        feat_id = (await db_execute("SELECT last_insert_rowid()"))[0][0]

        # Save signal history with feature_id reference
        await db_execute(
            "INSERT INTO signal_history (alert_id, symbol, interval, direction, strategy, entry_price, stop_loss, tp1, tp2, tp3, sl_percent, rsi, adx, trend, ai_prob, ai_confidence, ob_imbalance, ob_slippage, ob_stop_hunt, ob_iceberg_bids, ob_iceberg_asks, ob_quality_score, ob_rejection_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (alert_id, symbol, interval, signal["direction"], signal["strategy"], signal["entry_price"], signal["stop_loss"], signal["tp1"], signal["tp2"], signal["tp3"], signal["sl_percent"], signal["rsi"], signal["adx"], signal["trend"], prob, conf_label, ob["imbalance"], ob["slippage"], ob["stop_hunt_risk"], ob["iceberg_bids"], ob["iceberg_asks"], ob_quality_score, ob_quality_reason)
        )
        await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'total_signals'")

        LOGGER.info("SIGNAL SENT: {} | Strategy: {} | AI: {:.1%} (base: {:.1%}, mtf: {:+.1%}) | 4H: {} | 1H: {} | OB: {:.2f} | Quality: {:.2f} | Features recorded.".format(symbol, signal["strategy"], prob, base_prob if "base_prob" in dir() else prob, mtf_adjustment if "mtf_adjustment" in dir() else 0, h4_trend, h1_trend, ob_conf, ob_quality_score))

        await self.tg.notify_signal(signal, symbol, interval, prob, conf_label, ob, self.btc_trend, alert_id, gemini_result, ob_conf, ob_quality_reason)

    async def scanner_loop(self):
        async with aiohttp.ClientSession() as session:
            await self.update_btc(session)
            symbols = await self.get_symbols(session)
            btc_counter = 0

            while True:
                try:
                    btc_counter += 1
                    if btc_counter >= 15:
                        await self.update_btc(session)
                        btc_counter = 0

                    if time.time() < self.btc_pause_until:
                        await asyncio.sleep(5)
                        continue

                    for symbol in symbols:
                        vol = self.symbol_cache.get("volumes", {}).get(symbol, 0)
                        if vol <= 0:
                            LOGGER.debug("Skipping {}: no volume data".format(symbol))
                            continue

                        k4h = await fetch_klines(session, symbol, "4h")
                        k1d = await fetch_klines(session, symbol, "1d")
                        htf_s, htf_r = htf_sr(k4h, k1d)

                        for interval in TIMEFRAMES:
                            klines = await fetch_klines(session, symbol, interval)
                            if not klines:
                                continue

                            sig = analyze_signal(klines, symbol, interval, htf_s, htf_r, MAX_SL_PERCENT)
                            if not sig:
                                continue

                            if symbol != "BTCUSDT":
                                if sig["direction"] == "LONG" and self.btc_trend == "BEARISH":
                                    continue
                                if sig["direction"] == "SHORT" and self.btc_trend == "BULLISH":
                                    continue

                            alert_id = "{}_{}_{}_{}".format(symbol, interval, int(klines[-2][0]), sig['direction'])
                            exists = await db_execute("SELECT 1 FROM signal_history WHERE alert_id = ?", (alert_id,))
                            if exists:
                                continue

                            # Get MTF data
                            k4h_data = await fetch_klines(session, symbol, "4h")
                            k1h_data = await fetch_klines(session, symbol, "1h")

                            h4_trend, h4_levels, h4_ob = analyze_4h_direction(k4h_data)
                            h1_trend, h1_breaks, h1_fvg, h1_liq, h1_ob = analyze_1h_structure(k1h_data)

                            await self.process_signal(session, symbol, interval, sig, h4_trend, h1_trend, h1_ob, h1_fvg, h1_liq)
                            await asyncio.sleep(0.02)

                    symbols = await self.get_symbols(session)
                    await asyncio.sleep(5)
                except Exception as e:
                    LOGGER.error("Scanner: " + str(e))
                    await asyncio.sleep(15)

# ==========================================================
# 11. Web & Entry
# ==========================================================
async def health(request):
    return web.Response(text="Signal Bot Running", status=200)

async def main():
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    bot = SignalBot()
    await bot.start()
    await bot.scanner_loop()

if __name__ == "__main__":
    asyncio.run(main())
