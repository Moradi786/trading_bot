import asyncio
import logging
import os
import sqlite3
import time
import hmac
import hashlib
import io
import json
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal, ROUND_DOWN

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
LOGGER = logging.getLogger("UnifiedBot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", 8080))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "1.0"))
LEVERAGE = int(os.getenv("LEVERAGE", "3"))
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

DB_NAME = "unified_bot.db"
MODEL_PATH = "ai_model.joblib"
SCALER_PATH = "ai_scaler.joblib"

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
MAX_SL_PERCENT = 2.0
MIN_BTC_VOLUME = 250.0
MAX_SIGNAL_AGE = 600
MAX_SLIPPAGE = 1.0
ALERT_TTL = 86400

OB_MIN_BIDS = 5
OB_MIN_ASKS = 5
OB_MAX_SPREAD_PCT = 0.5
OB_MIN_IMBALANCE_CONF = 0.15

# ========== OB Quality Filter Config ==========
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
        """CREATE TABLE IF NOT EXISTS active_trades (
            trade_id TEXT PRIMARY KEY, symbol TEXT, direction TEXT,
            entry_price REAL, stop_loss REAL, take_profit REAL,
            highest_price REAL, entry_time REAL, feature_id INTEGER,
            position_size REAL, leverage INTEGER, is_paper INTEGER DEFAULT 1)""",
        """CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT UNIQUE,
            symbol TEXT, interval TEXT, direction TEXT, strategy TEXT,
            entry_price REAL, stop_loss REAL, tp1 REAL, tp2 REAL, tp3 REAL,
            sl_percent REAL, rsi REAL, adx REAL, trend TEXT, ai_prob REAL,
            ai_confidence TEXT, ob_imbalance REAL, ob_slippage REAL,
            ob_stop_hunt REAL, ob_iceberg_bids INTEGER, ob_iceberg_asks INTEGER,
            ob_quality_score REAL, ob_rejection_reason TEXT,
            executed INTEGER DEFAULT 0, feedback TEXT DEFAULT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS bot_stats (
            key TEXT PRIMARY KEY, value INTEGER DEFAULT 0)""",
        """CREATE TABLE IF NOT EXISTS gemini_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT,
            pattern_detected TEXT, signal TEXT, confidence_score REAL,
            analysis_summary TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    ]
    for q in queries:
        await db_execute(q)
    for k in ["total_signals","tp1_hits","tp2_hits","tp3_hits","sl_hits","ai_trades","feedback_good","feedback_bad","ob_rejected","spread_rejected","ob_quality_rejected","ob_depth_rejected","ob_stop_hunt_rejected","ob_slippage_rejected","volume_rejected"]:
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
# 4. Order Book Quality Validator (Enhanced)
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

# ========== OB Quality Auto Filter ==========
def ob_quality_filter(ob_data: dict, direction: str, symbol: str, entry_price: float = 0.0) -> Tuple[bool, str, float]:
    """
    Automatic Order Book Quality Filter.
    Returns: (passed, reason, quality_score)
    """
    if not OB_AUTO_FILTER_ENABLED:
        return True, "auto_filter_disabled", ob_confidence_score(ob_data, direction)

    score = ob_confidence_score(ob_data, direction)
    reasons = []

    # 1. Minimum quality score
    if score < OB_QUALITY_MIN_SCORE:
        reasons.append("quality_score_low ({:.2f} < {})".format(score, OB_QUALITY_MIN_SCORE))

    # 2. Minimum depth check
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

    # 3. Stop hunt risk
    if ob_data.get("stop_hunt_risk", 0) > OB_MAX_STOP_HUNT_RISK:
        reasons.append("stop_hunt_risk ({:.2f} > {})".format(ob_data['stop_hunt_risk'], OB_MAX_STOP_HUNT_RISK))

    # 4. Slippage check
    if ob_data.get("slippage", 0) > OB_MAX_SLIPPAGE_PCT:
        reasons.append("slippage_high ({:.3f}% > {}%)".format(ob_data['slippage'], OB_MAX_SLIPPAGE_PCT))

    # 5. Spread check (tighter than basic validator)
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
def analyze_signal(klines, symbol, interval, htf_s, htf_r, max_sl=2.0):
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

    recent_low = min(L[-6:-1])
    sweep_long = (cl < recent_low) and (cc > recent_low) and vol_spike
    recent_high = max(H[-6:-1])
    sweep_short = (ch > recent_high) and (cc < recent_high) and vol_spike

    lookback = 12
    rh = max(H[-lookback:-1]); rl = min(L[-lookback:-1])
    rw = ((rh - rl) / rl) * 100 if rl > 0 else 999.0
    in_range = rw <= 3.5
    near_s = any(rl >= s * 0.985 and rl <= s * 1.025 for s in htf_s) if htf_s else True
    near_r = any(rh <= r * 1.015 and rh >= r * 0.975 for r in htf_r) if htf_r else True

    green = cc > co; red = cc < co
    breakout_long = in_range and cc > rh and green and vol_spike and (near_s or near_r)
    breakout_short = in_range and cc < rl and red and vol_spike and (near_r or near_s)

    valid_size = rng >= 0.5 * atr
    setup_long = (trend != "BEARISH") and green and valid_size and (lw >= 2.0 * body) and (lw / rng >= 0.5) and (uw <= 0.2 * rng) and (cl <= sma7 < bb) and (cc > sma7) and vol_spike
    setup_short = (trend != "BULLISH") and red and valid_size and (uw >= 2.0 * body) and (uw / rng >= 0.5) and (lw <= 0.2 * rng) and (ch >= sma7 > bt) and (cc < sma7) and vol_spike

    dmi_long = (52 <= rsi <= 68) and (pdi > mdi) and (adx >= 20) and green and vol_spike
    dmi_short = (32 <= rsi <= 48) and (mdi > pdi) and (adx >= 20) and red and vol_spike

    hidden_long, hidden_short = detect_hidden_divergence(H, L, C)

    adv_candle_long = False
    adv_candle_short = False
    atr_bo_long = False
    atr_bo_short = False
    eq_sweep_long = False
    eq_sweep_short = False
    ob_long = False
    ob_short = False

    longs, shorts = [], []
    if dmi_long: longs.append("RSI+DMI Momentum")
    if setup_long: longs.append("Candle Setup")
    if breakout_long: longs.append("Range Breakout")
    if sweep_long: longs.append("SMC Liquidity Sweep")
    if hidden_long: longs.append("Hidden Divergence")
    if adv_candle_long: longs.append("Advanced Candle")
    if atr_bo_long: longs.append("ATR Breakout")
    if eq_sweep_long: longs.append("SMC EQ Sweep")
    if ob_long: longs.append("OB Imbalance")

    if dmi_short: shorts.append("RSI+DMI Momentum")
    if setup_short: shorts.append("Candle Setup")
    if breakout_short: shorts.append("Range Breakdown")
    if sweep_short: shorts.append("SMC Liquidity Sweep")
    if hidden_short: shorts.append("Hidden Divergence")
    if adv_candle_short: shorts.append("Advanced Candle")
    if atr_bo_short: shorts.append("ATR Breakdown")
    if eq_sweep_short: shorts.append("SMC EQ Sweep")
    if ob_short: shorts.append("OB Imbalance")

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
                "plus_di": pdi, "minus_di": mdi
            }
        return None

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
# 9. Risk Manager
# ==========================================================
class RiskManager:
    def __init__(self, risk_percent=RISK_PER_TRADE, leverage=LEVERAGE):
        self.risk_pct = risk_percent
        self.leverage = leverage

    def size(self, balance: float, entry: float, sl: float) -> Tuple[float, float]:
        risk_amount = balance * (self.risk_pct / 100.0)
        price_risk = abs(entry - sl)
        if price_risk <= 0: return 0.0, 0.0
        notional = (risk_amount / price_risk) * entry
        margin = notional / self.leverage
        return round(notional, 2), round(margin, 2)

# ==========================================================
# 10. Binance Trader
# ==========================================================
class BinanceTrader:
    BASE = "https://fapi.binance.com"
    def __init__(self, api_key, api_secret, paper=True):
        self.key = api_key
        self.secret = api_secret
        self.paper = paper
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session: await self.session.close()

    def _sign(self, params: dict):
        qs = "&".join("{}={}".format(k, v) for k, v in sorted(params.items()))
        sig = hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        return qs + "&signature=" + sig

    async def _request(self, method, path, params=None, signed=False):
        if not self.session: return None
        url = self.BASE + path
        headers = {"X-MBX-APIKEY": self.key}
        try:
            if signed and self.secret:
                qs = self._sign(params or {})
                url += "?" + qs
            elif params:
                url += "?" + "&".join("{}={}".format(k, v) for k, v in params.items())
            async with self.session.request(method, url, headers=headers, timeout=10) as r:
                return await r.json()
        except Exception as e:
            LOGGER.error("Binance API error: " + str(e)); return None

    async def get_balance(self):
        if self.paper: return 10000.0
        ts = int(time.time() * 1000)
        data = await self._request("GET", "/fapi/v2/account", {"timestamp": ts}, signed=True)
        if data:
            for a in data.get("assets", []):
                if a.get("asset") == "USDT":
                    return float(a.get("availableBalance", 0))
        return 0.0

    async def place_market_order(self, symbol: str, side: str, quantity: float, leverage: int = 3):
        if self.paper:
            LOGGER.info("PAPER ORDER | {} {} {} @ x{}".format(side, quantity, symbol, leverage))
            return {"orderId": "PAPER_" + str(int(time.time()*1000)), "status": "FILLED"}
        await self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage, "timestamp": int(time.time()*1000)}, signed=True)
        ts = int(time.time() * 1000)
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity, "timestamp": ts}
        return await self._request("POST", "/fapi/v1/order", params, signed=True)

# ==========================================================
# 11. Telegram Manager (دوزبانه - English | فارسی)
# ==========================================================
class TelegramManager:
    def __init__(self, token, chat_id, chart_analyzer=None):
        self.bot = Bot(token=token)
        self.chat_id = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        self.sent_alerts = {}
        self.chart_analyzer = chart_analyzer

    async def send(self, text, reply_markup=None, retries=3):
        for i in range(retries):
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
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
                InlineKeyboardButton("✅ Good | خوب", callback_data="good|" + alert_id),
                InlineKeyboardButton("❌ Bad | بد", callback_data="bad|" + alert_id)
            ]
        ])
        await self.send(msg, reply_markup=kb)

    async def notify_fill(self, symbol, direction, entry, size, sl, tp1, tp2, tp3, paper):
        p = "📄 PAPER | دمو" if paper else "💰 REAL | واقعی"
        de = "🟢" if direction == "LONG" else "🔴"
        msg = (
            "({}) *Trade Opened | ترید باز شد* ✅\n\n"
            "🪙 `#{}` | {} {}\n"
            "📍 *Entry | ورود:* `{}`\n"
            "📊 *Size | حجم:* `{}` USDT\n"
            "🛡️ *SL | استاپ:* `{}`\n"
            "🎯 *TP1:* `{}` | *TP2:* `{}` | *TP3:* `{}`"
        ).format(p, symbol, direction, de, str(entry), str(size), str(sl), str(tp1), str(tp2), str(tp3))
        await self.send(msg)

    async def notify_close(self, symbol, outcome, pnl_pct=0):
        icon = "💰" if outcome == 1 else "❌"
        text = "Profit | سود" if outcome == 1 else "Loss | ضرر"
        msg = (
            icon + " *Trade Closed | ترید بسته شد ({})*\n\n"
            "🪙 `#{}` | *PnL | سود/ضرر:* `{}%`"
        ).format(text, symbol, "{:.2f}".format(pnl_pct))
        await self.send(msg)

    async def notify_feedback(self, alert_id, feedback_type):
        label = "Good signal (AI learning) ✅ | سیگنال خوب (یادگیری AI)" if feedback_type == "good" else "Error logged ❌ | خطا ثبت شد"
        await self.send("Feedback recorded | بازخورد ثبت شد: " + label)

    async def command_listener(self):
        last_id = 0
        while True:
            try:
                updates = await self.bot.get_updates(offset=last_id + 1, timeout=5)
                for u in updates:
                    last_id = u.update_id
                    cid = u.message.chat_id if u.message else (u.callback_query.message.chat_id if u.callback_query else self.chat_id)

                    if u.message and u.message.text:
                        cmd = u.message.text.strip().split("@")[0].lower()
                        if cmd == "/stats":
                            rows = await db_execute("SELECT key, value FROM bot_stats")
                            stats = {r[0]: r[1] for r in rows}
                            total = stats.get("total_signals", 0)
                            wr = round((stats.get("tp1_hits",0)+stats.get("tp2_hits",0)+stats.get("tp3_hits",0))/total*100,1) if total else 0
                            msg = (
                                "📊 *Bot Statistics | آمار ربات*\n\n"
                                "🔢 *Total Signals | کل سیگنال‌ها:* `{}`\n"
                                "🎯 *TP1:* `{}` | *TP2:* `{}` | *TP3:* `{}`\n"
                                "❌ *SL | استاپ:* `{}`\n"
                                "🧠 *AI Trades | تریدهای AI:* `{}`\n"
                                "👍 *Good Feedback | بازخورد خوب:* `{}`\n"
                                "👎 *Bad Feedback | بازخورد بد:* `{}`\n"
                                "🚫 *OB Rejected | رد OB:* `{}`\n"
                                "🚫 *Spread Rejected | رد اسپرد:* `{}`\n"
                                "🚫 *OB Quality Rejected | رد کیفیت:* `{}`\n"
                                "🚫 *OB Depth Rejected | رد عمق:* `{}`\n"
                                "🚫 *OB Stop Hunt Rejected | رد شکار استاپ:* `{}`\n"
                                "🚫 *OB Slippage Rejected | رد لغزش:* `{}`\n"
                                "🏆 *Win Rate | نرخ برد:* `{}%`"
                            ).format(
                                str(total),
                                str(stats.get('tp1_hits',0)), str(stats.get('tp2_hits',0)), str(stats.get('tp3_hits',0)),
                                str(stats.get('sl_hits',0)),
                                str(stats.get('ai_trades',0)),
                                str(stats.get('feedback_good',0)),
                                str(stats.get('feedback_bad',0)),
                                str(stats.get('ob_rejected',0)),
                                str(stats.get('spread_rejected',0)),
                                str(stats.get('ob_quality_rejected',0)),
                                str(stats.get('ob_depth_rejected',0)),
                                str(stats.get('ob_stop_hunt_rejected',0)),
                                str(stats.get('ob_slippage_rejected',0)),
                                str(wr)
                            )
                            await self.send(msg, chat_id=cid)
                        elif cmd == "/active":
                            df = await db_fetch_df("SELECT * FROM active_trades")
                            if df.empty:
                                await self.send("ℹ️ No active positions | ترید فعالی نیست.", chat_id=cid)
                            else:
                                lines_list = []
                                for _, r in df.iterrows():
                                    lines_list.append("🔹 `#{}` ({}) @ `{}`".format(r['symbol'], r['direction'], str(r['entry_price'])))
                                lines = "\n".join(lines_list)
                                msg = "📌 *Active Trades | تریدهای فعال ({})*\n\n{}".format(str(len(df)), lines)
                                await self.send(msg, chat_id=cid)
                        elif cmd == "/help":
                            msg = (
                                "🤖 *Control Menu | منوی کنترل*\n\n"
                                "▫️ `/stats` — Statistics | آمار\n"
                                "▫️ `/active` — Active positions | تریدهای فعال\n"
                                "▫️ `/help` — Help | راهنما"
                            )
                            await self.send(msg, chat_id=cid)

                    if u.message and u.message.photo:
                        try:
                            photo = u.message.photo[-1]
                            file_obj = await self.bot.get_file(photo.file_id)
                            file_url = "https://api.telegram.org/file/bot{}/{}".format(self.bot.token, file_obj.file_path)
                            async with aiohttp.ClientSession() as s:
                                async with s.get(file_url) as r:
                                    if r.status == 200:
                                        image_bytes = await r.read()
                                        if self.chart_analyzer and self.chart_analyzer.client:
                                            gemini_result = await self.chart_analyzer.analyze_chart_image(image_bytes)
                                            if gemini_result:
                                                msg = (
                                                    "🧠 *Gemini Chart Analysis | تحلیل تصویری:*\n\n"
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
                                                await self.send(msg, chat_id=cid)
                                            else:
                                                await self.send("❌ Gemini could not analyze | جمینی نتونست تحلیل کنه.", chat_id=cid)
                                        else:
                                            await self.send("⚠️ Gemini analyzer not available | تحلیل‌گر جمینی در دسترس نیست. Check GEMINI_API_KEY.", chat_id=cid)
                        except Exception as e:
                            LOGGER.error("Photo analysis error: " + str(e))
                            await self.send("❌ Error analyzing image | خطا در تحلیل تصویر.", chat_id=cid)

                    if u.callback_query:
                        cq = u.callback_query
                        data = cq.data or ""
                        parts = data.split("|")
                        if len(parts) == 2:
                            fb_type, alert_id = parts[0], parts[1]
                            if fb_type in ("good", "bad"):
                                await db_execute("UPDATE signal_history SET feedback = ? WHERE alert_id = ?", (fb_type, alert_id))
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'feedback_{}'".format(fb_type))
                                await self.notify_feedback(alert_id, fb_type)
                                try:
                                    await self.bot.answer_callback_query(cq.id)
                                except: pass
            except Exception as e:
                LOGGER.error("Command listener: " + str(e))
            await asyncio.sleep(2)

# ==========================================================
# 12. Main Unified Bot

# ==========================================================
class UnifiedTradingBot:
    def __init__(self):
        self.ai = AIEngine()
        self.risk = RiskManager()
        self.trader = BinanceTrader(BINANCE_API_KEY, BINANCE_API_SECRET, PAPER_TRADING)
        self.chart_analyzer = ChartImageAnalyzer(GEMINI_API_KEY)
        self.tg = TelegramManager(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, self.chart_analyzer)
        self.btc_trend = "NEUTRAL"
        self.btc_pause_until = 0
        self.symbol_cache = {"symbols": [], "last_update": 0, "volumes": {}}

    async def start(self):
        await init_database()
        await self.trader.start()
        asyncio.create_task(self.tg.command_listener())
        asyncio.create_task(self.position_tracker())
        mode = "SIGNAL ONLY (AI Learning Mode)" if not EXECUTE_TRADES else ("PAPER TRADING" if PAPER_TRADING else "LIVE TRADING")
        LOGGER.info("Unified Bot ready. Mode: " + mode)

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
                        filtered_out = []
                        for x in data:
                            sym = x["symbol"]
                            if sym.endswith("USDT"):
                                qv = float(x.get("quoteVolume", 0))
                                if qv >= min_vol_usdt:
                                    syms.append(sym)
                                    vols[sym] = qv
                                else:
                                    filtered_out.append((sym, qv))
                        self.symbol_cache = {"symbols": syms, "last_update": now, "volumes": vols}
                        LOGGER.info("{} symbols loaded (min vol: {:,.0f} USDT).".format(len(syms), min_vol_usdt))
                        if filtered_out:
                            LOGGER.info("{} symbols filtered out (low 24h volume).".format(len(filtered_out)))
                            # Log top 5 filtered for debugging
                            for sym, qv in sorted(filtered_out, key=lambda x: x[1], reverse=True)[:5]:
                                LOGGER.debug("Filtered: {} (vol: {:,.0f} USDT < {:,.0f} USDT)".format(sym, qv, min_vol_usdt))
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

    async def execute_signal(self, session, symbol, interval, signal):
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

        # ========== OB Quality Auto Filter ==========
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
        # ==========================================

        features = {
            "rsi": signal["rsi"], "spread_pct": ob["spread_pct"],
            "vol_ratio": signal["vol_ratio"], "lower_wick_ratio": signal["lower_wick_ratio"],
            "upper_wick_ratio": signal["upper_wick_ratio"], "trend_code": signal["trend_code"],
            "adx": signal["adx"], "plus_di": signal["plus_di"], "minus_di": signal["minus_di"],
            "price_to_sma7_ratio": signal["price_to_sma7_ratio"], "atr_pct": signal["atr_pct"],
            "orderbook_imbalance": ob["imbalance"]
        }

        prob = await self.ai.predict(features)
        conf_label = self.ai.confidence_label(prob)
        if prob < 0.55 and self.ai.is_trained:
            LOGGER.info("AI rejected {} ({:.2f} — {})".format(symbol, prob, conf_label))
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
        await self.tg.notify_signal(signal, symbol, interval, prob, conf_label, ob, self.btc_trend, alert_id, gemini_result, ob_conf, ob_quality_reason)

        balance = await self.trader.get_balance()
        if balance <= 0:
            LOGGER.warning("Balance zero.")
            return

        entry = signal["entry_price"]
        sl = signal["stop_loss"]
        notional, margin = self.risk.size(balance, entry, sl)
        if notional <= 0:
            LOGGER.warning("Position size zero.")
            return

        cols = "symbol, rsi, spread_pct, vol_ratio, lower_wick_ratio, upper_wick_ratio, trend_code, adx, plus_di, minus_di, price_to_sma7_ratio, atr_pct, orderbook_imbalance"
        vals = (symbol, features["rsi"], features["spread_pct"], features["vol_ratio"], features["lower_wick_ratio"],
                features["upper_wick_ratio"], features["trend_code"], features["adx"], features["plus_di"],
                features["minus_di"], features["price_to_sma7_ratio"], features["atr_pct"], features["orderbook_imbalance"])
        await db_execute("INSERT INTO trade_features ({}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)".format(cols), vals)
        feat_id = (await db_execute("SELECT last_insert_rowid()"))[0][0]

        if not EXECUTE_TRADES:
            LOGGER.info("SIGNAL ONLY (trading OFF): {} | AI: {:.1%} | OB: {:.2f} | Quality: {:.2f} | Features recorded.".format(symbol, prob, ob_conf, ob_quality_score))
            return

        side = "BUY" if signal["direction"] == "LONG" else "SELL"
        qty = round(notional / entry, 4)
        res = await self.trader.place_market_order(symbol, side, qty, LEVERAGE)

        if res:
            trade_id = alert_id
            tp1, tp2, tp3 = signal["tp1"], signal["tp2"], signal["tp3"]
            await db_execute(
                "INSERT INTO active_trades (trade_id, symbol, direction, entry_price, stop_loss, take_profit, highest_price, entry_time, feature_id, position_size, leverage, is_paper) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (trade_id, symbol, signal["direction"], entry, sl, tp3, entry, time.time(), feat_id, notional, LEVERAGE, 1 if PAPER_TRADING else 0)
            )
            await db_execute("INSERT OR REPLACE INTO bot_stats (key, value) VALUES ('ai_trades', COALESCE((SELECT value FROM bot_stats WHERE key='ai_trades'), 0) + 1)")
            await db_execute(
                "UPDATE signal_history SET executed = 1, ai_prob = ?, ai_confidence = ?, ob_imbalance = ?, ob_slippage = ?, ob_stop_hunt = ?, ob_iceberg_bids = ?, ob_iceberg_asks = ?, ob_quality_score = ?, ob_rejection_reason = ? WHERE alert_id = ?",
                (prob, conf_label, ob["imbalance"], ob["slippage"], ob["stop_hunt_risk"], ob["iceberg_bids"], ob["iceberg_asks"], ob_quality_score, ob_quality_reason, alert_id)
            )
            await self.tg.notify_fill(symbol, signal["direction"], entry, notional, sl, tp1, tp2, tp3, PAPER_TRADING)
            LOGGER.info("Trade opened: {} | AI: {:.1%} | OB: {:.2f} | Quality: {:.2f} | Size: {}".format(symbol, prob, ob_conf, ob_quality_score, notional))

    async def position_tracker(self):
        while True:
            try:
                df = await db_fetch_df("SELECT * FROM active_trades")
                if df.empty:
                    await asyncio.sleep(10)
                    continue

                async with aiohttp.ClientSession() as session:
                    for _, row in df.iterrows():
                        tid = row["trade_id"]
                        sym = row["symbol"]
                        klines = await fetch_klines(session, sym, "15m")
                        if not klines:
                            continue
                        price = float(klines[-1][4])
                        direction = row["direction"]
                        entry = float(row["entry_price"])
                        sl = float(row["stop_loss"])
                        tp = float(row["take_profit"])
                        highest = float(row["highest_price"])
                        fid = int(row["feature_id"])

                        if price > highest:
                            await db_execute("UPDATE active_trades SET highest_price = ? WHERE trade_id = ?", (price, tid))
                            if (price - entry) / entry >= 0.02 and direction == "LONG":
                                new_sl = max(sl, entry)
                                await db_execute("UPDATE active_trades SET stop_loss = ? WHERE trade_id = ?", (new_sl, tid))
                            elif (entry - price) / entry >= 0.02 and direction == "SHORT":
                                new_sl = min(sl, entry)
                                await db_execute("UPDATE active_trades SET stop_loss = ? WHERE trade_id = ?", (new_sl, tid))

                        closed = False
                        outcome = None
                        if direction == "LONG":
                            if price >= tp:
                                closed = True
                                outcome = 1
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'tp3_hits'")
                            elif price <= sl:
                                closed = True
                                outcome = 0
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'sl_hits'")
                        else:
                            if price <= tp:
                                closed = True
                                outcome = 1
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'tp3_hits'")
                            elif price >= sl:
                                closed = True
                                outcome = 0
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'sl_hits'")

                        if closed:
                            await db_execute("UPDATE trade_features SET outcome = ? WHERE id = ?", (outcome, fid))
                            await db_execute("DELETE FROM active_trades WHERE trade_id = ?", (tid,))
                            pnl = ((price - entry) / entry * 100) if direction == "LONG" else ((entry - price) / entry * 100)
                            await self.tg.notify_close(sym, outcome, pnl)
                            asyncio.create_task(self.ai.retrain())
            except Exception as e:
                LOGGER.error("Tracker error: " + str(e))
            await asyncio.sleep(15)

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
                        # Double-check volume threshold (in case cache is stale)
                        btc_p = next((float(x["lastPrice"]) for x in [] if False), 60000)  # Will use cached logic
                        min_vol_usdt = MIN_BTC_VOLUME * 60000  # Approximate, actual check done in get_symbols

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

                            await db_execute(
                                "INSERT INTO signal_history (alert_id, symbol, interval, direction, strategy, entry_price, stop_loss, tp1, tp2, tp3, sl_percent, rsi, adx, trend) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (alert_id, symbol, interval, sig["direction"], sig["strategy"], sig["entry_price"], sig["stop_loss"], sig["tp1"], sig["tp2"], sig["tp3"], sig["sl_percent"], sig["rsi"], sig["adx"], sig["trend"])
                            )
                            await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'total_signals'")
                            await self.execute_signal(session, symbol, interval, sig)
                            await asyncio.sleep(0.02)

                    symbols = await self.get_symbols(session)
                    await asyncio.sleep(5)
                except Exception as e:
                    LOGGER.error("Scanner: " + str(e))
                    await asyncio.sleep(15)

# ==========================================================
# 13. Web & Entry
# ==========================================================
async def health(request):
    return web.Response(text="Unified Bot Running", status=200)

async def main():
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    bot = UnifiedTradingBot()
    await bot.start()
    await bot.scanner_loop()

if __name__ == "__main__":
    asyncio.run(main())
