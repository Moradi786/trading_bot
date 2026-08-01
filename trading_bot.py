
import asyncio
import logging
import os
import sqlite3
import time
import io
import json
from typing import Dict, Any, Optional, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import aiohttp
import numpy as np
from datetime import datetime, timedelta, timezone
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
# 0. Config - All API URLs from .env only
# ==========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("SignalBot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")
PORT = int(os.getenv("PORT", 8080))

# API URLs - MUST be set in .env (no defaults in code)
BINANCE_FUTURES_KLINES_URL = os.getenv("BINANCE_FUTURES_KLINES_URL")
BINANCE_FUTURES_DEPTH_URL = os.getenv("BINANCE_FUTURES_DEPTH_URL")
BINANCE_FUTURES_TICKER_URL = os.getenv("BINANCE_FUTURES_TICKER_URL")
BINANCE_SPOT_KLINES_URL = os.getenv("BINANCE_SPOT_KLINES_URL")
BINANCE_SPOT_DEPTH_URL = os.getenv("BINANCE_SPOT_DEPTH_URL")
BYBIT_KLINES_URL = os.getenv("BYBIT_KLINES_URL")
BYBIT_DEPTH_URL = os.getenv("BYBIT_DEPTH_URL")
OKX_KLINES_URL = os.getenv("OKX_KLINES_URL")
OKX_DEPTH_URL = os.getenv("OKX_DEPTH_URL")
BITGET_FUTURES_DEPTH_URL = os.getenv("BITGET_FUTURES_DEPTH_URL")
BITGET_SPOT_DEPTH_URL = os.getenv("BITGET_SPOT_DEPTH_URL")
COINMARKETCAP_URL = os.getenv("COINMARKETCAP_URL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# Validate required API URLs
def validate_api_urls():
    required_urls = {
        "BINANCE_FUTURES_KLINES_URL": BINANCE_FUTURES_KLINES_URL,
        "BINANCE_FUTURES_DEPTH_URL": BINANCE_FUTURES_DEPTH_URL,
        "BINANCE_FUTURES_TICKER_URL": BINANCE_FUTURES_TICKER_URL,
    }
    missing = [k for k, v in required_urls.items() if not v]
    if missing:
        LOGGER.error("Missing required API URLs in .env: {}".format(", ".join(missing)))
        LOGGER.error("Please set all API URLs in your .env file")
        raise ValueError("Missing API URLs: {}".format(", ".join(missing)))
    LOGGER.info("All API URLs validated successfully")

# Allowed users (comma-separated Telegram IDs). Empty = allow all.
ALLOWED_USER_IDS_STR = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS = [int(x.strip()) for x in ALLOWED_USER_IDS_STR.split(",") if x.strip().isdigit()] if ALLOWED_USER_IDS_STR else []

# Check interval for market scanner (seconds, minimum 5)
CHECK_INTERVAL_SECONDS = max(5, int(os.getenv("CHECK_INTERVAL_SECONDS", "5")))

DB_NAME = "signal_bot.db"
MODEL_PATH = "ai_model.joblib"
SCALER_PATH = "ai_scaler.joblib"

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
MAX_SL_PERCENT = 2.0
# فیلتر کوین‌ها: فقط حجم ۲۴ ساعته بالاتر از این مقدار BTC
MIN_BTC_VOLUME = float(os.getenv("MIN_BTC_VOLUME", "250.0"))

# ==========================================================
# SIGNAL FILTERS - همه از .env خوانده می‌شوند
# فقط .env را عوض کن، به کد دست نزن
# ==========================================================
# حداکثر تعداد سیگنال در روز (0 = نامحدود)
MAX_DAILY_SIGNALS = int(os.getenv("MAX_DAILY_SIGNALS", "0"))
# حداقل اطمینان AI برای ارسال سیگنال (0.0 - 1.0)
MIN_AI_CONFIDENCE = float(os.getenv("MIN_AI_CONFIDENCE", "0.55"))
# حداقل ADX (قدرت ترند) — 0 = غیرفعال
MIN_ADX = float(os.getenv("MIN_ADX", "0"))
# فیلتر جهت ترند بیت‌کوین (true/false)
FILTER_BTC_TREND = os.getenv("FILTER_BTC_TREND", "true").lower() == "true"
# فاصله بین دو سیگنال یک کوین (به دقیقه)
SIGNAL_COOLDOWN_MINUTES = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "240"))
# فیلتر AI حتی وقتی مدل هنوز آموزش ندیده — سیگنال با امتیاز زیر حد ارسال نمی‌شود
AI_FILTER_UNTRAINED = os.getenv("AI_FILTER_UNTRAINED", "true").lower() == "true"
# فیلتر ترند سراسری: LONG فقط بالای EMA / SHORT فقط زیر EMA (همه استراتژی‌ها)
TREND_FILTER_ENABLED = os.getenv("TREND_FILTER_ENABLED", "true").lower() == "true"
TREND_FILTER_EMA = int(os.getenv("TREND_FILTER_EMA", "50"))

# ============ NEWS FILTERS | فیلتر خبرها ============
# ⛔ توقف سیگنال موقع خبرهای بزرگ (FOMC خودکار + خبرهای دستی)
NEWS_BLACKOUT_ENABLED = os.getenv("NEWS_BLACKOUT_ENABLED", "true").lower() == "true"
NEWS_BLACKOUT_BEFORE_MIN = int(os.getenv("NEWS_BLACKOUT_BEFORE_MIN", "30"))   # 30 دقیقه قبل خبر توقف
NEWS_BLACKOUT_AFTER_MIN = int(os.getenv("NEWS_BLACKOUT_AFTER_MIN", "60"))     # 60 دقیقه بعد خبر توقف
# خبرهای دستی: فرمت UTC جدا با کاما → مثال: 2026-07-29 18:00,2026-08-01 12:30
NEWS_MANUAL_EVENTS = os.getenv("NEWS_MANUAL_EVENTS", "")
# 📰 فیلتر احساسات خبری CryptoPanic (نیاز به API Key رایگان از cryptopanic.com)
NEWS_SENTIMENT_FILTER = os.getenv("NEWS_SENTIMENT_FILTER", "false").lower() == "true"
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
NEWS_BLOCK_RATIO = float(os.getenv("NEWS_BLOCK_RATIO", "0.65"))   # اگر 65%+ خبرها برخلاف سیگنال بود، بلاک
NEWS_MIN_VOTES = int(os.getenv("NEWS_MIN_VOTES", "5"))            # حداقل تعداد خبر برای قضاوت
NEWS_CACHE_MINUTES = int(os.getenv("NEWS_CACHE_MINUTES", "15"))   # کش ۱۵ دقیقه‌ای
# 📊 فیچر ۳: تقویت AI Score با احساسات خبری (نیاز به CRYPTOPANIC_API_KEY)
NEWS_AI_SCORE = os.getenv("NEWS_AI_SCORE", "false").lower() == "true"
NEWS_AI_BOOST = float(os.getenv("NEWS_AI_BOOST", "0.05"))         # +5% اگر خبرها همجهت، -5% اگر برخلاف
# ⚠️ فیچر ۴: هشدار تلگرام قبل از خبرهای مهم
NEWS_ALERT_ENABLED = os.getenv("NEWS_ALERT_ENABLED", "true").lower() == "true"
NEWS_ALERT_BEFORE_MIN = int(os.getenv("NEWS_ALERT_BEFORE_MIN", "30"))  # 30 دقیقه قبل خبر هشدار

# ============ WEEKLY MOVE REPORT | گزارش هفتگی کوین‌های آماده حرکت ============
WEEKLY_REPORT_ENABLED = os.getenv("WEEKLY_REPORT_ENABLED", "true").lower() == "true"
WEEKLY_REPORT_DAY = int(os.getenv("WEEKLY_REPORT_DAY", "6"))      # 6=یکشنبه (پیش‌فرض) - گزارش قبل از شروع هفته معاملاتی
WEEKLY_REPORT_HOUR = int(os.getenv("WEEKLY_REPORT_HOUR", "6"))    # ساعت ارسال (UTC)
WEEKLY_TOP_N = int(os.getenv("WEEKLY_TOP_N", "10"))               # چند کوین در لیست باشه

# ============ TWO-TIER SMART SCANNER | اسکنر دولایه هوشمند ============
SCAN_FAST_TOP_N = int(os.getenv("SCAN_FAST_TOP_N", "60"))    # 60 کوین فعال (حجم بالا) = هر اسکن
SCAN_SLOW_EVERY = int(os.getenv("SCAN_SLOW_EVERY", "12"))    # بقیه هر 12 اسکن (≈ هر 1 دقیقه با اینتروال 5ثانیه)

# تقویم FOMC 2026 (ساعت ۱۸:۰۰ UTC = اعلان نرخ بهره)
FOMC_DATES_2026 = [
    "2026-01-28 18:00", "2026-03-18 17:00", "2026-04-29 18:00",
    "2026-06-17 18:00", "2026-07-29 18:00", "2026-09-16 18:00",
    "2026-10-28 18:00", "2026-12-09 19:00",
]
# تایم‌فریم‌های مجاز — اگر خالی باشد همه استفاده می‌شوند
ALLOWED_TIMEFRAMES_STR = os.getenv("ALLOWED_TIMEFRAMES", "")
if ALLOWED_TIMEFRAMES_STR:
    ALLOWED_TIMEFRAMES = [t.strip() for t in ALLOWED_TIMEFRAMES_STR.split(",") if t.strip()]
else:
    ALLOWED_TIMEFRAMES = TIMEFRAMES
# استراتژی‌های مجاز — اگر خالی باشد همه مجازند
ALLOWED_STRATEGIES_STR = os.getenv("ALLOWED_STRATEGIES", "")
ALLOWED_STRATEGIES = [s.strip() for s in ALLOWED_STRATEGIES_STR.split(",") if s.strip()] if ALLOWED_STRATEGIES_STR else []

# فیلتر RSI: سیگنال LONG فقط اگر RSI بالای این مقدار (0 = غیرفعال)
MIN_RSI_LONG = float(os.getenv("MIN_RSI_LONG", "0"))
# فیلتر RSI: سیگنال SHORT فقط اگر RSI پایین این مقدار (0 = غیرفعال)
MAX_RSI_SHORT = float(os.getenv("MAX_RSI_SHORT", "0"))

# ==========================================================
# Strategy 2 (Candle Setup) - قابل تنظیم از .env
# ==========================================================
S2_SHADOW_BIG = float(os.getenv("S2_SHADOW_BIG", "2.0"))        # سایه حداقل چند برابر بدنه
S2_SHADOW_SMALL = float(os.getenv("S2_SHADOW_SMALL", "0.25"))   # سایه سمت دیگر حداکثر
S2_VOLUME_RATIO = float(os.getenv("S2_VOLUME_RATIO", "1.2"))    # حداقل ضریب حجم
S2_SR_FILTER = os.getenv("S2_SR_FILTER", "true").lower() == "true"  # فقط نزدیک حمایت/مقاومت
S2_SR_PROXIMITY_PCT = float(os.getenv("S2_SR_PROXIMITY_PCT", "1.5"))  # فاصله مجاز به S/R (٪)

# ==========================================================
# Strategy 3 (HH/LL Breakout) - قابل تنظیم از .env
# ==========================================================
S3_SMA7_FILTER = os.getenv("S3_SMA7_FILTER", "true").lower() == "true"  # شکست فقط در جهت SMA7
S3_VOLUME_RATIO = float(os.getenv("S3_VOLUME_RATIO", "1.5"))            # حداقل ضریب حجم شکست

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

# 🧱 فیچر ۱: تشخیص دیوار (بزرگترین سفارش خرید/فروش)
OB_WALL_FILTER = os.getenv("OB_WALL_FILTER", "true").lower() == "true"
OB_WALL_MIN_MULT = float(os.getenv("OB_WALL_MIN_MULT", "5.0"))    # دیوار = 5 برابر سفارش متوسط
OB_WALL_BLOCK_PCT = float(os.getenv("OB_WALL_BLOCK_PCT", "0.5"))  # دیوار برخلاف در فاصله 0.5% → بلاک
# 📐 فیچر ۲: عدم تعادل سه‌لایه (تله: لایه نزدیک برخلاف لایه دور)
OB_BAND_TRAP_FILTER = os.getenv("OB_BAND_TRAP_FILTER", "true").lower() == "true"
# 📈 فیچر ۳: مومنتوم عدم تعادل (فشار پایدار خرید/فروش)
OB_MOMENTUM_FILTER = os.getenv("OB_MOMENTUM_FILTER", "true").lower() == "true"
OB_MOMENTUM_MIN = float(os.getenv("OB_MOMENTUM_MIN", "0.15"))     # آستانه فشار برخلاف
OB_MOMENTUM_SAMPLES = int(os.getenv("OB_MOMENTUM_SAMPLES", "3"))  # میانگین 3 قرائت اخیر
# 🎭 فیچر ۴: تشخیص دیوار فیک (اسپوفینگ - دیواری که ناگهان محو می‌شه)
OB_SPOOF_FILTER = os.getenv("OB_SPOOF_FILTER", "true").lower() == "true"

# ==========================================================
# SYMBOL FILTERING - فقط کریپتو و طلا
# ==========================================================
# نمادهای طلا که از CoinMarketCap گرفته می‌شوند
GOLD_SYMBOLS = ["PAXGUSDT", "XAUTUSDT"]

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
# Note: Exchanges with None URLs will be skipped
EXCHANGES = [
    {"name":"Binance","weight":10,"limiter":binance_limiter,
     "url": (BINANCE_FUTURES_KLINES_URL + "?symbol={symbol}&interval={interval}&limit=200") if BINANCE_FUTURES_KLINES_URL else None,
     "interval_map":{"15m":"15m","1h":"1h","4h":"4h","1d":"1d"},
     "parser": lambda d: d if isinstance(d, list) else None},
    {"name":"Bybit","weight":8,"limiter":bybit_limiter,
     "url": (BYBIT_KLINES_URL + "?category=linear&symbol={symbol}&interval={interval}&limit=200") if BYBIT_KLINES_URL else None,
     "interval_map":{"15m":"15","1h":"60","4h":"240","1d":"D"},
     "parser": lambda d: _parse_bybit(d)},
    {"name":"OKX","weight":8,"limiter":okx_limiter,
     "url": (OKX_KLINES_URL + "?instId={symbol}-SWAP&bar={interval}&limit=200") if OKX_KLINES_URL else None,
     "interval_map":{"15m":"15m","1h":"1H","4h":"4H","1d":"1D"},
     "parser": lambda d: _parse_okx(d)}
]

# Klines exchanges (skip ones without URL)
KLINES_EXCHANGES = [e for e in EXCHANGES if e.get("url")]

# Spot exchanges for GOLD symbols (PAXGUSDT, XAUTUSDT)
SPOT_EXCHANGES = [
    {"name":"Binance Spot","weight":10,"limiter":binance_limiter,
     "url": (BINANCE_SPOT_KLINES_URL + "?symbol={symbol}&interval={interval}&limit=200") if BINANCE_SPOT_KLINES_URL else None,
     "interval_map":{"15m":"15m","1h":"1h","4h":"4h","1d":"1d"},
     "parser": lambda d: d if isinstance(d, list) else None},
    {"name":"Bybit Spot","weight":8,"limiter":bybit_limiter,
     "url": (BYBIT_KLINES_URL + "?category=spot&symbol={symbol}&interval={interval}&limit=200") if BYBIT_KLINES_URL else None,
     "interval_map":{"15m":"15","1h":"60","4h":"240","1d":"D"},
     "parser": lambda d: _parse_bybit(d)},
]
SPOT_EXCHANGES = [e for e in SPOT_EXCHANGES if e.get("url")]

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

def weekly_move_score(klines):
    """Score how likely a coin is to make a big move this week.
    Combines: volatility (ATR%), range squeeze (tight range = explosion coming), volume surge."""
    try:
        H = [float(k[2]) for k in klines[:-1]]
        L = [float(k[3]) for k in klines[:-1]]
        C = [float(k[4]) for k in klines[:-1]]
        V = [float(k[5]) for k in klines[:-1]]
        if len(C) < 25:
            return None
        cc = C[-1]
        # 1) Volatility: ATR% (higher = moves more)
        atr = calc_atr(H, L, C, 14)
        atr_pct = (atr / cc) * 100 if cc > 0 else 0
        # 2) Squeeze: last 7 candles range vs previous 30 (tight range before big move)
        r7 = (max(H[-7:]) - min(L[-7:])) / cc * 100
        r30 = (max(H[-30:]) - min(L[-30:])) / cc * 100 if len(C) >= 30 else r7
        squeeze = 1.0 - min(1.0, r7 / r30) if r30 > 0 else 0   # 1 = very tight = ready to explode
        # 3) Volume surge: last 3 vs average 20
        avg_v = sum(V[-20:]) / 20
        vol_ratio = (sum(V[-3:]) / 3) / avg_v if avg_v > 0 else 1.0
        vol_score = min(1.0, (vol_ratio - 1.0))                 # >1 means volume growing
        # 4) Breakout proximity: price near 30d high/low = about to break
        hi30 = max(H[-30:]) if len(H) >= 30 else max(H)
        lo30 = min(L[-30:]) if len(L) >= 30 else min(L)
        rng = (hi30 - lo30) / cc * 100 if cc > 0 else 0
        breakout = 0.0
        if rng > 0:
            dist_hi = (hi30 - cc) / cc * 100
            dist_lo = (cc - lo30) / cc * 100
            near_edge = min(dist_hi, dist_lo)
            breakout = max(0.0, 1.0 - near_edge / (rng * 0.2))  # near edge of range
        score = (atr_pct / 8.0) * 0.30 + squeeze * 0.30 + max(0, vol_score) * 0.20 + breakout * 0.20
        score = min(1.0, max(0.0, score))
        reasons = []
        if squeeze >= 0.5: reasons.append("Squeeze | فشردگی رنج")
        if atr_pct >= 4: reasons.append("High volatility | نوسان بالا")
        if vol_ratio >= 1.3: reasons.append("Volume surge | جهش حجم")
        if breakout >= 0.6: reasons.append("Near breakout | نزدیک شکست")
        return {"score": score, "atr_pct": atr_pct, "squeeze": squeeze,
                "vol_ratio": vol_ratio, "breakout": breakout, "price": cc, "reasons": reasons}
    except Exception:
        return None

async def fetch_klines(session, symbol, interval):
    # اگر نماد طلا باشد، از Spot exchanges استفاده کن
    if symbol in GOLD_SYMBOLS:
        exchanges_to_try = SPOT_EXCHANGES
        LOGGER.info("Using Spot API for {}".format(symbol))
    else:
        exchanges_to_try = KLINES_EXCHANGES

    for ex in sorted(exchanges_to_try, key=lambda x: x["weight"], reverse=True):
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
# Note: Exchanges with None URLs are skipped automatically (no crash)
def _ob_url(base, suffix):
    """Build order book URL; returns None if base URL is not set in .env"""
    return (base + suffix) if base else None

EXCHANGES = [
    {
        "name": "Binance Futures",
        "limiter": binance_limiter,
        "url": _ob_url(BINANCE_FUTURES_DEPTH_URL, "?symbol={symbol}&limit={limit}"),
        "parser": lambda d: (d.get("bids", []), d.get("asks", []))
    },
    {
        "name": "Binance Spot",
        "limiter": binance_limiter,
        "url": _ob_url(BINANCE_SPOT_DEPTH_URL, "?symbol={symbol}&limit={limit}"),
        "parser": lambda d: (d.get("bids", []), d.get("asks", []))
    },
    {
        "name": "Bybit Linear",
        "limiter": bybit_limiter,
        "url": _ob_url(BYBIT_DEPTH_URL, "?category=linear&symbol={symbol}&limit={limit}"),
        "parser": lambda d: _parse_bybit_ob(d)
    },
    {
        "name": "Bybit Spot",
        "limiter": bybit_limiter,
        "url": _ob_url(BYBIT_DEPTH_URL, "?category=spot&symbol={symbol}&limit={limit}"),
        "parser": lambda d: _parse_bybit_ob(d)
    },
    {
        "name": "OKX",
        "limiter": okx_limiter,
        "url": _ob_url(OKX_DEPTH_URL, "?instId={symbol}-SWAP&sz={limit}"),
        "parser": lambda d: _parse_okx_ob(d)
    },
    {
        "name": "Bitget Futures",
        "limiter": bitget_limiter,
        "url": _ob_url(BITGET_FUTURES_DEPTH_URL, "?symbol={symbol}_UMCBL&limit={limit}&productType=USDT-FUTURES"),
        "parser": lambda d: _parse_bitget_ob(d)
    },
    {
        "name": "Bitget Spot",
        "limiter": bitget_limiter,
        "url": _ob_url(BITGET_SPOT_DEPTH_URL, "?symbol={symbol}&limit={limit}&type=step0"),
        "parser": lambda d: _parse_bitget_spot_ob(d)
    }
]

# Skip exchanges without URL - never crash because of missing .env URLs
EXCHANGES = [e for e in EXCHANGES if e["url"]]
# Futures OB exchanges (crypto) and Spot OB exchanges (gold)
OB_EXCHANGES = [e for e in EXCHANGES if "Spot" not in e["name"]]
SPOT_OB_EXCHANGES = [e for e in EXCHANGES if "Spot" in e["name"]]
LOGGER.info("Order book exchanges active: {}".format(", ".join(e["name"] for e in EXCHANGES)))

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
    # اگر نماد طلا باشد، از Spot OB استفاده کن
    if symbol in GOLD_SYMBOLS:
        ob_to_try = SPOT_OB_EXCHANGES
        LOGGER.info("Using Spot OB for {}".format(symbol))
    else:
        ob_to_try = OB_EXCHANGES

    for ex in ob_to_try:
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
            return {"imbalance": 0.0, "imb5": 0.0, "imb20": 0.0,
                    "bid_wall": {"price": 0.0, "size": 0.0, "mult": 0.0, "dist_pct": 99.0},
                    "ask_wall": {"price": 0.0, "size": 0.0, "mult": 0.0, "dist_pct": 99.0},
                    "spread_pct": 0.0, "slippage": 0.0,
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

        # 📐 Multi-band imbalance: near(5) / mid(10) / far(20) - trap detection
        bv5 = sum(float(b[1]) for b in bids[:5])
        av5 = sum(float(a[1]) for a in asks[:5])
        imb5 = (bv5 - av5) / (bv5 + av5 + 1e-9)
        imb20 = (bv20 - av20) / (bv20 + av20 + 1e-9)

        # 🧱 Wall detection: biggest order within ±2% of mid price
        mid = (bb + ba) / 2 if bb > 0 and ba > 0 else 0
        bid_wall = {"price": 0.0, "size": 0.0, "mult": 0.0, "dist_pct": 99.0}
        ask_wall = {"price": 0.0, "size": 0.0, "mult": 0.0, "dist_pct": 99.0}
        if mid > 0:
            for b in bids[:50]:
                p, s = float(b[0]), float(b[1])
                d = (mid - p) / mid * 100
                if 0 <= d <= 2.0 and avg_bid_size > 0 and (s / avg_bid_size) > bid_wall["mult"]:
                    bid_wall = {"price": p, "size": s, "mult": round(s / avg_bid_size, 1), "dist_pct": round(d, 3)}
            for a in asks[:50]:
                p, s = float(a[0]), float(a[1])
                d = (p - mid) / mid * 100
                if 0 <= d <= 2.0 and avg_ask_size > 0 and (s / avg_ask_size) > ask_wall["mult"]:
                    ask_wall = {"price": p, "size": s, "mult": round(s / avg_ask_size, 1), "dist_pct": round(d, 3)}

        return {
            "imbalance": round(float(imbalance), 4),
            "imb5": round(float(imb5), 4),
            "imb20": round(float(imb20), 4),
            "bid_wall": bid_wall,
            "ask_wall": ask_wall,
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
                model=GEMINI_MODEL,
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


def _parse_event_times(raw):
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(datetime.strptime(part, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc))
        except Exception:
            LOGGER.warning("Bad NEWS_MANUAL_EVENTS entry: " + part)
    return out

def news_blackout_active():
    """Returns (active: bool, reason: str). Checks FOMC calendar + manual events (all UTC)."""
    if not NEWS_BLACKOUT_ENABLED:
        return False, ""
    now = datetime.now(timezone.utc)
    events = _parse_event_times(",".join(FOMC_DATES_2026))
    if NEWS_MANUAL_EVENTS:
        events += _parse_event_times(NEWS_MANUAL_EVENTS)
    for ev in events:
        start = ev - timedelta(minutes=NEWS_BLACKOUT_BEFORE_MIN)
        end = ev + timedelta(minutes=NEWS_BLACKOUT_AFTER_MIN)
        if start <= now <= end:
            return True, ev.strftime("%Y-%m-%d %H:%M UTC")
    return False, ""

async def fetch_news_sentiment(session, symbol):
    """CryptoPanic sentiment for a coin. Returns (bull_ratio, n_votes) or (None, 0)."""
    if not NEWS_SENTIMENT_FILTER or not CRYPTOPANIC_API_KEY:
        return None, 0
    currency = symbol.replace("USDT", "").replace("USD", "")
    try:
        url = "https://cryptopanic.com/api/free/v1/posts/?auth_token={}&currencies={}&filter=rising".format(
            CRYPTOPANIC_API_KEY, currency)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                return None, 0
            data = await r.json()
        bull = bear = 0
        for p in (data.get("results") or [])[:20]:
            votes = p.get("votes") or {}
            bull += int(votes.get("positive", 0) or 0)
            bear += int(votes.get("negative", 0) or 0)
        total = bull + bear
        if total < NEWS_MIN_VOTES:
            return None, 0
        return bull / float(total), total
    except Exception as e:
        LOGGER.error("News sentiment error {}: {}".format(symbol, e))
        return None, 0

def calc_ema(values, period):
    """Exponential Moving Average; returns None if not enough data"""
    if not values or len(values) < period or period < 1:
        return None
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema

def _pivot_points(H, L, order):
    """Find swing highs and lows with indices: returns (ph, pl) lists of (index, price)"""
    ph = []
    pl = []
    for i in range(order, len(H) - order):
        if H[i] == max(H[i-order:i+order+1]):
            ph.append((i, H[i]))
        if L[i] == min(L[i-order:i+order+1]):
            pl.append((i, L[i]))
    return ph, pl

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
    # Strategy 1: RSI+DMI Breakout (Advanced v2)
    # Upgrades: EMA trend + Volume confirm + Dynamic ADX + Anti-Chase
    # (MTF confirm in scanner, DI exit alert after signal)
    # ==========================================================
    rsi_len = 14
    dmi_len = 14
    adx_min = 25
    di_lookback = 5
    rsi_lookback = 20

    # Need enough data
    if len(C) >= max(rsi_len, dmi_len, rsi_lookback, di_lookback) + 1:
        # Calculate RSI for lookback window
        # RSI high/low از ۲۰ کندل «قبلی» (بدون کندل فعلی) تا بریک‌اوت واقعی اندازه شود
        rsi_vals = [calc_rsi(C[:i+1]) for i in range(len(C)-rsi_lookback-1, len(C)-1)]
        if len(rsi_vals) >= rsi_lookback:
            rsi_hi = max(rsi_vals)
            rsi_lo = min(rsi_vals)

            # DI crossover check
            pdi_prev = pdi
            mdi_prev = mdi
            if len(C) > di_lookback + dmi_len * 2:
                pdi_prev, mdi_prev, _ = calc_dmi(H[:-di_lookback], L[:-di_lookback], C[:-di_lookback])

            # ADX ساده: فقط باید از حداقل قدرت ترند بیشتر باشد
            adx_ok = adx >= adx_min

            s1_long = (adx_ok and pdi > mdi and pdi > pdi_prev and rsi > rsi_hi)
            s1_short = (adx_ok and mdi > pdi and mdi > mdi_prev and rsi < rsi_lo)
        else:
            s1_long = s1_short = False
    else:
        s1_long = s1_short = False

    # ==========================================================
    # Strategy 2: Candle Setup (configurable + S/R proximity filter)
    # ==========================================================
    shadow_big = S2_SHADOW_BIG
    shadow_small = S2_SHADOW_SMALL
    s2_vol_ok = (avg_v20 > 0 and cv >= S2_VOLUME_RATIO * avg_v20)

    # S/R proximity: LONG only near support, SHORT only near resistance
    def _near_level(price, levels):
        if not S2_SR_FILTER or not levels:
            return True  # filter off or no levels found -> allow
        return any(abs(price - lvl) / lvl * 100 <= S2_SR_PROXIMITY_PCT for lvl in levels if lvl > 0)

    s2_near_support = _near_level(cl, htf_s)
    s2_near_resistance = _near_level(ch, htf_r)

    if body > 0:
        lower_shadow = lw
        upper_shadow = uw

        s2_long = (lower_shadow >= shadow_big * body and
                   upper_shadow <= shadow_small * body and
                   cl < sma7 < bb and s2_vol_ok and s2_near_support)
        s2_short = (upper_shadow >= shadow_big * body and
                    lower_shadow <= shadow_small * body and
                    bt < sma7 < ch and s2_vol_ok and s2_near_resistance)
    else:
        s2_long = s2_short = False

    # ==========================================================
    # Strategy 3: HH/LL Breakout with Volume (Swing High/Low)
    # ==========================================================
    swing_order = 5
    vol_period = 20

    if len(C) >= swing_order * 2 + 1 and len(V) >= vol_period:
        hh = _swing_high(H, swing_order)
        ll = _swing_low(L, swing_order)
        avg_vol = sum(V[-vol_period:]) / vol_period
        vol_ok = cv >= S3_VOLUME_RATIO * avg_vol

        s3_long = (hh is not None and cc > hh and vol_ok)
        s3_short = (ll is not None and cc < ll and vol_ok)

        # --- Upgrade: SMA7 direction filter (breakout only with short-term trend) ---
        # LONG only above SMA7, SHORT only below SMA7 - reduces fakeouts
        if S3_SMA7_FILTER and (s3_long or s3_short):
            if s3_long and cc < sma7:
                s3_long = False
            if s3_short and cc > sma7:
                s3_short = False
    else:
        s3_long = s3_short = False

    # ==========================================================
    # Voting System (votes_needed=1 - any strategy triggers)
    # ==========================================================
    longs, shorts = [], []

    if s1_long: longs.append("RSI+DMI Breakout")
    if s2_long: longs.append("Candle Setup")
    if s3_long: longs.append("HH/LL Breakout")

    if s1_short: shorts.append("RSI+DMI Breakout")
    if s2_short: shorts.append("Candle Setup")
    if s3_short: shorts.append("HH/LL Breakout")

    # ==========================================================
    # Global trend filter: LONG only above EMA / SHORT only below EMA
    # فیلتر ترند سراسری — جلوی سیگنال خلاف ترند را می‌گیرد
    # ==========================================================
    if TREND_FILTER_ENABLED and len(C) >= TREND_FILTER_EMA + 5:
        ema_tf = calc_ema(C, TREND_FILTER_EMA)
        if cc < ema_tf and longs:
            longs = []   # قیمت زیر EMA است → LONG ممنوع
        if cc > ema_tf and shorts:
            shorts = []  # قیمت بالای EMA است → SHORT ممنوع

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
            "🚨 *NEW TRADING SIGNAL*\n"
            "🪙 *Symbol |* `#{symbol}`        🤖 *AI |* {conf} `{ai}`\n"
            "📊 *Direction |* {direction} {emoji}\n"
            "🎯 *Strategy |* {strategy}\n"
            "⏱️ *Timeframe |* {interval}\n"
            "💵 *Entry Price |* `{entry}`\n"
            "📖 *Order Book |* Imb `{imb}` | Depth `{bid:,.0f}`/`{ask:,.0f}` | {src}"
        ).format(
            symbol=symbol,
            direction=signal['direction'], emoji=dir_emoji,
            strategy=signal['strategy'],
            interval=interval,
            entry=str(signal['entry_price']),
            conf=conf_emoji, ai="{:.0%}".format(ai_prob),
            imb="{:.2f}".format(ob_data['imbalance']),
            bid=ob_data['bid_depth'], ask=ob_data['ask_depth'],
            src=str(ob_data.get('source', 'unknown'))
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


    async def command_listener(self):
        last_id = 0
        while True:
            try:
                updates = await self.bot.get_updates(offset=last_id + 1, timeout=5, allowed_updates=["message", "callback_query"])
                for u in updates:
                    last_id = u.update_id

                    # Authorization check
                    user_id = None
                    if u.message and u.message.from_user:
                        user_id = u.message.from_user.id
                    elif u.callback_query and u.callback_query.from_user:
                        user_id = u.callback_query.from_user.id

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

                    if u.message and u.message.text:
                        cmd = u.message.text.strip().split("@")[0].lower()
                        parts = cmd.split()

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

                                # Update signal_history feedback
                                await db_execute("UPDATE signal_history SET feedback = ? WHERE alert_id = ?", (fb_type, alert_id))

                                # Update trade_features outcome
                                await db_execute(
                                    "UPDATE trade_features SET outcome = ? WHERE id = (SELECT id FROM signal_history WHERE alert_id = ?)",
                                    (outcome, alert_id)
                                )

                                # Update stats
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'feedback_{}'".format(fb_type))

                                await self.send(
                                    "✅ *Result recorded | نتیجه ثبت شد*\n\n"
                                    "Alert ID: `{}`\n"
                                    "Result: `{}`\n\n"
                                    "AI will learn from this feedback. | AI از این بازخورد یاد می‌گیره.".format(alert_id, result.upper()),
                                    chat_id=cid
                                )

                                # Trigger AI retrain
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

                    # Photo/image analysis removed per user request - images are ignored
                    if u.message and (u.message.photo or u.message.document):
                        LOGGER.info("Image received - image analysis is disabled, ignoring.")

                    if u.callback_query:
                        cq = u.callback_query
                        data = cq.data or ""
                        parts = data.split("|")
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

                                # Trigger AI retrain
                                asyncio.create_task(self.ai_engine.retrain())

                                # Answer callback immediately to remove loading indicator
                                try:
                                    await self.bot.answer_callback_query(cq.id)
                                except Exception as e:
                                    LOGGER.warning("Could not answer callback: {}".format(e))
            except Exception as e:
                LOGGER.error("Command listener: " + str(e))
            await asyncio.sleep(2)

# ==========================================================
# 10. Main Signal Bot
# ==========================================================
class SignalBot:
    def __init__(self):
        self.ai = AIEngine()
        self.chart_analyzer = ChartImageAnalyzer(GEMINI_API_KEY)
        self.tg = TelegramManager(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, self.chart_analyzer, self.ai, ALLOWED_USER_IDS)
        self.btc_trend = "NEUTRAL"
        self.btc_pause_until = 0
        self.symbol_cache = {"symbols": [], "last_update": 0, "volumes": {}}
        self.check_interval = CHECK_INTERVAL_SECONDS
        # Signal filtering counters
        self._daily_signals_count = 0
        self._last_signal_date = ""
        self._sent_signals_today = set()

    async def start(self):
        await init_database()
        asyncio.create_task(self.tg.command_listener())
        LOGGER.info("Signal Bot ready. Mode: CRYPTO + PAXG GOLD ONLY + AI Learning from Feedback")

    async def get_symbols(self, session):
        now = time.time()
        if now - self.symbol_cache["last_update"] > 300 or not self.symbol_cache["symbols"]:
            syms = []
            vols = {}
            
            # === ۱. گرفتن کریپتوها از Binance Futures ===
            try:
                url = BINANCE_FUTURES_TICKER_URL
                async with session.get(url, timeout=15) as r:
                    if r.status == 200:
                        data = await r.json()
                        btc_p = next((float(x["lastPrice"]) for x in data if x.get("symbol") == "BTCUSDT"), 60000)
                        min_vol_usdt = MIN_BTC_VOLUME * btc_p

                        for x in data:
                            sym = x["symbol"]
                            # فقط کریپتوهای USDT
                            if not sym.endswith("USDT"):
                                continue

                            qv = float(x.get("quoteVolume", 0))
                            if qv >= min_vol_usdt:
                                syms.append(sym)
                                vols[sym] = qv
                        LOGGER.info("{} crypto symbols loaded from Binance (24h vol > {} BTC = {:,.0f} USDT).".format(
                            len(syms), MIN_BTC_VOLUME, min_vol_usdt))
                    else:
                        LOGGER.error("Failed to fetch Binance 24hr ticker: HTTP {}".format(r.status))
            except Exception as e:
                LOGGER.error("Binance symbol fetch error: " + str(e))
            
            # === ۲. گرفتن PAX Gold از CoinMarketCap ===
            if COINMARKETCAP_API_KEY:
                try:
                    cmc_url = COINMARKETCAP_URL
                    headers = {
                        "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY,
                        "Accept": "application/json"
                    }
                    params = {
                        "symbol": "PAXG",
                        "convert": "USD"
                    }
                    async with session.get(cmc_url, headers=headers, params=params, timeout=15) as r:
                        if r.status == 200:
                            data = await r.json()
                            paxg_data = data.get("data", {}).get("PAXG", {})
                            quote = paxg_data.get("quote", {}).get("USD", {})
                            price = quote.get("price", 0)
                            volume_24h = quote.get("volume_24h", 0)
                            
                            if price > 0:
                                paxg_symbol = "PAXGUSDT"
                                if paxg_symbol not in syms:
                                    syms.append(paxg_symbol)
                                    vols[paxg_symbol] = volume_24h
                                    LOGGER.info("PAX Gold added from CMC: price=${:,.2f}, vol=${:,.0f}".format(price, volume_24h))
                        else:
                            LOGGER.warning("CMC API returned HTTP {}".format(r.status))
                except Exception as e:
                    LOGGER.error("CoinMarketCap PAXG fetch error: " + str(e))
            else:
                LOGGER.warning("COINMARKETCAP_API_KEY not set, skipping PAXG")
            
            self.symbol_cache = {"symbols": syms, "last_update": now, "volumes": vols}
            LOGGER.info("Total symbols loaded: {} (Crypto + PAXG Gold)".format(len(syms)))
            if syms:
                LOGGER.info("Symbols: {}".format(", ".join(syms[:10]) + ("..." if len(syms) > 10 else "")))
        
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
        
        # Deduplication: cooldown per symbol+direction (from .env, default 240 min)
        now = time.time()
        last_sent = getattr(self, "_last_signal_time", {})
        dedup_key = "{}_{}".format(symbol, signal["direction"])
        cooldown_sec = SIGNAL_COOLDOWN_MINUTES * 60
        if dedup_key in last_sent and (now - last_sent[dedup_key]) < cooldown_sec:
            LOGGER.info("Skipping {} {}: signal sent {}min ago (cooldown {}min)".format(
                symbol, signal["direction"], int((now - last_sent[dedup_key]) / 60), SIGNAL_COOLDOWN_MINUTES))
            return

        # Persistent dedup via DB - survives restarts and duplicate deploys
        try:
            _dd = await db_execute(
                "SELECT (julianday('now') - julianday(timestamp)) * 1440.0 FROM signal_history WHERE symbol = ? AND direction = ? ORDER BY id DESC LIMIT 1",
                (symbol, signal["direction"]))
            if _dd and _dd[0][0] is not None and _dd[0][0] < SIGNAL_COOLDOWN_MINUTES:
                LOGGER.info("Skipping {} {}: DB signal sent {:.0f}min ago (cooldown {}min)".format(
                    symbol, signal["direction"], _dd[0][0], SIGNAL_COOLDOWN_MINUTES))
                return
        except Exception as _e:
            LOGGER.error("Dedup DB check error: " + str(_e))

        # ⛔ News blackout: no signals around FOMC / manual major events
        _blk, _blk_ev = news_blackout_active()
        if _blk:
            LOGGER.info("Skipping {} {}: news blackout active (event {})".format(
                symbol, signal["direction"], _blk_ev))
            return

        # 📰 News sentiment (CryptoPanic) - cached per symbol, used by block filter AND AI boost
        _ratio, _votes = None, 0
        if (NEWS_SENTIMENT_FILTER or NEWS_AI_SCORE) and CRYPTOPANIC_API_KEY:
            try:
                if not hasattr(self, "_news_cache"):
                    self._news_cache = {}
                _nc = self._news_cache.get(symbol)
                if _nc and (now - _nc[0]) < NEWS_CACHE_MINUTES * 60:
                    _ratio, _votes = _nc[1], _nc[2]
                else:
                    _ratio, _votes = await fetch_news_sentiment(session, symbol)
                    self._news_cache[symbol] = (now, _ratio, _votes)
                if _ratio is not None and NEWS_SENTIMENT_FILTER:
                    if signal["direction"] == "LONG" and (1.0 - _ratio) >= NEWS_BLOCK_RATIO:
                        LOGGER.info("Skipping {} LONG: bearish news {:.0%} ({} votes)".format(symbol, 1.0 - _ratio, _votes))
                        return
                    if signal["direction"] == "SHORT" and _ratio >= NEWS_BLOCK_RATIO:
                        LOGGER.info("Skipping {} SHORT: bullish news {:.0%} ({} votes)".format(symbol, _ratio, _votes))
                        return
            except Exception as _ne:
                LOGGER.error("News filter error: " + str(_ne))
        
        # Check daily signal limit
        if MAX_DAILY_SIGNALS > 0 and self._daily_signals_count >= MAX_DAILY_SIGNALS:
            LOGGER.info("Daily signal limit reached ({}/{}). Skipping {}.".format(
                self._daily_signals_count, MAX_DAILY_SIGNALS, symbol))
            return
        bids, asks, ob_source = await fetch_order_book(session, symbol)
        ob = OrderBookAnalyzer.analyze(bids, asks, signal["entry_price"], signal["stop_loss"])
        ob["source"] = ob_source or "failed"

        # ============ ADVANCED OB FILTERS | فیلترهای پیشرفته سفارشات ============
        _dir_long = signal["direction"] == "LONG"

        # 🎭 Feature 4: Spoof detection - walls that vanish between scans
        if not hasattr(self, "_ob_walls_hist"):
            self._ob_walls_hist = {}
        if not hasattr(self, "_ob_spoof_until"):
            self._ob_spoof_until = {}   # symbol_side -> time until which walls are distrusted
        _hist = self._ob_walls_hist.get(symbol)
        _now_t = time.time()
        if OB_SPOOF_FILTER and _hist:
            for _side, _wall_key, _cur in (("bid", "bid_wall", ob["bid_wall"]), ("ask", "ask_wall", ob["ask_wall"])):
                _prev = _hist.get(_wall_key, {})
                # previous big wall at similar price disappeared now => spoof
                if _prev.get("mult", 0) >= OB_WALL_MIN_MULT and _cur.get("mult", 0) < OB_WALL_MIN_MULT \
                        and abs(_prev.get("price", 0) - _cur.get("price", 0)) / max(_prev.get("price", 1), 1) < 0.005:
                    self._ob_spoof_until["{}_{}".format(symbol, _side)] = _now_t + 1800  # 30 min distrust
                    LOGGER.warning("SPOOF detected {}: {} wall x{:.0f} vanished - distrusting {} walls 30min".format(
                        symbol, _side, _prev["mult"], _side))
        self._ob_walls_hist[symbol] = {"bid_wall": ob["bid_wall"], "ask_wall": ob["ask_wall"], "ts": _now_t}

        # 🧱 Feature 1: Wall filter - big opposing wall nearby blocks the signal
        if OB_WALL_FILTER:
            _blocking = ob["ask_wall"] if _dir_long else ob["bid_wall"]
            _side = "ask" if _dir_long else "bid"
            _spoofed = _now_t < self._ob_spoof_until.get("{}_{}".format(symbol, _side), 0)
            if _blocking["mult"] >= OB_WALL_MIN_MULT and _blocking["dist_pct"] <= OB_WALL_BLOCK_PCT:
                if _spoofed:
                    LOGGER.info("Wall block SKIPPED {} ({} wall recently spoofed - likely fake)".format(symbol, _side))
                else:
                    LOGGER.warning("WALL BLOCK {} {}: {} wall x{:.0f} at {:.3f}% away".format(
                        symbol, signal["direction"], _side, _blocking["mult"], _blocking["dist_pct"]))
                    return

        # 📐 Feature 2: Band trap - near layers oppose far layers (fake pressure)
        if OB_BAND_TRAP_FILTER:
            _near, _far = ob.get("imb5", 0), ob.get("imb20", 0)
            if (_dir_long and _near < -0.10 and _far > 0.10) or (not _dir_long and _near > 0.10 and _far < -0.10):
                LOGGER.warning("BAND TRAP {} {}: near={:.2f} vs far={:.2f} - fake pressure, blocked".format(
                    symbol, signal["direction"], _near, _far))
                return

        # 📈 Feature 3: OB momentum - sustained opposing pressure blocks signal
        if not hasattr(self, "_ob_imb_hist"):
            self._ob_imb_hist = {}
        _imbs = self._ob_imb_hist.setdefault(symbol, [])
        _imbs.append(ob["imbalance"])
        self._ob_imb_hist[symbol] = _imbs[-10:]
        if OB_MOMENTUM_FILTER and len(_imbs) >= OB_MOMENTUM_SAMPLES:
            _avg_imb = sum(_imbs[-OB_MOMENTUM_SAMPLES:]) / OB_MOMENTUM_SAMPLES
            if (_dir_long and _avg_imb < -OB_MOMENTUM_MIN) or (not _dir_long and _avg_imb > OB_MOMENTUM_MIN):
                LOGGER.warning("OB MOMENTUM BLOCK {} {}: sustained opposing pressure avg={:.2f}".format(
                    symbol, signal["direction"], _avg_imb))
                return

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
        
        # Filter: allowed strategies only
        if ALLOWED_STRATEGIES:
            signal_strategies = signal.get("strategy", "").split(" + ")
            if not any(s in ALLOWED_STRATEGIES for s in signal_strategies):
                LOGGER.info("Strategy filter rejected {}: {} not in allowed list".format(
                    symbol, signal.get("strategy", "")))
                return

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

        # 📊 Feature 3: News sentiment boost/penalty on AI score
        news_adjustment = 0.0
        if NEWS_AI_SCORE and _ratio is not None:
            if signal["direction"] == "LONG":
                if _ratio >= NEWS_BLOCK_RATIO:
                    news_adjustment = NEWS_AI_BOOST          # خبرها مثبت → تقویت لانگ
                elif (1.0 - _ratio) >= NEWS_BLOCK_RATIO:
                    news_adjustment = -NEWS_AI_BOOST         # خبرها منفی → جریمه لانگ
            else:
                if (1.0 - _ratio) >= NEWS_BLOCK_RATIO:
                    news_adjustment = NEWS_AI_BOOST          # خبرها منفی → تقویت شارت
                elif _ratio >= NEWS_BLOCK_RATIO:
                    news_adjustment = -NEWS_AI_BOOST         # خبرها مثبت → جریمه شارت
            if news_adjustment != 0.0:
                LOGGER.info("News AI adjustment {} {}: {:+.1%} (bull ratio {:.0%}, {} votes)".format(
                    symbol, signal["direction"], news_adjustment, _ratio, _votes))

        prob = min(1.0, max(0.0, base_prob + mtf_adjustment + news_adjustment))
        conf_label = self.ai.confidence_label(prob)
        
        # Filter: minimum AI confidence
        # AI filter only applies when the model is actually trained
        # (untrained model always returns 0.50 and would block everything)
        if prob < MIN_AI_CONFIDENCE and (self.ai.is_trained or AI_FILTER_UNTRAINED):
            LOGGER.info("AI confidence too low for {}: {:.1%} < {:.1%} (min required{})".format(
                symbol, prob, MIN_AI_CONFIDENCE, "" if self.ai.is_trained else " — untrained AI"))
            return
        if not self.ai.is_trained and not AI_FILTER_UNTRAINED:
            LOGGER.info("AI not trained yet ({}). Skipping AI confidence filter for {}.".format(
                "need {} feedbacks".format(self.ai.min_samples), symbol))
        
        # Filter: RSI range check
        if signal["direction"] == "LONG" and MIN_RSI_LONG > 0:
            if signal["rsi"] < MIN_RSI_LONG:
                LOGGER.info("RSI filter rejected LONG {}: RSI {:.1f} < {:.1f}".format(
                    symbol, signal["rsi"], MIN_RSI_LONG))
                return
        if signal["direction"] == "SHORT" and MAX_RSI_SHORT > 0:
            if signal["rsi"] > MAX_RSI_SHORT:
                LOGGER.info("RSI filter rejected SHORT {}: RSI {:.1f} > {:.1f}".format(
                    symbol, signal["rsi"], MAX_RSI_SHORT))
                return
        
        # Filter: minimum ADX
        if MIN_ADX > 0 and signal["adx"] < MIN_ADX:
            LOGGER.info("ADX filter rejected {}: ADX {:.1f} < {:.1f}".format(
                symbol, signal["adx"], MIN_ADX))
            return
        

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
        
        # Increment daily counter
        self._daily_signals_count += 1
        self._sent_signals_today.add(symbol)
        LOGGER.info("Daily signals: {}/{}".format(self._daily_signals_count, MAX_DAILY_SIGNALS if MAX_DAILY_SIGNALS > 0 else "unlimited"))

        LOGGER.info("SIGNAL SENT: {} | Strategy: {} | AI: {:.1%} (base: {:.1%}, mtf: {:+.1%}) | 4H: {} | 1H: {} | OB: {:.2f} | Quality: {:.2f} | Features recorded.".format(symbol, signal["strategy"], prob, base_prob if "base_prob" in dir() else prob, mtf_adjustment if "mtf_adjustment" in dir() else 0, h4_trend, h1_trend, ob_conf, ob_quality_score))
        
        # Record signal time for deduplication
        if not hasattr(self, "_last_signal_time"):
            self._last_signal_time = {}
        self._last_signal_time["{}_{}".format(symbol, signal["direction"])] = time.time()

        await self.tg.notify_signal(signal, symbol, interval, prob, conf_label, ob, self.btc_trend, alert_id, gemini_result, ob_conf, ob_quality_reason)

    async def scanner_loop(self):
        async with aiohttp.ClientSession() as session:
            await self.update_btc(session)
            symbols = await self.get_symbols(session)
            btc_counter = 0

            while True:
                try:
                    btc_counter += 1
                    
                    # Reset daily signal counter
                    today = time.strftime("%Y-%m-%d")
                    if today != self._last_signal_date:
                        self._daily_signals_count = 0
                        self._sent_signals_today = set()
                        self._last_signal_date = today
                        LOGGER.info("New day - signal counter reset to 0")
                    if btc_counter >= 15:
                        await self.update_btc(session)
                        btc_counter = 0

                    if time.time() < self.btc_pause_until:
                        await asyncio.sleep(5)
                        continue

                    # 🚀 Two-tier smart scanner: top-volume coins every cycle, rest every SCAN_SLOW_EVERY cycles
                    self._scan_cycle = getattr(self, "_scan_cycle", 0) + 1
                    _vols_map = self.symbol_cache.get("volumes", {})
                    _sorted_syms = sorted(symbols, key=lambda s: _vols_map.get(s, 0), reverse=True)
                    _fast_set = set(_sorted_syms[:SCAN_FAST_TOP_N])
                    _slow_cycle = (self._scan_cycle % SCAN_SLOW_EVERY == 0)
                    if _slow_cycle:
                        LOGGER.info("Scan cycle {}: FULL scan (fast {} + slow {})".format(
                            self._scan_cycle, len(_fast_set), len(symbols) - len(_fast_set)))

                    for symbol in symbols:
                        # دوباره چک کن که فقط کریپتو و طلا باشد
                        if not (symbol.endswith("USDT") or symbol.endswith("BUSD") or symbol in GOLD_SYMBOLS):
                            LOGGER.debug("Skipping non-crypto/gold: {}".format(symbol))
                            continue

                        # Slow-tier coins: only scanned on slow cycles
                        if symbol not in _fast_set and not _slow_cycle:
                            continue
                        
                        vol = self.symbol_cache.get("volumes", {}).get(symbol, 0)
                        if vol <= 0:
                            LOGGER.debug("Skipping {}: no volume data".format(symbol))
                            continue

                        k4h = await fetch_klines(session, symbol, "4h")
                        k1d = await fetch_klines(session, symbol, "1d")
                        htf_s, htf_r = htf_sr(k4h, k1d)

                        # Filter: allowed timeframes only
                        for interval in ALLOWED_TIMEFRAMES:
                            klines = await fetch_klines(session, symbol, interval)
                            if not klines:
                                continue

                            sig = analyze_signal(klines, symbol, interval, htf_s, htf_r, MAX_SL_PERCENT)
                            if not sig:
                                continue

                            # Filter: BTC trend alignment (configurable)
                            if FILTER_BTC_TREND and symbol != "BTCUSDT":
                                if sig["direction"] == "LONG" and self.btc_trend == "BEARISH":
                                    continue
                                if sig["direction"] == "SHORT" and self.btc_trend == "BULLISH":
                                    continue

                            # Use rounded timestamp to prevent duplicate signals within 5 min window
                            ts_rounded = (int(klines[-2][0]) // 300000) * 300000
                            alert_id = "{}_{}_{}_{}".format(symbol, interval, ts_rounded, sig['direction'])
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

                    # ⚠️ Feature 4: Telegram warning before major news events
                    if NEWS_ALERT_ENABLED:
                        try:
                            _now_utc = datetime.now(timezone.utc)
                            if not hasattr(self, "_news_alerted"):
                                self._news_alerted = set()
                            _events = _parse_event_times(",".join(FOMC_DATES_2026))
                            if NEWS_MANUAL_EVENTS:
                                _events += _parse_event_times(NEWS_MANUAL_EVENTS)
                            for _ev in _events:
                                _mins = (_ev - _now_utc).total_seconds() / 60.0
                                _key = _ev.strftime("%Y%m%d%H%M")
                                if 0 < _mins <= NEWS_ALERT_BEFORE_MIN and _key not in self._news_alerted:
                                    self._news_alerted.add(_key)
                                    await self.tg.send(
                                        "⚠️ *News Alert | هشدار خبر مهم*\n\n"
                                        "Major event in ~{:.0f} min | رویداد مهم در ~{:.0f} دقیقه دیگر\n"
                                        "Time | زمان: `{}`\n\n"
                                        "ربات در پنجره خبر ساکت می‌شه — مراقب پوزیشن‌های باز باش!".format(
                                            _mins, _mins, _ev.strftime("%Y-%m-%d %H:%M UTC")))
                                    LOGGER.info("News alert sent for event {}".format(_key))
                        except Exception as _ae:
                            LOGGER.error("News alert error: " + str(_ae))

                    # 📊 Weekly Move Report: coins likely to move this week
                    if WEEKLY_REPORT_ENABLED:
                        try:
                            _nw = datetime.now(timezone.utc)
                            _wk = _nw.isocalendar()[:2]  # (year, week) - send once per week
                            if _nw.weekday() == WEEKLY_REPORT_DAY and _nw.hour == WEEKLY_REPORT_HOUR \
                                    and getattr(self, "_weekly_sent", None) != _wk:
                                self._weekly_sent = _wk
                                LOGGER.info("Building weekly report for {} symbols...".format(len(symbols)))
                                _ranked, _new, _stats = [], [], []
                                for _sym in symbols:
                                    try:
                                        _dk = await fetch_klines(session, _sym, "1d")
                                        if not _dk:
                                            continue
                                        _sc = weekly_move_score(_dk)
                                        if _sc:
                                            _ranked.append((_sym, _sc))
                                        # weekly stats: 7d change + volume inflow vs previous week
                                        _C = [float(k[4]) for k in _dk[:-1]]
                                        _V = [float(k[5]) for k in _dk[:-1]]
                                        if len(_C) >= 15:
                                            _ch7 = (_C[-1] / _C[-8] - 1) * 100
                                            _vw = sum(_V[-7:]) / 7
                                            _vp = sum(_V[-14:-7]) / 7
                                            _vflow = _vw / _vp if _vp > 0 else 1.0
                                            _stats.append((_sym, _ch7, _vflow))
                                        if 3 <= len(_C) < 45:
                                            _new.append((_sym, len(_C), (_C[-1] / _C[0] - 1) * 100))
                                    except Exception:
                                        pass
                                    await asyncio.sleep(0.3)  # gentle on API rate limits

                                _ranked.sort(key=lambda x: x[1]["score"], reverse=True)
                                _lines = ["📊 *Weekly Report | گزارش هفتگی*\n"]

                                # 🔥 بخش ۱: کوین‌های آماده حرکت
                                _top = _ranked[:WEEKLY_TOP_N]
                                if _top:
                                    _lines.append("*Ready to move | آماده حرکت:*\n")
                                    for _sym, _sc in _top:
                                        _bar = "🔥" if _sc["score"] >= 0.6 else ("⚡" if _sc["score"] >= 0.4 else "📈")
                                        _rs = " + ".join(_sc["reasons"]) if _sc["reasons"] else "—"
                                        _lines.append(
                                            "{} `{}` — {:.0%} | ATR {:.1f}% | Vol x{:.1f}\n"
                                            "   {}\n".format(_bar, _sym, _sc["score"], _sc["atr_pct"], _sc["vol_ratio"], _rs))

                                # 🆕 بخش ۲: کوین‌های جدید
                                if _new:
                                    _lines.append("🆕 *New coins | کوین‌های جدید:*\n")
                                    for _sym, _age, _ch in sorted(_new, key=lambda x: -abs(x[2]))[:5]:
                                        _lines.append("`{}` — {} days old | {} روزه | {:+.1f}%\n".format(_sym, _age, _age, _ch))

                                # 💰 بخش ۳: ورود حجم = نشانه حرکت آینده (آینده‌نگر)
                                if _stats:
                                    _vf = sorted(_stats, key=lambda x: -x[2])[:5]
                                    _vf = [x for x in _vf if x[2] >= 1.3]
                                    if _vf:
                                        _lines.append("💰 *Volume inflow - move coming | ورود حجم، حرکت در راه:*")
                                        for _sym, _ch, _v in _vf:
                                            _lines.append("`{}` Vol x{:.1f} | حجم {:+.0f}% رشد کرده".format(_sym, _v, (_v - 1) * 100))

                                # 📰 بخش ۶: خبرهای داغ (اگر CryptoPanic فعال باشه)
                                if CRYPTOPANIC_API_KEY and _top:
                                    try:
                                        _news_lines = []
                                        for _sym, _ in _top[:3]:
                                            _cr = _sym.replace("USDT", "")
                                            _url = "https://cryptopanic.com/api/free/v1/posts/?auth_token={}&currencies={}&filter=hot".format(
                                                CRYPTOPANIC_API_KEY, _cr)
                                            async with session.get(_url, timeout=aiohttp.ClientTimeout(total=8)) as _r:
                                                if _r.status == 200:
                                                    _res = (await _r.json()).get("results") or []
                                                    if _res:
                                                        _news_lines.append("`{}`: {}".format(_sym, (_res[0].get("title") or "")[:90]))
                                    except Exception as _ne2:
                                        LOGGER.error("Weekly news error: " + str(_ne2))
                                    if _news_lines:
                                        _lines.append("\n📰 *Hot news | خبرهای داغ:*")
                                        _lines += _news_lines

                                if len(_lines) > 2:
                                    await self.tg.send("\n".join(_lines))
                                    LOGGER.info("Weekly report sent: {} movers, {} new, {} stats".format(len(_top), len(_new), len(_stats)))
                        except Exception as _we:
                            LOGGER.error("Weekly report error: " + str(_we))

                    await asyncio.sleep(self.check_interval)
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
