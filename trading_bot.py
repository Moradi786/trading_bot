import asyncio
import logging
import os
import sqlite3
import time
import aiohttp
from aiohttp import web
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
import libsql_client
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple
import traceback

# =========================================================
# ۱. تنظیمات اولیه و متغیرهای محیطی
# =========================================================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PORT = int(os.getenv("PORT", 8080))

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

if not TELEGRAM_BOT_TOKEN:
    LOGGER.error("❌ TELEGRAM_BOT_TOKEN تنظیم نشده!")
    raise SystemExit(1)

if not TELEGRAM_CHAT_ID:
    LOGGER.error("❌ TELEGRAM_CHAT_ID تنظیم نشده!")
    raise SystemExit(1)

try:
    TELEGRAM_CHAT_ID = int(TELEGRAM_CHAT_ID)
except ValueError:
    pass

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
MAX_SL_PERCENT = 5.0
MIN_BTC_VOLUME = 250.0
MAX_SIGNAL_AGE_SECONDS = 180
SLIPPAGE_WARNING_THRESHOLD = 0.3
CONCURRENT_SCAN_LIMIT = 10
VOLATILITY_PAUSE_MINUTES = 15
VOLATILITY_THRESHOLD_PERCENT = 2.5
ALERT_TTL = 86400
DB_NAME = "trading_ai_dataset.db"

active_trades = {}
active_trades_lock = asyncio.Lock()
GLOBAL_BTC_TREND = "NEUTRAL"
BTC_VOLATILITY_PAUSE_UNTIL = 0
_symbol_cache = {"symbols": [], "last_update": 0}

# =========================================================
# ۲. LRU Alert Cache (مشکل ۱۰ - FIXED)
# =========================================================
class LRUAlertCache:
    def __init__(self, max_size=5000, ttl=86400):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = asyncio.Lock()

    async def add(self, key):
        async with self.lock:
            now = time.time()
            expired = [k for k, v in self.cache.items() if now - v > self.ttl]
            for k in expired:
                del self.cache[k]

            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = now
            return True

    async def exists(self, key):
        async with self.lock:
            if key in self.cache:
                if time.time() - self.cache[key] <= self.ttl:
                    return True
                del self.cache[key]
            return False

alert_cache = LRUAlertCache(max_size=5000, ttl=86400)

# =========================================================
# ۳. مدیریت دیتابیس ابری Turso + SQLite Fallback
#    (مشکل ۹ - FIXED: context manager برای SQLite)
# =========================================================
def get_turso_client():
    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        try:
            url = TURSO_DATABASE_URL.strip()
            if url.startswith("https://"):
                url = url.replace("https://", "libsql://")
            return libsql_client.create_client_sync(url=url, auth_token=TURSO_AUTH_TOKEN.strip())
        except Exception as e:
            LOGGER.error(f"❌ Error connecting to Turso DB: {e}")
    return None

def execute_db_query(query, params=()):
    clean_params = []
    for p in params:
        if isinstance(p, (float, int, str)) or p is None:
            clean_params.append(p)
        else:
            clean_params.append(float(p))

    client = get_turso_client()
    if client:
        try:
            client.execute(query, clean_params)
            return True
        except Exception as e:
            LOGGER.error(f"❌ Turso Query Error: {e}")
        finally:
            try:
                client.close()
            except Exception:
                pass

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(query, clean_params)
            conn.commit()
            return True
    except Exception as e:
        LOGGER.error(f"❌ SQLite Query Error: {e}")
        return False

def fetch_db_df(query):
    client = get_turso_client()
    if client:
        try:
            res = client.execute(query)
            cols = res.columns
            rows = [list(r) for r in res.rows]
            client.close()
            return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            LOGGER.error(f"❌ Turso Fetch Error: {e}")
            try:
                client.close()
            except Exception:
                pass

    try:
        with sqlite3.connect(DB_NAME) as conn:
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        LOGGER.error(f"❌ SQLite Fetch Error: {e}")
        return pd.DataFrame()

def init_db():
    create_table_sql = '''
        CREATE TABLE IF NOT EXISTS trade_features (
            id TEXT PRIMARY KEY,
            symbol TEXT,
            direction TEXT,
            rsi REAL,
            spread_pct REAL,
            vol_ratio REAL,
            lower_wick_ratio REAL,
            upper_wick_ratio REAL,
            trend_code INTEGER,
            adx REAL,
            plus_di REAL,
            minus_di REAL,
            price_to_sma7_ratio REAL,
            atr_pct REAL,
            outcome INTEGER
        )
    '''
    if execute_db_query(create_table_sql):
        LOGGER.info("☁️ Cloud Database Initialized with Advanced Features.")

# =========================================================
# ۴. سیستم یادگیری هوش مصنوعی پیشرفته
#    (مشکل ۱ - FIXED: async retrain)
# =========================================================
class AdvancedSelfLearningAIEngine:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.2,
            eval_metric='logloss',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_samples_to_train = 8
        self.feature_columns = [
            'rsi', 'spread_pct', 'vol_ratio', 'lower_wick_ratio',
            'upper_wick_ratio', 'trend_code', 'adx', 'plus_di',
            'minus_di', 'price_to_sma7_ratio', 'atr_pct'
        ]
        self._training = False

    def retrain_model(self):
        if self._training:
            LOGGER.info("🧠 AI Learning: Training already in progress, skipping.")
            return False

        self._training = True
        try:
            df = fetch_db_df("SELECT * FROM trade_features WHERE outcome IS NOT NULL")
            if df.empty or len(df) < self.min_samples_to_train:
                LOGGER.info(f"🧠 AI Learning: Need at least {self.min_samples_to_train} closed trades. Currently in DB: {len(df)}")
                return False

            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0

            X = df[self.feature_columns].astype(float)
            y = df['outcome'].astype(int)

            if len(y.unique()) < 2:
                LOGGER.info("🧠 AI Learning: Both Win (1) and Loss (0) samples are required to train.")
                return False

            num_pos = (y == 1).sum()
            num_neg = (y == 0).sum()
            scale_pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
            self.model.set_params(scale_pos_weight=scale_pos_weight)

            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True

            importances = self.model.feature_importances_
            top_features = sorted(zip(self.feature_columns, importances), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"{f}: {imp*100:.1f}%" for f, imp in top_features])

            LOGGER.info(f"✅ Advanced AI Model Retrained on {len(df)} trades!")
            LOGGER.info(f"🔥 Top Features: {top_str}")
            return True
        except Exception as e:
            LOGGER.error(f"❌ Error during AI retraining: {e}")
            return False
        finally:
            self._training = False

    def predict_signal_quality(self, feature_dict):
        if not self.is_trained:
            return 0.80
        try:
            input_data = {col: feature_dict.get(col, 0.0) for col in self.feature_columns}
            X_input = pd.DataFrame([input_data])[self.feature_columns].astype(float)
            X_scaled = self.scaler.transform(X_input)
            prob = self.model.predict_proba(X_scaled)[0][1]
            return float(prob)
        except Exception as e:
            LOGGER.error(f"❌ Error during AI prediction: {e}")
            return 0.80

ai_engine = AdvancedSelfLearningAIEngine()

async def async_retrain_model():
    """Async wrapper for retrain_model to avoid blocking event loop (مشکل ۱ - FIXED)"""
    try:
        await asyncio.to_thread(ai_engine.retrain_model)
    except Exception as e:
        LOGGER.error(f"❌ Background retrain failed: {e}")

def update_trade_outcome(trade_id, outcome):
    """FIXED: فقط در صورت موفقیت retrain می‌کند و async است"""
    query = "UPDATE trade_features SET outcome = ? WHERE id = ?"
    success = execute_db_query(query, (outcome, trade_id))
    if success:
        asyncio.create_task(async_retrain_model())
    return success

# =========================================================
# ۵. Rate Limiter صرافی‌ها
# =========================================================
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
                wait_time = (1 - self.tokens) * (self.per / self.rate)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

binance_limiter = RateLimiter(rate=20, per=1)
bybit_limiter = RateLimiter(rate=10, per=1)
okx_limiter = RateLimiter(rate=10, per=1)

def _parse_bybit(data):
    try:
        if data.get("retCode") != 0:
            return None
        result = data.get("result", {}).get("list", [])
        return [[int(x[0]), x[1], x[2], x[3], x[4], x[5]] for x in reversed(result)]
    except Exception:
        return None

def _parse_okx(data):
    try:
        result = data.get("data", [])
        return [[int(x[0]), x[1], x[2], x[3], x[4], x[5]] for x in reversed(result)]
    except Exception:
        return None

EXCHANGES = [
    {
        "name": "Binance",
        "weight": 10,
        "limiter": binance_limiter,
        "url": "https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=100",
        "interval_map": {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"},
        "parser": lambda data: data if isinstance(data, list) else None,
    },
    {
        "name": "Bybit",
        "weight": 8,
        "limiter": bybit_limiter,
        "url": "https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit=100",
        "interval_map": {"15m": "15", "1h": "60", "4h": "240", "1d": "D"},
        "parser": lambda data: _parse_bybit(data),
    },
    {
        "name": "OKX",
        "weight": 8,
        "limiter": okx_limiter,
        "url": "https://www.okx.com/api/v5/market/history-candles?instId={symbol}-SWAP&bar={interval}&limit=100",
        "interval_map": {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"},
        "parser": lambda data: _parse_okx(data),
    }
]

# =========================================================
# ۶. دریافت اطلاعات دفتر سفارشات (Order Book Depth)
# =========================================================
async def fetch_order_book_metrics(session, symbol, depth_limit=50):
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={depth_limit}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=4)) as resp:
            if resp.status == 200:
                data = await resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])

                if not bids or not asks:
                    return None

                tot_bid_vol = sum(float(b[1]) for b in bids)
                tot_ask_vol = sum(float(a[1]) for a in asks)
                total_vol = tot_bid_vol + tot_ask_vol

                if total_vol == 0:
                    return None

                ob_imbalance = tot_bid_vol / total_vol

                avg_bid_vol = tot_bid_vol / len(bids)
                avg_ask_vol = tot_ask_vol / len(asks)

                max_bid_wall = max(float(b[1]) for b in bids)
                max_ask_wall = max(float(a[1]) for a in asks)

                bid_wall_ratio = max_bid_wall / avg_bid_vol if avg_bid_vol > 0 else 1.0
                ask_wall_ratio = max_ask_wall / avg_ask_vol if avg_ask_vol > 0 else 1.0

                return {
                    "ob_imbalance": round(ob_imbalance, 3),
                    "bid_wall_ratio": round(bid_wall_ratio, 2),
                    "ask_wall_ratio": round(ask_wall_ratio, 2)
                }
    except Exception as e:
        LOGGER.error(f"Order book fetch error for {symbol}: {e}")
    return None

# =========================================================
# ۷. اعتبارسنجی داده‌ها
# =========================================================
def validate_klines(klines, symbol):
    if not klines or len(klines) < 10:
        return False, "too_few_klines"
    try:
        last_close = float(klines[-1][4])
        prev_close = float(klines[-2][4])
    except (IndexError, ValueError, TypeError):
        return False, "invalid_format"

    if prev_close > 0:
        change = abs(last_close - prev_close) / prev_close
        if change > 0.5:
            return False, "suspicious_jump"

    if last_close > 1000000 or last_close < 0.000001:
        return False, "suspicious_range"

    return True, "ok"

async def cross_check_price(session, symbol):
    """FIXED: OKX هم اضافه شد (مشکل ۶)"""
    prices = {}

    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.json()
                prices["Binance"] = float(data.get("price", 0))
    except Exception:
        pass

    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("retCode") == 0:
                    tickers = data.get("result", {}).get("list", [])
                    if tickers:
                        prices["Bybit"] = float(tickers[0].get("lastPrice", 0))
    except Exception:
        pass

    try:
        url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}-SWAP"
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("code") == "0":
                    tickers = data.get("data", [])
                    if tickers:
                        prices["OKX"] = float(tickers[0].get("last", 0))
    except Exception:
        pass

    if len(prices) >= 2:
        vals = [v for v in prices.values() if v > 0]
        if len(vals) >= 2:
            max_diff = max(vals) / min(vals) - 1
            if max_diff > 0.05:
                return False, prices
    return True, prices

# =========================================================
# ۸. اندیکاتورها و محاسبات تکنیکال
#    (مشکل ۳ و ۴ - FIXED: pivot >= و Wilder RSI)
# =========================================================
def calculate_rsi(closes, period=14):
    """Wilder smoothing RSI (مشکل ۴ - FIXED)"""
    if len(closes) < period + 1:
        return 50.0

    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)

def calculate_dmi(highs, lows, closes, period=14):
    if len(highs) < period * 2:
        return 0.0, 0.0, 0.0

    tr_list, plus_dm, minus_dm = [], [], []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]

        p_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        m_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr_list.append(tr)
        plus_dm.append(p_dm)
        minus_dm.append(m_dm)

    if len(tr_list) < period:
        return 0.0, 0.0, 0.0

    smooth_tr = sum(tr_list[:period])
    smooth_p_dm = sum(plus_dm[:period])
    smooth_m_dm = sum(minus_dm[:period])

    dx_list = []
    for i in range(period, len(tr_list)):
        smooth_tr = smooth_tr - (smooth_tr / period) + tr_list[i]
        smooth_p_dm = smooth_p_dm - (smooth_p_dm / period) + plus_dm[i]
        smooth_m_dm = smooth_m_dm - (smooth_m_dm / period) + minus_dm[i]

        p_di = (smooth_p_dm / smooth_tr * 100) if smooth_tr > 0 else 0
        m_di = (smooth_m_dm / smooth_tr * 100) if smooth_tr > 0 else 0

        di_sum = p_di + m_di
        dx = (abs(p_di - m_di) / di_sum * 100) if di_sum > 0 else 0
        dx_list.append((p_di, m_di, dx))

    if not dx_list:
        return 0.0, 0.0, 0.0

    last_p_di, last_m_di, _ = dx_list[-1]
    adx = sum(x[2] for x in dx_list[-period:]) / period if len(dx_list) >= period else dx_list[-1][2]

    return round(last_p_di, 2), round(last_m_di, 2), round(adx, 2)

def calculate_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        tr_list.append(tr)
    atr = sum(tr_list[:period]) / period
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
    return atr

def find_pivots(highs, lows, left_right=3):
    """FIXED: >= برای pivotهای مساوی (مشکل ۳)"""
    pivot_highs, pivot_lows = [], []
    n = len(highs)
    for i in range(left_right, n - left_right - 1):
        if all(highs[i] >= highs[i - j] for j in range(1, left_right + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, left_right + 1)):
            pivot_highs.append((i, highs[i]))
        if all(lows[i] <= lows[i - j] for j in range(1, left_right + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, left_right + 1)):
            pivot_lows.append((i, lows[i]))
    return pivot_highs, pivot_lows

def check_dow_theory_trend(pivot_highs, pivot_lows):
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return "NEUTRAL"
    last_high1, last_high2 = pivot_highs[-1][1], pivot_highs[-2][1]
    last_low1, last_low2 = pivot_lows[-1][1], pivot_lows[-2][1]
    if last_high1 > last_high2 and last_low1 > last_low2:
        return "BULLISH"
    elif last_high1 < last_high2 and last_low1 < last_low2:
        return "BEARISH"
    return "NEUTRAL"

def extract_htf_sr_levels(klines_4h, klines_1d):
    supports, resistances = [], []
    for klines in [klines_4h, klines_1d]:
        if klines and len(klines) >= 30:
            h = [float(k[2]) for k in klines[:-1]]
            l = [float(k[3]) for k in klines[:-1]]
            ph, pl = find_pivots(h, l)
            resistances.extend([p[1] for p in ph[-3:]])
            supports.extend([p[1] for p in pl[-3:]])
    return supports, resistances

# =========================================================
# ۹. دریافت کندل‌ها و اسکن نمادها
# =========================================================
async def fetch_klines_with_failover(session, symbol, interval):
    sorted_exchanges = sorted(EXCHANGES, key=lambda x: x["weight"], reverse=True)
    for ex in sorted_exchanges:
        try:
            await ex["limiter"].acquire()
            mapped_interval = ex["interval_map"].get(interval, interval)
            url = ex["url"].format(symbol=symbol, interval=mapped_interval)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8), headers={"User-Agent": "TradingBot/1.0"}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    klines = ex["parser"](data)
                    if klines and len(klines) >= 50:
                        valid, _ = validate_klines(klines, symbol)
                        if valid:
                            return klines
        except Exception:
            pass
        await asyncio.sleep(0.02)
    return None

async def update_btc_trend_and_volatility(session):
    global GLOBAL_BTC_TREND, BTC_VOLATILITY_PAUSE_UNTIL
    try:
        klines_4h = await fetch_klines_with_failover(session, "BTCUSDT", "4h")
        if klines_4h:
            highs = [float(k[2]) for k in klines_4h[:-1]]
            lows = [float(k[3]) for k in klines_4h[:-1]]
            ph, pl = find_pivots(highs, lows)
            GLOBAL_BTC_TREND = check_dow_theory_trend(ph, pl)

        klines_15m = await fetch_klines_with_failover(session, "BTCUSDT", "15m")
        if klines_15m and len(klines_15m) >= 1:
            b_open = float(klines_15m[-1][1])
            b_close = float(klines_15m[-1][4])
            change_pct = abs((b_close - b_open) / b_open) * 100
            if change_pct >= VOLATILITY_THRESHOLD_PERCENT:
                BTC_VOLATILITY_PAUSE_UNTIL = time.time() + (VOLATILITY_PAUSE_MINUTES * 60)
                LOGGER.warning(f"⚠️ BTC Volatility Spike ({change_pct:.2f}%). Pausing signals for {VOLATILITY_PAUSE_MINUTES}m.")
    except Exception as e:
        LOGGER.error(f"Error updating BTC status: {e}")

async def get_all_usdt_symbols(session):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                btc_price = None
                for item in data:
                    if item.get("symbol") == "BTCUSDT":
                        btc_price = float(item.get("lastPrice", 0))
                        break
                if btc_price is None or btc_price <= 0:
                    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
                min_usdt_volume = MIN_BTC_VOLUME * btc_price
                valid_symbols = []
                for item in data:
                    symbol = item.get("symbol", "")
                    quote_volume = float(item.get("quoteVolume", 0))
                    if symbol.endswith("USDT") and quote_volume >= min_usdt_volume:
                        valid_symbols.append(symbol)
                return valid_symbols
    except Exception as e:
        LOGGER.error(f"Error fetching symbols: {e}")
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

async def get_all_usdt_symbols_cached(session):
    global _symbol_cache
    now = time.time()
    if now - _symbol_cache["last_update"] > 300 or not _symbol_cache["symbols"]:
        _symbol_cache["symbols"] = await get_all_usdt_symbols(session)
        _symbol_cache["last_update"] = now
        LOGGER.info(f"🔄 Symbol cache refreshed: {len(_symbol_cache['symbols'])} symbols")
    return _symbol_cache["symbols"]

# =========================================================
# ۱۰. استراتژی‌ها (مشکل ۵ - FIXED: Strategy Pattern)
# =========================================================
def check_rsi_dmi_long(features):
    return (features['is_trending'] and features['strong_buyers'] and 
            features['rsi_bullish_trigger'] and features['is_volume_spike'])

def check_rsi_dmi_short(features):
    return (features['is_trending'] and features['strong_sellers'] and 
            features['rsi_bearish_trigger'] and features['is_volume_spike'])

def check_candle_setup_long(features):
    return (features['trend'] != "BEARISH" and features['is_green_candle'] and 
            features['is_valid_size'] and features['is_strong_lower_wick'] and
            features['has_minimal_upper_wick'] and features['is_sma7_bounce'] and
            features['is_bounce_confirmed'] and features['is_volume_spike'])

def check_candle_setup_short(features):
    return (features['trend'] != "BULLISH" and features['is_red_candle'] and
            features['is_valid_size'] and features['is_strong_upper_wick'] and
            features['has_minimal_lower_wick'] and features['is_sma7_rejection'] and
            features['is_rejection_confirmed'] and features['is_volume_spike'])

def check_htf_range_breakout_long(features):
    return (features['is_in_range'] and features['is_green_candle'] and
            features['c_close'] > features['range_high'] and
            features['is_volume_spike'] and
            (features['is_near_htf_support'] or features['is_near_htf_resistance']))

def check_htf_range_breakout_short(features):
    return (features['is_in_range'] and features['is_red_candle'] and
            features['c_close'] < features['range_low'] and
            features['is_volume_spike'] and
            (features['is_near_htf_resistance'] or features['is_near_htf_support']))

def check_smc_long(features):
    return features['is_liquidity_sweep_long'] and features['is_volume_spike']

def check_smc_short(features):
    return features['is_liquidity_sweep_short'] and features['is_volume_spike']

LONG_STRATEGIES = [
    ("🔥 RSI + DMI Momentum", check_rsi_dmi_long),
    ("Candle Setup 📌", check_candle_setup_long),
    ("Range Breakout 🚀", check_htf_range_breakout_long),
    ("SMC Liquidity Sweep 🎯", check_smc_long),
]

SHORT_STRATEGIES = [
    ("🔻 RSI + DMI Breakdown", check_rsi_dmi_short),
    ("Candle Setup 📌", check_candle_setup_short),
    ("Range Breakdown 📉", check_htf_range_breakout_short),
    ("SMC Liquidity Sweep 🎯", check_smc_short),
]

# =========================================================
# ۱۱. تحلیل هوشمند سیگنال + فیلتر Order Book
# =========================================================
def build_feature_dict(klines, htf_supports, htf_resistances):
    if len(klines) < 50:
        return None

    closed_klines = klines[:-1]
    opens = [float(k[1]) for k in closed_klines]
    highs = [float(k[2]) for k in closed_klines]
    lows = [float(k[3]) for k in closed_klines]
    closes = [float(k[4]) for k in closed_klines]
    volumes = [float(k[5]) for k in closed_klines]

    current_live_price = float(klines[-1][4])
    rsi = calculate_rsi(closes)
    rsi_prev = calculate_rsi(closes[:-1]) if len(closes) > 1 else rsi
    plus_di, minus_di, adx = calculate_dmi(highs, lows, closes)
    atr = calculate_atr(highs, lows, closes)
    sma7 = sum(closes[-7:]) / 7

    c_open, c_high, c_low, c_close, c_vol = opens[-1], highs[-1], lows[-1], closes[-1], volumes[-1]
    body_bottom, body_top = min(c_open, c_close), max(c_open, c_close)
    body = abs(c_close - c_open)
    total_range = c_high - c_low

    if total_range == 0 or body == 0 or atr == 0:
        return None

    upper_wick = c_high - body_top
    lower_wick = body_bottom - c_low
    spread_pct = (total_range / c_low) * 100
    if spread_pct > 2.0:
        return None

    pivot_highs, pivot_lows = find_pivots(highs, lows)
    trend = check_dow_theory_trend(pivot_highs, pivot_lows)

    avg_vol_20 = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else c_vol
    is_volume_spike = (c_vol >= 1.5 * avg_vol_20)

    recent_min_low = min(lows[-6:-1])
    is_liquidity_sweep_long = (c_low < recent_min_low) and (c_close > recent_min_low)

    recent_max_high = max(highs[-6:-1])
    is_liquidity_sweep_short = (c_high > recent_max_high) and (c_close < recent_max_high)

    lookback = 12
    range_high = max(highs[-lookback:-1])
    range_low = min(lows[-lookback:-1])
    range_width_pct = ((range_high - range_low) / range_low) * 100 if range_low > 0 else 999.0
    is_in_range = (range_width_pct <= 3.5)

    is_near_htf_support = any(range_low >= supp * 0.985 and range_low <= supp * 1.025 for supp in htf_supports) if htf_supports else True
    is_near_htf_resistance = any(range_high <= res * 1.015 and range_high >= res * 0.975 for res in htf_resistances) if htf_resistances else True

    price_to_sma7_ratio = (c_close / sma7) if sma7 > 0 else 1.0
    atr_pct = (atr / c_close) * 100 if c_close > 0 else 0.0

    trend_map = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}

    return {
        'rsi': float(rsi), 'rsi_prev': float(rsi_prev),
        'plus_di': float(plus_di), 'minus_di': float(minus_di),
        'adx': float(adx), 'atr': float(atr), 'sma7': float(sma7),
        'c_open': float(c_open), 'c_high': float(c_high),
        'c_low': float(c_low), 'c_close': float(c_close), 'c_vol': float(c_vol),
        'body': float(body), 'total_range': float(total_range),
        'upper_wick': float(upper_wick), 'lower_wick': float(lower_wick),
        'spread_pct': float(spread_pct), 'trend': trend,
        'avg_vol_20': float(avg_vol_20), 'is_volume_spike': is_volume_spike,
        'is_liquidity_sweep_long': is_liquidity_sweep_long,
        'is_liquidity_sweep_short': is_liquidity_sweep_short,
        'range_high': float(range_high), 'range_low': float(range_low),
        'is_in_range': is_in_range,
        'is_near_htf_support': is_near_htf_support,
        'is_near_htf_resistance': is_near_htf_resistance,
        'price_to_sma7_ratio': float(price_to_sma7_ratio),
        'atr_pct': float(atr_pct),
        'trend_code': trend_map.get(trend, 0),
        'is_trending': adx >= 25.0,
        'strong_buyers': (plus_di - minus_di) >= 5.0,
        'strong_sellers': (minus_di - plus_di) >= 5.0,
        'rsi_bullish_trigger': (rsi > rsi_prev) and (55.0 <= rsi <= 72.0),
        'rsi_bearish_trigger': (rsi < rsi_prev) and (28.0 <= rsi <= 45.0),
        'is_green_candle': c_close > c_open,
        'is_red_candle': c_close < c_open,
        'is_valid_size': total_range >= 0.5 * atr,
        'is_strong_lower_wick': lower_wick >= 2.0 * body and lower_wick / total_range >= 0.50,
        'has_minimal_upper_wick': upper_wick <= 0.20 * total_range,
        'is_sma7_bounce': c_low <= sma7 and sma7 <= body_top,
        'is_bounce_confirmed': c_close > sma7,
        'is_strong_upper_wick': upper_wick >= 2.0 * body and upper_wick / total_range >= 0.50,
        'has_minimal_lower_wick': lower_wick <= 0.20 * total_range,
        'is_sma7_rejection': c_high >= sma7 and sma7 >= body_bottom,
        'is_rejection_confirmed': c_close < sma7,
        'current_live_price': float(current_live_price),
        'candle_time': closed_klines[-1][0],
    }

def _build_signal(direction, confirmed_strategies, features, symbol, interval, max_sl_percent):
    strategy_text = " + ".join(confirmed_strategies)

    ai_feature_dict = {
        'rsi': features['rsi'],
        'spread_pct': features['spread_pct'],
        'vol_ratio': features['c_vol'] / features['avg_vol_20'] if features['avg_vol_20'] > 0 else 1.0,
        'lower_wick_ratio': features['lower_wick'] / features['total_range'],
        'upper_wick_ratio': features['upper_wick'] / features['total_range'],
        'trend_code': features['trend_code'],
        'adx': features['adx'],
        'plus_di': features['plus_di'],
        'minus_di': features['minus_di'],
        'price_to_sma7_ratio': features['price_to_sma7_ratio'],
        'atr_pct': features['atr_pct']
    }

    win_probability = ai_engine.predict_signal_quality(ai_feature_dict)
    ai_score = round(win_probability * 100, 1)
    if ai_score < 50.0:
        return None

    entry_price = features['c_close']
    price_diff_percent = abs((features['current_live_price'] - entry_price) / entry_price) * 100

    if direction == "LONG":
        stop_loss = max(features['c_low'], entry_price - (1.5 * features['atr']))
        risk = entry_price - stop_loss
        if risk <= 0:
            return None
        sl_percent = (risk / entry_price) * 100
        if sl_percent > max_sl_percent:
            return None
        tp1 = round(entry_price + (risk * 2), 5)
        tp2 = round(entry_price + (risk * 5), 5)
        tp3 = round(entry_price + (risk * 7), 5)
        direction_label = "LONG 🟢"
    else:
        stop_loss = min(features['c_high'], entry_price + (1.5 * features['atr']))
        risk = stop_loss - entry_price
        if risk <= 0:
            return None
        sl_percent = (risk / entry_price) * 100
        if sl_percent > max_sl_percent:
            return None
        tp1 = round(entry_price - (risk * 2), 5)
        tp2 = round(entry_price - (risk * 5), 5)
        tp3 = round(entry_price - (risk * 7), 5)
        direction_label = "SHORT 🔴"

    trade_id = f"{symbol}_{int(time.time())}"

    insert_sql = "INSERT INTO trade_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)"
    execute_db_query(insert_sql, (
        trade_id, symbol, direction, ai_feature_dict['rsi'], ai_feature_dict['spread_pct'],
        ai_feature_dict['vol_ratio'], ai_feature_dict['lower_wick_ratio'],
        ai_feature_dict['upper_wick_ratio'], ai_feature_dict['trend_code'],
        ai_feature_dict['adx'], ai_feature_dict['plus_di'], ai_feature_dict['minus_di'],
        ai_feature_dict['price_to_sma7_ratio'], ai_feature_dict['atr_pct']
    ))

    return {
        "trade_id": trade_id,
        "strategy": strategy_text,
        "direction": direction_label,
        "entry_price": entry_price,
        "stop_loss": round(stop_loss, 5),
        "sl_percent": round(sl_percent, 2),
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "sma7": round(features['sma7'], 5),
        "rsi": features['rsi'],
        "ai_score": ai_score,
        "price_diff_percent": round(price_diff_percent, 2),
        "trend": features['trend'],
        "candle_time": features['candle_time']
    }

def analyze_market_signal(klines, symbol, interval, htf_supports, htf_resistances, max_sl_percent=5.0, ob_data=None):
    if time.time() < BTC_VOLATILITY_PAUSE_UNTIL:
        return None

    features = build_feature_dict(klines, htf_supports, htf_resistances)
    if features is None:
        return None

    current_time_ms = int(time.time() * 1000)
    current_candle_start_ms = int(klines[-1][0])
    elapsed_seconds = (current_time_ms - current_candle_start_ms) / 1000.0
    if elapsed_seconds > MAX_SIGNAL_AGE_SECONDS:
        return None

    # Check LONG strategies
    long_confirmed = []
    for name, check_fn in LONG_STRATEGIES:
        if check_fn(features):
            long_confirmed.append(f"{name} ({interval})")

    if long_confirmed and features['rsi'] <= 80.0:
        if ob_data:
            if ob_data["ob_imbalance"] < 0.40:
                LOGGER.info(f"⛔ LONG Rejected for {symbol}: Orderbook dominated by sellers ({ob_data['ob_imbalance']*100}% Bids)")
                return None
            if ob_data["ask_wall_ratio"] > 4.0:
                LOGGER.info(f"⛔ LONG Rejected for {symbol}: Massive Sell Wall in Order Book ({ob_data['ask_wall_ratio']}x avg)")
                return None
        return _build_signal("LONG", long_confirmed, features, symbol, interval, max_sl_percent)

    # Check SHORT strategies
    short_confirmed = []
    for name, check_fn in SHORT_STRATEGIES:
        if check_fn(features):
            short_confirmed.append(f"{name} ({interval})")

    if short_confirmed and features['rsi'] >= 20.0:
        if ob_data:
            if ob_data["ob_imbalance"] > 0.60:
                LOGGER.info(f"⛔ SHORT Rejected for {symbol}: Orderbook dominated by buyers ({ob_data['ob_imbalance']*100}% Bids)")
                return None
            if ob_data["bid_wall_ratio"] > 4.0:
                LOGGER.info(f"⛔ SHORT Rejected for {symbol}: Massive Buy Wall in Order Book ({ob_data['bid_wall_ratio']}x avg)")
                return None
        return _build_signal("SHORT", short_confirmed, features, symbol, interval, max_sl_percent)

    return None

# =========================================================
# ۱۲. تعقیب معاملات و به روزرسانی دیتابیس ابری
# =========================================================
async def track_active_trades(session, bot):
    if not active_trades:
        return

    async with active_trades_lock:
        trades = list(active_trades.items())

    now = time.time()

    for trade_key, trade in trades:
        symbol = trade["symbol"]

        trade_age = now - trade.get("created_at", now)
        if trade_age > 172800:
            async with active_trades_lock:
                if trade_key in active_trades:
                    del active_trades[trade_key]
            continue

        klines = await fetch_klines_with_failover(session, symbol, "15m")
        if not klines:
            if trade_age > 86400:
                async with active_trades_lock:
                    if trade_key in active_trades:
                        del active_trades[trade_key]
            continue

        current_price = float(klines[-1][4])

        async with active_trades_lock:
            if trade_key not in active_trades:
                continue

            if trade["direction"] == "LONG 🟢":
                if current_price <= trade["stop_loss"]:
                    msg = f"❌ **Stop Loss Hit!**\n🪙 `#{symbol}` | SL: `{trade['stop_loss']}` (-{trade['sl_percent']}%)"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    update_trade_outcome(trade["db_id"], 0)
                    del active_trades[trade_key]

                elif current_price >= trade["tp3"] and not trade.get("tp3_hit"):
                    msg = f"🎯🎯🎯 **ALL TARGETS HIT (TP3)!**\n🪙 `#{symbol}` | Final Price: `{current_price}` 🔥"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    update_trade_outcome(trade["db_id"], 1)
                    del active_trades[trade_key]

                elif current_price >= trade["tp2"] and not trade.get("tp2_hit"):
                    active_trades[trade_key]["tp2_hit"] = True
                    msg = f"🚀 **Target 2 Hit (TP2)!**\n🪙 `#{symbol}` | Price: `{current_price}`"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)

                elif current_price >= trade["tp1"] and not trade.get("tp1_hit"):
                    active_trades[trade_key]["tp1_hit"] = True
                    msg = f"✅ **Target 1 Hit (TP1)!**\n🪙 `#{symbol}` | Price: `{current_price}`"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)

            elif trade["direction"] == "SHORT 🔴":
                if current_price >= trade["stop_loss"]:
                    msg = f"❌ **Stop Loss Hit!**\n🪙 `#{symbol}` | SL: `{trade['stop_loss']}` (-{trade['sl_percent']}%)"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    update_trade_outcome(trade["db_id"], 0)
                    del active_trades[trade_key]

                elif current_price <= trade["tp3"] and not trade.get("tp3_hit"):
                    msg = f"🎯🎯🎯 **ALL TARGETS HIT (TP3)!**\n🪙 `#{symbol}` | Final Price: `{current_price}` 🔥"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    update_trade_outcome(trade["db_id"], 1)
                    del active_trades[trade_key]

                elif current_price <= trade["tp2"] and not trade.get("tp2_hit"):
                    active_trades[trade_key]["tp2_hit"] = True
                    msg = f"🚀 **Target 2 Hit (TP2)!**\n🪙 `#{symbol}` | Price: `{current_price}`"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)

                elif current_price <= trade["tp1"] and not trade.get("tp1_hit"):
                    active_trades[trade_key]["tp1_hit"] = True
                    msg = f"✅ **Target 1 Hit (TP1)!**\n🪙 `#{symbol}` | Price: `{current_price}`"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)

# =========================================================
# ۱۳. مدیریت دستورات تلگرام و بازخورد کاربر
#    (مشکل ۷ - FIXED: long polling بدون sleep اضافی)
# =========================================================
async def send_telegram_message(bot, chat_id, text, reply_markup=None, retries=3):
    for i in range(retries):
        try:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
            return True
        except Exception as e:
            if i == retries - 1:
                LOGGER.error(f"Telegram error after {retries} retries: {e}")
                return False
            wait = 2 ** i
            await asyncio.sleep(wait)

async def telegram_command_listener(bot):
    last_update_id = 0
    while True:
        try:
            updates = await bot.get_updates(offset=last_update_id + 1, timeout=30)
            for update in updates:
                last_update_id = update.update_id

                if update.callback_query:
                    query = update.callback_query
                    data = query.data

                    if data.startswith("fb_bad_"):
                        trade_id = data.replace("fb_bad_", "")
                        update_trade_outcome(trade_id, 0)
                        await query.answer("❌ این سیگنال به عنوان خطای الگویی ثبت شد و مدل AI بازآموزی گردید.")
                        try:
                            await query.edit_message_text(
                                text=query.message.text + "\n\n⚠️ **بازخورد شما ثبت شد: سیگنال اشتباه (یادگیری AI)**",
                                parse_mode=ParseMode.MARKDOWN
                            )
                        except Exception:
                            pass

                    elif data.startswith("fb_good_"):
                        trade_id = data.replace("fb_good_", "")
                        update_trade_outcome(trade_id, 1)
                        await query.answer("✅ این سیگنال به عنوان الگوی موفق ثبت شد.")
                        try:
                            await query.edit_message_text(
                                text=query.message.text + "\n\n✅ **بازخورد شما ثبت شد: سیگنال موفق (یادگیری AI)**",
                                parse_mode=ParseMode.MARKDOWN
                            )
                        except Exception:
                            pass

                if update.message and update.message.text:
                    raw_text = update.message.text.strip()
                    cmd = raw_text.split('@')[0].lower()
                    chat_id = update.message.chat_id

                    if cmd == "/stats":
                        df_all = fetch_db_df("SELECT * FROM trade_features")
                        total = len(df_all)
                        df_closed = df_all[df_all['outcome'].notnull()] if not df_all.empty else pd.DataFrame()
                        wins = len(df_closed[df_closed['outcome'] == 1]) if not df_closed.empty else 0
                        sl = len(df_closed[df_closed['outcome'] == 0]) if not df_closed.empty else 0
                        win_rate = round((wins / len(df_closed) * 100), 1) if not df_closed.empty and len(df_closed) > 0 else 0.0

                        msg = (
                            f"📊 **Bot Performance & Win Rate Stats (Turso Cloud)**\n\n"
                            f"🔢 **Total Signals Saved:** `{total}`\n"
                            f"🎯 **Successful Trades (Wins):** `{wins}`\n"
                            f"❌ **Stop Loss Hits:** `{sl}`\n\n"
                            f"🏆 **Current Win Rate:** `{win_rate}%`"
                        )
                        await send_telegram_message(bot, chat_id, msg)

                    elif cmd == "/active":
                        async with active_trades_lock:
                            if not active_trades:
                                await send_telegram_message(bot, chat_id, "ℹ️ هیچ پوزیشن فعالی در حال حاضر وجود ندارد.")
                            else:
                                active_list = "\n".join([f"🔹 `#{v['symbol']}` ({v['direction']}) - Entry: `{v['entry_price']}`" for k, v in active_trades.items()])
                                msg = f"📌 **Active Tracked Trades ({len(active_trades)}):**\n\n{active_list}"
                                await send_telegram_message(bot, chat_id, msg)

                    elif cmd == "/pause":
                        global BTC_VOLATILITY_PAUSE_UNTIL
                        BTC_VOLATILITY_PAUSE_UNTIL = 0
                        await send_telegram_message(bot, chat_id, "⏸️ **Volatility pause deactivated.**\n✅ Bot is now active.")

                    elif cmd == "/debug":
                        df_all = fetch_db_df("SELECT * FROM trade_features")
                        msg = (
                            "🔍 **Debug Info:**\n\n"
                            f"🌐 BTC Trend: `{GLOBAL_BTC_TREND}`\n"
                            f"⏸️ Volatility Pause: `{'YES' if time.time() < BTC_VOLATILITY_PAUSE_UNTIL else 'NO'}`\n"
                            f"🤖 AI Model Trained: `{'YES' if ai_engine.is_trained else 'NO (Collecting Data)'}`\n"
                            f"☁️ Cloud DB Saved Signals: `{len(df_all)}`\n"
                            f"📊 Active Trades: `{len(active_trades)}`\n"
                            f"💾 Cached Symbols: `{len(_symbol_cache['symbols'])}`"
                        )
                        await send_telegram_message(bot, chat_id, msg)

                    elif cmd in ["/start", "/help"]:
                        msg = (
                            "🤖 **Trading Bot Control Menu**\n\n"
                            "▫️ `/stats` : آمار دیتابیس ابری\n"
                            "▫️ `/active` : پوزیشن‌های فعال\n"
                            "▫️ `/pause` : لغو غیرفعال بودن نوسان\n"
                            "▫️ `/debug` : اطلاعات دیباگ و دیتابیس\n"
                            "▫️ `/help` : راهنما"
                        )
                        await send_telegram_message(bot, chat_id, msg)
        except Exception as e:
            LOGGER.error(f"Command Listener Error: {e}")

# =========================================================
# ۱۴. اسکن موازی نمادها
#    (مشکل ۲ - FIXED: همه timeframeها چک می‌شوند)
# =========================================================
async def process_single_symbol(symbol, session, bot, semaphore):
    async with semaphore:
        try:
            async with active_trades_lock:
                has_active_trade = any(v["symbol"] == symbol for v in active_trades.values())
            if has_active_trade:
                return

            if symbol in ["ANTHROPICUSDT", "ANTHRCUSDT"]:
                ok, _ = await cross_check_price(session, symbol)
                if not ok:
                    return

            klines_4h = await fetch_klines_with_failover(session, symbol, "4h")
            klines_1d = await fetch_klines_with_failover(session, symbol, "1d")
            htf_supports, htf_resistances = extract_htf_sr_levels(klines_4h, klines_1d)
            ob_data = await fetch_order_book_metrics(session, symbol)

            # مشکل ۲ - FIXED: همه timeframeها رو چک کن و بهترین رو انتخاب کن
            all_signals = []
            for interval in TIMEFRAMES:
                klines = await fetch_klines_with_failover(session, symbol, interval)
                if not klines:
                    continue

                signal = analyze_market_signal(klines, symbol, interval, htf_supports, htf_resistances, MAX_SL_PERCENT, ob_data=ob_data)
                if signal:
                    signal['interval'] = interval
                    all_signals.append(signal)

            if not all_signals:
                return

            # بهترین سیگنال (بالاترین AI Score)
            best_signal = max(all_signals, key=lambda s: s['ai_score'])
            signal = best_signal
            interval = signal['interval']

            alert_key = f"{symbol}_{interval}_{signal['candle_time']}"
            if await alert_cache.exists(alert_key):
                return
            await alert_cache.add(alert_key)

            slippage_warning_text = ""
            if signal['price_diff_percent'] > SLIPPAGE_WARNING_THRESHOLD:
                slippage_warning_text = f"\n⚠️ **هشدار حرکت قیمت:** قیمت به میزان `{signal['price_diff_percent']}%` حرکت کرده است."

            msg = (
                f"🚨 **NEW TRADING SIGNAL** 🚨\n\n"
                f"🪙 **Symbol:** `#{symbol}`\n"
                f"📊 **Direction:** `{signal['direction']}`\n"
                f"🎯 **Strategy:** `{signal['strategy']}`\n"
                f"🤖 **AI Score:** `{signal['ai_score']}%` Confidence\n"
                f"⏱️ **Timeframe:** `{interval}`\n"
                f"{slippage_warning_text}\n"
                f"💵 **Entry Price:** `{signal['entry_price']}`\n"
                f"🛑 **Stop Loss:** `{signal['stop_loss']}` (-{signal['sl_percent']}%)\n\n"
                f"🎯 **TP1:** `{signal['tp1']}`\n"
                f"🚀 **TP2:** `{signal['tp2']}`\n"
                f"🔥 **TP3:** `{signal['tp3']}`\n\n"
                f"📈 **RSI:** `{signal['rsi']}` | Trend: `{signal['trend']}`"
            )

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("❌ سیگنال اشتباه (ثبت خطا)", callback_data=f"fb_bad_{signal['trade_id']}"),
                    InlineKeyboardButton("✅ سیگنال خوب", callback_data=f"fb_good_{signal['trade_id']}")
                ]
            ])

            await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg, reply_markup=keyboard)

            trade_key = f"{symbol}_{interval}"
            async with active_trades_lock:
                active_trades[trade_key] = {
                    "symbol": symbol,
                    "direction": signal["direction"],
                    "entry_price": signal["entry_price"],
                    "stop_loss": signal["stop_loss"],
                    "sl_percent": signal["sl_percent"],
                    "tp1": signal["tp1"],
                    "tp2": signal["tp2"],
                    "tp3": signal["tp3"],
                    "db_id": signal["trade_id"],
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "tp3_hit": False,
                    "created_at": time.time()
                }

        except asyncio.TimeoutError:
            LOGGER.warning(f"⏱️ Timeout processing {symbol}")
        except aiohttp.ClientError as e:
            LOGGER.warning(f"🌐 Network error for {symbol}: {e}")
        except Exception as e:
            LOGGER.error(f"❌ Unexpected error processing {symbol}: {e}")
            LOGGER.error(traceback.format_exc())

# =========================================================
# ۱۵. اجرای اصلی و وب‌سرور Render
# =========================================================
async def handle_health_check(request):
    return web.Response(text="Trading Bot AI service is live & active!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle_health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    LOGGER.info(f"🌐 Web Server listening on port {PORT}")

async def scanner_task():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        me = await bot.get_me()
        LOGGER.info(f"🤖 Bot Connected: @{me.username}")
    except Exception as e:
        LOGGER.error(f"❌ Telegram Auth Error: {e}")
        return

    asyncio.create_task(telegram_command_listener(bot))

    async with aiohttp.ClientSession() as session:
        await update_btc_trend_and_volatility(session)
        symbols = await get_all_usdt_symbols_cached(session)
        btc_counter = 0

        semaphore = asyncio.Semaphore(CONCURRENT_SCAN_LIMIT)

        while True:
            try:
                btc_counter += 1
                if btc_counter >= 15:
                    await update_btc_trend_and_volatility(session)
                    btc_counter = 0

                await track_active_trades(session, bot)

                tasks = [process_single_symbol(symbol, session, bot, semaphore) for symbol in symbols]
                await asyncio.gather(*tasks)

                await asyncio.sleep(5)

            except Exception as e:
                LOGGER.error(f"Error in scanner loop: {e}")
                await asyncio.sleep(5)

async def main():
    init_db()
    ai_engine.retrain_model()
    await start_web_server()
    await scanner_task()

if __name__ == "__main__":
    asyncio.run(main())
