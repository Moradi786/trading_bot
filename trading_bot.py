import asyncio
import logging
import os
import sqlite3
import time
import traceback
import hmac
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal, ROUND_DOWN

import aiohttp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from aiohttp import web

# ==========================================================
# ۰. تنظیمات و کانفیگ
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger("UnifiedBot")

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
PORT = int(os.getenv("PORT", 8080))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "1.0"))  # درصد
LEVERAGE = int(os.getenv("LEVERAGE", "3"))
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

DB_NAME = "unified_bot.db"
MODEL_PATH = "ai_model.joblib"
SCALER_PATH = "ai_scaler.joblib"

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
MAX_SL_PERCENT = 2.0
MIN_BTC_VOLUME = 250.0
MAX_SIGNAL_AGE = 180
MAX_SLIPPAGE = 0.2
ALERT_TTL = 86400

# ==========================================================
# ۱. دیتابیس Async (ترکیب ربات ۱)
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
            executed INTEGER DEFAULT 0, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS bot_stats (
            key TEXT PRIMARY KEY, value INTEGER DEFAULT 0)"""
    ]
    for q in queries:
        await db_execute(q)
    # Init stats
    for k in ["total_signals","tp1_hits","tp2_hits","tp3_hits","sl_hits","ai_trades"]:
        await db_execute("INSERT OR IGNORE INTO bot_stats (key, value) VALUES (?, 0)", (k,))
    LOGGER.info("✅ دیتابیس یکپارچه مقداردهی شد.")

# ==========================================================
# ۲. Rate Limiting (ترکیب ربات ۲)
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

# ==========================================================
# ۳. صرافی‌ها و دیتا (ترکیب ربات ۲)
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

async def fetch_order_book(session, symbol, limit=10):
    try:
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={limit}"
        async with session.get(url, timeout=5) as r:
            if r.status == 200:
                data = await r.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                return bids, asks
    except: pass
    return [], []

# ==========================================================
# ۴. اندیکاتورها (ترکیب ربات ۲)
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
# ۵. Order Book Analyzer (ترکیب ربات ۱)
# ==========================================================
class OrderBookAnalyzer:
    @staticmethod
    def analyze(bids, asks):
        if not bids or not asks: return {"imbalance": 0.0, "spread_pct": 0.0}
        bv = sum(float(b[1]) for b in bids[:10])
        av = sum(float(a[1]) for a in asks[:10])
        tv = bv + av + 1e-9
        imb = (bv - av) / tv
        bb = float(bids[0][0]); ba = float(asks[0][0])
        sp = ((ba - bb) / bb) * 100 if bb > 0 else 0.0
        return {"imbalance": float(imb), "spread_pct": float(sp)}

# ==========================================================
# ۶. موتور AI (ترکیب ربات ۱)
# ==========================================================
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
            LOGGER.info("💾 مدل AI ذخیره شد.")
        except Exception as e:
            LOGGER.error(f"❌ خطا در ذخیره مدل: {e}")

    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.is_trained = True
                LOGGER.info("🧠 مدل AI بارگذاری شد.")
            except Exception as e:
                LOGGER.error(f"❌ خطا در بارگذاری: {e}")

    async def predict(self, features: Dict[str, float]) -> float:
        if not self.is_trained: return 0.50
        def _pred():
            df = pd.DataFrame([features])[self.FEATURE_COLS].fillna(0.0)
            Xs = self.scaler.transform(df)
            return float(self.model.predict_proba(Xs)[0][1])
        return await asyncio.to_thread(_pred)

    async def retrain(self):
        if self._training: return False
        self._training = True
        try:
            df = await db_fetch_df("SELECT * FROM trade_features WHERE outcome IS NOT NULL")
            if len(df) < self.min_samples or len(df["outcome"].unique()) < 2:
                LOGGER.info(f"🧠 داده کافی برای آموزش نیست ({len(df)}).")
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
            LOGGER.info(f"✅ AI بازآموز شد روی {len(df)} معامله.")
            return True
        except Exception as e:
            LOGGER.error(f"❌ خطا در بازآموزی: {e}"); return False
        finally:
            self._training = False

# ==========================================================
# ۷. تحلیل سیگنال (ترکیب ربات ۲)
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
    if spread_pct > 2.0: return None

    ph, pl = find_pivots(H, L)
    trend = dow_trend(ph, pl)
    avg_v20 = sum(V[-21:-1]) / 20 if len(V) >= 21 else cv
    vol_spike = cv >= 1.5 * avg_v20

    # Strategies
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

    longs, shorts = [], []
    if dmi_long: longs.append(f"RSI+DMI ⚡({interval})")
    if setup_long: longs.append(f"Candle 📌({interval})")
    if breakout_long: longs.append(f"Breakout 🚀({interval})")
    if sweep_long: longs.append(f"SMC Sweep 🎯({interval})")

    if dmi_short: shorts.append(f"RSI+DMI ⚡({interval})")
    if setup_short: shorts.append(f"Candle 📌({interval})")
    if breakout_short: shorts.append(f"Breakdown 📉({interval})")
    if sweep_short: shorts.append(f"SMC Sweep 🎯({interval})")

    def build(direction, strategies, entry, sl, risk):
        sl_pct = (risk / entry) * 100 if entry > 0 else 999
        if sl_pct <= max_sl and risk > 0:
            return {
                "strategy": " + ".join(strategies), "direction": direction,
                "entry_price": entry, "stop_loss": round(sl, 5), "sl_percent": round(sl_pct, 2),
                "tp1": round(entry + (risk * 2), 5), "tp2": round(entry + (risk * 5), 5), "tp3": round(entry + (risk * 7), 5),
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
        res = build("LONG 🟢", longs, cc, sl, risk)
        if res and abs((live - cc) / cc) * 100 <= MAX_SLIPPAGE:
            return res
    if shorts:
        sl = min(ch, cc + 1.5 * atr)
        risk = sl - cc
        res = build("SHORT 🔴", shorts, cc, sl, risk)
        if res and abs((cc - live) / cc) * 100 <= MAX_SLIPPAGE:
            return res
    return None

# ==========================================================
# ۸. Risk Manager & Position Sizing (جدید)
# ==========================================================
class RiskManager:
    def __init__(self, risk_percent=RISK_PER_TRADE, leverage=LEVERAGE):
        self.risk_pct = risk_percent
        self.leverage = leverage

    def size(self, balance: float, entry: float, sl: float) -> Tuple[float, float]:
        risk_amount = balance * (self.risk_pct / 100.0)
        price_risk = abs(entry - sl)
        if price_risk <= 0: return 0.0, 0.0
        # Position size in USDT (notional)
        notional = (risk_amount / price_risk) * entry
        # Apply leverage
        margin = notional / self.leverage
        return round(notional, 2), round(margin, 2)

# ==========================================================
# ۹. Binance Futures Trader (جدید)
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
        qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
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
                url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
            async with self.session.request(method, url, headers=headers, timeout=10) as r:
                return await r.json()
        except Exception as e:
            LOGGER.error(f"Binance API error: {e}"); return None

    async def get_balance(self):
        if self.paper: return 10000.0  # Paper balance
        ts = int(time.time() * 1000)
        data = await self._request("GET", "/fapi/v2/account", {"timestamp": ts}, signed=True)
        if data:
            for a in data.get("assets", []):
                if a.get("asset") == "USDT":
                    return float(a.get("availableBalance", 0))
        return 0.0

    async def place_market_order(self, symbol: str, side: str, quantity: float, leverage: int = 3):
        if self.paper:
            LOGGER.info(f"📄 PAPER ORDER | {side} {quantity} {symbol} @ x{leverage}")
            return {"orderId": f"PAPER_{int(time.time()*1000)}", "status": "FILLED"}
        # Set leverage first
        await self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage, "timestamp": int(time.time()*1000)}, signed=True)
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol, "side": side, "type": "MARKET",
            "quantity": quantity, "timestamp": ts
        }
        return await self._request("POST", "/fapi/v1/order", params, signed=True)

# ==========================================================
# ۱۰. Telegram Manager (ترکیب ربات ۲)
# ==========================================================
class TelegramManager:
    def __init__(self, token, chat_id):
        self.bot = Bot(token=token)
        self.chat_id = int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id
        self.sent_alerts = {}

    async def send(self, text, reply_markup=None, retries=3):
        for i in range(retries):
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
                return True
            except Exception as e:
                if i == retries - 1: LOGGER.error(f"TG error: {e}"); return False
                await asyncio.sleep(2 ** i)

    async def notify_signal(self, signal, symbol, interval, ai_prob, btc_trend):
        tv = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}"
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("📊 TradingView", url=tv)]])
        msg = (
            f"🎯 **سیگنال ترکیبی (AI + Technical)**\n\n"
            f"⚙️ **استراتژی:** `{signal['strategy']}`\n"
            f"🪙 **نماد:** `#{symbol}` | **تایم‌فریم:** `{interval}`\n"
            f"🚦 **جهت:** {signal['direction']}\n"
            f"📈 **روند بازار:** `{signal['trend']}`\n"
            f"🌐 **روند BTC:** `{btc_trend}`\n"
            f"🧠 **احتمال برد AI:** `{ai_prob:.0%}`\n"
            f"📊 **RSI / ADX:** `{signal['rsi']}` / `{signal['adx']}`\n\n"
            f"📍 **ورود:** `{signal['entry_price']}`\n"
            f"🛡️ **حد ضرر:** `{signal['stop_loss']}` (`{signal['sl_percent']}%`)\n\n"
            f"🎯 **اهداف سود:**\n"
            f"🔹 TP1 (1:2): `{signal['tp1']}`\n"
            f"🔹 TP2 (1:5): `{signal['tp2']}`\n"
            f"🔹 TP3 (1:7): `{signal['tp3']}`"
        )
        await self.send(msg, reply_markup=kb)

    async def notify_fill(self, symbol, direction, entry, size, sl, tp1, tp2, tp3, paper):
        p = "📄 PAPER" if paper else "💰 REAL"
        await self.send(
            f"✅ **معامله باز شد ({p})**\n\n"
            f"🪙 `#{symbol}` | {direction}\n"
            f"📍 Entry: `{entry}`\n"
            f"📊 Size: `{size}` USDT\n"
            f"🛡️ SL: `{sl}`\n"
            f"🎯 TP1: `{tp1}` | TP2: `{tp2}` | TP3: `{tp3}`"
        )

    async def notify_close(self, symbol, outcome, pnl_pct=0):
        icon = "💰" if outcome == 1 else "❌"
        text = "سود" if outcome == 1 else "ضرر"
        await self.send(f"{icon} **معامله بسته شد ({text})**\n\n🪙 `#{symbol}` | PnL: `{pnl_pct:.2f}%`")

    async def command_listener(self):
        last_id = 0
        while True:
            try:
                updates = await self.bot.get_updates(offset=last_id + 1, timeout=5)
                for u in updates:
                    last_id = u.update_id
                    if u.message and u.message.text:
                        cmd = u.message.text.strip().split("@")[0].lower()
                        cid = u.message.chat_id
                        if cmd == "/stats":
                            rows = await db_execute("SELECT key, value FROM bot_stats")
                            stats = {r[0]: r[1] for r in rows}
                            total = stats.get("total_signals", 0)
                            wr = round((stats.get("tp1_hits",0)+stats.get("tp2_hits",0)+stats.get("tp3_hits",0))/total*100,1) if total else 0
                            await self.send(
                                f"📊 **آمار ربات**\n\n"
                                f"🔢 کل سیگنال‌ها: `{total}`\n"
                                f"🎯 TP1: `{stats.get('tp1_hits',0)}` | TP2: `{stats.get('tp2_hits',0)}` | TP3: `{stats.get('tp3_hits',0)}`\n"
                                f"❌ SL: `{stats.get('sl_hits',0)}`\n"
                                f"🧠 AI Trades: `{stats.get('ai_trades',0)}`\n"
                                f"🏆 Win Rate: `{wr}%`", chat_id=cid
                            )
                        elif cmd == "/active":
                            df = await db_fetch_df("SELECT * FROM active_trades")
                            if df.empty:
                                await self.send("ℹ️ هیچ پوزیشن فعالی وجود ندارد.", chat_id=cid)
                            else:
                                lines = "\n".join([f"🔹 `{r['symbol']}` ({r['direction']}) @ `{r['entry_price']}`" for _, r in df.iterrows()])
                                await self.send(f"📌 **پوزیشن‌های فعال ({len(df)}):**\n\n{lines}", chat_id=cid)
                        elif cmd == "/help":
                            await self.send(
                                "🤖 **منوی کنترل**\n\n"
                                "▫️ `/stats` — آمار\n"
                                "▫️ `/active` — پوزیشن‌ها\n"
                                "▫️ `/help` — راهنما", chat_id=cid
                            )
            except Exception as e:
                LOGGER.error(f"Command listener: {e}")
            await asyncio.sleep(2)

# ==========================================================
# ۱۱. موتور اصلی یکپارچه
# ==========================================================
class UnifiedTradingBot:
    def __init__(self):
        self.ai = AIEngine()
        self.risk = RiskManager()
        self.trader = BinanceTrader(BINANCE_API_KEY, BINANCE_API_SECRET, PAPER_TRADING)
        self.tg = TelegramManager(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.btc_trend = "NEUTRAL"
        self.btc_pause_until = 0
        self.symbol_cache = {"symbols": [], "last_update": 0}
        self.lock = asyncio.Lock()

    async def start(self):
        await init_database()
        await self.trader.start()
        asyncio.create_task(self.tg.command_listener())
        asyncio.create_task(self.position_tracker())
        LOGGER.info("🚀 ربات یکپارچه (AI + Signal + Execution) آماده است.")

    async def get_symbols(self, session):
        now = time.time()
        if now - self.symbol_cache["last_update"] > 300 or not self.symbol_cache["symbols"]:
            try:
                url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                async with session.get(url, timeout=10) as r:
                    if r.status == 200:
                        data = await r.json()
                        btc_p = next((float(x["lastPrice"]) for x in data if x.get("symbol") == "BTCUSDT"), 60000)
                        min_vol = MIN_BTC_VOLUME * btc_p
                        syms = [x["symbol"] for x in data if x["symbol"].endswith("USDT") and float(x.get("quoteVolume",0)) >= min_vol]
                        self.symbol_cache = {"symbols": syms, "last_update": now}
                        LOGGER.info(f"✅ {len(syms)} symbol loaded.")
            except Exception as e:
                LOGGER.error(f"Symbol fetch: {e}")
                self.symbol_cache["symbols"] = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"]
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
                    LOGGER.warning(f"⚠️ BTC Volatility Spike {change:.1f}% — 30m pause.")
        except Exception as e:
            LOGGER.error(f"BTC update: {e}")

    async def execute_signal(self, session, symbol, interval, signal, ai_prob):
        # 1. Order Book
        bids, asks = await fetch_order_book(session, symbol)
        ob = OrderBookAnalyzer.analyze(bids, asks)

        # 2. Prepare AI features
        features = {
            "rsi": signal["rsi"], "spread_pct": ob["spread_pct"],
            "vol_ratio": signal["vol_ratio"], "lower_wick_ratio": signal["lower_wick_ratio"],
            "upper_wick_ratio": signal["upper_wick_ratio"], "trend_code": signal["trend_code"],
            "adx": signal["adx"], "plus_di": signal["plus_di"], "minus_di": signal["minus_di"],
            "price_to_sma7_ratio": signal["price_to_sma7_ratio"], "atr_pct": signal["atr_pct"],
            "orderbook_imbalance": ob["imbalance"]
        }

        # 3. AI Filter
        prob = await self.ai.predict(features)
        if prob < 0.55 and self.ai.is_trained:
            LOGGER.info(f"🧠 AI rejected {symbol} ({prob:.2f})")
            return

        # 4. Check balance
        balance = await self.trader.get_balance()
        if balance <= 0:
            LOGGER.warning("⚠️ Balance zero.")
            return

        # 5. Position Sizing
        entry = signal["entry_price"]
        sl = signal["stop_loss"]
        notional, margin = self.risk.size(balance, entry, sl)
        if notional <= 0:
            LOGGER.warning("⚠️ Position size zero.")
            return

        # 6. Record features for future training
        cols = "symbol, rsi, spread_pct, vol_ratio, lower_wick_ratio, upper_wick_ratio, trend_code, adx, plus_di, minus_di, price_to_sma7_ratio, atr_pct, orderbook_imbalance"
        vals = (symbol, features["rsi"], features["spread_pct"], features["vol_ratio"], features["lower_wick_ratio"],
                features["upper_wick_ratio"], features["trend_code"], features["adx"], features["plus_di"],
                features["minus_di"], features["price_to_sma7_ratio"], features["atr_pct"], features["orderbook_imbalance"])
        await db_execute(f"INSERT INTO trade_features ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", vals)
        feat_id = (await db_execute("SELECT last_insert_rowid()"))[0][0]

        # 7. Execute trade
        side = "BUY" if "LONG" in signal["direction"] else "SELL"
        # Convert notional to quantity (approximate)
        qty = round(notional / entry, 4)
        res = await self.trader.place_market_order(symbol, side, qty, LEVERAGE)

        if res:
            trade_id = f"{symbol}_{interval}_{int(time.time())}"
            tp1, tp2, tp3 = signal["tp1"], signal["tp2"], signal["tp3"]
            await db_execute(
                "INSERT INTO active_trades (trade_id, symbol, direction, entry_price, stop_loss, take_profit, highest_price, entry_time, feature_id, position_size, leverage, is_paper) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (trade_id, symbol, signal["direction"], entry, sl, tp3, entry, time.time(), feat_id, notional, LEVERAGE, 1 if PAPER_TRADING else 0)
            )
            await db_execute("INSERT OR REPLACE INTO bot_stats (key, value) VALUES ('ai_trades', COALESCE((SELECT value FROM bot_stats WHERE key='ai_trades'), 0) + 1)")
            await self.tg.notify_fill(symbol, signal["direction"], entry, notional, sl, tp1, tp2, tp3, PAPER_TRADING)
            LOGGER.info(f"✅ Trade opened: {symbol} | AI Prob: {prob:.2f} | Size: {notional}")

    async def position_tracker(self):
        while True:
            try:
                df = await db_fetch_df("SELECT * FROM active_trades")
                if df.empty:
                    await asyncio.sleep(10); continue

                async with aiohttp.ClientSession() as session:
                    for _, row in df.iterrows():
                        tid = row["trade_id"]; sym = row["symbol"]
                        klines = await fetch_klines(session, sym, "15m")
                        if not klines: continue
                        price = float(klines[-1][4])
                        direction = row["direction"]
                        entry = float(row["entry_price"])
                        sl = float(row["stop_loss"])
                        tp = float(row["take_profit"])
                        highest = float(row["highest_price"])
                        fid = int(row["feature_id"])

                        # Update highest
                        if price > highest:
                            await db_execute("UPDATE active_trades SET highest_price = ? WHERE trade_id = ?", (price, tid))
                            # Risk-free after 2%
                            if (price - entry) / entry >= 0.02 and "LONG" in direction:
                                new_sl = max(sl, entry)
                                await db_execute("UPDATE active_trades SET stop_loss = ? WHERE trade_id = ?", (new_sl, tid))
                            elif (entry - price) / entry >= 0.02 and "SHORT" in direction:
                                new_sl = min(sl, entry)
                                await db_execute("UPDATE active_trades SET stop_loss = ? WHERE trade_id = ?", (new_sl, tid))

                        # Check exits
                        closed = False; outcome = None
                        if "LONG" in direction:
                            if price >= tp:
                                closed = True; outcome = 1
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'tp3_hits'")
                            elif price <= sl:
                                closed = True; outcome = 0
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'sl_hits'")
                        else:
                            if price <= tp:
                                closed = True; outcome = 1
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'tp3_hits'")
                            elif price >= sl:
                                closed = True; outcome = 0
                                await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'sl_hits'")

                        if closed:
                            await db_execute("UPDATE trade_features SET outcome = ? WHERE id = ?", (outcome, fid))
                            await db_execute("DELETE FROM active_trades WHERE trade_id = ?", (tid,))
                            pnl = ((price - entry) / entry * 100) if "LONG" in direction else ((entry - price) / entry * 100)
                            await self.tg.notify_close(sym, outcome, pnl)
                            asyncio.create_task(self.ai.retrain())
            except Exception as e:
                LOGGER.error(f"Tracker error: {e}")
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
                        await self.update_btc(session); btc_counter = 0

                    if time.time() < self.btc_pause_until:
                        await asyncio.sleep(5); continue

                    for symbol in symbols:
                        k4h = await fetch_klines(session, symbol, "4h")
                        k1d = await fetch_klines(session, symbol, "1d")
                        htf_s, htf_r = htf_sr(k4h, k1d)

                        for interval in TIMEFRAMES:
                            klines = await fetch_klines(session, symbol, interval)
                            if not klines: continue

                            sig = analyze_signal(klines, symbol, interval, htf_s, htf_r, MAX_SL_PERCENT)
                            if not sig: continue

                            # BTC Filter
                            if symbol != "BTCUSDT":
                                if "LONG" in sig["direction"] and self.btc_trend == "BEARISH": continue
                                if "SHORT" in sig["direction"] and self.btc_trend == "BULLISH": continue

                            alert_id = f"{symbol}_{interval}_{klines[-2][0]}_{sig['direction']}"
                            exists = await db_execute("SELECT 1 FROM signal_history WHERE alert_id = ?", (alert_id,))
                            if exists: continue

                            await db_execute(
                                "INSERT INTO signal_history (alert_id, symbol, interval, direction, strategy, entry_price, stop_loss, tp1, tp2, tp3, sl_percent, rsi, adx, trend) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (alert_id, symbol, interval, sig["direction"], sig["strategy"], sig["entry_price"], sig["stop_loss"], sig["tp1"], sig["tp2"], sig["tp3"], sig["sl_percent"], sig["rsi"], sig["adx"], sig["trend"])
                            )
                            await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'total_signals'")

                            # AI + Execution
                            await self.execute_signal(session, symbol, interval, sig, 0.0)
                            await asyncio.sleep(0.02)

                    symbols = await self.get_symbols(session)
                    await asyncio.sleep(5)
                except Exception as e:
                    LOGGER.error(f"Scanner: {e}")
                    await asyncio.sleep(15)

# ==========================================================
# ۱۲. Web Server & Main
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
