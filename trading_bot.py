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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from aiohttp import web

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()

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


def _sync_execute(query: str, params: tuple = ()):
    with sqlite3.connect(DB_NAME, timeout=20.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.fetchall()

def _sync_fetch_df(query: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_NAME, timeout=20.0) as conn:
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
            sl_percent REAL, rsi REAL, adx REAL, atr REAL, trend TEXT,
            ai_prob REAL, ai_confidence TEXT,
            ob_imbalance REAL, ob_slippage REAL, ob_stop_hunt REAL,
            ob_iceberg_bids INTEGER, ob_iceberg_asks INTEGER,
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
    for k in ["total_signals","tp1_hits","tp2_hits","tp3_hits","sl_hits","ai_trades","feedback_good","feedback_bad"]:
        await db_execute("INSERT OR IGNORE INTO bot_stats (key, value) VALUES (?, 0)", (k,))
    LOGGER.info("Database initialized.")


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

async def fetch_order_book(session, symbol, limit=50):
    try:
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={limit}"
        async with session.get(url, timeout=5) as r:
            if r.status == 200:
                data = await r.json()
                return data.get("bids", []), data.get("asks", [])
    except: pass
    return [], []


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

def detect_hidden_divergence(highs, lows, closes):
    if len(closes) < 20: return False, False
    price_lows = lows[-10:]
    rsi_vals = [calc_rsi(closes[:len(closes)-10+i+1]) for i in range(10)]
    if len(rsi_vals) < 10: return False, False
    hidden_bull = (lows[-1] > min(price_lows)) and (rsi_vals[-1] < min(rsi_vals))
    hidden_bear = (highs[-1] < max(highs[-10:])) and (rsi_vals[-1] > max(rsi_vals))
    return hidden_bull, hidden_bear


class OrderBookAnalyzer:
    @staticmethod
    def analyze(bids, asks, entry_price=0.0, sl_price=0.0):
        if not bids or not asks:
            return {"imbalance": 0.0, "spread_pct": 0.0, "slippage": 0.0,
                    "stop_hunt_risk": 0.0, "iceberg_bids": 0, "iceberg_asks": 0}
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
            "iceberg_asks": int(iceberg_asks)
        }

    @staticmethod
    async def get_depth_imbalance(session, symbol):
        bids, asks = await fetch_order_book(session, symbol, 20)
        if not bids or not asks: return 0.0
        bv = sum(float(b[1]) for b in bids[:20])
        av = sum(float(a[1]) for a in asks[:20])
        if bv + av == 0: return 0.0
        return (bv - av) / (bv + av)


class ChartGenerator:
    @staticmethod
    def generate_chart_image(df: pd.DataFrame, symbol: str, signal: dict) -> bytes:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.plot(df.index, df['close'], label="Price", color="#1f77b4", linewidth=1.5)
        entry = signal.get('entry_price', signal.get('entry', 0))
        sl = signal.get('stop_loss', signal.get('sl', 0))
        tp1 = signal.get('tp1', 0)
        if entry: ax.axhline(y=entry, color='blue', linestyle='--', linewidth=1, label=f"Entry: {entry}")
        if sl: ax.axhline(y=sl, color='red', linestyle='--', linewidth=1, label=f"SL: {round(sl, 4)}")
        if tp1: ax.axhline(y=tp1, color='green', linestyle='--', linewidth=1, label=f"TP1: {round(tp1, 4)}")
        ax.set_title(f"{symbol} - {signal.get('direction', 'SIGNAL')} Chart", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="upper left")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

def klines_to_df(klines):
    df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', '_1', '_2', '_3', '_4', '_5', '_6'])
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)


class ChartImageAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key) if api_key and GEMINI_AVAILABLE else None

    async def analyze_chart_image(self, image_bytes: bytes) -> dict:
        if not self.client:
            return None
        try:
            prompt = """
            This is a trading chart image. Analyze it carefully and return ONLY the following JSON format:
            {
                "pattern_detected": "pattern name like Head and Shoulders, Bullish Flag, SMC Sweep, or None",
                "signal": "LONG or SHORT or NEUTRAL",
                "confidence_score": number between 0 and 100,
                "analysis_summary": "one short sentence about the analysis"
            }
            """
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt
                ]
            )
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            LOGGER.error(f"Gemini analysis error: {e}")
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
            LOGGER.error(f"Save error: {e}")

    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.is_trained = True
                LOGGER.info("AI model loaded.")
            except Exception as e:
                LOGGER.error(f"Load error: {e}")

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
                LOGGER.info(f"Not enough data for training ({len(df)}).")
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
            LOGGER.info(f"AI retrained on {len(df)} trades.")
            return True
        except Exception as e:
            LOGGER.error(f"Retrain error: {e}"); return False
        finally:
            self._training = False


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
            LOGGER.info(f"PAPER ORDER | {side} {quantity} {symbol} @ x{leverage}")
            return {"orderId": f"PAPER_{int(time.time()*1000)}", "status": "FILLED"}
        await self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage, "timestamp": int(time.time()*1000)}, signed=True)
        ts = int(time.time() * 1000)
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity, "timestamp": ts}
        return await self._request("POST", "/fapi/v1/order", params, signed=True)


def analyze_signal(klines, symbol, interval, htf_s, htf_r, ob_imbalance=0.0, max_sl=2.0):
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

    prev_high = H[-2] if len(H) >= 2 else ch
    prev_low = L[-2] if len(L) >= 2 else cl

    # --- Bot 2/3 Strategies ---
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

    # --- Bot 4 Additional Strategies ---
    hidden_bull, hidden_bear = detect_hidden_divergence(H, L, C)
    hidden_long = hidden_bull and vol_spike
    hidden_short = hidden_bear and vol_spike

    adv_candle_long = (lw >= 3.0 * body) and (cc > prev_high) and vol_spike
    adv_candle_short = (uw >= 3.0 * body) and (cc < prev_low) and vol_spike

    resistance_20 = max(H[-20:-1]) if len(H) >= 21 else rh
    support_20 = min(L[-20:-1]) if len(L) >= 21 else rl
    atr_bo_long = (cc > resistance_20 + 0.5 * atr) and vol_spike
    atr_bo_short = (cc < support_20 - 0.5 * atr) and vol_spike

    eq_high = False; eq_low = False
    if len(H) >= 11 and len(L) >= 11:
        eq_high = abs(max(H[-5:-1]) - max(H[-10:-5])) < (atr * 0.1)
        eq_low = abs(min(L[-5:-1]) - min(L[-10:-5])) < (atr * 0.1)
    eq_sweep_long = eq_low and lw >= 2.5 * body
    eq_sweep_short = eq_high and uw >= 2.5 * body

    ob_long = ob_imbalance > 0.30
    ob_short = ob_imbalance < -0.30

    # Build strategy lists
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
            return {
                "strategy": " + ".join(strategies), "direction": direction,
                "entry_price": entry, "stop_loss": round(sl, 5), "sl_percent": round(sl_pct, 2),
                "tp1": round(entry + (risk * 2), 5), "tp2": round(entry + (risk * 5), 5), "tp3": round(entry + (risk * 7), 5),
                "rsi": rsi, "adx": adx, "atr": atr, "trend": trend, "sma7": round(sma7, 5),
                "live": live, "vol_spike": vol_spike,
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


async def send_telegram_msg(session: aiohttp.ClientSession, text: str, photo_bytes: bytes = None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if photo_bytes:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        data = aiohttp.FormData()
        data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
        data.add_field("caption", text)
        data.add_field("parse_mode", "Markdown")
        data.add_field("photo", photo_bytes, filename="chart.png", content_type="image/png")
        try:
            async with session.post(url, data=data, timeout=10) as resp:
                if resp.status != 200:
                    LOGGER.error(f"Telegram photo send error: {await resp.text()}")
        except Exception as e:
            LOGGER.error(f"Telegram photo error: {e}")
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        try:
            async with session.post(url, json=payload, timeout=5) as resp:
                if resp.status != 200:
                    LOGGER.error(f"Telegram msg error: {await resp.text()}")
        except Exception as e:
            LOGGER.error(f"Telegram msg error: {e}")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = await db_execute("SELECT key, value FROM bot_stats")
    stats = {r[0]: r[1] for r in rows}
    total = stats.get("total_signals", 0)
    wr = round((stats.get("tp1_hits",0)+stats.get("tp2_hits",0)+stats.get("tp3_hits",0))/total*100,1) if total else 0
    msg = (
        f"📊 *Bot Statistics*\n\n"
        f"🔢 Total Signals: `{total}`\n"
        f"🎯 TP1: `{stats.get('tp1_hits',0)}` | TP2: `{stats.get('tp2_hits',0)}` | TP3: `{stats.get('tp3_hits',0)}`\n"
        f"❌ SL: `{stats.get('sl_hits',0)}`\n"
        f"🧠 AI Trades: `{stats.get('ai_trades',0)}`\n"
        f"👍 Good Feedback: `{stats.get('feedback_good',0)}`\n"
        f"👎 Bad Feedback: `{stats.get('feedback_bad',0)}`\n"
        f"🏆 Win Rate: `{wr}%`"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def cmd_active(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = await db_fetch_df("SELECT * FROM active_trades")
    if df.empty:
        await update.message.reply_text("ℹ️ No active positions.", parse_mode=ParseMode.MARKDOWN)
    else:
        lines = "\n".join([f"🔹 `#{r['symbol']}` ({r['direction']}) @ `{r['entry_price']}`" for _, r in df.iterrows()])
        await update.message.reply_text(f"📌 *Active Trades ({len(df)}):*\n\n{lines}", parse_mode=ParseMode.MARKDOWN)

async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get BTC trend from global (we'll store it in a global or module var)
    # For now just show basic stats
    df = await db_fetch_df("SELECT * FROM active_trades")
    sa = await db_fetch_df("SELECT * FROM signal_history WHERE datetime(timestamp) > datetime('now', '-1 day')")
    msg = (
        f"🔍 *Debug Info:*\n\n"
        f"📊 Active Trades: `{len(df)}`\n"
        f"📨 Signals (24h): `{len(sa)}`\n"
        f"💾 DB: `{DB_NAME}`"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"🤖 *Control Menu*\n\n"
        f"▫️ `/stats` — Statistics\n"
        f"▫️ `/active` — Active positions\n"
        f"▫️ `/debug` — Debug info\n"
        f"▫️ `/help` — Help\n\n"
        f"📸 Send a chart photo for Gemini AI analysis!"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📸 Chart image received. Sending to Gemini AI for analysis...")
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
    except Exception as e:
        await update.message.reply_text(f"❌ Error downloading photo: {e}")
        return

    analyzer = ChartImageAnalyzer(GEMINI_API_KEY)
    result = await analyzer.analyze_chart_image(bytes(photo_bytes))

    if result:
        sig = result.get('signal', 'NEUTRAL')
        sig_emoji = "🟢" if sig == "LONG" else ("🔴" if sig == "SHORT" else "⚪")
        msg = (
            f"🔍 *Gemini AI Analysis Result:*\n\n"
            f"📌 Pattern: `{result.get('pattern_detected', 'None')}`\n"
            f"📊 Signal: `{sig}` {sig_emoji}\n"
            f"🎯 Confidence: `{result.get('confidence_score', 0)}%`\n"
            f"💡 Summary: {result.get('analysis_summary', 'No summary')}"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        # Save to DB
        await db_execute(
            "INSERT INTO gemini_analysis (symbol, pattern_detected, signal, confidence_score, analysis_summary) VALUES (?, ?, ?, ?, ?)",
            ("USER_UPLOAD", result.get('pattern_detected'), sig, result.get('confidence_score', 0), result.get('analysis_summary'))
        )
    else:
        await update.message.reply_text("❌ Gemini could not analyze the image.")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cq = update.callback_query
    data = cq.data or ""
    parts = data.split("|")
    if len(parts) == 2:
        fb_type, alert_id = parts[0], parts[1]
        if fb_type in ("good", "bad"):
            await db_execute("UPDATE signal_history SET feedback = ? WHERE alert_id = ?", (fb_type, alert_id))
            await db_execute(f"UPDATE bot_stats SET value = value + 1 WHERE key = 'feedback_{fb_type}'")
            label = "✅ Good signal recorded (AI learning)" if fb_type == "good" else "❌ Error recorded"
            await cq.answer(text=label, show_alert=False)
            return
    await cq.answer(text="Unknown action", show_alert=False)


class UnifiedTradingBot:
    def __init__(self):
        self.ai = AIEngine()
        self.risk = RiskManager()
        self.trader = BinanceTrader(BINANCE_API_KEY, BINANCE_API_SECRET, PAPER_TRADING)
        self.btc_trend = "NEUTRAL"
        self.btc_pause_until = 0
        self.symbol_cache = {"symbols": [], "last_update": 0}
        self.recent_signals = {}

    async def start(self):
        await self.trader.start()
        asyncio.create_task(self.position_tracker())
        mode = "SIGNAL ONLY (AI Learning Mode)" if not EXECUTE_TRADES else ("PAPER TRADING" if PAPER_TRADING else "LIVE TRADING")
        LOGGER.info(f"Unified Bot ready. Mode: {mode}")

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
                        LOGGER.info(f"{len(syms)} symbols loaded.")
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
                    LOGGER.warning(f"BTC Volatility Spike {change:.1f}% — 30m pause.")
        except Exception as e:
            LOGGER.error(f"BTC update: {e}")

    async def execute_signal(self, session, symbol, interval, signal):
        # Cooldown check (Bot 4 style)
        current_time = time.time()
        if symbol in self.recent_signals and (current_time - self.recent_signals[symbol]) < 900:
            return
        self.recent_signals[symbol] = current_time

        # Order Book
        bids, asks = await fetch_order_book(session, symbol)
        ob = OrderBookAnalyzer.analyze(bids, asks, signal["entry_price"], signal["stop_loss"])

        # AI features
        features = {
            "rsi": signal["rsi"], "spread_pct": ob["spread_pct"],
            "vol_ratio": signal["vol_ratio"], "lower_wick_ratio": signal["lower_wick_ratio"],
            "upper_wick_ratio": signal["upper_wick_ratio"], "trend_code": signal["trend_code"],
            "adx": signal["adx"], "plus_di": signal["plus_di"], "minus_di": signal["minus_di"],
            "price_to_sma7_ratio": signal["price_to_sma7_ratio"], "atr_pct": signal["atr_pct"],
            "orderbook_imbalance": ob["imbalance"]
        }

        # AI Filter
        prob = await self.ai.predict(features)
        conf_label = self.ai.confidence_label(prob)
        if prob < 0.55 and self.ai.is_trained:
            LOGGER.info(f"AI rejected {symbol} ({prob:.2f} — {conf_label})")
            return

        # Alert ID
        alert_id = f"{symbol}_{interval}_{int(time.time())}"

        # Generate chart
        df_chart = klines_to_df(await fetch_klines(session, symbol, interval) or [])
        chart_bytes = None
        if not df_chart.empty:
            chart_bytes = ChartGenerator.generate_chart_image(df_chart, symbol, signal)

        # Telegram notification with chart
        dir_emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
        conf_emoji = "🔥" if prob >= 0.75 else ("✅" if prob >= 0.60 else ("⚠️" if prob >= 0.55 else "❓"))
        msg = (
            f"🚨 *NEW TRADING SIGNAL* 🚨\n\n"
            f"🪙 *Symbol:* `#{symbol}`\n"
            f"📊 *Direction:* {signal['direction']} {dir_emoji}\n"
            f"🎯 *Strategy:* {signal['strategy']} ({interval})\n"
            f"{conf_emoji} *AI Score:* `{prob:.1%}` Confidence\n"
            f"⏱️ *Timeframe:* {interval}\n\n"
            f"💵 *Entry Price:* `{signal['entry_price']}`\n"
            f"🛡️ *Stop Loss:* `{signal['stop_loss']}` (`{signal['sl_percent']}%`)\n\n"
            f"🎯 *Take Profit Targets:*\n"
            f"🔹 *TP1:* `{signal['tp1']}`\n"
            f"🔹 *TP2:* `{signal['tp2']}`\n"
            f"🔹 *TP3:* `{signal['tp3']}`\n\n"
            f"📉 *RSI:* `{signal['rsi']}` | *Trend:* `{signal['trend']}`\n"
            f"🌐 *BTC Trend:* `{self.btc_trend}`\n\n"
            f"📖 *Order Book Microstructure:*\n"
            f"• Imbalance Ratio: `{ob['imbalance']:.2f}`\n"
            f"• Slippage: `{ob['slippage']:.2f}%`\n"
            f"• Stop Hunt Risk: `{ob['stop_hunt_risk']}`\n"
            f"• Iceberg Bids/Asks: `{ob['iceberg_bids']}` / `{ob['iceberg_asks']}`"
        )

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 TradingView", url=f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}")],
            [
                InlineKeyboardButton("✅ Good Signal", callback_data=f"good|{alert_id}"),
                InlineKeyboardButton("❌ Bad Signal", callback_data=f"bad|{alert_id}")
            ]
        ])

        # Send via direct API for photo support
        if chart_bytes:
            await send_telegram_msg(session, msg, photo_bytes=chart_bytes)
        else:
            await send_telegram_msg(session, msg)

        # Record features for AI learning (even without execution)
        cols = "symbol, rsi, spread_pct, vol_ratio, lower_wick_ratio, upper_wick_ratio, trend_code, adx, plus_di, minus_di, price_to_sma7_ratio, atr_pct, orderbook_imbalance"
        vals = (symbol, features["rsi"], features["spread_pct"], features["vol_ratio"], features["lower_wick_ratio"],
                features["upper_wick_ratio"], features["trend_code"], features["adx"], features["plus_di"],
                features["minus_di"], features["price_to_sma7_ratio"], features["atr_pct"], features["orderbook_imbalance"])
        await db_execute(f"INSERT INTO trade_features ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", vals)
        feat_id = (await db_execute("SELECT last_insert_rowid()"))[0][0]

        # Update signal_history with AI data
        await db_execute(
            "UPDATE signal_history SET ai_prob = ?, ai_confidence = ?, ob_imbalance = ?, ob_slippage = ?, ob_stop_hunt = ?, ob_iceberg_bids = ?, ob_iceberg_asks = ? WHERE alert_id = ?",
            (prob, conf_label, ob["imbalance"], ob["slippage"], ob["stop_hunt_risk"], ob["iceberg_bids"], ob["iceberg_asks"], alert_id)
        )

        if not EXECUTE_TRADES:
            LOGGER.info(f"📡 SIGNAL ONLY (trading OFF): {symbol} | AI: {prob:.1%} | Features recorded for learning.")
            return

        # Balance & sizing
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

        # Execute trade
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
                "UPDATE signal_history SET executed = 1 WHERE alert_id = ?",
                (alert_id,)
            )
            LOGGER.info(f"Trade opened: {symbol} | AI: {prob:.1%} | Size: {notional}")

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

                        if price > highest:
                            await db_execute("UPDATE active_trades SET highest_price = ? WHERE trade_id = ?", (price, tid))
                            if (price - entry) / entry >= 0.02 and direction == "LONG":
                                new_sl = max(sl, entry)
                                await db_execute("UPDATE active_trades SET stop_loss = ? WHERE trade_id = ?", (new_sl, tid))
                            elif (entry - price) / entry >= 0.02 and direction == "SHORT":
                                new_sl = min(sl, entry)
                                await db_execute("UPDATE active_trades SET stop_loss = ? WHERE trade_id = ?", (new_sl, tid))

                        closed = False; outcome = None
                        if direction == "LONG":
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
                            pnl = ((price - entry) / entry * 100) if direction == "LONG" else ((entry - price) / entry * 100)
                            await send_telegram_msg(session, f"{'💰' if outcome==1 else '❌'} *Trade Closed ({'Profit' if outcome==1 else 'Loss'})*\n\n🪙 `#{sym}` | PnL: `{pnl:.2f}%`")
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

                            ob_imbalance = await OrderBookAnalyzer.get_depth_imbalance(session, symbol)
                            sig = analyze_signal(klines, symbol, interval, htf_s, htf_r, ob_imbalance, MAX_SL_PERCENT)
                            if not sig: continue

                            if symbol != "BTCUSDT":
                                if sig["direction"] == "LONG" and self.btc_trend == "BEARISH": continue
                                if sig["direction"] == "SHORT" and self.btc_trend == "BULLISH": continue

                            alert_id = f"{symbol}_{interval}_{int(klines[-2][0])}_{sig['direction']}"
                            exists = await db_execute("SELECT 1 FROM signal_history WHERE alert_id = ?", (alert_id,))
                            if exists: continue

                            await db_execute(
                                "INSERT INTO signal_history (alert_id, symbol, interval, direction, strategy, entry_price, stop_loss, tp1, tp2, tp3, sl_percent, rsi, adx, atr, trend) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (alert_id, symbol, interval, sig["direction"], sig["strategy"], sig["entry_price"], sig["stop_loss"], sig["tp1"], sig["tp2"], sig["tp3"], sig["sl_percent"], sig["rsi"], sig["adx"], sig["atr"], sig["trend"])
                            )
                            await db_execute("UPDATE bot_stats SET value = value + 1 WHERE key = 'total_signals'")
                            await self.execute_signal(session, symbol, interval, sig)
                            await asyncio.sleep(0.02)

                    symbols = await self.get_symbols(session)
                    await asyncio.sleep(5)
                except Exception as e:
                    LOGGER.error(f"Scanner: {e}")
                    await asyncio.sleep(15)


async def health(request):
    return web.Response(text="Unified Bot Running", status=200)

async def main():
    await init_database()

    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("active", cmd_active))
    application.add_handler(CommandHandler("debug", cmd_debug))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("start", cmd_help))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CallbackQueryHandler(handle_callback))

    bot = UnifiedTradingBot()
    await bot.start()
    asyncio.create_task(bot.scanner_loop())

    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    LOGGER.info("🚀 All systems operational. Bot is running.")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())