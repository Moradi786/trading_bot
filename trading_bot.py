import os
import asyncio
import sqlite3
import math
import logging
import json
from decimal import Decimal
import aiohttp
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from dotenv import load_dotenv
from google import genai
from google.genai import types

# بارگذاری متغیرها از فایل .env
load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
DB_NAME = "alerts.db"
RISK_PER_TRADE = 0.02
LEVERAGE = 10
AI_MIN_SAMPLES = 50

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EXCHANGES = {
    "binance": {
        "url": "https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=100",
        "depth_url": "https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=20",
        "exchange_info": "https://fapi.binance.com/fapi/v1/exchangeInfo"
    },
    "okx": {
        "url": "https://www.okx.com/api/v5/market/history-candles?instId={okx_symbol}-SWAP&bar={interval}&limit=100"
    }
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================================
# 2. TELEGRAM NOTIFIER SYSTEM
# ==========================================
async def send_telegram_msg(session: aiohttp.ClientSession, text: str):
    """ارسال مستقیم پیام به کانال/گروه تلگرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        async with session.post(url, json=payload, timeout=5) as resp:
            if resp.status != 200:
                logging.error(f"خطا در ارسال پیام تلگرام: {await resp.text()}")
    except Exception as e:
        logging.error(f"خطای ارتباط با تلگرام: {e}")

# ==========================================
# 3. DATABASE (WAL MODE)
# ==========================================
def sync_execute(query: str, params: tuple = ()):
    with sqlite3.connect(DB_NAME, timeout=20.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.fetchall()

def init_db():
    sync_execute("""
    CREATE TABLE IF NOT EXISTS active_trades (
        id TEXT PRIMARY KEY,
        symbol TEXT,
        direction TEXT,
        entry_price REAL,
        stop_loss REAL,
        tp1 REAL, tp2 REAL, tp3 REAL,
        quantity REAL, remaining_qty REAL,
        position_size_usdt REAL,
        tp1_hit INTEGER DEFAULT 0, tp2_hit INTEGER DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    sync_execute("""
    CREATE TABLE IF NOT EXISTS trade_history (
        id TEXT PRIMARY KEY,
        symbol TEXT, direction TEXT,
        entry_price REAL, close_price REAL,
        pnl REAL, pnl_pct REAL, win INTEGER, rsi REAL, adx REAL, atr REAL, imbalance REAL, close_reason TEXT
    );
    """)

async def db_execute(query: str, params: tuple = ()):
    return await asyncio.to_thread(sync_execute, query, params)

# ==========================================
# 4. EXCHANGE & PRECISION MANAGER
# ==========================================
class ExchangeManager:
    def __init__(self):
        self.symbol_info = {}

    async def load_exchange_info(self, session: aiohttp.ClientSession):
        try:
            async with session.get(EXCHANGES["binance"]["exchange_info"]) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for s in data["symbols"]:
                        symbol = s["symbol"]
                        step_size = "1"
                        for f in s["filters"]:
                            if f["filterType"] == "LOT_SIZE":
                                step_size = f["stepSize"]
                        self.symbol_info[symbol] = {"step_size": float(step_size)}
        except Exception as e:
            logging.error(f"خطا در دریافت قوانین صرافی: {e}")

    def format_quantity(self, symbol: str, quantity: float) -> float:
        info = self.symbol_info.get(symbol, {"step_size": 0.001})
        step = Decimal(str(info["step_size"]))
        qty_dec = Decimal(str(quantity))
        return float((qty_dec // step) * step)

    async def fetch_klines(self, session: aiohttp.ClientSession, symbol: str, interval: str = "15m"):
        url = EXCHANGES["binance"]["url"].format(symbol=symbol, interval=interval)
        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', '_1', '_2', '_3', '_4', '_5', '_6'])
                    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception:
            pass

        okx_sym = f"{symbol[:-4]}-{symbol[-4:]}" if symbol.endswith("USDT") else symbol
        url_okx = EXCHANGES["okx"]["url"].format(okx_symbol=okx_sym, interval=interval)
        try:
            async with session.get(url_okx, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raw = data.get("data", [])
                    df = pd.DataFrame(raw, columns=['time', 'open', 'high', 'low', 'close', 'volume', '_1', '_2', '_3'])
                    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    return df.iloc[::-1].reset_index(drop=True)
        except Exception as e:
            logging.error(f"خطا در دریافت کندل {symbol}: {e}")
        return None

# ==========================================
# 5. TECHNICAL & ORDERBOOK ANALYZER
# ==========================================
class TechnicalAnalyzer:
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr_smooth = tr.rolling(14).sum()
        df['plus_di'] = 100 * (pd.Series(plus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
        df['minus_di'] = 100 * (pd.Series(minus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
        dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-9)
        df['adx'] = dx.rolling(14).mean()

        df['vol_ma20'] = df['volume'].rolling(20).mean()
        return df

class OrderBookAnalyzer:
    @staticmethod
    async def get_depth_imbalance(session: aiohttp.ClientSession, symbol: str) -> float:
        url = EXCHANGES["binance"]["depth_url"].format(symbol=symbol)
        try:
            async with session.get(url, timeout=3) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bid_vol = sum([float(b[1]) for b in data.get("bids", [])[:20]])
                    ask_vol = sum([float(a[1]) for a in data.get("asks", [])[:20]])
                    if bid_vol + ask_vol == 0: return 0.0
                    return (bid_vol - ask_vol) / (bid_vol + ask_vol)
        except Exception:
            pass
        return 0.0

# ==========================================
# 6. AI ENGINE & STRATEGY
# ==========================================
class AdvancedAIEngine:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, random_state=42)
        self.is_trained = False

    def train_model(self):
        rows = sync_execute("SELECT rsi, adx, atr, imbalance, win FROM trade_history")
        if len(rows) < AI_MIN_SAMPLES: return
        df = pd.DataFrame(rows, columns=['rsi', 'adx', 'atr', 'imbalance', 'win'])
        self.model.fit(df[['rsi', 'adx', 'atr', 'imbalance']], df['win'])
        self.is_trained = True

    def predict_win_probability(self, rsi: float, adx: float, atr: float, imbalance: float) -> float:
        if not self.is_trained: return 0.60
        return float(self.model.predict_proba(np.array([[rsi, adx, atr, imbalance]]))[0][1])

class StrategyEngine:
    @staticmethod
    def analyze_signal(df: pd.DataFrame, imbalance: float) -> dict:
        if df is None or len(df) < 30: return None
        curr = df.iloc[-1]
        atr, price = curr['atr'], curr['close']
        c_long, c_short = 0, 0

        if curr['rsi'] > 52 and curr['adx'] > 22 and curr['plus_di'] > curr['minus_di']: c_long += 1
        elif curr['rsi'] < 48 and curr['adx'] > 22 and curr['minus_di'] > curr['plus_di']: c_short += 1

        body = abs(curr['close'] - curr['open'])
        if (min(curr['open'], curr['close']) - curr['low']) > body * 2.5 and curr['volume'] > curr['vol_ma20'] * 1.3: c_long += 1
        elif (curr['high'] - max(curr['open'], curr['close'])) > body * 2.5 and curr['volume'] > curr['vol_ma20'] * 1.3: c_short += 1

        if imbalance > 0.25: c_long += 1
        elif imbalance < -0.25: c_short += 1

        direction = "LONG" if c_long >= 2 else ("SHORT" if c_short >= 2 else None)
        if not direction: return None

        risk = atr * 1.5
        
        # اصلاح حد سودها بر اساس جهت معامله (اصلاح خطای محاسباتی عکس)
        if direction == "LONG":
            sl = price - risk
            tp1, tp2, tp3 = price + (risk * 1.5), price + (risk * 3.0), price + (risk * 5.0)
        else: # SHORT
            sl = price + risk
            tp1, tp2, tp3 = price - (risk * 1.5), price - (risk * 3.0), price - (risk * 5.0)

        return {"direction": direction, "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rsi": curr['rsi'], "adx": curr['adx'], "atr": atr, "imbalance": imbalance}

# ==========================================
# 7. PAPER TRADING MANAGER & NOTIFICATIONS
# ==========================================
class TradeManager:
    def __init__(self, ex_manager: ExchangeManager, ai_engine: AdvancedAIEngine):
        self.ex_manager = ex_manager
        self.ai_engine = ai_engine

    async def execute_trade(self, session: aiohttp.ClientSession, symbol: str, signal: dict, account_balance: float):
        entry = signal['entry']
        raw_qty = ((account_balance * RISK_PER_TRADE) / abs(entry - signal['sl'])) * LEVERAGE
        qty = self.ex_manager.format_quantity(symbol, raw_qty)
        if qty <= 0: return

        size_usdt = round(qty * entry, 1)
        trade_id = f"{symbol}_{int(asyncio.get_event_loop().time())}"

        await db_execute("""
        INSERT INTO active_trades (id, symbol, direction, entry_price, stop_loss, tp1, tp2, tp3, quantity, remaining_qty, position_size_usdt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, symbol, signal['direction'], entry, signal['sl'], signal['tp1'], signal['tp2'], signal['tp3'], qty, qty, size_usdt))

        dir_emoji = "🔴" if signal['direction'] == "SHORT" else "🟢"
        
        # فرمت دقیق پیام مطابق با تصویر شما
        msg = f"📄 **(PAPER) Trade Opened** ✅\n\n" \
              f"🪐 #{symbol} | {signal['direction']} {dir_emoji}\n" \
              f"📍 Entry: {entry}\n" \
              f"📊 Size: {size_usdt} USDT\n" \
              f"🛡 SL: {round(signal['sl'], 5)}\n" \
              f"🎯 TP1: {round(signal['tp1'], 5)} | TP2: {round(signal['tp2'], 5)} | TP3: {round(signal['tp3'], 5)}"

        await send_telegram_msg(session, msg)
        logging.info(f"🚀 Paper Trade باز شد: {symbol}")

    async def track_active_positions(self, session: aiohttp.ClientSession):
        while True:
            trades = await db_execute("SELECT id, symbol, direction, entry_price, stop_loss, tp1, tp2, tp3, quantity, remaining_qty, position_size_usdt, tp1_hit, tp2_hit FROM active_trades")
            for t in trades:
                trade_id, symbol, direction, entry, sl, tp1, tp2, tp3, qty, rem_qty, size_usdt, tp1_hit, tp2_hit = t
                df = await self.ex_manager.fetch_klines(session, symbol, "1m")
                if df is None or df.empty: continue
                
                curr_price = df.iloc[-1]['close']
                close_reason, pnl = None, 0

                if (direction == "LONG" and curr_price <= sl) or (direction == "SHORT" and curr_price >= sl):
                    close_reason = "Loss"
                    pnl = (sl - entry) * rem_qty if direction == "LONG" else (entry - sl) * rem_qty

                elif not tp1_hit and ((direction == "LONG" and curr_price >= tp1) or (direction == "SHORT" and curr_price <= tp1)):
                    await db_execute("UPDATE active_trades SET remaining_qty = ?, stop_loss = ?, tp1_hit = 1 WHERE id = ?", (rem_qty * 0.5, entry, trade_id))
                    continue

                elif tp1_hit and not tp2_hit and ((direction == "LONG" and curr_price >= tp2) or (direction == "SHORT" and curr_price <= tp2)):
                    await db_execute("UPDATE active_trades SET remaining_qty = ?, stop_loss = ?, tp2_hit = 1 WHERE id = ?", (rem_qty * 0.4, tp1, trade_id))
                    continue

                elif (direction == "LONG" and curr_price >= tp3) or (direction == "SHORT" and curr_price <= tp3):
                    close_reason = "Profit"
                    pnl = (tp3 - entry) * rem_qty if direction == "LONG" else (entry - tp3) * rem_qty

                if close_reason:
                    win = 1 if pnl > 0 else 0
                    pnl_pct = ((curr_price - entry) / entry * 100) if direction == "LONG" else ((entry - curr_price) / entry * 100)
                    
                    await db_execute("DELETE FROM active_trades WHERE id = ?", (trade_id,))
                    await db_execute("""
                    INSERT INTO trade_history (id, symbol, direction, entry_price, close_price, pnl, pnl_pct, win, rsi, adx, atr, imbalance, close_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, ?)
                    """, (trade_id, symbol, direction, entry, curr_price, pnl, pnl_pct, win, close_reason))

                    # فرمت پیام بسته‌شدن معامله مطابق عکس
                    status_icon = "❌" if close_reason == "Loss" else "💰"
                    close_msg = f"{status_icon} **Trade Closed ({close_reason})**\n\n" \
                                f"🪐 #{symbol} | PnL: {pnl_pct:+.2f}%"
                    
                    await send_telegram_msg(session, close_msg)
                    logging.info(f"🏁 معامله بسته‌شد: {symbol} | PnL: {pnl_pct:.2f}%")
                    self.ai_engine.train_model()

            await asyncio.sleep(3)

# ==========================================
# 8. MAIN EXECUTION LOOP
# ==========================================
async def main():
    init_db()
    ex_manager = ExchangeManager()
    ai_engine = AdvancedAIEngine()
    trade_manager = TradeManager(ex_manager, ai_engine)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AAVEUSDT", "LINKUSDT", "SUIUSDT", "TRUMPUSDT"]
    account_balance = 1000.0

    async with aiohttp.ClientSession() as session:
        await ex_manager.load_exchange_info(session)
        ai_engine.train_model()
        asyncio.create_task(trade_manager.track_active_positions(session))

        logging.info("🤖 Paper Trading Bot فعال شد...")
        while True:
            for symbol in symbols:
                df = await ex_manager.fetch_klines(session, symbol, "15m")
                if df is None: continue
                df = TechnicalAnalyzer.compute_indicators(df)
                imbalance = await OrderBookAnalyzer.get_depth_imbalance(session, symbol)

                signal = StrategyEngine.analyze_signal(df, imbalance)
                if signal:
                    win_prob = ai_engine.predict_win_probability(signal['rsi'], signal['adx'], signal['atr'], signal['imbalance'])
                    if win_prob >= 0.55:
                        existing = await db_execute("SELECT id FROM active_trades WHERE symbol = ?", (symbol,))
                        if not existing:
                            await trade_manager.execute_trade(session, symbol, signal, account_balance)
            await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())
