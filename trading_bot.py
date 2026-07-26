import os
import asyncio
import sqlite3
import logging
import json
import time
import io
import aiohttp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
DB_NAME = "alerts.db"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EXCHANGES = {
    "binance": {
        "url": "https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=100",
        "depth_url": "https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=20"
    }
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================================
# 2. GEMINI VISION ANALYZER (تحلیل عکس چارت فرستاده‌شده توسط شما)
# ==========================================
class ChartImageAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key) if api_key else None

    async def analyze_chart_image(self, image_bytes: bytes) -> dict:
        if not self.client:
            return None
        try:
            prompt = """
            این تصویر یک چارت معاملاتی است. لطفا آن را دقیق بررسی کن و خروجی را فقط به فرمت JSON زیر بده:
            {
                "pattern_detected": "نام الگو مثل Head and Shoulders یا Bullish Flag یا SMC Sweep یا None",
                "signal": "LONG یا SHORT یا NEUTRAL",
                "confidence_score": عدد بین 0 تا 100,
                "analysis_summary": "یک جمله کوتاه درباره تحلیل"
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
            logging.error(f"خطا در تحلیل تصویر توسط Gemini: {e}")
            return None

# ==========================================
# 3. TELEGRAM NOTIFIER & PHOTO HANDLER
# ==========================================
async def send_telegram_msg(session: aiohttp.ClientSession, text: str, photo_bytes: bytes = None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    if photo_bytes:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        data = aiohttp.FormData()
        data.add_field("chat_id", TELEGRAM_CHAT_ID)
        data.add_field("caption", text)
        data.add_field("parse_mode", "Markdown")
        data.add_field("photo", photo_bytes, filename="chart.png", content_type="image/png")
        try:
            async with session.post(url, data=data, timeout=10) as resp:
                if resp.status != 200:
                    logging.error(f"خطا در ارسال عکس به تلگرام: {await resp.text()}")
        except Exception as e:
            logging.error(f"خطای ارتباط با تلگرام: {e}")
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        try:
            async with session.post(url, json=payload, timeout=5) as resp:
                if resp.status != 200:
                    logging.error(f"خطا در ارسال پیام تلگرام: {await resp.text()}")
        except Exception as e:
            logging.error(f"خطای ارتباط با تلگرام: {e}")

async def handle_user_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """دریافت و تحلیل تصویر چارت فرستاده‌شده توسط کاربر با Gemini"""
    await update.message.reply_text("📸 تصویر چارت دریافت شد. در حال ارسال به Gemini AI برای تحلیل...")
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()

    analyzer = ChartImageAnalyzer(GEMINI_API_KEY)
    result = await analyzer.analyze_chart_image(bytes(photo_bytes))

    if result:
        sig_emoji = "🟢" if result.get('signal') == "LONG" else ("🔴" if result.get('signal') == "SHORT" else "⚪")
        msg = f"🔍 **نتیجه تحلیل Gemini AI:**\n\n" \
              f"📌 الگوی شناسایی‌شده: `{result.get('pattern_detected', 'هیچکدام')}`\n" \
              f"📊 سیگنال پیشنهادی: `{result.get('signal', 'NEUTRAL')}` {sig_emoji}\n" \
              f"🎯 میزان اطمینان: `{result.get('confidence_score', 0)}%`\n" \
              f"💡 توضیحات: {result.get('analysis_summary', 'بدون توضیح')}"
        await update.message.reply_text(msg, parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ خطا در پردازش تصویر توسط Gemini.")

# ==========================================
# 4. CHART GENERATOR SYSTEM (رسم خودکار چارت)
# ==========================================
class ChartGenerator:
    @staticmethod
    def generate_chart_image(df: pd.DataFrame, symbol: str, signal: dict) -> bytes:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.plot(df.index, df['close'], label="Price", color="#1f77b4", linewidth=1.5)
        
        entry, sl, tp1 = signal['entry'], signal['sl'], signal['tp1']
        ax.axhline(y=entry, color='blue', linestyle='--', linewidth=1, label=f"Entry: {entry}")
        ax.axhline(y=sl, color='red', linestyle='--', linewidth=1, label=f"SL: {round(sl, 4)}")
        ax.axhline(y=tp1, color='green', linestyle='--', linewidth=1, label=f"TP1: {round(tp1, 4)}")
        
        ax.set_title(f"{symbol} - {signal['direction']} Signal Chart", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="upper left")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

# ==========================================
# 5. DATABASE
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
    CREATE TABLE IF NOT EXISTS signal_history (
        id TEXT PRIMARY KEY, symbol TEXT, direction TEXT,
        entry_price REAL, stop_loss REAL, tp1 REAL, tp2 REAL, tp3 REAL,
        rsi REAL, adx REAL, atr REAL, imbalance REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

async def db_execute(query: str, params: tuple = ()):
    return await asyncio.to_thread(sync_execute, query, params)

# ==========================================
# 6. DATA FETCHING & TECHNICAL ANALYSIS
# ==========================================
class ExchangeManager:
    async def fetch_klines(self, session: aiohttp.ClientSession, symbol: str, interval: str = "15m"):
        url = EXCHANGES["binance"]["url"].format(symbol=symbol, interval=interval)
        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', '_1', '_2', '_3', '_4', '_5', '_6'])
                    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except Exception as e:
            logging.error(f"خطا در دریافت کندل {symbol}: {e}")
        return None

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
# 7. ADVANCED STRATEGY ENGINE & AI FILTER
# ==========================================
class AdvancedAIEngine:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, random_state=42)
        self.is_trained = False

    def predict_win_probability(self, rsi: float, adx: float, atr: float, imbalance: float) -> float:
        if not self.is_trained: return 0.60
        return float(self.model.predict_proba(np.array([[rsi, adx, atr, imbalance]]))[0][1])

class StrategyEngine:
    @staticmethod
    def detect_hidden_divergence(df: pd.DataFrame) -> tuple[bool, bool]:
        if len(df) < 20: return False, False
        price_lows = df['low'].iloc[-10:]
        rsi_lows = df['rsi'].iloc[-10:]
        
        hidden_bullish = (df['low'].iloc[-1] > price_lows.min()) and (df['rsi'].iloc[-1] < rsi_lows.min())
        hidden_bearish = (df['high'].iloc[-1] < df['high'].iloc[-10:].max()) and (df['rsi'].iloc[-1] > df['rsi'].iloc[-10:].max())
        
        return hidden_bullish, hidden_bearish

    @staticmethod
    def analyze_signal(df: pd.DataFrame, imbalance: float) -> dict:
        if df is None or len(df) < 30: return None
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        atr, price = curr['atr'], curr['close']
        c_long, c_short = 0, 0

        # ۱. استراتژی RSI + DMI پیشرفته (ADX > 25 و واگرایی مخفی)
        hidden_bull, hidden_bear = StrategyEngine.detect_hidden_divergence(df)
        if curr['adx'] > 25:
            if curr['plus_di'] > curr['minus_di'] and (curr['rsi'] > 50 or hidden_bull):
                c_long += 1
            elif curr['minus_di'] > curr['plus_di'] and (curr['rsi'] < 50 or hidden_bear):
                c_short += 1

        # ۲. پیشرفته‌سازی Candle Setup (سایه ۳ برابر بدنه + Volume Spike ۱.۵ برابر)
        body = abs(curr['close'] - curr['open'])
        lower_wick = min(curr['open'], curr['close']) - curr['low']
        upper_wick = curr['high'] - max(curr['open'], curr['close'])
        volume_spike = curr['volume'] > (curr['vol_ma20'] * 1.5)

        if lower_wick >= (body * 3.0) and volume_spike and curr['close'] > prev['high']:
            c_long += 1
        elif upper_wick >= (body * 3.0) and volume_spike and curr['close'] < prev['low']:
            c_short += 1

        # ۳. شکست معتبر Breakout + 0.5 * ATR
        resistance = df['high'].iloc[-20:-1].max()
        support = df['low'].iloc[-20:-1].min()
        
        if curr['close'] > (resistance + 0.5 * atr) and volume_spike:
            c_long += 1
        elif curr['close'] < (support - 0.5 * atr) and volume_spike:
            c_short += 1

        # ۴. SMC Liquidity Sweep (جاروی نقدینگی سقف و کف‌های برابر)
        eq_high = abs(df['high'].iloc[-5:-1].max() - df['high'].iloc[-10:-5].max()) < (atr * 0.1)
        eq_low = abs(df['low'].iloc[-5:-1].min() - df['low'].iloc[-10:-5].min()) < (atr * 0.1)

        if eq_low and lower_wick >= (body * 2.5):
            c_long += 1
        elif eq_high and upper_wick >= (body * 2.5):
            c_short += 1

        # ۵. Orderbook Imbalance (عمق نقدینگی)
        if imbalance > 0.30: c_long += 1
        elif imbalance < -0.30: c_short += 1

        # صدور سیگنال فقط با حداقل ۳ تاییدیه هم‌زمان
        direction = "LONG" if c_long >= 3 else ("SHORT" if c_short >= 3 else None)
        if not direction: return None

        risk = atr * 1.5
        if direction == "LONG":
            sl = price - risk
            tp1, tp2, tp3 = price + (risk * 1.5), price + (risk * 3.0), price + (risk * 5.0)
        else:
            sl = price + risk
            tp1, tp2, tp3 = price - (risk * 1.5), price - (risk * 3.0), price - (risk * 5.0)

        return {
            "direction": direction, "entry": price, "sl": sl, 
            "tp1": tp1, "tp2": tp2, "tp3": tp3, 
            "rsi": curr['rsi'], "adx": curr['adx'], "atr": atr, "imbalance": imbalance
        }

# ==========================================
# 8. SIGNAL MANAGER & MARKET SCANNER
# ==========================================
class SignalManager:
    def __init__(self, ai_engine: AdvancedAIEngine):
        self.ai_engine = ai_engine
        self.recent_signals = {}

    async def process_signal(self, session: aiohttp.ClientSession, symbol: str, signal: dict, df: pd.DataFrame):
        current_time = time.time()
        if symbol in self.recent_signals and (current_time - self.recent_signals[symbol]) < 900:
            return

        self.recent_signals[symbol] = current_time
        entry = signal['entry']
        signal_id = f"{symbol}_{int(current_time)}"

        await db_execute("""
        INSERT INTO signal_history (id, symbol, direction, entry_price, stop_loss, tp1, tp2, tp3, rsi, adx, atr, imbalance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (signal_id, symbol, signal['direction'], entry, signal['sl'], signal['tp1'], signal['tp2'], signal['tp3'], signal['rsi'], signal['adx'], signal['atr'], signal['imbalance']))

        dir_emoji = "🔴" if signal['direction'] == "SHORT" else "🟢"
        
        msg = f"🚨 **New Advanced Signal** 📢\n\n" \
              f"🪐 #{symbol} | {signal['direction']} {dir_emoji}\n" \
              f"📍 Entry Price: {entry}\n" \
              f"🛡 SL: {round(signal['sl'], 5)}\n" \
              f"🎯 TP1: {round(signal['tp1'], 5)}\n" \
              f"🎯 TP2: {round(signal['tp2'], 5)}\n" \
              f"🎯 TP3: {round(signal['tp3'], 5)}"

        chart_bytes = ChartGenerator.generate_chart_image(df, symbol, signal)
        await send_telegram_msg(session, msg, photo_bytes=chart_bytes)
        logging.info(f"📢 سیگنال پیشرفته به همراه چارت ارسال شد: {symbol} | {signal['direction']}")

async def start_market_scanner(signal_manager: SignalManager, ai_engine: AdvancedAIEngine):
    ex_manager = ExchangeManager()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]

    async with aiohttp.ClientSession() as session:
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
                        await signal_manager.process_signal(session, symbol, signal, df)
            await asyncio.sleep(15)

# ==========================================
# 9. MAIN EXECUTION
# ==========================================
async def main():
    init_db()
    ai_engine = AdvancedAIEngine()
    signal_manager = SignalManager(ai_engine)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_user_photo))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    logging.info("🤖 ربات با تمام استراتژی‌های پیشرفته و Gemini Vision آماده کار است...")

    await start_market_scanner(signal_manager, ai_engine)

if __name__ == "__main__":
    asyncio.run(main())
