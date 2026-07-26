import asyncio
import sqlite3
import math
import logging
from decimal import Decimal, ROUND_DOWN
import aiohttp
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ==========================================
# 1. CONFIGURATION & GLOBAL SETTINGS
# ==========================================
DB_NAME = "trading_bot.db"
RISK_PER_TRADE = 0.02  # 2% Risk per trade
LEVERAGE = 10
AI_MIN_SAMPLES = 50

EXCHANGES = {
    "binance": {
        "url": "https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=100",
        "depth_url": "https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=20",
        "exchange_info": "https://fapi.binance.com/fapi/v1/exchangeInfo"
    },
    "okx": {
        "url": "https://www.okx.com/api/v5/market/history-candles?instId={okx_symbol}-SWAP&bar={interval}&limit=100",
        "depth_url": "https://www.okx.com/api/v5/market/books?instId={okx_symbol}-SWAP&sz=20"
    }
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================================
# 2. DATABASE MANAGEMENT (WAL MODE ENABLED)
# ==========================================
def sync_execute(query: str, params: tuple = ()):
    """اجرای ایمن دستورات SQLite با فعال‌سازی WAL جهت جلوگیری از قفل شدن دیتابیس"""
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
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        quantity REAL,
        remaining_qty REAL,
        tp1_hit INTEGER DEFAULT 0,
        tp2_hit INTEGER DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    sync_execute("""
    CREATE TABLE IF NOT EXISTS trade_history (
        id TEXT PRIMARY KEY,
        symbol TEXT,
        direction TEXT,
        entry_price REAL,
        close_price REAL,
        pnl REAL,
        win INTEGER,
        rsi REAL,
        adx REAL,
        atr REAL,
        imbalance REAL,
        close_reason TEXT
    );
    """)

async def db_execute(query: str, params: tuple = ()):
    return await asyncio.to_thread(sync_execute, query, params)

# ==========================================
# 3. EXCHANGE DATA & PRECISION MANAGER
# ==========================================
class ExchangeManager:
    def __init__(self):
        self.symbol_info = {}

    async def load_exchange_info(self, session: aiohttp.ClientSession):
        """دریافت قوانین اعشار و گام حجم (LOT_SIZE) تمام نمادها از بایننس"""
        try:
            async with session.get(EXCHANGES["binance"]["exchange_info"]) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for s in data["symbols"]:
                        symbol = s["symbol"]
                        step_size = "1"
                        tick_size = "0.01"
                        for f in s["filters"]:
                            if f["filterType"] == "LOT_SIZE":
                                step_size = f["stepSize"]
                            elif f["filterType"] == "PRICE_FILTER":
                                tick_size = f["tickSize"]
                        self.symbol_info[symbol] = {
                            "step_size": float(step_size),
                            "tick_size": float(tick_size)
                        }
        except Exception as e:
            logging.error(f"خطا در دریافت قوانین اعشار صرافی: {e}")

    def format_quantity(self, symbol: str, quantity: float) -> float:
        """فرمت‌دهی دقیق حجم بر اساس گام مجاز صرافی (جلوگیری از خطای LOT_SIZE)"""
        info = self.symbol_info.get(symbol, {"step_size": 0.001})
        step = Decimal(str(info["step_size"]))
        qty_dec = Decimal(str(quantity))
        formatted = (qty_dec // step) * step
        return float(formatted)

    async def fetch_klines(self, session: aiohttp.ClientSession, symbol: str, interval: str = "15m"):
        """دریافت کندل‌ها با قابلیت پشتیبانی از چند صرافی (Failover)"""
        # 1. تلاش در بایننس
        url = EXCHANGES["binance"]["url"].format(symbol=symbol, interval=interval)
        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', '_1', '_2', '_3', '_4', '_5', '_6'])
                    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    return df
        except Exception:
            pass

        # 2. رزرو (Fallback) روی OKX در صورت قطعی بایننس
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
            logging.error(f"تلاش ناکام دریافت کندل برای {symbol}: {e}")
        return None

# ==========================================
# 4. ADVANCED TECHNICAL & ORDERBOOK ANALYZER
# ==========================================
class TechnicalAnalyzer:
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # ADX & DMI
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr_smooth = tr.rolling(14).sum()
        df['plus_di'] = 100 * (pd.Series(plus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
        df['minus_di'] = 100 * (pd.Series(minus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
        dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-9)
        df['adx'] = dx.rolling(14).mean()

        # SMA & Volume Profile
        df['sma7'] = df['close'].rolling(7).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        return df

class OrderBookAnalyzer:
    @staticmethod
    async def get_depth_imbalance(session: aiohttp.ClientSession, symbol: str) -> float:
        """تحلیل پیشرفته دفتر سفارشات و محاسبه نسبت عدم تعادل عمق (Depth Imbalance)"""
        url = EXCHANGES["binance"]["depth_url"].format(symbol=symbol)
        try:
            async with session.get(url, timeout=3) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    bid_vol = sum([float(b[1]) for b in bids[:20]])
                    ask_vol = sum([float(a[1]) for a in asks[:20]])
                    if bid_vol + ask_vol == 0:
                        return 0.0
                    return (bid_vol - ask_vol) / (bid_vol + ask_vol)  # بین 1.0- تا 1.0+
        except Exception:
            pass
        return 0.0

# ==========================================
# 5. AI ENGINE (XGBOOST WITH FEATURE ENG)
# ==========================================
class AdvancedAIEngine:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            random_state=42
        )
        self.is_trained = False

    def train_model(self):
        rows = sync_execute("SELECT rsi, adx, atr, imbalance, win FROM trade_history")
        if len(rows) < AI_MIN_SAMPLES:
            return
        
        df = pd.DataFrame(rows, columns=['rsi', 'adx', 'atr', 'imbalance', 'win'])
        X = df[['rsi', 'adx', 'atr', 'imbalance']]
        y = df['win']
        
        self.model.fit(X, y)
        self.is_trained = True
        logging.info(f"مدل هوش مصنوعی با موفقیت روی {len(df)} داده آموزش داده شد.")

    def predict_win_probability(self, rsi: float, adx: float, atr: float, imbalance: float) -> float:
        if not self.is_trained:
            return 0.60  # مقدار پیش‌فرض در صورت عدم وجود داده کافی
        features = np.array([[rsi, adx, atr, imbalance]])
        prob = self.model.predict_proba(features)[0][1]
        return float(prob)

# ==========================================
# 6. CORE STRATEGY ENGINE & SIGNAL GENERATION
# ==========================================
class StrategyEngine:
    @staticmethod
    def analyze_signal(df: pd.DataFrame, imbalance: float) -> dict:
        if df is None or len(df) < 30:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        atr = curr['atr']
        price = curr['close']

        confirmations_long = 0
        confirmations_short = 0

        # 1. استراتژی مومنتوم RSI + DMI
        if curr['rsi'] > 52 and curr['adx'] > 22 and curr['plus_di'] > curr['minus_di']:
            confirmations_long += 1
        elif curr['rsi'] < 48 and curr['adx'] > 22 and curr['minus_di'] > curr['plus_di']:
            confirmations_short += 1

        # 2. استراتژی کندل استیک و اکشن (Pinbar + Volume Spike)
        body = abs(curr['close'] - curr['open'])
        lower_wick = min(curr['open'], curr['close']) - curr['low']
        upper_wick = curr['high'] - max(curr['open'], curr['close'])
        
        if lower_wick > body * 2.5 and curr['volume'] > curr['vol_ma20'] * 1.3:
            confirmations_long += 1
        elif upper_wick > body * 2.5 and curr['volume'] > curr['vol_ma20'] * 1.3:
            confirmations_short += 1

        # 3. استراتژی SMC & Order Book Imbalance
        if imbalance > 0.25:
            confirmations_long += 1
        elif imbalance < -0.25:
            confirmations_short += 1

        # شرط ورود: حداقل ۲ تاییدیه همزمان از استراتژی‌های مختلف
        direction = None
        if confirmations_long >= 2:
            direction = "LONG"
        elif confirmations_short >= 2:
            direction = "SHORT"

        if not direction:
            return None

        # تعیین حد ضرر و تارگت‌های ۳گانه پله‌ای بر اساس ATR
        if direction == "LONG":
            sl = price - (atr * 1.5)
            risk = price - sl
            tp1 = price + (risk * 2.0)
            tp2 = price + (risk * 4.0)
            tp3 = price + (risk * 6.0)
        else:
            sl = price + (atr * 1.5)
            risk = sl - price
            tp1 = price - (risk * 2.0)
            tp2 = price - (risk * 4.0)
            tp3 = price - (risk * 6.0)

        return {
            "direction": direction,
            "entry": price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rsi": curr['rsi'],
            "adx": curr['adx'],
            "atr": atr,
            "imbalance": imbalance
        }

# ==========================================
# 7. POSITION TRACKER & SCALING EXITS
# ==========================================
class TradeManager:
    def __init__(self, ex_manager: ExchangeManager, ai_engine: AdvancedAIEngine):
        self.ex_manager = ex_manager
        self.ai_engine = ai_engine

    async def execute_trade(self, symbol: str, signal: dict, account_balance: float):
        """محاسبه دقیق حجم معامله و ثبت پوزیشن با مدیریت اعشار"""
        entry = signal['entry']
        risk_dist = abs(entry - signal['sl'])
        risk_amount = account_balance * RISK_PER_TRADE
        
        raw_qty = (risk_amount / risk_dist) * LEVERAGE
        formatted_qty = self.ex_manager.format_quantity(symbol, raw_qty)

        if formatted_qty <= 0:
            return

        trade_id = f"{symbol}_{int(asyncio.get_event_loop().time())}"
        
        await db_execute("""
        INSERT INTO active_trades (id, symbol, direction, entry_price, stop_loss, tp1, tp2, tp3, quantity, remaining_qty)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, symbol, signal['direction'], entry, signal['sl'], signal['tp1'], signal['tp2'], signal['tp3'], formatted_qty, formatted_qty))

        logging.info(f"🚀 معامله جدید باز شد: {symbol} | جهت: {signal['direction']} | قیمت ورود: {entry} | حجم: {formatted_qty}")

    async def track_active_positions(self, session: aiohttp.ClientSession):
        """مدیریت پوزیشن‌های فعال، خروج پله‌ای (TP1/TP2) و ریسک‌فری کردن اتوماتیک"""
        while True:
            trades = await db_execute("SELECT id, symbol, direction, entry_price, stop_loss, tp1, tp2, tp3, quantity, remaining_qty, tp1_hit, tp2_hit FROM active_trades")
            
            for t in trades:
                trade_id, symbol, direction, entry, sl, tp1, tp2, tp3, qty, rem_qty, tp1_hit, tp2_hit = t
                df = await self.ex_manager.fetch_klines(session, symbol, "1m")
                if df is None or df.empty:
                    continue
                
                curr_price = df.iloc[-1]['close']
                close_reason = None
                pnl = 0

                # 1. بررسی حد ضرر (Stop Loss)
                if (direction == "LONG" and curr_price <= sl) or (direction == "SHORT" and curr_price >= sl):
                    close_reason = "STOP_LOSS"
                    pnl = (sl - entry) * rem_qty if direction == "LONG" else (entry - sl) * rem_qty

                # 2. بررسی TP1: خروج 50% حجم + ریسک‌فری کردن معامله (انتقال SL به نقطه ورود)
                elif not tp1_hit and ((direction == "LONG" and curr_price >= tp1) or (direction == "SHORT" and curr_price <= tp1)):
                    exit_qty = rem_qty * 0.5
                    new_rem_qty = rem_qty - exit_qty
                    await db_execute("""
                    UPDATE active_trades SET remaining_qty = ?, stop_loss = ?, tp1_hit = 1 WHERE id = ?
                    """, (new_rem_qty, entry, trade_id))
                    logging.info(f"🎯 TP1 لمس شد برای {symbol}! خروج ۵۰٪ حجم و ریسک‌فری شدن معامله.")
                    continue

                # 3. بررسی TP2: خروج 30% دیگر + تریل کردن حد ضرر روی TP1
                elif tp1_hit and not tp2_hit and ((direction == "LONG" and curr_price >= tp2) or (direction == "SHORT" and curr_price <= tp2)):
                    exit_qty = rem_qty * 0.6  # معادل ۳۰٪ از کل حجم اولیه
                    new_rem_qty = rem_qty - exit_qty
                    await db_execute("""
                    UPDATE active_trades SET remaining_qty = ?, stop_loss = ?, tp2_hit = 1 WHERE id = ?
                    """, (new_rem_qty, tp1, trade_id))
                    logging.info(f"🎯 TP2 لمس شد برای {symbol}! خروج ۳۰٪ حجم دیگر و انتقال SL به TP1.")
                    continue

                # 4. بررسی TP3: خروج کامل (تکمیل معامله)
                elif (direction == "LONG" and curr_price >= tp3) or (direction == "SHORT" and curr_price <= tp3):
                    close_reason = "TP3_FULL"
                    pnl = (tp3 - entry) * rem_qty if direction == "LONG" else (entry - tp3) * rem_qty

                # بستن نهایی معامله در دیتابیس در صورت خروج کامل
                if close_reason:
                    win = 1 if pnl > 0 else 0
                    await db_execute("DELETE FROM active_trades WHERE id = ?", (trade_id,))
                    await db_execute("""
                    INSERT INTO trade_history (id, symbol, direction, entry_price, close_price, pnl, win, rsi, adx, atr, imbalance, close_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, ?)
                    """, (trade_id, symbol, direction, entry, curr_price, pnl, win, close_reason))
                    
                    logging.info(f"🏁 معامله بسته‌ شد: {symbol} | دلیل: {close_reason} | سود/زیان: {pnl:.2f}")
                    
                    # آموزش مجدد مدل در صورت جمع‌آوری داده کافی
                    self.ai_engine.train_model()

            await asyncio.sleep(3)

# ==========================================
# 8. MAIN SCANNER & BOT EXECUTION LOOP
# ==========================================
async def main():
    init_db()
    ex_manager = ExchangeManager()
    ai_engine = AdvancedAIEngine()
    trade_manager = TradeManager(ex_manager, ai_engine)

    symbols_to_scan = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT"]
    account_balance = 1000.0  # موجودی فرضی حساب به تتر

    async with aiohttp.ClientSession() as session:
        await ex_manager.load_exchange_info(session)
        ai_engine.train_model()

        # اجرای همزمان حلقه زیرنظر داشتن پوزیشن‌ها
        asyncio.create_task(trade_manager.track_active_positions(session))

        logging.info("🤖 ربات معاملاتی پیشرفته فعال شد. در حال اسکن بازار...")

        while True:
            for symbol in symbols_to_scan:
                # 1. دریافت کندل‌ها و محاسبه اندیکاتورها
                df = await ex_manager.fetch_klines(session, symbol, "15m")
                if df is None:
                    continue
                
                df = TechnicalAnalyzer.compute_indicators(df)
                
                # 2. تحلیل عمق دفتر سفارشات
                imbalance = await OrderBookAnalyzer.get_depth_imbalance(session, symbol)

                # 3. بررسی سیگنال توسط موتور استراتژی
                signal = StrategyEngine.analyze_signal(df, imbalance)

                if signal:
                    # 4. ارزیابی احتمال برد توسط هوش مصنوعی
                    win_prob = ai_engine.predict_win_probability(
                        signal['rsi'], signal['adx'], signal['atr'], signal['imbalance']
                    )

                    # ورود به معامله تنها در صورت تایید هوش مصنوعی (احتمال بالای ۵۵٪)
                    if win_prob >= 0.55:
                        # بررسی عدم وجود پوزیشن فعال روی همین نماد
                        existing = await db_execute("SELECT id FROM active_trades WHERE symbol = ?", (symbol,))
                        if not existing:
                            await trade_manager.execute_trade(symbol, signal, account_balance)

            await asyncio.sleep(15)  # اسکن مجدد هر ۱۵ ثانیه

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("ربات متوقف شد.")
