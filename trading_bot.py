import asyncio
import logging
import os
import time
import aiohttp
from aiohttp import web
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

# ---------------------------------------------------------
# ۱. تنظیمات اولیه و متغیرهای عمومی
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PORT = int(os.getenv("PORT", 8080))

if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
    LOGGER.error("❌ TELEGRAM_BOT_TOKEN تنظیم نشده!")
    raise SystemExit(1)

if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
    LOGGER.error("❌ TELEGRAM_CHAT_ID تنظیم نشده!")
    raise SystemExit(1)

try:
    TELEGRAM_CHAT_ID = int(TELEGRAM_CHAT_ID)
except ValueError:
    pass

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
MAX_SL_PERCENT = 2.0
MIN_BTC_VOLUME = 250.0
MAX_SIGNAL_AGE_SECONDS = 180
MAX_SLIPPAGE_PERCENT = 0.2

# تنظیمات Volatility Pause
VOLATILITY_PAUSE_MINUTES = 15      # از ۳۰ دقیقه به ۱۵ دقیقه
VOLATILITY_THRESHOLD_PERCENT = 2.5  # از ۱.۵٪ به ۲.۵٪

sent_alerts = {}
active_trades = {}
active_trades_lock = asyncio.Lock()
ALERT_TTL = 86400
GLOBAL_BTC_TREND = "NEUTRAL"
BTC_VOLATILITY_PAUSE_UNTIL = 0

STATS = {
    "total_signals": 0,
    "tp1_hits": 0,
    "tp2_hits": 0,
    "tp3_hits": 0,
    "sl_hits": 0
}

_symbol_cache = {"symbols": [], "last_update": 0}


# ---------------------------------------------------------
# ۲. Rate Limiting
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# ۳. صرافی‌ها
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# ۴. پارسرها
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# ۵. اعتبارسنجی داده‌های API
# ---------------------------------------------------------
def validate_klines(klines, symbol):
    if not klines or len(klines) < 10:
        LOGGER.warning(f"⚠️ [{symbol}] Too few klines: {len(klines) if klines else 0}")
        return False, "too_few_klines"

    try:
        last_close = float(klines[-1][4])
        prev_close = float(klines[-2][4])
    except (IndexError, ValueError, TypeError) as e:
        LOGGER.warning(f"⚠️ [{symbol}] Invalid kline format: {e}")
        return False, "invalid_format"

    if prev_close > 0:
        change = abs(last_close - prev_close) / prev_close
        if change > 0.5:
            LOGGER.warning(f"⚠️ [{symbol}] Suspicious price jump: {change*100:.1f}%")
            return False, "suspicious_jump"

    if last_close > 1000000 or last_close < 0.000001:
        LOGGER.warning(f"⚠️ [{symbol}] Suspicious price range: {last_close}")
        return False, "suspicious_range"

    if last_close <= 0:
        LOGGER.warning(f"⚠️ [{symbol}] Zero or negative price: {last_close}")
        return False, "zero_price"

    closes = [float(k[4]) for k in klines[-10:] if len(k) >= 5]
    if len(closes) >= 2:
        avg_close = sum(closes) / len(closes)
        max_dev = max(abs(c - avg_close) for c in closes) / avg_close if avg_close > 0 else 0
        if max_dev > 0.8:
            LOGGER.warning(f"⚠️ [{symbol}] High price variance: {max_dev*100:.1f}%")
            return False, "high_variance"

    return True, "ok"


async def cross_check_price(session, symbol):
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

    if len(prices) >= 2:
        vals = [v for v in prices.values() if v > 0]
        if len(vals) >= 2:
            max_diff = max(vals) / min(vals) - 1
            if max_diff > 0.05:
                LOGGER.error(f"❌ [{symbol}] Price mismatch: {prices} | Diff: {max_diff*100:.1f}%")
                return False, prices
    return True, prices


# ---------------------------------------------------------
# ۶. اندیکاتورها
# ---------------------------------------------------------
def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    alpha = 1.0 / period
    for i in range(period, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


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
    pivot_highs, pivot_lows = [], []
    n = len(highs)
    for i in range(left_right, n - left_right - 1):
        if all(highs[i] > highs[i - j] for j in range(1, left_right + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, left_right + 1)):
            pivot_highs.append((i, highs[i]))
        if all(lows[i] < lows[i - j] for j in range(1, left_right + 1)) and \
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


# ---------------------------------------------------------
# ۷. دریافت داده‌ها
# ---------------------------------------------------------
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
                        valid, reason = validate_klines(klines, symbol)
                        if valid:
                            return klines
                        else:
                            LOGGER.warning(f"⚠️ [{symbol}] Data validation failed on {ex['name']}: {reason}")
        except Exception:
            pass
        await asyncio.sleep(0.05)
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

        # اصلاح: بررسی کندل در حال اجرا (klines[-1]) نه کندل بسته شده (klines[-2])
        klines_15m = await fetch_klines_with_failover(session, "BTCUSDT", "15m")
        if klines_15m and len(klines_15m) >= 1:
            b_open = float(klines_15m[-1][1])   # کندل در حال اجرا
            b_close = float(klines_15m[-1][4])  # کندل در حال اجرا
            change_pct = abs((b_close - b_open) / b_open) * 100
            if change_pct >= VOLATILITY_THRESHOLD_PERCENT:  # ۲.۵٪
                BTC_VOLATILITY_PAUSE_UNTIL = time.time() + (VOLATILITY_PAUSE_MINUTES * 60)  # ۱۵ دقیقه
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
                    LOGGER.warning("⚠️ Could not fetch BTC price, using fallback list.")
                    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
                min_usdt_volume = MIN_BTC_VOLUME * btc_price
                valid_symbols = []
                for item in data:
                    symbol = item.get("symbol", "")
                    quote_volume = float(item.get("quoteVolume", 0))
                    if symbol.endswith("USDT") and quote_volume >= min_usdt_volume:
                        valid_symbols.append(symbol)
                LOGGER.info(f"✅ {len(valid_symbols)} Symbols Selected (>250 BTC Vol)")
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


# ---------------------------------------------------------
# ۸. تحلیل تکنیکال — کندل ستاپ اصلاح شده
# ---------------------------------------------------------
def analyze_market_signal(klines, symbol, interval, htf_supports, htf_resistances, max_sl_percent=2.0):
    if time.time() < BTC_VOLATILITY_PAUSE_UNTIL:
        return None
    if len(klines) < 50:
        return None

    current_time_ms = int(time.time() * 1000)
    current_candle_start_ms = int(klines[-1][0])
    elapsed_seconds = (current_time_ms - current_candle_start_ms) / 1000.0
    if elapsed_seconds > MAX_SIGNAL_AGE_SECONDS:
        return None

    closed_klines = klines[:-1]
    opens = [float(k[1]) for k in closed_klines]
    highs = [float(k[2]) for k in closed_klines]
    lows = [float(k[3]) for k in closed_klines]
    closes = [float(k[4]) for k in closed_klines]
    volumes = [float(k[5]) for k in closed_klines]

    current_live_price = float(klines[-1][4])
    rsi = calculate_rsi(closes)
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

    # ==================== LONG SETUP ====================
    is_green_candle = (c_close > c_open)
    is_valid_size = (total_range >= 0.5 * atr)
    is_strong_lower_wick = (lower_wick >= 2.0 * body) and (lower_wick / total_range >= 0.50)
    has_minimal_upper_wick = (upper_wick <= 0.20 * total_range)
    is_sma7_bounce = (c_low <= sma7) and (sma7 <= body_top)
    is_bounce_confirmed = c_close > sma7

    is_candle_setup_long = (
        (trend != "BEARISH") and
        is_green_candle and
        is_valid_size and
        is_strong_lower_wick and
        has_minimal_upper_wick and
        is_sma7_bounce and
        is_bounce_confirmed and
        is_volume_spike
    )

    is_htf_range_breakout_long = (
        is_in_range and
        (c_close > range_high) and
        is_green_candle and
        is_volume_spike and
        (is_near_htf_support or is_near_htf_resistance)
    )

    if is_candle_setup_long or is_htf_range_breakout_long:
        if symbol != "BTCUSDT" and GLOBAL_BTC_TREND == "BEARISH":
            return None
        if rsi > 68.0:
            return None

        entry_price = c_close
        price_diff_percent = ((current_live_price - entry_price) / entry_price) * 100
        if price_diff_percent > MAX_SLIPPAGE_PERCENT:
            return None

        stop_loss = max(c_low, entry_price - (1.5 * atr))
        risk = entry_price - stop_loss

        if risk > 0:
            sl_percent = (risk / entry_price) * 100
            if sl_percent <= max_sl_percent:
                confirmed = []
                if is_candle_setup_long:
                    confirmed.append(f"Candle Setup 📌 ({interval})")
                if is_htf_range_breakout_long:
                    confirmed.append(f"Range Breakout 🚀 ({interval})")
                if is_liquidity_sweep_long:
                    confirmed.append(f"SMC Liquidity Sweep 🎯 ({interval})")
                strategy_text = " + ".join(confirmed)
                return {
                    "strategy": strategy_text,
                    "direction": "LONG 🟢",
                    "entry_price": entry_price,
                    "stop_loss": round(stop_loss, 5),
                    "sl_percent": round(sl_percent, 2),
                    "tp1": round(entry_price + (risk * 2), 5),
                    "tp2": round(entry_price + (risk * 5), 5),
                    "tp3": round(entry_price + (risk * 7), 5),
                    "sma7": round(sma7, 5),
                    "rsi": rsi,
                    "trend": trend,
                    "candle_time": closed_klines[-1][0]
                }

    # ==================== SHORT SETUP ====================
    is_red_candle = (c_close < c_open)
    is_strong_upper_wick = (upper_wick >= 2.0 * body) and (upper_wick / total_range >= 0.50)
    has_minimal_lower_wick = (lower_wick <= 0.20 * total_range)
    is_sma7_rejection = (c_high >= sma7) and (sma7 >= body_bottom)
    is_rejection_confirmed = c_close < sma7

    is_candle_setup_short = (
        (trend != "BULLISH") and
        is_red_candle and
        is_valid_size and
        is_strong_upper_wick and
        has_minimal_lower_wick and
        is_sma7_rejection and
        is_rejection_confirmed and
        is_volume_spike
    )

    is_htf_range_breakout_short = (
        is_in_range and
        (c_close < range_low) and
        is_red_candle and
        is_volume_spike and
        (is_near_htf_resistance or is_near_htf_support)
    )

    if is_candle_setup_short or is_htf_range_breakout_short:
        if symbol != "BTCUSDT" and GLOBAL_BTC_TREND == "BULLISH":
            return None
        if rsi < 32.0:
            return None

        entry_price = c_close
        price_diff_percent = ((entry_price - current_live_price) / entry_price) * 100
        if price_diff_percent > MAX_SLIPPAGE_PERCENT:
            return None

        stop_loss = min(c_high, entry_price + (1.5 * atr))
        risk = stop_loss - entry_price

        if risk > 0:
            sl_percent = (risk / entry_price) * 100
            if sl_percent <= max_sl_percent:
                confirmed = []
                if is_candle_setup_short:
                    confirmed.append(f"Candle Setup 📌 ({interval})")
                if is_htf_range_breakout_short:
                    confirmed.append(f"Range Breakdown 📉 ({interval})")
                if is_liquidity_sweep_short:
                    confirmed.append(f"SMC Liquidity Sweep 🎯 ({interval})")
                strategy_text = " + ".join(confirmed)
                return {
                    "strategy": strategy_text,
                    "direction": "SHORT 🔴",
                    "entry_price": entry_price,
                    "stop_loss": round(stop_loss, 5),
                    "sl_percent": round(sl_percent, 2),
                    "tp1": round(entry_price - (risk * 2), 5),
                    "tp2": round(entry_price - (risk * 5), 5),
                    "tp3": round(entry_price - (risk * 7), 5),
                    "sma7": round(sma7, 5),
                    "rsi": rsi,
                    "trend": trend,
                    "candle_time": closed_klines[-1][0]
                }

    return None


# ---------------------------------------------------------
# ۹. تعقیب پوزیشن‌ها
# ---------------------------------------------------------
async def track_active_trades(session, bot):
    if not active_trades:
        return

    async with active_trades_lock:
        trades = list(active_trades.items())

    for trade_id, trade in trades:
        symbol = trade["symbol"]
        klines = await fetch_klines_with_failover(session, symbol, "15m")
        if not klines:
            continue

        current_price = float(klines[-1][4])

        async with active_trades_lock:
            if trade_id not in active_trades:
                continue

            if trade["direction"] == "LONG 🟢":
                if current_price <= trade["stop_loss"]:
                    msg = f"❌ **Stop Loss Hit!**\n🪙 `#{symbol}` | SL: `{trade['stop_loss']}` (-{trade['sl_percent']}%)"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    STATS["sl_hits"] += 1
                    del active_trades[trade_id]

                elif current_price >= trade["tp3"] and not trade.get("tp3_hit"):
                    msg = f"🎯🎯🎯 **ALL TARGETS HIT (TP3)!**\n🪙 `#{symbol}` | Final Price: `{current_price}` 🔥"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    STATS["tp3_hits"] += 1
                    del active_trades[trade_id]

                elif current_price >= trade["tp2"] and not trade.get("tp2_hit"):
                    active_trades[trade_id]["tp2_hit"] = True
                    msg = f"🚀 **Target 2 Hit (TP2)!**\n🪙 `#{symbol}` | Price: `{current_price}`"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    STATS["tp2_hits"] += 1

                elif current_price >= trade["tp1"] and not trade.get("tp1_hit"):
                    active_trades[trade_id]["tp1_hit"] = True
                    msg = f"✅ **Target 1 Hit (TP1)!**\n🪙 `#{symbol}` | Price: `{current_price}`"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    STATS["tp1_hits"] += 1

            elif trade["direction"] == "SHORT 🔴":
                if current_price >= trade["stop_loss"]:
                    msg = f"❌ **Stop Loss Hit!**\n🪙 `#{symbol}` | SL: `{trade['stop_loss']}` (-{trade['sl_percent']}%)"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    STATS["sl_hits"] += 1
                    del active_trades[trade_id]

                elif current_price <= trade["tp3"] and not trade.get("tp3_hit"):
                    msg = f"🎯🎯🎯 **ALL TARGETS HIT (TP3)!**\n🪙 `#{symbol}` | Final Price: `{current_price}` 🔥"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    STATS["tp3_hits"] += 1
                    del active_trades[trade_id]

                elif current_price <= trade["tp2"] and not trade.get("tp2_hit"):
                    active_trades[trade_id]["tp2_hit"] = True
                    msg = f"🚀 **Target 2 Hit (TP2)!**\n🪙 `#{symbol}` | Price: `{current_price}`"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    STATS["tp2_hits"] += 1

                elif current_price <= trade["tp1"] and not trade.get("tp1_hit"):
                    active_trades[trade_id]["tp1_hit"] = True
                    msg = f"✅ **Target 1 Hit (TP1)!**\n🪙 `#{symbol}` | Price: `{current_price}`"
                    await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)
                    STATS["tp1_hits"] += 1


# ---------------------------------------------------------
# ۱۰. ارسال تلگرام و دستورات
# ---------------------------------------------------------
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
            LOGGER.warning(f"Telegram send failed (attempt {i+1}/{retries}), retrying in {wait}s...")
            await asyncio.sleep(wait)


async def telegram_command_listener(bot):
    last_update_id = 0
    while True:
        try:
            updates = await bot.get_updates(offset=last_update_id + 1, timeout=5)
            for update in updates:
                last_update_id = update.update_id
                if update.message and update.message.text:
                    raw_text = update.message.text.strip()
                    cmd = raw_text.split('@')[0].lower()
                    chat_id = update.message.chat_id

                    if cmd == "/stats":
                        total = STATS["total_signals"]
                        tp1 = STATS["tp1_hits"]
                        tp2 = STATS["tp2_hits"]
                        tp3 = STATS["tp3_hits"]
                        sl = STATS["sl_hits"]
                        win_rate = round(((tp1 + tp2 + tp3) / total * 100), 1) if total > 0 else 0.0
                        msg = (
                            f"📊 **Bot Performance & Win Rate Stats**\n\n"
                            f"🔢 **Total Signals Sent:** `{total}`\n"
                            f"🎯 **TP1 Hits:** `{tp1}`\n"
                            f"🚀 **TP2 Hits:** `{tp2}`\n"
                            f"🔥 **TP3 Hits:** `{tp3}`\n"
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
                        LOGGER.info("🟢 Volatility pause manually deactivated via /pause command")

                    elif cmd == "/debug":
                        msg = (
                            "🔍 **Debug Info:**\n\n"
                            f"🌐 BTC Trend: `{GLOBAL_BTC_TREND}`\n"
                            f"⏸️ Volatility Pause: `{'YES' if time.time() < BTC_VOLATILITY_PAUSE_UNTIL else 'NO'}`\n"
                            f"📊 Active Trades: `{len(active_trades)}`\n"
                            f"📨 Sent Alerts: `{len(sent_alerts)}`\n"
                            f"💾 Cached Symbols: `{len(_symbol_cache['symbols'])}`"
                        )
                        await send_telegram_message(bot, chat_id, msg)

                    elif cmd in ["/start", "/help"]:
                        msg = (
                            "🤖 **Trading Bot Control Menu**\n\n"
                            "▫️ `/stats` : مشاهده آمار\n"
                            "▫️ `/active` : پوزیشن‌های فعال\n"
                            "▫️ `/pause` : غیرفعال کردن Volatility Pause\n"
                            "▫️ `/debug` : اطلاعات دیباگ\n"
                            "▫️ `/help` : راهنما"
                        )
                        await send_telegram_message(bot, chat_id, msg)

        except Exception as e:
            LOGGER.error(f"Command Listener Error: {e}")
        await asyncio.sleep(2)


def cleanup_old_alerts():
    now = time.time()
    expired = [k for k, v in sent_alerts.items() if now - v > ALERT_TTL]
    for k in expired:
        del sent_alerts[k]


# ---------------------------------------------------------
# ۱۱. اسکنر اصلی
# ---------------------------------------------------------
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

        while True:
            try:
                btc_counter += 1
                if btc_counter >= 15:
                    await update_btc_trend_and_volatility(session)
                    btc_counter = 0

                await track_active_trades(session, bot)

                for symbol in symbols:
                    if symbol in ["ANTHROPICUSDT", "ANTHRCUSDT"]:
                        ok, prices = await cross_check_price(session, symbol)
                        if not ok:
                            LOGGER.error(f"🚫 Skipping {symbol} due to price mismatch")
                            continue

                    klines_4h = await fetch_klines_with_failover(session, symbol, "4h")
                    klines_1d = await fetch_klines_with_failover(session, symbol, "1d")
                    htf_supports, htf_resistances = extract_htf_sr_levels(klines_4h, klines_1d)

                    for interval in TIMEFRAMES:
                        klines = await fetch_klines_with_failover(session, symbol, interval)
                        if not klines:
                            continue

                        signal = analyze_market_signal(
                            klines=klines,
                            symbol=symbol,
                            interval=interval,
                            htf_supports=htf_supports,
                            htf_resistances=htf_resistances,
                            max_sl_percent=MAX_SL_PERCENT
                        )

                        if signal:
                            alert_id = f"{symbol}_{interval}_{signal['candle_time']}_{signal['direction']}"
                            if alert_id not in sent_alerts:
                                sent_alerts[alert_id] = time.time()
                                async with active_trades_lock:
                                    active_trades[alert_id] = {**signal, "symbol": symbol}
                                STATS["total_signals"] += 1

                                tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}"
                                keyboard = InlineKeyboardMarkup([
                                    [InlineKeyboardButton("📊 مشاهده چارت در TradingView", url=tv_url)]
                                ])

                                msg = (
                                    f"🎯 **High-Precision Signal Detected!**\n\n"
                                    f"⚙️ **Strategy:** `{signal['strategy']}`\n"
                                    f"🪙 **Coin:** `#{symbol}` | **Timeframe:** `{interval}`\n"
                                    f"🚦 **Direction:** {signal['direction']}\n"
                                    f"📈 **Market Trend:** `{signal['trend']}`\n"
                                    f"🌐 **BTC Trend:** `{GLOBAL_BTC_TREND}`\n"
                                    f"📊 **RSI (14):** `{signal['rsi']}`\n\n"
                                    f"📍 **Entry:** `{signal['entry_price']}`\n"
                                    f"🛡️ **Stop Loss:** `{signal['stop_loss']}` (`{signal['sl_percent']}% Risk`)\n\n"
                                    f"🎯 **Take Profit Targets:**\n"
                                    f"🔹 **TP1 (1:2):** `{signal['tp1']}`\n"
                                    f"🔹 **TP2 (1:5):** `{signal['tp2']}`\n"
                                    f"🔹 **TP3 (1:7):** `{signal['tp3']}`\n\n"
                                    f"📉 **SMA 7:** `{signal['sma7']}`"
                                )
                                await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg, reply_markup=keyboard)

                        await asyncio.sleep(0.02)

                cleanup_old_alerts()
                symbols = await get_all_usdt_symbols_cached(session)
                await asyncio.sleep(5)

            except Exception as e:
                LOGGER.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(15)


async def health_check_handler(request):
    return web.Response(text="Bot Running Fresh & Fast", status=200)


async def main():
    app = web.Application()
    app.router.add_get("/", health_check_handler)
    app.router.add_get("/health", health_check_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    asyncio.create_task(scanner_task())
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
