import asyncio
import logging
import os
import time
import aiohttp
from aiohttp import web
from telegram import Bot
from telegram.constants import ParseMode

# ---------------------------------------------------------
# ۱. تنظیمات اولیه
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
MIN_BTC_VOLUME = 250.0          # حداقل حجم ۲۴ ساعته: بالای ۲۵۰ بیت‌کوین
MAX_SIGNAL_AGE_SECONDS = 180  # حداکثر زمان مجاز ارسال سیگنال (۳ دقیقه)
MAX_SLIPPAGE_PERCENT = 0.2    # حداکثر جابه‌جایی مجاز قیمت بازار نسبت به Entry

sent_alerts = {}
ALERT_TTL = 86400
GLOBAL_BTC_TREND = "NEUTRAL"


# ---------------------------------------------------------
# ۲. تعریف صرافی‌ها
# ---------------------------------------------------------
EXCHANGES = [
    {
        "name": "Binance",
        "weight": 10,
        "url": "https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=100",
        "interval_map": {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"},
        "parser": lambda data: data if isinstance(data, list) else None,
    },
    {
        "name": "Bybit",
        "weight": 8,
        "url": "https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit=100",
        "interval_map": {"15m": "15", "1h": "60", "4h": "240", "1d": "D"},
        "parser": lambda data: _parse_bybit(data),
    },
    {
        "name": "OKX",
        "weight": 8,
        "url": "https://www.okx.com/api/v5/market/history-candles?instId={symbol}-SWAP&bar={interval}&limit=100",
        "interval_map": {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"},
        "parser": lambda data: _parse_okx(data),
    }
]


# ---------------------------------------------------------
# ۳. پارسرها
# ---------------------------------------------------------
def _parse_bybit(data):
    try:
        if data.get("retCode") != 0: return None
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
# ۴. اندیکاتورها و محاسبات تکنیکال
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

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

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
    """استخراج سطوح حمایت و مقاومت تایم‌فریم‌های ۴ ساعته و روزانه"""
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
# ۵. دریافت داده‌ها
# ---------------------------------------------------------
async def fetch_klines_with_failover(session, symbol, interval):
    sorted_exchanges = sorted(EXCHANGES, key=lambda x: x["weight"], reverse=True)
    for ex in sorted_exchanges:
        try:
            mapped_interval = ex["interval_map"].get(interval, interval)
            url = ex["url"].format(symbol=symbol, interval=mapped_interval)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8), headers={"User-Agent": "TradingBot/1.0"}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    klines = ex["parser"](data)
                    if klines and len(klines) >= 50:
                        return klines
        except Exception:
            pass
        await asyncio.sleep(0.05)
    return None


async def update_btc_trend(session):
    global GLOBAL_BTC_TREND
    try:
        klines = await fetch_klines_with_failover(session, "BTCUSDT", "4h")
        if klines:
            highs = [float(k[2]) for k in klines[:-1]]
            lows = [float(k[3]) for k in klines[:-1]]
            ph, pl = find_pivots(highs, lows)
            GLOBAL_BTC_TREND = check_dow_theory_trend(ph, pl)
            LOGGER.info(f"🌐 BTC 4H Trend Updated: {GLOBAL_BTC_TREND}")
    except Exception as e:
        LOGGER.error(f"Error updating BTC trend: {e}")


async def get_all_usdt_symbols(session):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                btc_price = 60000.0
                for item in data:
                    if item.get("symbol") == "BTCUSDT":
                        btc_price = float(item.get("lastPrice", 60000.0))
                        break

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
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


# ---------------------------------------------------------
# ۶. تحلیل تکنیکال (Candle Setup + HTF Range Breakout)
# ---------------------------------------------------------
def analyze_market_signal(klines, symbol, htf_supports, htf_resistances, max_sl_percent=2.0):
    if len(klines) < 50:
        return None

    # ۱. فیلتر زمان تازگی سیگنال (حداکثر ۳ دقیقه پس از بسته شدن)
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

    pivot_highs, pivot_lows = find_pivots(highs, lows)
    trend = check_dow_theory_trend(pivot_highs, pivot_lows)

    # -----------------------------------------------------
    # ۲. تشخیص محدوده رنج (Consolidation Range Detection)
    # -----------------------------------------------------
    lookback = 12
    range_high = max(highs[-lookback:-1])
    range_low = min(lows[-lookback:-1])
    range_width_pct = ((range_high - range_low) / range_low) * 100 if range_low > 0 else 999.0

    # محدوده رنج زمانی است که نوسان ۱۲ کندل قبلی کمتر از ۳.۵ درصد باشد
    is_in_range = (range_width_pct <= 3.5)

    # بررسی موقعیت محدوده نسبت به سطوح 4H و 1D
    is_near_htf_support = any(range_low >= supp * 0.985 and range_low <= supp * 1.025 for supp in htf_supports) if htf_supports else True
    is_near_htf_resistance = any(range_high <= res * 1.015 and range_high >= res * 0.975 for res in htf_resistances) if htf_resistances else True

    # -----------------------------------------------------
    # ۳. قوانین CANDLE SETUP + RANGE BREAKOUT (LONG)
    # -----------------------------------------------------
    is_green_candle = (c_close > c_open)
    is_valid_size = (total_range >= 0.5 * atr)
    is_strong_lower_wick = (lower_wick >= 1.8 * body) and (lower_wick / total_range >= 0.45)
    has_minimal_upper_wick = (upper_wick <= 0.25 * total_range)
    is_sma7_bounce = (c_low <= sma7) and (c_close > sma7)

    # کندل ستاپ استاندارد
    is_base_candle_setup_long = (
        (trend != "BEARISH") and 
        is_green_candle and 
        is_valid_size and 
        is_strong_lower_wick and 
        has_minimal_upper_wick and 
        is_sma7_bounce
    )

    # شکست محدوده رنج با کندل ستاپ در بالای حمایت یا زیر مقاومت ۴ ساعته/روزانه
    is_htf_range_breakout_long = (
        is_in_range and 
        (c_close > range_high) and 
        is_green_candle and 
        (is_near_htf_support or is_near_htf_resistance)
    )

    if is_base_candle_setup_long or is_htf_range_breakout_long:
        if symbol != "BTCUSDT" and GLOBAL_BTC_TREND == "BEARISH":
            return None
        if rsi > 68.0:
            return None

        entry_price = c_close

        # فیلتر پرش قیمت لحظه‌ای
        price_diff_percent = ((current_live_price - entry_price) / entry_price) * 100
        if price_diff_percent > MAX_SLIPPAGE_PERCENT:
            return None

        stop_loss = max(c_low, entry_price - (1.5 * atr))
        risk = entry_price - stop_loss

        if risk > 0:
            sl_percent = (risk / entry_price) * 100
            if sl_percent <= max_sl_percent:
                confirmed = []
                if is_base_candle_setup_long: confirmed.append("Candle Setup 📌")
                if is_htf_range_breakout_long: confirmed.append("4H/1D Range Breakout 🚀")

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

    # -----------------------------------------------------
    # ۴. قوانین CANDLE SETUP + RANGE BREAKOUT (SHORT)
    # -----------------------------------------------------
    is_red_candle = (c_close < c_open)
    is_strong_upper_wick = (upper_wick >= 1.8 * body) and (upper_wick / total_range >= 0.45)
    has_minimal_lower_wick = (lower_wick <= 0.25 * total_range)
    is_sma7_rejection = (c_high >= sma7) and (c_close < sma7)

    is_base_candle_setup_short = (
        (trend != "BULLISH") and 
        is_red_candle and 
        is_valid_size and 
        is_strong_upper_wick and 
        has_minimal_lower_wick and 
        is_sma7_rejection
    )

    is_htf_range_breakout_short = (
        is_in_range and 
        (c_close < range_low) and 
        is_red_candle and 
        (is_near_htf_resistance or is_near_htf_support)
    )

    if is_base_candle_setup_short or is_htf_range_breakout_short:
        if symbol != "BTCUSDT" and GLOBAL_BTC_TREND == "BULLISH":
            return None
        if rsi < 32.0:
            return None

        entry_price = c_close

        # فیلتر پرش قیمت لحظه‌ای
        price_diff_percent = ((entry_price - current_live_price) / entry_price) * 100
        if price_diff_percent > MAX_SLIPPAGE_PERCENT:
            return None

        stop_loss = min(c_high, entry_price + (1.5 * atr))
        risk = stop_loss - entry_price

        if risk > 0:
            sl_percent = (risk / entry_price) * 100
            if sl_percent <= max_sl_percent:
                confirmed = []
                if is_base_candle_setup_short: confirmed.append("Candle Setup 📌")
                if is_htf_range_breakout_short: confirmed.append("4H/1D Range Breakdown 📉")

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
# ۷. ارسال تلگرام و چرخه اصلی
# ---------------------------------------------------------
async def send_telegram_message(bot, chat_id, text):
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
        return True
    except Exception as e:
        LOGGER.error(f"Telegram error: {e}")
        return False


def cleanup_old_alerts():
    now = time.time()
    expired = [k for k, v in sent_alerts.items() if now - v > ALERT_TTL]
    for k in expired:
        del sent_alerts[k]


async def scanner_task():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        me = await bot.get_me()
        LOGGER.info(f"🤖 Bot Connected: @{me.username}")
    except Exception as e:
        LOGGER.error(f"❌ Telegram Auth Error: {e}")
        return

    async with aiohttp.ClientSession() as session:
        await update_btc_trend(session)
        symbols = await get_all_usdt_symbols(session)
        btc_counter = 0

        while True:
            try:
                btc_counter += 1
                if btc_counter >= 15:
                    await update_btc_trend(session)
                    btc_counter = 0

                for symbol in symbols:
                    # دریافت سطوح ۴ ساعته و روزانه برای فیلتر سطوح کلیدی
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
                            htf_supports=htf_supports, 
                            htf_resistances=htf_resistances, 
                            max_sl_percent=MAX_SL_PERCENT
                        )

                        if signal:
                            alert_id = f"{symbol}_{interval}_{signal['candle_time']}_{signal['direction']}"
                            if alert_id not in sent_alerts:
                                sent_alerts[alert_id] = time.time()

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
                                await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)

                        await asyncio.sleep(0.02)

                cleanup_old_alerts()
                symbols = await get_all_usdt_symbols(session)
                await asyncio.sleep(5)

            except Exception as e:
                LOGGER.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(15)


async def health_check_handler(request):
    return web.Response(text="Bot Running Fresh & Fast", status=200)


async def main():
    app = web.Application()
    app.router.add_get("/", health_check_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    asyncio.create_task(scanner_task())
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
