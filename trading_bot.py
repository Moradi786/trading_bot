import asyncio
import logging
import os
import time
import random
import aiohttp
from aiohttp import web
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden

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

TIMEFRAMES = ["15m", "1h", "4h"]
MAX_SL_PERCENT = 2.0
MIN_BTC_VOLUME = 30.0  # حداقل حجم ۲۴ ساعته بر حسب بیت‌کوین
sent_alerts = {}
ALERT_TTL = 86400


# ---------------------------------------------------------
# ۲. تعریف صرافی‌ها با Endpoint
# ---------------------------------------------------------
EXCHANGES = [
    {
        "name": "Binance",
        "weight": 10,
        "url": "https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=100",
        "interval_map": {"15m": "15m", "1h": "1h", "4h": "4h"},
        "parser": lambda data: data if isinstance(data, list) else None,
    },
    {
        "name": "Bybit",
        "weight": 8,
        "url": "https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit=100",
        "interval_map": {"15m": "15", "1h": "60", "4h": "240"},
        "parser": lambda data: _parse_bybit(data),
    },
    {
        "name": "OKX",
        "weight": 8,
        "url": "https://www.okx.com/api/v5/market/history-candles?instId={symbol}-SWAP&bar={interval}&limit=100",
        "interval_map": {"15m": "15m", "1h": "1H", "4h": "4H"},
        "parser": lambda data: _parse_okx(data),
    },
    {
        "name": "KuCoin",
        "weight": 6,
        "url": "https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol}&limit=100",
        "interval_map": {"15m": "15min", "1h": "1hour", "4h": "4hour"},
        "parser": lambda data: _parse_kucoin(data),
    },
    {
        "name": "Gate.io",
        "weight": 6,
        "url": "https://api.gateio.ws/api/v4/futures/usdt/candlesticks?contract={symbol}&interval={interval}&limit=100",
        "interval_map": {"15m": "15m", "1h": "1h", "4h": "4h"},
        "parser": lambda data: _parse_gateio(data),
    },
    {
        "name": "Bitget",
        "weight": 5,
        "url": "https://api.bitget.com/api/v2/mix/market/candles?symbol={symbol}&granularity={interval}&limit=100&productType=USDT-FUTURES",
        "interval_map": {"15m": "15m", "1h": "1H", "4h": "4H"},
        "parser": lambda data: _parse_bitget(data),
    },
    {
        "name": "HTX",
        "weight": 4,
        "url": "https://api.hbdm.com/linear-swap-ex/market/history/kline?contract_code={symbol}&period={interval}&size=100",
        "interval_map": {"15m": "15min", "1h": "60min", "4h": "4hour"},
        "parser": lambda data: _parse_htx(data),
    },
    {
        "name": "Kraken",
        "weight": 3,
        "url": "https://futures.kraken.com/api/charts/v1/trade/{symbol}/{interval}?limit=100",
        "interval_map": {"15m": "1", "1h": "60", "4h": "240"},
        "parser": lambda data: _parse_kraken(data),
    },
    {
        "name": "Coinbase",
        "weight": 3,
        "url": "https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={interval}&limit=100",
        "interval_map": {"15m": "900", "1h": "3600", "4h": "14400"},
        "parser": lambda data: _parse_coinbase(data),
    },
]


# ---------------------------------------------------------
# ۳. پارسرها
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


def _parse_kucoin(data):
    try:
        result = data.get("data", [])
        return [[int(x[0]), x[1], x[3], x[4], x[2], x[5]] for x in result]
    except Exception:
        return None


def _parse_gateio(data):
    try:
        return [[int(x[0]), x[5], x[3], x[4], x[2], x[1]] for x in data]
    except Exception:
        return None


def _parse_bitget(data):
    try:
        result = data.get("data", [])
        return [[int(x[6]), x[0], x[1], x[2], x[3], x[4]] for x in result]
    except Exception:
        return None


def _parse_htx(data):
    try:
        result = data.get("data", [])
        return [[int(x[0]), x[1], x[4], x[3], x[2], x[5]] for x in result]
    except Exception:
        return None


def _parse_kraken(data):
    try:
        candles = data.get("candles", [])
        return [[int(c["time"]), c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in candles]
    except Exception:
        return None


def _parse_coinbase(data):
    try:
        return [[int(x[0]), x[3], x[2], x[1], x[4], x[5]] for x in data]
    except Exception:
        return None


# ---------------------------------------------------------
# ۴. دریافت کندل با Failover
# ---------------------------------------------------------
async def fetch_klines_with_failover(session, symbol, interval):
    sorted_exchanges = sorted(EXCHANGES, key=lambda x: x["weight"], reverse=True)

    for ex in sorted_exchanges:
        try:
            mapped_interval = ex["interval_map"].get(interval, interval)
            url = ex["url"].format(symbol=symbol, interval=mapped_interval)

            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=8),
                headers={"User-Agent": "TradingBot/1.0"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    klines = ex["parser"](data)
                    if klines and len(klines) >= 50:
                        return klines
        except Exception:
            pass

        await asyncio.sleep(0.05)

    return None


# ---------------------------------------------------------
# ۵. دریافت لیست نمادها و فیلتر حجم بالای ۲۵۰ بیت‌کوین
# ---------------------------------------------------------
async def get_all_usdt_symbols(session):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                
                # پیدا کردن قیمت لحظه‌ای بیت‌کوین
                btc_price = 60000.0  # مقدار پیش‌فرض در صورت عدم دریافت
                for item in data:
                    if item.get("symbol") == "BTCUSDT":
                        btc_price = float(item.get("lastPrice", 60000.0))
                        break

                # حداقل حجم به تتر معادل 250 بیت‌کوین
                min_usdt_volume = MIN_BTC_VOLUME * btc_price

                valid_symbols = []
                for item in data:
                    symbol = item.get("symbol", "")
                    quote_volume = float(item.get("quoteVolume", 0))

                    if symbol.endswith("USDT") and quote_volume >= min_usdt_volume:
                        valid_symbols.append(symbol)

                LOGGER.info(f"✅ {len(valid_symbols)} نماد با حجم بالای 250 BTC انتخاب شدند (حداقل حجم: ${min_usdt_volume:,.0f})")
                return valid_symbols
    except Exception as e:
        LOGGER.error(f"Error fetching symbols & volume filter: {e}")
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XLMUSDT", "ZECUSDT"]


# ---------------------------------------------------------
# ۶. توابع تئوری داو و مدیریت حافظه
# ---------------------------------------------------------
def find_pivots(highs, lows, left_right=3):
    pivot_highs = []
    pivot_lows = []
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


def cleanup_old_alerts():
    now = time.time()
    expired = [k for k, v in sent_alerts.items() if now - v > ALERT_TTL]
    for k in expired:
        del sent_alerts[k]


# ---------------------------------------------------------
# ۷. الگوریتم اصلی تحلیل کندل ستاپ
# ---------------------------------------------------------
def analyze_candle_setup(klines, max_sl_percent=2.0):
    if len(klines) < 50:
        return None

    opens = [float(k[1]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    sma7 = sum(closes[-8:-1]) / 7

    c_open = opens[-2]
    c_high = highs[-2]
    c_low = lows[-2]
    c_close = closes[-2]

    body_bottom = min(c_open, c_close)
    body_top = max(c_open, c_close)
    body = abs(c_close - c_open)
    total_range = c_high - c_low

    if total_range == 0 or body == 0:
        return None

    upper_wick = c_high - body_top
    lower_wick = body_bottom - c_low

    pivot_highs, pivot_lows = find_pivots(highs[:-1], lows[:-1])
    trend = check_dow_theory_trend(pivot_highs, pivot_lows)

    latest_resistance = max([ph[1] for ph in pivot_highs[-5:]]) if pivot_highs else c_high
    latest_support = min([pl[1] for pl in pivot_lows[-5:]]) if pivot_lows else c_low

    # ----------------------------------------------------
    # LONG Setup
    # ----------------------------------------------------
    is_long_wick = (lower_wick >= 1.5 * body) or (lower_wick / total_range >= 0.45)
    
    # شرط ۱: بدنه حداقل ۳ برابر شدوی بالا باشد
    body_gt_upper = (body >= 3 * upper_wick)
    
    # شرط ۲: SMA 7 تنها و دقیقاً از داخل شدوی پایین رد شده باشد
    sma7_only_in_lower_wick = (c_low <= sma7 < body_bottom)
    
    dow_long_valid = (trend == "BULLISH") or (c_close >= latest_resistance)

    if is_long_wick and body_gt_upper and sma7_only_in_lower_wick and dow_long_valid:
        entry_price = c_close
        stop_loss = c_low
        risk = entry_price - stop_loss

        if risk <= 0:
            return None

        sl_percent = (risk / entry_price) * 100
        if sl_percent > max_sl_percent:
            return None

        return {
            "direction": "LONG 🟢",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "sl_percent": round(sl_percent, 2),
            "tp1": round(entry_price + (risk * 2), 5),
            "tp2": round(entry_price + (risk * 5), 5),
            "tp3": round(entry_price + (risk * 7), 5),
            "sma7": round(sma7, 5),
            "trend": trend,
            "candle_time": klines[-2][0]
        }

    # ----------------------------------------------------
    # SHORT Setup
    # ----------------------------------------------------
    is_short_wick = (upper_wick >= 1.5 * body) or (upper_wick / total_range >= 0.45)
    
    # شرط ۱: بدنه حداقل ۳ برابر شدوی پایین باشد
    body_gt_lower = (body >= 3 * lower_wick)
    
    # شرط ۲: SMA 7 تنها و دقیقاً از داخل شدوی بالا رد شده باشد
    sma7_only_in_upper_wick = (body_top < sma7 <= c_high)
    
    dow_short_valid = (trend == "BEARISH") or (c_close <= latest_support)

    if is_short_wick and body_gt_lower and sma7_only_in_upper_wick and dow_short_valid:
        entry_price = c_close
        stop_loss = c_high
        risk = stop_loss - entry_price

        if risk <= 0:
            return None

        sl_percent = (risk / entry_price) * 100
        if sl_percent > max_sl_percent:
            return None

        return {
            "direction": "SHORT 🔴",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "sl_percent": round(sl_percent, 2),
            "tp1": round(entry_price - (risk * 2), 5),
            "tp2": round(entry_price - (risk * 5), 5),
            "tp3": round(entry_price - (risk * 7), 5),
            "sma7": round(sma7, 5),
            "trend": trend,
            "candle_time": klines[-2][0]
        }

    return None


# ---------------------------------------------------------
# ۸. ارسال پیام تلگرام
# ---------------------------------------------------------
async def send_telegram_message(bot, chat_id, text):
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN
        )
        return True

    except BadRequest as e:
        err = str(e).lower()
        if "chat not found" in err:
            LOGGER.error(f"❌ Chat not found: {chat_id}")
        elif "parse" in err:
            try:
                await bot.send_message(chat_id=chat_id, text=text)
                return True
            except Exception as e2:
                LOGGER.error(f"❌ Fail without format: {e2}")
        return False
    except Exception as e:
        LOGGER.error(f"❌ Telegram error: {e}")
        return False


# ---------------------------------------------------------
# ۹. چرخه اصلی اسکن
# ---------------------------------------------------------
async def scanner_task():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    try:
        me = await bot.get_me()
        LOGGER.info(f"🤖 Bot connected: @{me.username}")
    except Exception as e:
        LOGGER.error(f"❌ Telegram Auth Error: {e}")
        return

    LOGGER.info("✅ Starting Multi-Exchange Scanner...")

    async with aiohttp.ClientSession() as session:
        symbols = await get_all_usdt_symbols(session)

        while True:
            try:
                for symbol in symbols:
                    for interval in TIMEFRAMES:
                        klines = await fetch_klines_with_failover(session, symbol, interval)
                        if not klines:
                            continue

                        signal = analyze_candle_setup(klines, max_sl_percent=MAX_SL_PERCENT)
                        if signal:
                            alert_id = f"{symbol}_{interval}_{signal['candle_time']}_{signal['direction']}"
                            if alert_id not in sent_alerts:
                                sent_alerts[alert_id] = time.time()

                                msg = (
                                    f"🎯 **Candle Setup Signal Found!**\n\n"
                                    f"🪙 **Coin:** `#{symbol}` | **Timeframe:** `{interval}`\n"
                                    f"🚦 **Direction:** {signal['direction']}\n"
                                    f"📈 **Dow Market Trend:** `{signal['trend']}`\n\n"
                                    f"📍 **Entry (Next Candle Open):** `{signal['entry_price']}`\n"
                                    f"🛡️ **Stop Loss (Behind Wick):** `{signal['stop_loss']}` (`{signal['sl_percent']}% Risk`)\n\n"
                                    f"🎯 **Take Profit Targets:**\n"
                                    f"🔹 **TP1 (R:R 1:2):** `{signal['tp1']}`\n"
                                    f"🔹 **TP2 (R:R 1:5):** `{signal['tp2']}`\n"
                                    f"🔹 **TP3 (R:R 1:7):** `{signal['tp3']}`\n\n"
                                    f"📉 **SMA 7 Level:** `{signal['sma7']}`"
                                )
                                await send_telegram_message(bot, TELEGRAM_CHAT_ID, msg)

                        await asyncio.sleep(0.05)

                cleanup_old_alerts()
                symbols = await get_all_usdt_symbols(session)
                await asyncio.sleep(15)

            except Exception as e:
                LOGGER.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(30)


# ---------------------------------------------------------
# ۱۰. وب‌سرور سلامت
# ---------------------------------------------------------
async def health_check_handler(request):
    return web.Response(
        text=f"Multi-Exchange Bot running | Alerts: {len(sent_alerts)}",
        status=200
    )


async def main():
    app = web.Application()
    app.router.add_get("/", health_check_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    LOGGER.info(f"🌐 Web server started on port {PORT}")

    asyncio.create_task(scanner_task())
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
