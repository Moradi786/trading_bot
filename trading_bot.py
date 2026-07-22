import asyncio
import logging
import os
import time
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

# اعتبارسنجی متغیرهای محیطی
if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
    LOGGER.error("❌ TELEGRAM_BOT_TOKEN تنظیم نشده!")
    raise SystemExit(1)

if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
    LOGGER.error("❌ TELEGRAM_CHAT_ID تنظیم نشده!")
    raise SystemExit(1)

# تبدیل chat_id به int اگر عددی باشد
try:
    TELEGRAM_CHAT_ID = int(TELEGRAM_CHAT_ID)
except ValueError:
    pass

TIMEFRAMES = ["15m", "1h", "4h"]
MAX_SL_PERCENT = 2.0

# ذخیره آلرت‌ها با timestamp برای جلوگیری از پر شدن حافظه
sent_alerts = {}  # {alert_id: timestamp}
ALERT_TTL = 86400  # ۲۴ ساعت


# ---------------------------------------------------------
# ۲. دریافت لیست کل ارزهای USDT فیوچرز
# ---------------------------------------------------------
async def get_all_usdt_symbols(session):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [
                    s["symbol"] for s in data["symbols"]
                    if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
                ]
    except Exception as e:
        LOGGER.error(f"Error fetching symbols: {e}")
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XLMUSDT", "ZECUSDT"]


# ---------------------------------------------------------
# ۳. توابع کمکی تئوری داو
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


# ---------------------------------------------------------
# ۴. پاک‌سازی آلرت‌های قدیمی
# ---------------------------------------------------------
def cleanup_old_alerts():
    now = time.time()
    expired = [k for k, v in sent_alerts.items() if now - v > ALERT_TTL]
    for k in expired:
        del sent_alerts[k]
    if expired:
        LOGGER.info(f"🧹 {len(expired)} آلرت قدیمی پاک شد")


# ---------------------------------------------------------
# ۵. الگوریتم اصلی تحلیل کندل ستاپ
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

    if total_range == 0:
        return None

    upper_wick = c_high - body_top
    lower_wick = body_bottom - c_low

    if body_bottom <= sma7 <= body_top:
        return None

    pivot_highs, pivot_lows = find_pivots(highs[:-1], lows[:-1])
    trend = check_dow_theory_trend(pivot_highs, pivot_lows)

    latest_resistance = max([ph[1] for ph in pivot_highs[-5:]]) if pivot_highs else c_high
    latest_support = min([pl[1] for pl in pivot_lows[-5:]]) if pivot_lows else c_low

    # LONG
    is_long_wick = (lower_wick >= 1.5 * body) or (lower_wick / total_range >= 0.45)
    sma7_in_lower_wick = (c_low <= sma7 < body_bottom)
    dow_long_valid = (trend == "BULLISH") or (c_close >= latest_resistance)

    if is_long_wick and sma7_in_lower_wick and dow_long_valid:
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

    # SHORT
    is_short_wick = (upper_wick >= 1.5 * body) or (upper_wick / total_range >= 0.45)
    sma7_in_upper_wick = (body_top < sma7 <= c_high)
    dow_short_valid = (trend == "BEARISH") or (c_close <= latest_support)

    if is_short_wick and sma7_in_upper_wick and dow_short_valid:
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


async def fetch_klines(session, symbol, interval):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=100"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------
# ۶. ارسال پیام تلگرام با هندل دقیق خطا
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
            LOGGER.error("   → اگر گروه/کاناله، بوت رو عضو کن")
            LOGGER.error("   → اگر کاربر خصوصیه، اول /start بزن")
        elif "can't parse entities" in err or "parse" in err:
            LOGGER.warning("⚠️ Markdown خراب بود، بدون فرمت ارسال می‌شه")
            try:
                await bot.send_message(chat_id=chat_id, text=text)
                return True
            except Exception as e2:
                LOGGER.error(f"❌ ارسال بدون فرمت هم ناموفق: {e2}")
        else:
            LOGGER.error(f"❌ BadRequest: {e}")
        return False

    except Forbidden as e:
        LOGGER.error(f"🚫 Forbidden: {e} (کاربر بوت رو بلاک کرده)")
        return False

    except Exception as e:
        LOGGER.error(f"❌ خطای تلگرام: {e}")
        return False


# ---------------------------------------------------------
# ۷. چرخه اصلی اسکن
# ---------------------------------------------------------
async def scanner_task():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    # تست اتصال اولیه
    try:
        me = await bot.get_me()
        LOGGER.info(f"🤖 بوت متصل شد: @{me.username}")
    except Exception as e:
        LOGGER.error(f"❌ خطا در اتصال تلگرام: {e}")
        return

    LOGGER.info("✅ Starting Market Scanner with Dow Theory...")

    async with aiohttp.ClientSession() as session:
        symbols = await get_all_usdt_symbols(session)

        while True:
            try:
                for symbol in symbols:
                    for interval in TIMEFRAMES:
                        klines = await fetch_klines(session, symbol, interval)
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
                LOGGER.error(f"❌ خطای حلقه اصلی: {e}")
                await asyncio.sleep(30)


# ---------------------------------------------------------
# ۸. وب‌سرور سلامت
# ---------------------------------------------------------
async def health_check_handler(request):
    return web.Response(
        text=f"Bot running | Alerts: {len(sent_alerts)}",
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
