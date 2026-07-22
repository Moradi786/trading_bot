import asyncio
import logging
import os
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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")
PORT = int(os.getenv("PORT", 8080))  # پورت مورد نیاز رندر

TIMEFRAMES = ["15m", "1h", "4h"]
MAX_SL_PERCENT = 2.0  # حداکثر درصد حد زیان مجاز
sent_alerts = set()


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
                    s['symbol'] for s in data['symbols'] 
                    if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
                ]
    except Exception as e:
        LOGGER.error(f"Error fetching symbols: {e}")
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XLMUSDT", "ZECUSDT"]


# ---------------------------------------------------------
# ۳. الگوریتم تشخیص کندل ستاپ
# ---------------------------------------------------------
def analyze_candle_setup(klines, max_sl_percent=2.0):
    if len(klines) < 20:
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

    body = abs(c_close - c_open)
    total_range = c_high - c_low

    if total_range == 0:
        return None

    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low

    # LONG Setup
    is_long_wick = (lower_wick >= 1.5 * body) or (lower_wick / total_range >= 0.45)
    sma7_in_lower_wick = c_low <= sma7 <= min(c_open, c_close)

    if is_long_wick and sma7_in_lower_wick:
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
            "candle_time": klines[-2][0]
        }

    # SHORT Setup
    is_short_wick = (upper_wick >= 1.5 * body) or (upper_wick / total_range >= 0.45)
    sma7_in_upper_wick = max(c_open, c_close) <= sma7 <= c_high

    if is_short_wick and sma7_in_upper_wick:
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
            "candle_time": klines[-2][0]
        }

    return None


async def fetch_klines(session, symbol, interval):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=30"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------
# ۴. چرخه اصلی اسکن و ارسال تلگرام
# ---------------------------------------------------------
async def scanner_task():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    LOGGER.info("Starting Market Scanner Task...")

    async with aiohttp.ClientSession() as session:
        symbols = await get_all_usdt_symbols(session)

        while True:
            for symbol in symbols:
                for interval in TIMEFRAMES:
                    klines = await fetch_klines(session, symbol, interval)
                    if not klines:
                        continue

                    signal = analyze_candle_setup(klines, max_sl_percent=MAX_SL_PERCENT)
                    if signal:
                        alert_id = f"{symbol}_{interval}_{signal['candle_time']}_{signal['direction']}"
                        if alert_id not in sent_alerts:
                            sent_alerts.add(alert_id)

                            msg = (
                                f"🎯 **Candle Setup Signal Found!**\n\n"
                                f"🪙 **Coin:** `#{symbol}` | **Timeframe:** `{interval}`\n"
                                f"🚦 **Direction:** {signal['direction']}\n\n"
                                f"📍 **Entry (Next Candle Open):** `{signal['entry_price']}`\n"
                                f"🛡️ **Stop Loss (Behind Wick):** `{signal['stop_loss']}` (`{signal['sl_percent']}% Risk`)\n\n"
                                f"🎯 **Take Profit Targets:**\n"
                                f"🔹 **TP1 (R:R 1:2):** `{signal['tp1']}`\n"
                                f"🔹 **TP2 (R:R 1:5):** `{signal['tp2']}`\n"
                                f"🔹 **TP3 (R:R 1:7):** `{signal['tp3']}`\n\n"
                                f"📉 **SMA 7 Level:** `{signal['sma7']}`"
                            )
                            try:
                                await bot.send_message(
                                    chat_id=TELEGRAM_CHAT_ID,
                                    text=msg,
                                    parse_mode=ParseMode.MARKDOWN
                                )
                            except Exception as e:
                                LOGGER.error(f"Telegram error: {e}")

                    await asyncio.sleep(0.05)

            symbols = await get_all_usdt_symbols(session)
            await asyncio.sleep(15)


# ---------------------------------------------------------
# ۵. پاسخ به پینگ cron-job.org برای زنده نگه داشتن Render
# ---------------------------------------------------------
async def health_check_handler(request):
    return web.Response(text="Trading Bot is running & scanning!", status=200)


async def main():
    # ایجاد وب‌سرور برای cron-job.org
    app = web.Application()
    app.router.add_get('/', health_check_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    LOGGER.info(f"Web server started on port {PORT}")

    # اجرای همزمان اسکنر در پس‌زمینه
    asyncio.create_task(scanner_task())

    # نگه داشتن برنامه در حال اجرا
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
