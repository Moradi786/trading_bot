import asyncio
import logging
import os
import aiohttp
from telegram import Bot
from telegram.constants import ParseMode

# ---------------------------------------------------------
# ۱. تنظیمات اولیه ربات
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

# توکن ربات و آیدی چت تلگرام (از کلیدهای محیطی یا مستقیم جایگزین کنید)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# لیست جفت‌ارزها و تایم‌فریم‌های مورد نظر برای اسکن
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XLMUSDT", "ZECUSDT", "SOLUSDT"]
TIMEFRAMES = ["15m", "1h", "4h"]
MAX_SL_PERCENT = 2.0  # حداکثر درصد حد زیان مجاز

# حافظه موقت برای جلوگیری از ارسال پیام‌های تکراری
sent_alerts = set()


# ---------------------------------------------------------
# ۲. تابع الگوریتم تشخیص کندل ستاپ
# ---------------------------------------------------------
def analyze_candle_setup(klines, max_sl_percent=2.0):
    """
    بررسی کندل ستاپ، نقطه ورود در کندل بعدی، استاپ پشت شدو و TPهای 2, 5, 7
    """
    if len(klines) < 20:
        return None

    # استخراج قیمت‌های Open, High, Low, Close
    opens = [float(k[1]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    # محاسبه SMA 7 روی کندل‌های بسته شده
    sma7 = sum(closes[-8:-1]) / 7

    # کندل ستاپ (آخرین کندل بسته شده - Index -2)
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

    # ----------------------------------------------------
    # ستاپ LONG (صعودی)
    # ----------------------------------------------------
    is_long_wick = (lower_wick >= 1.5 * body) or (lower_wick / total_range >= 0.45)
    sma7_in_lower_wick = c_low <= sma7 <= min(c_open, c_close)

    if is_long_wick and sma7_in_lower_wick:
        entry_price = c_close  # ورود در شروع کندل بعدی (معادل Close کندل ستاپ)
        stop_loss = c_low       # استاپ پشت شدوی بلند کندل ستاپ
        risk = entry_price - stop_loss

        if risk <= 0:
            return None

        sl_percent = (risk / entry_price) * 100
        if sl_percent > max_sl_percent:
            return None  # رد سیگنال در صورت استاپ‌لاس بزرگ

        # محاسبه تارگت‌ها بر اساس ریسک به ریوارد
        tp1 = entry_price + (risk * 2)  # R:R = 2
        tp2 = entry_price + (risk * 5)  # R:R = 5
        tp3 = entry_price + (risk * 7)  # R:R = 7

        return {
            "direction": "LONG 🟢",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "sl_percent": round(sl_percent, 2),
            "tp1": round(tp1, 5),
            "tp2": round(tp2, 5),
            "tp3": round(tp3, 5),
            "sma7": round(sma7, 5),
            "candle_time": klines[-2][0]
        }

    # ----------------------------------------------------
    # ستاپ SHORT (نزولی)
    # ----------------------------------------------------
    is_short_wick = (upper_wick >= 1.5 * body) or (upper_wick / total_range >= 0.45)
    sma7_in_upper_wick = max(c_open, c_close) <= sma7 <= c_high

    if is_short_wick and sma7_in_upper_wick:
        entry_price = c_close  # ورود در شروع کندل بعدی
        stop_loss = c_high      # استاپ پشت شدوی بلند کندل ستاپ
        risk = stop_loss - entry_price

        if risk <= 0:
            return None

        sl_percent = (risk / entry_price) * 100
        if sl_percent > max_sl_percent:
            return None  # رد سیگنال در صورت استاپ‌لاس بزرگ

        # محاسبه تارگت‌ها بر اساس ریسک به ریوارد
        tp1 = entry_price - (risk * 2)  # R:R = 2
        tp2 = entry_price - (risk * 5)  # R:R = 5
        tp3 = entry_price - (risk * 7)  # R:R = 7

        return {
            "direction": "SHORT 🔴",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "sl_percent": round(sl_percent, 2),
            "tp1": round(tp1, 5),
            "tp2": round(tp2, 5),
            "tp3": round(tp3, 5),
            "sma7": round(sma7, 5),
            "candle_time": klines[-2][0]
        }

    return None


# ---------------------------------------------------------
# ۳. دریافت آنلاین کندل‌ها از API صرافی بایننس
# ---------------------------------------------------------
async def fetch_klines(session, symbol, interval):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=30"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception as e:
        LOGGER.error(f"Error fetching data for {symbol} ({interval}): {e}")
    return None


# ---------------------------------------------------------
# ۴. چرخه اصلی اسکن بازار و ارسال پیام
# ---------------------------------------------------------
async def main_scanner_loop():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    LOGGER.info("Starting Candle Setup Bot Scanner...")

    async with aiohttp.ClientSession() as session:
        while True:
            for symbol in SYMBOLS:
                for interval in TIMEFRAMES:
                    klines = await fetch_klines(session, symbol, interval)
                    if not klines:
                        continue

                    signal = analyze_candle_setup(klines, max_sl_percent=MAX_SL_PERCENT)
                    
                    if signal:
                        # کلید یکتا برای جلوگیری از ارسال مجدد یک سیگنال
                        alert_id = f"{symbol}_{interval}_{signal['candle_time']}_{signal['direction']}"

                        if alert_id not in sent_alerts:
                            sent_alerts.add(alert_id)

                            # ساخت متن پیام سیگنال
                            message_text = (
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
                                    text=message_text,
                                    parse_mode=ParseMode.MARKDOWN
                                )
                                LOGGER.info(f"Alert sent for {symbol} ({interval})")
                            except Exception as err:
                                LOGGER.error(f"Failed to send Telegram alert: {err}")

            # ۳۰ ثانیه صبر پیش از اسکن بعدی
            await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(main_scanner_loop())
