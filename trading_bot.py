import asyncio
import logging
import os
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# تنظیمات لاگ‌گیری
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# فرض بر این است که شیء client یا دیتابیس شما از قبل تعریف شده است
# async def get_market_data(session, symbol): ...

async def add_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تابع ثبت آلرت جدید با قابلیت جلوگیری از ثبت تکراری"""
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    text = message.text or message.caption or ""
    photo_file_id = message.photo[-1].file_id if message.photo else None

    # استخراج اطلاعات سمبل، جهت و تارگت (بسته به الگوی پارسر متن شما)
    # مثال ساده: فرض کنید این مقادیر از متن استخراج شده‌اند
    symbol = "BTC"  # به عنوان مثال
    direction = "LONG"
    target = 50000.0
    entry_price = 48000.0
    created_by = message.from_user.username or "Unknown"
    source_link = message.link or ""

    # ۱. جلوگیری از ثبت آلرت‌های تکراری
    try:
        check_dup = await client.execute(
            "SELECT id FROM alerts WHERE chat_id = ? AND symbol = ? AND direction = ? AND target_price = ?",
            (chat_id, symbol, direction, target)
        )
        if len(check_dup.rows) > 0:
            bot_msg = await message.reply_text(
                f"⚠️ Alert for <b>#{symbol} {direction} {target:g}</b> already exists!", 
                parse_mode=ParseMode.HTML
            )
            # اگر تابعی برای پاکسازی خودکار پیام دارید اینجا فراخوانی کنید
            return

        # درج آلرت جدید در دیتابیس
        await client.execute(
            "INSERT INTO alerts (chat_id, symbol, direction, target_price, entry_price, created_by, photo_file_id, source_link) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, symbol, direction, target, entry_price, created_by, photo_file_id, source_link)
        )
        
        confirmation_msg = await message.reply_text(f"✅ Alert saved for #{symbol} {direction} @ {target:g}")
        
    except Exception as e:
        LOGGER.error(f"Error in add_alert: {e}")


async def check_alerts(application: Application, session):
    """تابع پس‌زمینه برای بررسی و ارسال آلرت‌ها بدون مشکل اسپم"""
    while True:
        try:
            # دریافت آلرت‌هایی که قیمتشان به تارگت رسیده است
            # (فرض بر این است که تابع دیتابیس شما این رکوردها را برمی‌گرداند)
            active_alerts = await client.execute("SELECT * FROM alerts") # یا کوئری شرطی بررسی قیمت
            
            for alert in active_alerts.rows:
                alert_id = alert['id']
                chat_id = alert['chat_id']
                symbol = alert['symbol']
                direction = alert['direction']
                target = alert['target_price']
                entry_price = alert['entry_price']
                photo_file_id = alert['photo_file_id']
                created_by = alert['created_by']
                source_link = alert['source_link']

                # شبیه‌سازی شرط رسیدن به تارگت
                reached = True  # این مقدار بر اساس قیمت بازار محاسبه می‌شود

                if reached:
                    dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                    m_data = await get_market_data(session, symbol)
                    
                    success_text = (
                        f"🎯 <b>Target Reached!</b>\n"
                        f"#{symbol} {dir_emoji}\n"
                        f"Target: <code>{target:g}</code>\n"
                        f"📊 Vol: <code>{m_data.get('volume', 0)}</code> | 🕒 RSI: <code>{m_data.get('rsi', 0)}</code>\n\n"
                        f"{m_data.get('ai_analysis', '')}"
                    )
                    
                    # ۲. اصلاح حیاتی: اول در تاریخچه ثبت و از جدول اصلی پاک می‌کنیم تا قطعی تلگرام باعث اسپم نشود
                    try:
                        await client.execute(
                            "INSERT INTO alert_history (chat_id, symbol, direction, target_price, entry_price, created_by, photo_file_id, source_link) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (chat_id, symbol, direction, target, entry_price, 0.0, created_by, photo_file_id, source_link)
                        )
                        await client.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
                    except Exception as db_err:
                        LOGGER.error(f"Error updating DB for alert #{alert_id}: {db_err}")
                        continue # اگر دیتابیس خطا داد، فعلاً به پیام تلگرام دست نزنیم تا لوپ نشود

                    # ۳. سپس پیام را به تلگرام می‌فرستیم
                    try:
                        if photo_file_id:
                            await application.bot.send_photo(
                                chat_id=chat_id, photo=photo_file_id, caption=success_text, parse_mode=ParseMode.HTML
                            )
                        else:
                            await application.bot.send_message(
                                chat_id=chat_id, text=success_text, parse_mode=ParseMode.HTML
                            )
                    except Exception as msg_err:
                        LOGGER.error(f"Failed to dispatch alert notification to chat {chat_id} for ID {alert_id}: {msg_err}")
                    
        except Exception as loop_err:
            LOGGER.error(f"Error in check_alerts loop: {loop_err}")
            
        await asyncio.sleep(30) # بررسی هر ۳۰ ثانیه یک‌بار


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    application = Application.builder().token(token).build()

    # ثبت هندلرها
    application.add_handler(MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), add_alert))

    # راه‌اندازی ربات
    LOGGER.info("Bot is starting...")
    application.run_polling()

if __name__ == "__main__":
    main()
