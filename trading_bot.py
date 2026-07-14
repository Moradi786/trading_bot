import logging
import asyncio
import requests
import re
import os
import json
from aiohttp import web
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# ---------------------------------------------------------
TOKEN = "8766875036:AAEpSseVagPrhMph_Jr5iwFZusc3QxyLWW4"
ADMIN_ID = 6147760453
FRIEND_1_ID = 6673849133
FRIEND_2_ID = 5544021969
# ---------------------------------------------------------

ERLAUBTE_USER = {ADMIN_ID, FRIEND_1_ID, FRIEND_2_ID}
active_alerts = {}
DATABASE_URL = f"https://kvdb.io/Trade786Bot_SecureBucket_{ADMIN_ID}/active_alerts"

USER_NAMES = {ADMIN_ID: "Admin", FRIEND_1_ID: "Amiri", FRIEND_2_ID: "Ali"}

# --- CLOUD SPEICHER ---
def load_alerts():
    global active_alerts
    try:
        response = requests.get(DATABASE_URL, timeout=8)
        if response.status_code == 200:
            active_alerts = {int(k): v for k, v in response.json().items()}
    except:
        active_alerts = {}

def save_alerts():
    try:
        requests.put(DATABASE_URL, data=json.dumps({str(k): v for k, v in active_alerts.items()}), headers={'Content-type': 'application/json'}, timeout=8)
    except:
        pass

# --- HILFSFUNKTIONEN ---
def get_crypto_price(symbol):
    symbol = symbol.upper()
    try:
        for url in [f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}", f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"]:
            data = requests.get(url, timeout=5).json()
            if "price" in data: return float(data["price"])
    except: return None
    return None

async def delete_message_after_delay(context, chat_id, message_id, delay):
    await asyncio.sleep(delay)
    try: await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except: pass

# --- HANDLER ---
async def status_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # SOFORT LÖSCHEN BEVOR LOGIK STARTET
    try: await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
    except: pass

    if not (update.effective_user.id == ADMIN_ID or update.effective_user.id in ERLAUBTE_USER):
        return

    chat_id = update.effective_chat.id
    alerts = active_alerts.get(chat_id, [])

    if not alerts:
        msg = await update.message.reply_text("🔔 Aktuell sind keine aktiven Alarme eingerichtet.")
        asyncio.create_task(delete_message_after_delay(context, chat_id, msg.message_id, 5))
        return

    text = ["📊 **Aktive Alarme:**\n"]
    for idx, alert in enumerate(alerts, 1):
        price = get_crypto_price(alert["symbol"])
        text.append(f"{idx}. #{alert['symbol']} | {alert['trade_type']} | Target: {alert['target_price']} | Aktuell: {price}")
    
    msg = await update.message.reply_text("\n".join(text), parse_mode="Markdown")
    asyncio.create_task(delete_message_after_delay(context, chat_id, msg.message_id, 60))

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not (update.effective_user.id == ADMIN_ID or update.effective_user.id in ERLAUBTE_USER): return
    caption = update.message.caption
    if not caption: return
    
    symbol = caption.split()[0].upper()
    if not symbol.endswith("USDT") and symbol not in ["BTC", "ETH"]: symbol += "USDT"
    
    curr = get_crypto_price(symbol)
    if curr is None: return

    chat_id = update.effective_chat.id
    if chat_id not in active_alerts: active_alerts[chat_id] = []
    
    # Simple Parsing
    for dir in ["LONG", "SHORT"]:
        match = re.search(f'(?i){dir}[:\s]+([0-9.,]+)', caption)
        if match:
            price = float(match.group(1).replace(",", "."))
            active_alerts[chat_id].append({
                "symbol": symbol, "target_price": price, "trade_type": dir,
                "photo_id": update.message.photo[-1].file_id, "direction": "above" if price >= curr else "below"
            })
    save_alerts()

async def post_init(application):
    load_alerts()

async def main():
    application = Application.builder().token(TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler(["alarm", "alarms"], status_alerts))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    while True: await asyncio.sleep(3600)

if __name__ == '__main__':
    asyncio.run(main())
