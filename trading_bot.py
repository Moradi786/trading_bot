import logging
import asyncio
import requests
import json
import os
import re
from aiohttp import web
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- KONFIGURATION ---
TOKEN = "8766875036:AAEpSseVagPrhMph_Jr5iwFZusc3QxyLWW4"
ADMIN_ID = 6147760453
FRIEND_1_ID = 6673849133
FRIEND_2_ID = 5544021969
ERLAUBTE_USER = {ADMIN_ID, FRIEND_1_ID, FRIEND_2_ID}
DATABASE_URL = f"https://kvdb.io/Trade786Bot_SecureBucket_{ADMIN_ID}/active_alerts"

logging.basicConfig(level=logging.INFO)

active_alerts = {}

# --- FUNKTIONEN ---
def load_alerts():
    global active_alerts
    try:
        response = requests.get(DATABASE_URL, timeout=8)
        if response.status_code == 200:
            active_alerts = {int(k): v for k, v in response.json().items()}
    except: active_alerts = {}

def save_alerts():
    try:
        requests.put(DATABASE_URL, data=json.dumps({str(k): v for k, v in active_alerts.items()}), timeout=8)
    except: pass

def get_crypto_price(symbol):
    try:
        for url in [f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"]:
            data = requests.get(url, timeout=3).json()
            if "price" in data: return float(data["price"])
    except: return None
    return None

async def delete_msg(context, chat_id, message_id, delay=0):
    if delay > 0: await asyncio.sleep(delay)
    try: await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except: pass

# --- HANDLER ---
async def status_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Sofort das Kommando löschen
    await delete_msg(context, update.effective_chat.id, update.message.message_id)
    
    chat_id = update.effective_chat.id
    alerts = active_alerts.get(chat_id, [])

    if not alerts:
        msg = await update.message.reply_text("🔔 Keine aktiven Alarme.")
        asyncio.create_task(delete_msg(context, chat_id, msg.message_id, 5))
        return

    text = ["📊 **Aktive Alarme:**"]
    for idx, a in enumerate(alerts, 1):
        text.append(f"{idx}. #{a['symbol']} | {a['trade_type']} | Target: {a['target_price']}")
    
    msg = await update.message.reply_text("\n".join(text), parse_mode="Markdown")
    asyncio.create_task(delete_msg(context, chat_id, msg.message_id, 30))

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Logik zum Speichern von Signalen (vereinfacht)
    chat_id = update.effective_chat.id
    caption = update.message.caption or ""
    # Hier deine Parsing-Logik einfügen...
    save_alerts()

async def post_init(application):
    load_alerts()

async def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler(["alarm", "alarms"], status_alerts))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    
    # Webserver für Render Keep-Alive
    runner = web.AppRunner(web.Application())
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", 10000))).start()
    
    while True: await asyncio.sleep(3600)

if __name__ == '__main__':
    asyncio.run(main())
