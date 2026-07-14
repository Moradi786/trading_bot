import logging, asyncio, requests, re, os, json
from aiohttp import web
from telegram.ext import Application, CommandHandler, MessageHandler, filters

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- KONFIGURATION ---
TOKEN = "8766875036:AAEpSseVagPrhMph_Jr5iwFZusc3QxyLWW4"
ADMIN_ID = 6147760453
ERLAUBTE_USER = {ADMIN_ID, 6673849133, 5544021969}
DATABASE_URL = f"https://kvdb.io/Trade786Bot_SecureBucket_{ADMIN_ID}/active_alerts"
active_alerts = {}

# --- CLOUD SPEICHER ---
def load_alerts():
    global active_alerts
    try:
        response = requests.get(DATABASE_URL, timeout=10)
        if response.status_code == 200: active_alerts = response.json()
        else: active_alerts = {}
    except: active_alerts = {}

def save_alerts():
    try:
        requests.put(DATABASE_URL, data=json.dumps(active_alerts), headers={'Content-type': 'application/json'}, timeout=10)
    except: pass

def get_crypto_price(symbol):
    clean_symbol = symbol.replace("#", "").upper()
    if not clean_symbol.endswith("USDT"): clean_symbol += "USDT"
    try:
        data = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={clean_symbol}", timeout=5).json()
        return float(data.get("price", 0.0))
    except: return 0.0

# --- FUNKTIONEN ---
async def delete_after_delay(context, chat_id, message_id, delay=30):
    await asyncio.sleep(delay)
    try: await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except: pass

async def status_alerts(update, context):
    chat_id = str(update.effective_chat.id)
    alerts = active_alerts.get(chat_id, [])
    
    if not alerts:
        msg = await update.message.reply_text("🔔 Keine aktiven Alarme.")
        asyncio.create_task(delete_after_delay(context, chat_id, msg.message_id, 30))
        return

    text = ["📊 **Aktive Alarme & aktuelle Kurse:**\n"]
    for idx, alert in enumerate(alerts, 1):
        curr = get_crypto_price(alert["symbol"])
        target = alert["target_price"]
        entry = alert.get("entry_price", curr)
        
        pct = (abs(curr - entry) / abs(target - entry) * 100) if abs(target - entry) != 0 else 0
        pct = max(0, min(100, pct))
        
        # Farbige Balken: 🟩 für Long, 🟥 für Short
        bar_char = "🟩" if alert["trade_type"] == "LONG" else "🟥"
        bar = bar_char * int(pct/10) + "⬜" * (10 - int(pct/10))
        icon = "🟢" if alert["trade_type"] == "LONG" else "🔴"
        
        text.append(f"{idx}. **#{alert['symbol'].replace('USDT', '')}** | BY ( {alert.get('created_by', 'Admin')} )\n"
                    f"{icon} {alert['trade_type']} | 🎯 T: `{target}` | ⚡ Now: `{curr}`\n"
                    f"📈 To Target: {bar} {int(pct)}%\n")
    
    msg = await update.message.reply_text("\n".join(text), parse_mode="Markdown", disable_web_page_preview=True)
    asyncio.create_task(delete_after_delay(context, chat_id, msg.message_id, 30))

async def handle_photo(update, context):
    if update.effective_user.id not in ERLAUBTE_USER: return
    caption = update.message.caption or ""
    chat_id = str(update.effective_chat.id)
    
    parts = caption.split()
    if not parts: return
    symbol = parts[0].upper()
    match = re.search(r'(LONG|SHORT)[:\s]+([0-9.,]+)', caption, re.IGNORECASE)
    
    if match:
        dir, price = match.group(1).upper(), float(match.group(2).replace(",", "."))
        if chat_id not in active_alerts: active_alerts[chat_id] = []
        
        active_alerts[chat_id].append({
            "symbol": symbol, "target_price": price, "trade_type": dir,
            "message_id": update.message.message_id, "entry_price": get_crypto_price(symbol),
            "created_by": update.effective_user.first_name
        })
        save_alerts()
        msg = await update.message.reply_text(f"✅ {symbol} {dir} Alarm gespeichert!")
        asyncio.create_task(delete_after_delay(context, chat_id, msg.message_id, 30))

# --- START (RENDER STABILISIERT) ---
async def run_bot():
    load_alerts()
    app_bot = Application.builder().token(TOKEN).build()
    app_bot.add_handler(CommandHandler(["alarm", "alarms"], status_alerts))
    app_bot.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    await app_bot.initialize()
    await app_bot.start()
    await app_bot.updater.start_polling()
    await asyncio.Event().wait()

async def run_server():
    app = web.Application()
    app.router.add_get("/", lambda r: web.Response(text="Bot is running!"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", 10000))).start()
    await asyncio.Event().wait()

async def main():
    await asyncio.gather(run_bot(), run_server())

if __name__ == '__main__':
    asyncio.run(main())
