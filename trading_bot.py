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
ADMIN_ID = 6147760453       # Deine ID (Master-Admin)
FRIEND_1_ID = 6673849133     # ID von Amiri
FRIEND_2_ID = 5544021969     # ID von Ali
# ---------------------------------------------------------

# In-Memory-Liste der erlaubten User
ERLAUBTE_USER = {ADMIN_ID, FRIEND_1_ID, FRIEND_2_ID}
active_alerts = {}

# Kostenlose Cloud-Datenbank URL
DATABASE_URL = f"https://kvdb.io/Trade786Bot_SecureBucket_{ADMIN_ID}/active_alerts"

# Namens-Mapping
USER_NAMES = {
    ADMIN_ID: "Admin",
    FRIEND_1_ID: "Amiri",
    FRIEND_2_ID: "Ali"
}

# --- CLOUD SPEICHER FUNKTIONEN ---
def load_alerts():
    global active_alerts
    try:
        response = requests.get(DATABASE_URL, timeout=8)
        if response.status_code == 200:
            loaded = response.json()
            active_alerts = {int(k): v for k, v in loaded.items()}
            logging.info("Alarme erfolgreich aus der Cloud geladen!")
        elif response.status_code == 404:
            active_alerts = {}
            logging.info("Keine alten Alarme in der Cloud gefunden.")
        else:
            logging.error(f"Fehler beim Laden. Status: {response.status_code}")
            active_alerts = {}
    except Exception as e:
        logging.error(f"Verbindungsfehler beim Laden: {e}")
        active_alerts = {}

def save_alerts():
    try:
        headers = {'Content-type': 'application/json'}
        data_to_send = {str(k): v for k, v in active_alerts.items()}
        response = requests.put(DATABASE_URL, data=json.dumps(data_to_send), headers=headers, timeout=8)
        if response.status_code in [200, 201]:
            logging.info("Alarme erfolgreich in der Cloud gesichert!")
        else:
            logging.error(f"Fehler beim Speichern. Status: {response.status_code}")
    except Exception as e:
        logging.error(f"Verbindungsfehler beim Speichern: {e}")

# ---------------------------------------------------------

def get_crypto_price(symbol):
    symbol = symbol.upper()
    try:
        url_futures = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        response = requests.get(url_futures, timeout=5)
        data = response.json()
        if "price" in data:
            return float(data["price"])
            
        url_spot = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url_spot, timeout=5)
        data = response.json()
        if "price" in data:
            return float(data["price"])
            
    except Exception as e:
        print(f"Fehler beim Abrufen des Preises für {symbol}: {e}")
    return None

def ist_erlaubt(user_id):
    return user_id == ADMIN_ID or user_id in ERLAUBTE_USER

def generate_progress_bar(current, target, start_price, trade_type):
    if start_price is None or start_price == target:
        return "⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0%"
    
    total_distance = abs(target - start_price)
    
    if trade_type == "LONG" and current < start_price:
        percentage = 0
    elif trade_type == "SHORT" and current > start_price:
        percentage = 0
    else:
        if total_distance > 0:
            current_distance = abs(current - start_price) if trade_type == "LONG" else abs(start_price - current)
            percentage = int((current_distance / total_distance) * 100)
        else:
            percentage = 0

    percentage = min(max(percentage, 0), 100)
    
    filled_blocks = percentage // 10
    empty_blocks = 10 - filled_blocks
    
    fill_emoji = "🟩" if trade_type == "LONG" else "🟥"
    empty_emoji = "⬛"
    
    bar = (fill_emoji * filled_blocks) + (empty_emoji * empty_blocks)
    return f"{bar} **{percentage}%**"

# --- HILFSFUNKTION FÜR AUTOMATISCHES LÖSCHEN ---
async def delete_message_after_delay(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int, delay: int):
    """Löscht eine bestimmte Nachricht nach einer Verzögerung (Sekunden)."""
    await asyncio.sleep(delay)
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception as e:
        logging.warning(f"Nachricht konnte nicht gelöscht werden (vielleicht bereits manuell gelöscht): {e}")

# --- ADMIN BEFEHLE ---
async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Nur der Master-Admin darf Nutzer hinzufügen.")
        return
    if not context.args:
        await update.message.reply_text("Bitte gib eine ID an. Beispiel: `/add 123456789`", parse_mode="Markdown")
        return
    try:
        neue_id = int(context.args[0])
        ERLAUBTE_USER.add(neue_id)
        await update.message.reply_text(f"✅ User-ID `{neue_id}` wurde erfolgreich hinzugefügt!", parse_mode="Markdown")
    except ValueError:
        await update.message.reply_text("❌ Ungültige ID.")

async def remove_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Nur der Master-Admin darf Nutzer löschen.")
        return
    if not context.args:
        await update.message.reply_text("Bitte gib eine ID an. Beispiel: `/remove 123456789`", parse_mode="Markdown")
        return
    try:
        ziel_id = int(context.args[0])
        if ziel_id == ADMIN_ID:
            await update.message.reply_text("❌ Du kannst dich nicht selbst löschen!")
            return
        if ziel_id in ERLAUBTE_USER:
            ERLAUBTE_USER.remove(ziel_id)
            await update.message.reply_text(f"🗑️ User-ID `{ziel_id}` wurde gelöscht.", parse_mode="Markdown")
        else:
            await update.message.reply_text("❌ Diese ID war nicht in der Liste.")
    except ValueError:
        await update.message.reply_text("❌ Ungültige ID.")

async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Nur der Master-Admin darf die Liste sehen.")
        return
    if not ERLAUBTE_USER:
        await update.message.reply_text("Die Liste ist leer.")
        return
    liste = "\n".join([f"• `{uid}`" for uid in ERLAUBTE_USER])
    await update.message.reply_text(f"👥 **Erlaubte Nutzer-IDs:**\n\n{liste}", parse_mode="Markdown")

# --- ALARME ANZEIGEN ---
async def status_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ist_erlaubt(update.effective_user.id):
        await update.message.reply_text("Zugriff verweigert.")
        return

    chat_id = update.effective_chat.id
    user_message_id = update.message.message_id

    # 1. Schritt: Sofort die getippte "/alarm" Nachricht des Users löschen
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=user_message_id)
    except Exception as e:
        logging.warning(f"Konnte User-Befehl nicht löschen: {e}")

    alerts = active_alerts.get(chat_id, [])

    # 2. Schritt: Wenn keine Alarme aktiv sind -> "Keine Alarme"-Nachricht für GENAU 5 Sekunden zeigen und löschen!
    if not alerts:
        no_alerts_msg = await update.message.reply_text(
            "🔔 Aktuell sind **keine** aktiven Alarme für diesen Chat eingerichtet.", 
            parse_mode="Markdown"
        )
        # Lösch-Task starten (5 Sekunden)
        asyncio.create_task(delete_message_after_delay(context, chat_id, no_alerts_msg.message_id, 5))
        return

    # 3. Schritt: Alarme existieren -> Zeigen für 60 Sekunden (1 Minute)
    text_lines = ["📊 **Aktive Alarme & aktuelle Kurse:**\n"]
    for idx, alert in enumerate(alerts, 1):
        symbol = alert["symbol"]
        target = alert["target_price"]
        start_price = alert["start_price"]
        trade_type = alert["trade_type"]
        emoji = alert["emoji"]
        creator = alert.get("created_by", "Unbekannt")
        msg_id = alert.get("message_id")
        
        current = get_crypto_price(symbol)
        
        chat_username = update.effective_chat.username
        if chat_username:
            img_link = f"[🖼️ Bild anzeigen](https://t.me/{chat_username}/{msg_id})"
        else:
            cleaned_chat_id = str(chat_id).replace("-100", "")
            img_link = f"[🖼️ Bild anzeigen](https://t.me/c/{cleaned_chat_id}/{msg_id})"
        
        if current is not None:
            current_text = f"{current} USDT"
            progress_bar = generate_progress_bar(current, target, start_price, trade_type)
        else:
            current_text = "Fehler"
            progress_bar = "⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0%"
        
        text_lines.append(
            f"{idx}. **#{symbol}** {' ' * (28 - len(symbol))} BY ( {creator} )\n"
            f"   {emoji} **{trade_type}**\n"
            f"   🎯 Target Preis: `{target} USDT`\n"
            f"   ⚡ Aktuell: `{current_text}`\n"
            f"   📈 To Target: {progress_bar}\n"
            f"   🔗 {img_link}\n"
        )
    
    alerts_msg = await update.message.reply_text(
        "\n".join(text_lines), 
        parse_mode="Markdown", 
        disable_web_page_preview=True
    )
    # Lösch-Task starten (60 Sekunden = 1 Minute)
    asyncio.create_task(delete_message_after_delay(context, chat_id, alerts_msg.message_id, 60))

# --- STANDARDFUNKTIONEN ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ist_erlaubt(update.effective_user.id):
        await update.message.reply_text("Keine Zugriffsberechtigung.")
        return
    await update.message.reply_text(
        "Hi! Ich bin dein Trading-Alarm-Bot.\n\n"
        "Du kannst mir ein Bild mit einer oder beiden Richtungen schicken:\n"
        "`#MNT long 0.4263 short 0.4183`"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ist_erlaubt(update.effective_user.id):
        return

    chat_id = update.effective_chat.id
    caption = update.message.caption
    user = update.effective_user
    message_id = update.message.message_id

    if user.id in USER_NAMES:
        creator_name = USER_NAMES[user.id]
    else:
        creator_name = user.first_name if user.first_name else (user.username if user.username else str(user.id))

    if not caption:
        return

    words = caption.replace("#", " ").strip().split()
    if not words:
        return
    
    symbol = words[0].upper()
    if not symbol.endswith("USDT") and symbol not in ["BTC", "ETH"]:
        symbol = symbol + "USDT"

    current_price = get_crypto_price(symbol)
    if current_price is None:
        return

    long_match = re.search(r'(?i)long[:\s]+([0-9.,]+)', caption)
    short_match = re.search(r'(?i)short[:\s]+([0-9.,]+)', caption)

    found_alerts = []

    if long_match:
        price = float(long_match.group(1).replace(",", "."))
        direction = "above" if price >= current_price else "below"
        found_alerts.append({"type": "LONG", "price": price, "direction": direction, "emoji": "🟢"})

    if short_match:
        price = float(short_match.group(1).replace(",", "."))
        direction = "above" if price >= current_price else "below"
        found_alerts.append({"type": "SHORT", "price": price, "direction": direction, "emoji": "🔴"})

    if not found_alerts:
        return

    if chat_id not in active_alerts:
        active_alerts[chat_id] = []

    photo_id = update.message.photo[-1].file_id

    for a in found_alerts:
        active_alerts[chat_id].append({
            "symbol": symbol,
            "target_price": a["price"],
            "start_price": current_price,
            "photo_id": photo_id,
            "message_id": message_id,
            "direction": a["direction"],
            "trade_type": a["type"],
            "emoji": a["emoji"],
            "created_by": creator_name
        })
    
    save_alerts()

async def price_checker_loop(application: Application):
    while True:
        await asyncio.sleep(30)
        changes_made = False
        for chat_id, alerts in list(active_alerts.items()):
            for alert in list(alerts):
                symbol = alert["symbol"]
                target_price = alert["target_price"]
                photo_id = alert["photo_id"]
                direction = alert["direction"]
                trade_type = alert["trade_type"]
                emoji = alert["emoji"]
                creator = alert.get("created_by", "Unbekannt")

                current_price = get_crypto_price(symbol)
                if current_price is None:
                    continue

                triggered = False
                if direction == "above" and current_price >= target_price:
                    triggered = True
                elif direction == "below" and current_price <= target_price:
                    triggered = True

                if triggered:
                    try:
                        message_text = f"Trade Signal: {emoji} {trade_type} BY ( {creator} )\n\n📊 Pair: #{symbol}\n\n🎯 Entry: {target_price}"
                        await application.bot.send_photo(chat_id=chat_id, photo=photo_id, caption=message_text)
                    except Exception as e:
                        print(f"Fehler beim Senden: {e}")
                    alerts.remove(alert)
                    changes_made = True
        
        if changes_made:
            save_alerts()

async def post_init(application: Application):
    load_alerts()
    asyncio.create_task(price_checker_loop(application))

async def handle_ping(request):
    return web.Response(text="Bot is running!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle_ping)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

async def main():
    await start_web_server()
    application = Application.builder().token(TOKEN).post_init(post_init).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("add", add_user))
    application.add_handler(CommandHandler("remove", remove_user))
    application.add_handler(CommandHandler("list", list_users))
    application.add_handler(CommandHandler(["alarms", "alarm"], status_alerts))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    while True:
        await asyncio.sleep(3600)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
