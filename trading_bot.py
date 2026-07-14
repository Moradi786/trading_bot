import logging
import asyncio
import requests
import re
import os
from aiohttp import web
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# ---------------------------------------------------------
TOKEN = "8766875036:AAEpSseVagPrhMph_Jr5iwFZusc3QxyLWW4"
ADMIN_ID = @Mo_radi786  # <--- ERSETZE DIESE ZAHL MIT DEINER ECHTEN TELEGRAM-ID!
# ---------------------------------------------------------

# In-Memory-Liste der erlaubten User (du bist automatisch immer drin)
ERLAUBTE_USER = {ADMIN_ID}
active_alerts = {}

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

# --- AUTHENTIFIZIERUNG ---
def ist_erlaubt(user_id):
    return user_id == ADMIN_ID or user_id in ERLAUBTE_USER

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
        await update.message.reply_text("❌ Ungültige ID. Die ID muss eine reine Zahl sein.")

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
        await update.message.reply_text("Die Liste ist leer (außer dir als Admin).")
        return

    liste = "\n".join([f"• `{uid}`" for uid in ERLAUBTE_USER])
    await update.message.reply_text(f"👥 **Erlaubte Nutzer-IDs:**\n\n{liste}", parse_mode="Markdown")

# --- STANDARDFUNKTIONEN ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ist_erlaubt(update.effective_user.id):
        await update.message.reply_text("Tut mir leid, dieser Bot ist privat. Du hast keine Zugriffsberechtigung.")
        return

    await update.message.reply_text(
        "Hi! Ich bin dein Trading-Alarm-Bot.\n\n"
        "Schicke mir ein Bild mit einer Unterschrift wie z.B.:\n"
        "`#SUIUSDT long Preis 0.7072` oder `#AVAXUSDT short 6.483`"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ist_erlaubt(update.effective_user.id):
        await update.message.reply_text("Zugriff verweigert.")
        return

    chat_id = update.effective_chat.id
    caption = update.message.caption

    if not caption:
        await update.message.reply_text("Bitte füge dem Bild eine Unterschrift mit Paar und Preis hinzu.")
        return

    user_trade_type = None
    if "long" in caption.lower():
        user_trade_type = "LONG"
    elif "short" in caption.lower():
        user_trade_type = "SHORT"

    try:
        cleaned_text = caption.replace("#", " ").replace("(", " ").replace(")", " ")
        cleaned_text = re.sub(r'(?i)alarm|long|short|preis|limit', ' ', cleaned_text)
        
        parts = cleaned_text.strip().split()
        symbol = None
        target_price = None
        
        for part in parts:
            part = part.strip()
            try:
                potential_price = float(part.replace(",", "."))
                target_price = potential_price
            except ValueError:
                if len(part) >= 3 and part.isalpha():
                    symbol = part.upper()

        if not symbol or target_price is None:
            raise ValueError
        
        if not symbol.endswith("USDT") and symbol not in ["BTC", "ETH"]:
            symbol = symbol + "USDT"

    except ValueError:
        await update.message.reply_text("Ungültiges Format. Beispiel: `#SUIUSDT long Preis 0.7072`")
        return

    current_price = get_crypto_price(symbol)
    if current_price is None:
        await update.message.reply_text(f"Fehler: Konnte keinen Preis für {symbol} finden.")
        return

    direction = "above" if target_price >= current_price else "below"
    
    if not user_trade_type:
        user_trade_type = "LONG" if direction == "above" else "SHORT"

    emoji = "🟢" if user_trade_type == "LONG" else "🔴"

    if chat_id not in active_alerts:
        active_alerts[chat_id] = []

    photo_id = update.message.photo[-1].file_id
    active_alerts[chat_id].append({
        "symbol": symbol,
        "target_price": target_price,
        "photo_id": photo_id,
        "direction": direction,
        "trade_type": user_trade_type,
        "emoji": emoji
    })

    await update.message.reply_text(
        f"✅ **Alarm eingerichtet!**\n\n"
        f"Trade Signal: {emoji} {user_trade_type}\n"
        f"📊 Paar: #{symbol}\n"
        f"🎯 Entry-Ziel: {target_price} USDT\n"
        f"(Aktueller Kurs: {current_price} USDT)"
    )

async def price_checker_loop(application: Application):
    while True:
        await asyncio.sleep(30)
        
        for chat_id, alerts in list(active_alerts.items()):
            for alert in list(alerts):
                symbol = alert["symbol"]
                target_price = alert["target_price"]
                photo_id = alert["photo_id"]
                direction = alert["direction"]
                trade_type = alert["trade_type"]
                emoji = alert["emoji"]

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
                        message_text = (
                            f"Trade Signal: {emoji} {trade_type}\n\n"
                            f"📊 Pair: #{symbol}\n\n"
                            f"🎯 Entry: {target_price}"
                        )
                        
                        await application.bot.send_photo(
                            chat_id=chat_id,
                            photo=photo_id,
                            caption=message_text
                        )
                    except Exception as e:
                        print(f"Fehler beim Senden: {e}")
                    
                    alerts.remove(alert)

async def post_init(application: Application):
    asyncio.create_task(price_checker_loop(application))

# --- DUMMY WEB SERVER ---
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
    print(f"Web-Server läuft auf Port {port}...")

async def main():
    await start_web_server()

    application = Application.builder().token(TOKEN).post_init(post_init).build()
    
    # Handlers registrieren
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("add", add_user))
    application.add_handler(CommandHandler("remove", remove_user))
    application.add_handler(CommandHandler("list", list_users))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    print("Trading-Bot läuft und überwacht die Märkte...")
    
    while True:
        await asyncio.sleep(3600)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
