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
ADMIN_ID = 6147760453  # Deine ID bleibt fest
# ---------------------------------------------------------

# In-Memory-Liste der erlaubten User
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
    alerts = active_alerts.get(chat_id, [])

    if not alerts:
        await update.message.reply_text("🔔 Aktuell sind **keine** aktiven Alarme für diesen Chat eingerichtet.", parse_mode="Markdown")
        return

    text_lines = ["📊 **Aktive Alarme & aktuelle Kurse:**\n"]
    for idx, alert in enumerate(alerts, 1):
        symbol = alert["symbol"]
        target = alert["target_price"]
        trade_type = alert["trade_type"]
        emoji = alert["emoji"]
        
        current = get_crypto_price(symbol)
        current_text = f"{current} USDT" if current is not None else "Fehler beim Abrufen"
        
        text_lines.append(
            f"{idx}. {emoji} **{trade_type}** | #{symbol}\n"
            f"   🎯 Ziel: `{target} USDT`\n"
            f"   ⚡ Aktuell: `{current_text}`\n"
        )
    await update.message.reply_text("\n".join(text_lines), parse_mode="Markdown")

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
        await update.message.reply_text("Zugriff verweigert.")
        return

    chat_id = update.effective_chat.id
    caption = update.message.caption

    if not caption:
        await update.message.reply_text("Bitte füge dem Bild eine Unterschrift hinzu.")
        return

    # Symbol extrahieren (Erstes Wort oder Wort mit #)
    words = caption.replace("#", " ").strip().split()
    if not words:
        await update.message.reply_text("Konnte kein Krypto-Paar finden.")
        return
    
    symbol = words[0].upper()
    if not symbol.endswith("USDT") and symbol not in ["BTC", "ETH"]:
        symbol = symbol + "USDT"

    current_price = get_crypto_price(symbol)
    if current_price is None:
        await update.message.reply_text(f"Fehler: Kurs für {symbol} nicht gefunden.")
        return

    # Suche nach Long und Short Preisen mit Regex
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

    # Fallback: Falls kein "long" oder "short" Text gefunden wurde, alten Parser nutzen
    if not found_alerts:
        try:
            cleaned_text = re.sub(r'(?i)alarm|preis|limit', ' ', caption.replace("#", " "))
            parts = cleaned_text.strip().split()
            target_price = None
            for part in parts:
                try:
                    target_price = float(part.replace(",", "."))
                except ValueError:
                    continue
            if target_price is not None:
                direction = "above" if target_price >= current_price else "below"
                ttype = "LONG" if direction == "above" else "SHORT"
                emoji = "🟢" if ttype == "LONG" else "🔴"
                found_alerts.append({"type": ttype, "price": target_price, "direction": direction, "emoji": emoji})
        except Exception:
            pass

    if not found_alerts:
        await update.message.reply_text("Ungültiges Format. Beispiel:\nMnt\nShort: 0.4183\nLong: 0.4263")
        return

    if chat_id not in active_alerts:
        active_alerts[chat_id] = []

    photo_id = update.message.photo[-1].file_id
    response_text = f"✅ **Alarme für #{symbol} eingerichtet!**\n\n(Aktueller Kurs: {current_price} USDT)\n\n"

    for a in found_alerts:
        active_alerts[chat_id].append({
            "symbol": symbol,
            "target_price": a["price"],
            "photo_id": photo_id,
            "direction": a["direction"],
            "trade_type": a["type"],
            "emoji": a["emoji"]
        })
        response_text += f"{a['emoji']} **{a['type']}** bei `{a['price']} USDT`\n"

    await update.message.reply_text(response_text, parse_mode="Markdown")

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
                        message_text = f"Trade Signal: {emoji} {trade_type}\n\n📊 Pair: #{symbol}\n\n🎯 Entry: {target_price}"
                        await application.bot.send_photo(chat_id=chat_id, photo=photo_id, caption=message_text)
                    except Exception as e:
                        print(f"Fehler beim Senden: {e}")
                    alerts.remove(alert)

async def post_init(application: Application):
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
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("add", add_user))
    application.add_handler(CommandHandler("remove", remove_user))
    application.add_handler(CommandHandler("list", list_users))
    application.add_handler(CommandHandler("alarms", status_alerts))
    application.add_handler(CommandHandler("alarm", status_alerts))
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
