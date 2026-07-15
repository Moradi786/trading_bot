import os
import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError
from libsql_client import create_client
from aiohttp import web

# 1. LOGGING EINRICHTEN
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 2. UMGEBUNGSVARIABLEN LADEN
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_URL = os.getenv("TURSO_DATABASE_URL")
DB_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

# Admin IDs aus der .env auslesen (Komma-separiert, z. B. "6147760453,554402196")
ADMIN_IDS = [
    int(x.strip()) 
    for x in os.getenv("ADMIN_USER_IDS", "").split(",") 
    if x.strip().isdigit()
]

# AUTOMATISCHER PROGROMM-FIX:
# Konvertiert libsql:// zu https://, um den WebSocket-Fehler auf Render zu umgehen!
if DB_URL and DB_URL.startswith("libsql://"):
    DB_URL = DB_URL.replace("libsql://", "https://")


# 3. AUTOMATISCHE LÖSCH-FUNKTION (CLEAN CHAT)
async def delete_messages_later(bot, chat_id, message_ids, delay=30):
    """Löscht eine Liste von Nachrichten-IDs nach einer Verzögerung im Hintergrund."""
    await asyncio.sleep(delay)
    for msg_id in message_ids:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except TelegramError:
            # Fehler ignorieren, wenn Nachricht bereits gelöscht wurde
            # oder der Bot im Privat-Chat die User-Nachricht nicht löschen darf
            pass


# 4. DATENBANK INITIALISIERUNG
async def init_db():
    """Erstellt die Tabelle in Turso, falls sie noch nicht existiert."""
    try:
        async with create_client(url=DB_URL, auth_token=DB_TOKEN) as client:
            await client.execute(
                "CREATE TABLE IF NOT EXISTS allowed_users (user_id TEXT PRIMARY KEY, name TEXT)"
            )
            logger.info("Datenbank erfolgreich geladen und Tabelle überprüft.")
    except Exception as e:
        logger.error(f"Fehler bei der Datenbank-Initialisierung: {e}")


# 5. TELEGRAM COMMAND: /addid <ID> <NAME>
async def addid_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message
    if not user_msg:
        return

    # Prüfen, ob der Absender Admin ist
    if update.effective_user.id not in ADMIN_IDS:
        bot_msg = await user_msg.reply_text("❌ Du hast keine Rechte für diesen Befehl.")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    args = context.args
    if len(args) < 2:
        bot_msg = await user_msg.reply_text("⚠️ Syntax: `/addid <ID> <NAME>`", parse_mode="Markdown")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    target_id = args[0]
    name = " ".join(args[1:])

    try:
        # In Turso-Datenbank speichern
        async with create_client(url=DB_URL, auth_token=DB_TOKEN) as client:
            await client.execute(
                "INSERT OR REPLACE INTO allowed_users (user_id, name) VALUES (?, ?)",
                (target_id, name)
            )
        bot_msg = await user_msg.reply_text(
            f"✅ User can now add alerts:\n• {name} — `{target_id}`",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Datenbankfehler bei /addid: {e}")
        bot_msg = await user_msg.reply_text(f"❌ Fehler beim Speichern: {e}")

    # Beide Nachrichten nach 30 Sekunden im Hintergrund löschen
    asyncio.create_task(
        delete_messages_later(
            context.bot, 
            update.effective_chat.id, 
            [user_msg.message_id, bot_msg.message_id], 
            30
        )
    )


# 6. TELEGRAM COMMAND: /list
async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message
    if not user_msg:
        return

    # Prüfen, ob der Absender Admin ist
    if update.effective_user.id not in ADMIN_IDS:
        bot_msg = await user_msg.reply_text("❌ Du hast keine Rechte für diesen Befehl.")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    try:
        # Aus Turso-Datenbank lesen
        async with create_client(url=DB_URL, auth_token=DB_TOKEN) as client:
            rs = await client.execute("SELECT user_id, name FROM allowed_users")
            users = rs.rows

        if not users:
            bot_msg = await user_msg.reply_text("No extra user IDs have been added yet.")
        else:
            text = "Added user IDs:\n" + "\n".join([f"• {u[1]} — `{u[0]}`" for u in users])
            bot_msg = await user_msg.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Datenbankfehler bei /list: {e}")
        bot_msg = await user_msg.reply_text(f"❌ Fehler beim Abrufen der Liste: {e}")

    # Beide Nachrichten nach 30 Sekunden im Hintergrund löschen
    asyncio.create_task(
        delete_messages_later(
            context.bot, 
            update.effective_chat.id, 
            [user_msg.message_id, bot_msg.message_id], 
            30
        )
    )


# 7. DUMMY-WEB-SERVER (Wichtig für das Render Port-Binding)
async def handle_ping(request):
    return web.Response(text="Bot is running active!")

async def run_server():
    """Startet einen Mini-Webserver auf dem von Render verlangten Port."""
    app = web.Application()
    app.add_routes([web.get('/', handle_ping)])
    port = int(os.getenv("PORT", 10000))  # Standardmäßig Port 10000 (Render Standard)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    logger.info(f"Webserver gestartet auf Port {port}...")
    await site.start()
    while True:
        await asyncio.sleep(3600)


# 8. BOT STARTEN
async def run_bot():
    if not TOKEN:
        logger.error("Kein TELEGRAM_BOT_TOKEN gefunden!")
        return

    await init_db()
    
    # Telegram Application erstellen
    application = Application.builder().token(TOKEN).build()

    # Befehle registrieren
    application.add_handler(CommandHandler("addid", addid_command))
    application.add_handler(CommandHandler("list", list_command))

    # Bot starten
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    logger.info("Telegram Bot ist aktiv und lauscht...")
    
    while True:
        await asyncio.sleep(3600)


# 9. MAIN ENTRYPOINT (Startet Bot und Webserver gleichzeitig)
async def main():
    await asyncio.gather(run_bot(), run_server())

if __name__ == "__main__":
    asyncio.run(main())
