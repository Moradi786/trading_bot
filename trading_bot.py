import logging
import os
import re
import html
import asyncio
from contextlib import suppress
from pathlib import Path

import aiohttp
import libsql_client
from aiohttp import web
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)
from telegram.error import TelegramError

# Logging initialisieren
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

load_dotenv()

DB_PATH = Path("alerts.db")
BINANCE_PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
SYMBOL_PATTERN = re.compile(r"^\s*#?([A-Za-z0-9/_-]+)")
DIRECTION_TARGET_PATTERN = re.compile(r"(LONG|SHORT)\s+([\d.,]+)", re.IGNORECASE)

# Admins & Erlaubte User IDs aus .env laden
ADMIN_USER_IDS = [
    int(x.strip()) 
    for x in os.getenv("ADMIN_USER_IDS", "").split(",") 
    if x.strip().isdigit()
]
ALLOWED_USER_IDS = [
    int(x.strip()) 
    for x in os.getenv("ALLOWED_USER_IDS", "").split(",") 
    if x.strip().isdigit()
]
CHECK_INTERVAL_SECONDS = max(10, int(os.getenv("CHECK_INTERVAL_SECONDS", "30")))


# 1. DATENBANK-SETUP (Turso)
async def initialise_database(client) -> None:
    await client.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
            target_price REAL NOT NULL CHECK(target_price > 0),
            entry_price REAL NOT NULL CHECK(entry_price > 0),
            created_by TEXT NOT NULL,
            source_link TEXT,
            photo_file_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await client.execute(
        """
        CREATE TABLE IF NOT EXISTS authorised_users (
            user_id INTEGER PRIMARY KEY,
            added_by INTEGER NOT NULL,
            display_name TEXT,
            added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


# 2. HILFSFUNKTIONEN
def normalise_symbol(value: str) -> str:
    return value.upper().strip().replace("/", "").replace("-", "").replace("_", "")


def message_link(chat_id: int, chat_username: str | None, chat_type: str, message_id: int) -> str | None:
    if chat_type in {"group", "supergroup", "channel"} and chat_username:
        return f"https://t.me/{chat_username}/{message_id}"
    raw_chat_id = str(chat_id)
    if chat_type in {"supergroup", "channel"} and raw_chat_id.startswith("-100"):
        return f"https://t.me/c/{raw_chat_id[4:]}/{message_id}"
    return None


async def get_price(session: aiohttp.ClientSession, symbol: str) -> float | None:
    try:
        norm_symbol = normalise_symbol(symbol)
        if not norm_symbol.endswith("USDT") and not norm_symbol.endswith("BUSD"):
            norm_symbol += "USDT"
        async with session.get(BINANCE_PRICE_URL, params={"symbol": norm_symbol}) as response:
            if response.status == 200:
                data = await response.json()
                return float(data["price"])
    except Exception:
        LOGGER.exception("Fehler beim Abrufen des Preises für %s", symbol)
    return None


def parse_caption(caption: str) -> list[tuple[str, str, float]]:
    alerts = []
    caption_clean = caption.replace(",", ".")  # Deutsche Kommasetzung abfangen
    lines = [line.strip() for line in caption_clean.split("\n") if line.strip()]
    if not lines:
        return alerts
        
    symbol_match = SYMBOL_PATTERN.search(lines[0])
    if not symbol_match:
        return alerts
    symbol = symbol_match.group(1).upper()
    
    for line in lines[1:]:
        match = DIRECTION_TARGET_PATTERN.search(line)
        if match:
            direction = match.group(1).upper()
            try:
                target = float(match.group(2))
                alerts.append((symbol, direction, target))
            except ValueError:
                continue
    return alerts


# 3. SAUBERER CHAT: AUTOMATISCHES LÖSCHEN
async def delete_messages_later(bot, chat_id, message_ids, delay=30):
    """Löscht eine Liste von Nachrichten-IDs nach exakt 'delay' Sekunden."""
    await asyncio.sleep(delay)
    for msg_id in message_ids:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except TelegramError:
            # Fehler ignorieren, falls die Nachricht schon manuell gelöscht wurde 
            # oder der Bot im Privat-Chat keine User-Nachrichten löschen darf
            pass


# 4. RECHTEPRÜFUNG
async def is_authorised(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    if user is None:
        return False
    if user.id in ADMIN_USER_IDS or user.id in ALLOWED_USER_IDS:
        return True
    if not ADMIN_USER_IDS and not ALLOWED_USER_IDS:
        return True
    client = context.application.bot_data["db_client"]
    result = await client.execute("SELECT 1 FROM authorised_users WHERE user_id = ?", (user.id,))
    return len(result.rows) > 0


def is_admin(update: Update) -> bool:
    user = update.effective_user
    return user is not None and user.id in ADMIN_USER_IDS


# 5. COMMANDS: USER-VERWALTUNG (ADMINS)
async def add_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not is_admin(update):
        bot_msg = await user_msg.reply_text("Only an admin can add allowed user IDs.")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    if len(context.args) < 2:
        bot_msg = await user_msg.reply_text("Usage: <code>/addid 123456789 AMIRI</code>", parse_mode=ParseMode.HTML)
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    try:
        u_id = int(context.args[0])
        display_name = " ".join(context.args[1:])
        entries = [(u_id, display_name)]
    except ValueError:
        bot_msg = await user_msg.reply_text("Invalid User ID.")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    client = context.application.bot_data["db_client"]
    for u_id, name in entries:
        await client.execute(
            """
            INSERT INTO authorised_users (user_id, added_by, display_name) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET added_by = excluded.added_by, display_name = excluded.display_name
            """,
            (u_id, update.effective_user.id, name),
        )

    added = "\n".join(
        f"• {html.escape(name) if name else 'No name'} — <code>{u_id}</code>"
        for u_id, name in entries
    )
    bot_msg = await user_msg.reply_text(
        f"✅ User{'s' if len(entries) > 1 else ''} can now add alerts:\n{added}", parse_mode=ParseMode.HTML
    )
    asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))


async def delete_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not is_admin(update):
        bot_msg = await user_msg.reply_text("Only an admin can delete allowed user IDs.")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    if not context.args or not context.args[0].isdigit():
        bot_msg = await user_msg.reply_text("Usage: <code>/deleteid 123456789</code>", parse_mode=ParseMode.HTML)
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    user_id = int(context.args[0])
    client = context.application.bot_data["db_client"]
    result = await client.execute("DELETE FROM authorised_users WHERE user_id = ?", (user_id,))
    
    message = (
        f"🗑️ User ID <code>{user_id}</code> removed."
        if result.rows_affected > 0
        else "User ID was not in the added list."
    )
    bot_msg = await user_msg.reply_text(message, parse_mode=ParseMode.HTML)
    asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))


async def list_user_ids(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not is_admin(update):
        bot_msg = await user_msg.reply_text("Only an admin can view allowed user IDs.")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "SELECT user_id, display_name FROM authorised_users ORDER BY added_at, user_id"
    )
    users = result.rows
    if not users:
        bot_msg = await user_msg.reply_text("No extra user IDs have been added yet.")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    lines = ["👥 <b>Allowed users</b>"]
    for number, row in enumerate(users, start=1):
        u_id, display_name = row[0], row[1]
        name = html.escape(display_name) if display_name else "No name"
        lines.append(f"{number}. <b>{name}</b> — <code>{u_id}</code>")

    bot_msg = await user_msg.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))


# 6. COMMANDS: ALARM-VERWALTUNG (SIGNALE)
async def add_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Wird getriggert, wenn ein User ein Signalbild mit Caption postet."""
    if not await is_authorised(update, context) or update.message is None:
        return

    parsed_alerts = parse_caption(update.message.caption or "")
    if not parsed_alerts:
        return

    photo_file_id = update.message.photo[-1].file_id if update.message.photo else None
    chat = update.effective_chat
    message_id = update.message.message_id
    source_link = message_link(chat.id, chat.username, chat.type, message_id) if chat else None

    client = context.application.bot_data["db_client"]
    session = context.application.bot_data["http_session"]
    alert_ids: list[int] = []
    creator = update.effective_user.username or update.effective_user.first_name or "Unknown"

    for symbol, direction, target in parsed_alerts:
        current_price = await get_price(session, symbol) or target
        result = await client.execute(
            "INSERT INTO alerts (chat_id, symbol, direction, target_price, entry_price, created_by, source_link, photo_file_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (update.effective_chat.id, symbol, direction, target, current_price, creator, source_link, photo_file_id),
        )
        alert_ids.append(result.last_insert_rowid)

    saved = "\n".join(
        f"• #{alert_id} {direction} → Ziel: <code>{target:g}</code>"
        for alert_id, (_, direction, target) in zip(alert_ids, parsed_alerts)
    )
    
    # Signale bleiben zur Übersicht im Kanal stehen, keine automatische Löschung!
    await update.message.reply_text(
        f"🔔 <b>Alarme gespeichert!</b>\n{saved}", parse_mode=ParseMode.HTML
    )


async def list_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not await is_authorised(update, context) or update.effective_chat is None or user_msg is None:
        return

    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "SELECT id, symbol, direction, target_price, entry_price, created_by, source_link, photo_file_id "
        "FROM alerts WHERE chat_id = ? ORDER BY id",
        (update.effective_chat.id,),
    )
    alerts = result.rows
    if not alerts:
        bot_msg = await user_msg.reply_text("🔔 No active alerts.")
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    session = context.application.bot_data["http_session"]
    symbols = {alert[1] for alert in alerts}
    fetched_prices = await asyncio.gather(*(get_price(session, s) for s in symbols))
    prices = dict(zip(symbols, fetched_prices))

    lines = ["📊 <b>Active Alerts &amp; Current Prices</b>"]
    keyboard = []

    for number, alert in enumerate(alerts, start=1):
        alert_id, symbol, direction, target, entry, creator, source_link, photo_file_id = alert
        current = prices.get(symbol)
        if current is None:
            current_text, progress = "unavailable", 0
        else:
            current_text = f"{current:g} USDT"
            total_diff = abs(target - entry)
            if total_diff > 0:
                current_diff = abs(current - entry)
                progress = min(100, int((current_diff / total_diff) * 100))
            else:
                progress = 100

        lines.append(
            f"{number}. <b>#{symbol}</b> {direction}\n"
            f"   Einstieg: <code>{entry:g}</code> → Ziel: <code>{target:g}</code>\n"
            f"   Kurs: <code>{current_text}</code> | Fortschritt: <code>{progress}%</code>"
        )
        if photo_file_id:
            keyboard.append([InlineKeyboardButton(f"🖼️ Show Image #{alert_id}", callback_data=f"show_img:{alert_id}")])

    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    bot_msg = await user_msg.reply_text("\n\n".join(lines), reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    
    # Liste nach 30 Sekunden im Hintergrund sauber löschen
    asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))


async def show_trade_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None or not query.data.startswith("show_img:"):
        return
    await query.answer()
    alert_id = int(query.data.split(":", maxsplit=1)[1])
    
    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "SELECT symbol, direction, target_price, photo_file_id FROM alerts WHERE id = ? AND chat_id = ?",
        (alert_id, query.message.chat_id),
    )
    alert = result.rows[0] if result.rows else None
    if alert is None or not alert[3]:
        await query.message.reply_text("The original image is no longer available.")
        return

    symbol, direction, target, photo_file_id = alert[0], alert[1], alert[2], alert[3]
    await query.message.reply_photo(
        photo=photo_file_id,
        caption=f"#{symbol} {direction} | Target: {target:g} USDT",
    )


async def delete_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not await is_authorised(update, context) or update.effective_chat is None or user_msg is None:
        return

    if len(context.args) != 1 or not context.args[0].isdigit():
        bot_msg = await user_msg.reply_text("Verwendung: <code>/delete 12</code>", parse_mode=ParseMode.HTML)
        asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))
        return

    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "DELETE FROM alerts WHERE id = ? AND chat_id = ?", (int(context.args[0]), update.effective_chat.id)
    )
    
    bot_msg = await user_msg.reply_text(
        "🗑️ Alarm gelöscht." if result.rows_affected > 0 else "Kein Alarm mit dieser Nummer gefunden."
    )
    asyncio.create_task(delete_messages_later(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30))


# 7. HINTERGRUND-TASK: ALARME PRÜFEN
async def check_alerts(application: Application) -> None:
    session: aiohttp.ClientSession = application.bot_data["http_session"]
    client = application.bot_data["db_client"]
    while True:
        try:
            result = await client.execute(
                "SELECT id, chat_id, symbol, direction, target_price FROM alerts ORDER BY id"
            )
            alerts = result.rows

            prices: dict[str, float | None] = {}
            for row in alerts:
                symbol = row[2]
                if symbol not in prices:
                    prices[symbol] = await get_price(session, symbol)

            for row in alerts:
                alert_id, chat_id, symbol, direction, target = row[0], row[1], row[2], row[3], row[4]
                price = prices[symbol]
                reached = price is not None and (
                    (direction == "LONG" and price >= target) or 
                    (direction == "SHORT" and price <= target)
                )
                if not reached:
                    continue
                try:
                    await application.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"🎯 <b>Ziel erreicht!</b>\n#{symbol} {direction}\n"
                            f"Ziel: <code>{target:g}</code> | Kurs: <code>{price:g}</code>"
                        ),
                        parse_mode=ParseMode.HTML,
                    )
                    await client.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
                except Exception:
                    LOGGER.exception("Could not notify chat %s for alert %s", chat_id, alert_id)
        except Exception as e:
            LOGGER.error("Fehler im Alert-Check-Loop: %s", e)
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


# 8. MINI-WEBSERVER FÜR PORT-BINDING
async def handle_ping(request):
    return web.Response(text="Bot is running!")


# 9. TELEGRAM INITIALISIERUNG & SHUTDOWN
async def post_init(application: Application) -> None:
    db_url = os.getenv("TURSO_DATABASE_URL")
    db_token = os.getenv("TURSO_AUTH_TOKEN")
    if not db_url or not db_token:
        raise RuntimeError("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN environment variables must be set.")
    
    # Bypass für den Render WebSocket-Fehler: libsql:// wird in stabile HTTPS-Anfragen geändert
    if db_url.startswith("libsql://"):
        db_url = db_url.replace("libsql://", "https://")

    db_client = libsql_client.create_client(url=db_url, auth_token=db_token)
    application.bot_data["db_client"] = db_client
    
    await initialise_database(db_client)
    application.bot_data["http_session"] = aiohttp.ClientSession()
    application.bot_data["alert_task"] = asyncio.create_task(check_alerts(application))
    
    # Webserver starten, um Renders Port-Binding Fehler zu verhindern
    app = web.Application()
    app.router.add_get("/", handle_ping)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    application.bot_data["web_runner"] = runner
    LOGGER.info("Web server started on port %s", port)


async def post_shutdown(application: Application) -> None:
    task = application.bot_data.get("alert_task")
    if task:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
            
    session = application.bot_data.get("http_session")
    if session:
        await session.close()
        
    db_client = application.bot_data.get("db_client")
    if db_client:
        await db_client.close()
        
    runner = application.bot_data.get("web_runner")
    if runner:
        await runner.cleanup()


# 10. ENTRYPOINT
def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        LOGGER.error("Kein TELEGRAM_BOT_TOKEN in der .env gefunden!")
        return

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Befehle registrieren
    application.add_handler(CommandHandler("addid", add_user_id))
    application.add_handler(CommandHandler("deleteid", delete_user_id))
    application.add_handler(CommandHandler("list", list_user_ids))
    application.add_handler(CommandHandler("delete", delete_alert))
    application.add_handler(CommandHandler("alerts", list_alerts))
    
    # Callback für Bildanzeige bei Inline-Buttons
    application.add_handler(CallbackQueryHandler(show_trade_image, pattern="^show_img:"))
    
    # Handler für Bilder mit Alarmsignalen (z.B. TradingView-Bild mit Signal im Text)
    application.add_handler(MessageHandler(filters.PHOTO, add_alert))

    LOGGER.info("Starte Polling...")
    application.run_polling()


if __name__ == "__main__":
    main()
