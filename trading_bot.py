"""Telegram crypto target-alert bot.

Send a photo with a caption such as:
    BTC LONG 70000
    ETHUSDT SHORT 2200

The bot stores the target, checks Binance Futures prices, and sends a message
to the same chat when the target is reached.
"""

import asyncio
import html
import logging
import os
import re
from contextlib import suppress

import aiohttp
import libsql_client
from aiohttp import web
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

load_dotenv()

BINANCE_PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
SYMBOL_PATTERN = re.compile(r"^\s*#?([A-Za-z0-9/_-]+)")
DIRECTION_TARGET_PATTERN = re.compile(
    r"\b(LONG|SHORT)\b\s*(?:TARGET(?:\s+PRICE)?|TP|PRICE)?\s*[:=@-]?\s*([0-9][0-9.,]*)",
    re.IGNORECASE,
)


def allowed_users() -> set[int]:
    raw_value = os.getenv("ALLOWED_USER_IDS", "").strip()
    if not raw_value:
        return set()
    try:
        return {int(value.strip()) for value in raw_value.split(",") if value.strip()}
    except ValueError as error:
        raise RuntimeError("ALLOWED_USER_IDS must contain only comma-separated numeric IDs.") from error


ALLOWED_USER_IDS = allowed_users()
ADMIN_USER_IDS = {
    int(value.strip())
    for value in os.getenv("ADMIN_USER_IDS", "").split(",")
    if value.strip()
}
CHECK_INTERVAL_SECONDS = max(10, int(os.getenv("CHECK_INTERVAL_SECONDS", "30")))


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


def normalise_symbol(value: str) -> str:
    symbol = re.sub(r"[^A-Z0-9]", "", value.upper())
    if not symbol:
        raise ValueError("Kein gültiges Symbol gefunden.")
    return symbol if symbol.endswith("USDT") else f"{symbol}USDT"


def message_link(chat_id: int, chat_username: str | None, chat_type: str, message_id: int) -> str | None:
    """Return the Telegram link to a source photo when Telegram permits one."""
    if chat_type in {"group", "supergroup", "channel"} and chat_username:
        return f"https://t.me/{chat_username}/{message_id}"
    raw_chat_id = str(chat_id)
    if chat_type in {"supergroup", "channel"} and raw_chat_id.startswith("-100"):
        return f"https://t.me/c/{raw_chat_id[4:]}/{message_id}"
    return None


def parse_caption(caption: str) -> list[tuple[str, str, float]]:
    """Read one or two targets, e.g. ``BTC LONG 70000 SHORT 62000``."""
    symbol_match = SYMBOL_PATTERN.match(caption)
    if not symbol_match:
        return []
    symbol = normalise_symbol(symbol_match.group(1))
    alerts: list[tuple[str, str, float]] = []
    for match in DIRECTION_TARGET_PATTERN.finditer(caption):
        direction, raw_target = match.groups()
        target = float(raw_target.replace(",", "."))
        if target > 0:
            alerts.append((symbol, direction.upper(), target))
    return alerts


async def get_price(session: aiohttp.ClientSession, symbol: str) -> float | None:
    try:
        async with session.get(BINANCE_PRICE_URL, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                LOGGER.warning("Price request for %s returned HTTP %s", symbol, response.status)
                return None
            payload = await response.json()
            return float(payload["price"])
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, TypeError, ValueError) as error:
        LOGGER.warning("Could not fetch %s price: %s", symbol, error)
        return None


async def reply_in_chunks(message, text: str, parse_mode: ParseMode = ParseMode.HTML) -> None:
    """Telegram messages are limited to 4,096 characters; keep long alarm lists readable."""
    while text:
        if len(text) <= 3800:
            await message.reply_text(text, parse_mode=parse_mode)
            return
        split_at = text.rfind("\n\n", 0, 3800)
        if split_at < 1:
            split_at = text.rfind("\n", 0, 3800)
        if split_at < 1:
            split_at = 3800
        await message.reply_text(text[:split_at], parse_mode=parse_mode)
        text = text[split_at:].lstrip()


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
    return update.effective_user is not None and update.effective_user.id in ADMIN_USER_IDS


async def add_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update):
        await update.message.reply_text("Only an admin can add user IDs.")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "Usage: <code>/addid 123456789 AMIRI</code>", parse_mode=ParseMode.HTML
        )
        return
    entries: list[tuple[int, str | None]] = []
    user_id: int | None = None
    name_parts: list[str] = []
    for part in context.args:
        if part.lower().startswith("/addid"):
            if user_id is not None:
                entries.append((user_id, " ".join(name_parts).strip() or None))
            user_id, name_parts = None, []
        elif user_id is None and part.isdigit():
            user_id = int(part)
        elif user_id is not None:
            name_parts.append(part)
    if user_id is not None:
        entries.append((user_id, " ".join(name_parts).strip() or None))
    if not entries:
        await update.message.reply_text(
            "Usage: <code>/addid 123456789 AMIRI</code>", parse_mode=ParseMode.HTML
        )
        return
    client = context.application.bot_data["db_client"]
    for u_id, display_name in entries:
        await client.execute(
            """
            INSERT INTO authorised_users (user_id, added_by, display_name) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET added_by = excluded.added_by, display_name = excluded.display_name
            """,
            (u_id, update.effective_user.id, display_name),
        )
    added = "\n".join(
        f"• {html.escape(name) if name else 'No name'} — <code>{u_id}</code>"
        for u_id, name in entries
    )
    await update.message.reply_text(
        f"✅ User{'s' if len(entries) > 1 else ''} can now add alerts:\n{added}", parse_mode=ParseMode.HTML
    )


async def delete_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update):
        await update.message.reply_text("Only an admin can remove user IDs.")
        return
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Usage: <code>/deleteid 123456789</code>", parse_mode=ParseMode.HTML)
        return
    user_id = int(context.args[0])
    client = context.application.bot_data["db_client"]
    result = await client.execute("DELETE FROM authorised_users WHERE user_id = ?", (user_id,))
    message = (
        f"🗑️ User ID <code>{user_id}</code> removed."
        if result.rows_affected > 0
        else "User ID was not in the added list."
    )
    await update.message.reply_text(message, parse_mode=ParseMode.HTML)


async def list_user_ids(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update):
        await update.message.reply_text("Only an admin can view allowed user IDs.")
        return
    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "SELECT user_id, display_name FROM authorised_users ORDER BY added_at, user_id"
    )
    users = result.rows
    if not users:
        await update.message.reply_text("No extra user IDs have been added yet.")
        return
    lines = ["👥 <b>Allowed users</b>"]
    for number, row in enumerate(users, start=1):
        u_id, display_name = row[0], row[1]
        name = html.escape(display_name) if display_name else "No name"
        lines.append(f"{number}. <b>{name}</b> — <code>{u_id}</code>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def add_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await is_authorised(update, context) or update.message is None:
        return
    parsed_alerts = parse_caption(update.message.caption or "")
    if not parsed_alerts:
        await update.message.reply_text(
            "Ich konnte den Alarm nicht lesen. Schreibe in die Bildbeschreibung z. B.:\n"
            "<code>BTC LONG 70000</code>\noder für beide Richtungen:\n"
            "<code>BTC LONG 70000 SHORT 62000</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    symbol = parsed_alerts[0][0]
    session: aiohttp.ClientSession = context.application.bot_data["http_session"]
    current_price = await get_price(session, symbol)
    if current_price is None:
        await update.message.reply_text(f"Für {symbol} wurde kein Futures-Kurs gefunden. Prüfe das Symbol.")
        return

    assert update.effective_chat is not None
    creator = update.effective_user.full_name
    source_link = message_link(
        update.effective_chat.id,
        update.effective_chat.username,
        update.effective_chat.type,
        update.message.message_id,
    )
    photo_file_id = update.message.photo[-1].file_id
    client = context.application.bot_data["db_client"]
    alert_ids: list[int] = []
    for _, direction, target in parsed_alerts:
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
    await update.message.reply_text(
        f"✅ Alarm{'e' if len(alert_ids) > 1 else ''} gespeichert – #{symbol} | BY {html.escape(creator)}\n"
        f"{saved}\nAktueller Kurs: <code>{current_price:g}</code>",
        parse_mode=ParseMode.HTML,
    )


async def list_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await is_authorised(update, context) or update.effective_chat is None or update.message is None:
        return
    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "SELECT id, symbol, direction, target_price, entry_price, created_by, source_link, photo_file_id "
        "FROM alerts WHERE chat_id = ? ORDER BY id",
        (update.effective_chat.id,),
    )
    alerts = result.rows
    if not alerts:
        await update.message.reply_text("🔔 No active alerts.")
        return
    session: aiohttp.ClientSession = context.application.bot_data["http_session"]
    symbols = list(dict.fromkeys(alert[1] for alert in alerts))
    fetched_prices = await asyncio.gather(*(get_price(session, symbol) for symbol in symbols))
    prices = dict(zip(symbols, fetched_prices))
    lines = ["📊 <b>Active Alerts &amp; Current Prices</b>"]
    image_buttons: list[list[InlineKeyboardButton]] = []
    for number, alert in enumerate(alerts, start=1):
        alert_id, symbol, direction, target, entry, creator, source_link, photo_file_id = alert
        current = prices.get(symbol)
        if current is None:
            current_text, progress = "unavailable", 0
        else:
            current_text = f"<code>{current:g} USDT</code>"
            distance = target - entry if direction == "LONG" else entry - target
            travelled = current - entry if direction == "LONG" else entry - current
            progress = 0 if distance <= 0 else max(0, min(100, round(travelled / distance * 100)))
        filled = round(progress / 10)
        colour = "🟩" if direction == "LONG" else "🟥"
        bar = colour * filled + "⬛" * (10 - filled)
        if photo_file_id:
            image_buttons.append([InlineKeyboardButton(f"🔗 🖼 View Image #{symbol}", callback_data=f"image:{alert_id}")])
        lines.append(
            f"\n{number}. <b>#{symbol}</b> | BY {html.escape(creator)}\n"
            f"{'🟢' if direction == 'LONG' else '🔴'} {direction}\n"
            f"🎯 Target: <code>{target:g} USDT</code>\n"
            f"⚡ Current: {current_text}\n"
            f"📈 To Target: {bar} {progress}%"
        )
    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(image_buttons) if image_buttons else None,
    )


async def show_trade_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.message is None:
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
    if not await is_authorised(update, context) or update.effective_chat is None or update.message is None:
        return
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Verwendung: <code>/delete 12</code>", parse_mode=ParseMode.HTML)
        return
    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "DELETE FROM alerts WHERE id = ? AND chat_id = ?", (int(context.args[0]), update.effective_chat.id)
    )
    await update.message.reply_text("🗑️ Alarm gelöscht." if result.rows_affected > 0 else "Kein Alarm mit dieser Nummer gefunden.")


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
                reached = price is not None and ((direction == "LONG" and price >= target) or (direction == "SHORT" and price <= target))
                if not reached:
                    continue
                try:
                    await application.bot.send_message(
                        chat_id=chat_id,
                        text=(f"🎯 <b>Ziel erreicht!</b>\n#{symbol} {direction}\n"
                              f"Ziel: <code>{target:g}</code> | Kurs: <code>{price:g}</code>"),
                        parse_mode=ParseMode.HTML,
                    )
                    await client.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
                except Exception:
                    LOGGER.exception("Could not notify chat %s for alert %s", chat_id, alert_id)
        except Exception as e:
            LOGGER.error("Fehler im Alert-Check-Loop: %s", e)
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


async def handle_ping(request):
    return web.Response(text="Bot is running!")


async def post_init(application: Application) -> None:
    db_url = os.getenv("TURSO_DATABASE_URL")
    db_token = os.getenv("TURSO_AUTH_TOKEN")
    if not db_url or not db_token:
        raise RuntimeError("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN environment variables must be set.")
    
    db_client = libsql_client.create_client(url=db_url, auth_token=db_token)
    application.bot_data["db_client"] = db_client
    
    await initialise_database(db_client)
    application.bot_data["http_session"] = aiohttp.ClientSession()
    application.bot_data["alert_task"] = asyncio.create_task(check_alerts(application))
    
    # Start web server to prevent Render from idling or failing to bind to port
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


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the bot.")
    application = (
        Application.builder().token(token).post_init(post_init).post_shutdown(post_shutdown).build()
    )
    application.add_handler(CommandHandler(("alarms", "alarm"), list_alerts))
    application.add_handler(CommandHandler("delete", delete_alert))
    application.add_handler(CommandHandler("addid", add_user_id))
    application.add_handler(CommandHandler("deleteid", delete_user_id))
    application.add_handler(CommandHandler(("list", "listid"), list_user_ids))
    application.add_handler(CallbackQueryHandler(show_trade_image, pattern=r"^image:\d+$"))
    application.add_handler(MessageHandler(filters.PHOTO, add_alert))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
