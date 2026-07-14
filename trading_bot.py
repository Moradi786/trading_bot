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
import sqlite3
from contextlib import suppress
from pathlib import Path

import aiohttp
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

DB_PATH = Path("alerts.db")
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


def initialise_database() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
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
        # Allows an existing database made by an earlier bot version to keep working.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(alerts)")}
        if "entry_price" not in columns:
            connection.execute("ALTER TABLE alerts ADD COLUMN entry_price REAL")
            connection.execute("UPDATE alerts SET entry_price = target_price WHERE entry_price IS NULL")
        if "source_link" not in columns:
            connection.execute("ALTER TABLE alerts ADD COLUMN source_link TEXT")
        if "photo_file_id" not in columns:
            connection.execute("ALTER TABLE alerts ADD COLUMN photo_file_id TEXT")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS authorised_users (
                user_id INTEGER PRIMARY KEY,
                added_by INTEGER NOT NULL,
                display_name TEXT,
                added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        user_columns = {row[1] for row in connection.execute("PRAGMA table_info(authorised_users)")}
        if "display_name" not in user_columns:
            connection.execute("ALTER TABLE authorised_users ADD COLUMN display_name TEXT")


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
    # t.me/c links work for private supergroups and channels (IDs begin with -100).
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


def is_authorised(update: Update) -> bool:
    user = update.effective_user
    if user is None:
        return False
    if user.id in ADMIN_USER_IDS or user.id in ALLOWED_USER_IDS:
        return True
    if not ADMIN_USER_IDS and not ALLOWED_USER_IDS:
        return True
    with sqlite3.connect(DB_PATH) as connection:
        return connection.execute(
            "SELECT 1 FROM authorised_users WHERE user_id = ?", (user.id,)
        ).fetchone() is not None


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
    with sqlite3.connect(DB_PATH) as connection:
        for user_id, display_name in entries:
            connection.execute(
                """
                INSERT INTO authorised_users (user_id, added_by, display_name) VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET added_by = excluded.added_by, display_name = excluded.display_name
                """,
                (user_id, update.effective_user.id, display_name),
            )
    added = "\n".join(
        f"• {html.escape(name) if name else 'No name'} — <code>{user_id}</code>"
        for user_id, name in entries
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
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("DELETE FROM authorised_users WHERE user_id = ?", (user_id,))
    message = (
        f"🗑️ User ID <code>{user_id}</code> removed."
        if cursor.rowcount
        else "User ID was not in the added list."
    )
    await update.message.reply_text(message, parse_mode=ParseMode.HTML)


async def list_user_ids(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update):
        await update.message.reply_text("Only an admin can view allowed user IDs.")
        return
    with sqlite3.connect(DB_PATH) as connection:
        users = connection.execute(
            "SELECT user_id, display_name FROM authorised_users ORDER BY added_at, user_id"
        ).fetchall()
    if not users:
        await update.message.reply_text("No extra user IDs have been added yet.")
        return
    lines = ["👥 <b>Allowed users</b>"]
    for number, (user_id, display_name) in enumerate(users, start=1):
        name = html.escape(display_name) if display_name else "No name"
        lines.append(f"{number}. <b>{name}</b> — <code>{user_id}</code>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def add_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorised(update) or update.message is None:
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
    with sqlite3.connect(DB_PATH) as connection:
        alert_ids: list[int] = []
        for _, direction, target in parsed_alerts:
            cursor = connection.execute(
                "INSERT INTO alerts (chat_id, symbol, direction, target_price, entry_price, created_by, source_link, photo_file_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (update.effective_chat.id, symbol, direction, target, current_price, creator, source_link, photo_file_id),
            )
            alert_ids.append(cursor.lastrowid)

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
    if not is_authorised(update) or update.effective_chat is None or update.message is None:
        return
    with sqlite3.connect(DB_PATH) as connection:
        alerts = connection.execute(
            "SELECT id, symbol, direction, target_price, entry_price, created_by, source_link, photo_file_id "
            "FROM alerts WHERE chat_id = ? ORDER BY id",
            (update.effective_chat.id,),
        ).fetchall()
    if not alerts:
        await update.message.reply_text("🔔 No active alerts.")
        return
    session: aiohttp.ClientSession = context.application.bot_data["http_session"]
    symbols = list(dict.fromkeys(alert[1] for alert in alerts))
    fetched_prices = await asyncio.gather(*(get_price(session, symbol) for symbol in symbols))
    prices = dict(zip(symbols, fetched_prices))
    lines = ["📊 <b>Active Alerts &amp; Current Prices</b>"]
    image_buttons: list[list[InlineKeyboardButton]] = []
    for number, (alert_id, symbol, direction, target, entry, creator, source_link, photo_file_id) in enumerate(alerts, start=1):
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
    with sqlite3.connect(DB_PATH) as connection:
        alert = connection.execute(
            "SELECT symbol, direction, target_price, photo_file_id FROM alerts WHERE id = ? AND chat_id = ?",
            (alert_id, query.message.chat_id),
        ).fetchone()
    if alert is None or not alert[3]:
        await query.message.reply_text("The original image is no longer available.")
        return
    symbol, direction, target, photo_file_id = alert
    await query.message.reply_photo(
        photo=photo_file_id,
        caption=f"#{symbol} {direction} | Target: {target:g} USDT",
    )


async def delete_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorised(update) or update.effective_chat is None or update.message is None:
        return
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Verwendung: <code>/delete 12</code>", parse_mode=ParseMode.HTML)
        return
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            "DELETE FROM alerts WHERE id = ? AND chat_id = ?", (int(context.args[0]), update.effective_chat.id)
        )
    await update.message.reply_text("🗑️ Alarm gelöscht." if cursor.rowcount else "Kein Alarm mit dieser Nummer gefunden.")


async def check_alerts(application: Application) -> None:
    session: aiohttp.ClientSession = application.bot_data["http_session"]
    while True:
        with sqlite3.connect(DB_PATH) as connection:
            alerts = connection.execute(
                "SELECT id, chat_id, symbol, direction, target_price FROM alerts ORDER BY id"
            ).fetchall()

        prices: dict[str, float | None] = {}
        for _, _, symbol, _, _ in alerts:
            if symbol not in prices:
                prices[symbol] = await get_price(session, symbol)

        for alert_id, chat_id, symbol, direction, target in alerts:
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
                with sqlite3.connect(DB_PATH) as connection:
                    connection.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            except Exception:
                LOGGER.exception("Could not notify chat %s for alert %s", chat_id, alert_id)
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


async def post_init(application: Application) -> None:
    initialise_database()
    application.bot_data["http_session"] = aiohttp.ClientSession()
    application.bot_data["alert_task"] = asyncio.create_task(check_alerts(application))


async def post_shutdown(application: Application) -> None:
    task = application.bot_data.get("alert_task")
    if task:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    session = application.bot_data.get("http_session")
    if session:
        await session.close()


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
