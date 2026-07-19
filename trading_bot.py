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

# Initialize Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

load_dotenv()

DB_PATH = Path("alerts.db")
BINANCE_PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price"

# Global set to keep strong references to running background tasks
RUNNING_TASKS = set()

# Regex Patterns
SYMBOL_PATTERN = re.compile(r"^\s*#([A-Za-z0-9/_-]+)")
DIRECTION_TARGET_PATTERN = re.compile(r"(LONG|SHORT)\s+([\d.,]+)", re.IGNORECASE)

ADMIN_USER_IDS = [
    int(num) 
    for num in re.findall(r"\d+", os.getenv("ADMIN_USER_IDS", "6147760453 MORADI"))
]
ALLOWED_USER_IDS = [
    int(num) 
    for num in re.findall(r"\d+", os.getenv("ALLOWED_USER_IDS", "6673849133 AMIRI , 5544021969 ALI"))
]
CHECK_INTERVAL_SECONDS = max(10, int(os.getenv("CHECK_INTERVAL_SECONDS", "30")))


# 1. DATABASE SETUP
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
            near_1_sent INTEGER DEFAULT 0,
            near_05_sent INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    with suppress(Exception):
        await client.execute("ALTER TABLE alerts ADD COLUMN near_1_sent INTEGER DEFAULT 0")
    with suppress(Exception):
        await client.execute("ALTER TABLE alerts ADD COLUMN near_05_sent INTEGER DEFAULT 0")

    await client.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            target_price REAL NOT NULL,
            entry_price REAL NOT NULL,
            triggered_price REAL NOT NULL,
            created_by TEXT NOT NULL,
            photo_file_id TEXT,
            source_link TEXT,
            triggered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    with suppress(Exception):
        await client.execute("ALTER TABLE alert_history ADD COLUMN photo_file_id TEXT")
    with suppress(Exception):
        await client.execute("ALTER TABLE alert_history ADD COLUMN source_link TEXT")

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


# 2. HELPER FUNCTIONS FOR RSI & VOLUME CALCULATION
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    seed = deltas[:period]
    up = sum([d for d in seed if d > 0]) / period
    down = sum([-d for d in seed if d < 0]) / period
    
    if down == 0:
        rs = float('inf')
    else:
        rs = up / down
    rsi = 100 - (100 / (1 + rs))
    
    for d in deltas[period:]:
        gain = d if d > 0 else 0
        loss = -d if d < 0 else 0
        up = (up * (period - 1) + gain) / period
        down = (down * (period - 1) + loss) / period
        if down == 0:
            rs = float('inf')
        else:
            rs = up / down
        rsi = 100 - (100 / (1 + rs))
    return rsi


async def get_market_data(session: aiohttp.ClientSession, symbol: str, interval: str = "1h") -> dict:
    norm_symbol = normalise_symbol(symbol)
    if not norm_symbol.endswith("USDT") and not norm_symbol.endswith("BUSD"):
        norm_symbol += "USDT"
        
    market_metrics = {"price": None, "volume": "N/A", "rsi": "N/A"}
    
    # Fetch 24h Ticker data (Price and Volume)
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        async with session.get(url, params={"symbol": norm_symbol}, timeout=aiohttp.ClientTimeout(total=3)) as response:
            if response.status == 200:
                res_data = await response.json()
                market_metrics["price"] = float(res_data["lastPrice"])
                raw_vol = float(res_data["quoteVolume"]) # Volume in USDT
                if raw_vol >= 1_000_000:
                    market_metrics["volume"] = f"{raw_vol / 1_000_000:.1f}M"
                elif raw_vol >= 1_000:
                    market_metrics["volume"] = f"{raw_vol / 1_000:.1f}K"
                else:
                    market_metrics["volume"] = f"{raw_vol:.0f}"
    except Exception:
        market_metrics["price"] = await get_price(session, symbol)

    # Fetch Klines for standard RSI(14) calculation
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": norm_symbol, "interval": interval, "limit": 50}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as response:
            if response.status == 200:
                klines = await response.json()
                close_prices = [float(k[4]) for k in klines]
                rsi_val = calculate_rsi(close_prices)
                if rsi_val is not None:
                    market_metrics["rsi"] = f"{rsi_val:.1f}"
    except Exception:
        pass
        
    return market_metrics


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
    norm_symbol = normalise_symbol(symbol)
    if not norm_symbol.endswith("USDT") and not norm_symbol.endswith("BUSD"):
        norm_symbol += "USDT"

    try:
        async with session.get(
            BINANCE_PRICE_URL,
            params={"symbol": norm_symbol},
            timeout=aiohttp.ClientTimeout(total=3)
        ) as response:
            if response.status == 200:
                data = await response.json()
                return float(data["price"])
    except Exception:
        pass

    try:
        bybit_url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "linear", "symbol": norm_symbol}
        async with session.get(
            bybit_url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=3)
        ) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("retCode") == 0 and data["result"]["list"]:
                    return float(data["result"]["list"][0]["lastPrice"])
    except Exception:
        pass

    return None


def parse_price(price_str: str) -> float | None:
    price_str = price_str.strip()
    if "," in price_str and "." in price_str:
        if price_str.find(".") < price_str.find(","):
            price_str = price_str.replace(".", "").replace(",", ".")
        else:
            price_str = price_str.replace(",", "")
    elif "," in price_str:
        price_str = price_str.replace(",", ".")
    try:
        return float(price_str)
    except ValueError:
        return None


def parse_caption(caption: str) -> list[tuple[str, str, float]]:
    alerts = []
    symbol_match = re.search(r"#([A-Za-z0-9/_-]+)", caption)
    if not symbol_match:
        return alerts
    symbol = symbol_match.group(1).upper()
    
    inline_match = re.search(r"\b(LONG|SHORT)\b\s+([\d.,]+)", caption, re.IGNORECASE)
    if inline_match:
        direction = inline_match.group(1).upper()
        target = parse_price(inline_match.group(2))
        if target is not None:
            alerts.append((symbol, direction, target))
            return alerts
            
    dir_match = re.search(r"\b(LONG|SHORT)\b", caption, re.IGNORECASE)
    target_match = re.search(r"\b(?:target|ziel|tp|entry|price)\s*:?\s*([\d.,]+)", caption, re.IGNORECASE)
    
    if dir_match and target_match:
        direction = dir_match.group(1).upper()
        target = parse_price(target_match.group(1))
        if target is not None:
            alerts.append((symbol, direction, target))
            return alerts

    lines = [line.strip() for line in caption.split("\n") if line.strip()]
    for line in lines:
        match = DIRECTION_TARGET_PATTERN.search(line)
        if match:
            direction = match.group(1).upper()
            target = parse_price(match.group(2))
            if target is not None:
                alerts.append((symbol, direction, target))
                return alerts
                
    return alerts


# AUTOMATIC MESSAGE CLEANUP
async def delete_messages_later(bot, chat_id, message_ids, delay=30):
    await asyncio.sleep(delay)
    for msg_id in message_ids:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except TelegramError as e:
            LOGGER.warning(f"Could not delete message {msg_id} in chat {chat_id}: {e}")


def track_background_cleanup(bot, chat_id, message_ids, delay=30):
    task = asyncio.create_task(delete_messages_later(bot, chat_id, message_ids, delay))
    RUNNING_TASKS.add(task)
    task.add_done_callback(RUNNING_TASKS.discard)


# AUTHORISATION CHECKS
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


# 3. COMMANDS: USER MANAGEMENT (ADMIN ONLY)
async def add_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not is_admin(update):
        bot_msg = await user_msg.reply_text("Only an admin can add allowed user IDs.")
        track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)
        return

    if len(context.args) < 2:
        bot_msg = await user_msg.reply_text("Usage: <code>/addid 123456789 AMIRI</code>", parse_mode=ParseMode.HTML)
        track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)
        return

    try:
        u_id = int(context.args[0])
        display_name = " ".join(context.args[1:])
        entries = [(u_id, display_name)]
    except ValueError:
        bot_msg = await user_msg.reply_text("Invalid User ID.")
        track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)
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
        f"✅ User{'s' if len(entries) > 1 else ''} authorized successfully:\n{added}", parse_mode=ParseMode.HTML
    )
    track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)


async def delete_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not is_admin(update):
        bot_msg = await user_msg.reply_text("Only an admin can delete allowed user IDs.")
        track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)
        return

    if not context.args or not context.args[0].isdigit():
        bot_msg = await user_msg.reply_text("Usage: <code>/deleteid 123456789</code>", parse_mode=ParseMode.HTML)
        track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)
        return

    user_id = int(context.args[0])
    client = context.application.bot_data["db_client"]
    result = await client.execute("DELETE FROM authorised_users WHERE user_id = ?", (user_id,))
    
    message = (
        f"🗑️ User ID <code>{user_id}</code> removed."
        if result.rows_affected > 0
        else "User ID was not found in the authorized list."
    )
    bot_msg = await user_msg.reply_text(message, parse_mode=ParseMode.HTML)
    track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)


async def list_user_ids(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not is_admin(update):
        bot_msg = await user_msg.reply_text("Only an admin can view allowed user IDs.")
        track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)
        return

    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "SELECT user_id, display_name FROM authorised_users ORDER BY added_at, user_id"
    )
    users = result.rows
    if not users:
        bot_msg = await user_msg.reply_text("No extra user IDs authorized yet.")
        track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)
        return

    lines = ["👥 <b>Allowed Users List</b>"]
    for number, row in enumerate(users, start=1):
        u_id, display_name = row[0], row[1]
        name = html.escape(display_name) if display_name else "No name"
        lines.append(f"{number}. <b>{name}</b> — <code>{u_id}</code>")

    bot_msg = await user_msg.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)


# 4. COMMANDS: ALERT MANAGEMENT
async def add_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await is_authorised(update, context) or update.message is None:
        return

    message = update.message
    text_to_parse = ""
    photo_file_id = None

    if message.photo:
        text_to_parse = message.caption or ""
        photo_file_id = message.photo[-1].file_id
    elif message.reply_to_message:
        replied = message.reply_to_message
        if replied.photo:
            text_to_parse = replied.caption or ""
            photo_file_id = replied.photo[-1].file_id
        else:
            text_to_parse = replied.text or ""
    else:
        text_to_parse = message.text or ""

    if text_to_parse.lower().startswith("/alarm"):
        text_to_parse = re.sub(r"^/alarm\s*", "", text_to_parse, flags=re.IGNORECASE).strip()

    parsed_alerts = parse_caption(text_to_parse)
    
    if not parsed_alerts:
        if message.text and message.text.lower().startswith("/alarm"):
            bot_msg = await message.reply_text(
                "❌ <b>Error:</b> Invalid Format!\n\n"
                "The ticker/coin <b>must</b> start with a <code>#</code>.\n"
                "Example: <code>/alarm #TRXUSDT SHORT 0.3238</code>",
                parse_mode=ParseMode.HTML
            )
            track_background_cleanup(context.bot, update.effective_chat.id, [bot_msg.message_id], 30)
        return

    chat = update.effective_chat
    message_id = message.message_id
    source_link = message_link(chat.id, chat.username, chat.type, message_id) if chat else None

    client = context.application.bot_data["db_client"]
    session = context.application.bot_data["http_session"]
    creator = update.effective_user.username or update.effective_user.first_name or "Unknown"

    if len(parsed_alerts) == 1:
        symbol, direction, target = parsed_alerts[0]
        market = await get_market_data(session, symbol)
        current_price = market["price"] or target
        
        result = await client.execute(
            "INSERT INTO alerts (chat_id, symbol, direction, target_price, entry_price, created_by, source_link, photo_file_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (update.effective_chat.id, symbol, direction, target, current_price, creator, source_link, photo_file_id),
        )
        alert_id = result.last_insert_rowid
        
        dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
        reply_text = (
            f"✅ <b>Alert Saved - #{symbol} | BY {creator}</b>\n"
            f"• #{alert_id} {dir_emoji} → Target: <code>{target:g}</code>\n"
            f"Current Price: <code>{current_price:g}</code>\n"
            f"📊 Vol (24h): <code>{market['volume']}</code> | 🕒 RSI (1h): <code>{market['rsi']}</code>"
        )
    else:
        saved_lines = []
        for symbol, direction, target in parsed_alerts:
            market = await get_market_data(session, symbol)
            current_price = market["price"] or target
            result = await client.execute(
                "INSERT INTO alerts (chat_id, symbol, direction, target_price, entry_price, created_by, source_link, photo_file_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (update.effective_chat.id, symbol, direction, target, current_price, creator, source_link, photo_file_id),
            )
            alert_id = result.last_insert_rowid
            dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
            saved_lines.append(f"• #{alert_id} {dir_emoji} → Target: <code>{target:g}</code> (RSI: {market['rsi']})")
            
        reply_text = (
            f"🔔 <b>Multiple Alerts Saved!</b>\n" + "\n".join(saved_lines)
        )
    
    bot_msg = await message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    track_background_cleanup(context.bot, update.effective_chat.id, [bot_msg.message_id], 60)


# CRASH-PROOF LIST_ALERTS SHOWING LIVE RSI AND 24H VOLUME DATA
async def list_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not await is_authorised(update, context) or update.effective_chat is None or user_msg is None:
        return

    messages_to_delete = [user_msg.message_id]

    try:
        client = context.application.bot_data["db_client"]
        result = await client.execute(
            "SELECT id, symbol, direction, target_price, entry_price, created_by, source_link, photo_file_id "
            "FROM alerts WHERE chat_id = ? ORDER BY id",
            (update.effective_chat.id,),
        )
        alerts = result.rows
        if not alerts:
            bot_msg = await user_msg.reply_text("🔔 No active alerts.")
            messages_to_delete.append(bot_msg.message_id)
            return

        session = context.application.bot_data["http_session"]
        symbols = {alert[1] for alert in alerts}
        
        # Parallel gathering of high-frequency price, rsi and volume indicators
        market_responses = await asyncio.gather(*(get_market_data(session, s) for s in symbols), return_exceptions=True)
        
        market_data = {}
        for s, res in zip(symbols, market_responses):
            if isinstance(res, Exception) or res is None:
                market_data[s] = {"price": None, "volume": "N/A", "rsi": "N/A"}
            else:
                market_data[s] = res

        header_msg = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="📊 <b>Active Alerts &amp; Market Prices</b>",
            parse_mode=ParseMode.HTML
        )
        messages_to_delete.append(header_msg.message_id)

        for number, alert in enumerate(alerts, start=1):
            try:
                alert_id, symbol, direction, target, entry, creator, source_link, photo_file_id = alert
                metrics = market_data.get(symbol, {"price": None, "volume": "N/A", "rsi": "N/A"})
                current = metrics["price"]
                
                dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                
                if current is None:
                    current_text = "unavailable"
                    progress_line = ""
                else:
                    current_text = f"{current:g} USDT"
                    
                    denom = (target - entry) if direction == "LONG" else (entry - target)
                    if denom == 0:
                        progress = 100
                    else:
                        if direction == "LONG":
                            if current <= entry:
                                progress = 0
                            elif current >= target:
                                progress = 100
                            else:
                                progress = int(((current - entry) / denom) * 100)
                        else:  # SHORT
                            if current >= entry:
                                progress = 0
                            elif current <= target:
                                progress = 100
                            else:
                                progress = int(((entry - current) / denom) * 100)

                    progress_clamped = min(100, max(0, progress))
                    filled_blocks = int(progress_clamped / 10)
                    empty_blocks = 10 - filled_blocks
                    bar = "█" * filled_blocks + "░" * empty_blocks
                    progress_line = f"\n📈 To Target: [<code>{bar}</code>] <code>{progress_clamped}%</code>"

                alert_text = (
                    f"{number}. <b>#{symbol}</b> | BY {creator}\n"
                    f"{dir_emoji}\n"
                    f"🎯 Target: <code>{target:g}</code> USDT\n"
                    f"⚡ Current: <code>{current_text}</code>\n"
                    f"📊 Vol (24h): <code>{metrics['volume']}</code> | 🕒 RSI (1h): <code>{metrics['rsi']}</code>"
                    f"{progress_line}"
                )
                
                keyboard = []
                if photo_file_id:
                    keyboard.append([InlineKeyboardButton(f"🔗 🖼️ View Chart #{symbol}", callback_data=f"show_img:{alert_id}")])

                reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
                
                alert_msg = await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=alert_text,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
                messages_to_delete.append(alert_msg.message_id)
            except Exception as e:
                LOGGER.error(f"Error compiling list item row {number} (#{symbol}): {e}")
                continue

    except Exception as general_error:
        LOGGER.error(f"General processing error inside list_alerts pipeline: {general_error}")
    finally:
        track_background_cleanup(context.bot, update.effective_chat.id, messages_to_delete, 30)


# CRASH-PROOF LIST_HISTORY FUNCTION WITH AUTO-DELETE AND CHART BUTTONS
async def list_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not await is_authorised(update, context) or update.effective_chat is None or user_msg is None:
        return

    messages_to_delete = [user_msg.message_id]

    try:
        client = context.application.bot_data["db_client"]
        result = await client.execute(
            "SELECT id, symbol, direction, target_price, triggered_price, created_by, triggered_at, photo_file_id "
            "FROM alert_history WHERE chat_id = ? ORDER BY triggered_at DESC LIMIT 30",
            (update.effective_chat.id,),
        )
        history = result.rows
        if not history:
            bot_msg = await user_msg.reply_text("📜 No alerts have been triggered yet.")
            messages_to_delete.append(bot_msg.message_id)
            return

        header_msg = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="📜 <b>History of Triggered Targets (Last 30)</b>",
            parse_mode=ParseMode.HTML
        )
        messages_to_delete.append(header_msg.message_id)

        for number, row in enumerate(history, start=1):
            try:
                hist_id, symbol, direction, target, triggered, creator, timestamp, photo_file_id = row
                dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                
                history_text = (
                    f"{number}. <b>#{symbol}</b> {dir_emoji}\n"
                    f"🎯 Target: <code>{target:g}</code> | Hit at: <code>{triggered:g}</code>\n"
                    f"👤 By: {creator} | 🕒 {timestamp}"
                )
                
                keyboard = []
                if photo_file_id:
                    keyboard.append([InlineKeyboardButton(f"🔗 🖼️ View Chart #{symbol}", callback_data=f"show_hist_img:{hist_id}")])
                
                reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
                
                msg = await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=history_text,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
                messages_to_delete.append(msg.message_id)
            except Exception as e:
                LOGGER.error(f"Error compiling history item row {number}: {e}")
                continue

    except Exception as general_error:
        LOGGER.error(f"General processing error inside list_history pipeline: {general_error}")
    finally:
        track_background_cleanup(context.bot, update.effective_chat.id, messages_to_delete, 30)


async def show_trade_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None:
        return
    await query.answer()
    
    client = context.application.bot_data["db_client"]
    
    if query.data.startswith("show_img:"):
        alert_id = int(query.data.split(":", maxsplit=1)[1])
        result = await client.execute(
            "SELECT symbol, direction, target_price, photo_file_id FROM alerts WHERE id = ? AND chat_id = ?",
            (alert_id, query.message.chat_id),
        )
    elif query.data.startswith("show_hist_img:"):
        hist_id = int(query.data.split(":", maxsplit=1)[1])
        result = await client.execute(
            "SELECT symbol, direction, target_price, photo_file_id FROM alert_history WHERE id = ? AND chat_id = ?",
            (hist_id, query.message.chat_id),
        )
    else:
        return

    alert = result.rows[0] if result.rows else None
    if alert is None or not alert[3]:
        await query.message.reply_text("The original chart image is no longer available.")
        return

    symbol, direction, target, photo_file_id = alert[0], alert[1], alert[2], alert[3]
    dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    await query.message.reply_photo(
        photo=photo_file_id,
        caption=f"#{symbol} {dir_emoji} | Target: {target:g} USDT",
    )


async def delete_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message
    if not await is_authorised(update, context) or update.effective_chat is None or user_msg is None:
        return

    if len(context.args) != 1 or not context.args[0].isdigit():
        bot_msg = await user_msg.reply_text("Usage: <code>/delete 12</code>", parse_mode=ParseMode.HTML)
        track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)
        return

    client = context.application.bot_data["db_client"]
    result = await client.execute(
        "DELETE FROM alerts WHERE id = ? AND chat_id = ?", (int(context.args[0]), update.effective_chat.id)
    )
    
    bot_msg = await user_msg.reply_text(
        "🗑️ Alert deleted successfully." if result.rows_affected > 0 else "No active alert found with that ID."
    )
    track_background_cleanup(context.bot, update.effective_chat.id, [user_msg.message_id, bot_msg.message_id], 30)


# 5. BACKGROUND ENGINE: TICKER WATCHER
async def check_alerts(application: Application) -> None:
    session: aiohttp.ClientSession = application.bot_data["http_session"]
    client = application.bot_data["db_client"]
    while True:
        try:
            result = await client.execute(
                "SELECT id, chat_id, symbol, direction, target_price, entry_price, created_by, photo_file_id, near_1_sent, near_05_sent, source_link FROM alerts ORDER BY id"
            )
            alerts = result.rows

            prices: dict[str, float | None] = {}
            for row in alerts:
                symbol = row[2]
                if symbol not in prices:
                    prices[symbol] = await get_price(session, symbol)

            for row in alerts:
                alert_id, chat_id, symbol, direction, target, entry_price, created_by, photo_file_id, near_1_sent, near_05_sent, source_link = (
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]
                )
                price = prices[symbol]
                
                if price is None:
                    continue

                reached = (
                    (direction == "LONG" and price >= target) or 
                    (direction == "SHORT" and price <= target)
                )
                
                if reached:
                    dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                    
                    # Live contextual market scan right as the trigger fires
                    m_data = await get_market_data(session, symbol)
                    
                    success_text = (
                        f"🎯 <b>Target Reached!</b>\n"
                        f"#{symbol} {dir_emoji}\n"
                        f"Target: <code>{target:g}</code> | Price Hit: <code>{price:g}</code>\n"
                        f"📊 Vol (24h): <code>{m_data['volume']}</code> | 🕒 RSI (1h): <code>{m_data['rsi']}</code>"
                    )
                    try:
                        if photo_file_id:
                            await application.bot.send_photo(
                                chat_id=chat_id, photo=photo_file_id, caption=success_text, parse_mode=ParseMode.HTML
                            )
                        else:
                            await application.bot.send_message(
                                chat_id=chat_id, text=success_text, parse_mode=ParseMode.HTML
                            )
                        
                        await client.execute(
                            "INSERT INTO alert_history (chat_id, symbol, direction, target_price, entry_price, triggered_price, created_by, photo_file_id, source_link) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (chat_id, symbol, direction, target, entry_price, price, created_by, photo_file_id, source_link)
                        )
                        
                        await client.execute(
                            "DELETE FROM alert_history WHERE chat_id = ? AND id NOT IN ("
                            "SELECT id FROM alert_history WHERE chat_id = ? ORDER BY triggered_at DESC, id DESC LIMIT 30"
                            ")",
                            (chat_id, chat_id)
                        )
                        
                        await client.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
                    except Exception:
                        LOGGER.exception("Failed to dispatch alert notification to chat %s for ID %s", chat_id, alert_id)
                    continue

                is_near_1 = False
                is_near_05 = False

                if direction == "LONG":
                    if (target * 0.995) <= price < target:
                        is_near_05 = True
                    elif (target * 0.99) <= price < target:
                        is_near_1 = True
                else:  # SHORT
                    if (target * 1.005) >= price > target:
                        is_near_05 = True
                    elif (target * 1.01) >= price > target:
                        is_near_1 = True

                if is_near_05 and not near_05_sent:
                    dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                    warning_text = (
                        f"⚠️ <b>Extremely Close to Target! (0.5% Remaining)</b>\n"
                        f"#{symbol} {dir_emoji}\n"
                        f"🎯 Target: <code>{target:g}</code> | Current Price: <code>{price:g}</code> (Less than 0.5% away!)"
                    )
                    try:
                        await application.bot.send_message(
                            chat_id=chat_id, text=warning_text, parse_mode=ParseMode.HTML
                        )
                        await client.execute("UPDATE alerts SET near_1_sent = 1, near_05_sent = 1 WHERE id = ?", (alert_id,))
                    except Exception:
                        LOGGER.exception("Failed to dispatch 0.5%% proximity engine alert for ID %s", alert_id)

                elif is_near_1 and not near_1_sent:
                    dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                    warning_text = (
                        f"⚠️ <b>Closing In On Target! (1% Remaining)</b>\n"
                        f"#{symbol} {dir_emoji}\n"
                        f"🎯 Target: <code>{target:g}</code> | Current Price: <code>{price:g}</code> (Less than 1% away!)"
                    )
                    try:
                        await application.bot.send_message(
                            chat_id=chat_id, text=warning_text, parse_mode=ParseMode.HTML
                        )
                        await client.execute("UPDATE alerts SET near_1_sent = 1 WHERE id = ?", (alert_id,))
                    except Exception:
                        LOGGER.exception("Failed to dispatch 1%% proximity engine alert for ID %s", alert_id)

        except Exception as e:
            LOGGER.error("Error inside target loop execution: %s", e)
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


# 6. MINI WEB SERVER
async def handle_ping(request):
    return web.Response(text="Bot ecosystem is running live!")


# 7. TELEGRAM LIFECYCLE HANDLERS
async def post_init(application: Application) -> None:
    db_url = os.getenv("TURSO_DATABASE_URL")
    db_token = os.getenv("TURSO_AUTH_TOKEN")
    if not db_url or not db_token:
        raise RuntimeError("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN variables must be configured properly.")
    
    if db_url.startswith("libsql://"):
        db_url = db_url.replace("libsql://", "https://")

    db_client = libsql_client.create_client(url=db_url, auth_token=db_token)
    application.bot_data["db_client"] = db_client
    
    await initialise_database(db_client)
    application.bot_data["http_session"] = aiohttp.ClientSession()
    application.bot_data["alert_task"] = asyncio.create_task(check_alerts(application))
    
    app = web.Application()
    app.router.add_get("/", handle_ping)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    application.bot_data["web_runner"] = runner
    LOGGER.info("Internal status engine bound on port %s", port)


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


# 8. BOT RUNTIME MANAGER
def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        LOGGER.error("Missing TELEGRAM_BOT_TOKEN inside configuration profile (.env)")
        return

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("addid", add_user_id))
    application.add_handler(CommandHandler("deleteid", delete_user_id))
    application.add_handler(CommandHandler("list", list_user_ids))
    application.add_handler(CommandHandler("delete", delete_alert))
    application.add_handler(CommandHandler("alerts", list_alerts))
    application.add_handler(CommandHandler("history", list_history))  
    application.add_handler(CommandHandler("alarm", add_alert))
    
    application.add_handler(CallbackQueryHandler(show_trade_image, pattern="^(show_img:|show_hist_img:)"))
    application.add_handler(MessageHandler(filters.PHOTO, add_alert))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, add_alert))

    LOGGER.info("Starting Polling loop context...")
    application.run_polling()


if __name__ == "__main__":
    main()
