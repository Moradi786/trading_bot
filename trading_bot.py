import asyncio
import logging
import time
import math
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import aiohttp

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
LOGGER = logging.getLogger("TradingBot")

# ==========================================
# 1. DATA STRUCTURES
# ==========================================

@dataclass
class AdvancedOBMetrics:
    symbol: str
    cvd: float
    bid_slippage_1k: float
    bid_slippage_10k: float
    ask_slippage_1k: float
    ask_slippage_10k: float
    bid_walls: List[Tuple[float, float]]
    ask_walls: List[Tuple[float, float]]
    iceberg_bids: int
    iceberg_asks: int
    stop_hunt_risk: float
    liquidity_imbalance: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Signal:
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    ob_metrics: Optional[AdvancedOBMetrics] = None


# ==========================================
# 2. ADVANCED ORDER BOOK ANALYZER
# ==========================================

class AdvancedOrderBookAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def analyze(self, bids: List[List[Any]], asks: List[List[Any]], current_price: float, atr: float = 0.0) -> AdvancedOBMetrics:
        # Convert bids/asks to float tuples
        clean_bids = [(float(b[0]), float(b[1])) for b in bids if float(b[1]) > 0]
        clean_asks = [(float(a[0]), float(a[1])) for a in asks if float(a[1]) > 0]

        cvd = self._calculate_cvd(clean_bids, clean_asks)
        bid_slip_1k = self._estimate_slippage(clean_bids, amount_usd=1000)
        bid_slip_10k = self._estimate_slippage(clean_bids, amount_usd=10000)
        ask_slip_1k = self._estimate_slippage(clean_asks, amount_usd=1000)
        ask_slip_10k = self._estimate_slippage(clean_asks, amount_usd=10000)
        
        bid_walls = self._detect_walls(clean_bids)
        ask_walls = self._detect_walls(clean_asks)
        
        iceberg_bids = self._detect_icebergs(clean_bids)
        iceberg_asks = self._detect_icebergs(clean_asks)
        
        stop_hunt_risk = self._detect_stop_hunt(clean_bids, clean_asks, current_price, atr)
        imbalance = self._calculate_imbalance(clean_bids, clean_asks)

        return AdvancedOBMetrics(
            symbol=self.symbol,
            cvd=cvd,
            bid_slippage_1k=bid_slip_1k,
            bid_slippage_10k=bid_slip_10k,
            ask_slippage_1k=ask_slip_1k,
            ask_slippage_10k=ask_slip_10k,
            bid_walls=bid_walls,
            ask_walls=ask_walls,
            iceberg_bids=iceberg_bids,
            iceberg_asks=iceberg_asks,
            stop_hunt_risk=stop_hunt_risk,
            liquidity_imbalance=imbalance
        )

    def _calculate_cvd(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
        bid_vol = sum(p * v for p, v in bids[:20])
        ask_vol = sum(p * v for p, v in asks[:20])
        return bid_vol - ask_vol

    def _estimate_slippage(self, order_side: List[Tuple[float, float]], amount_usd: float) -> float:
        if not order_side:
            return 999.0  # High penalty if empty
        
        initial_price = order_side[0][0]
        remaining_usd = amount_usd
        total_qty = 0.0

        for price, qty in order_side:
            level_usd = price * qty
            if remaining_usd <= level_usd:
                total_qty += remaining_usd / price
                remaining_usd = 0.0
                break
            else:
                total_qty += qty
                remaining_usd -= level_usd

        if remaining_usd > 0 or total_qty == 0:
            return 5.0  # Max slippage cap

        avg_price = amount_usd / total_qty
        slippage_pct = abs(avg_price - initial_price) / initial_price * 100.0
        return slippage_pct

    def _detect_walls(self, order_side: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not order_side:
            return []
        
        slice_depth = order_side[:20]
        slice_len = len(slice_depth)
        if slice_len == 0:
            return []

        avg_volume = sum(v for _, v in slice_depth) / slice_len
        walls = []
        for price, qty in slice_depth:
            if qty >= avg_volume * 3.5:  # Wall threshold
                walls.append((price, qty))
        return walls

    def _detect_icebergs(self, order_side: List[Tuple[float, float]]) -> int:
        if not order_side:
            return 0
        slice_depth = order_side[:15]
        slice_len = len(slice_depth)
        if slice_len == 0:
            return 0
        avg_qty = sum(v for _, v in slice_depth) / slice_len
        
        # Icebergs show dense clustering around tight ranges with equal sizes
        iceberg_count = sum(1 for _, qty in slice_depth if 1.8 * avg_qty <= qty <= 3.0 * avg_qty)
        return iceberg_count

    def _detect_stop_hunt(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], current_price: float, atr: float) -> float:
        if atr <= 0 or not bids or not asks:
            return 0.0

        near_bid_vol = sum(p * v for p, v in bids if abs(current_price - p) <= 1.2 * atr)
        near_ask_vol = sum(p * v for p, v in asks if abs(current_price - p) <= 1.2 * atr)

        total_near_vol = near_bid_vol + near_ask_vol
        if total_near_vol == 0:
            return 0.0

        risk_score = min(1.0, total_near_vol / (current_price * 100.0))
        return risk_score

    def _calculate_imbalance(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
        bid_vol = sum(p * v for p, v in bids[:30])
        ask_vol = sum(p * v for p, v in asks[:30])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


# OB Analyzer Factory
_OB_ANALYZERS: Dict[str, AdvancedOrderBookAnalyzer] = {}

def get_ob_analyzer(symbol: str) -> AdvancedOrderBookAnalyzer:
    if symbol not in _OB_ANALYZERS:
        _OB_ANALYZERS[symbol] = AdvancedOrderBookAnalyzer(symbol)
    return _OB_ANALYZERS[symbol]

# ==========================================
# 3. MULTI-EXCHANGE FAILOVER FETCHERS
# ==========================================

async def fetch_advanced_order_book(session: aiohttp.ClientSession, symbol: str, depth_limit: int = 100, atr: float = 0.0) -> Optional[AdvancedOBMetrics]:
    """Fetch order book from Binance, Bybit, or OKX with automatic failover."""
    
    clean_symbol = symbol.replace("/", "").replace("-", "").upper()
    
    exchanges = [
        {
            "name": "Binance",
            "url": f"https://fapi.binance.com/fapi/v1/depth?symbol={clean_symbol}&limit={depth_limit}",
            "parser": lambda d: (d.get("bids", []), d.get("asks", []))
        },
        {
            "name": "Bybit",
            "url": f"https://api.bybit.com/v5/market/orderbook?category=linear&symbol={clean_symbol}&limit=50",
            "parser": lambda d: (d.get("result", {}).get("b", []), d.get("result", {}).get("a", []))
        },
        {
            "name": "OKX",
            "url": f"https://www.okx.com/api/v5/market/books?instId={clean_symbol[:3]}-USDT-SWAP&sz=100",
            "parser": lambda d: (d.get("data", [{}])[0].get("bids", []), d.get("data", [{}])[0].get("asks", []))
        }
    ]

    for ex in exchanges:
        try:
            async with session.get(ex["url"], timeout=aiohttp.ClientTimeout(total=4)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bids, asks = ex["parser"](data)
                    
                    if bids and asks:
                        top_bid = float(bids[0][0])
                        top_ask = float(asks[0][0])
                        current_price = (top_bid + top_ask) / 2.0
                        
                        analyzer = get_ob_analyzer(symbol)
                        return analyzer.analyze(bids, asks, current_price, atr=atr)
        except Exception as e:
            LOGGER.debug(f"OrderBook fetch failover [{ex['name']}] for {symbol}: {e}")
            continue

    LOGGER.warning(f"Could not fetch Order Book for {symbol} from any exchange.")
    return None


async def fetch_klines_with_failover(session: aiohttp.ClientSession, symbol: str, interval: str = "15m", limit: int = 100) -> List[Dict[str, float]]:
    """Fetch OHLCV klines with multi-exchange failover."""
    clean_symbol = symbol.replace("/", "").replace("-", "").upper()

    exchanges = [
        {
            "name": "Binance",
            "url": f"https://fapi.binance.com/fapi/v1/klines?symbol={clean_symbol}&interval={interval}&limit={limit}",
            "parser": lambda data: [
                {
                    "time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                } for k in data
            ]
        },
        {
            "name": "Bybit",
            "url": f"https://api.bybit.com/v5/market/kline?category=linear&symbol={clean_symbol}&interval=15&limit={limit}",
            "parser": lambda data: [
                {
                    "time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                } for k in reversed(data.get("result", {}).get("list", []))
            ]
        }
    ]

    for ex in exchanges:
        try:
            async with session.get(ex["url"], timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    klines = ex["parser"](data)
                    if klines:
                        return klines
        except Exception as e:
            LOGGER.debug(f"Kline fetch failover [{ex['name']}] for {symbol}: {e}")
            continue

    return []

# ==========================================
# 4. TECHNICAL ANALYSIS & FEATURE BUILDING
# ==========================================

def calculate_ema(prices: List[float], period: int) -> float:
    if not prices:
        return 0.0
    k = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = (p * k) + (ema * (1 - k))
    return ema

def calculate_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
            
    avg_gain = gains / period
    avg_loss = losses / period
    
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            avg_gain = (avg_gain * (period - 1) + diff) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) - diff) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_atr(klines: List[Dict[str, float]], period: int = 14) -> float:
    if len(klines) < period + 1:
        return 0.0
    
    tr_list = []
    for i in range(1, len(klines)):
        high = klines[i]["high"]
        low = klines[i]["low"]
        prev_close = klines[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
        
    if not tr_list:
        return 0.0
    return sum(tr_list[-period:]) / min(len(tr_list), period)


def build_feature_dict(klines: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not klines or len(klines) < 30:
        return None

    closes = [k["close"] for k in klines]
    current_close = closes[-1]
    
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    rsi14 = calculate_rsi(closes, 14)
    atr14 = calculate_atr(klines, 14)

    return {
        "close": current_close,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi14,
        "atr": atr14
    }

# ==========================================
# 5. SIGNAL GENERATION & ORDER BOOK FILTER
# ==========================================

def generate_technical_signal(symbol: str, features: Dict[str, float]) -> Optional[Signal]:
    close = features["close"]
    ema20 = features["ema20"]
    ema50 = features["ema50"]
    rsi = features["rsi"]
    atr = features["atr"]

    # Trend & Momentum Strategy
    if ema20 > ema50 and 52 < rsi < 70:
        stop_loss = close - (1.5 * atr)
        take_profit = close + (3.0 * atr)
        return Signal(
            symbol=symbol,
            direction="BUY",
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="Bullish EMA Crossover with healthy RSI momentum"
        )
    elif ema20 < ema50 and 30 < rsi < 48:
        stop_loss = close + (1.5 * atr)
        take_profit = close - (3.0 * atr)
        return Signal(
            symbol=symbol,
            direction="SELL",
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="Bearish EMA Crossover with momentum decline"
        )

    return None


def advanced_ob_filter(signal: Signal, ob: Optional[AdvancedOBMetrics]) -> Tuple[bool, str]:
    """Filters signals based on microstructural Order Book data."""
    if not ob:
        return True, "Passed (No OB data available)"

    if signal.direction == "BUY":
        if ob.liquidity_imbalance < -0.35:
            return False, f"Rejected BUY: High Sell Liquidity Imbalance ({ob.liquidity_imbalance:.2f})"
        
        if ob.ask_slippage_10k > 0.8:
            return False, f"Rejected BUY: Excessive Ask Slippage ({ob.ask_slippage_10k:.2f}%)"
        
        if ob.stop_hunt_risk > 0.85:
            return False, f"Rejected BUY: Critical Stop Hunt Risk ({ob.stop_hunt_risk:.2f})"
        
        if len(ob.ask_walls) >= 2:
            return False, "Rejected BUY: Multiple Ask Walls detected ahead"

    elif signal.direction == "SELL":
        if ob.liquidity_imbalance > 0.35:
            return False, f"Rejected SELL: High Buy Liquidity Imbalance ({ob.liquidity_imbalance:.2f})"
        
        if ob.bid_slippage_10k > 0.8:
            return False, f"Rejected SELL: Excessive Bid Slippage ({ob.bid_slippage_10k:.2f}%)"
        
        if ob.stop_hunt_risk > 0.85:
            return False, f"Rejected SELL: Critical Stop Hunt Risk ({ob.stop_hunt_risk:.2f})"
        
        if len(ob.bid_walls) >= 2:
            return False, "Rejected SELL: Multiple Bid Walls detected below"

    return True, "Passed Order Book Filters"

# ==========================================
# 6. CACHE & RATE LIMITER
# ==========================================

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()

    async def consume(self, amount: float = 1.0):
        while True:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            
            if self.tokens >= amount:
                self.tokens -= amount
                return
            
            await asyncio.sleep(0.1)

class SignalDeduplicator:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self.history: Dict[str, float] = {}

    def is_duplicate(self, symbol: str, direction: str) -> bool:
        key = f"{symbol}_{direction}"
        now = time.time()
        if key in self.history:
            if now - self.history[key] < self.ttl:
                return True
        self.history[key] = now
        return False

# ==========================================
# 7. ASYNC DATABASE OPERATIONS
# ==========================================

def sync_execute_db_query(query: str, params: tuple = ()):
    """Synchronous DB function wrapper (e.g., LibSQL/Turso/SQLite)."""
    LOGGER.debug(f"[DB] Executed: {query} with {params}")
    return True

async def execute_db_query_async(query: str, params: tuple = ()):
    """Non-blocking async DB wrapper using asyncio.to_thread."""
    try:
        return await asyncio.to_thread(sync_execute_db_query, query, params)
    except Exception as e:
        LOGGER.error(f"Async DB Error: {e}")
        return None

# ==========================================
# 8. TELEGRAM BOT NOTIFIER & LISTENER
# ==========================================

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    async def send_message(self, session: aiohttp.ClientSession, text: str):
        if not self.bot_token or not self.chat_id:
            LOGGER.info(f"[Telegram Alert Simulation]:
{text}")
            return
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    LOGGER.error(f"Failed to send Telegram message: {resp.status}")
        except Exception as e:
            LOGGER.error(f"Telegram Send Error: {e}")

async def telegram_command_listener(bot_token: str):
    """Listens for commands from Telegram without freezing CPU on errors."""
    if not bot_token:
        LOGGER.info("Telegram Bot Token not configured. Command listener disabled.")
        return

    offset = 0
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                params = {"offset": offset, "timeout": 10}
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for update in data.get("result", []):
                            offset = update["update_id"] + 1
                            message = update.get("message", {})
                            text = message.get("text", "")
                            
                            if text == "/status":
                                LOGGER.info("Received Telegram Command: /status")
                            elif text == "/scan":
                                LOGGER.info("Received Telegram Command: /scan")
                                
            except Exception as e:
                LOGGER.error(f"Telegram Command Listener Error: {e}")
                await asyncio.sleep(2)  # Prevents CPU 100% spin loop on internet drops
                
            await asyncio.sleep(1)

# ==========================================
# 9. MAIN SCANNING ENGINE
# ==========================================

async def process_single_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    rate_limiter: TokenBucket,
    deduplicator: SignalDeduplicator,
    telegram: TelegramNotifier
):
    await rate_limiter.consume(1.0)
    
    # 1. Fetch Klines
    klines = await fetch_klines_with_failover(session, symbol, interval="15m", limit=100)
    if not klines:
        return

    # 2. Build Technical Features & extract ATR
    features = build_feature_dict(klines)
    if not features:
        return

    current_atr = features.get("atr", 0.0)

    # 3. Generate Technical Signal
    raw_signal = generate_technical_signal(symbol, features)
    if not raw_signal:
        return

    # Deduplication check
    if deduplicator.is_duplicate(symbol, raw_signal.direction):
        LOGGER.debug(f"Duplicate signal suppressed for {symbol}")
        return

    # 4. Fetch Order Book metrics using real ATR with multi-exchange failover
    ob_metrics = await fetch_advanced_order_book(session, symbol, depth_limit=100, atr=current_atr)

    # 5. Filter Signal using Order Book Metrics
    is_valid, filter_reason = advanced_ob_filter(raw_signal, ob_metrics)

    if is_valid:
        raw_signal.ob_metrics = ob_metrics
        msg = (
            f"🚨 <b>QUANT TRADING SIGNAL: {symbol}</b> 🚨\n\n"
            f"<b>Direction:</b> {raw_signal.direction}\n"
            f"<b>Entry:</b> ${raw_signal.entry_price:.4f}\n"
            f"<b>Stop Loss:</b> ${raw_signal.stop_loss:.4f}\n"
            f"<b>Take Profit:</b> ${raw_signal.take_profit:.4f}\n"
            f"<b>Reason:</b> {raw_signal.reason}\n\n"
            f"📊 <b>Order Book Microstructure:</b>\n"
            f"• Imbalance Ratio: {ob_metrics.liquidity_imbalance if ob_metrics else 0.0:.2f}\n"
            f"• 10k Slippage: {(ob_metrics.ask_slippage_10k if raw_signal.direction == 'BUY' else ob_metrics.bid_slippage_10k) if ob_metrics else 0.0:.2f}%\n"
            f"• Stop Hunt Risk: {ob_metrics.stop_hunt_risk if ob_metrics else 0.0:.2f}\n"
            f"• Iceberg Bids/Asks: {ob_metrics.iceberg_bids if ob_metrics else 0} / {ob_metrics.iceberg_asks if ob_metrics else 0}"
        )
        
        LOGGER.info(f"ACCEPTED SIGNAL FOR {symbol}: {raw_signal.direction}")
        await telegram.send_message(session, msg)
        await execute_db_query_async("INSERT INTO signals (symbol, direction) VALUES (?, ?)", (symbol, raw_signal.direction))
    else:
        LOGGER.info(f"REJECTED SIGNAL FOR {symbol}: {filter_reason}")


async def main():
    LOGGER.info("Starting Quant Trading Bot Engine...")
    
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]
    
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    
    rate_limiter = TokenBucket(rate=10.0, capacity=20.0)
    deduplicator = SignalDeduplicator(ttl_seconds=1800)
    telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    if TELEGRAM_BOT_TOKEN:
        asyncio.create_task(telegram_command_listener(TELEGRAM_BOT_TOKEN))

    async with aiohttp.ClientSession() as session:
        # Run one quick test iteration
        LOGGER.info("--- Running Initial Market Scan Test ---")
        tasks = [
            process_single_symbol(session, symbol, rate_limiter, deduplicator, telegram)
            for symbol in SYMBOLS
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        LOGGER.info("--- Initial Test Completed Successfully ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Bot execution terminated gracefully by user.")
