"""
Grid Trading Bot - Main Entry Point
Phase 4: Dynamic Monitoring & Order Flipping with Atomic State Persistence
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from core.exchange import BinanceManager
from core.strategy import (
    GridStrategy, 
    save_state, 
    load_state, 
    create_initial_state
)
from core.notifier import TelegramNotifier
from core.database import DatabaseManager

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Lock file path
LOCK_FILE = 'data/bot.lock'

# PHASE 6.2: Monitoring Lock (prevents race conditions)
MONITORING_LOCK = asyncio.Lock()

# Emergency Stop Configuration
STOP_LOSS_BUFFER = 0.02  # 2% below lower bound

# PHASE 7.2: Trailing Stop Configuration
TRAILING_STOP_PCT = 0.03  # 3% trailing stop for open positions

# PHASE 7.2: Holdings Tracker (tracks open positions with peak prices)
# Structure: {level: {'amount': float, 'entry_price': float, 'peak_price': float, 'entry_time': str}}
HOLDINGS: Dict[int, Dict[str, Any]] = {}


def calculate_atr(ohlcv_data: list, period: int = 14) -> float:
    """Calculate Average True Range (ATR) from OHLCV data"""
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    atr = df['tr'].rolling(window=period).mean().iloc[-1]
    return atr


def calculate_ema(ohlcv_data: list, period: int = 50) -> float:
    """
    Calculate Exponential Moving Average (EMA) from OHLCV data
    
    Args:
        ohlcv_data: List of OHLCV candles
        period: EMA period (default: 50)
    
    Returns:
        float: Current EMA value
    """
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    ema = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
    return ema


def get_trend_filter(current_price: float, ema_value: float, buffer_pct: float = 0.02) -> Dict[str, bool]:
    """
    Determine which order sides are allowed based on trend
    
    Args:
        current_price: Current market price
        ema_value: EMA value for trend detection
        buffer_pct: Buffer percentage to avoid whipsaw (default: 2%)
    
    Returns:
        Dict with 'allow_buy' and 'allow_sell' booleans
    
    Logic:
        - Uptrend (price > EMA * 1.02): Allow BUY orders only
        - Downtrend (price < EMA * 0.98): Allow SELL orders only
        - Neutral zone: Allow both (ranging market)
    """
    upper_threshold = ema_value * (1 + buffer_pct)
    lower_threshold = ema_value * (1 - buffer_pct)
    
    if current_price > upper_threshold:
        # Strong uptrend - only BUY (we want to accumulate in uptrend)
        return {'allow_buy': True, 'allow_sell': True, 'trend': 'UPTREND'}
    elif current_price < lower_threshold:
        # Strong downtrend - only SELL (avoid catching falling knife)
        return {'allow_buy': False, 'allow_sell': True, 'trend': 'DOWNTREND'}
    else:
        # Neutral/Ranging - allow both (grid bot's sweet spot)
        return {'allow_buy': True, 'allow_sell': True, 'trend': 'NEUTRAL'}


def add_holding(level: int, amount: float, entry_price: float):
    """
    Add a new holding position to track for trailing stop
    
    Args:
        level: Grid level of the position
        amount: Amount of base currency held
        entry_price: Entry price of the position
    """
    global HOLDINGS
    HOLDINGS[level] = {
        'amount': amount,
        'entry_price': entry_price,
        'peak_price': entry_price,  # Initially, peak = entry
        'entry_time': datetime.now().isoformat()
    }
    logger.debug(f"[HOLDINGS] Added position at Level {level}: {amount:.6f} @ ${entry_price:,.2f}")


def remove_holding(level: int):
    """Remove a holding position after it's been closed"""
    global HOLDINGS
    if level in HOLDINGS:
        del HOLDINGS[level]
        logger.debug(f"[HOLDINGS] Removed position at Level {level}")


def update_peak_prices(current_price: float):
    """
    Update peak prices for all holdings (called on each monitoring cycle)
    
    Args:
        current_price: Current market price
    """
    global HOLDINGS
    for level, holding in HOLDINGS.items():
        if current_price > holding['peak_price']:
            old_peak = holding['peak_price']
            holding['peak_price'] = current_price
            logger.debug(f"[HOLDINGS] Level {level} peak updated: ${old_peak:,.2f} → ${current_price:,.2f}")


def check_trailing_stops(current_price: float) -> List[Dict[str, Any]]:
    """
    Check if any holdings should be closed due to trailing stop
    
    Args:
        current_price: Current market price
    
    Returns:
        List of holdings that triggered trailing stop
    """
    global HOLDINGS
    triggered = []
    
    for level, holding in list(HOLDINGS.items()):
        peak_price = holding['peak_price']
        trailing_stop_price = peak_price * (1 - TRAILING_STOP_PCT)
        
        if current_price < trailing_stop_price:
            # Trailing stop triggered!
            triggered.append({
                'level': level,
                'amount': holding['amount'],
                'entry_price': holding['entry_price'],
                'peak_price': peak_price,
                'stop_price': trailing_stop_price,
                'current_price': current_price,
                'loss_from_peak_pct': ((peak_price - current_price) / peak_price) * 100
            })
            logger.warning(f"[TRAILING STOP] Level {level} triggered!")
            logger.warning(f"  Peak: ${peak_price:,.2f} → Current: ${current_price:,.2f}")
            logger.warning(f"  Stop Price: ${trailing_stop_price:,.2f} | Loss from peak: {((peak_price - current_price) / peak_price) * 100:.2f}%")
    
    return triggered


async def execute_trailing_stop(binance: BinanceManager, symbol: str, holding_info: dict,
                                state: dict, db: Optional[DatabaseManager] = None,
                                notifier: Optional[TelegramNotifier] = None) -> bool:
    """
    Execute a market sell for a trailing stop exit
    
    Args:
        binance: BinanceManager instance
        symbol: Trading symbol
        holding_info: Dict with holding details from check_trailing_stops
        state: Current state dictionary
        db: Optional database manager
        notifier: Optional Telegram notifier
    
    Returns:
        bool: True if executed successfully
    """
    try:
        level = holding_info['level']
        amount = holding_info['amount']
        entry_price = holding_info['entry_price']
        current_price = holding_info['current_price']
        
        logger.info(f"[TRAILING STOP] Executing market sell for Level {level}...")
        
        # Place market sell order
        order = await binance.exchange.create_market_sell_order(symbol, amount)
        
        if order:
            # Calculate P&L
            pnl = (current_price - entry_price) * amount
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            logger.info(f"[TRAILING STOP] Executed @ ~${current_price:,.2f}")
            logger.info(f"[TRAILING STOP] P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            
            # Update stats
            stats = state.get('stats', {})
            stats['trailing_stops_count'] = stats.get('trailing_stops_count', 0) + 1
            stats['total_profit_usd'] = stats.get('total_profit_usd', 0) + pnl
            
            # Add to trade history
            trade_history = state.get('trade_history', [])
            trade_history.append({
                'id': len(trade_history) + 1,
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'side': 'SELL',
                'type': 'TRAILING_STOP',
                'entry_price': entry_price,
                'exit_price': current_price,
                'amount': amount,
                'profit_usd': pnl,
                'peak_price': holding_info['peak_price'],
                'status': 'filled'
            })
            
            # Insert into database
            if db:
                await db.insert_trade({
                    'symbol': symbol,
                    'side': 'SELL',
                    'price': current_price,
                    'amount': amount,
                    'profit': pnl,
                    'timestamp': datetime.now().isoformat(),
                    'order_id': str(order.get('id', '')),
                    'level': level,
                    'client_order_id': f'trailing_stop_{level}_{int(time.time())}',
                    'status': 'filled'
                })
            
            # Remove from holdings
            remove_holding(level)
            
            # Send notification
            if notifier:
                await notifier.send_message(
                    f"🛑 *Trailing Stop Executed*\n"
                    f"Level: {level}\n"
                    f"Exit Price: ${current_price:,.2f}\n"
                    f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)"
                )
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"[TRAILING STOP ERROR] Failed to execute: {str(e)}")
        return False


def generate_client_order_id(level: int, side: str, iteration: int = 0) -> str:
    """Generate a unique client order ID for tracking"""
    timestamp = int(time.time())
    if iteration > 0:
        return f"grid_{timestamp}_{level}_{side}_i{iteration}"
    return f"grid_{timestamp}_{level}_{side}"


# PHASE 7.5: ATR Change Threshold for Grid Recalculation
ATR_RECALC_THRESHOLD = 0.20  # 20% change triggers recalculation

def should_recalculate_grid(state: dict, current_atr: float) -> bool:
    """
    PHASE 7.5: Check if grid should be recalculated due to significant ATR change
    
    Args:
        state: Current state dictionary containing original grid params
        current_atr: Current ATR value
    
    Returns:
        bool: True if grid needs recalculation
    """
    grid_params = state.get('grid_params', {})
    original_atr = grid_params.get('atr_value', current_atr)
    
    if original_atr <= 0:
        return False
    
    atr_change_pct = abs(current_atr - original_atr) / original_atr
    
    if atr_change_pct > ATR_RECALC_THRESHOLD:
        logger.warning(f"[GRID RECALC] ATR changed {atr_change_pct*100:.1f}% (threshold: {ATR_RECALC_THRESHOLD*100:.0f}%)")
        logger.warning(f"[GRID RECALC] Original ATR: ${original_atr:,.2f} → Current: ${current_atr:,.2f}")
        return True
    
    return False


def create_lock_file() -> bool:
    """Create lock file to prevent multiple instances"""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = f.read().strip()
            logger.error(f"[ERROR] Bot is already running (PID: {pid})")
            logger.error("If sure no instance is running, delete data/bot.lock")
            return False
        except:
            pass
    
    try:
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        logger.debug(f"[LOCK] Created lock file with PID: {os.getpid()}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to create lock file: {str(e)}")
        return False


def remove_lock_file():
    """Remove lock file on shutdown"""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            logger.debug("[LOCK] Removed lock file")
    except Exception as e:
        logger.error(f"[ERROR] Failed to remove lock file: {str(e)}")


def check_startup_safety(state: Optional[dict]) -> bool:
    """
    Check if bot is safe to start based on previous emergency stops
    
    Args:
        state: Current state dictionary
    
    Returns:
        bool: True if safe to start, False if emergency stop was triggered
    """
    if state is None:
        return True  # No previous state, safe to start
    
    is_emergency = state.get('is_emergency_stopped', False)
    
    if is_emergency:
        logger.critical("=" * 80)
        logger.critical("⚠️  EMERGENCY STOP ACTIVE - BOT CANNOT START")
        logger.critical("=" * 80)
        logger.critical("")
        logger.critical("The bot was previously stopped due to an emergency condition.")
        logger.critical("")
        
        # Display emergency details
        stop_params = state.get('stop_loss_params', {})
        logger.critical(f"Emergency Stop Price: ${stop_params.get('emergency_stop_price', 0):,.2f}")
        logger.critical(f"Lower Bound: ${stop_params.get('lower_bound', 0):,.2f}")
        logger.critical(f"Last Update: {state.get('last_update', 'Unknown')}")
        logger.critical("")
        logger.critical("MANUAL INTERVENTION REQUIRED:")
        logger.critical("1. Review current market conditions")
        logger.critical("2. Edit data/state.json and set 'is_emergency_stopped': false")
        logger.critical("3. Consider adjusting grid parameters if needed")
        logger.critical("4. Restart the bot")
        logger.critical("")
        logger.critical("=" * 80)
        
        return False
    
    return True


async def place_grid_order(binance: BinanceManager, symbol: str, level: int, side: str, 
                          price: float, amount: float, iteration: int = 0) -> Optional[Dict[str, Any]]:
    """Place a single grid order"""
    client_order_id = generate_client_order_id(level, side, iteration)
    
    order_result = await binance.place_limit_order(
        symbol=symbol,
        side=side,
        amount=amount,
        price=price,
        client_order_id=client_order_id
    )
    
    if order_result:
        return {
            'level': level,
            'order_id': str(order_result.get('id')),
            'client_order_id': client_order_id,
            'side': side,
            'price': price,
            'amount': amount,
            'status': 'open',
            'placed_at': datetime.now().isoformat()
        }
    return None


async def sync_with_binance(binance: BinanceManager, symbol: str, state: dict,
                           db: Optional[DatabaseManager] = None,
                           notifier: Optional[TelegramNotifier] = None) -> tuple:
    """
    Sync state with Binance to find orders that filled while bot was offline
    
    Returns:
        tuple: (filled_orders, cancelled_orders)
    """
    logger.info("\n[SYNCING] Checking for changes while bot was offline...")
    
    active_orders = state.get('active_orders', {})
    filled_orders = []
    cancelled_orders = []
    
    # Fetch current open orders
    open_orders = await binance.get_open_orders(symbol)
    
    if open_orders is None:
        logger.error("[ERROR] Failed to fetch open orders during sync")
        return ([], [])
    
    # Create set of current open client IDs
    current_open_client_ids = {
        order.get('clientOrderId') for order in open_orders 
        if order.get('clientOrderId')
    }
    
    # Check each active order
    for client_order_id, order_info in list(active_orders.items()):
        if client_order_id not in current_open_client_ids:
            # Order is not in open orders, check its status
            logger.info(f"[SYNC] Checking order: {client_order_id}")
            
            order_details = await binance.fetch_order(order_info['order_id'], symbol)
            
            if order_details is None:
                logger.warning(f"[SYNC] Could not fetch order details for {client_order_id}")
                continue
            
            status = order_details.get('status', '').lower()
            
            if status in ['closed', 'filled']:
                filled_orders.append(order_info)
                logger.info(f"[SYNC] Order {client_order_id} was FILLED while offline")
            elif status in ['canceled', 'cancelled']:
                cancelled_orders.append(order_info)
                logger.info(f"[SYNC] Order {client_order_id} was CANCELLED while offline")
            else:
                logger.warning(f"[SYNC] Order {client_order_id} has unexpected status: {status}")
    
    logger.info(f"[SYNC] Found {len(filled_orders)} filled and {len(cancelled_orders)} cancelled orders")
    
    return (filled_orders, cancelled_orders)


async def process_filled_order(binance: BinanceManager, symbol: str, filled_order: dict,
                               grid_spacing: float, state: dict, 
                               db: Optional[DatabaseManager] = None,
                               notifier: Optional[TelegramNotifier] = None,
                               trend_filter: Optional[Dict] = None) -> Optional[Dict]:
    """
    Process a filled order and flip it
    
    Args:
        binance: BinanceManager instance
        symbol: Trading symbol
        filled_order: The filled order info dict
        grid_spacing: Grid spacing value
        state: Current state dictionary
        db: Optional database manager
        notifier: Optional Telegram notifier
        trend_filter: Optional trend filter dict with 'allow_buy', 'allow_sell', 'trend'
    
    Returns:
        dict: New order info if flipped successfully, None otherwise
    """
    level = filled_order['level']
    side = filled_order['side']
    price = filled_order['price']
    amount = filled_order['amount']
    
    # Calculate REAL profit (only for SELL orders)
    # BUY orders don't realize profit until they're sold
    real_profit = 0.0
    if side == 'SELL':
        # Profit = (Sell_Price - Buy_Price) * Amount
        # In grid bot: Sell_Price - Buy_Price = grid_spacing
        real_profit = grid_spacing * amount
        logger.info(f"[TRADE COMPLETED] Level {level} {side} @ ${price:,.2f} | Real Profit: +${real_profit:,.2f}")
    else:
        logger.info(f"[TRADE COMPLETED] Level {level} {side} @ ${price:,.2f} | Position opened")
    
    # Update stats
    stats = state.get('stats', {})
    stats['fills_count'] = stats.get('fills_count', 0) + 1
    stats['total_trades'] = stats.get('total_trades', 0) + 1
    
    # Only add to profit if this was a SELL (realized profit)
    if side == 'SELL':
        stats['total_profit_usd'] = stats.get('total_profit_usd', 0) + real_profit
    
    stats['last_trade_at'] = datetime.now().isoformat()
    
    # Add to trade history (state.json - for backward compatibility)
    trade_history = state.get('trade_history', [])
    trade_history.append({
        'id': len(trade_history) + 1,
        'timestamp': datetime.now().isoformat(),
        'client_order_id': filled_order['client_order_id'],
        'level': level,
        'side': side,
        'price': price,
        'amount': amount,
        'status': 'filled',
        'profit_usd': real_profit  # Only SELL orders have profit > 0
    })
    
    # PHASE 6: Insert into SQLite database (persistent cold storage)
    if db:
        await db.insert_trade({
            'symbol': state.get('symbol', symbol),
            'side': side,
            'price': price,
            'amount': amount,
            'profit': real_profit,
            'timestamp': datetime.now().isoformat(),
            'order_id': filled_order.get('order_id', ''),
            'level': level,
            'client_order_id': filled_order['client_order_id'],
            'status': 'filled'
        })
    
    # Determine flip action
    iteration_counter = state.get('iteration_counter', {})
    iteration_key = str(level)
    iteration_counter[iteration_key] = iteration_counter.get(iteration_key, 0) + 1
    iteration = iteration_counter[iteration_key]
    
    new_order = None
    
    if side == 'BUY':
        # PHASE 7.2: Track holding for trailing stop
        add_holding(level, amount, price)
        
        # BUY filled → Place SELL at next level
        next_level = level + 1
        if next_level <= 10:  # Assuming 10 levels
            flip_price = price + grid_spacing
            logger.info(f"  [FLIPPING] BUY @ Level {level} → SELL @ Level {next_level} (${flip_price:,.2f})")
            
            new_order = await place_grid_order(
                binance, symbol, next_level, 'SELL', flip_price, amount, iteration
            )
            
            if new_order:
                logger.info(f"  [SUCCESS] Flipped to SELL order @ Level {next_level}")
        else:
            logger.warning(f"  [SKIP] Cannot flip Level {level} BUY - already at top")
    
    elif side == 'SELL':
        # PHASE 7.2: Remove holding (position closed)
        remove_holding(level)
        
        # SELL filled → Place BUY at previous level
        prev_level = level - 1
        if prev_level >= 1:
            flip_price = price - grid_spacing
            
            # PHASE 7: Check trend filter before placing BUY
            if trend_filter and not trend_filter.get('allow_buy', True):
                logger.warning(f"  [TREND FILTER] Skipping BUY @ Level {prev_level} - DOWNTREND detected")
                logger.warning(f"  [TREND FILTER] Trend: {trend_filter.get('trend', 'N/A')} - Avoiding falling knife")
                # Don't place BUY order, but still consider the SELL processed
            else:
                logger.info(f"  [FLIPPING] SELL @ Level {level} → BUY @ Level {prev_level} (${flip_price:,.2f})")
                
                new_order = await place_grid_order(
                    binance, symbol, prev_level, 'BUY', flip_price, amount, iteration
                )
                
                if new_order:
                    logger.info(f"  [SUCCESS] Flipped to BUY order @ Level {prev_level}")
        else:
            logger.warning(f"  [SKIP] Cannot flip Level {level} SELL - already at bottom")
    
    # Send Telegram notification
    if notifier and new_order:
        await notifier.send_trade_notification(filled_order, real_profit)
    
    return new_order


async def check_emergency_stop(binance: BinanceManager, symbol: str, state: dict,
                               notifier: Optional[TelegramNotifier] = None) -> bool:
    """
    Check if emergency stop should be triggered
    
    Args:
        binance: Binance manager instance
        symbol: Trading symbol
        state: Current state dictionary
        notifier: Optional Telegram notifier
    
    Returns:
        bool: True if emergency triggered, False if safe
    """
    try:
        # Get current price
        current_price = await binance.get_current_price(symbol)
        if current_price is None:
            logger.warning("[EMERGENCY CHECK] Could not fetch price, skipping check")
            return False
        
        # Get emergency stop parameters
        stop_params = state.get('stop_loss_params', {})
        emergency_threshold = stop_params.get('emergency_stop_price', 0)
        lower_bound = stop_params.get('lower_bound', 0)
        
        # Log current status
        logger.debug(f"[SAFETY CHECK] Price: ${current_price:,.2f} | Threshold: ${emergency_threshold:,.2f}")
        
        # Check if price has fallen below emergency threshold
        if current_price < emergency_threshold:
            logger.critical("")
            logger.critical("=" * 80)
            logger.critical("🚨 EMERGENCY STOP TRIGGERED! 🚨")
            logger.critical("=" * 80)
            logger.critical("")
            logger.critical(f"Current Price: ${current_price:,.2f}")
            logger.critical(f"Emergency Threshold: ${emergency_threshold:,.2f}")
            logger.critical(f"Lower Bound: ${lower_bound:,.2f}")
            logger.critical("")
            logger.critical("Price has fallen below safety zone!")
            logger.critical(f"Drop: {((emergency_threshold - current_price) / emergency_threshold * 100):.2f}% below threshold")
            logger.critical("")
            
            # Cancel all open orders
            active_orders = state.get('active_orders', {})
            logger.critical(f"Cancelling {len(active_orders)} open orders...")
            
            cancelled_count = await binance.cancel_all_open_orders(symbol)
            logger.critical(f"Cancelled {cancelled_count} orders")
            
            # Update state
            state['is_emergency_stopped'] = True
            state['active_orders'] = {}  # Clear active orders
            
            # Add emergency event to trade history
            trade_history = state.get('trade_history', [])
            trade_history.append({
                'id': len(trade_history) + 1,
                'timestamp': datetime.now().isoformat(),
                'event': 'EMERGENCY_STOP',
                'price': current_price,
                'threshold': emergency_threshold,
                'reason': f'Price ${current_price:,.2f} below threshold ${emergency_threshold:,.2f}'
            })
            
            # Save state
            save_state(state)
            
            # Send Telegram alert
            if notifier:
                await notifier.send_emergency_alert(current_price, emergency_threshold, cancelled_count)
            
            logger.critical("")
            logger.critical("⚠️  BOT STOPPED - MANUAL INTERVENTION REQUIRED")
            logger.critical("Review market conditions before restarting.")
            logger.critical("=" * 80)
            logger.critical("")
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"[ERROR] Emergency check failed: {str(e)}")
        return False


async def monitor_and_flip(binance: BinanceManager, symbol: str, state: dict,
                           db: Optional[DatabaseManager] = None,
                           notifier: Optional[TelegramNotifier] = None):
    """
    Monitor open orders and flip filled orders
    
    PHASE 6.2: Uses asyncio.Lock to prevent race conditions.
    Only one monitoring cycle can execute the critical section at a time.
    """
    
    active_orders = state.get('active_orders', {})
    grid_spacing = state.get('grid_params', {}).get('grid_spacing', 0)
    
    logger.info("\n" + "=" * 80)
    logger.info("STARTING REAL-TIME MONITORING & ORDER FLIPPING")
    logger.info("=" * 80)
    logger.info("Press Ctrl+C to stop the bot")
    logger.info("=" * 80 + "\n")
    
    cycle = 0
    
    try:
        while True:
            cycle += 1
            logger.info(f"\n[MONITORING CYCLE #{cycle}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # PHASE 6.2: CRITICAL SECTION - Protected by Lock
            # Prevents race condition where two cycles process the same filled order
            async with MONITORING_LOCK:
                logger.debug("[LOCK] Acquired monitoring lock")
                
                # CRITICAL: Check emergency stop FIRST before any other operations
                emergency_triggered = await check_emergency_stop(binance, symbol, state, notifier)
                if emergency_triggered:
                    logger.critical("[EMERGENCY] Exiting monitoring loop due to emergency stop")
                    break
                
                # PHASE 7: Fetch fresh EMA data and calculate trend filter
                current_price = await binance.get_current_price(symbol)
                if current_price:
                    ohlcv_data = await binance.fetch_ohlcv(symbol, '1h', limit=60)  # 60 candles for EMA-50
                    if ohlcv_data:
                        ema_value = calculate_ema(ohlcv_data, period=50)
                        trend_filter = get_trend_filter(current_price, ema_value)
                        
                        # Log trend status periodically (every 10 cycles)
                        if cycle % 10 == 1:
                            logger.info(f"[TREND] Price: ${current_price:,.2f} | EMA-50: ${ema_value:,.2f} | Trend: {trend_filter['trend']}")
                    else:
                        trend_filter = {'allow_buy': True, 'allow_sell': True, 'trend': 'UNKNOWN'}
                    
                    # PHASE 7.2: Trailing Stop Management
                    # Update peak prices for all holdings
                    update_peak_prices(current_price)
                    
                    # Check for triggered trailing stops
                    triggered_stops = check_trailing_stops(current_price)
                    
                    if triggered_stops:
                        logger.warning(f"[TRAILING STOP] {len(triggered_stops)} position(s) triggered!")
                        for stop_info in triggered_stops:
                            success = await execute_trailing_stop(
                                binance, symbol, stop_info, state, db, notifier
                            )
                            if success:
                                # Remove from active orders if there was a pending SELL
                                level = stop_info['level']
                                for cid, order in list(active_orders.items()):
                                    if order.get('level') == level and order.get('side') == 'SELL':
                                        await binance.exchange.cancel_order(order['order_id'], symbol)
                                        del active_orders[cid]
                                        logger.info(f"[TRAILING STOP] Cancelled pending SELL @ Level {level}")
                                        break
                        save_state(state)
                    
                    # Log holdings count periodically
                    if HOLDINGS and cycle % 10 == 1:
                        logger.info(f"[HOLDINGS] Tracking {len(HOLDINGS)} position(s) for trailing stop")
                else:
                    trend_filter = {'allow_buy': True, 'allow_sell': True, 'trend': 'UNKNOWN'}
            
                # Fetch current open orders from Binance
                open_orders = await binance.get_open_orders(symbol)
            
                if open_orders is None:
                    logger.error("[ERROR] Failed to fetch open orders. Retrying in 30s...")
                    # Release lock before sleeping
                    logger.debug("[LOCK] Releasing lock due to error")
                    # Lock will auto-release at end of 'async with' block
                    await asyncio.sleep(30)
                    continue
            
                # Create set of current open order client IDs
                current_open_client_ids = {
                    order.get('clientOrderId') for order in open_orders 
                    if order.get('clientOrderId')
                }
                
                logger.info(f"Active orders in state: {len(active_orders)}")
                logger.info(f"Open orders on Binance: {len(open_orders)}")
                
                # Find orders that are no longer open
                missing_orders = []
                for client_order_id, order_info in list(active_orders.items()):
                    if client_order_id not in current_open_client_ids:
                        missing_orders.append((client_order_id, order_info))
            
                # Process missing orders
                if missing_orders:
                    logger.info(f"\n[DETECTED] {len(missing_orders)} order(s) no longer open, checking status...")
                
                    for client_order_id, order_info in missing_orders:
                        # Fetch order details to determine if filled or cancelled
                        order_details = await binance.fetch_order(order_info['order_id'], symbol)
                    
                        if order_details is None:
                            logger.error(f"[ERROR] Could not fetch order details for {client_order_id}")
                            continue
                    
                        status = order_details.get('status', '').lower()
                    
                        if status in ['closed', 'filled']:
                            # Process filled order with trend filter
                            new_order = await process_filled_order(
                                binance, symbol, order_info, grid_spacing, state, db, notifier,
                                trend_filter=trend_filter  # PHASE 7: Pass trend filter
                            )
                            
                            # Remove from active orders
                            del active_orders[client_order_id]
                            
                            # Add new flipped order to active orders
                            if new_order:
                                active_orders[new_order['client_order_id']] = new_order
                            
                            # Save state immediately (still within lock)
                            save_state(state)
                        
                        elif status in ['canceled', 'cancelled']:
                            # Order was manually cancelled
                            logger.warning(f"[WARNING] Level {order_info['level']} {order_info['side']} "
                                          f"@ ${order_info['price']:,.2f} was manually cancelled - NOT flipping!")
                            
                            # Update stats
                            stats = state.get('stats', {})
                            stats['cancels_count'] = stats.get('cancels_count', 0) + 1
                            
                            # Add to trade history
                            trade_history = state.get('trade_history', [])
                            trade_history.append({
                                'id': len(trade_history) + 1,
                                'timestamp': datetime.now().isoformat(),
                                'client_order_id': order_info['client_order_id'],
                                'level': order_info['level'],
                                'side': order_info['side'],
                                'price': order_info['price'],
                                'amount': order_info['amount'],
                                'status': 'cancelled'
                            })
                            
                            # Remove from active orders
                            del active_orders[client_order_id]
                            
                            # Save state
                            save_state(state)
                        else:
                            logger.warning(f"[WARNING] Order {client_order_id} has unexpected status: {status}")
                
                    # Log stats
                    stats = state.get('stats', {})
                    logger.info(f"\n[STATS] Total trades: {stats.get('total_trades', 0)}, "
                               f"Fills: {stats.get('fills_count', 0)}, "
                               f"Cancels: {stats.get('cancels_count', 0)}, "
                               f"Est. Profit: ${stats.get('total_profit_usd', 0):,.2f}")
                else:
                    logger.info("[NO CHANGES] All orders still open, waiting...")
                
                logger.debug("[LOCK] Releasing monitoring lock")
                # Lock automatically released at end of 'async with' block
            
            # Wait 30 seconds before next check
            logger.info(f"Next check in 30 seconds...")
            await asyncio.sleep(30)
            
    except asyncio.CancelledError:
        logger.info("\n[SHUTDOWN] Monitoring stopped")
        raise


async def main():
    """Main entry point for Grid Trading Bot - Phase 6: Database Integration"""
    
    # Check for lock file
    if not create_lock_file():
        return
    
    # Initialize database (will be closed in finally block)
    db = None
    
    try:
        logger.info("=" * 80)
        logger.info("Grid Trading Bot - Phase 6: SQLite Database Integration")
        logger.info("=" * 80)
        
        # Configuration
        SYMBOL = 'BTC/USDT'
        TIMEFRAME = '1h'
        ATR_PERIOD = 14
        EMA_PERIOD = 50  # PHASE 7: Trend filter
        N_LEVELS = 10
        ATR_MULTIPLIER = 2.0
        ALLOCATION_PERCENT = 2.0
        
        # Initialize Binance Manager
        binance = BinanceManager()
        
        # Initialize Telegram Notifier (optional)
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        notifier = None
        if telegram_token and telegram_chat_id and telegram_token != 'your_telegram_bot_token_here':
            try:
                notifier = TelegramNotifier(telegram_token, telegram_chat_id)
                logger.info("[TELEGRAM] Notifier initialized successfully")
            except Exception as e:
                logger.warning(f"[TELEGRAM] Failed to initialize notifier: {str(e)}")
                logger.warning("[TELEGRAM] Bot will continue without notifications")
        else:
            logger.info("[TELEGRAM] Notifier not configured (optional feature)")
        
        # PHASE 6: Initialize Database Manager
        logger.info("\n[DATABASE] Initializing SQLite database...")
        db = DatabaseManager()
        await db.init_db()
        logger.info("[DATABASE] Database ready for trade history storage")
        
        # Try to load previous state
        state = load_state()
        
        # CRITICAL: Check if emergency stop was triggered in previous session
        if not check_startup_safety(state):
            logger.error("[STARTUP] Bot cannot start due to emergency stop condition")
            return
        
        # PHASE 6: Migrate existing trades from state.json to database (one-time)
        if state and db:
            migrated = await db.migrate_from_state(state)
            if migrated > 0:
                logger.info(f"[MIGRATION] Successfully migrated {migrated} trades to database")
        
        if state:
            # Resume from previous state
            logger.info("\n[RESUMING] From previous state...")
            logger.info("=" * 80)
            
            # Sync with Binance
            filled_orders, cancelled_orders = await sync_with_binance(binance, SYMBOL, state, db, notifier)
            
            # Process filled orders from offline period
            if filled_orders:
                logger.info(f"\n[PROCESSING OFFLINE FILLS] {len(filled_orders)} order(s)")
                
                for filled_order in filled_orders:
                    new_order = await process_filled_order(
                        binance, SYMBOL, filled_order, 
                        state['grid_params']['grid_spacing'], state, db, notifier
                    )
                    
                    # Remove from active orders
                    client_id = filled_order['client_order_id']
                    if client_id in state['active_orders']:
                        del state['active_orders'][client_id]
                    
                    # Add new flipped order
                    if new_order:
                        state['active_orders'][new_order['client_order_id']] = new_order
                
                # Save updated state
                save_state(state)
            
            # Process cancelled orders
            if cancelled_orders:
                for cancelled_order in cancelled_orders:
                    client_id = cancelled_order['client_order_id']
                    if client_id in state['active_orders']:
                        del state['active_orders'][client_id]
                save_state(state)
            
            # Send startup notification
            if notifier:
                await notifier.send_startup_message(state['grid_params'], SYMBOL)
            
            # Start monitoring
            await monitor_and_flip(binance, SYMBOL, state, db, notifier)
            
        else:
            # Fresh start
            logger.info("\n[FRESH START] Initializing new grid...")
            
            # Check connection
            connection_ok = await binance.check_connection()
            if not connection_ok:
                logger.error("[ERROR] Failed to connect to Binance. Exiting...")
                return
            
            balance = await binance.exchange.fetch_balance()
            total_balance = balance.get('USDT', {}).get('total', 0)
            
            # Cancel existing orders
            await binance.cancel_all_open_orders(SYMBOL)
            
            # Calculate grid
            current_price = await binance.get_current_price(SYMBOL)
            if not current_price:
                return
            
            # Fetch enough candles for both ATR and EMA
            candles_needed = max(ATR_PERIOD + 10, EMA_PERIOD + 10)
            ohlcv_data = await binance.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=candles_needed)
            if not ohlcv_data:
                return
            
            atr_value = calculate_atr(ohlcv_data, period=ATR_PERIOD)
            
            # PHASE 7: Calculate EMA and determine trend filter
            ema_value = calculate_ema(ohlcv_data, period=EMA_PERIOD)
            trend_filter = get_trend_filter(current_price, ema_value)
            
            logger.info(f"\n[TREND FILTER] EMA-{EMA_PERIOD}: ${ema_value:,.2f}")
            logger.info(f"[TREND FILTER] Current Trend: {trend_filter['trend']}")
            logger.info(f"[TREND FILTER] Allow BUY: {trend_filter['allow_buy']}, Allow SELL: {trend_filter['allow_sell']}")
            
            strategy = GridStrategy(N_LEVELS, ATR_MULTIPLIER, ALLOCATION_PERCENT)
            grid_params = strategy.calculate_grid_params(current_price, atr_value)
            
            # PHASE 7.3: Calculate dynamic allocation based on ATR
            dynamic_allocation = strategy.calculate_dynamic_allocation(ALLOCATION_PERCENT)
            strategy.allocation_percent = dynamic_allocation  # Update strategy with dynamic value
            
            grid_orders = strategy.generate_grid_orders(total_balance)
            
            # Create initial state
            config = {
                'n_levels': N_LEVELS,
                'atr_multiplier': ATR_MULTIPLIER,
                'allocation_percent': ALLOCATION_PERCENT
            }
            state = create_initial_state(SYMBOL, grid_params, config)
            
            # Place initial orders with trend filter
            logger.info(f"\n[PLACING INITIAL GRID] {N_LEVELS} orders (with trend filter)...")
            active_orders = {}
            skipped_count = 0
            
            for grid_order in grid_orders:
                # PHASE 7: Apply trend filter
                side = grid_order['side']
                
                if side == 'BUY' and not trend_filter['allow_buy']:
                    logger.warning(f"  [SKIPPED] Level {grid_order['level']} BUY - Trend filter active (DOWNTREND)")
                    skipped_count += 1
                    continue
                
                if side == 'SELL' and not trend_filter['allow_sell']:
                    logger.warning(f"  [SKIPPED] Level {grid_order['level']} SELL - Trend filter active (UPTREND)")
                    skipped_count += 1
                    continue
                
                new_order = await place_grid_order(
                    binance, SYMBOL, grid_order['level'], grid_order['side'],
                    grid_order['price'], grid_order['amount'], 0
                )
                
                if new_order:
                    active_orders[new_order['client_order_id']] = new_order
                
                await asyncio.sleep(0.5)
            
            logger.info(f"[SUCCESS] {len(active_orders)} orders placed, {skipped_count} skipped by trend filter")
            
            # Update state with active orders
            state['active_orders'] = active_orders
            save_state(state)
            
            # Send startup notification
            if notifier:
                await notifier.send_startup_message(grid_params, SYMBOL)
            
            # Start monitoring
            await monitor_and_flip(binance, SYMBOL, state, db, notifier)
        
    except KeyboardInterrupt:
        logger.info("\n[SHUTDOWN] Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Cleanup
        if db:
            await db.close()
            logger.info("[DATABASE] Connection closed")
        await binance.close()
        remove_lock_file()
        logger.info("[SHUTDOWN] State saved. Safe to restart. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
