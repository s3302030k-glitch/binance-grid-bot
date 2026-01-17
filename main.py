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


def calculate_atr(ohlcv_data: list, period: int = 14) -> float:
    """Calculate Average True Range (ATR) from OHLCV data"""
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    atr = df['tr'].rolling(window=period).mean().iloc[-1]
    return atr


def generate_client_order_id(level: int, side: str, iteration: int = 0) -> str:
    """Generate a unique client order ID for tracking"""
    timestamp = int(time.time())
    if iteration > 0:
        return f"grid_{timestamp}_{level}_{side}_i{iteration}"
    return f"grid_{timestamp}_{level}_{side}"


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
                               notifier: Optional[TelegramNotifier] = None) -> Optional[Dict]:
    """
    Process a filled order and flip it
    
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
        # SELL filled → Place BUY at previous level
        prev_level = level - 1
        if prev_level >= 1:
            flip_price = price - grid_spacing
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
                            # Process filled order
                            new_order = await process_filled_order(
                                binance, symbol, order_info, grid_spacing, state, db, notifier
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
            
            ohlcv_data = await binance.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=ATR_PERIOD + 10)
            if not ohlcv_data:
                return
            
            atr_value = calculate_atr(ohlcv_data, period=ATR_PERIOD)
            
            strategy = GridStrategy(N_LEVELS, ATR_MULTIPLIER, ALLOCATION_PERCENT)
            grid_params = strategy.calculate_grid_params(current_price, atr_value)
            grid_orders = strategy.generate_grid_orders(total_balance)
            
            # Create initial state
            config = {
                'n_levels': N_LEVELS,
                'atr_multiplier': ATR_MULTIPLIER,
                'allocation_percent': ALLOCATION_PERCENT
            }
            state = create_initial_state(SYMBOL, grid_params, config)
            
            # Place initial orders
            logger.info(f"\n[PLACING INITIAL GRID] {N_LEVELS} orders...")
            active_orders = {}
            
            for grid_order in grid_orders:
                new_order = await place_grid_order(
                    binance, SYMBOL, grid_order['level'], grid_order['side'],
                    grid_order['price'], grid_order['amount'], 0
                )
                
                if new_order:
                    active_orders[new_order['client_order_id']] = new_order
                
                await asyncio.sleep(0.5)
            
            logger.info(f"[SUCCESS] {len(active_orders)} orders placed")
            
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
