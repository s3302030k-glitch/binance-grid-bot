"""
Grid Trading Strategy
Implements ATR-based dynamic grid calculation
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class GridStrategy:
    """
    Grid Trading Strategy Implementation
    
    Uses ATR (Average True Range) to dynamically calculate grid levels
    based on market volatility.
    """
    
    def __init__(self, n_levels: int = 10, atr_multiplier: float = 2.0, allocation_percent: float = 2.0):
        """
        Initialize Grid Strategy
        
        Args:
            n_levels: Number of grid levels (default: 10)
            atr_multiplier: Multiplier for ATR to define bounds (default: 2.0)
            allocation_percent: Percentage of balance per grid level (default: 2.0%)
        """
        self.n_levels = n_levels
        self.atr_multiplier = atr_multiplier
        self.allocation_percent = allocation_percent
        
        # Grid parameters (calculated by calculate_grid_params)
        self.lower_bound = None
        self.upper_bound = None
        self.grid_spacing = None
        self.current_price = None
        
        logger.info(f"GridStrategy initialized: {n_levels} levels, ATR multiplier: {atr_multiplier}x, "
                   f"allocation: {allocation_percent}% per level")
    
    def calculate_grid_params(self, current_price: float, atr_value: float) -> Dict[str, float]:
        """
        Calculate grid parameters based on current price and ATR
        
        Args:
            current_price: Current market price
            atr_value: Average True Range value
        
        Returns:
            Dict containing lower_bound, upper_bound, and grid_spacing
        """
        logger.info(f"Calculating grid parameters...")
        logger.info(f"  Current Price: ${current_price:,.2f}")
        logger.info(f"  ATR Value: ${atr_value:,.2f}")
        logger.info(f"  ATR Multiplier: {self.atr_multiplier}x")
        
        # Calculate bounds based on ATR
        atr_offset = atr_value * self.atr_multiplier
        self.lower_bound = current_price - atr_offset
        self.upper_bound = current_price + atr_offset
        self.current_price = current_price
        self.atr_value = atr_value  # PHASE 7.3: Store for dynamic allocation
        
        # Calculate spacing between grid levels
        self.grid_spacing = (self.upper_bound - self.lower_bound) / (self.n_levels - 1)
        
        logger.info(f"[GRID BOUNDS]")
        logger.info(f"  Lower Bound: ${self.lower_bound:,.2f}")
        logger.info(f"  Upper Bound: ${self.upper_bound:,.2f}")
        logger.info(f"  Grid Spacing: ${self.grid_spacing:,.2f}")
        
        return {
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'grid_spacing': self.grid_spacing,
            'atr_offset': atr_offset,
            'atr_value': atr_value  # Include ATR in params
        }
    
    def calculate_dynamic_allocation(self, base_allocation: float = 2.0) -> float:
        """
        PHASE 7.3: Calculate allocation based on volatility
        
        Higher ATR = smaller allocation (risk normalization)
        Lower ATR = larger allocation (capitalize on stability)
        
        Args:
            base_allocation: Base allocation percentage (default: 2.0%)
        
        Returns:
            float: Adjusted allocation percentage (clamped between 0.5% and 5.0%)
        
        Formula:
            allocation = base_allocation × (baseline_atr_pct / current_atr_pct)
        """
        if self.current_price is None or self.atr_value is None:
            logger.warning("[DYNAMIC ALLOCATION] No price/ATR data, using base allocation")
            return base_allocation
        
        # Calculate ATR as percentage of price
        atr_pct = (self.atr_value / self.current_price) * 100
        
        # Baseline ATR percentage (typical BTC volatility ~2%)
        baseline_atr_pct = 2.0
        
        # Calculate adjusted allocation (inverse relationship)
        adjusted_allocation = base_allocation * (baseline_atr_pct / max(atr_pct, 0.5))
        
        # Clamp between 0.5% and 5.0%
        final_allocation = max(0.5, min(5.0, adjusted_allocation))
        
        logger.info(f"[DYNAMIC ALLOCATION] ATR%: {atr_pct:.2f}% | Base: {base_allocation}% | Adjusted: {final_allocation:.2f}%")
        
        return final_allocation
    
    def generate_grid_orders(self, total_balance: float) -> List[Dict[str, Any]]:
        """
        Generate grid order levels with prices, sides, and amounts
        
        Args:
            total_balance: Total USDT balance available
        
        Returns:
            List of grid orders with 'level', 'price', 'side', 'amount', 'usdt_amount'
        """
        if self.lower_bound is None or self.upper_bound is None:
            raise ValueError("Grid parameters not calculated. Call calculate_grid_params() first.")
        
        logger.info(f"\n[GENERATING GRID ORDERS]")
        logger.info(f"  Total Balance: ${total_balance:,.2f} USDT")
        logger.info(f"  Allocation per level: {self.allocation_percent}%")
        
        grid_orders = []
        usdt_per_level = (total_balance * self.allocation_percent) / 100
        
        for i in range(self.n_levels):
            # Calculate price for this level
            price = self.lower_bound + (i * self.grid_spacing)
            
            # Determine order side (BUY below current price, SELL above)
            if price < self.current_price:
                side = 'BUY'
            elif price > self.current_price:
                side = 'SELL'
            else:
                # If exactly at current price, skip or make it BUY
                side = 'BUY'
            
            # Calculate amount (in base currency, e.g., BTC)
            amount = usdt_per_level / price
            
            order = {
                'level': i + 1,
                'price': price,
                'side': side,
                'amount': amount,  # Amount in base currency (BTC/ETH)
                'usdt_amount': usdt_per_level  # Amount in USDT
            }
            
            grid_orders.append(order)
        
        logger.info(f"[SUCCESS] Generated {len(grid_orders)} grid orders")
        
        return grid_orders
    
    def print_grid_summary(self, grid_orders: List[Dict[str, Any]], symbol: str = "BTC/USDT"):
        """
        Print a formatted summary of grid orders
        
        Args:
            grid_orders: List of grid orders from generate_grid_orders()
            symbol: Trading pair symbol
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"GRID TRADING STRATEGY SUMMARY - {symbol}")
        logger.info(f"{'='*80}")
        logger.info(f"{'Level':<8} {'Price':<15} {'Side':<8} {'Amount':<15} {'USDT Value':<15}")
        logger.info(f"{'-'*80}")
        
        buy_orders = [o for o in grid_orders if o['side'] == 'BUY']
        sell_orders = [o for o in grid_orders if o['side'] == 'SELL']
        
        for order in grid_orders:
            logger.info(
                f"{order['level']:<8} "
                f"${order['price']:>13,.2f} "
                f"{order['side']:<8} "
                f"{order['amount']:>13.6f} "
                f"${order['usdt_amount']:>13.2f}"
            )
        
        logger.info(f"{'-'*80}")
        logger.info(f"Total BUY orders:  {len(buy_orders)}")
        logger.info(f"Total SELL orders: {len(sell_orders)}")
        logger.info(f"Total orders:      {len(grid_orders)}")
        logger.info(f"{'='*80}\n")
    
    
    def __repr__(self) -> str:
        """String representation of GridStrategy"""
        return (f"<GridStrategy(levels={self.n_levels}, "
                f"atr_multiplier={self.atr_multiplier}, "
                f"allocation={self.allocation_percent}%)>")


# State Management Functions

import os
import json
import shutil
from typing import Optional
from datetime import datetime

STATE_FILE = 'data/state.json'
BACKUP_FILE = 'data/state.backup.json'
TMP_FILE = 'data/state.tmp.json'


def save_state(state_data: dict) -> bool:
    """
    Save state to JSON file using atomic writing
    
    Uses atomic writing pattern:
    1. Write to temporary file
    2. Backup existing state
    3. Atomically replace with temp file
    
    Args:
        state_data: State dictionary to save
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Update timestamp
        state_data['last_update'] = datetime.now().isoformat()
        
        # Step 1: Write to temporary file
        with open(TMP_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        # Step 2: Backup existing state (if exists)
        if os.path.exists(STATE_FILE):
            shutil.copy(STATE_FILE, BACKUP_FILE)
        
        # Step 3: Atomically replace with temp file
        # os.replace() is atomic on both Windows and Unix
        os.replace(TMP_FILE, STATE_FILE)
        
        logger.debug("[STATE] Saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to save state: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(TMP_FILE):
            try:
                os.remove(TMP_FILE)
            except:
                pass
        return False


def load_state() -> Optional[dict]:
    """
    Load state from JSON file
    
    Attempts to load state.json, falls back to backup if corrupted
    
    Returns:
        dict: State dictionary, or None if no valid state found
    """
    try:
        if not os.path.exists(STATE_FILE):
            logger.info("[STATE] No previous state found, starting fresh")
            return None
        
        # Try to load main state file
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Validate state structure
        if validate_state(state):
            logger.info(f"[STATE] Loaded previous state from {STATE_FILE}")
            logger.info(f"  Last update: {state.get('last_update', 'N/A')}")
            logger.info(f"  Active orders: {len(state.get('active_orders', {}))}")
            logger.info(f"  Total trades: {state.get('stats', {}).get('total_trades', 0)}")
            return state
        else:
            logger.warning("[STATE] Main state file is invalid, trying backup...")
            raise ValueError("Invalid state structure")
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to load main state: {str(e)}")
        
        # Try backup
        try:
            if os.path.exists(BACKUP_FILE):
                logger.info("[STATE] Attempting to load backup...")
                with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                if validate_state(state):
                    logger.info("[STATE] Loaded from backup successfully")
                    # Restore backup as main state
                    shutil.copy(BACKUP_FILE, STATE_FILE)
                    return state
                else:
                    logger.error("[STATE] Backup is also invalid")
            else:
                logger.error("[STATE] No backup file found")
        except Exception as backup_error:
            logger.error(f"[ERROR] Backup load failed: {str(backup_error)}")
        
        logger.warning("[STATE] Could not load any valid state, starting fresh")
        return None


def validate_state(state: dict) -> bool:
    """
    Validate state structure
    
    Args:
        state: State dictionary to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ['active_orders', 'trade_history', 'stats', 'grid_params', 'is_emergency_stopped']
    
    for key in required_keys:
        if key not in state:
            logger.error(f"[STATE VALIDATION] Missing required key: {key}")
            return False
    
    # Validate active_orders structure
    if not isinstance(state['active_orders'], dict):
        logger.error("[STATE VALIDATION] active_orders must be a dict")
        return False
    
    # Validate stats structure
    if not isinstance(state['stats'], dict):
        logger.error("[STATE VALIDATION] stats must be a dict")
        return False
    
    # Validate is_emergency_stopped is boolean
    if not isinstance(state['is_emergency_stopped'], bool):
        logger.error("[STATE VALIDATION] is_emergency_stopped must be a boolean")
        return False
    
    return True


def create_initial_state(symbol: str, grid_params: dict, config: dict) -> dict:
    """
    Create initial state structure
    
    Args:
        symbol: Trading pair symbol
        grid_params: Grid parameters from calculate_grid_params()
        config: Configuration dict with n_levels, atr_multiplier, etc.
    
    Returns:
        dict: Initial state structure
    """
    # Calculate emergency stop threshold
    stop_loss_buffer = 0.02  # 2% below lower bound
    lower_bound = grid_params.get('lower_bound', 0)
    emergency_stop_price = lower_bound * (1 - stop_loss_buffer) if lower_bound > 0 else 0
    
    return {
        'version': '1.0',
        'last_update': datetime.now().isoformat(),
        'symbol': symbol,
        'config': config,
        'grid_params': grid_params,
        'active_orders': {},
        'trade_history': [],
        'stats': {
            'total_trades': 0,
            'total_profit_usd': 0.0,
            'fills_count': 0,
            'cancels_count': 0,
            'started_at': datetime.now().isoformat(),
            'last_trade_at': None
        },
        'iteration_counter': {},
        'is_emergency_stopped': False,
        'stop_loss_params': {
            'stop_loss_buffer': stop_loss_buffer,
            'emergency_stop_price': emergency_stop_price,
            'lower_bound': lower_bound
        }
    }
