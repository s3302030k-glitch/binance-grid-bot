"""
Backtesting Engine for Grid Trading Bot
Simulates grid trading strategy on historical data
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from core.exchange import BinanceManager
from core.strategy import GridStrategy

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting Engine for Grid Trading Strategy
    
    Simulates grid trading on historical OHLCV data without placing real orders.
    Calculates performance metrics: Return %, Max Drawdown, Win Rate.
    """
    
    def __init__(self, binance: BinanceManager, strategy: GridStrategy):
        """
        Initialize Backtester
        
        Args:
            binance: BinanceManager instance for fetching historical data
            strategy: GridStrategy instance with configuration
        """
        self.binance = binance
        self.strategy = strategy
        self.historical_data = []
        
    async def load_data(self, symbol: str, days: int = 30, timeframe: str = '1h') -> bool:
        """
        Load historical OHLCV data from exchange
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            days: Number of days to fetch
            timeframe: Candle timeframe ('1h', '4h', '1d', etc.)
        
        Returns:
            bool: True if data loaded successfully
        """
        try:
            logger.info(f"[BACKTEST] Loading {days} days of {timeframe} data for {symbol}...")
            
            # Calculate how many candles needed
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }
            
            minutes_per_candle = timeframe_minutes.get(timeframe, 60)
            candles_per_day = (24 * 60) // minutes_per_candle
            limit = days * candles_per_day
            
            # Fetch OHLCV data
            ohlcv = await self.binance.fetch_ohlcv(symbol, timeframe, limit=min(limit, 1000))
            
            if not ohlcv:
                logger.error("[BACKTEST] Failed to load historical data")
                return False
            
            # Convert to pandas DataFrame for easier processing
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            self.historical_data = df
            
            logger.info(f"[BACKTEST] Loaded {len(df)} candles")
            logger.info(f"[BACKTEST] Period: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"[BACKTEST ERROR] Failed to load data: {str(e)}")
            return False
    
    async def run_backtest(self, initial_balance: float = 10000.0) -> Dict[str, Any]:
        """
        Run backtest simulation on loaded historical data
        
        Args:
            initial_balance: Starting USDT balance
        
        Returns:
            dict: Backtest results with metrics
        """
        if self.historical_data is None or len(self.historical_data) == 0:
            logger.error("[BACKTEST] No historical data loaded. Call load_data() first.")
            return {}
        
        try:
            logger.info("=" * 80)
            logger.info("[BACKTEST] Starting backtest simulation...")
            logger.info("=" * 80)
            
            df = self.historical_data
            
            # Initialize grid based on first candle
            first_price = df['close'].iloc[0]
            atr_value = self._calculate_atr(df.head(20))
            
            grid_params = self.strategy.calculate_grid_params(first_price, atr_value)
            grid_orders = self.strategy.generate_grid_orders(initial_balance)
            
            logger.info(f"[BACKTEST] Initial price: ${first_price:,.2f}")
            logger.info(f"[BACKTEST] Grid: ${grid_params['lower_bound']:,.2f} - ${grid_params['upper_bound']:,.2f}")
            logger.info(f"[BACKTEST] Grid spacing: ${grid_params['grid_spacing']:,.2f}")
            
            # Initialize state
            balance = initial_balance
            holdings = {}  # {level: {amount, entry_price}}
            active_orders = {order['level']: order for order in grid_orders}
            
            trade_history = []
            equity_curve = []
            max_equity = initial_balance
            max_drawdown = 0.0
            
            # Simulate candle by candle
            for idx, candle in df.iterrows():
                current_price = candle['close']
                candle_low = candle['low']
                candle_high = candle['high']
                timestamp = candle['datetime']
                
                # Check for order fills
                filled_orders = []
                
                for level, order in list(active_orders.items()):
                    if order['side'] == 'BUY' and candle_low <= order['price']:
                        # BUY order filled
                        filled_orders.append(('BUY', level, order))
                        
                    elif order['side'] == 'SELL' and candle_high >= order['price']:
                        # SELL order filled
                        filled_orders.append(('SELL', level, order))
                
                # Process filled orders
                for side, level, order in filled_orders:
                    if side == 'BUY':
                        # Open position
                        holdings[level] = {
                            'amount': order['amount'],
                            'entry_price': order['price']
                        }
                        balance -= order['price'] * order['amount']
                        
                        # Remove BUY order, place SELL order at next level
                        del active_orders[level]
                        next_level = level + 1
                        if next_level <= len(grid_orders):
                            sell_price = order['price'] + grid_params['grid_spacing']
                            active_orders[next_level] = {
                                'level': next_level,
                                'side': 'SELL',
                                'price': sell_price,
                                'amount': order['amount']
                            }
                        
                        trade_history.append({
                            'timestamp': timestamp,
                            'side': 'BUY',
                            'level': level,
                            'price': order['price'],
                            'amount': order['amount'],
                            'profit': 0.0
                        })
                        
                    elif side == 'SELL':
                        # Close position
                        if level in holdings:
                            holding = holdings[level]
                            profit = (order['price'] - holding['entry_price']) * order['amount']
                            balance += order['price'] * order['amount']
                            del holdings[level]
                            
                            # Remove SELL order, place BUY order at previous level
                            del active_orders[level]
                            prev_level = level - 1
                            if prev_level >= 1:
                                buy_price = order['price'] - grid_params['grid_spacing']
                                active_orders[prev_level] = {
                                    'level': prev_level,
                                    'side': 'BUY',
                                    'price': buy_price,
                                    'amount': order['amount']
                                }
                            
                            trade_history.append({
                                'timestamp': timestamp,
                                'side': 'SELL',
                                'level': level,
                                'price': order['price'],
                                'amount': order['amount'],
                                'profit': profit
                            })
                
                # Calculate current equity (balance + value of holdings)
                holdings_value = sum(
                    holding['amount'] * current_price 
                    for holding in holdings.values()
                )
                current_equity = balance + holdings_value
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity,
                    'balance': balance,
                    'holdings_value': holdings_value
                })
                
                # Track max drawdown
                if current_equity > max_equity:
                    max_equity = current_equity
                drawdown = (max_equity - current_equity) / max_equity * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate final results
            final_equity = equity_curve[-1]['equity']
            total_return = (final_equity - initial_balance) / initial_balance * 100
            
            # Calculate win rate
            profitable_trades = [t for t in trade_history if t['side'] == 'SELL' and t['profit'] > 0]
            sell_trades = [t for t in trade_history if t['side'] == 'SELL']
            win_rate = (len(profitable_trades) / len(sell_trades) * 100) if sell_trades else 0
            
            # Total profit
            total_profit = sum(t['profit'] for t in trade_history if t['side'] == 'SELL')
            
            results = {
                'initial_balance': initial_balance,
                'final_equity': final_equity,
                'total_return_pct': total_return,
                'total_profit': total_profit,
                'max_drawdown_pct': max_drawdown,
                'total_trades': len(trade_history),
                'sell_trades': len(sell_trades),
                'profitable_trades': len(profitable_trades),
                'win_rate_pct': win_rate,
                'equity_curve': equity_curve,
                'trade_history': trade_history,
                'grid_params': grid_params
            }
            
            self._print_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"[BACKTEST ERROR] Simulation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range from OHLCV DataFrame"""
        df = df.copy()
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        atr = df['tr'].rolling(window=period).mean().iloc[-1]
        return atr
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted backtest results"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 BACKTEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Initial Balance:   ${results['initial_balance']:,.2f}")
        logger.info(f"Final Equity:      ${results['final_equity']:,.2f}")
        logger.info(f"Total Profit:      ${results['total_profit']:,.2f}")
        logger.info(f"Total Return:      {results['total_return_pct']:,.2f}%")
        logger.info(f"Max Drawdown:      {results['max_drawdown_pct']:,.2f}%")
        logger.info(f"Total Trades:      {results['total_trades']}")
        logger.info(f"Sell Trades:       {results['sell_trades']}")
        logger.info(f"Profitable Trades: {results['profitable_trades']}")
        logger.info(f"Win Rate:          {results['win_rate_pct']:,.2f}%")
        logger.info("=" * 80)
        
        # Performance rating
        if results['total_return_pct'] > 10:
            logger.info("🟢 Excellent Performance!")
        elif results['total_return_pct'] > 5:
            logger.info("🟡 Good Performance")
        elif results['total_return_pct'] > 0:
            logger.info("🟠 Moderate Performance")
        else:
            logger.info("🔴 Poor Performance - Strategy needs adjustment")
        
        logger.info("=" * 80 + "\n")
