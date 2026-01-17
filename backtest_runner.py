"""
Backtest Runner Script
Command-line interface for running backtests
"""

import asyncio
import argparse
import logging
import os
from dotenv import load_dotenv

from core.exchange import BinanceManager
from core.strategy import GridStrategy
from core.backtest import Backtester

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for backtest runner"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Grid Trading Bot - Backtest Runner')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', 
                       help='Trading pair symbol (default: BTC/USDT)')
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of days to backtest (default: 30)')
    parser.add_argument('--timeframe', type=str, default='1h', 
                       help='Candle timeframe (default: 1h)')
    parser.add_argument('--balance', type=float, default=10000.0, 
                       help='Initial balance in USDT (default: 10000)')
    parser.add_argument('--levels', type=int, default=10, 
                       help='Number of grid levels (default: 10)')
    parser.add_argument('--atr-multiplier', type=float, default=2.0, 
                       help='ATR multiplier for grid bounds (default: 2.0)')
    parser.add_argument('--allocation', type=float, default=2.0, 
                       help='Allocation percentage per level (default: 2.0%)')
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("Grid Trading Bot - Backtest Mode")
        logger.info("=" * 80)
        logger.info(f"Symbol:         {args.symbol}")
        logger.info(f"Period:         {args.days} days")
        logger.info(f"Timeframe:      {args.timeframe}")
        logger.info(f"Initial Balance: ${args.balance:,.2f}")
        logger.info(f"Grid Levels:    {args.levels}")
        logger.info(f"ATR Multiplier: {args.atr_multiplier}x")
        logger.info(f"Allocation:     {args.allocation}% per level")
        logger.info("=" * 80 + "\n")
        
        # Initialize Binance Manager
        binance = BinanceManager()
        
        # Check connection
        connection_ok = await binance.check_connection()
        if not connection_ok:
            logger.error("[ERROR] Failed to connect to Binance. Exiting...")
            return
        
        # Initialize GridStrategy
        strategy = GridStrategy(
            n_levels=args.levels,
            atr_multiplier=args.atr_multiplier,
            allocation_percent=args.allocation
        )
        
        # Initialize Backtester
        backtester = Backtester(binance, strategy)
        
        # Load historical data
        data_loaded = await backtester.load_data(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe
        )
        
        if not data_loaded:
            logger.error("[ERROR] Failed to load historical data. Exiting...")
            return
        
        # Run backtest
        results = await backtester.run_backtest(initial_balance=args.balance)
        
        if not results:
            logger.error("[ERROR] Backtest failed. Check logs for details.")
            return
        
        # Save results to file (optional)
        save_results = input("\n💾 Save results to file? (y/n): ").lower()
        if save_results == 'y':
            import json
            from datetime import datetime
            
            filename = f"backtest_{args.symbol.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join('data', filename)
            
            # Prepare results for JSON (convert timestamps)
            json_results = {
                'config': {
                    'symbol': args.symbol,
                    'days': args.days,
                    'timeframe': args.timeframe,
                    'initial_balance': args.balance,
                    'levels': args.levels,
                    'atr_multiplier': args.atr_multiplier,
                    'allocation_percent': args.allocation
                },
                'metrics': {
                    'initial_balance': results['initial_balance'],
                    'final_equity': results['final_equity'],
                    'total_return_pct': results['total_return_pct'],
                    'total_profit': results['total_profit'],
                    'max_drawdown_pct': results['max_drawdown_pct'],
                    'total_trades': results['total_trades'],
                    'sell_trades': results['sell_trades'],
                    'profitable_trades': results['profitable_trades'],
                    'win_rate_pct': results['win_rate_pct']
                },
                'grid_params': results['grid_params']
            }
            
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"✅ Results saved to: {filepath}")
        
        # Close connection
        await binance.close()
        
        logger.info("\n✅ Backtest completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n[SHUTDOWN] Backtest stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if 'binance' in locals():
            await binance.close()


if __name__ == "__main__":
    asyncio.run(main())
