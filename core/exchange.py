"""
Binance Exchange Manager
Handles all interactions with Binance exchange using async ccxt
"""

import ccxt.async_support as ccxt
import logging
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/exchange.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class BinanceManager:
    """
    Async Binance Exchange Manager
    
    Manages connection, authentication, and operations with Binance exchange.
    Supports both Testnet and Mainnet based on USE_TESTNET environment variable.
    """
    
    def __init__(self):
        """Initialize Binance exchange with async support"""
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.use_testnet = os.getenv('USE_TESTNET', 'True').lower() == 'true'
        
        if not self.api_key or not self.api_secret:
            logger.error("API credentials not found in environment variables")
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file")
        
        # Initialize exchange with async support
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,  # Critical for Binance to avoid rate limit errors
            'options': {
                'defaultType': 'spot',  # IMPORTANT: Testnet only supports 'spot' market
            }
        })
        
        # Configure testnet if enabled
        if self.use_testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("[TESTNET] Binance Testnet mode enabled")
        else:
            logger.warning("[MAINNET] Binance MAINNET mode - Real trading enabled!")
        
        logger.info(f"BinanceManager initialized - Testnet: {self.use_testnet}")
    
    async def check_connection(self) -> bool:
        """
        Check connection to Binance and verify API credentials
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Checking Binance connection...")
            
            # Fetch balance to verify authentication and connection
            balance = await self.exchange.fetch_balance()
            
            # Get USDT balance
            usdt_balance = balance.get('USDT', {})
            total_usdt = usdt_balance.get('total', 0)
            free_usdt = usdt_balance.get('free', 0)
            used_usdt = usdt_balance.get('used', 0)
            
            logger.info("[SUCCESS] Connection successful!")
            logger.info("[BALANCE] USDT Balance:")
            logger.info(f"   - Total: {total_usdt:.2f} USDT")
            logger.info(f"   - Free:  {free_usdt:.2f} USDT")
            logger.info(f"   - Used:  {used_usdt:.2f} USDT")
            
            return True
            
        except ccxt.AuthenticationError as e:
            logger.error(f"[AUTH ERROR] Authentication failed: {str(e)}")
            logger.error("Please check your API_KEY and API_SECRET in .env file")
            return False
            
        except ccxt.NetworkError as e:
            logger.error(f"[NETWORK ERROR] Network error: {str(e)}")
            logger.error("Please check your internet connection or Binance API status")
            return False
            
        except ccxt.ExchangeError as e:
            logger.error(f"[EXCHANGE ERROR] Exchange error: {str(e)}")
            return False
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error: {str(e)}")
            return False
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetch the current price for a given symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
        Returns:
            float: Current 'last' price, or None if error occurs
        """
        try:
            logger.info(f"Fetching current price for {symbol}...")
            
            # Fetch ticker data
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Get last price
            last_price = ticker.get('last')
            
            if last_price is None:
                logger.warning(f"[WARNING] No price data available for {symbol}")
                return None
            
            logger.info(f"[PRICE] {symbol} Current Price: ${last_price:,.2f}")
            return last_price
            
        except ccxt.NetworkError as e:
            logger.error(f"[NETWORK ERROR] Network error while fetching price for {symbol}: {str(e)}")
            return None
            
        except ccxt.ExchangeError as e:
            logger.error(f"[EXCHANGE ERROR] Exchange error while fetching price for {symbol}: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error while fetching price for {symbol}: {str(e)}")
            return None
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[list]:
        """
        Fetch historical OHLCV (Open, High, Low, Close, Volume) data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to fetch (default: 100)
        
        Returns:
            List of OHLCV data, or None if error occurs
            Format: [[timestamp, open, high, low, close, volume], ...]
        """
        try:
            logger.info(f"Fetching {limit} candles for {symbol} ({timeframe})...")
            
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                logger.warning(f"[WARNING] No OHLCV data returned for {symbol}")
                return None
            
            logger.info(f"[SUCCESS] Fetched {len(ohlcv)} candles for {symbol}")
            return ohlcv
            
        except ccxt.NetworkError as e:
            logger.error(f"[NETWORK ERROR] Network error while fetching OHLCV for {symbol}: {str(e)}")
            return None
            
        except ccxt.ExchangeError as e:
            logger.error(f"[EXCHANGE ERROR] Exchange error while fetching OHLCV for {symbol}: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error while fetching OHLCV for {symbol}: {str(e)}")
            return None
    
    async def place_limit_order(self, symbol: str, side: str, amount: float, price: float, 
                               client_order_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Place a limit order on the exchange
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Limit price
            client_order_id: Optional custom client order ID for tracking
        
        Returns:
            Order response dict, or None if error occurs
        """
        try:
            logger.info(f"Placing {side.upper()} limit order for {symbol}...")
            logger.info(f"  Amount: {amount:.6f}, Price: ${price:,.2f}")
            
            # Prepare order params
            params = {}
            if client_order_id:
                params['clientOrderId'] = client_order_id
                logger.info(f"  Client Order ID: {client_order_id}")
            
            # Place limit order
            order = await self.exchange.create_limit_order(
                symbol=symbol,
                side=side.lower(),
                amount=amount,
                price=price,
                params=params
            )
            
            order_id = order.get('id', 'N/A')
            logger.info(f"[SUCCESS] Order placed successfully!")
            logger.info(f"  Order ID: {order_id}")
            logger.info(f"  Status: {order.get('status', 'N/A')}")
            
            return order
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"[INSUFFICIENT FUNDS] Not enough balance to place {side.upper()} order")
            logger.error(f"  Symbol: {symbol}, Amount: {amount:.6f}, Price: ${price:,.2f}")
            logger.error(f"  Error: {str(e)}")
            return None
            
        except ccxt.InvalidOrder as e:
            logger.error(f"[INVALID ORDER] Order parameters are invalid")
            logger.error(f"  Symbol: {symbol}, Side: {side.upper()}, Amount: {amount:.6f}, Price: ${price:,.2f}")
            logger.error(f"  Error: {str(e)}")
            return None
            
        except ccxt.NetworkError as e:
            logger.error(f"[NETWORK ERROR] Network error while placing order: {str(e)}")
            return None
            
        except ccxt.ExchangeError as e:
            logger.error(f"[EXCHANGE ERROR] Exchange error while placing order: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error while placing order: {str(e)}")
            return None
    
    async def get_open_orders(self, symbol: str = None) -> Optional[list]:
        """
        Fetch all open (active) orders
        
        Args:
            symbol: Optional trading pair symbol to filter orders
        
        Returns:
            List of open orders, or None if error occurs
        """
        try:
            logger.info(f"Fetching open orders{f' for {symbol}' if symbol else ''}...")
            
            # Fetch open orders
            orders = await self.exchange.fetch_open_orders(symbol)
            
            logger.info(f"[SUCCESS] Found {len(orders)} open order(s)")
            return orders
            
        except ccxt.NetworkError as e:
            logger.error(f"[NETWORK ERROR] Network error while fetching open orders: {str(e)}")
            return None
            
        except ccxt.ExchangeError as e:
            logger.error(f"[EXCHANGE ERROR] Exchange error while fetching open orders: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error while fetching open orders: {str(e)}")
            return None
    
    async def cancel_all_open_orders(self, symbol: str) -> int:
        """
        Cancel all open orders for a specific symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
        Returns:
            Number of orders cancelled
        """
        try:
            logger.info(f"Cancelling all open orders for {symbol}...")
            
            # Fetch open orders
            orders = await self.get_open_orders(symbol)
            
            if orders is None:
                logger.error("[ERROR] Failed to fetch open orders")
                return 0
            
            if len(orders) == 0:
                logger.info("[INFO] No open orders to cancel")
                return 0
            
            # Cancel each order
            cancelled_count = 0
            for order in orders:
                try:
                    order_id = order.get('id')
                    await self.exchange.cancel_order(order_id, symbol)
                    cancelled_count += 1
                    logger.info(f"  Cancelled order: {order_id}")
                except Exception as e:
                    logger.error(f"  Failed to cancel order {order_id}: {str(e)}")
            
            logger.info(f"[SUCCESS] Cancelled {cancelled_count} order(s)")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error while cancelling orders: {str(e)}")
            return 0
    
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch details of a specific order by ID
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
        
        Returns:
            Order details dict, or None if error occurs
        """
        try:
            logger.info(f"Fetching order details for ID: {order_id}...")
            
            order = await self.exchange.fetch_order(order_id, symbol)
            
            logger.info(f"[SUCCESS] Order {order_id} status: {order.get('status', 'N/A')}")
            return order
            
        except ccxt.OrderNotFound as e:
            logger.error(f"[ORDER NOT FOUND] Order {order_id} does not exist: {str(e)}")
            return None
            
        except ccxt.NetworkError as e:
            logger.error(f"[NETWORK ERROR] Network error while fetching order: {str(e)}")
            return None
            
        except ccxt.ExchangeError as e:
            logger.error(f"[EXCHANGE ERROR] Exchange error while fetching order: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error while fetching order: {str(e)}")
            return None
    
    async def close(self):
        """Close the exchange connection"""
        try:
            await self.exchange.close()
            logger.info("[CLOSED] Exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing exchange connection: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of BinanceManager"""
        mode = "Testnet" if self.use_testnet else "Mainnet"
        return f"<BinanceManager(mode={mode})>"
