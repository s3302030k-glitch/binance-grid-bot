"""
Database Manager for Grid Trading Bot
Phase 6: SQLite-based Trade History Storage
"""

import aiosqlite
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = 'data/trades.db'


class DatabaseManager:
    """
    Async SQLite Database Manager for Trade History
    
    Handles persistent storage of completed trades for analytics and reporting.
    Uses dual-storage architecture:
    - state.json: Hot state (active orders, emergency flags)
    - SQLite: Cold storage (historical trade records)
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize Database Manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection: Optional[aiosqlite.Connection] = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        logger.info(f"[DB] DatabaseManager initialized - Path: {db_path}")
    
    async def init_db(self):
        """
        Initialize database and create tables if they don't exist
        
        Creates the 'trades' table with proper schema.
        Safe to call multiple times (CREATE TABLE IF NOT EXISTS).
        """
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            
            # Enable foreign keys and WAL mode for better concurrency
            await self.connection.execute("PRAGMA foreign_keys = ON")
            await self.connection.execute("PRAGMA journal_mode = WAL")
            
            # Create trades table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    profit REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    order_id TEXT,
                    level INTEGER,
                    client_order_id TEXT UNIQUE,
                    status TEXT DEFAULT 'filled'
                )
            """)
            
            # Create indexes for common queries
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
                ON trades(timestamp DESC)
            """)
            
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                ON trades(symbol, timestamp DESC)
            """)
            
            await self.connection.commit()
            
            logger.info("[DB] Database initialized successfully")
            logger.info(f"[DB] Tables created/verified in {self.db_path}")
            
        except Exception as e:
            logger.error(f"[DB ERROR] Failed to initialize database: {str(e)}")
            raise
    
    async def insert_trade(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """
        Insert a completed trade into the database
        
        Args:
            trade_data: Dictionary containing trade information
                Required keys: symbol, side, price, amount, timestamp
                Optional keys: profit, order_id, level, client_order_id
        
        Returns:
            int: Trade ID (primary key) if successful, None otherwise
        """
        try:
            if not self.connection:
                logger.error("[DB ERROR] Database not initialized. Call init_db() first.")
                return None
            
            # Extract fields
            symbol = trade_data.get('symbol')
            side = trade_data.get('side')
            price = trade_data.get('price')
            amount = trade_data.get('amount')
            profit = trade_data.get('profit', 0.0)
            timestamp = trade_data.get('timestamp', datetime.now().isoformat())
            order_id = trade_data.get('order_id', '')
            level = trade_data.get('level', 0)
            client_order_id = trade_data.get('client_order_id', '')
            status = trade_data.get('status', 'filled')
            
            # Validate required fields
            if not all([symbol, side, price, amount]):
                logger.error("[DB ERROR] Missing required fields for trade insertion")
                return None
            
            # Insert trade
            cursor = await self.connection.execute("""
                INSERT INTO trades (
                    symbol, side, price, amount, profit, 
                    timestamp, order_id, level, client_order_id, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, side, price, amount, profit, timestamp, 
                  order_id, level, client_order_id, status))
            
            await self.connection.commit()
            
            trade_id = cursor.lastrowid
            logger.info(f"[DB] Trade inserted successfully - ID: {trade_id}, "
                       f"{side} {amount:.6f} @ ${price:,.2f}, Profit: ${profit:,.2f}")
            
            return trade_id
            
        except aiosqlite.IntegrityError as e:
            logger.warning(f"[DB WARNING] Duplicate trade (client_order_id already exists): {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"[DB ERROR] Failed to insert trade: {str(e)}")
            return None
    
    async def get_trades(self, symbol: Optional[str] = None, 
                        limit: int = 100, 
                        offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve trade history from database
        
        Args:
            symbol: Optional symbol filter (e.g., 'BTC/USDT')
            limit: Maximum number of trades to retrieve
            offset: Number of trades to skip (for pagination)
        
        Returns:
            List of trade dictionaries
        """
        try:
            if not self.connection:
                logger.error("[DB ERROR] Database not initialized")
                return []
            
            if symbol:
                query = """
                    SELECT * FROM trades 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """
                cursor = await self.connection.execute(query, (symbol, limit, offset))
            else:
                query = """
                    SELECT * FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """
                cursor = await self.connection.execute(query, (limit, offset))
            
            rows = await cursor.fetchall()
            
            # Convert to list of dictionaries
            trades = []
            for row in rows:
                trades.append({
                    'id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'price': row[3],
                    'amount': row[4],
                    'profit': row[5],
                    'timestamp': row[6],
                    'order_id': row[7],
                    'level': row[8],
                    'client_order_id': row[9],
                    'status': row[10]
                })
            
            logger.info(f"[DB] Retrieved {len(trades)} trades from database")
            return trades
            
        except Exception as e:
            logger.error(f"[DB ERROR] Failed to retrieve trades: {str(e)}")
            return []
    
    async def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trading statistics from database
        
        Args:
            symbol: Optional symbol filter
        
        Returns:
            Dictionary with statistics (total trades, profit, win rate, etc.)
        """
        try:
            if not self.connection:
                logger.error("[DB ERROR] Database not initialized")
                return {}
            
            # Build query based on symbol filter
            where_clause = "WHERE symbol = ?" if symbol else ""
            params = (symbol,) if symbol else ()
            
            # Total trades and profit
            query = f"""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(profit) as total_profit,
                    AVG(profit) as avg_profit,
                    MAX(profit) as max_profit,
                    MIN(profit) as min_profit,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                    SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sell_count
                FROM trades 
                {where_clause}
            """
            
            cursor = await self.connection.execute(query, params)
            row = await cursor.fetchone()
            
            if row and row[0] > 0:
                total_trades = row[0] or 0
                winning_trades = row[5] or 0
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                stats = {
                    'total_trades': total_trades,
                    'total_profit': row[1] or 0.0,
                    'avg_profit': row[2] or 0.0,
                    'max_profit': row[3] or 0.0,
                    'min_profit': row[4] or 0.0,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'buy_count': row[6] or 0,
                    'sell_count': row[7] or 0
                }
                
                logger.info(f"[DB] Stats calculated - Total trades: {total_trades}, "
                           f"Total profit: ${stats['total_profit']:,.2f}, Win rate: {win_rate:.2f}%")
                return stats
            else:
                logger.info("[DB] No trades found in database")
                return {
                    'total_trades': 0,
                    'total_profit': 0.0,
                    'avg_profit': 0.0,
                    'max_profit': 0.0,
                    'min_profit': 0.0,
                    'winning_trades': 0,
                    'win_rate': 0.0,
                    'buy_count': 0,
                    'sell_count': 0
                }
                
        except Exception as e:
            logger.error(f"[DB ERROR] Failed to calculate stats: {str(e)}")
            return {}
    
    async def migrate_from_state(self, state_data: Dict[str, Any]) -> int:
        """
        Migrate existing trade history from state.json to database
        
        Args:
            state_data: State dictionary loaded from state.json
        
        Returns:
            Number of trades migrated
        """
        try:
            trade_history = state_data.get('trade_history', [])
            
            if not trade_history:
                logger.info("[DB] No trades to migrate from state.json")
                return 0
            
            migrated_count = 0
            
            for trade in trade_history:
                # Skip events (like EMERGENCY_STOP)
                if 'event' in trade:
                    continue
                
                # Prepare trade data for insertion
                trade_data = {
                    'symbol': state_data.get('symbol', 'UNKNOWN'),
                    'side': trade.get('side', 'UNKNOWN'),
                    'price': trade.get('price', 0.0),
                    'amount': trade.get('amount', 0.0),
                    'profit': trade.get('profit_usd', 0.0),
                    'timestamp': trade.get('timestamp', datetime.now().isoformat()),
                    'client_order_id': trade.get('client_order_id', ''),
                    'level': trade.get('level', 0),
                    'status': trade.get('status', 'filled')
                }
                
                # Insert into database (skip if duplicate)
                trade_id = await self.insert_trade(trade_data)
                if trade_id:
                    migrated_count += 1
            
            logger.info(f"[DB] Migrated {migrated_count} trades from state.json to database")
            return migrated_count
            
        except Exception as e:
            logger.error(f"[DB ERROR] Failed to migrate trades: {str(e)}")
            return 0
    
    async def close(self):
        """Close database connection"""
        try:
            if self.connection:
                await self.connection.close()
                logger.info("[DB] Database connection closed")
        except Exception as e:
            logger.error(f"[DB ERROR] Error closing database: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of DatabaseManager"""
        return f"<DatabaseManager(db_path={self.db_path})>"
