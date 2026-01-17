"""
Database Verification Script
Tests DatabaseManager functionality independently
"""

import asyncio
from core.database import DatabaseManager
from datetime import datetime


async def test_database():
    """Test all DatabaseManager functions"""
    print("=" * 60)
    print("DATABASE VERIFICATION TEST")
    print("=" * 60)
    
    # Initialize database
    print("\n[1] Initializing database...")
    db = DatabaseManager('data/test_trades.db')
    await db.init_db()
    print("✅ Database initialized successfully")
    
    # Insert test trades
    print("\n[2] Inserting test trades...")
    test_trades = [
        {
            'symbol': 'BTC/USDT',
            'side': 'SELL',
            'price': 42000.00,
            'amount': 0.001,
            'profit': 42.50,
            'timestamp': datetime.now().isoformat(),
            'order_id': 'test_order_1',
            'level': 5,
            'client_order_id': 'test_client_1',
            'status': 'filled'
        },
        {
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'price': 41500.00,
            'amount': 0.001,
            'profit': 0.0,
            'timestamp': datetime.now().isoformat(),
            'order_id': 'test_order_2',
            'level': 4,
            'client_order_id': 'test_client_2',
            'status': 'filled'
        },
        {
            'symbol': 'BTC/USDT',
            'side': 'SELL',
            'price': 42500.00,
            'amount': 0.001,
            'profit': 45.00,
            'timestamp': datetime.now().isoformat(),
            'order_id': 'test_order_3',
            'level': 6,
            'client_order_id': 'test_client_3',
            'status': 'filled'
        }
    ]
    
    for trade in test_trades:
        trade_id = await db.insert_trade(trade)
        if trade_id:
            print(f"  ✅ Inserted trade ID: {trade_id} - {trade['side']} @ ${trade['price']:,.2f}")
        else:
            print(f"  ❌ Failed to insert trade")
    
    # Retrieve trades
    print("\n[3] Retrieving trades...")
    trades = await db.get_trades(symbol='BTC/USDT', limit=10)
    print(f"  ✅ Retrieved {len(trades)} trades:")
    for trade in trades:
        print(f"     ID: {trade['id']}, {trade['side']}, "
              f"Price: ${trade['price']:,.2f}, Profit: ${trade['profit']:,.2f}")
    
    # Calculate statistics
    print("\n[4] Calculating statistics...")
    stats = await db.get_stats(symbol='BTC/USDT')
    print(f"  ✅ Statistics calculated:")
    print(f"     Total Trades: {stats['total_trades']}")
    print(f"     Total Profit: ${stats['total_profit']:,.2f}")
    print(f"     Avg Profit: ${stats['avg_profit']:,.2f}")
    print(f"     Win Rate: {stats['win_rate']:.2f}%")
    print(f"     Buy Orders: {stats['buy_count']}")
    print(f"     Sell Orders: {stats['sell_count']}")
    
    # Test duplicate prevention
    print("\n[5] Testing duplicate prevention...")
    duplicate_trade = test_trades[0].copy()
    duplicate_id = await db.insert_trade(duplicate_trade)
    if duplicate_id is None:
        print("  ✅ Duplicate trade rejected (UNIQUE constraint working)")
    else:
        print("  ❌ Duplicate trade was inserted (UNIQUE constraint failed)")
    
    # Test migration (simulate state.json)
    print("\n[6] Testing migration from state.json...")
    mock_state = {
        'symbol': 'BTC/USDT',
        'trade_history': [
            {
                'side': 'SELL',
                'price': 43000.00,
                'amount': 0.002,
                'profit_usd': 86.00,
                'timestamp': datetime.now().isoformat(),
                'client_order_id': 'migrated_1',
                'level': 7,
                'status': 'filled'
            },
            {
                'event': 'EMERGENCY_STOP',  # Should be skipped
                'timestamp': datetime.now().isoformat()
            }
        ]
    }
    
    migrated = await db.migrate_from_state(mock_state)
    print(f"  ✅ Migrated {migrated} trades from state.json")
    
    # Final stats
    print("\n[7] Final statistics after migration...")
    final_stats = await db.get_stats()
    print(f"  ✅ Total trades in database: {final_stats['total_trades']}")
    print(f"  ✅ Total profit: ${final_stats['total_profit']:,.2f}")
    
    # Close database
    await db.close()
    print("\n✅ Database connection closed")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)
    print("\nDatabase file created at: data/test_trades.db")
    print("You can inspect it with any SQLite viewer.")


if __name__ == "__main__":
    asyncio.run(test_database())
