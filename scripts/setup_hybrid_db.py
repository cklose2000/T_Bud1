#!/usr/bin/env python3
"""
Set up a hybrid database approach:
- Read market data from IB database
- Store Trading Buddy specific data in local database
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from trading_buddy.core.ddl import initialize_database


def setup_hybrid_database():
    """Create a hybrid setup with views to IB data."""
    
    # Path to IB database
    ib_db_path = "/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb"
    
    # Create local Trading Buddy database
    local_db_path = Path("data/trading_buddy.duckdb")
    local_db_path.parent.mkdir(exist_ok=True)
    
    print(f"Creating local Trading Buddy database: {local_db_path}")
    
    conn = duckdb.connect(str(local_db_path))
    
    # Attach the IB database as read-only
    print(f"Attaching IB database (read-only): {ib_db_path}")
    conn.execute(f"""
        ATTACH DATABASE '{ib_db_path}' AS ib_data (READ_ONLY)
    """)
    
    # Create Trading Buddy tables
    print("Creating Trading Buddy tables...")
    initialize_database(conn)
    
    # Create views that map to IB data
    print("Creating views to IB market data...")
    
    # Main bars view
    conn.execute("""
        CREATE OR REPLACE VIEW bars AS
        SELECT 
            symbol,
            timeframe,
            date as ts,
            open,
            high,
            low,
            close,
            volume
        FROM ib_data.market_data
    """)
    
    # Create timeframe-specific views
    timeframes = ["1min", "5min", "15min", "30min", "1hour", "4hour", "1day"]
    
    for tf in timeframes:
        view_name = f"bars_{tf.replace('min', 'm').replace('hour', 'h').replace('day', 'd')}"
        
        conn.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT 
                symbol,
                timeframe,
                date as ts,
                open,
                high,
                low,
                close,
                volume
            FROM ib_data.market_data
            WHERE timeframe = '{tf}'
        """)
        
        print(f"  Created view: {view_name}")
    
    # Create meta table if not exists and save column mapping
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    conn.execute("""
        INSERT OR REPLACE INTO meta (key, value)
        VALUES ('column_mapping', ?)
    """, ['{"timestamp": "date", "symbol": "symbol", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "timeframe": "timeframe"}'])
    
    # Test the views
    print("\nTesting views...")
    
    # Count records
    total = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    print(f"  Total bars: {total:,}")
    
    # Sample data
    sample = conn.execute("""
        SELECT symbol, timeframe, ts, close, volume
        FROM bars
        WHERE timeframe = '5min'
        ORDER BY ts DESC
        LIMIT 5
    """).fetchall()
    
    print("\n  Recent 5min bars:")
    for row in sample:
        print(f"    {row[0]} {row[1]} {row[2]} close={row[3]:.2f} vol={row[4]:,}")
    
    # Check available symbols
    symbols = conn.execute("""
        SELECT DISTINCT symbol 
        FROM bars 
        ORDER BY symbol
    """).fetchall()
    
    print(f"\n  Available symbols: {', '.join([s[0] for s in symbols])}")
    
    # Check available timeframes
    timeframes = conn.execute("""
        SELECT DISTINCT timeframe, COUNT(*) as cnt
        FROM bars 
        GROUP BY timeframe
        ORDER BY timeframe
    """).fetchall()
    
    print("\n  Available timeframes:")
    for tf, cnt in timeframes:
        print(f"    {tf}: {cnt:,} bars")
    
    conn.commit()
    conn.close()
    
    # Update .env to point to local database
    env_content = f"""# DuckDB Configuration
DUCKDB_PATH=./data/trading_buddy.duckdb

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_RELOAD=true

# Logging
LOG_LEVEL=INFO

# Trading Configuration
DEFAULT_LOOKBACK_DAYS=365
DEFAULT_HORIZON_BARS=10
DEFAULT_RISK_BUDGET=0.01

# Alert Configuration
ALERT_CHECK_INTERVAL_SECONDS=60
PRECURSOR_PROBABILITY_THRESHOLD=0.7
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("\n✅ Hybrid database setup complete!")
    print("\nThe Trading Buddy database now has:")
    print("  - Read-only views to IB market data")
    print("  - Local tables for events, contexts, and CI matrix")
    print("\nNext steps:")
    print("  1. Run: make backfill")
    print("  2. Run: make ci")
    print("  3. Run: make run")


if __name__ == "__main__":
    setup_hybrid_database()