#!/usr/bin/env python3
"""
Connect Trading Buddy to existing IB backtest system data.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from trading_buddy.core.duck import DuckDBManager, save_column_mapping


def explore_ib_database():
    """Explore the IB backtest system database structure."""
    # Path to the IB backtest system database
    ib_db_path = "/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb"
    
    if not Path(ib_db_path).exists():
        print(f"Error: Database not found at {ib_db_path}")
        return
    
    print(f"Connecting to IB database: {ib_db_path}")
    
    conn = duckdb.connect(ib_db_path, read_only=True)
    
    # List all tables
    tables = conn.execute("""
        SELECT table_name, table_type
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()
    
    print("\nTables found:")
    for table, type_ in tables:
        print(f"  - {table} ({type_})")
    
    # Look for OHLCV-like tables
    print("\nExploring table structures...")
    
    for table_name, _ in tables:
        # Get columns
        columns = conn.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            AND table_schema = 'main'
            ORDER BY ordinal_position
        """).fetchall()
        
        col_names = [col[0].lower() for col in columns]
        
        # Check if this looks like an OHLCV table
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if any(col in col_names for col in ohlcv_cols):
            print(f"\n{table_name} columns:")
            for col_name, col_type in columns:
                print(f"    - {col_name} ({col_type})")
            
            # Show sample data
            try:
                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
                if sample:
                    print(f"  Sample data:")
                    for row in sample:
                        print(f"    {row}")
            except:
                pass
    
    conn.close()
    return ib_db_path


def setup_trading_buddy_connection():
    """Set up Trading Buddy to use the IB database."""
    # Update .env file
    env_content = f"""# DuckDB Configuration
DUCKDB_PATH=/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb

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
    
    print("\nCreated .env file pointing to IB database")
    
    # Now test the connection with Trading Buddy
    print("\nTesting Trading Buddy connection...")
    
    with DuckDBManager() as db:
        # Check if we can query the database
        try:
            tables = db.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                LIMIT 5
            """).fetchall()
            
            print(f"Successfully connected! Found {len(tables)} tables")
            
            # Try to find OHLCV data
            # Common table names to try
            possible_tables = ['bars', 'ohlcv', 'market_data', 'price_data', 'candles']
            
            for table in possible_tables:
                try:
                    count = db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    print(f"  - Table '{table}' has {count} rows")
                    
                    # Get column info
                    cols = db.conn.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                    """).fetchall()
                    
                    print(f"    Columns: {', '.join([c[0] for c in cols])}")
                    
                except:
                    continue
                    
        except Exception as e:
            print(f"Error connecting: {e}")


def main():
    print("Connecting Trading Buddy to IB Backtest System data...\n")
    
    # First explore the IB database
    ib_db_path = explore_ib_database()
    
    if ib_db_path:
        # Set up the connection
        setup_trading_buddy_connection()
        
        print("\n" + "="*60)
        print("Setup complete!")
        print("\nNext steps:")
        print("1. Run: make init-db")
        print("2. The system will auto-discover the OHLCV table schema")
        print("3. Run: make backfill")
        print("4. Run: make ci")
        print("5. Run: make run")


if __name__ == "__main__":
    main()