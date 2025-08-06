#!/usr/bin/env python3
"""
Initialize database and auto-discover schema.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.ddl import initialize_database
from trading_buddy.core.duck import DuckDBManager, initialize_schema_discovery


def main():
    print("Initializing Trading Buddy database...")
    
    with DuckDBManager() as db:
        # Create tables
        print("Creating tables...")
        initialize_database(db.conn)
        
        # Auto-discover schema
        print("Auto-discovering schema...")
        if initialize_schema_discovery(db.conn):
            print("✓ Schema discovery successful")
            
            # List discovered views
            views = db.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                AND table_name LIKE 'bars%'
                ORDER BY table_name
            """).fetchall()
            
            print(f"\nDiscovered views:")
            for view in views:
                print(f"  - {view[0]}")
        else:
            print("✗ Schema discovery failed - no OHLCV table found")
            print("\nPlease ensure your DuckDB database contains a table with:")
            print("  - timestamp/time column")
            print("  - symbol/ticker column")
            print("  - open, high, low, close columns")
            print("  - volume column")
            print("  - timeframe column (optional)")
        
        # Show all tables
        tables = db.conn.execute("""
            SELECT table_name, table_type
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            ORDER BY table_type, table_name
        """).fetchall()
        
        print(f"\nAll database objects:")
        for table, type_ in tables:
            print(f"  - {table} ({type_})")
        
        db.conn.commit()
        print("\nDatabase initialization complete!")


if __name__ == "__main__":
    main()