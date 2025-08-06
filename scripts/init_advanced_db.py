#!/usr/bin/env python3
"""
Initialize database with advanced schema including OOS splits
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.core.ddl import initialize_database


def main():
    print("Initializing Advanced Database Schema")
    print("=" * 80)
    
    with DuckDBManager() as db:
        # Initialize all tables
        initialize_database(db.conn)
        print("✓ Created core tables")
        
        # Check if events table has oos_split column
        result = db.conn.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'events' 
        AND column_name = 'oos_split'
        """).fetchone()
        
        if result:
            print("✓ Events table has oos_split column")
        else:
            # Add column if missing
            try:
                db.conn.execute("""
                ALTER TABLE events 
                ADD COLUMN oos_split TEXT DEFAULT 'train'
                """)
                print("✓ Added oos_split column to events table")
            except:
                print("⚠️  Could not add oos_split column (may already exist)")
        
        # Insert some sample events with OOS splits for testing
        print("\nInserting sample events...")
        
        # Sample events
        sample_events = [
            ("SPY", "5min", "2025-08-05 10:50:00", "double_bottom", 0.0025, "test"),
            ("SPY", "5min", "2025-08-05 11:40:00", "w_pattern", 0.0031, "test"),
            ("SPY", "5min", "2025-08-05 14:30:00", "macd_bull_cross", -0.0012, "test"),
            ("SPY", "15min", "2025-08-05 10:45:00", "double_bottom", 0.0042, "test"),
            ("SPY", "5min", "2025-08-04 09:45:00", "double_bottom", 0.0018, "train"),
            ("SPY", "5min", "2025-08-04 11:15:00", "w_pattern", 0.0022, "train"),
        ]
        
        for symbol, tf, ts, pattern, ret, split in sample_events:
            try:
                db.conn.execute("""
                INSERT INTO events (symbol, timeframe, event_ts, pattern, fwd_ret, oos_split, lead_bars)
                VALUES (?, ?, ?, ?, ?, ?, 10)
                ON CONFLICT DO NOTHING
                """, [symbol, tf, ts, pattern, ret, split])
            except Exception as e:
                print(f"  Warning: {e}")
        
        # Add some consistency matrix data
        print("\nInserting consistency matrix data...")
        
        cm_data = [
            ("5min", "double_bottom", "15min", "trend_up", 50, 0.002, 0.65, 0.8, 0.7),
            ("5min", "w_pattern", "15min", "trend_up", 45, 0.0015, 0.62, 0.75, 0.65),
            ("5min", "double_bottom", "60min", "trend_up", 30, 0.003, 0.70, 0.85, 0.75),
        ]
        
        for row in cm_data:
            try:
                db.conn.execute("""
                INSERT INTO consistency_matrix 
                (base_tf, base_pattern, ctx_tf, ctx_pattern, n, exp_lift, hit_rate, stability, ci)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """, row)
            except Exception as e:
                print(f"  Warning: {e}")
        
        # Verify
        event_count = db.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        cm_count = db.conn.execute("SELECT COUNT(*) FROM consistency_matrix").fetchone()[0]
        
        print(f"\n✓ Database initialized:")
        print(f"  - Events: {event_count} records")
        print(f"  - Consistency Matrix: {cm_count} records")
        print(f"  - Reports Daily table ready")
        
        # Show schema
        print("\nTables created:")
        tables = db.conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_name
        """).fetchall()
        
        for table in tables:
            print(f"  - {table[0]}")


if __name__ == "__main__":
    main()