#!/usr/bin/env python3
"""
Backfill historical events and contexts.
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.detectors.contexts import (
    detect_large_drop,
    detect_regime,
    detect_trend_context,
)
from trading_buddy.detectors.outcomes import update_event_outcomes
from trading_buddy.detectors.patterns import (
    detect_compound_pattern,
    detect_double_bottom,
    detect_macd_bull_cross,
)


def backfill_symbol(conn, symbol: str, start_date: str, end_date: str):
    """Backfill events for a single symbol."""
    print(f"\nProcessing {symbol}...")
    
    # Detect 5m patterns
    print(f"  - Detecting double bottoms...")
    double_bottoms = detect_double_bottom(
        conn, symbol, "5m",
        start_ts=start_date,
        end_ts=end_date
    )
    
    for event in double_bottoms:
        conn.execute("""
            INSERT OR IGNORE INTO events 
            (symbol, timeframe, event_ts, pattern, attrs)
            VALUES (?, ?, ?, ?, ?)
        """, [symbol, "5m", event["event_ts"], event["pattern"], event["attrs"]])
    
    print(f"    Found {len(double_bottoms)} double bottoms")
    
    # MACD crosses
    print(f"  - Detecting MACD crosses...")
    macd_crosses = detect_macd_bull_cross(
        conn, symbol, "5m",
        start_ts=start_date,
        end_ts=end_date
    )
    
    for event in macd_crosses:
        conn.execute("""
            INSERT OR IGNORE INTO events 
            (symbol, timeframe, event_ts, pattern, attrs)
            VALUES (?, ?, ?, ?, ?)
        """, [symbol, "5m", event["event_ts"], event["pattern"], event["attrs"]])
    
    print(f"    Found {len(macd_crosses)} MACD crosses")
    
    # Compound patterns
    print(f"  - Detecting compound patterns...")
    compounds = detect_compound_pattern(
        conn, symbol, "5m",
        start_ts=start_date,
        end_ts=end_date
    )
    
    for event in compounds:
        conn.execute("""
            INSERT OR IGNORE INTO events 
            (symbol, timeframe, event_ts, pattern, attrs)
            VALUES (?, ?, ?, ?, ?)
        """, [symbol, "5m", event["event_ts"], event["pattern"], event["attrs"]])
    
    print(f"    Found {len(compounds)} compound patterns")
    
    # Detect 15m contexts
    print(f"  - Detecting 15m trends...")
    trends = detect_trend_context(
        conn, symbol, "15m",
        start_ts=start_date,
        end_ts=end_date
    )
    
    for ctx in trends:
        conn.execute("""
            INSERT OR IGNORE INTO contexts
            (symbol, timeframe, event_ts, context, attrs)
            VALUES (?, ?, ?, ?, ?)
        """, [symbol, "15m", ctx["event_ts"], ctx["context"], ctx["attrs"]])
    
    print(f"    Found {len(trends)} trend contexts")
    
    # Detect 60m regimes
    print(f"  - Detecting 60m regimes...")
    regimes = detect_regime(
        conn, symbol, "60m",
        start_ts=start_date,
        end_ts=end_date
    )
    
    for ctx in regimes:
        conn.execute("""
            INSERT OR IGNORE INTO contexts
            (symbol, timeframe, event_ts, context, attrs)
            VALUES (?, ?, ?, ?, ?)
        """, [symbol, "60m", ctx["event_ts"], ctx["context"], ctx["attrs"]])
    
    print(f"    Found {len(regimes)} regime contexts")
    
    conn.commit()


def update_outcomes(conn):
    """Update outcomes for all events without forward returns."""
    print("\nUpdating outcomes...")
    
    events = conn.execute("""
        SELECT symbol, timeframe, event_ts, pattern
        FROM events
        WHERE fwd_ret IS NULL
        ORDER BY event_ts
    """).fetchall()
    
    event_dicts = [
        {
            "symbol": e[0],
            "timeframe": e[1],
            "event_ts": e[2],
            "pattern": e[3],
        }
        for e in events
    ]
    
    updated = update_event_outcomes(conn, event_dicts)
    print(f"  Updated {updated}/{len(event_dicts)} events with outcomes")
    
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Backfill historical events")
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,NVDA",
        help="Comma-separated list of symbols"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to backfill"
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Calculate date range
    if args.to_date:
        end_date = datetime.fromisoformat(args.to_date)
    else:
        end_date = datetime.now()
    
    if args.from_date:
        start_date = datetime.fromisoformat(args.from_date)
    else:
        start_date = end_date - timedelta(days=args.days)
    
    print(f"Backfilling events from {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {', '.join(symbols)}")
    
    with DuckDBManager() as db:
        # Attach IB database if needed
        try:
            db.conn.execute("""
                ATTACH DATABASE '/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb' 
                AS ib_data (READ_ONLY)
            """)
        except:
            pass  # Already attached
            
        # Check if we have data
        sample = db.conn.execute("""
            SELECT COUNT(*) FROM bars LIMIT 1
        """).fetchone()
        
        if not sample or sample[0] == 0:
            print("\nError: No data found in bars table!")
            print("Please load OHLCV data before running backfill.")
            return
        
        # Backfill each symbol
        for symbol in symbols:
            backfill_symbol(
                db.conn,
                symbol,
                start_date.isoformat(),
                end_date.isoformat()
            )
        
        # Update outcomes
        update_outcomes(db.conn)
        
        # Show summary
        total_events = db.conn.execute("""
            SELECT COUNT(*) FROM events
        """).fetchone()[0]
        
        total_contexts = db.conn.execute("""
            SELECT COUNT(*) FROM contexts
        """).fetchone()[0]
        
        print(f"\nBackfill complete!")
        print(f"  Total events: {total_events}")
        print(f"  Total contexts: {total_contexts}")


if __name__ == "__main__":
    main()