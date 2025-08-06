#!/usr/bin/env python3
"""
Check for latest patterns in available data
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.hybrid_data_manager import HybridDataManager
from trading_buddy.detectors.patterns import detect_swing_points


def main():
    print("Checking latest available data and patterns...")
    print("=" * 60)
    
    with HybridDataManager() as db:
        # Check data coverage for all symbols
        symbols = ['SPY', 'QQQ', 'NVDA']
        
        for symbol in symbols:
            print(f"\n{symbol} Data Coverage:")
            for tf in ['5min', '15min', '60min']:
                freshness = db.check_data_freshness(symbol, tf)
                if freshness['latest']:
                    print(f"  {tf}: up to {freshness['latest']} ({freshness['gap_minutes']:.0f} min ago)")
        
        # Look for patterns in the most recent available data
        print("\n" + "=" * 60)
        print("Searching for patterns in recent data...")
        
        # Get latest SPY data date
        result = db.conn.execute("""
            SELECT MAX(ts) FROM bars 
            WHERE symbol = 'SPY' AND timeframe = '5min'
        """).fetchone()
        
        if result and result[0]:
            latest_date = result[0]
            start_date = latest_date - timedelta(days=3)
            
            print(f"\nAnalyzing SPY from {start_date} to {latest_date}")
            
            # Get price action for context
            price_data = db.conn.execute("""
                SELECT ts, open, high, low, close, volume
                FROM bars
                WHERE symbol = 'SPY' 
                AND timeframe = '5min'
                AND ts >= ?
                ORDER BY ts DESC
                LIMIT 20
            """, [start_date]).fetchall()
            
            print("\nRecent SPY 5min bars:")
            for row in price_data[:10]:
                ts, o, h, l, c, v = row
                print(f"  {ts}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} V={v:,}")
            
            # Check for swing points
            swings = detect_swing_points(
                db.conn, "SPY", "5min",
                neighborhood=5,  # Use larger neighborhood
                start_ts=str(start_date),
                end_ts=str(latest_date)
            )
            
            print(f"\nSwing Points Found:")
            print(f"  Swing Lows: {len(swings['lows'])}")
            if swings['lows']:
                for ts, price in swings['lows'][-3:]:
                    print(f"    {ts}: ${price:.2f}")
            
            print(f"  Swing Highs: {len(swings['highs'])}")
            if swings['highs']:
                for ts, price in swings['highs'][-3:]:
                    print(f"    {ts}: ${price:.2f}")
        
        # Check existing patterns in database
        print("\n" + "=" * 60)
        print("Existing patterns in database:")
        
        patterns = db.get_latest_patterns(hours=72)
        
        pattern_counts = {}
        for p in patterns:
            key = f"{p['symbol']} {p['pattern']}"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        
        for pattern, count in sorted(pattern_counts.items()):
            print(f"  {pattern}: {count}")
        
        # Show most recent patterns
        if patterns:
            print("\nMost recent patterns:")
            for p in patterns[:5]:
                print(f"  {p['symbol']} {p['pattern']} at {p['event_ts']}")
                if p['fwd_ret'] is not None:
                    print(f"    Forward return: {p['fwd_ret']:.2%} ({p['label']})")


if __name__ == "__main__":
    main()