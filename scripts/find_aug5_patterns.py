#!/usr/bin/env python3
"""
Find patterns in August 5 SPY data using enhanced detectors
"""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.hybrid_data_manager import HybridDataManager
from trading_buddy.detectors.patterns_realtime import (
    detect_double_bottom_flexible,
    detect_intraday_double_bottom
)


def main():
    with HybridDataManager() as db:
        print("Searching for double bottom patterns on August 5, 2025...")
        print("=" * 80)
        
        # Method 1: Flexible double bottom with various tolerances
        tolerances = [50, 100, 150, 200, 250, 300]
        
        for tolerance in tolerances:
            print(f"\nTolerance: {tolerance} bps ({tolerance/100:.1f}%)")
            patterns = detect_double_bottom_flexible(
                db.conn,
                'SPY',
                '5min',
                lookback_hours=24,
                tolerance_bps=tolerance,
                min_bars_between=5,
                max_bars_between=200,
                start_ts='2025-08-05 00:00:00',
                end_ts='2025-08-05 23:59:59'
            )
            
            if patterns:
                print(f"  Found {len(patterns)} patterns:")
                for p in patterns:
                    attrs = p['attrs']
                    print(f"    - Bottom 1: {attrs['low1_ts']} @ ${attrs['low1_price']:.2f}")
                    print(f"      Bottom 2: {attrs['low2_ts']} @ ${attrs['low2_price']:.2f}")
                    print(f"      Neckline: ${attrs['neckline_price']:.2f}")
                    print(f"      Distance: {attrs['bar_distance']} bars")
                    print(f"      Similarity: {attrs['price_similarity_pct']:.1f}%")
                    print()
            else:
                print("  No patterns found")
        
        # Method 2: Intraday specific search
        print("\n" + "=" * 80)
        print("Intraday double bottom search (focused on lowest points):")
        
        intraday_patterns = detect_intraday_double_bottom(
            db.conn,
            'SPY',
            '2025-08-05',
            tolerance_bps=150
        )
        
        if intraday_patterns:
            print(f"\nFound {len(intraday_patterns)} intraday patterns:")
            for p in intraday_patterns:
                attrs = p['attrs']
                print(f"\n  Pattern at {p['event_ts']}:")
                print(f"    Bottom 1: {attrs['low1_ts']} @ ${attrs['low1_price']:.2f}")
                print(f"    Bottom 2: {attrs['low2_ts']} @ ${attrs['low2_price']:.2f}")
                print(f"    Price diff: {attrs['price_diff_pct']:.2f}%")
                print(f"    Bars between: {attrs['bars_between']}")
                print(f"    Bounce after: ${attrs['bounce_height']:.2f}")
        
        # Method 3: Manual inspection of specific time periods
        print("\n" + "=" * 80)
        print("Manual inspection of key lows:")
        
        # Get the two lowest points that are far apart
        result = db.conn.execute("""
        WITH lows AS (
            SELECT 
                ts,
                low,
                high,
                close,
                ROW_NUMBER() OVER (ORDER BY low) as rank,
                ROW_NUMBER() OVER (ORDER BY ts) as bar_num
            FROM bars
            WHERE symbol = 'SPY' 
            AND timeframe = '5min'
            AND DATE(ts) = '2025-08-05'
        ),
        key_lows AS (
            SELECT * FROM lows WHERE rank <= 20
        )
        SELECT 
            l1.ts as ts1, l1.low as low1, l1.bar_num as bar1,
            l2.ts as ts2, l2.low as low2, l2.bar_num as bar2,
            ABS(l2.low - l1.low) / l1.low * 100 as price_diff_pct,
            l2.bar_num - l1.bar_num as bars_apart
        FROM key_lows l1
        JOIN key_lows l2 ON l1.ts < l2.ts
        WHERE l2.bar_num - l1.bar_num >= 10  -- At least 50 minutes apart
        AND ABS(l2.low - l1.low) / l1.low <= 0.02  -- Within 2%
        ORDER BY price_diff_pct
        LIMIT 10
        """).fetchall()
        
        if result:
            print(f"\nPotential double bottom candidates (within 2% price):")
            for ts1, low1, bar1, ts2, low2, bar2, diff_pct, bars in result:
                print(f"\n  Candidate:")
                print(f"    Bottom 1: {ts1} @ ${low1:.2f} (bar {bar1})")
                print(f"    Bottom 2: {ts2} @ ${low2:.2f} (bar {bar2})")
                print(f"    Price difference: {diff_pct:.2f}%")
                print(f"    Bars apart: {bars} ({bars*5} minutes)")
                
                # Check for bounce after second bottom
                bounce = db.conn.execute(f"""
                SELECT MAX(high) - {low2} as bounce
                FROM bars
                WHERE symbol = 'SPY'
                AND timeframe = '5min'
                AND ts > '{ts2}'
                AND DATE(ts) = '2025-08-05'
                """).fetchone()
                
                if bounce and bounce[0]:
                    print(f"    Bounce after: ${bounce[0]:.2f}")


if __name__ == "__main__":
    main()