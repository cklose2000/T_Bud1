#!/usr/bin/env python3
"""
Analyze the specific W pattern on August 5 that the user mentioned
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.hybrid_data_manager import HybridDataManager
from trading_buddy.detectors.w_pattern_detector import detect_w_pattern, analyze_aug5_w_pattern


def main():
    with HybridDataManager() as db:
        print("Analyzing August 5, 2025 W Pattern (10:50 -> 11:45)")
        print("=" * 80)
        
        # Get the specific bars mentioned
        result = db.conn.execute("""
        SELECT ts, open, high, low, close, volume
        FROM bars
        WHERE symbol = 'SPY' 
        AND timeframe = '5min'
        AND ts IN ('2025-08-05 10:50:00', '2025-08-05 11:45:00')
        ORDER BY ts
        """).fetchall()
        
        if len(result) == 2:
            print("\nKey Points:")
            print(f"First bottom:  {result[0][0]} - Low: ${result[0][3]:.2f} (O:{result[0][1]:.2f} H:{result[0][2]:.2f} C:{result[0][4]:.2f})")
            print(f"Second bottom: {result[1][0]} - Low: ${result[1][3]:.2f} (O:{result[1][1]:.2f} H:{result[1][2]:.2f} C:{result[1][4]:.2f})")
            print(f"Difference: ${result[1][3] - result[0][3]:.2f} ({(result[1][3] - result[0][3])/result[0][3]*100:.2f}% higher)")
        
        # Analyze MACD during this period
        print("\n" + "-" * 80)
        print("MACD Analysis during W pattern formation:")
        
        macd_result = db.conn.execute("""
        WITH macd_calc AS (
            SELECT 
                ts,
                close,
                AVG(close) OVER (ORDER BY ts ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as sma12,
                AVG(close) OVER (ORDER BY ts ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) as sma26
            FROM bars
            WHERE symbol = 'SPY'
            AND timeframe = '5min'
            AND ts >= '2025-08-05 10:30:00'
            AND ts <= '2025-08-05 12:00:00'
        )
        SELECT 
            ts,
            close,
            (sma12 - sma26) as macd,
            LAG(sma12 - sma26, 1) OVER (ORDER BY ts) as prev_macd
        FROM macd_calc
        WHERE ts IN ('2025-08-05 10:50:00', '2025-08-05 11:45:00')
        ORDER BY ts
        """).fetchall()
        
        if macd_result:
            for ts, close, macd, prev_macd in macd_result:
                print(f"{ts} - MACD: {macd:.4f}, Price: ${close:.2f}")
                if prev_macd and macd > prev_macd:
                    print(f"  → MACD improving (less negative)")
        
        # Run W pattern detection
        print("\n" + "-" * 80)
        print("Running W Pattern Detection:")
        
        patterns = detect_w_pattern(
            db.conn,
            'SPY',
            '5min',
            lookback_hours=3,
            second_bottom_tolerance_pct=0.5,
            min_bars_between=8,
            max_bars_between=20,
            start_ts='2025-08-05 10:00:00',
            end_ts='2025-08-05 12:30:00'
        )
        
        if patterns:
            print(f"\nFound {len(patterns)} W patterns:")
            for p in patterns:
                attrs = p['attrs']
                print(f"\nPattern at {p['event_ts']}:")
                print(f"  Bottom 1: {attrs['low1_ts']} @ ${attrs['low1_price']:.2f}")
                print(f"  Bottom 2: {attrs['low2_ts']} @ ${attrs['low2_price']:.2f} ({attrs['second_bottom_higher_pct']:.2f}% higher)")
                print(f"  Neckline: ${attrs['neckline_price']:.2f}")
                print(f"  W Depth: {attrs['w_depth_pct']:.1f}%")
                print(f"  MACD Divergence: {attrs['macd_divergence']}")
                if attrs['macd1'] and attrs['macd2']:
                    print(f"  MACD values: {attrs['macd1']:.4f} → {attrs['macd2']:.4f}")
        
        # Get detailed analysis
        print("\n" + "-" * 80)
        print("Detailed W Pattern Analysis:")
        
        analysis = analyze_aug5_w_pattern(db.conn)
        if analysis:
            pattern = analysis['pattern']
            details = analysis['details']
            attrs = pattern['attrs']
            
            print(f"\nTHE W PATTERN (10:50 → 11:45):")
            print(f"  First Bottom:  {attrs['low1_ts']} @ ${attrs['low1_price']:.2f}")
            print(f"  Second Bottom: {attrs['low2_ts']} @ ${attrs['low2_price']:.2f}")
            print(f"  Second bottom is {attrs['second_bottom_higher_pct']:.2f}% higher")
            print(f"\n  Pattern Stats:")
            print(f"    Bars in pattern: {details['bars_in_pattern']}")
            print(f"    Pattern range: ${details['pattern_range']:.2f}")
            print(f"    Total volume: {details['total_volume']:,}")
            print(f"    Exit price: ${details['exit_price']:.2f}")
            
            # Show what happened after the pattern
            print("\n  Post-Pattern Performance:")
            post_pattern = db.conn.execute(f"""
            SELECT 
                MIN(ts) as first_bar,
                MAX(ts) as last_bar,
                MAX(high) as max_high,
                MIN(low) as min_low,
                LAST(close ORDER BY ts) as final_close
            FROM bars
            WHERE symbol = 'SPY'
            AND timeframe = '5min'
            AND ts > '{attrs["low2_ts"]}'
            AND ts <= '2025-08-05 16:00:00'
            """).fetchone()
            
            if post_pattern:
                gain = post_pattern[2] - attrs['low2_price']
                gain_pct = gain / attrs['low2_price'] * 100
                print(f"    Maximum gain: ${gain:.2f} ({gain_pct:.2f}%)")
                print(f"    High reached: ${post_pattern[2]:.2f}")


if __name__ == "__main__":
    main()