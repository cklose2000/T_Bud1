#!/usr/bin/env python3
"""
Analyze August 5 SPY data to find patterns
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.hybrid_data_manager import HybridDataManager
from datetime import datetime


def main():
    with HybridDataManager() as db:
        # Get August 5 SPY data
        result = db.conn.execute("""
        SELECT 
            ts,
            open,
            high,
            low,
            close,
            volume
        FROM bars
        WHERE symbol = 'SPY' 
        AND timeframe = '5min'
        AND DATE(ts) = '2025-08-05'
        ORDER BY ts
        """).fetchall()
        
        print(f'SPY 5min bars for August 5: {len(result)}')
        
        if result:
            # Extract data
            timestamps = [r[0] for r in result]
            lows = [r[3] for r in result]
            highs = [r[2] for r in result]
            closes = [r[4] for r in result]
            
            # Find swing lows
            swing_lows = []
            for i in range(5, len(lows) - 5):
                # Check if this is a local minimum
                if all(lows[i] <= lows[j] for j in range(i-5, i+6) if j != i):
                    swing_lows.append((i, timestamps[i], lows[i]))
            
            print(f'\nSwing lows detected: {len(swing_lows)}')
            for idx, ts, low in swing_lows:
                print(f'  {ts}: ${low:.2f}')
            
            # Find lowest points
            lows_with_idx = [(i, timestamps[i], lows[i]) for i in range(len(lows))]
            lows_sorted = sorted(lows_with_idx, key=lambda x: x[2])
            
            print(f'\nLowest 5 points on August 5:')
            for i, (idx, ts, low) in enumerate(lows_sorted[:5]):
                print(f'{i+1}. {ts}: ${low:.2f} (bar #{idx})')
            
            # Look for double bottom pattern
            print('\nChecking for double bottom patterns...')
            
            # Method 1: Look for two lows within 2% of each other
            min_low = min(lows)
            potential_bottoms = []
            
            for i, low in enumerate(lows):
                if low <= min_low * 1.02:  # Within 2% of lowest
                    potential_bottoms.append((i, timestamps[i], low))
            
            print(f'\nBars within 2% of lowest ({min_low:.2f}):')
            for idx, ts, low in potential_bottoms:
                print(f'  {ts}: ${low:.2f}')
            
            # Check spacing between potential bottoms
            if len(potential_bottoms) >= 2:
                # Group nearby bottoms
                bottom_groups = []
                current_group = [potential_bottoms[0]]
                
                for i in range(1, len(potential_bottoms)):
                    # If bars are within 20 bars of each other, same group
                    if potential_bottoms[i][0] - current_group[-1][0] <= 20:
                        current_group.append(potential_bottoms[i])
                    else:
                        bottom_groups.append(current_group)
                        current_group = [potential_bottoms[i]]
                
                bottom_groups.append(current_group)
                
                print(f'\nBottom groups found: {len(bottom_groups)}')
                for i, group in enumerate(bottom_groups):
                    print(f'\nGroup {i+1}:')
                    for idx, ts, low in group:
                        print(f'  {ts}: ${low:.2f}')
                
                # Check if we have a double bottom (2 distinct groups)
                if len(bottom_groups) >= 2:
                    print('\n*** POTENTIAL DOUBLE BOTTOM DETECTED ***')
                    print(f'First bottom: {bottom_groups[0][0][1]} at ${bottom_groups[0][0][2]:.2f}')
                    print(f'Second bottom: {bottom_groups[-1][0][1]} at ${bottom_groups[-1][0][2]:.2f}')
            
            # Show price action context
            print('\n\nPrice action summary:')
            print(f'Open: ${result[0][1]:.2f}')
            print(f'High: ${max(highs):.2f}')
            print(f'Low: ${min(lows):.2f}')
            print(f'Close: ${result[-1][4]:.2f}')
            print(f'Range: ${max(highs) - min(lows):.2f}')
            
            # Look at specific time periods
            print('\n\nPrice by hour:')
            hourly_data = {}
            for ts, o, h, l, c, v in result:
                hour = ts.hour
                if hour not in hourly_data:
                    hourly_data[hour] = {'high': h, 'low': l, 'open': o, 'close': c}
                else:
                    hourly_data[hour]['high'] = max(hourly_data[hour]['high'], h)
                    hourly_data[hour]['low'] = min(hourly_data[hour]['low'], l)
                    hourly_data[hour]['close'] = c
            
            for hour in sorted(hourly_data.keys()):
                data = hourly_data[hour]
                print(f'{hour:02d}:00 - O:{data["open"]:.2f} H:{data["high"]:.2f} L:{data["low"]:.2f} C:{data["close"]:.2f}')


if __name__ == "__main__":
    main()