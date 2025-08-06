#!/usr/bin/env python3
"""
Test real-time pattern detection with live IB data
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.hybrid_data_manager import HybridDataManager
from trading_buddy.detectors.patterns import detect_all_patterns
from trading_buddy.data.ib_realtime_fetcher import IBRealtimeFetcher


async def test_realtime_patterns():
    """Test pattern detection with real-time data"""
    print("Testing Real-Time Pattern Detection")
    print("=" * 80)
    
    # Initialize components
    fetcher = IBRealtimeFetcher(
        db_path="./data/trading_buddy.duckdb",
        ib_host="localhost", 
        ib_port=4002
    )
    
    try:
        # Connect to IB
        await fetcher.connect()
        print("✓ Connected to IB Gateway")
        
        # Fetch latest data for SPY
        print("\nFetching latest SPY data...")
        bars = await fetcher.get_historical_data(
            'SPY',
            '5min',
            duration='1 D'
        )
        print(f"  Fetched {len(bars)} bars")
        
        # Use hybrid data manager to detect patterns
        with HybridDataManager() as db:
            print("\nDetecting patterns in latest data...")
            
            # Get all patterns
            patterns = detect_all_patterns(
                db.conn,
                'SPY',
                '5min',
                lookback_hours=8,
                include_w_patterns=True
            )
            
            # Display results
            for pattern_type, pattern_list in patterns.items():
                if pattern_list:
                    print(f"\n{pattern_type.upper()} Patterns Found: {len(pattern_list)}")
                    for p in pattern_list[:3]:  # Show first 3
                        print(f"  - {p['event_ts']}")
                        if pattern_type == 'w_pattern':
                            attrs = p['attrs']
                            print(f"    Bottom 1: ${attrs['low1_price']:.2f} → Bottom 2: ${attrs['low2_price']:.2f}")
                            print(f"    Second bottom {attrs['second_bottom_higher_pct']:.2f}% higher")
                            print(f"    MACD divergence: {attrs['macd_divergence']}")
            
            # Test real-time monitoring
            print("\n" + "-" * 80)
            print("Starting real-time pattern monitoring (5 iterations)...")
            
            for i in range(5):
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteration {i+1}/5")
                
                # Fetch latest bar
                bars = await fetcher.get_historical_data(
                    'SPY',
                    '5min',
                    duration='1 H'
                )
                
                # Check for new patterns
                latest_patterns = detect_all_patterns(
                    db.conn,
                    'SPY',
                    '5min',
                    lookback_hours=2,
                    include_w_patterns=True
                )
                
                new_patterns_found = False
                for pattern_type, pattern_list in latest_patterns.items():
                    if pattern_list:
                        # Check if any patterns are from the last 30 minutes
                        recent_patterns = [
                            p for p in pattern_list
                            if datetime.fromisoformat(str(p['event_ts'])) > datetime.now() - timedelta(minutes=30)
                        ]
                        if recent_patterns:
                            new_patterns_found = True
                            print(f"  NEW {pattern_type}: {len(recent_patterns)} pattern(s)")
                
                if not new_patterns_found:
                    print("  No new patterns in last 30 minutes")
                
                # Wait 10 seconds before next iteration
                if i < 4:
                    await asyncio.sleep(10)
    
    finally:
        fetcher.disconnect()
        print("\n✓ Disconnected from IB Gateway")
        print("\nReal-time pattern detection test completed!")


def main():
    asyncio.run(test_realtime_patterns())


if __name__ == "__main__":
    main()