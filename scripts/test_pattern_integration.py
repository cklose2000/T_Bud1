#!/usr/bin/env python3
"""
Test pattern detection integration with existing data
"""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.hybrid_data_manager import HybridDataManager
from trading_buddy.detectors.patterns import detect_all_patterns


def main():
    print("Testing Pattern Detection Integration")
    print("=" * 80)
    
    with HybridDataManager() as db:
        # Test on August 5 data
        print("\nTesting on August 5, 2025 SPY data:")
        
        # Detect all patterns
        patterns = detect_all_patterns(
            db.conn,
            'SPY',
            '5min',
            lookback_hours=24,
            include_w_patterns=True
        )
        
        # Summary
        print("\nPattern Detection Summary:")
        print("-" * 40)
        total_patterns = 0
        
        for pattern_type, pattern_list in patterns.items():
            count = len(pattern_list)
            total_patterns += count
            print(f"{pattern_type:20} {count:3} patterns")
        
        print("-" * 40)
        print(f"{'TOTAL':20} {total_patterns:3} patterns")
        
        # Show W patterns in detail
        if patterns.get('w_pattern'):
            print("\n\nW Pattern Details:")
            print("=" * 80)
            
            for i, p in enumerate(patterns['w_pattern'][:5], 1):
                attrs = p['attrs']
                print(f"\nPattern {i}:")
                print(f"  Time: {p['event_ts']}")
                print(f"  Bottom 1: {attrs['low1_ts']} @ ${attrs['low1_price']:.2f}")
                print(f"  Bottom 2: {attrs['low2_ts']} @ ${attrs['low2_price']:.2f}")
                print(f"  Second bottom is {attrs['second_bottom_higher_pct']:.2f}% higher")
                print(f"  Neckline: ${attrs['neckline_price']:.2f}")
                print(f"  W Depth: {attrs['w_depth_pct']:.1f}%")
                print(f"  MACD Divergence: {attrs['macd_divergence']}")
                
                if attrs['macd1'] and attrs['macd2']:
                    print(f"  MACD: {attrs['macd1']:.4f} → {attrs['macd2']:.4f}")
                    if attrs['macd2'] > attrs['macd1']:
                        print("  → Bullish momentum building")
        
        # Test multi-symbol detection
        print("\n\nTesting Multi-Symbol Pattern Detection:")
        print("=" * 80)
        
        symbols = ['SPY', 'QQQ', 'NVDA']
        
        for symbol in symbols:
            # Check if we have data for this symbol
            check = db.conn.execute(f"""
            SELECT COUNT(*) 
            FROM bars 
            WHERE symbol = '{symbol}' 
            AND timeframe = '5min'
            """).fetchone()
            
            if check and check[0] > 0:
                patterns = detect_all_patterns(
                    db.conn,
                    symbol,
                    '5min',
                    lookback_hours=24,
                    include_w_patterns=True
                )
                
                # Count total patterns
                total = sum(len(p_list) for p_list in patterns.values())
                print(f"\n{symbol}: {total} total patterns")
                
                # Show breakdown
                for p_type, p_list in patterns.items():
                    if p_list:
                        print(f"  - {p_type}: {len(p_list)}")
            else:
                print(f"\n{symbol}: No data available")
        
        print("\n✓ Pattern detection integration test completed!")


if __name__ == "__main__":
    main()