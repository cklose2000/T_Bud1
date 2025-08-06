#!/usr/bin/env python3
"""
Backfill August 5, 2025 SPY patterns
Specifically looks for the double bottom pattern mentioned by the user
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.hybrid_data_manager import HybridDataManager
from trading_buddy.detectors.patterns import (
    detect_double_bottom,
    detect_macd_bull_cross,
    detect_compound_pattern,
    detect_swing_points
)
from trading_buddy.detectors.contexts import (
    detect_trend_context,
    detect_regime
)
from trading_buddy.detectors.outcomes import update_event_outcomes
from trading_buddy.council.consistency import update_consistency_matrix


def main():
    print("Backfilling August 5, 2025 SPY patterns...")
    print("=" * 60)
    
    # Use hybrid data manager
    with HybridDataManager() as db:
        # Focus on August 5, 2025
        start_date = "2025-08-04"
        end_date = "2025-08-06"
        
        # Check data availability
        freshness = db.check_data_freshness("SPY", "5min")
        print(f"SPY 5min data:")
        print(f"  Latest: {freshness['latest']}")
        print(f"  Gap: {freshness['gap_minutes']:.1f} minutes")
        print(f"  Fresh: {freshness['is_fresh']}")
        print()
        
        # Look for swing points first
        print("Detecting swing points...")
        swings = detect_swing_points(
            db.conn, "SPY", "5m",
            start_ts=start_date,
            end_ts=end_date
        )
        
        print(f"  Found {len(swings['lows'])} swing lows")
        print(f"  Found {len(swings['highs'])} swing highs")
        
        if swings['lows']:
            print("\n  Recent swing lows:")
            for ts, price in swings['lows'][-5:]:
                print(f"    {ts}: ${price:.2f}")
        
        # Detect double bottoms
        print("\nDetecting double bottoms...")
        double_bottoms = detect_double_bottom(
            db.conn, "SPY", "5m",
            lookback_hours=4.0,  # Increase lookback
            tolerance_bps=100,   # Increase tolerance to 1%
            start_ts=start_date,
            end_ts=end_date
        )
        
        print(f"  Found {len(double_bottoms)} double bottoms")
        
        for db_pattern in double_bottoms:
            print(f"\n  Double Bottom at {db_pattern['event_ts']}:")
            attrs = db_pattern['attrs']
            print(f"    Low 1: ${attrs['low1_price']:.2f} at {attrs['low1_ts']}")
            print(f"    Low 2: ${attrs['low2_price']:.2f} at {attrs['low2_ts']}")
            print(f"    Neckline: ${attrs['neckline_price']:.2f}")
            print(f"    Confirmed at: ${attrs['confirm_price']:.2f}")
            
            # Save to database
            exists = db.conn.execute("""
                SELECT 1 FROM events
                WHERE symbol = 'SPY' AND timeframe = '5m'
                AND event_ts = ? AND pattern = 'double_bottom'
            """, [db_pattern['event_ts']]).fetchone()
            
            if not exists:
                db.conn.execute("""
                    INSERT INTO events 
                    (symbol, timeframe, event_ts, pattern, attrs)
                    VALUES ('SPY', '5m', ?, 'double_bottom', ?)
                """, [db_pattern['event_ts'], db_pattern['attrs']])
                print("    ✓ Saved to database")
        
        # Detect MACD crosses
        print("\nDetecting MACD crosses...")
        macd_crosses = detect_macd_bull_cross(
            db.conn, "SPY", "5m",
            start_ts=start_date,
            end_ts=end_date
        )
        
        print(f"  Found {len(macd_crosses)} MACD bull crosses")
        
        for cross in macd_crosses:
            print(f"\n  MACD Cross at {cross['event_ts']}:")
            attrs = cross['attrs']
            print(f"    MACD: {attrs['macd']:.4f}")
            print(f"    Signal: {attrs['signal']:.4f}")
            print(f"    Histogram: {attrs['histogram']:.4f}")
            
            # Save to database
            exists = db.conn.execute("""
                SELECT 1 FROM events
                WHERE symbol = 'SPY' AND timeframe = '5m'
                AND event_ts = ? AND pattern = 'macd_bull_cross'
            """, [cross['event_ts']]).fetchone()
            
            if not exists:
                db.conn.execute("""
                    INSERT INTO events 
                    (symbol, timeframe, event_ts, pattern, attrs)
                    VALUES ('SPY', '5m', ?, 'macd_bull_cross', ?)
                """, [cross['event_ts'], cross['attrs']])
                print("    ✓ Saved to database")
        
        # Detect compound patterns
        print("\nDetecting compound patterns...")
        compounds = detect_compound_pattern(
            db.conn, "SPY", "5m",
            within_bars=6,  # Allow more bars between patterns
            start_ts=start_date,
            end_ts=end_date
        )
        
        print(f"  Found {len(compounds)} compound patterns")
        
        for comp in compounds:
            print(f"\n  Compound Pattern at {comp['event_ts']}:")
            attrs = comp['attrs']
            print(f"    Double Bottom confirmed")
            print(f"    MACD Cross at {attrs['macd_cross_ts']}")
            
            # Save to database
            exists = db.conn.execute("""
                SELECT 1 FROM events
                WHERE symbol = 'SPY' AND timeframe = '5m'
                AND event_ts = ? AND pattern = 'double_bottom_macd_bull'
            """, [comp['event_ts']]).fetchone()
            
            if not exists:
                db.conn.execute("""
                    INSERT INTO events 
                    (symbol, timeframe, event_ts, pattern, attrs)
                    VALUES ('SPY', '5m', ?, 'double_bottom_macd_bull', ?)
                """, [comp['event_ts'], comp['attrs']])
                print("    ✓ Saved to database")
        
        # Detect contexts
        print("\nDetecting 15m trend contexts...")
        trends = detect_trend_context(
            db.conn, "SPY", "15m",
            start_ts=start_date,
            end_ts=end_date
        )
        
        print(f"  Found {len(trends)} trend contexts")
        
        # Detect regimes
        print("\nDetecting 60m regimes...")
        regimes = detect_regime(
            db.conn, "SPY", "60m",
            start_ts="2025-07-01",  # Longer lookback for regime
            end_ts=end_date
        )
        
        regime_counts = {}
        for r in regimes:
            regime_type = r['context']
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
        
        print(f"  Regime distribution:")
        for regime, count in regime_counts.items():
            print(f"    {regime}: {count}")
        
        # Commit changes
        db.conn.commit()
        
        # Update outcomes for new events
        print("\nUpdating outcomes...")
        events = db.conn.execute("""
            SELECT symbol, timeframe, event_ts, pattern
            FROM events
            WHERE symbol = 'SPY'
            AND event_ts >= ? AND event_ts <= ?
            AND fwd_ret IS NULL
        """, [start_date, end_date]).fetchall()
        
        if events:
            event_dicts = [
                {
                    "symbol": e[0],
                    "timeframe": e[1],
                    "event_ts": e[2],
                    "pattern": e[3],
                }
                for e in events
            ]
            
            updated = update_event_outcomes(db.conn, event_dicts)
            print(f"  Updated {updated} event outcomes")
            db.conn.commit()
        
        # Update consistency matrix
        print("\nUpdating consistency matrix...")
        update_consistency_matrix(
            db.conn,
            base_tf="5m",
            base_pattern="double_bottom_macd_bull",
            context_configs=[
                ("15m", "trend_up_ema20"),
                ("60m", "regime_trend"),
                ("60m", "regime_mrv"),
            ]
        )
        db.conn.commit()
        
        # Show summary
        print("\n" + "=" * 60)
        print("Backfill Summary:")
        
        total_events = db.conn.execute("""
            SELECT COUNT(*) FROM events
            WHERE symbol = 'SPY'
            AND event_ts >= ? AND event_ts <= ?
        """, [start_date, end_date]).fetchone()[0]
        
        print(f"  Total events for Aug 5: {total_events}")
        
        # Get latest patterns
        latest_patterns = db.get_latest_patterns(hours=48)
        spy_patterns = [p for p in latest_patterns if p['symbol'] == 'SPY']
        
        print(f"\nRecent SPY patterns:")
        for pattern in spy_patterns[:5]:
            print(f"  {pattern['pattern']} at {pattern['event_ts']}")
            if pattern['fwd_ret'] is not None:
                print(f"    Forward return: {pattern['fwd_ret']:.2%}")
        
        print("\n✅ Backfill complete!")


if __name__ == "__main__":
    main()