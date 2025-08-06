#!/usr/bin/env python3
"""
Build/refresh the Consistency Index matrix.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.council.consistency import update_consistency_matrix
from trading_buddy.core.duck import DuckDBManager


def main():
    print("Building Consistency Index matrix...")
    
    with DuckDBManager() as db:
        # Check if we have events
        event_count = db.conn.execute("""
            SELECT COUNT(*) FROM events WHERE fwd_ret IS NOT NULL
        """).fetchone()[0]
        
        if event_count == 0:
            print("Error: No events with outcomes found!")
            print("Please run backfill first.")
            return
        
        print(f"Found {event_count} events with outcomes")
        
        # Define pattern and context combinations
        base_configs = [
            ("5m", "double_bottom"),
            ("5m", "macd_bull_cross"),
            ("5m", "double_bottom_macd_bull"),
        ]
        
        context_configs = [
            ("15m", "trend_up_ema20"),
            ("60m", "regime_trend"),
            ("60m", "regime_mrv"),
            ("60m", "regime_chop"),
        ]
        
        # Update CI for each combination
        total_updated = 0
        
        for base_tf, base_pattern in base_configs:
            print(f"\nProcessing {base_pattern} on {base_tf}...")
            
            update_consistency_matrix(
                db.conn,
                base_tf,
                base_pattern,
                context_configs
            )
            
            # Count updates
            updated = db.conn.execute("""
                SELECT COUNT(*) 
                FROM consistency_matrix
                WHERE base_tf = ? AND base_pattern = ?
            """, [base_tf, base_pattern]).fetchone()[0]
            
            total_updated += updated
            print(f"  Updated {updated} context combinations")
        
        db.conn.commit()
        
        # Show top CI values
        print("\nTop Consistency Index values:")
        top_ci = db.conn.execute("""
            SELECT 
                base_tf, base_pattern, ctx_tf, ctx_pattern,
                n, exp_lift, hit_rate, ci
            FROM consistency_matrix
            WHERE ci > 0
            ORDER BY ci DESC
            LIMIT 10
        """).fetchall()
        
        for row in top_ci:
            base_tf, base_pattern, ctx_tf, ctx_pattern, n, lift, hr, ci = row
            print(f"  {base_pattern}({base_tf}) + {ctx_pattern}({ctx_tf}): "
                  f"CI={ci:.3f}, lift={lift:.3f}, hr={hr:.2f}, n={n}")
        
        print(f"\nConsistency Index build complete!")
        print(f"Total combinations: {total_updated}")


if __name__ == "__main__":
    main()