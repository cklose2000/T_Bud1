#!/usr/bin/env python3
"""
Test daily report generation
"""
import sys
from pathlib import Path
from datetime import date

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.reports.build_daily_report_v2 import build_daily_report_simple as build_daily_report
from trading_buddy.reports.get_reports import get_reports, get_report_summary


def main():
    print("Testing Daily Report Generation")
    print("=" * 80)
    
    with DuckDBManager() as db:
        # Build a report for SPY
        print("\nBuilding daily report for SPY...")
        result = build_daily_report(db.conn, "SPY", date.today())
        
        print(f"\nReport built:")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Date: {result['as_of']}")
        print(f"  Total size: {result['size_bytes']} bytes")
        print(f"  Sections:")
        for section, size in result['sections'].items():
            print(f"    - {section}: {size} bytes")
        
        # Retrieve the report
        print("\n" + "-" * 80)
        print("Retrieving report...")
        
        reports = get_reports(db.conn, "SPY", days=1)
        if reports:
            report = reports[0]
            print(f"\nReport summary:")
            print(report['summary_md'])
            
            print(f"\nTop edges: {len(report['top_edges'])}")
            for edge in report['top_edges'][:3]:
                print(f"  - {edge['pattern']} ({edge['timeframe']}): "
                      f"n={edge['n']}, R={edge['median_r']:.3f}, "
                      f"Hit={edge['hit_rate']:.1%}")
            
            if report['drift'].get('retired'):
                print(f"\nRetired patterns: {len(report['drift']['retired'])}")
            
            print(f"\nExamples: {len(report['examples'])}")
        
        # Test report summary across multiple days
        print("\n" + "-" * 80)
        print("Testing multi-day summary...")
        
        summary = get_report_summary(db.conn, "SPY", days=7)
        print(f"\nSummary for {summary['days_available']} days:")
        print(summary['summary'])
        
        # Check report size constraint
        print("\n" + "-" * 80)
        print("Checking size constraints...")
        
        total_size = result['size_bytes']
        if total_size <= 5000:
            print(f"✓ Report size OK: {total_size} bytes (< 5KB)")
        else:
            print(f"⚠️  Report size warning: {total_size} bytes (> 5KB)")


if __name__ == "__main__":
    main()