#!/usr/bin/env python3
"""
Test PR6 audit system integration with real detector functions.
"""
import sys
sys.path.append('.')

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.detectors.indicators import compute_ema, compute_macd, compute_atr, compute_ema_slope  
from trading_buddy.detectors.patterns import detect_swing_points
from trading_buddy.core.detector_audit import get_violations, clear_violations


def test_real_detector_integration():
    """Test that real detector functions work with audit system."""
    print("🧪 Testing Real Detector Integration...")
    
    # Create test data
    with DuckDBManager() as db:
        # Clean up existing data
        try:
            db.conn.execute("DROP VIEW IF EXISTS bars")
        except:
            pass
        try:
            db.conn.execute("DROP TABLE IF EXISTS bars")
        except:
            pass
        
        db.conn.execute("""
            CREATE TABLE bars AS
            SELECT 
                '2025-08-01 09:30:00'::TIMESTAMP + INTERVAL (i * 5) MINUTE as ts,
                'TSLA' as symbol,
                '5m' as timeframe,
                100 + (i % 10) as open,
                100 + (i % 10) + 2 as high, 
                100 + (i % 10) - 2 as low,
                100 + (i % 10) + 1 as close,
                1000 + (i * 10) as volume
            FROM generate_series(0, 200) as t(i)
        """)
        
        print("✅ Created test data (200 bars)")
        
        # Test each real detector function
        detectors = [
            ("EMA", lambda: compute_ema(db.conn, "TSLA", "5m", 20)),
            ("MACD", lambda: compute_macd(db.conn, "TSLA", "5m", 12, 26, 9)), 
            ("ATR", lambda: compute_atr(db.conn, "TSLA", "5m", 14)),
            ("EMA Slope", lambda: compute_ema_slope(db.conn, "TSLA", "5m", 20, 5)),
            ("Swing Points", lambda: detect_swing_points(db.conn, "TSLA", "5m", 3)),
        ]
        
        results = []
        
        for detector_name, detector_func in detectors:
            print(f"\n🔍 Testing {detector_name}:")
            
            clear_violations()
            
            try:
                result = detector_func()
                violations = get_violations()
                
                if violations:
                    print(f"❌ UNEXPECTED VIOLATIONS: {len(violations)}")
                    for v in violations[:2]:
                        print(f"   - {v.violation_type}: {v.description}")
                    results.append((detector_name, False, f"{len(violations)} violations"))
                else:
                    print(f"✅ PASSED: No violations, returned {len(result)} results")
                    results.append((detector_name, True, "Clean execution"))
                    
            except Exception as e:
                print(f"❌ ERROR: {str(e)[:100]}...")
                results.append((detector_name, False, f"Exception: {type(e).__name__}"))
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Real Detector Integration Summary:")
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        for name, success, details in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {name}: {details}")
        
        print(f"\nOverall: {passed}/{total} detectors passed")
        
        if passed == total:
            print("\n🎉 ALL REAL DETECTORS INTEGRATE SUCCESSFULLY!")
            print("✅ No future peeking violations detected in production code")
            return True
        else:
            print(f"\n💥 INTEGRATION ISSUES FOUND!")
            print("❌ Some detectors have audit violations or errors")
            return False


if __name__ == "__main__":
    print("🛡️  PR6 Real Detector Integration Test")
    print("="*60)
    
    success = test_real_detector_integration()
    exit(0 if success else 1)