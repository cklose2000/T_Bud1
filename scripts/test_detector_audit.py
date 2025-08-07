#!/usr/bin/env python3
"""
PR6 Detector Audit Hooks Test - Prove future peeking detection works.

This script demonstrates the audit system catching various future peeking violations.
"""
import sys
sys.path.append('.')

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.core.detector_audit import (
    audit_detector, audit_sql_query, detector_audit_context, 
    get_violations, clear_violations, create_audit_report,
    FuturePeekError
)


@audit_detector  
def malicious_detector_with_lead(conn, symbol, timeframe, allow_future=False):
    """Intentionally bad detector that uses LEAD() - should be caught."""
    
    # This SQL contains LEAD() which peeks into future
    bad_query = f"""
    SELECT 
        ts,
        close,
        LEAD(close, 1) OVER (ORDER BY ts) as next_close,  -- FUTURE PEEKING!
        LEAD(high, 2) OVER (ORDER BY ts) as future_high   -- MORE FUTURE PEEKING!
    FROM bars_5m
    WHERE symbol = '{symbol}'
    ORDER BY ts
    """
    
    audit_sql_query(bad_query, "malicious_detector_with_lead")
    return conn.execute(bad_query).fetchall()


@audit_detector
def malicious_detector_with_following(conn, symbol, timeframe, allow_future=False):
    """Bad detector that uses FOLLOWING window - should be caught."""
    
    bad_query = f"""
    SELECT 
        ts,
        close,
        AVG(close) OVER (
            ORDER BY ts 
            ROWS BETWEEN CURRENT ROW AND 5 FOLLOWING  -- FUTURE PEEKING!
        ) as future_avg
    FROM bars_5m  
    WHERE symbol = '{symbol}'
    ORDER BY ts
    """
    
    audit_sql_query(bad_query, "malicious_detector_with_following")
    return conn.execute(bad_query).fetchall()


@audit_detector
def malicious_detector_max_timestamp(conn, symbol, timeframe, allow_future=False):
    """Bad detector that queries max timestamp - should be caught."""
    
    bad_query = f"""
    SELECT 
        MAX(ts) as latest_timestamp,  -- FUTURE PEEKING!
        COUNT(*) as total_bars
    FROM bars_5m
    WHERE symbol = '{symbol}'
    """
    
    audit_sql_query(bad_query, "malicious_detector_max_timestamp")
    return conn.execute(bad_query).fetchall()


@audit_detector
def good_causal_detector(conn, symbol, timeframe, allow_future=False):
    """Properly written causal detector - should pass audit."""
    
    good_query = f"""
    SELECT 
        ts,
        close,
        LAG(close, 1) OVER (ORDER BY ts) as prev_close,  -- CAUSAL - OK
        AVG(close) OVER (
            ORDER BY ts 
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW  -- CAUSAL - OK
        ) as trailing_avg
    FROM bars_5m
    WHERE symbol = '{symbol}'
    ORDER BY ts
    """
    
    audit_sql_query(good_query, "good_causal_detector") 
    return conn.execute(good_query).fetchall()


def test_audit_system_catches_violations():
    """Test that audit system catches various future peeking violations."""
    print("🧪 Testing Detector Audit System...")
    
    # Create test data
    with DuckDBManager() as db:
        # Handle existing bars_5m view/table (try both)
        try:
            db.conn.execute("DROP VIEW IF EXISTS bars_5m")
        except:
            pass
        try:
            db.conn.execute("DROP TABLE IF EXISTS bars_5m")
        except:
            pass
        
        db.conn.execute("""
            CREATE TABLE bars_5m AS
            SELECT 
                '2025-08-01 09:30:00'::TIMESTAMP + INTERVAL (i * 5) MINUTE as ts,
                'TEST' as symbol,
                '5m' as timeframe,
                100 + (i % 10) as open,
                100 + (i % 10) + 1 as high, 
                100 + (i % 10) - 1 as low,
                100 + (i % 10) + 0.5 as close,
                1000 as volume
            FROM generate_series(0, 100) as t(i)
        """)
        
        print("✅ Created test data")
        
        test_cases = [
            ("LEAD Function", malicious_detector_with_lead, True),
            ("FOLLOWING Window", malicious_detector_with_following, True), 
            ("MAX Timestamp", malicious_detector_max_timestamp, True),
            ("Good Causal Detector", good_causal_detector, False),
        ]
        
        results = []
        
        for test_name, detector_func, should_violate in test_cases:
            print(f"\n🔍 Testing {test_name}:")
            
            clear_violations()  # Reset violations
            
            try:
                with detector_audit_context(allow_future=False):
                    detector_func(db.conn, "TEST", "5m", allow_future=False)
                
                violations = get_violations()
                
                if should_violate and len(violations) > 0:
                    print(f"✅ CORRECTLY CAUGHT {len(violations)} violations:")
                    for v in violations[:2]:  # Show first 2
                        print(f"   - {v.violation_type}: {v.description}")
                    results.append((test_name, True, "Violations correctly caught"))
                    
                elif should_violate and len(violations) == 0:
                    print(f"❌ MISSED VIOLATION: Expected violations but found none")
                    results.append((test_name, False, "Failed to catch violation"))
                    
                elif not should_violate and len(violations) == 0:
                    print(f"✅ CORRECTLY PASSED: No violations found")
                    results.append((test_name, True, "Correctly passed audit"))
                    
                else:  # not should_violate and len(violations) > 0
                    print(f"❌ FALSE POSITIVE: Found {len(violations)} unexpected violations")
                    for v in violations:
                        print(f"   - {v.violation_type}: {v.description}")
                    results.append((test_name, False, "False positive violations"))
                
            except FuturePeekError as e:
                if should_violate:
                    print(f"✅ CORRECTLY BLOCKED: {e}")
                    results.append((test_name, True, "Correctly blocked with exception"))
                else:
                    print(f"❌ FALSE POSITIVE BLOCK: {e}")
                    results.append((test_name, False, "Incorrectly blocked"))
                    
            except Exception as e:
                print(f"❌ UNEXPECTED ERROR: {e}")
                results.append((test_name, False, f"Unexpected error: {e}"))
        
        return results


def test_audit_context_manager():
    """Test that audit context manager works properly."""
    print("\n🧪 Testing Audit Context Manager...")
    
    with DuckDBManager() as db:
        # Create minimal test data  
        db.conn.execute("""
            CREATE OR REPLACE TABLE test_audit AS
            SELECT 1 as id, 'test' as value
        """)
        
        # Test allow_future=True allows violations
        clear_violations()
        
        try:
            with detector_audit_context(allow_future=True):
                audit_sql_query("SELECT LEAD(value, 1) OVER () FROM test_audit", "test_func")
            
            violations = get_violations()
            if len(violations) == 0:
                print("✅ allow_future=True correctly bypassed audit")
                return True
            else:
                print(f"❌ allow_future=True still caught {len(violations)} violations")
                return False
                
        except FuturePeekError:
            print("❌ allow_future=True still threw FuturePeekError")
            return False


def run_detector_audit_tests():
    """Run comprehensive detector audit tests."""
    print("🛡️  PR6 Detector Audit Hooks Validation")
    print("=" * 60)
    
    # Test main audit system
    main_results = test_audit_system_catches_violations()
    
    # Test context manager
    context_result = test_audit_context_manager()
    
    print("\n" + "=" * 60)
    print("📊 Detector Audit Test Summary:")
    
    passed = 0
    total = len(main_results)
    
    for test_name, success, details in main_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}: {details}")
        if success:
            passed += 1
    
    context_status = "✅ PASS" if context_result else "❌ FAIL"
    print(f"   {context_status} Context Manager Test")
    if context_result:
        passed += 1
    total += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 DETECTOR AUDIT SYSTEM VALIDATED!")
        print("✅ Future peeking violations are correctly detected and blocked")
        print("✅ Causal detectors pass audit without issues")  
        print("✅ Context manager properly controls audit behavior")
        return True
    else:
        print("\n💥 DETECTOR AUDIT VALIDATION FAILED!")
        print("❌ Critical issues in future peek detection system")
        return False


if __name__ == "__main__":
    success = run_detector_audit_tests()
    sys.exit(0 if success else 1)