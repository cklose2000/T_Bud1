#!/usr/bin/env python3
"""
Debug the detector audit system step by step.
"""
import sys
sys.path.append('.')

from trading_buddy.core.detector_audit import (
    audit_sql_query, detector_audit_context, get_violations, 
    clear_violations, _audit_context
)


def test_sql_pattern_matching():
    """Test that SQL patterns are being matched correctly."""
    print("🧪 Testing SQL Pattern Matching...")
    
    test_queries = [
        ("LEAD Function", "SELECT LEAD(close, 1) OVER (ORDER BY ts) FROM test", True),
        ("FOLLOWING Window", "SELECT AVG(close) OVER (ROWS BETWEEN CURRENT ROW AND 5 FOLLOWING) FROM test", True),
        ("MAX Timestamp", "SELECT MAX(ts) FROM test", True),
        ("Good LAG Query", "SELECT LAG(close, 1) OVER (ORDER BY ts) FROM test", False),
        ("Good Trailing Window", "SELECT AVG(close) OVER (ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) FROM test", False),
    ]
    
    for test_name, query, should_violate in test_queries:
        print(f"\n🔍 Testing {test_name}:")
        print(f"   Query: {query}")
        
        clear_violations()
        
        # Test with audit context enabled
        try:
            with detector_audit_context(allow_future=False):
                try:
                    audit_sql_query(query, "test_function")
                    violations = get_violations()
                    print(f"   Violations found: {len(violations)}")
                except Exception as e:
                    violations = get_violations()
                    print(f"   Violations found: {len(violations)} (raised: {type(e).__name__})")
            
            if violations:
                for v in violations:
                    print(f"     - {v.violation_type}: {v.description}")
            
            if should_violate and len(violations) > 0:
                print("   ✅ CORRECTLY CAUGHT violation")
            elif should_violate and len(violations) == 0:
                print("   ❌ MISSED violation")
            elif not should_violate and len(violations) == 0:
                print("   ✅ CORRECTLY PASSED")
            else:
                print("   ❌ FALSE POSITIVE")
                
        except Exception as e:
            print(f"   ❌ CONTEXT ERROR: {e}")


def test_audit_context_state():
    """Test that audit context state is working."""
    print("\n🧪 Testing Audit Context State...")
    
    print(f"Initial state:")
    print(f"  allow_future: {_audit_context.allow_future}")
    print(f"  in_detector_call: {_audit_context.in_detector_call}")
    
    with detector_audit_context(allow_future=False):
        print(f"Inside context (allow_future=False):")
        print(f"  allow_future: {_audit_context.allow_future}")
        print(f"  in_detector_call: {_audit_context.in_detector_call}")
        
        # Test audit behavior
        clear_violations()
        try:
            audit_sql_query("SELECT LEAD(close, 1) OVER (ORDER BY ts) FROM test", "debug_test")
            violations = get_violations()
            print(f"  Violations after LEAD query: {len(violations)}")
        except Exception as e:
            violations = get_violations()
            print(f"  Violations after LEAD query: {len(violations)} (raised: {type(e).__name__})")
    
    print(f"After context:")
    print(f"  allow_future: {_audit_context.allow_future}")
    print(f"  in_detector_call: {_audit_context.in_detector_call}")


def test_pattern_regex():
    """Test individual regex patterns."""
    print("\n🧪 Testing Regex Patterns...")
    
    import re
    from trading_buddy.core.detector_audit import _audit_context
    
    test_cases = [
        ("LEAD(close, 1)", r'\bLEAD\s*\('),
        ("ROWS BETWEEN CURRENT ROW AND 5 FOLLOWING", r'ROWS\s+BETWEEN\s+.*FOLLOWING'),
        ("MAX(ts)", r'MAX\s*\(\s*ts\s*\)'),
    ]
    
    for text, pattern in test_cases:
        match = re.search(pattern, text.upper(), re.IGNORECASE)
        print(f"   Text: '{text}'")
        print(f"   Pattern: '{pattern}'")
        print(f"   Match: {'✅ YES' if match else '❌ NO'}")
        if match:
            print(f"   Matched: '{match.group()}'")


if __name__ == "__main__":
    print("🛡️  Debugging PR6 Detector Audit System")
    print("=" * 50)
    
    test_pattern_regex()
    test_audit_context_state()
    test_sql_pattern_matching()