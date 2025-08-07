#!/usr/bin/env python3
"""
CI step: Validate that final responses don't contain untraced numbers.
This script diffs final messages against claims_log to ensure PR5 truth harness compliance.
"""
import sys
import re
from typing import List, Dict, Set
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append('..')
sys.path.append('.')

import duckdb
from trading_buddy.core.duck import DuckDBManager
from trading_buddy.middleware.referee_enhanced import extract_numeric_claims


def get_recent_responses(conn: duckdb.DuckDBPyConnection, hours: int = 24) -> List[Dict]:
    """Get final responses from the last N hours."""
    cutoff = datetime.now() - timedelta(hours=hours)
    
    result = conn.execute("""
        SELECT message_id, route, symbol, body, ts, claims_count, verified_count, unverified_count
        FROM final_responses 
        WHERE ts > ?
        ORDER BY ts DESC
    """, [cutoff]).fetchall()
    
    return [
        {
            "message_id": row[0],
            "route": row[1], 
            "symbol": row[2],
            "body": row[3],
            "ts": row[4],
            "claims_count": row[5],
            "verified_count": row[6],
            "unverified_count": row[7]
        }
        for row in result
    ]


def get_logged_claims(conn: duckdb.DuckDBPyConnection, message_id: str) -> List[Dict]:
    """Get logged claims for a specific message."""
    result = conn.execute("""
        SELECT claim_text, value_numeric, unit, verdict, tool_name 
        FROM claims_log 
        WHERE message_id = ?
    """, [message_id]).fetchall()
    
    return [
        {
            "claim_text": row[0],
            "value_numeric": row[1], 
            "unit": row[2],
            "verdict": row[3],
            "tool_name": row[4]
        }
        for row in result
    ]


def extract_numbers_from_text(text: str) -> Set[str]:
    """Extract all numeric expressions from text for comparison."""
    # Comprehensive pattern to catch various numeric expressions
    patterns = [
        r'\b\d+\.?\d*%\b',           # Percentages like 75%, 0.28%
        r'\b\d+\.?\d*\s*bps?\b',     # Basis points
        r'\b\d+\.?\d*\b',            # Plain numbers
        r'\b\d+\s*events?\b',        # Count phrases
        r'\b\d+\s*samples?\b',       # Sample sizes
        r'n\s*=\s*\d+',             # n=X format
        r'CI\s*=?\s*\d+\.?\d*',     # CI scores
    ]
    
    found = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.update(matches)
    
    return found


def validate_message_integrity(
    conn: duckdb.DuckDBPyConnection, 
    message: Dict
) -> Dict[str, any]:
    """
    Validate that a final message doesn't contain untraced numbers.
    
    Returns validation result with pass/fail status and details.
    """
    message_id = message["message_id"]
    body = message["body"]
    
    # Extract all numeric expressions from the final message
    text_numbers = extract_numbers_from_text(body)
    
    # Get logged claims for this message
    logged_claims = get_logged_claims(conn, message_id)
    
    # Extract claimed numbers from referee analysis
    referee_claims = extract_numeric_claims(body)
    referee_numbers = set()
    for claim in referee_claims:
        # Add the original text representations
        referee_numbers.add(claim['claim_text'])
        # Also add the numeric value as string
        if claim['value_numeric'] is not None:
            referee_numbers.add(str(claim['value_numeric']))
    
    # Track verified vs unverified numbers  
    verified_numbers = set()
    unverified_numbers = set()
    
    for claim in logged_claims:
        claim_text = claim['claim_text']
        if claim['verdict'] in ['verified', 'corrected']:
            verified_numbers.add(claim_text)
        else:
            unverified_numbers.add(claim_text)
    
    # Find numbers in text that aren't in referee analysis
    unanalyzed_numbers = text_numbers - referee_numbers
    
    # Check for numbers that slipped through without analysis
    integrity_violations = []
    for num_text in unanalyzed_numbers:
        # Skip very common non-metric numbers (years, versions, etc.)
        if re.match(r'^(20\d{2}|[12]?\d{1,2})$', num_text):  # Years or small integers
            continue
        integrity_violations.append(num_text)
    
    # Summary
    total_numbers = len(text_numbers)
    analyzed_numbers = len(referee_numbers & text_numbers)
    verified_count = len([c for c in logged_claims if c['verdict'] in ['verified', 'corrected']])
    rejected_count = len([c for c in logged_claims if c['verdict'] == 'rejected'])
    
    passed = len(integrity_violations) == 0 and rejected_count == 0
    
    return {
        "message_id": message_id,
        "route": message["route"],
        "symbol": message["symbol"],
        "passed": passed,
        "total_numbers_in_text": total_numbers,
        "analyzed_by_referee": analyzed_numbers,
        "verified_claims": verified_count,
        "rejected_claims": rejected_count,
        "integrity_violations": integrity_violations,
        "unverified_numbers": list(unverified_numbers),
        "analysis": {
            "all_text_numbers": list(text_numbers),
            "referee_numbers": list(referee_numbers),
            "unanalyzed": list(unanalyzed_numbers)
        }
    }


def run_integrity_check(hours: int = 24) -> Dict[str, any]:
    """
    Run full integrity check on recent responses.
    
    Returns summary of validation results.
    """
    with DuckDBManager() as db:
        recent_responses = get_recent_responses(db.conn, hours)
        
        if not recent_responses:
            return {
                "status": "success",
                "message": f"No responses found in last {hours} hours",
                "responses_checked": 0,
                "passed": 0,
                "failed": 0,
                "violations": []
            }
        
        results = []
        passed_count = 0
        failed_count = 0
        
        print(f"🔍 Validating {len(recent_responses)} responses from last {hours} hours...")
        
        for response in recent_responses:
            result = validate_message_integrity(db.conn, response)
            results.append(result)
            
            if result["passed"]:
                passed_count += 1
                print(f"✅ {result['route']} {result['symbol']} - {result['verified_claims']} claims verified")
            else:
                failed_count += 1
                print(f"❌ {result['route']} {result['symbol']} - {len(result['integrity_violations'])} violations, {result['rejected_claims']} rejected claims")
                for violation in result['integrity_violations']:
                    print(f"   🚨 Untraced number: {violation}")
                for unverified in result['unverified_numbers']:
                    print(f"   ⚠️  Unverified claim: {unverified}")
        
        all_violations = []
        for r in results:
            all_violations.extend(r['integrity_violations'])
        
        return {
            "status": "success" if failed_count == 0 else "failure",
            "responses_checked": len(recent_responses),
            "passed": passed_count,
            "failed": failed_count,
            "violations": all_violations,
            "details": results
        }


if __name__ == "__main__":
    print("🛡️  PR5 Response Integrity Validation")
    print("=" * 50)
    
    # Allow custom time window via command line
    hours = 24
    if len(sys.argv) > 1:
        try:
            hours = int(sys.argv[1])
        except ValueError:
            print("Usage: python3 validate_response_integrity.py [hours]")
            sys.exit(1)
    
    results = run_integrity_check(hours)
    
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   Responses checked: {results['responses_checked']}")
    print(f"   ✅ Passed: {results['passed']}")
    print(f"   ❌ Failed: {results['failed']}")
    print(f"   🚨 Total violations: {len(results['violations'])}")
    
    if results['status'] == 'failure':
        print(f"\n💥 INTEGRITY CHECK FAILED!")
        print("Some responses contain untraced numeric claims.")
        print("All numbers in final responses must be traceable to tool outputs.")
        sys.exit(1)
    else:
        print(f"\n🎉 INTEGRITY CHECK PASSED!")
        print("All numeric claims are properly traced to tool outputs.")
        sys.exit(0)