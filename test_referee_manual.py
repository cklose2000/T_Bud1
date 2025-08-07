#!/usr/bin/env python3
"""
Manual test script for PR5 referee truth harness.
"""
import sys
sys.path.append('.')

from trading_buddy.middleware.referee_enhanced import extract_numeric_claims, verify_claims_against_tools

# Test cases from the pytest file
SNEAKY = [
    "up about a third of a percent in 10 bars",
    "roughly +28 bps median outcome", 
    "hit rate is high (around sixty percent)",
    "mae is a hair under 0.2%",
    "expectancy modestly positive; call it 0.003",
    "return about 25 bps with strong lift",
    "council score approximately 0.68",
    "sample size of roughly 45 trades",
    "CI around seventy percent confidence",
# "performance slightly less than expected"  # This is qualitative, no numbers
]

def test_pattern_extraction():
    """Test that all sneaky patterns are caught."""
    print("Testing pattern extraction...")
    
    for text in SNEAKY:
        print(f"\nTesting: '{text}'")
        claims = extract_numeric_claims(text)
        
        if len(claims) == 0:
            print(f"❌ FAIL: No claims found in: {text}")
            return False
        
        print(f"✅ Found {len(claims)} claims:")
        for claim in claims:
            print(f"   - {claim['claim_text']} -> {claim['value_numeric']} ({claim['unit']})")
    
    return True

def test_tool_verification():
    """Test claim verification against tool results."""
    print("\n\nTesting tool verification...")
    
    text = "median return 0.0034, hit rate 75%, sample size 24"
    claims = extract_numeric_claims(text)
    print(f"Extracted {len(claims)} claims from: '{text}'")
    
    # Simulate matching tool results
    tool_hits = [
        {
            'tool_name': 'council_vote',
            'data': {
                'median_return': 0.0034,
                'hit_rate': 0.75,  # As decimal
                'n': 24
            }
        }
    ]
    
    results = verify_claims_against_tools(claims, tool_hits)
    
    verified_count = sum(1 for r in results if r["verdict"] in ["verified", "corrected"])
    rejected_count = sum(1 for r in results if r["verdict"] == "rejected")
    
    print(f"Verification results: {verified_count} verified, {rejected_count} rejected")
    
    for result in results:
        status = "✅" if result["verdict"] in ["verified", "corrected"] else "❌"
        print(f"   {status} {result['claim_text']} -> {result['verdict']}")
    
    return verified_count > 0

def test_rejection_case():
    """Test that unverified claims are rejected."""
    print("\n\nTesting rejection case...")
    
    text = "hit rate 85%, median return +0.45%"
    claims = extract_numeric_claims(text)
    print(f"Extracted {len(claims)} claims from: '{text}'")
    
    # No tool results - should reject all
    results = verify_claims_against_tools(claims, tool_hits=[])
    
    rejected_count = sum(1 for r in results if r["verdict"] == "rejected")
    
    print(f"All {rejected_count} claims rejected (as expected)")
    
    return rejected_count == len(claims)

if __name__ == "__main__":
    print("🧪 PR5 Referee Truth Harness Manual Tests")
    print("=" * 50)
    
    success = True
    success &= test_pattern_extraction()
    success &= test_tool_verification()  
    success &= test_rejection_case()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests PASSED!")
        print("Enhanced referee successfully catches sneaky numeric claims!")
    else:
        print("💥 Some tests FAILED!")
        sys.exit(1)