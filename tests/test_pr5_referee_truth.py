"""
PR5 Truth Harness Tests - Prove every number is tool-traceable.
"""
import re
import json
import pytest

from trading_buddy.middleware.referee_enhanced import extract_numeric_claims, verify_claims_against_tools

# Sneaky prompts that try to slip numbers via synonyms
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
    "performance slightly less than expected"
]


@pytest.mark.parametrize("text", SNEAKY)
def test_referee_extracts_implied_metrics(text):
    """Test that referee catches sneaky numeric claims."""
    claims = extract_numeric_claims(text)
    # Expect at least one normalized numeric claim
    assert len(claims) > 0, f"No claims found in: {text}"
    assert any(c["value_numeric"] is not None for c in claims), f"No numeric values found in: {text}"
    
    # Log what we found for debugging
    print(f"Text: {text}")
    print(f"Claims found: {[c['claim_text'] + ' -> ' + str(c['value_numeric']) for c in claims]}")


def test_referee_catches_pct_decimal_mismatch():
    """Test detection of ambiguous unit representations."""
    text = "median 0.28 over 10 bars"  # Could be 0.28 or 0.28% 
    claims = extract_numeric_claims(text)
    
    assert len(claims) > 0
    # Should flag potential unit disambiguation
    assert any(c.get("needs_unit_disambiguation") for c in claims), "Should detect unit ambiguity"


def test_referee_requires_tool_source():
    """Test that claims without tool sources are rejected."""
    text = "median +0.28% over 10 bars, hit rate 62%"
    claims = extract_numeric_claims(text)
    
    # Simulate no tool hits -> should reject all claims
    results = verify_claims_against_tools(claims, tool_hits=[])
    
    assert len(results) > 0
    assert all(r["verdict"] == "rejected" for r in results), "All claims should be rejected without tool sources"


def test_referee_accepts_verified_claims():
    """Test that claims with matching tool sources are accepted."""
    text = "median return 0.0034, hit rate 75%, sample size 24"
    claims = extract_numeric_claims(text)
    
    # Simulate tool hit with matching data
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
    assert verified_count > 0, "Should verify at least some claims with tool data"


def test_referee_corrects_mismatched_values():
    """Test auto-correction of slightly mismatched values."""
    text = "hit rate 74%"  # Model claims 74%
    claims = extract_numeric_claims(text)
    
    # Tool actually returned 74.2%
    tool_hits = [
        {
            'tool_name': 'council_vote',
            'data': {
                'hit_rate': 0.742  # Actual value from tool
            }
        }
    ]
    
    results = verify_claims_against_tools(claims, tool_hits)
    
    assert len(results) > 0
    corrected = [r for r in results if r["verdict"] == "corrected"]
    assert len(corrected) > 0, "Should auto-correct mismatched values"
    
    # Check the corrected value
    corrected_claim = corrected[0] 
    assert abs(corrected_claim["value_numeric"] - 0.742) < 0.001, "Should correct to tool value"


def test_comprehensive_pattern_matching():
    """Test comprehensive pattern matching including edge cases."""
    
    test_cases = [
        ("CI of 0.67", "score"),
        ("return of +0.003", "return"), 
        ("n=45 events", "count"),
        ("about third of a percent", "implied_pct"),
        ("roughly 28 bps", "implied_bps"),
        ("modestly positive", "implied_small_pos"),
        ("call it 0.68", "approximation"),
        ("high (around 70%)", "implied_high"),
    ]
    
    for text, expected_unit_type in test_cases:
        claims = extract_numeric_claims(text)
        assert len(claims) > 0, f"No claims found for: {text}"
        
        # Check that we found the expected unit type
        units_found = [c["unit"] for c in claims]
        assert any(expected_unit_type in unit for unit in units_found), \
            f"Expected unit '{expected_unit_type}' not found in {units_found} for text: {text}"


def test_no_false_positives():
    """Test that non-numeric text doesn't generate false claims."""
    
    non_numeric_texts = [
        "The pattern looks good",
        "Strong momentum building",
        "Market trending higher",
        "No clear signal yet"
    ]
    
    for text in non_numeric_texts:
        claims = extract_numeric_claims(text)
        # Should find no claims in purely qualitative text
        assert len(claims) == 0, f"False positive claims found in: {text}"


def test_deduplication():
    """Test that duplicate claims are properly deduplicated."""
    text = "median 0.003 return, median 0.003 return"  # Duplicate
    claims = extract_numeric_claims(text)
    
    # Should deduplicate identical claims
    unique_claim_texts = set(c["claim_text"] for c in claims)
    assert len(claims) == len(unique_claim_texts), "Should deduplicate identical claims"


@pytest.mark.parametrize("text,expected_min_claims", [
    ("SPY shows 75% hit rate with median +0.28% over 24 events, CI 0.67", 4),
    ("Council score 0.85 based on n=156 samples, expectancy roughly 45 bps", 3),
    ("Performance about a third better than baseline with high precision", 2),
])
def test_complex_multi_claim_extraction(text, expected_min_claims):
    """Test extraction from complex sentences with multiple claims."""
    claims = extract_numeric_claims(text)
    
    assert len(claims) >= expected_min_claims, \
        f"Expected at least {expected_min_claims} claims, got {len(claims)} in: {text}"
    
    # All claims should have valid numeric values
    assert all(c["value_numeric"] is not None for c in claims)
    assert all(isinstance(c["value_numeric"], (int, float)) for c in claims)


def test_unit_normalization():
    """Test that different unit representations are normalized correctly."""
    
    normalization_tests = [
        ("75%", 0.75),  # Percentage to decimal
        ("28 bps", 0.0028),  # Basis points to decimal
        ("third of a percent", 0.0033),  # Approximate
        ("hair under 0.5", 0.475),  # 95% of 0.5
    ]
    
    for text, expected_normalized in normalization_tests:
        claims = extract_numeric_claims(text)
        assert len(claims) > 0, f"No claims found for: {text}"
        
        normalized_value = claims[0]["value_numeric"]
        assert abs(normalized_value - expected_normalized) < 0.001, \
            f"Expected {expected_normalized}, got {normalized_value} for: {text}"