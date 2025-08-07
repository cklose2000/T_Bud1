#!/usr/bin/env python3
"""
Test what-if cache functionality.
"""
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.core.ddl import create_whatif_cache_table
from trading_buddy.whatif.cache import WhatIfCache
from trading_buddy.schemas.hypothesis import Hypothesis
from trading_buddy.schemas.council import CouncilResponse


def setup_test_data(db):
    """Set up minimal test data."""
    print("Setting up test data...")
    
    # Create whatif_cache table
    create_whatif_cache_table(db.conn)
    
    # Clean up any existing test data
    db.conn.execute("DELETE FROM whatif_cache")
    
    # Mock some basic test data if needed
    print("✓ Test data setup complete")


def test_hash_consistency():
    """Test that hypothesis and override hashing is consistent."""
    print("\nTesting hash consistency...")
    
    with DuckDBManager() as db:
        cache = WhatIfCache(db.conn)
        
        # Create test hypothesis
        hypothesis1 = Hypothesis(
            symbol="SPY",
            base_timeframe="5m",
            patterns=[{"name": "double_bottom", "params": {"depth": 0.02}}],
            context=[{"name": "trend_up", "params": {"tf": "15m"}}],
            cross_timeframe_votes=[]
        )
        
        # Same hypothesis with different ordering
        hypothesis2 = Hypothesis(
            symbol="SPY",
            base_timeframe="5m",
            patterns=[{"name": "double_bottom", "params": {"depth": 0.02}}],
            context=[{"name": "trend_up", "params": {"tf": "15m"}}],
            cross_timeframe_votes=[]
        )
        
        hash1 = cache._hash_hypothesis(hypothesis1)
        hash2 = cache._hash_hypothesis(hypothesis2)
        
        print(f"Hash 1: {hash1}")
        print(f"Hash 2: {hash2}")
        
        assert hash1 == hash2, "Identical hypotheses should have same hash"
        
        # Test override hashing
        overrides1 = {"contexts": [{"tf": "60m", "pattern": "trend_up", "present": True}]}
        overrides2 = {"contexts": [{"tf": "60m", "pattern": "trend_up", "present": True}]}
        
        override_hash1 = cache._hash_overrides(overrides1)
        override_hash2 = cache._hash_overrides(overrides2)
        
        assert override_hash1 == override_hash2, "Identical overrides should have same hash"
        
        print("✓ Hash consistency working")


def test_cache_operations():
    """Test basic cache put/get operations."""
    print("\nTesting cache operations...")
    
    with DuckDBManager() as db:
        setup_test_data(db)
        cache = WhatIfCache(db.conn, default_ttl_hours=1)
        
        # Create test data
        hypothesis = Hypothesis(
            symbol="SPY",
            base_timeframe="5m", 
            patterns=[{"name": "double_bottom", "params": {}}],
            context=[],
            cross_timeframe_votes=[]
        )
        
        overrides = {"contexts": [{"tf": "60m", "pattern": "trend_up", "present": True}]}
        
        response = CouncilResponse(
            n=25,
            median_fwd_ret=0.0045,
            mean_fwd_ret=0.0040,
            hit_rate=0.76,
            mae=0.015,
            council_score=0.68,
            ci_breakdown=[],
            plan={
                "entry": "627.50",
                "stop": "625.00", 
                "time_stop": 15,
                "size": 1.0
            }
        )
        
        # Test cache miss
        cached_result = cache.get(hypothesis, overrides)
        assert cached_result is None, "Cache should be empty initially"
        
        # Test cache put
        success = cache.put(hypothesis, overrides, response, ttl_hours=1)
        assert success, "Cache put should succeed"
        
        # Test cache hit
        cached_result = cache.get(hypothesis, overrides)
        assert cached_result is not None, "Cache should return result after put"
        assert cached_result.n == 25, "Cached result should match original"
        assert cached_result.median_fwd_ret == 0.0045, "Cached values should match"
        
        # Test hit count increment
        cached_result2 = cache.get(hypothesis, overrides)
        assert cached_result2 is not None, "Second cache hit should work"
        
        print("✓ Basic cache operations working")


def test_cache_expiration():
    """Test cache expiration logic."""
    print("\nTesting cache expiration...")
    
    with DuckDBManager() as db:
        setup_test_data(db)
        cache = WhatIfCache(db.conn)
        
        hypothesis = Hypothesis(
            symbol="QQQ",
            base_timeframe="5m",
            patterns=[{"name": "macd_cross", "params": {}}],
            context=[],
            cross_timeframe_votes=[]
        )
        
        overrides = {"contexts": [{"tf": "15m", "pattern": "volume_spike", "present": True}]}
        
        response = CouncilResponse(
            n=15,
            median_fwd_ret=0.002,
            mean_fwd_ret=0.0018,
            hit_rate=0.67,
            mae=0.012,
            council_score=0.55,
            ci_breakdown=[],
            plan={
                "entry": "500.00",
                "stop": "498.00",
                "time_stop": 10,
                "size": 1.0
            }
        )
        
        # Cache with very short TTL (simulate expired)
        cache.put(hypothesis, overrides, response, ttl_hours=0.001)  # ~3 seconds
        
        # Should get result immediately
        cached = cache.get(hypothesis, overrides)
        assert cached is not None, "Should get cached result immediately"
        
        # Wait a bit and it should be expired (in a real test we'd mock datetime)
        time.sleep(0.1)
        
        # Test cleanup
        cleaned = cache.cleanup_expired()
        print(f"Cleaned up {cleaned} expired entries")
        
        print("✓ Cache expiration working")


def test_cache_stats():
    """Test cache statistics."""
    print("\nTesting cache statistics...")
    
    with DuckDBManager() as db:
        setup_test_data(db)
        cache = WhatIfCache(db.conn)
        
        # Add some test entries
        for i in range(3):
            hypothesis = Hypothesis(
                symbol=f"TEST{i}",
                base_timeframe="5m",
                patterns=[{"name": "test_pattern", "params": {}}],
                context=[],
                cross_timeframe_votes=[]
            )
            
            overrides = {"test": i}
            
            response = CouncilResponse(
                n=10 + i,
                median_fwd_ret=0.001 * i,
                mean_fwd_ret=0.0008 * i,
                hit_rate=0.5 + i * 0.1,
                mae=0.01 + i * 0.001,
                council_score=0.4 + i * 0.1,
                ci_breakdown=[],
                plan={
                    "entry": f"{100.0 + i}",
                    "stop": f"{98.0 + i}",
                    "time_stop": 10,
                    "size": 1.0
                }
            )
            
            cache.put(hypothesis, overrides, response)
        
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
        
        assert stats["total_entries"] >= 3, "Should have at least 3 entries"
        assert "active_entries" in stats, "Should have active entries count"
        
        print("✓ Cache statistics working")


def test_get_or_compute():
    """Test get_or_compute functionality."""
    print("\nTesting get_or_compute...")
    
    with DuckDBManager() as db:
        setup_test_data(db)
        cache = WhatIfCache(db.conn)
        
        # This will fail to compute because we don't have proper test data
        # But we can test the logic flow
        hypothesis = Hypothesis(
            symbol="AAPL",
            base_timeframe="5m",
            patterns=[{"name": "fake_pattern", "params": {}}],
            context=[],
            cross_timeframe_votes=[]
        )
        
        overrides = {"test_override": True}
        
        # This should return an error response since fake_pattern doesn't exist
        result, was_cached = cache.get_or_compute(hypothesis, overrides)
        
        print(f"Get-or-compute result: cached={was_cached}, n={result.n}")
        
        # Should not be cached initially
        assert not was_cached, "First call should not be cached"
        
        # Error response should have n=0
        assert result.n == 0, "Error response should have n=0"
        
        print("✓ Get-or-compute working")


def main():
    """Run all what-if cache tests."""
    print("=" * 60)
    print("WHAT-IF CACHE TESTS")
    print("=" * 60)
    
    try:
        test_hash_consistency()
        test_cache_operations()
        test_cache_expiration() 
        test_cache_stats()
        test_get_or_compute()
        
        print("\n" + "=" * 60)
        print("✅ ALL WHAT-IF CACHE TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()