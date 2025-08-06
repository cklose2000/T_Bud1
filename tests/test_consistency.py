import pytest
from datetime import datetime, timedelta

import duckdb

from trading_buddy.council.consistency import (
    calculate_consistency_index,
    compute_expectancy_with_context,
    compute_hit_rate_with_context,
    compute_stability,
)
from trading_buddy.core.ddl import initialize_database


def create_test_events_and_contexts(conn):
    """Create test data for consistency calculations."""
    initialize_database(conn)
    
    base_time = datetime.now() - timedelta(days=180)
    
    # Create events with known outcomes
    for i in range(100):
        ts = base_time + timedelta(days=i*2)
        
        # Half with positive returns, half negative
        fwd_ret = 0.01 if i % 2 == 0 else -0.01
        
        # Every 3rd event has context
        has_context = i % 3 == 0
        
        conn.execute("""
            INSERT INTO events (symbol, timeframe, event_ts, pattern, attrs, 
                              lead_bars, fwd_ret, max_dd, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            "TEST", "5m", ts, "test_pattern", {},
            10, fwd_ret, -0.005, "up" if fwd_ret > 0 else "down"
        ])
        
        if has_context:
            conn.execute("""
                INSERT INTO contexts (symbol, timeframe, event_ts, context, attrs)
                VALUES (?, ?, ?, ?, ?)
            """, ["TEST", "15m", ts, "test_context", {}])


class TestConsistencyIndex:
    def setup_method(self):
        self.conn = duckdb.connect(":memory:")
        create_test_events_and_contexts(self.conn)
    
    def teardown_method(self):
        self.conn.close()
    
    def test_calculate_consistency_index(self):
        # Test CI calculation
        ci = calculate_consistency_index(
            exp_with=0.02,
            exp_without=0.01,
            hit_rate=0.6,
            stability=0.8
        )
        
        assert ci == pytest.approx(0.0048, rel=1e-4)  # 0.01 * 0.6 * 0.8
        
        # Test with negative lift
        ci = calculate_consistency_index(
            exp_with=0.01,
            exp_without=0.02,
            hit_rate=0.6,
            stability=0.8
        )
        
        assert ci == 0.0  # Should be 0 for negative lift
    
    def test_compute_expectancy_with_context(self):
        exp_with, exp_without, n = compute_expectancy_with_context(
            self.conn,
            base_tf="5m",
            base_pattern="test_pattern",
            ctx_tf="15m",
            ctx_pattern="test_context"
        )
        
        assert n > 0  # Should have samples
        # With our test data, expectancy should differ
        assert exp_with != exp_without
    
    def test_compute_hit_rate_with_context(self):
        hit_rate = compute_hit_rate_with_context(
            self.conn,
            base_tf="5m",
            base_pattern="test_pattern",
            ctx_tf="15m",
            ctx_pattern="test_context"
        )
        
        assert 0 <= hit_rate <= 1
        # With our test data setup
        assert hit_rate == pytest.approx(0.5, abs=0.1)
    
    def test_compute_stability(self):
        # Add more recent data for stability calculation
        recent_time = datetime.now() - timedelta(days=30)
        
        for i in range(20):
            ts = recent_time + timedelta(days=i)
            fwd_ret = 0.015  # Consistent positive returns
            
            conn.execute("""
                INSERT INTO events (symbol, timeframe, event_ts, pattern, attrs,
                                  lead_bars, fwd_ret, max_dd, label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                "TEST", "5m", ts, "test_pattern", {},
                10, fwd_ret, -0.005, "up"
            ])
            
            conn.execute("""
                INSERT INTO contexts (symbol, timeframe, event_ts, context, attrs)
                VALUES (?, ?, ?, ?, ?)
            """, ["TEST", "15m", ts, "test_context", {}])
        
        stability = compute_stability(
            self.conn,
            base_tf="5m",
            base_pattern="test_pattern",
            ctx_tf="15m",
            ctx_pattern="test_context"
        )
        
        assert 0 <= stability <= 1
        # Should show some stability
        assert stability > 0.3