import pytest
from datetime import datetime

import duckdb

from trading_buddy.council.vote import (
    calculate_council_score,
    check_current_context,
    council_vote,
)
from trading_buddy.core.ddl import initialize_database
from trading_buddy.schemas.council import CIBreakdown
from trading_buddy.schemas.hypothesis import (
    CrossTimeframeVote,
    Hypothesis,
    PatternConfig,
)


def create_test_council_data(conn):
    """Create test data for council voting."""
    initialize_database(conn)
    
    # Create test bars view
    conn.execute("""
        CREATE OR REPLACE VIEW bars AS
        SELECT 
            'TEST' as symbol,
            '5m' as timeframe,
            CURRENT_TIMESTAMP as ts,
            100.0 as open,
            101.0 as high,
            99.0 as low,
            100.5 as close,
            1000.0 as volume
    """)
    
    # Add test events with outcomes
    now = datetime.now()
    
    for i in range(50):
        conn.execute("""
            INSERT INTO events (symbol, timeframe, event_ts, pattern, attrs,
                              lead_bars, fwd_ret, max_dd, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            "TEST", "5m", now, "double_bottom_macd_bull", 
            {"neckline_price": 100.0, "low2_price": 99.0},
            10, 0.02, -0.01, "up"
        ])
    
    # Add test contexts
    conn.execute("""
        INSERT INTO contexts (symbol, timeframe, event_ts, context, attrs)
        VALUES (?, ?, ?, ?, ?)
    """, ["TEST", "15m", now, "trend_up_ema20", {}])
    
    # Add consistency matrix entry
    conn.execute("""
        INSERT INTO consistency_matrix 
        (base_tf, base_pattern, ctx_tf, ctx_pattern, n, exp_lift, hit_rate, stability, ci)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        "5m", "double_bottom_macd_bull", "15m", "trend_up_ema20",
        50, 0.01, 0.65, 0.8, 0.0052
    ])


class TestCouncilVoting:
    def setup_method(self):
        self.conn = duckdb.connect(":memory:")
        create_test_council_data(self.conn)
    
    def teardown_method(self):
        self.conn.close()
    
    def test_calculate_council_score(self):
        ci_breakdown = [
            CIBreakdown(
                timeframe="15m",
                pattern="trend_up_ema20",
                ci=0.8,
                present=True,
                exp_lift=0.01
            ),
            CIBreakdown(
                timeframe="60m",
                pattern="regime_trend",
                ci=0.6,
                present=False,
                exp_lift=None
            ),
        ]
        
        score = calculate_council_score(ci_breakdown)
        
        # (0.8 * 1.0 + 0.6 * 0.0) / (0.8 + 0.6) = 0.8 / 1.4
        assert score == pytest.approx(0.571, rel=0.01)
    
    def test_check_current_context(self):
        now = datetime.now()
        
        # Should find the context we inserted
        present = check_current_context(
            self.conn, "TEST", "15m", "trend_up_ema20", now
        )
        assert present is True
        
        # Should not find non-existent context
        present = check_current_context(
            self.conn, "TEST", "60m", "regime_chop", now
        )
        assert present is False
    
    def test_council_vote(self):
        hypothesis = Hypothesis(
            symbol="TEST",
            base_timeframe="5m",
            patterns=[
                PatternConfig(name="double_bottom_macd_bull", params={})
            ],
            cross_timeframe_votes=[
                CrossTimeframeVote(
                    tf="15m",
                    validators=[{"name": "trend_up_ema20"}],
                    lead_window="45m"
                )
            ],
            now_ts=datetime.now()
        )
        
        response = council_vote(self.conn, hypothesis)
        
        assert response.n == 50
        assert response.median_fwd_ret == pytest.approx(0.02, rel=0.01)
        assert response.hit_rate == 1.0  # All test events are "up"
        assert response.council_score > 0
        assert len(response.ci_breakdown) > 0
        assert response.plan.size > 0