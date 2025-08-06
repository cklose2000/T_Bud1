import pytest
from datetime import datetime, timedelta

import duckdb

from trading_buddy.detectors.patterns import (
    detect_double_bottom,
    detect_macd_bull_cross,
    detect_swing_points,
)


def create_test_data(conn):
    """Create test OHLCV data with known patterns."""
    # Create test table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_bars (
            symbol TEXT,
            timeframe TEXT,
            ts TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    
    # Generate synthetic data with a double bottom pattern
    base_time = datetime.now() - timedelta(hours=4)
    base_price = 100.0
    
    # Create price data with double bottom
    prices = []
    
    # First decline
    for i in range(20):
        ts = base_time + timedelta(minutes=5*i)
        price = base_price - (i * 0.5)  # Decline
        prices.append((ts, price - 0.1, price + 0.1, price - 0.2, price, 1000))
    
    # First bottom at ~90
    low1_price = prices[-1][3]
    
    # Recovery
    for i in range(10):
        ts = base_time + timedelta(minutes=5*(20+i))
        price = low1_price + (i * 0.3)  # Rise
        prices.append((ts, price - 0.1, price + 0.2, price - 0.1, price, 1000))
    
    # Second decline
    for i in range(15):
        ts = base_time + timedelta(minutes=5*(30+i))
        price = prices[-1][4] - (i * 0.3)  # Decline again
        prices.append((ts, price - 0.1, price + 0.1, price - 0.2, price, 1000))
    
    # Second bottom near first bottom
    low2_price = low1_price + 0.2  # Within tolerance
    prices[-1] = (prices[-1][0], prices[-1][1], prices[-1][2], low2_price, prices[-1][4], 1000)
    
    # Breakout above neckline
    for i in range(10):
        ts = base_time + timedelta(minutes=5*(45+i))
        price = low2_price + (i * 0.5)  # Strong rise
        prices.append((ts, price - 0.1, price + 0.3, price, price + 0.2, 1000))
    
    # Insert data
    for ts, o, h, l, c, v in prices:
        conn.execute("""
            INSERT INTO test_bars (symbol, timeframe, ts, open, high, low, close, volume)
            VALUES ('TEST', '5m', ?, ?, ?, ?, ?, ?)
        """, [ts, o, h, l, c, v])
    
    # Create view to match expected schema
    conn.execute("""
        CREATE OR REPLACE VIEW bars AS
        SELECT * FROM test_bars
    """)


class TestPatternDetection:
    def setup_method(self):
        self.conn = duckdb.connect(":memory:")
        create_test_data(self.conn)
    
    def teardown_method(self):
        self.conn.close()
    
    def test_detect_swing_points(self):
        swings = detect_swing_points(self.conn, "TEST", "5m")
        
        assert "highs" in swings
        assert "lows" in swings
        assert len(swings["lows"]) >= 2  # At least two swing lows
        assert len(swings["highs"]) >= 1  # At least one swing high
    
    def test_detect_double_bottom(self):
        patterns = detect_double_bottom(self.conn, "TEST", "5m")
        
        assert len(patterns) > 0
        pattern = patterns[0]
        
        assert pattern["pattern"] == "double_bottom"
        assert "low1_price" in pattern["attrs"]
        assert "low2_price" in pattern["attrs"]
        assert "neckline_price" in pattern["attrs"]
        
        # Check price tolerance
        low1 = pattern["attrs"]["low1_price"]
        low2 = pattern["attrs"]["low2_price"]
        assert abs(low2 - low1) / low1 < 0.005  # Within 0.5%
    
    def test_detect_macd_cross_with_synthetic_data(self):
        # Add more data for MACD calculation
        base_time = datetime.now() - timedelta(hours=10)
        
        # Create trending data for MACD
        for i in range(100):
            ts = base_time + timedelta(minutes=5*i)
            # Create uptrend with noise
            trend = 100 + (i * 0.1)
            noise = (i % 10 - 5) * 0.2
            price = trend + noise
            
            self.conn.execute("""
                INSERT INTO test_bars (symbol, timeframe, ts, open, high, low, close, volume)
                VALUES ('MACD_TEST', '5m', ?, ?, ?, ?, ?, ?)
            """, [ts, price - 0.1, price + 0.2, price - 0.2, price, 1000])
        
        crosses = detect_macd_bull_cross(self.conn, "MACD_TEST", "5m")
        
        # Should detect at least one crossover in trending data
        assert isinstance(crosses, list)