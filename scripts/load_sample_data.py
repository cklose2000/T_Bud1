#!/usr/bin/env python3
"""
Load sample OHLCV data for testing.
Creates synthetic data if no real data is available.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
import random
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.duck import DuckDBManager


def generate_synthetic_ohlcv(symbol: str, days: int = 365):
    """Generate synthetic OHLCV data with realistic patterns."""
    data = []
    
    # Base parameters
    base_price = {"SPY": 400.0, "QQQ": 350.0, "NVDA": 500.0}.get(symbol, 100.0)
    volatility = {"SPY": 0.01, "QQQ": 0.015, "NVDA": 0.025}.get(symbol, 0.02)
    trend = 0.0002  # Slight upward bias
    
    # Generate data for multiple timeframes
    timeframes = [
        ("1m", 1),
        ("5m", 5),
        ("15m", 15),
        ("60m", 60),
        ("1d", 1440),
    ]
    
    end_time = datetime.now()
    
    for tf_name, tf_minutes in timeframes:
        current_price = base_price
        bars_per_day = 1440 // tf_minutes
        total_bars = bars_per_day * days
        
        # Skip 1d for now if too many bars
        if tf_name == "1d":
            total_bars = days
        
        for i in range(total_bars):
            # Calculate timestamp
            ts = end_time - timedelta(minutes=tf_minutes * (total_bars - i))
            
            # Add some intraday seasonality
            hour = ts.hour
            intraday_factor = 1.0
            if 9 <= hour <= 16:  # Market hours
                intraday_factor = 1.2
            
            # Generate price movement
            returns = random.gauss(trend, volatility * intraday_factor)
            
            # Add some mean reversion
            mean_reversion = -0.001 * ((current_price / base_price) - 1.0)
            returns += mean_reversion
            
            # Calculate OHLC
            open_price = current_price
            close_price = current_price * (1 + returns)
            
            # High and low with realistic wicks
            wick_size = abs(returns) * random.uniform(0.5, 1.5)
            high_price = max(open_price, close_price) + wick_size * current_price
            low_price = min(open_price, close_price) - wick_size * current_price
            
            # Volume with some randomness
            base_volume = 1000000 if tf_name == "5m" else 100000
            volume = base_volume * random.uniform(0.5, 2.0) * intraday_factor
            
            # Add some patterns occasionally
            if random.random() < 0.01:  # 1% chance of larger move
                multiplier = random.choice([2, 3, -2, -3])
                close_price = open_price * (1 + returns * multiplier)
                high_price = max(open_price, close_price) * 1.01
                low_price = min(open_price, close_price) * 0.99
                volume *= 3
            
            data.append({
                "symbol": symbol,
                "timeframe": tf_name,
                "ts": ts,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": int(volume),
            })
            
            current_price = close_price
        
        # Reset price for next timeframe
        current_price = base_price
    
    return data


def main():
    print("Loading sample data into Trading Buddy database...")
    
    with DuckDBManager() as db:
        # Create raw data table
        print("Creating raw data table...")
        db.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_ohlcv (
                symbol TEXT,
                timeframe TEXT,
                ts TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, timeframe, ts)
            )
        """)
        
        # Generate and load data for each symbol
        symbols = ["SPY", "QQQ", "NVDA"]
        
        for symbol in symbols:
            print(f"\nGenerating data for {symbol}...")
            data = generate_synthetic_ohlcv(symbol, days=365)
            
            print(f"  Generated {len(data)} bars")
            
            # Bulk insert
            for row in data:
                db.conn.execute("""
                    INSERT OR REPLACE INTO raw_ohlcv 
                    (symbol, timeframe, ts, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row["symbol"], row["timeframe"], row["ts"],
                    row["open"], row["high"], row["low"], row["close"], row["volume"]
                ])
            
            print(f"  Loaded {symbol} data")
        
        db.conn.commit()
        
        # Verify data
        count = db.conn.execute("SELECT COUNT(*) FROM raw_ohlcv").fetchone()[0]
        print(f"\nTotal bars loaded: {count}")
        
        # Show sample
        print("\nSample data:")
        samples = db.conn.execute("""
            SELECT symbol, timeframe, ts, close, volume
            FROM raw_ohlcv
            WHERE timeframe = '5m'
            ORDER BY ts DESC
            LIMIT 5
        """).fetchall()
        
        for s in samples:
            print(f"  {s[0]} {s[1]} {s[2]} close={s[3]} vol={s[4]}")
        
        print("\nSample data loaded successfully!")
        print("\nNow run: make init-db")


if __name__ == "__main__":
    main()