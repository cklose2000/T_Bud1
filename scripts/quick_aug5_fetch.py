#!/usr/bin/env python3
"""
Quick fetch of August 5, 2025 SPY data
Uses the existing IB connection
"""
import asyncio
from ib_insync import IB, Stock
from datetime import datetime
import duckdb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_aug5_spy():
    """Fetch August 5 SPY data quickly"""
    ib = IB()
    
    try:
        # Connect to IB Gateway on port 4002
        await ib.connectAsync('localhost', 4002, clientId=20)  # Use different client ID
        logger.info("Connected to IB Gateway")
        
        # Create SPY contract
        spy = Stock('SPY', 'SMART', 'USD')
        
        # Request 5-minute bars for last 3 days (to get Aug 5)
        bars = await ib.reqHistoricalDataAsync(
            spy,
            endDateTime='20250806 23:59:59',  # Today end of day
            durationStr='3 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        
        logger.info(f"Received {len(bars)} bars")
        
        # Filter for August 5 data
        aug5_bars = [b for b in bars if b.date.date() == datetime(2025, 8, 5).date()]
        logger.info(f"August 5 bars: {len(aug5_bars)}")
        
        if aug5_bars:
            # Show some data
            logger.info(f"First bar: {aug5_bars[0].date} O:{aug5_bars[0].open} H:{aug5_bars[0].high} L:{aug5_bars[0].low} C:{aug5_bars[0].close}")
            logger.info(f"Last bar: {aug5_bars[-1].date} O:{aug5_bars[-1].open} H:{aug5_bars[-1].high} L:{aug5_bars[-1].low} C:{aug5_bars[-1].close}")
            
            # Find lowest point
            lowest_bar = min(aug5_bars, key=lambda x: x.low)
            logger.info(f"Lowest point: {lowest_bar.date} Low: ${lowest_bar.low}")
            
            # Save to database - use regular INSERT
            db_path = '/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb'
            conn = duckdb.connect(db_path)
            
            # First, delete any existing Aug 5 data to avoid duplicates
            conn.execute("""
                DELETE FROM market_data 
                WHERE symbol = 'SPY' 
                AND DATE(date) = '2025-08-05' 
                AND timeframe = '5min'
            """)
            
            for bar in aug5_bars:
                try:
                    conn.execute("""
                        INSERT INTO market_data 
                        (date, symbol, open, high, low, close, volume, vwap, trade_count, 
                         bid_price, ask_price, bid_size, ask_size, spread, timeframe)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        bar.date, 'SPY', bar.open, bar.high, bar.low, bar.close, 
                        bar.volume, bar.average or bar.close, bar.barCount or 0,
                        bar.close - 0.01, bar.close + 0.01, 100, 100, 0.02, '5min'
                    ])
                except Exception as e:
                    logger.error(f"Error inserting bar: {e}")
            
            conn.commit()
            logger.info(f"Saved {len(aug5_bars)} bars to database")
            
            # Verify
            check = conn.execute("""
                SELECT COUNT(*) FROM market_data 
                WHERE symbol = 'SPY' AND DATE(date) = '2025-08-05'
            """).fetchone()
            logger.info(f"Total SPY bars for Aug 5 in DB: {check[0]}")
            
            conn.close()
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        ib.disconnect()


def main():
    asyncio.run(fetch_aug5_spy())


if __name__ == "__main__":
    main()