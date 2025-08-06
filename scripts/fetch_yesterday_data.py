#!/usr/bin/env python3
"""
Fetch yesterday's (August 5, 2025) data from IB Gateway
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.data.ib_realtime_fetcher import IBRealtimeFetcher
from trading_buddy.core.hybrid_data_manager import HybridDataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def fetch_august_5_data():
    """Fetch August 5, 2025 data from IB Gateway"""
    
    # IB database path
    ib_db_path = "/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb"
    
    # Create fetcher
    fetcher = IBRealtimeFetcher(
        db_path=ib_db_path,
        ib_host="localhost",
        ib_port=4002,  # IB Gateway is on port 4002
        client_id=10
    )
    
    # Focus on SPY for the double bottom
    fetcher.symbols = ['SPY', 'QQQ', 'NVDA']
    fetcher.timeframes = ['5 mins', '15 mins', '1 hour']
    
    logger.info("Connecting to IB Gateway...")
    connected = await fetcher.connect()
    
    if not connected:
        logger.error("Failed to connect to IB Gateway!")
        logger.error("Please ensure:")
        logger.error("1. IB Gateway or TWS is running")
        logger.error("2. API connections are enabled (port 7497)")
        logger.error("3. Settings -> API -> Settings -> Enable ActiveX and Socket Clients is checked")
        return
    
    logger.info("Connected to IB Gateway successfully!")
    
    try:
        # Get August 5 data specifically
        yesterday = datetime(2025, 8, 5)
        today = datetime(2025, 8, 6)
        
        # Calculate how many days back we need
        days_back = (today - yesterday).days + 1  # 2 days to cover Aug 5
        
        for symbol in fetcher.symbols:
            logger.info(f"\nFetching {symbol} data for August 5, 2025...")
            
            # Fetch 5-minute data
            bars_5min = await fetcher.get_historical_data(
                symbol=symbol,
                timeframe='5 mins',
                duration=f'{days_back} D'
            )
            
            if bars_5min:
                logger.info(f"  Retrieved {len(bars_5min)} 5-minute bars")
                
                # Check if we got August 5 data
                aug5_bars = [b for b in bars_5min if b.date.date() == yesterday.date()]
                logger.info(f"  August 5 bars: {len(aug5_bars)}")
                
                if aug5_bars:
                    # Show price range for August 5
                    lows = [b.low for b in aug5_bars]
                    highs = [b.high for b in aug5_bars]
                    logger.info(f"  August 5 price range: ${min(lows):.2f} - ${max(highs):.2f}")
            
            # Also fetch 15-minute data
            bars_15min = await fetcher.get_historical_data(
                symbol=symbol,
                timeframe='15 mins',
                duration=f'{days_back} D'
            )
            
            if bars_15min:
                logger.info(f"  Retrieved {len(bars_15min)} 15-minute bars")
            
            # Small delay between symbols
            await asyncio.sleep(1)
        
        # Save data to database
        logger.info("\nSaving data to database...")
        import duckdb
        conn = duckdb.connect(ib_db_path)
        
        # Flush any queued data
        if fetcher.data_queue:
            fetcher._flush_data_queue(conn)
        
        # Check what we have now
        result = conn.execute("""
            SELECT 
                symbol,
                timeframe,
                MIN(date) as earliest,
                MAX(date) as latest,
                COUNT(*) as count
            FROM market_data
            WHERE date >= '2025-08-05'
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """).fetchall()
        
        logger.info("\nUpdated data coverage:")
        for row in result:
            logger.info(f"  {row[0]} {row[1]}: {row[2]} to {row[3]} ({row[4]} bars)")
        
        conn.close()
        
    finally:
        fetcher.disconnect()
    
    logger.info("\n✅ Data fetch complete!")
    logger.info("Now run: python3 scripts/backfill_aug5_patterns.py")


def main():
    # Check if IB Gateway is accessible
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("localhost", 4002))
    sock.close()
    
    if result != 0:
        logger.error("Cannot connect to IB Gateway on port 4002!")
        logger.error("Please ensure IB Gateway is running and API connections are enabled.")
        return 1
    
    logger.info("IB Gateway is accessible on port 4002")
    
    # Run async fetch
    asyncio.run(fetch_august_5_data())
    return 0


if __name__ == "__main__":
    sys.exit(main())