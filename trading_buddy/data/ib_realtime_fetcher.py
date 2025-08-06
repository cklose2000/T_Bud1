"""
IB Gateway Real-Time Data Fetcher
Fetches historical and real-time data from Interactive Brokers
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import time
from collections import deque
import threading

try:
    from ib_insync import IB, Stock, util
except ImportError:
    raise ImportError("Please install ib_insync: pip install ib_insync")

import duckdb
from duckdb import DuckDBPyConnection

from trading_buddy.core.config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for IB API requests"""
    def __init__(self, requests_per_period: int = 60, period_seconds: int = 600):
        self.requests_per_period = requests_per_period
        self.period_seconds = period_seconds
        self.tokens = requests_per_period
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
    def acquire(self, timeout: int = 60) -> bool:
        """Acquire a token, blocking if necessary"""
        deadline = time.time() + timeout
        
        while True:
            with self.lock:
                # Refill tokens based on time passed
                now = time.time()
                elapsed = now - self.last_refill
                new_tokens = elapsed * (self.requests_per_period / self.period_seconds)
                self.tokens = min(self.requests_per_period, self.tokens + new_tokens)
                self.last_refill = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                    
            if time.time() >= deadline:
                return False
                
            time.sleep(0.1)


class IBRealtimeFetcher:
    """
    Real-time data fetcher for Interactive Brokers
    Handles both historical backfill and real-time streaming
    """
    
    def __init__(self, db_path: str, ib_host: str = "localhost", ib_port: int = 7497, client_id: int = 10):
        self.db_path = db_path
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.client_id = client_id
        
        self.ib: Optional[IB] = None
        self.rate_limiter = RateLimiter()
        self.subscriptions: Dict[str, Dict] = {}
        self.running = False
        
        # Symbols to track
        self.symbols = ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT', 'TSLA', 'GOOGL', 'META', 'AMZN']
        self.timeframes = ['5 mins', '15 mins', '1 hour']
        
        # Data queue for batch inserts
        self.data_queue: deque = deque()
        self.batch_size = 100
        
    async def connect(self) -> bool:
        """Connect to IB Gateway"""
        try:
            self.ib = IB()
            await self.ib.connectAsync(self.ib_host, self.ib_port, clientId=self.client_id)
            logger.info(f"Connected to IB Gateway at {self.ib_host}:{self.ib_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IB Gateway"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        duration: str = "1 D"
    ) -> Optional[List]:
        """Fetch historical data for a symbol"""
        if not self.rate_limiter.acquire():
            logger.warning(f"Rate limit reached, skipping {symbol} {timeframe}")
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=timeframe,
                whatToShow='TRADES',
                useRTH=False,  # Include extended hours
                formatDate=1
            )
            
            if bars:
                logger.info(f"Fetched {len(bars)} bars for {symbol} {timeframe}")
                return bars
            else:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _convert_timeframe(self, ib_timeframe: str) -> str:
        """Convert IB timeframe to database format"""
        mapping = {
            '5 mins': '5min',
            '15 mins': '15min',
            '1 hour': '60min',
            '1 day': '1day'
        }
        return mapping.get(ib_timeframe, ib_timeframe)
    
    async def backfill_missing_data(self, conn: DuckDBPyConnection):
        """Backfill missing data for all symbols"""
        logger.info("Starting backfill of missing data...")
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                # Check latest data in database
                tf_db = self._convert_timeframe(timeframe)
                
                result = conn.execute(f"""
                    SELECT MAX(date) as latest
                    FROM market_data
                    WHERE symbol = ? AND timeframe = ?
                """, [symbol, tf_db]).fetchone()
                
                latest_date = result[0] if result and result[0] else None
                
                # Calculate duration needed
                if latest_date:
                    days_missing = (datetime.now() - latest_date).days
                    if days_missing > 0:
                        duration = f"{min(days_missing + 1, 30)} D"
                    else:
                        continue  # Data is up to date
                else:
                    duration = "30 D"  # Default backfill
                
                logger.info(f"Backfilling {symbol} {timeframe} for {duration}")
                
                # Fetch historical data
                bars = await self.get_historical_data(symbol, timeframe, duration)
                
                if bars:
                    # Convert to database format
                    for bar in bars:
                        self.data_queue.append({
                            'date': bar.date,
                            'symbol': symbol,
                            'timeframe': tf_db,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'average': bar.average,
                            'barCount': bar.barCount
                        })
                    
                    # Flush if queue is large
                    if len(self.data_queue) >= self.batch_size:
                        self._flush_data_queue(conn)
                
                # Small delay between requests
                await asyncio.sleep(0.5)
        
        # Flush remaining data
        if self.data_queue:
            self._flush_data_queue(conn)
        
        logger.info("Backfill completed")
    
    def _flush_data_queue(self, conn: DuckDBPyConnection):
        """Flush data queue to database"""
        if not self.data_queue:
            return
        
        data_list = list(self.data_queue)
        self.data_queue.clear()
        
        # Batch insert
        for data in data_list:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO market_data 
                    (date, symbol, open, high, low, close, volume, vwap, trade_count, 
                     bid_price, ask_price, bid_size, ask_size, spread, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    data['date'], data['symbol'], data['open'], data['high'],
                    data['low'], data['close'], data['volume'], 
                    data.get('average', data['close']),  # Use average as vwap
                    data.get('barCount', 0),  # trade_count
                    data['close'] - 0.01,  # bid_price (approximation)
                    data['close'] + 0.01,  # ask_price (approximation)
                    100,  # bid_size (default)
                    100,  # ask_size (default)
                    0.02,  # spread (default)
                    data['timeframe']
                ])
            except Exception as e:
                logger.error(f"Error inserting data: {e}")
        
        conn.commit()
        logger.info(f"Flushed {len(data_list)} records to database")
    
    async def start_realtime_streaming(self, conn: DuckDBPyConnection):
        """Start real-time data streaming for all symbols"""
        logger.info("Starting real-time data streaming...")
        
        for symbol in self.symbols:
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                
                # Request 5-second bars (real-time)
                bars = await self.ib.reqRealTimeBarsAsync(
                    contract,
                    barSize=5,
                    whatToShow='TRADES',
                    useRTH=False
                )
                
                # Set up callback for new bars
                bars.updateEvent += lambda bars, symbol=symbol: self._on_bar_update(bars, symbol, conn)
                
                self.subscriptions[symbol] = {
                    'contract': contract,
                    'bars': bars
                }
                
                logger.info(f"Started real-time streaming for {symbol}")
                
            except Exception as e:
                logger.error(f"Error starting real-time stream for {symbol}: {e}")
        
        # Keep running
        self.running = True
        while self.running:
            await asyncio.sleep(1)
            
            # Periodic flush of data queue
            if len(self.data_queue) >= 10:
                self._flush_data_queue(conn)
    
    def _on_bar_update(self, bars, symbol: str, conn: DuckDBPyConnection):
        """Handle real-time bar updates"""
        if not bars:
            return
        
        latest_bar = bars[-1]
        
        # Add to queue for 5-second data
        self.data_queue.append({
            'date': latest_bar.time,
            'symbol': symbol,
            'timeframe': '5s',
            'open': latest_bar.open,
            'high': latest_bar.high,
            'low': latest_bar.low,
            'close': latest_bar.close,
            'volume': latest_bar.volume,
            'average': latest_bar.average,
            'barCount': latest_bar.count
        })
        
        # Also aggregate to 5-minute bars
        # This would need more sophisticated logic to properly aggregate
        
        logger.debug(f"New bar for {symbol}: {latest_bar.time} close={latest_bar.close}")
    
    def stop(self):
        """Stop real-time streaming"""
        self.running = False
        
        # Cancel all subscriptions
        for symbol, sub in self.subscriptions.items():
            if 'bars' in sub:
                self.ib.cancelRealTimeBars(sub['bars'])
        
        self.subscriptions.clear()
        logger.info("Stopped real-time streaming")