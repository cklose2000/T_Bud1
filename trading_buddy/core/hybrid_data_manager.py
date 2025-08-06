"""
Hybrid Data Manager
Seamlessly manages data access between IB real-time data and DuckDB storage
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import threading
from collections import deque
import time

import duckdb
from duckdb import DuckDBPyConnection

from trading_buddy.core.config import settings
from trading_buddy.core.duck import DuckDBManager

logger = logging.getLogger(__name__)


class DataCache:
    """Thread-safe cache for frequently accessed data"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[any, float]] = {}
        self.access_order: deque = deque()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[any]:
        """Get value from cache if not expired"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (most recently used)
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return value
                else:
                    # Expired
                    del self.cache[key]
                    self.access_order.remove(key)
        return None
    
    def set(self, key: str, value: any):
        """Set value in cache"""
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = (value, time.time())
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


class HybridDataManager(DuckDBManager):
    """
    Enhanced DuckDB manager that provides seamless access to both
    historical and real-time data from IB Gateway
    """
    
    def __init__(
        self, 
        tb_db_path: Optional[Path] = None,
        ib_db_path: str = "/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb"
    ):
        super().__init__(tb_db_path)
        self.ib_db_path = Path(ib_db_path)
        self.cache = DataCache()
        
        # Track data freshness
        self.data_freshness: Dict[str, Dict[str, datetime]] = {}
        self.freshness_threshold = timedelta(minutes=5)
        
    @property
    def conn(self) -> DuckDBPyConnection:
        """Get connection with IB database attached"""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            
            # Attach IB database if it exists
            if self.ib_db_path.exists():
                try:
                    self._conn.execute(f"""
                        ATTACH DATABASE '{self.ib_db_path}' AS ib_data (READ_ONLY)
                    """)
                    logger.info(f"Attached IB database: {self.ib_db_path}")
                except Exception as e:
                    # Database might already be attached or locked
                    logger.debug(f"Could not attach IB database: {e}")
                    
        return self._conn
    
    def get_latest_data(
        self, 
        symbol: str, 
        timeframe: str = "5min",
        bars: int = 100
    ) -> List[Dict]:
        """
        Get latest data for a symbol, using cache if available
        """
        cache_key = f"{symbol}_{timeframe}_{bars}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Query database
        query = f"""
        SELECT 
            ts as timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM bars
        WHERE symbol = ? AND timeframe = ?
        ORDER BY ts DESC
        LIMIT ?
        """
        
        result = self.conn.execute(query, [symbol, timeframe, bars]).fetchall()
        
        # Convert to list of dicts
        data = [
            {
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5]
            }
            for row in reversed(result)  # Oldest first
        ]
        
        # Cache the result
        self.cache.set(cache_key, data)
        
        return data
    
    def get_data_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> List[Dict]:
        """Get data for a specific date range"""
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        
        query = f"""
        SELECT 
            ts as timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM bars
        WHERE symbol = ? 
        AND timeframe = ?
        AND ts >= ?
        AND ts <= ?
        ORDER BY ts
        """
        
        result = self.conn.execute(
            query, 
            [symbol, timeframe, start_date, end_date]
        ).fetchall()
        
        return [
            {
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5]
            }
            for row in result
        ]
    
    def check_data_freshness(self, symbol: str, timeframe: str) -> Dict:
        """Check how fresh the data is for a symbol/timeframe"""
        query = f"""
        SELECT 
            MIN(ts) as earliest,
            MAX(ts) as latest,
            COUNT(*) as bar_count
        FROM bars
        WHERE symbol = ? AND timeframe = ?
        """
        
        result = self.conn.execute(query, [symbol, timeframe]).fetchone()
        
        if result and result[1]:
            latest = result[1]
            gap = datetime.now() - latest
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "earliest": result[0],
                "latest": latest,
                "bar_count": result[2],
                "gap_minutes": gap.total_seconds() / 60,
                "is_fresh": gap < self.freshness_threshold
            }
        else:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "earliest": None,
                "latest": None,
                "bar_count": 0,
                "gap_minutes": float('inf'),
                "is_fresh": False
            }
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols"""
        cache_key = "available_symbols"
        
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        result = self.conn.execute("""
            SELECT DISTINCT symbol 
            FROM bars 
            ORDER BY symbol
        """).fetchall()
        
        symbols = [row[0] for row in result]
        self.cache.set(cache_key, symbols)
        
        return symbols
    
    def get_timeframes(self, symbol: str) -> List[str]:
        """Get available timeframes for a symbol"""
        result = self.conn.execute("""
            SELECT DISTINCT timeframe, COUNT(*) as cnt
            FROM bars
            WHERE symbol = ?
            GROUP BY timeframe
            ORDER BY timeframe
        """, [symbol]).fetchall()
        
        return [(row[0], row[1]) for row in result]
    
    @contextmanager
    def streaming_cursor(
        self, 
        symbol: str, 
        timeframe: str,
        start_date: Optional[str] = None
    ):
        """
        Context manager for streaming large amounts of data
        """
        where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
        if start_date:
            where_clause += f" AND ts >= '{start_date}'"
        
        query = f"""
        SELECT 
            ts as timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM bars
        {where_clause}
        ORDER BY ts
        """
        
        # Use a separate connection for streaming
        stream_conn = duckdb.connect(str(self.db_path))
        
        # Attach IB database
        if self.ib_db_path.exists():
            try:
                stream_conn.execute(f"""
                    ATTACH DATABASE '{self.ib_db_path}' AS ib_data (READ_ONLY)
                """)
            except:
                pass
        
        cursor = stream_conn.execute(query)
        
        try:
            yield cursor
        finally:
            stream_conn.close()
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cleared data cache")
    
    def get_latest_patterns(
        self, 
        hours: int = 24,
        pattern_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get recently detected patterns"""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        where_clause = f"WHERE event_ts >= '{since}'"
        if pattern_types:
            patterns = "','".join(pattern_types)
            where_clause += f" AND pattern IN ('{patterns}')"
        
        query = f"""
        SELECT 
            symbol,
            timeframe,
            event_ts,
            pattern,
            attrs,
            fwd_ret,
            label
        FROM events
        {where_clause}
        ORDER BY event_ts DESC
        """
        
        result = self.conn.execute(query).fetchall()
        
        return [
            {
                "symbol": row[0],
                "timeframe": row[1],
                "event_ts": row[2],
                "pattern": row[3],
                "attrs": row[4],
                "fwd_ret": row[5],
                "label": row[6]
            }
            for row in result
        ]