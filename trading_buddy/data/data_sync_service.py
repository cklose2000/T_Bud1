"""
Data Synchronization Service
Manages continuous data updates between IB Gateway and DuckDB
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import threading
import time

import duckdb
from duckdb import DuckDBPyConnection

from trading_buddy.core.config import settings
from trading_buddy.data.ib_realtime_fetcher import IBRealtimeFetcher
from trading_buddy.detectors.patterns import (
    detect_double_bottom,
    detect_macd_bull_cross,
    detect_compound_pattern
)
from trading_buddy.detectors.contexts import (
    detect_trend_context,
    detect_regime
)
from trading_buddy.detectors.outcomes import update_event_outcomes

logger = logging.getLogger(__name__)


class DataSyncService:
    """
    Service for continuous data synchronization and pattern detection
    """
    
    def __init__(
        self,
        ib_db_path: str = "/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb",
        tb_db_path: str = "./data/trading_buddy.duckdb",
        ib_host: str = "localhost",
        ib_port: int = 7497,
        client_id: int = 10
    ):
        self.ib_db_path = ib_db_path
        self.tb_db_path = tb_db_path
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.client_id = client_id
        
        # Components
        self.fetcher = IBRealtimeFetcher(
            db_path=ib_db_path,
            ib_host=ib_host,
            ib_port=ib_port,
            client_id=client_id
        )
        
        # State
        self.running = False
        self.last_sync_times: Dict[str, datetime] = {}
        self.pattern_detection_enabled = True
        self.sync_interval = 60  # seconds
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def get_ib_connection(self) -> DuckDBPyConnection:
        """Get connection to IB database"""
        return duckdb.connect(self.ib_db_path)
    
    def get_tb_connection(self) -> DuckDBPyConnection:
        """Get connection to Trading Buddy database with IB attached"""
        conn = duckdb.connect(self.tb_db_path)
        
        # Attach IB database
        try:
            conn.execute(f"""
                ATTACH DATABASE '{self.ib_db_path}' AS ib_data (READ_ONLY)
            """)
        except:
            pass  # Already attached
            
        return conn
    
    async def check_data_gaps(self) -> Dict[str, Dict]:
        """Check for data gaps that need to be filled"""
        gaps = {}
        
        with self.get_ib_connection() as conn:
            for symbol in self.fetcher.symbols:
                gaps[symbol] = {}
                
                for timeframe in ['5min', '15min', '60min']:
                    result = conn.execute(f"""
                        SELECT 
                            MAX(date) as latest,
                            MIN(date) as earliest,
                            COUNT(*) as count
                        FROM market_data
                        WHERE symbol = ? AND timeframe = ?
                    """, [symbol, timeframe]).fetchone()
                    
                    if result and result[0]:
                        latest = result[0]
                        gap_hours = (datetime.now() - latest).total_seconds() / 3600
                        
                        gaps[symbol][timeframe] = {
                            'latest': latest,
                            'gap_hours': gap_hours,
                            'needs_update': gap_hours > 0.1  # More than 6 minutes
                        }
                    else:
                        gaps[symbol][timeframe] = {
                            'latest': None,
                            'gap_hours': float('inf'),
                            'needs_update': True
                        }
        
        return gaps
    
    async def sync_data(self):
        """Main sync loop"""
        logger.info("Starting data sync service...")
        
        # Connect to IB Gateway
        connected = await self.fetcher.connect()
        if not connected:
            logger.error("Failed to connect to IB Gateway")
            return
        
        try:
            # Initial backfill
            logger.info("Performing initial backfill...")
            with self.get_ib_connection() as conn:
                await self.fetcher.backfill_missing_data(conn)
            
            # Check data gaps
            gaps = await self.check_data_gaps()
            for symbol, timeframes in gaps.items():
                for tf, info in timeframes.items():
                    if info['needs_update']:
                        logger.info(f"{symbol} {tf}: {info['gap_hours']:.1f} hours behind")
            
            # Start real-time streaming
            logger.info("Starting real-time data streaming...")
            
            # Run pattern detection in background
            pattern_task = asyncio.create_task(self._pattern_detection_loop())
            
            # Start real-time streaming (this blocks)
            with self.get_ib_connection() as conn:
                await self.fetcher.start_realtime_streaming(conn)
                
        except Exception as e:
            logger.error(f"Error in sync loop: {e}")
            
        finally:
            self.fetcher.disconnect()
            if 'pattern_task' in locals():
                pattern_task.cancel()
    
    async def _pattern_detection_loop(self):
        """Background loop for pattern detection"""
        while self.running:
            try:
                # Run pattern detection every minute
                await asyncio.sleep(60)
                
                if self.pattern_detection_enabled:
                    await self._detect_patterns()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pattern detection: {e}")
    
    async def _detect_patterns(self):
        """Detect patterns in recent data"""
        logger.info("Running pattern detection...")
        
        with self.get_tb_connection() as conn:
            # Look for patterns in last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            for symbol in ['SPY', 'QQQ', 'NVDA']:
                try:
                    # Detect 5m patterns
                    double_bottoms = detect_double_bottom(
                        conn, symbol, "5m",
                        start_ts=start_time.isoformat(),
                        end_ts=end_time.isoformat()
                    )
                    
                    for event in double_bottoms:
                        # Check if already exists
                        exists = conn.execute("""
                            SELECT 1 FROM events
                            WHERE symbol = ? AND timeframe = ? 
                            AND event_ts = ? AND pattern = ?
                        """, [symbol, "5m", event["event_ts"], event["pattern"]]).fetchone()
                        
                        if not exists:
                            conn.execute("""
                                INSERT INTO events 
                                (symbol, timeframe, event_ts, pattern, attrs)
                                VALUES (?, ?, ?, ?, ?)
                            """, [symbol, "5m", event["event_ts"], event["pattern"], event["attrs"]])
                            
                            logger.info(f"New pattern detected: {symbol} {event['pattern']} at {event['event_ts']}")
                    
                    # Detect MACD crosses
                    macd_crosses = detect_macd_bull_cross(
                        conn, symbol, "5m",
                        start_ts=start_time.isoformat(),
                        end_ts=end_time.isoformat()
                    )
                    
                    for event in macd_crosses:
                        exists = conn.execute("""
                            SELECT 1 FROM events
                            WHERE symbol = ? AND timeframe = ? 
                            AND event_ts = ? AND pattern = ?
                        """, [symbol, "5m", event["event_ts"], event["pattern"]]).fetchone()
                        
                        if not exists:
                            conn.execute("""
                                INSERT INTO events 
                                (symbol, timeframe, event_ts, pattern, attrs)
                                VALUES (?, ?, ?, ?, ?)
                            """, [symbol, "5m", event["event_ts"], event["pattern"], event["attrs"]])
                            
                            logger.info(f"New pattern detected: {symbol} {event['pattern']} at {event['event_ts']}")
                    
                    # Detect contexts
                    contexts = detect_trend_context(
                        conn, symbol, "15m",
                        start_ts=start_time.isoformat(),
                        end_ts=end_time.isoformat()
                    )
                    
                    for ctx in contexts:
                        exists = conn.execute("""
                            SELECT 1 FROM contexts
                            WHERE symbol = ? AND timeframe = ? 
                            AND event_ts = ? AND context = ?
                        """, [symbol, "15m", ctx["event_ts"], ctx["context"]]).fetchone()
                        
                        if not exists:
                            conn.execute("""
                                INSERT INTO contexts
                                (symbol, timeframe, event_ts, context, attrs)
                                VALUES (?, ?, ?, ?, ?)
                            """, [symbol, "15m", ctx["event_ts"], ctx["context"], ctx["attrs"]])
                    
                except Exception as e:
                    logger.error(f"Error detecting patterns for {symbol}: {e}")
            
            conn.commit()
            
            # Update outcomes for new events
            self._update_outcomes(conn)
    
    def _update_outcomes(self, conn: DuckDBPyConnection):
        """Update outcomes for events without forward returns"""
        events = conn.execute("""
            SELECT symbol, timeframe, event_ts, pattern
            FROM events
            WHERE fwd_ret IS NULL
            AND event_ts < datetime('now', '-1 hour')  -- Only events older than 1 hour
            ORDER BY event_ts
            LIMIT 100
        """).fetchall()
        
        if events:
            event_dicts = [
                {
                    "symbol": e[0],
                    "timeframe": e[1],
                    "event_ts": e[2],
                    "pattern": e[3],
                }
                for e in events
            ]
            
            updated = update_event_outcomes(conn, event_dicts)
            if updated > 0:
                logger.info(f"Updated outcomes for {updated} events")
                conn.commit()
    
    def start(self):
        """Start the sync service"""
        self.running = True
        asyncio.run(self.sync_data())
    
    def stop(self):
        """Stop the sync service"""
        self.running = False
        self.fetcher.stop()
        logger.info("Data sync service stopped")