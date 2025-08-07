"""
PR6: Out-of-Sample (OOS) Sanctity Enforcer

This module provides a strict temporal boundary layer to prevent future peeking.
All data access must go through this enforcer to maintain backtest validity.
"""
import hashlib
import logging
import re
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class FuturePeekError(Exception):
    """Raised when a query attempts to access future data."""
    pass


class OOSConfig:
    """OOS configuration loaded from database."""
    
    def __init__(self, current_cutoff: datetime, buffer_hours: int, enforced: bool):
        self.current_cutoff = current_cutoff
        self.buffer_hours = buffer_hours
        self.enforced = enforced
        self.safe_cutoff = current_cutoff - timedelta(hours=buffer_hours)
    
    @classmethod
    def load(cls, conn: DuckDBPyConnection, config_id: str = "default") -> "OOSConfig":
        """Load OOS config from database."""
        result = conn.execute("""
            SELECT current_cutoff, buffer_hours, enforced 
            FROM oos_config 
            WHERE id = ?
        """, [config_id]).fetchone()
        
        if not result:
            raise ValueError(f"OOS config '{config_id}' not found")
        
        return cls(
            current_cutoff=result[0],
            buffer_hours=result[1],
            enforced=result[2]
        )


class OOSEnforcer:
    """
    Enforces out-of-sample sanctity by intercepting and validating all data queries.
    
    Key principles:
    1. No query can access data after current_cutoff
    2. Buffer zone prevents near-future peeking
    3. All access is logged for audit
    4. Violations trigger immediate failure
    """
    
    def __init__(self, conn: DuckDBPyConnection, user_context: str = "unknown"):
        self.conn = conn
        self.user_context = user_context
        self.config = OOSConfig.load(conn)
        
    def execute_safe_query(
        self, 
        query: str, 
        params: Optional[List] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Tuple]:
        """
        Execute query with OOS enforcement.
        
        Args:
            query: SQL query to execute
            params: Query parameters  
            symbol: Symbol being queried (for logging)
            timeframe: Timeframe being queried (for logging)
            
        Returns:
            Query results
            
        Raises:
            FuturePeekError: If query violates temporal boundaries
        """
        if not self.config.enforced:
            logger.warning("OOS enforcement is DISABLED - query executed without checks")
            return self.conn.execute(query, params or []).fetchall()
        
        # Analyze query for temporal violations
        violation = self._analyze_query_safety(query, params)
        
        if violation:
            self._log_violation(query, params, symbol, timeframe, violation)
            raise FuturePeekError(f"Query violates OOS sanctity: {violation}")
        
        # Execute query and log access
        result = self.conn.execute(query, params or []).fetchall()
        self._log_access(query, params, symbol, timeframe, result)
        
        return result
    
    def get_safe_bars(
        self,
        symbol: str,
        timeframe: str,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Tuple]:
        """
        Get bars with automatic OOS enforcement.
        
        This is the preferred way to access OHLCV data.
        """
        # Force end_ts to be within safe bounds
        if end_ts is None or end_ts > self.config.safe_cutoff:
            end_ts = self.config.safe_cutoff
        
        query_parts = [
            "SELECT symbol, timeframe, ts, open, high, low, close, volume",
            "FROM bars",
            "WHERE symbol = ? AND timeframe = ?"
        ]
        params = [symbol, timeframe]
        
        if start_ts:
            query_parts.append("AND ts >= ?")
            params.append(start_ts)
        
        query_parts.append("AND ts <= ?")
        params.append(end_ts)
        
        query_parts.append("ORDER BY ts")
        
        if limit:
            query_parts.append(f"LIMIT {limit}")
        
        query = " ".join(query_parts)
        
        return self.execute_safe_query(query, params, symbol, timeframe)
    
    def compute_forward_returns_oos(
        self,
        symbol: str,
        timeframe: str,
        event_ts: datetime,
        lead_bars: int = 10
    ) -> Optional[Dict]:
        """
        Compute forward returns with strict OOS enforcement.
        
        This replaces the dangerous compute_forward_returns function.
        Only computes if event_ts + lead_bars is within safe bounds.
        """
        # Estimate when the forward window would end
        # Conservative estimate: assume 1 bar per minute for safety
        estimated_end = event_ts + timedelta(minutes=lead_bars)
        
        if estimated_end > self.config.safe_cutoff:
            logger.warning(
                f"Cannot compute forward returns for {symbol} {event_ts}: "
                f"would require future data (estimated end: {estimated_end}, cutoff: {self.config.safe_cutoff})"
            )
            return None
        
        # Query with explicit temporal boundaries
        query = """
        WITH entry AS (
            SELECT close as entry_price
            FROM bars
            WHERE symbol = ? AND timeframe = ? AND ts = ?
            LIMIT 1
        ),
        forward_bars AS (
            SELECT 
                ts,
                close,
                low,
                ROW_NUMBER() OVER (ORDER BY ts) as bar_num
            FROM bars
            WHERE symbol = ? AND timeframe = ? 
            AND ts > ? AND ts <= ?
            ORDER BY ts
            LIMIT ?
        )
        SELECT 
            e.entry_price,
            f.close as exit_price,
            MIN(f.low) as min_low,
            COUNT(*) as actual_bars
        FROM entry e
        CROSS JOIN forward_bars f
        WHERE f.bar_num = ?
        GROUP BY e.entry_price, f.close
        """
        
        params = [
            symbol, timeframe, event_ts,           # entry query
            symbol, timeframe, event_ts, self.config.safe_cutoff,  # forward window
            lead_bars,                             # limit
            lead_bars                              # final filter
        ]
        
        result = self.execute_safe_query(query, params, symbol, timeframe)
        
        if not result:
            return None
        
        entry_price, exit_price, min_low, actual_bars = result[0]
        
        if actual_bars < lead_bars:
            logger.warning(f"Insufficient forward data: got {actual_bars}, needed {lead_bars}")
            return None
        
        return_pct = (exit_price - entry_price) / entry_price
        drawdown_pct = (min_low - entry_price) / entry_price
        
        return {
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "return_pct": float(return_pct),
            "drawdown_pct": float(drawdown_pct),
            "actual_bars": int(actual_bars),
            "computed_at": datetime.now(),
            "oos_cutoff": self.config.safe_cutoff
        }
    
    def _analyze_query_safety(self, query: str, params: Optional[List]) -> Optional[str]:
        """
        Analyze query for potential temporal violations.
        
        Returns violation description or None if safe.
        """
        query_lower = query.lower()
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (r'ts\s*>\s*[\'"][^\'"]*(now\(\)|current_timestamp)', 'uses current_timestamp in comparison'),
            (r'where.*ts\s*>\s*[\'"][^\'"]*[\'"].*order\s+by\s+ts(?:\s+desc)?.*limit\s+\d+', 'unbounded future query with limit'),
            (r'max\s*\(\s*ts\s*\)', 'queries maximum timestamp'),
            (r'order\s+by\s+ts\s+desc\s+limit\s+1', 'queries most recent data'),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, query_lower):
                return description
        
        # Check if query explicitly filters by timestamp
        if 'ts' not in query_lower and 'timestamp' not in query_lower:
            return "query lacks temporal filtering - could access all data"
        
        # If params contain timestamps, validate them
        if params:
            for param in params:
                if isinstance(param, datetime) and param > self.config.safe_cutoff:
                    return f"parameter timestamp {param} exceeds safe cutoff {self.config.safe_cutoff}"
        
        return None
    
    def _log_access(
        self, 
        query: str, 
        params: Optional[List],
        symbol: Optional[str],
        timeframe: Optional[str], 
        result: List[Tuple]
    ):
        """Log successful data access."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        # Extract timestamp bounds from result if possible
        min_ts, max_ts = self._extract_timestamp_bounds(result)
        
        try:
            self.conn.execute("""
                INSERT INTO data_access_log 
                (query_hash, user_context, symbol, timeframe, min_timestamp, max_timestamp, 
                 row_count, violation_type, query_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """, [
                query_hash,
                self.user_context,
                symbol,
                timeframe,
                min_ts,
                max_ts,
                len(result),
                query[:2000]  # Truncate long queries
            ])
        except Exception as e:
            logger.error(f"Failed to log data access: {e}")
    
    def _log_violation(
        self,
        query: str,
        params: Optional[List],
        symbol: Optional[str],
        timeframe: Optional[str],
        violation: str
    ):
        """Log temporal violation for forensic analysis."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        stack = traceback.format_stack()
        
        try:
            self.conn.execute("""
                INSERT INTO data_access_log 
                (query_hash, user_context, symbol, timeframe, row_count, 
                 violation_type, stack_trace, query_text)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?)
            """, [
                query_hash,
                self.user_context,
                symbol,
                timeframe,
                violation,
                '\n'.join(stack[-5:]),  # Last 5 stack frames
                query[:2000]
            ])
        except Exception as e:
            logger.error(f"Failed to log violation: {e}")
    
    def _extract_timestamp_bounds(self, result: List[Tuple]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Try to extract min/max timestamps from query result."""
        if not result:
            return None, None
        
        timestamps = []
        
        # Look for datetime objects in each row
        for row in result:
            for value in row:
                if isinstance(value, datetime):
                    timestamps.append(value)
        
        if timestamps:
            return min(timestamps), max(timestamps)
        
        return None, None
    
    @staticmethod
    def create_purged_splits(
        conn: DuckDBPyConnection,
        symbol: str,
        timeframe: str,
        n_splits: int = 5,
        purge_pct: float = 0.1
    ) -> List[Dict]:
        """
        Create purged k-fold splits for walk-forward backtesting.
        
        Purge gaps prevent future information leakage.
        """
        # Get data bounds
        result = conn.execute("""
            SELECT MIN(ts), MAX(ts), COUNT(*) 
            FROM bars 
            WHERE symbol = ? AND timeframe = ?
        """, [symbol, timeframe]).fetchone()
        
        if not result or result[2] < 100:
            raise ValueError(f"Insufficient data for {symbol} {timeframe}")
        
        min_ts, max_ts, total_bars = result
        total_duration = max_ts - min_ts
        split_duration = total_duration / n_splits
        purge_duration = split_duration * purge_pct
        
        splits = []
        
        for i in range(n_splits):
            # Each split: train on earlier data, purge gap, test on later data
            test_start = min_ts + (i * split_duration)
            test_end = test_start + split_duration
            
            purge_start = test_start - purge_duration
            train_end = purge_start
            train_start = min_ts
            
            split = {
                "fold_id": i,
                "train_start": train_start,
                "train_end": train_end,
                "purge_start": purge_start,
                "purge_end": test_start,
                "test_start": test_start,
                "test_end": test_end,
                "symbol": symbol,
                "timeframe": timeframe,
                "metadata": {
                    "total_duration_hours": total_duration.total_seconds() / 3600,
                    "purge_pct": purge_pct,
                    "n_splits": n_splits
                }
            }
            
            splits.append(split)
            
            # Save to database
            conn.execute("""
                INSERT INTO backtest_splits 
                (fold_id, train_start, train_end, purge_start, purge_end, 
                 test_start, test_end, symbol, timeframe, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                split["fold_id"],
                split["train_start"],
                split["train_end"], 
                split["purge_start"],
                split["purge_end"],
                split["test_start"],
                split["test_end"],
                split["symbol"],
                split["timeframe"],
                split["metadata"]
            ])
        
        logger.info(f"Created {n_splits} purged splits for {symbol} {timeframe}")
        return splits


def update_oos_cutoff(conn: DuckDBPyConnection, new_cutoff: datetime, config_id: str = "default"):
    """Update the OOS cutoff timestamp."""
    conn.execute("""
        UPDATE oos_config 
        SET current_cutoff = ?, ts = CURRENT_TIMESTAMP 
        WHERE id = ?
    """, [new_cutoff, config_id])
    
    logger.info(f"Updated OOS cutoff to {new_cutoff}")


def get_violation_report(conn: DuckDBPyConnection, hours: int = 24) -> Dict[str, Any]:
    """Get report of recent violations."""
    cutoff = datetime.now() - timedelta(hours=hours)
    
    violations = conn.execute("""
        SELECT user_context, violation_type, COUNT(*), MAX(ts)
        FROM data_access_log
        WHERE violation_type IS NOT NULL AND ts > ?
        GROUP BY user_context, violation_type
        ORDER BY COUNT(*) DESC
    """, [cutoff]).fetchall()
    
    total_queries = conn.execute("""
        SELECT COUNT(*) FROM data_access_log WHERE ts > ?
    """, [cutoff]).fetchone()[0]
    
    return {
        "hours": hours,
        "total_queries": total_queries,
        "violations": [
            {
                "user_context": v[0],
                "violation_type": v[1], 
                "count": v[2],
                "last_seen": v[3]
            }
            for v in violations
        ],
        "clean_queries": total_queries - sum(v[2] for v in violations)
    }