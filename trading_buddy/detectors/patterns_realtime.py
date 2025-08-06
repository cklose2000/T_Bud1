from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


def detect_double_bottom_flexible(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str = "5m",
    lookback_hours: float = 24.0,
    tolerance_bps: int = 200,  # More tolerance for real-time
    min_bars_between: int = 5,
    max_bars_between: int = 100,
    confirmation_bars: int = 5,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Dict]:
    """
    Enhanced double bottom detection with flexible parameters for real-time data.
    
    Parameters:
    - lookback_hours: How far back to search
    - tolerance_bps: Price tolerance in basis points (200 = 2%)
    - min_bars_between: Minimum bars between the two bottoms
    - max_bars_between: Maximum bars between the two bottoms
    - confirmation_bars: Bars to wait for confirmation above neckline
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # Get all price data with row numbers
    query = f"""
    WITH numbered_bars AS (
        SELECT 
            ts,
            open,
            high,
            low,
            close,
            volume,
            ROW_NUMBER() OVER (ORDER BY ts) as bar_num
        FROM bars
        {where_clause}
        ORDER BY ts
    ),
    swing_lows AS (
        SELECT 
            bar_num,
            ts,
            low,
            LAG(low, 2) OVER (ORDER BY ts) as prev2_low,
            LAG(low, 1) OVER (ORDER BY ts) as prev1_low,
            LEAD(low, 1) OVER (ORDER BY ts) as next1_low,
            LEAD(low, 2) OVER (ORDER BY ts) as next2_low
        FROM numbered_bars
    )
    SELECT bar_num, ts, low
    FROM swing_lows
    WHERE low <= prev1_low 
    AND low <= next1_low
    AND low <= COALESCE(prev2_low, low + 1)
    AND low <= COALESCE(next2_low, low + 1)
    ORDER BY ts
    """
    
    swing_lows = conn.execute(query).fetchall()
    
    if len(swing_lows) < 2:
        return []
    
    patterns = []
    tolerance_pct = tolerance_bps / 10000.0
    
    # Check all pairs of swing lows
    for i in range(len(swing_lows) - 1):
        for j in range(i + 1, len(swing_lows)):
            bar_num1, ts1, low1 = swing_lows[i]
            bar_num2, ts2, low2 = swing_lows[j]
            
            # Check bar distance
            bar_distance = bar_num2 - bar_num1
            if bar_distance < min_bars_between or bar_distance > max_bars_between:
                continue
            
            # Check price similarity
            price_diff = abs(low2 - low1) / low1
            if price_diff > tolerance_pct:
                continue
            
            # Find neckline (highest high between the two lows)
            neckline_query = f"""
            SELECT MAX(high) as neckline_price
            FROM bars
            WHERE symbol = '{symbol}' 
            AND timeframe = '{timeframe}'
            AND ts > '{ts1}'
            AND ts < '{ts2}'
            """
            
            neckline_result = conn.execute(neckline_query).fetchone()
            if not neckline_result or not neckline_result[0]:
                continue
            
            neckline_price = neckline_result[0]
            
            # Look for confirmation (price breaking above neckline)
            confirm_query = f"""
            SELECT ts, close, high
            FROM bars
            WHERE symbol = '{symbol}' 
            AND timeframe = '{timeframe}'
            AND ts > '{ts2}'
            ORDER BY ts
            LIMIT {confirmation_bars}
            """
            
            confirm_bars = conn.execute(confirm_query).fetchall()
            
            # Check if any bar breaks above neckline
            confirmed = False
            confirm_ts = None
            confirm_price = None
            
            for bar_ts, bar_close, bar_high in confirm_bars:
                if bar_close > neckline_price or bar_high > neckline_price * 1.001:
                    confirmed = True
                    confirm_ts = bar_ts
                    confirm_price = max(bar_close, bar_high)
                    break
            
            if confirmed:
                patterns.append({
                    "event_ts": confirm_ts,
                    "pattern": "double_bottom",
                    "attrs": {
                        "low1_ts": str(ts1),
                        "low1_price": low1,
                        "low2_ts": str(ts2),
                        "low2_price": low2,
                        "neckline_price": neckline_price,
                        "confirm_price": confirm_price,
                        "bar_distance": bar_distance,
                        "price_similarity_pct": (1 - price_diff) * 100,
                    }
                })
                logger.info(f"Found double bottom: {ts1} (${low1:.2f}) -> {ts2} (${low2:.2f}), neckline: ${neckline_price:.2f}")
    
    return patterns


def detect_intraday_double_bottom(
    conn: DuckDBPyConnection,
    symbol: str,
    date: str,  # Format: 'YYYY-MM-DD'
    tolerance_bps: int = 100,
) -> List[Dict]:
    """
    Specialized detector for intraday double bottoms on a specific date.
    More lenient parameters for catching subtle patterns.
    """
    # Get all 5-minute bars for the specified date
    query = f"""
    WITH price_data AS (
        SELECT 
            ts,
            low,
            high,
            close,
            ROW_NUMBER() OVER (ORDER BY ts) as bar_num
        FROM bars
        WHERE symbol = '{symbol}'
        AND timeframe = '5min'
        AND DATE(ts) = '{date}'
        ORDER BY ts
    ),
    local_minima AS (
        SELECT 
            bar_num,
            ts,
            low,
            LAG(low, 1) OVER (ORDER BY ts) as prev_low,
            LEAD(low, 1) OVER (ORDER BY ts) as next_low
        FROM price_data
    )
    SELECT bar_num, ts, low
    FROM local_minima
    WHERE low <= prev_low AND low <= next_low
    ORDER BY low
    LIMIT 10  -- Focus on the 10 lowest points
    """
    
    lowest_points = conn.execute(query).fetchall()
    
    if len(lowest_points) < 2:
        return []
    
    patterns = []
    tolerance_pct = tolerance_bps / 10000.0
    
    # Sort by timestamp to check chronological pairs
    lowest_points.sort(key=lambda x: x[1])
    
    # Check pairs that are at least 10 bars apart
    for i in range(len(lowest_points) - 1):
        for j in range(i + 1, len(lowest_points)):
            bar_num1, ts1, low1 = lowest_points[i]
            bar_num2, ts2, low2 = lowest_points[j]
            
            # Require at least 10 bars between bottoms (50 minutes)
            if bar_num2 - bar_num1 < 10:
                continue
            
            # Check price similarity
            price_diff = abs(low2 - low1) / low1
            if price_diff <= tolerance_pct:
                # Calculate bounce height after second bottom
                bounce_query = f"""
                SELECT MAX(high) as max_bounce
                FROM bars
                WHERE symbol = '{symbol}'
                AND timeframe = '5min'
                AND ts > '{ts2}'
                AND DATE(ts) = '{date}'
                """
                
                bounce_result = conn.execute(bounce_query).fetchone()
                bounce_height = bounce_result[0] - low2 if bounce_result and bounce_result[0] else 0
                
                patterns.append({
                    "event_ts": ts2,
                    "pattern": "intraday_double_bottom",
                    "attrs": {
                        "low1_ts": str(ts1),
                        "low1_price": low1,
                        "low2_ts": str(ts2),
                        "low2_price": low2,
                        "price_diff_pct": price_diff * 100,
                        "bars_between": bar_num2 - bar_num1,
                        "bounce_height": bounce_height,
                    }
                })
    
    return patterns


def scan_for_patterns(
    conn: DuckDBPyConnection,
    symbols: List[str],
    timeframes: List[str],
    lookback_hours: float = 24.0,
) -> Dict[str, List[Dict]]:
    """
    Scan multiple symbols and timeframes for patterns.
    Returns dict keyed by symbol.
    """
    results = {}
    
    for symbol in symbols:
        symbol_patterns = []
        
        for timeframe in timeframes:
            # Regular double bottom
            patterns = detect_double_bottom_flexible(
                conn, symbol, timeframe, 
                lookback_hours=lookback_hours,
                tolerance_bps=200
            )
            symbol_patterns.extend(patterns)
            
            # Also check with tighter tolerance
            tight_patterns = detect_double_bottom_flexible(
                conn, symbol, timeframe,
                lookback_hours=lookback_hours,
                tolerance_bps=50
            )
            symbol_patterns.extend(tight_patterns)
        
        if symbol_patterns:
            # Remove duplicates based on event_ts
            unique_patterns = []
            seen_ts = set()
            for p in symbol_patterns:
                if p["event_ts"] not in seen_ts:
                    seen_ts.add(p["event_ts"])
                    unique_patterns.append(p)
            
            results[symbol] = unique_patterns
    
    return results