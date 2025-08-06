from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from duckdb import DuckDBPyConnection


def detect_swing_points(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    neighborhood: int = 3,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> Dict[str, List[Tuple]]:
    """
    Detect swing highs and lows using neighborhood comparison.
    Returns dict with 'highs' and 'lows' lists of (ts, price) tuples.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # Detect swing highs
    high_query = f"""
    WITH windowed AS (
        SELECT 
            ts,
            high,
            MAX(high) OVER (
                ORDER BY ts 
                ROWS BETWEEN {neighborhood} PRECEDING AND {neighborhood} FOLLOWING
            ) as window_max,
            LAG(high, 1) OVER (ORDER BY ts) as prev_high,
            LEAD(high, 1) OVER (ORDER BY ts) as next_high
        FROM bars
        {where_clause}
    )
    SELECT ts, high
    FROM windowed
    WHERE high = window_max
    AND high > prev_high
    AND high > next_high
    ORDER BY ts
    """
    
    # Detect swing lows
    low_query = f"""
    WITH windowed AS (
        SELECT 
            ts,
            low,
            MIN(low) OVER (
                ORDER BY ts 
                ROWS BETWEEN {neighborhood} PRECEDING AND {neighborhood} FOLLOWING
            ) as window_min,
            LAG(low, 1) OVER (ORDER BY ts) as prev_low,
            LEAD(low, 1) OVER (ORDER BY ts) as next_low
        FROM bars
        {where_clause}
    )
    SELECT ts, low
    FROM windowed
    WHERE low = window_min
    AND low < prev_low
    AND low < next_low
    ORDER BY ts
    """
    
    highs = conn.execute(high_query).fetchall()
    lows = conn.execute(low_query).fetchall()
    
    return {"highs": highs, "lows": lows}


def detect_double_bottom(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str = "5m",
    lookback_hours: float = 2.0,
    tolerance_bps: int = 50,  # basis points
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Dict]:
    """
    Detect double bottom patterns.
    Returns list of pattern occurrences with attributes.
    """
    swing_points = detect_swing_points(conn, symbol, timeframe, start_ts=start_ts, end_ts=end_ts)
    lows = swing_points["lows"]
    highs = swing_points["highs"]
    
    if len(lows) < 2:
        return []
    
    patterns = []
    tolerance_pct = tolerance_bps / 10000.0
    
    for i in range(1, len(lows)):
        low1_ts, low1_price = lows[i-1]
        low2_ts, low2_price = lows[i]
        
        # Check time constraint
        time_diff = (datetime.fromisoformat(str(low2_ts)) - 
                    datetime.fromisoformat(str(low1_ts))).total_seconds() / 3600
        if time_diff > lookback_hours:
            continue
        
        # Check price similarity
        price_diff = abs(low2_price - low1_price) / low1_price
        if price_diff > tolerance_pct:
            continue
        
        # Find neckline (highest high between the two lows)
        neckline_highs = [h for h in highs if low1_ts < h[0] < low2_ts]
        if not neckline_highs:
            continue
        
        neckline = max(neckline_highs, key=lambda x: x[1])
        
        # Check for confirmation (close above neckline)
        confirm_query = f"""
        SELECT ts, close
        FROM bars
        WHERE symbol = '{symbol}' 
        AND timeframe = '{timeframe}'
        AND ts > '{low2_ts}'
        AND close > {neckline[1]}
        ORDER BY ts
        LIMIT 1
        """
        
        confirm_result = conn.execute(confirm_query).fetchone()
        
        if confirm_result:
            patterns.append({
                "event_ts": confirm_result[0],
                "pattern": "double_bottom",
                "attrs": {
                    "low1_ts": str(low1_ts),
                    "low1_price": low1_price,
                    "low2_ts": str(low2_ts),
                    "low2_price": low2_price,
                    "neckline_price": neckline[1],
                    "confirm_price": confirm_result[1],
                }
            })
    
    return patterns


def detect_macd_bull_cross(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Dict]:
    """
    Detect MACD bullish crossovers (MACD line crosses above signal line).
    Returns list of crossover events.
    """
    from trading_buddy.detectors.indicators_simple import compute_sma_based_macd
    
    macd_data = compute_sma_based_macd(conn, symbol, timeframe, fast, slow, signal, start_ts, end_ts)
    
    if len(macd_data) < 2:
        return []
    
    crosses = []
    
    for i in range(1, len(macd_data)):
        prev_ts, prev_macd, prev_signal, _ = macd_data[i-1]
        curr_ts, curr_macd, curr_signal, _ = macd_data[i]
        
        # Skip if any values are None
        if any(v is None for v in [prev_macd, prev_signal, curr_macd, curr_signal]):
            continue
        
        # Detect bullish crossover
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            crosses.append({
                "event_ts": curr_ts,
                "pattern": "macd_bull_cross",
                "attrs": {
                    "macd": curr_macd,
                    "signal": curr_signal,
                    "histogram": curr_macd - curr_signal,
                    "fast": fast,
                    "slow": slow,
                    "signal_period": signal,
                }
            })
    
    return crosses


def detect_compound_pattern(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str = "5m",
    within_bars: int = 3,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Dict]:
    """
    Detect compound pattern: double_bottom + macd_bull_cross within N bars.
    """
    # Get double bottoms
    double_bottoms = detect_double_bottom(conn, symbol, timeframe, start_ts=start_ts, end_ts=end_ts)
    
    # Get MACD crosses
    macd_crosses = detect_macd_bull_cross(conn, symbol, timeframe, start_ts=start_ts, end_ts=end_ts)
    
    if not double_bottoms or not macd_crosses:
        return []
    
    compounds = []
    
    for db in double_bottoms:
        db_ts = datetime.fromisoformat(str(db["event_ts"]))
        
        # Find MACD crosses near the double bottom confirmation
        nearby_crosses = [
            mc for mc in macd_crosses
            if abs((datetime.fromisoformat(str(mc["event_ts"])) - db_ts).total_seconds()) <= within_bars * 300  # 5m bars
        ]
        
        if nearby_crosses:
            # Take the closest cross
            closest_cross = min(nearby_crosses, 
                              key=lambda x: abs((datetime.fromisoformat(str(x["event_ts"])) - db_ts).total_seconds()))
            
            compounds.append({
                "event_ts": db["event_ts"],
                "pattern": "double_bottom_macd_bull",
                "attrs": {
                    **db["attrs"],
                    "macd_cross_ts": str(closest_cross["event_ts"]),
                    "macd_attrs": closest_cross["attrs"],
                }
            })
    
    return compounds


def detect_all_patterns(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str = "5m",
    lookback_hours: float = 24.0,
    include_w_patterns: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Detect all pattern types for a symbol.
    Returns dict keyed by pattern type.
    """
    from trading_buddy.detectors.w_pattern_detector import detect_w_pattern
    
    end_ts = datetime.now()
    start_ts = end_ts - timedelta(hours=lookback_hours)
    
    results = {
        "double_bottom": detect_double_bottom(
            conn, symbol, timeframe, 
            lookback_hours=lookback_hours,
            start_ts=str(start_ts),
            end_ts=str(end_ts)
        ),
        "macd_bull_cross": detect_macd_bull_cross(
            conn, symbol, timeframe,
            start_ts=str(start_ts),
            end_ts=str(end_ts)
        ),
        "compound": detect_compound_pattern(
            conn, symbol, timeframe,
            start_ts=str(start_ts),
            end_ts=str(end_ts)
        )
    }
    
    if include_w_patterns:
        results["w_pattern"] = detect_w_pattern(
            conn, symbol, timeframe,
            lookback_hours=lookback_hours,
            start_ts=str(start_ts),
            end_ts=str(end_ts)
        )
    
    return results