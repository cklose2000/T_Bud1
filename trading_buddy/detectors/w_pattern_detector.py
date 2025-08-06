from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


def detect_w_pattern(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str = "5m",
    lookback_hours: float = 4.0,
    second_bottom_tolerance_pct: float = 0.5,  # Allow second bottom to be up to 0.5% higher
    min_bars_between: int = 5,
    max_bars_between: int = 30,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Dict]:
    """
    Detect W patterns (imperfect double bottoms where second bottom can be slightly higher).
    Also checks for MACD momentum divergence.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # Get price data with MACD
    query = f"""
    WITH price_data AS (
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
    sma_fast AS (
        SELECT 
            ts,
            AVG(close) OVER (ORDER BY ts ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as sma12
        FROM price_data
    ),
    sma_slow AS (
        SELECT 
            ts,
            AVG(close) OVER (ORDER BY ts ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) as sma26
        FROM price_data
    ),
    macd_line AS (
        SELECT 
            p.ts,
            p.bar_num,
            p.low,
            p.high,
            p.close,
            (f.sma12 - s.sma26) as macd,
            LAG(f.sma12 - s.sma26, 1) OVER (ORDER BY p.ts) as prev_macd
        FROM price_data p
        JOIN sma_fast f ON p.ts = f.ts
        JOIN sma_slow s ON p.ts = s.ts
    ),
    swing_lows AS (
        SELECT 
            bar_num,
            ts,
            low,
            macd,
            prev_macd,
            LAG(low, 2) OVER (ORDER BY ts) as prev2_low,
            LAG(low, 1) OVER (ORDER BY ts) as prev1_low,
            LEAD(low, 1) OVER (ORDER BY ts) as next1_low,
            LEAD(low, 2) OVER (ORDER BY ts) as next2_low
        FROM macd_line
    )
    SELECT bar_num, ts, low, macd, prev_macd
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
    
    # Check all pairs of swing lows
    for i in range(len(swing_lows) - 1):
        for j in range(i + 1, len(swing_lows)):
            bar_num1, ts1, low1, macd1, prev_macd1 = swing_lows[i]
            bar_num2, ts2, low2, macd2, prev_macd2 = swing_lows[j]
            
            # Check bar distance
            bar_distance = bar_num2 - bar_num1
            if bar_distance < min_bars_between or bar_distance > max_bars_between:
                continue
            
            # Check if second bottom is within tolerance (can be slightly higher)
            price_diff_pct = (low2 - low1) / low1 * 100
            if price_diff_pct < 0 or price_diff_pct > second_bottom_tolerance_pct:
                continue
            
            # Find neckline (highest high between the two lows)
            neckline_query = f"""
            SELECT MAX(high) as neckline_price, AVG(high) as avg_high
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
            avg_high = neckline_result[1]
            
            # Calculate W pattern quality metrics
            w_depth = (neckline_price - (low1 + low2) / 2) / neckline_price * 100
            
            # Check MACD divergence (bearish momentum slowing)
            macd_divergence = False
            if macd1 is not None and macd2 is not None:
                # Price made similar or lower low, but MACD is less negative (bullish divergence)
                if low2 >= low1 and macd2 > macd1:
                    macd_divergence = True
            
            # Get MACD histogram trend between the two lows
            macd_trend_query = f"""
            WITH macd_calc AS (
                SELECT 
                    ts,
                    AVG(close) OVER (ORDER BY ts ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) -
                    AVG(close) OVER (ORDER BY ts ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) as macd
                FROM bars
                WHERE symbol = '{symbol}' 
                AND timeframe = '{timeframe}'
                AND ts >= '{ts1}'
                AND ts <= '{ts2}'
            )
            SELECT 
                MIN(macd) as min_macd,
                MAX(macd) as max_macd,
                LAST(macd ORDER BY ts) as final_macd
            FROM macd_calc
            """
            
            macd_stats = conn.execute(macd_trend_query).fetchone()
            
            patterns.append({
                "event_ts": ts2,
                "pattern": "w_pattern",
                "attrs": {
                    "low1_ts": str(ts1),
                    "low1_price": low1,
                    "low2_ts": str(ts2),
                    "low2_price": low2,
                    "neckline_price": neckline_price,
                    "avg_high_between": avg_high,
                    "bar_distance": bar_distance,
                    "second_bottom_higher_pct": price_diff_pct,
                    "w_depth_pct": w_depth,
                    "macd_divergence": macd_divergence,
                    "macd1": macd1,
                    "macd2": macd2,
                    "macd_min": macd_stats[0] if macd_stats else None,
                    "macd_max": macd_stats[1] if macd_stats else None,
                    "macd_final": macd_stats[2] if macd_stats else None,
                }
            })
            
            logger.info(f"Found W pattern: {ts1} (${low1:.2f}) -> {ts2} (${low2:.2f}), "
                       f"second bottom {price_diff_pct:.2f}% higher, "
                       f"MACD divergence: {macd_divergence}")
    
    return patterns


def analyze_aug5_w_pattern(conn: DuckDBPyConnection) -> Dict:
    """
    Specifically analyze the August 5 W pattern around 10:50-11:45
    """
    # Look for W patterns in the morning session
    patterns = detect_w_pattern(
        conn,
        'SPY',
        '5min',
        lookback_hours=2,
        second_bottom_tolerance_pct=0.5,
        min_bars_between=8,  # At least 40 minutes
        max_bars_between=20,  # Max 100 minutes
        start_ts='2025-08-05 09:30:00',
        end_ts='2025-08-05 13:00:00'
    )
    
    if patterns:
        # Find the specific pattern around 10:50-11:45
        for p in patterns:
            attrs = p['attrs']
            ts1 = datetime.fromisoformat(attrs['low1_ts'])
            ts2 = datetime.fromisoformat(attrs['low2_ts'])
            
            # Check if this matches the timeframe we're looking for
            if (ts1.hour == 10 and ts1.minute >= 45) or (ts1.hour == 11 and ts1.minute <= 0):
                if (ts2.hour == 11 and ts2.minute >= 40) or (ts2.hour == 11 and ts2.minute <= 50):
                    # This is our pattern! Get detailed analysis
                    detail_query = f"""
                    WITH price_action AS (
                        SELECT 
                            ts,
                            high,
                            low,
                            close,
                            volume
                        FROM bars
                        WHERE symbol = 'SPY'
                        AND timeframe = '5min'
                        AND ts >= '{attrs["low1_ts"]}'
                        AND ts <= '{ts2 + timedelta(minutes=30)}'
                        ORDER BY ts
                    )
                    SELECT 
                        COUNT(*) as bars_in_pattern,
                        MAX(high) as pattern_high,
                        MIN(low) as pattern_low,
                        SUM(volume) as total_volume,
                        LAST(close ORDER BY ts) as exit_price
                    FROM price_action
                    """
                    
                    details = conn.execute(detail_query).fetchone()
                    
                    return {
                        "pattern": p,
                        "details": {
                            "bars_in_pattern": details[0],
                            "pattern_high": details[1],
                            "pattern_low": details[2],
                            "total_volume": details[3],
                            "exit_price": details[4],
                            "pattern_range": details[1] - details[2],
                            "risk_reward": (details[4] - attrs['low2_price']) / (attrs['low2_price'] - details[2])
                                          if details[2] < attrs['low2_price'] else 0
                        }
                    }
    
    return None