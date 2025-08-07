from typing import List, Optional, Tuple

from duckdb import DuckDBPyConnection
from trading_buddy.core.detector_audit import audit_detector, audit_sql_query


@audit_detector
def compute_ema(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    period: int,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    allow_future: bool = False,
) -> List[Tuple]:
    """
    Compute EMA with CAUSAL warm-up masking.
    
    *** PR6 CRITICAL FIX: 4×period warm-up, null until stable ***
    
    EMA values are masked (null) until 4×period bars for stability.
    This prevents using unstable EMA values in backtests.
    
    Args:
        allow_future: Audit flag for future peek detection
        
    Returns list of (ts, ema_value) tuples where ema_value is null during warm-up.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    warmup_bars = 4 * period  # User specification: 4×period warm-up
    
    query = f"""
    WITH ema_with_warmup AS (
        SELECT 
            ts,
            close,
            exponential_moving_average(close, {period}) OVER (
                ORDER BY ts 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as raw_ema,
            ROW_NUMBER() OVER (ORDER BY ts) as bar_num
        FROM bars
        {where_clause}
    )
    SELECT 
        ts,
        CASE 
            WHEN bar_num >= {warmup_bars} THEN raw_ema  -- Only after warm-up
            ELSE NULL  -- Masked during warm-up period
        END as ema_{period}
    FROM ema_with_warmup
    ORDER BY ts
    """
    
    # Audit SQL query for future peeking patterns
    audit_sql_query(query, "compute_ema")
    
    return conn.execute(query).fetchall()


@audit_detector
def compute_macd(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    allow_future: bool = False,
) -> List[Tuple]:
    """
    Compute MACD indicator with CAUSAL warm-up masking.
    
    *** PR6 CRITICAL FIX: Composite warm-up for MACD stability ***
    
    MACD requires warm-up for:
    - Fast EMA: 4×fast bars
    - Slow EMA: 4×slow bars  
    - Signal EMA: 4×signal bars AFTER MACD line stabilizes
    
    Total warm-up = 4×slow + 4×signal bars minimum.
    
    Returns list of (ts, macd, signal_line, histogram) tuples with null during warm-up.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # Composite warm-up: slow EMA needs to stabilize first, then signal EMA
    slow_warmup = 4 * slow  # Slow EMA warm-up  
    signal_warmup = 4 * signal  # Additional warm-up for signal line
    total_warmup = slow_warmup + signal_warmup  # Conservative: both must be stable
    
    query = f"""
    WITH ema_calc AS (
        SELECT 
            ts,
            close,
            exponential_moving_average(close, {fast}) OVER (
                ORDER BY ts 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as raw_ema_fast,
            exponential_moving_average(close, {slow}) OVER (
                ORDER BY ts 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as raw_ema_slow,
            ROW_NUMBER() OVER (ORDER BY ts) as bar_num
        FROM bars
        {where_clause}
    ),
    macd_calc AS (
        SELECT 
            ts,
            close,
            bar_num,
            -- Mask EMAs during warm-up
            CASE WHEN bar_num >= {4 * fast} THEN raw_ema_fast ELSE NULL END as ema_fast,
            CASE WHEN bar_num >= {4 * slow} THEN raw_ema_slow ELSE NULL END as ema_slow,
            -- MACD line only after both EMAs are stable
            CASE 
                WHEN bar_num >= {slow_warmup} THEN raw_ema_fast - raw_ema_slow 
                ELSE NULL 
            END as macd_line_raw
        FROM ema_calc
    ),
    signal_calc AS (
        SELECT 
            ts,
            bar_num,
            macd_line_raw,
            exponential_moving_average(macd_line_raw, {signal}) OVER (
                ORDER BY ts 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as raw_signal_line
        FROM macd_calc
    )
    SELECT 
        ts,
        -- Final masking: all components must be stable
        CASE WHEN bar_num >= {total_warmup} THEN macd_line_raw ELSE NULL END as macd_line,
        CASE WHEN bar_num >= {total_warmup} THEN raw_signal_line ELSE NULL END as signal_line,
        CASE 
            WHEN bar_num >= {total_warmup} THEN macd_line_raw - raw_signal_line 
            ELSE NULL 
        END as histogram
    FROM signal_calc
    ORDER BY ts
    """
    
    # Audit SQL query for future peeking patterns
    audit_sql_query(query, "compute_macd")
    
    return conn.execute(query).fetchall()


@audit_detector 
def compute_atr(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    period: int = 14,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    allow_future: bool = False,
) -> List[Tuple]:
    """
    Compute Average True Range (ATR) with CAUSAL warm-up masking.
    
    *** PR6 CRITICAL FIX: ATR warm-up for stability ***
    
    ATR uses moving average, so needs period+1 bars minimum for stability.
    Returns null during insufficient data period.
    
    Returns list of (ts, atr) tuples with null during warm-up.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # ATR needs at least 'period' bars for meaningful calculation
    min_bars = period  
    
    query = f"""
    WITH tr_calc AS (
        SELECT 
            ts,
            GREATEST(
                high - low,
                ABS(high - LAG(close) OVER (ORDER BY ts)),
                ABS(low - LAG(close) OVER (ORDER BY ts))
            ) as true_range,
            ROW_NUMBER() OVER (ORDER BY ts) as bar_num
        FROM bars
        {where_clause}
    )
    SELECT 
        ts,
        CASE 
            WHEN bar_num >= {min_bars} THEN 
                AVG(true_range) OVER (
                    ORDER BY ts 
                    ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                )
            ELSE NULL  -- Masked during insufficient data
        END as atr
    FROM tr_calc
    ORDER BY ts
    """
    
    # Audit SQL query for future peeking patterns
    audit_sql_query(query, "compute_atr")
    
    return conn.execute(query).fetchall()


@audit_detector
def compute_ema_slope(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    period: int,
    lookback_bars: int = 5,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    allow_future: bool = False,
) -> List[Tuple]:
    """
    Compute EMA slope over lookback period with CAUSAL warm-up.
    
    *** PR6 CRITICAL FIX: Uses warm-up aware EMA + additional slope warm-up ***
    
    Requires EMA to be stable (4×period) PLUS lookback_bars for slope calculation.
    
    Returns list of (ts, ema, slope) tuples with null during warm-up.
    """
    # Use warm-up aware EMA computation
    ema_data = compute_ema(conn, symbol, timeframe, period, start_ts, end_ts, allow_future=allow_future)
    
    if not ema_data:
        return []
    
    # Filter out null EMA values and prepare for slope calculation
    valid_ema_data = [(ts, ema) for ts, ema in ema_data if ema is not None]
    
    if len(valid_ema_data) < lookback_bars + 1:
        return []  # Not enough data for slope calculation
    
    # Convert to SQL for slope calculation
    values = ", ".join([f"('{ts}'::TIMESTAMP, {ema})" for ts, ema in valid_ema_data])
    
    query = f"""
    WITH ema_data (ts, ema) AS (
        SELECT * FROM (VALUES {values}) AS t(ts, ema)
    ),
    slope_calc AS (
        SELECT 
            ts,
            ema,
            LAG(ema, {lookback_bars}) OVER (ORDER BY ts) as lagged_ema,
            ROW_NUMBER() OVER (ORDER BY ts) as row_num
        FROM ema_data
    )
    SELECT 
        ts,
        ema,
        CASE 
            WHEN row_num > {lookback_bars} AND lagged_ema IS NOT NULL AND lagged_ema != 0 THEN
                (ema - lagged_ema) / lagged_ema  -- Relative slope
            ELSE NULL  -- Insufficient data for slope  
        END as slope
    FROM slope_calc
    ORDER BY ts
    """
    
    # Audit SQL query for future peeking patterns
    audit_sql_query(query, "compute_ema_slope")
    
    return conn.execute(query).fetchall()