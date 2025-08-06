from typing import List, Optional, Tuple

from duckdb import DuckDBPyConnection


def compute_ema(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    period: int,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Tuple]:
    """
    Compute EMA using DuckDB's exponential_moving_average function.
    Returns list of (ts, ema_value) tuples.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    query = f"""
    SELECT 
        ts,
        exponential_moving_average(close, {period}) OVER (
            ORDER BY ts 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as ema_{period}
    FROM bars
    {where_clause}
    ORDER BY ts
    """
    
    return conn.execute(query).fetchall()


def compute_macd(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Tuple]:
    """
    Compute MACD indicator.
    Returns list of (ts, macd, signal_line, histogram) tuples.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    query = f"""
    WITH ema_calc AS (
        SELECT 
            ts,
            close,
            exponential_moving_average(close, {fast}) OVER (
                ORDER BY ts 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as ema_fast,
            exponential_moving_average(close, {slow}) OVER (
                ORDER BY ts 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as ema_slow
        FROM bars
        {where_clause}
    ),
    macd_calc AS (
        SELECT 
            ts,
            close,
            ema_fast - ema_slow as macd_line
        FROM ema_calc
    )
    SELECT 
        ts,
        macd_line,
        exponential_moving_average(macd_line, {signal}) OVER (
            ORDER BY ts 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as signal_line,
        macd_line - exponential_moving_average(macd_line, {signal}) OVER (
            ORDER BY ts 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as histogram
    FROM macd_calc
    ORDER BY ts
    """
    
    return conn.execute(query).fetchall()


def compute_atr(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    period: int = 14,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Tuple]:
    """
    Compute Average True Range (ATR).
    Returns list of (ts, atr) tuples.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    query = f"""
    WITH tr_calc AS (
        SELECT 
            ts,
            GREATEST(
                high - low,
                ABS(high - LAG(close) OVER (ORDER BY ts)),
                ABS(low - LAG(close) OVER (ORDER BY ts))
            ) as true_range
        FROM bars
        {where_clause}
    )
    SELECT 
        ts,
        AVG(true_range) OVER (
            ORDER BY ts 
            ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
        ) as atr
    FROM tr_calc
    ORDER BY ts
    """
    
    return conn.execute(query).fetchall()


def compute_ema_slope(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    period: int,
    lookback_bars: int = 5,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Tuple]:
    """
    Compute EMA slope over lookback period.
    Returns list of (ts, ema, slope) tuples.
    """
    ema_data = compute_ema(conn, symbol, timeframe, period, start_ts, end_ts)
    
    if not ema_data:
        return []
    
    # Convert to SQL for slope calculation
    values = ", ".join([f"('{ts}'::TIMESTAMP, {ema})" for ts, ema in ema_data if ema is not None])
    
    query = f"""
    WITH ema_data (ts, ema) AS (
        SELECT * FROM (VALUES {values}) AS t(ts, ema)
    )
    SELECT 
        ts,
        ema,
        (ema - LAG(ema, {lookback_bars}) OVER (ORDER BY ts)) / 
        NULLIF(LAG(ema, {lookback_bars}) OVER (ORDER BY ts), 0) as slope
    FROM ema_data
    ORDER BY ts
    """
    
    return conn.execute(query).fetchall()