"""
Simplified indicator calculations for DuckDB without built-in EMA.
"""
from typing import List, Optional, Tuple
import numpy as np
from duckdb import DuckDBPyConnection


def calculate_ema_python(prices: List[float], period: int) -> List[float]:
    """Calculate EMA using Python."""
    if not prices or period <= 0:
        return []
    
    ema = []
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first value
    if len(prices) >= period:
        ema.append(sum(prices[:period]) / period)
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema_val = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_val)
    
    # Pad with None for initial values
    return [None] * (len(prices) - len(ema)) + ema


def compute_macd_simple(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Tuple]:
    """
    Compute MACD using simple moving averages as approximation.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # Get price data
    query = f"""
    SELECT ts, close
    FROM bars
    {where_clause}
    ORDER BY ts
    """
    
    data = conn.execute(query).fetchall()
    
    if len(data) < slow + signal_period:
        return []
    
    # Extract prices
    timestamps = [row[0] for row in data]
    prices = [row[1] for row in data]
    
    # Calculate EMAs
    ema_fast = calculate_ema_python(prices, fast)
    ema_slow = calculate_ema_python(prices, slow)
    
    # Calculate MACD line
    macd_line = []
    for i in range(len(prices)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    # Calculate signal line (EMA of MACD)
    macd_values = [v for v in macd_line if v is not None]
    signal_ema = calculate_ema_python(macd_values, signal_period)
    
    # Align signal line with original data
    signal_line = [None] * (len(macd_line) - len(signal_ema)) + signal_ema
    
    # Build result
    result = []
    for i in range(len(timestamps)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram = macd_line[i] - signal_line[i]
            result.append((timestamps[i], macd_line[i], signal_line[i], histogram))
    
    return result


def compute_sma_based_macd(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Tuple]:
    """
    Compute MACD using SQL with SMA approximation.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # Use simple moving averages as approximation
    query = f"""
    WITH price_data AS (
        SELECT 
            ts,
            close,
            ROW_NUMBER() OVER (ORDER BY ts) as rn
        FROM bars
        {where_clause}
    ),
    sma_calc AS (
        SELECT 
            ts,
            close,
            AVG(close) OVER (ORDER BY ts ROWS BETWEEN {fast-1} PRECEDING AND CURRENT ROW) as sma_fast,
            AVG(close) OVER (ORDER BY ts ROWS BETWEEN {slow-1} PRECEDING AND CURRENT ROW) as sma_slow,
            rn
        FROM price_data
    ),
    macd_calc AS (
        SELECT 
            ts,
            close,
            sma_fast - sma_slow as macd_line,
            rn
        FROM sma_calc
        WHERE rn >= {slow}  -- Ensure we have enough data
    )
    SELECT 
        ts,
        macd_line,
        AVG(macd_line) OVER (ORDER BY ts ROWS BETWEEN {signal_period-1} PRECEDING AND CURRENT ROW) as signal_line,
        macd_line - AVG(macd_line) OVER (ORDER BY ts ROWS BETWEEN {signal_period-1} PRECEDING AND CURRENT ROW) as histogram
    FROM macd_calc
    WHERE rn >= {slow + signal_period - 1}
    ORDER BY ts
    """
    
    return conn.execute(query).fetchall()