from typing import Dict, List, Optional

from duckdb import DuckDBPyConnection


def detect_trend_context(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str = "15m",
    ema_period: int = 20,
    slope_lookback: int = 5,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Dict]:
    """
    Detect trend context: price above SMA and SMA slope positive.
    Using SMA as approximation for EMA.
    """
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # Use SMA instead of EMA for simplicity
    query = f"""
    WITH sma_calc AS (
        SELECT 
            ts,
            close,
            AVG(close) OVER (
                ORDER BY ts 
                ROWS BETWEEN {ema_period - 1} PRECEDING AND CURRENT ROW
            ) as sma,
            ROW_NUMBER() OVER (ORDER BY ts) as rn
        FROM bars
        {where_clause}
    ),
    slope_calc AS (
        SELECT 
            ts,
            close,
            sma,
            (sma - LAG(sma, {slope_lookback}) OVER (ORDER BY ts)) / 
            NULLIF(LAG(sma, {slope_lookback}) OVER (ORDER BY ts), 0) as slope
        FROM sma_calc
        WHERE rn >= {ema_period}
    )
    SELECT ts, close, sma, slope
    FROM slope_calc
    WHERE slope IS NOT NULL
    ORDER BY ts
    """
    
    results = conn.execute(query).fetchall()
    contexts = []
    
    for ts, close, sma, slope in results:
        # Trend up: close > SMA and slope > 0
        if close > sma and slope > 0:
            contexts.append({
                "event_ts": ts,
                "context": "trend_up_ema20",
                "attrs": {
                    "close": close,
                    "ema": sma,  # Using SMA as approximation
                    "slope": slope,
                    "distance_pct": (close - sma) / sma,
                }
            })
    
    return contexts


def detect_regime(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str = "60m",
    lookback_periods: int = 20,
    atr_period: int = 14,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Dict]:
    """
    Detect market regime: trend/mrv/chop based on composite indicators.
    """
    # Simplified ATR calculation inline
    
    where_clause = f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
    if start_ts:
        where_clause += f" AND ts >= '{start_ts}'"
    if end_ts:
        where_clause += f" AND ts <= '{end_ts}'"
    
    # Simplified ATR calculation will be done inline
    
    # Calculate price slope and ADX (simplified)
    regime_query = f"""
    WITH price_data AS (
        SELECT 
            ts,
            close,
            -- Price return over lookback
            (close - LAG(close, {lookback_periods}) OVER (ORDER BY ts)) / 
            NULLIF(LAG(close, {lookback_periods}) OVER (ORDER BY ts), 0) as return_slope,
            -- Simplified directional movement
            high - LAG(high, 1) OVER (ORDER BY ts) as plus_dm,
            LAG(low, 1) OVER (ORDER BY ts) - low as minus_dm
        FROM bars
        {where_clause}
    ),
    adx_calc AS (
        SELECT 
            ts,
            close,
            return_slope,
            -- Simplified ADX calculation
            AVG(ABS(plus_dm - minus_dm) / NULLIF(plus_dm + minus_dm, 0)) 
                OVER (ORDER BY ts ROWS BETWEEN {adx_period - 1} PRECEDING AND CURRENT ROW) * 100 as adx
        FROM price_data
    )
    SELECT ts, close, return_slope, adx
    FROM adx_calc
    ORDER BY ts
    """
    
    regime_data = conn.execute(regime_query).fetchall()
    
    contexts = []
    
    for ts, close, return_slope, adx in regime_data:
        if return_slope is None:
            continue
        
        # Use a simple volatility measure instead of ATR
        atr = abs(return_slope) * close * 0.01  # Simplified volatility proxy
        
        # Determine regime
        if adx and adx > adx_threshold and abs(return_slope) > 0.02:
            # Strong directional movement
            regime = "regime_trend" if return_slope > 0 else "regime_trend_down"
        elif adx and adx < 20 and abs(return_slope) < 0.01:
            # Low directional movement, small returns
            regime = "regime_chop"
        else:
            # Mean reversion conditions
            regime = "regime_mrv"
        
        contexts.append({
            "event_ts": ts,
            "context": regime,
            "attrs": {
                "return_slope": return_slope,
                "adx": adx or 0,
                "atr": atr,
                "close": close,
            }
        })
    
    return contexts


def detect_large_drop(
    conn: DuckDBPyConnection,
    symbol: str,
    lookback: str = "1d",
    threshold_pct: float = -1.5,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> List[Dict]:
    """
    Detect large drop context over specified lookback period.
    """
    # Convert lookback to hours
    lookback_hours = {
        "1h": 1, "4h": 4, "12h": 12, "1d": 24, "2d": 48, "1w": 168
    }.get(lookback, 24)
    
    where_clause = "WHERE symbol = ?"
    params = [symbol]
    
    if start_ts:
        where_clause += " AND ts >= ?"
        params.append(start_ts)
    if end_ts:
        where_clause += " AND ts <= ?"
        params.append(end_ts)
    
    query = f"""
    WITH price_changes AS (
        SELECT 
            ts,
            close,
            MIN(close) OVER (
                ORDER BY ts 
                RANGE BETWEEN INTERVAL '{lookback_hours} hours' PRECEDING AND CURRENT ROW
            ) as min_price,
            MAX(close) OVER (
                ORDER BY ts 
                RANGE BETWEEN INTERVAL '{lookback_hours} hours' PRECEDING AND CURRENT ROW
            ) as max_price
        FROM bars
        {where_clause}
    )
    SELECT 
        ts,
        close,
        (close - max_price) / max_price * 100 as drop_pct
    FROM price_changes
    WHERE (close - max_price) / max_price * 100 <= ?
    ORDER BY ts
    """
    
    params.append(threshold_pct)
    results = conn.execute(query, params).fetchall()
    
    contexts = []
    for ts, close, drop_pct in results:
        contexts.append({
            "event_ts": ts,
            "context": "large_drop",
            "attrs": {
                "close": close,
                "drop_pct": drop_pct,
                "lookback": lookback,
                "threshold_pct": threshold_pct,
            }
        })
    
    return contexts