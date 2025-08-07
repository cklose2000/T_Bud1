import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from duckdb import DuckDBPyConnection
from trading_buddy.core.oos_enforcer import OOSEnforcer


def compute_forward_returns(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    event_ts: str,
    lead_bars: int = 10,
) -> Optional[Dict]:
    """
    Compute forward returns and maximum drawdown over lead_bars horizon.
    Returns None if insufficient data.
    """
    query = f"""
    WITH entry AS (
        SELECT close as entry_price
        FROM bars
        WHERE symbol = '{symbol}' 
        AND timeframe = '{timeframe}'
        AND ts = '{event_ts}'
        LIMIT 1
    ),
    forward_bars AS (
        SELECT 
            ts,
            close,
            low,
            ROW_NUMBER() OVER (ORDER BY ts) as bar_num
        FROM bars
        WHERE symbol = '{symbol}'
        AND timeframe = '{timeframe}'
        AND ts > '{event_ts}'
        ORDER BY ts
        LIMIT {lead_bars}
    )
    SELECT 
        e.entry_price,
        f.close as exit_price,
        MIN(f.low) as min_low,
        COUNT(*) as actual_bars
    FROM entry e
    CROSS JOIN forward_bars f
    WHERE f.bar_num = {lead_bars}
    GROUP BY e.entry_price, f.close
    """
    
    result = conn.execute(query).fetchone()
    
    if not result or result[3] < lead_bars:
        return None
    
    entry_price, exit_price, min_low, actual_bars = result
    
    # Calculate log return
    fwd_ret = math.log(exit_price / entry_price)
    
    # Calculate maximum drawdown
    max_dd = (min_low / entry_price) - 1.0
    
    return {
        "fwd_ret": fwd_ret,
        "max_dd": max_dd,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "actual_bars": actual_bars,
    }


def label_outcome(fwd_ret: float, epsilon: float = 0.0001) -> str:
    """
    Label outcome as up/flat/down based on forward return.
    """
    if fwd_ret > epsilon:
        return "up"
    elif fwd_ret < -epsilon:
        return "down"
    else:
        return "flat"


def update_event_outcomes(
    conn: DuckDBPyConnection,
    events: List[Dict],
    lead_bars: int = 10,
) -> int:
    """
    Update events table with computed outcomes.
    Returns number of events updated.
    """
    updated = 0
    
    for event in events:
        outcomes = compute_forward_returns(
            conn,
            event["symbol"],
            event["timeframe"],
            event["event_ts"],
            lead_bars,
        )
        
        if outcomes:
            label = label_outcome(outcomes["fwd_ret"])
            
            # Update the event record
            update_query = """
            UPDATE events
            SET 
                lead_bars = ?,
                fwd_ret = ?,
                max_dd = ?,
                label = ?
            WHERE 
                symbol = ?
                AND timeframe = ?
                AND event_ts = ?
                AND pattern = ?
            """
            
            conn.execute(update_query, [
                lead_bars,
                outcomes["fwd_ret"],
                outcomes["max_dd"],
                label,
                event["symbol"],
                event["timeframe"],
                event["event_ts"],
                event["pattern"],
            ])
            
            updated += 1
    
    return updated


def compute_pattern_stats(
    conn: DuckDBPyConnection,
    pattern: str,
    timeframe: str = "5m",
    min_samples: int = 30,
) -> Optional[Dict]:
    """
    Compute aggregate statistics for a pattern.
    """
    query = f"""
    WITH pattern_outcomes AS (
        SELECT 
            fwd_ret,
            max_dd,
            label
        FROM events
        WHERE pattern = '{pattern}'
        AND timeframe = '{timeframe}'
        AND fwd_ret IS NOT NULL
    )
    SELECT 
        COUNT(*) as n,
        AVG(fwd_ret) as mean_fwd_ret,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fwd_ret) as median_fwd_ret,
        STDDEV(fwd_ret) as std_fwd_ret,
        AVG(max_dd) as mean_max_dd,
        SUM(CASE WHEN label = 'up' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as hit_rate,
        SUM(CASE WHEN fwd_ret > 0 THEN fwd_ret ELSE 0 END) as sum_wins,
        SUM(CASE WHEN fwd_ret < 0 THEN ABS(fwd_ret) ELSE 0 END) as sum_losses
    FROM pattern_outcomes
    """
    
    result = conn.execute(query).fetchone()
    
    if not result or result[0] < min_samples:
        return None
    
    n, mean_ret, median_ret, std_ret, mean_dd, hit_rate, sum_wins, sum_losses = result
    
    # Calculate profit factor
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else float('inf')
    
    # Calculate Sharpe ratio (simplified)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    
    return {
        "n": n,
        "mean_fwd_ret": mean_ret,
        "median_fwd_ret": median_ret,
        "std_fwd_ret": std_ret,
        "mae": abs(mean_dd),  # Mean Adverse Excursion
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
    }