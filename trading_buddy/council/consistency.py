from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from duckdb import DuckDBPyConnection


def compute_expectancy_with_context(
    conn: DuckDBPyConnection,
    base_tf: str,
    base_pattern: str,
    ctx_tf: str,
    ctx_pattern: str,
    lookback_days: int = 365,
) -> Tuple[float, float, int]:
    """
    Compute expectancy with and without context.
    Returns (exp_with, exp_without, n_samples).
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Get events with context present
    with_context_query = f"""
    WITH base_events AS (
        SELECT 
            e.symbol,
            e.event_ts,
            e.fwd_ret
        FROM events e
        WHERE e.pattern = '{base_pattern}'
        AND e.timeframe = '{base_tf}'
        AND e.fwd_ret IS NOT NULL
        AND e.event_ts >= '{start_date}'
        AND e.event_ts <= '{end_date}'
    ),
    context_window AS (
        SELECT 
            be.symbol,
            be.event_ts,
            be.fwd_ret,
            COUNT(c.event_ts) > 0 as has_context
        FROM base_events be
        LEFT JOIN contexts c ON 
            c.symbol = be.symbol
            AND c.timeframe = '{ctx_tf}'
            AND c.context = '{ctx_pattern}'
            AND c.event_ts BETWEEN be.event_ts - INTERVAL '1 hour' AND be.event_ts + INTERVAL '1 hour'
        GROUP BY be.symbol, be.event_ts, be.fwd_ret
    )
    SELECT 
        AVG(CASE WHEN has_context THEN fwd_ret END) as exp_with,
        AVG(CASE WHEN NOT has_context THEN fwd_ret END) as exp_without,
        SUM(CASE WHEN has_context THEN 1 ELSE 0 END) as n_with,
        SUM(CASE WHEN NOT has_context THEN 1 ELSE 0 END) as n_without
    FROM context_window
    """
    
    result = conn.execute(with_context_query).fetchone()
    
    if not result:
        return 0.0, 0.0, 0
    
    exp_with = result[0] or 0.0
    exp_without = result[1] or 0.0
    n_with = result[2] or 0
    
    return exp_with, exp_without, n_with


def compute_hit_rate_with_context(
    conn: DuckDBPyConnection,
    base_tf: str,
    base_pattern: str,
    ctx_tf: str,
    ctx_pattern: str,
    lookback_days: int = 365,
) -> float:
    """
    Compute hit rate when context is present.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    query = f"""
    WITH base_events AS (
        SELECT 
            e.symbol,
            e.event_ts,
            e.label
        FROM events e
        WHERE e.pattern = '{base_pattern}'
        AND e.timeframe = '{base_tf}'
        AND e.label IS NOT NULL
        AND e.event_ts >= '{start_date}'
        AND e.event_ts <= '{end_date}'
    ),
    with_context AS (
        SELECT 
            be.label
        FROM base_events be
        INNER JOIN contexts c ON 
            c.symbol = be.symbol
            AND c.timeframe = '{ctx_tf}'
            AND c.context = '{ctx_pattern}'
            AND c.event_ts BETWEEN be.event_ts - INTERVAL '1 hour' AND be.event_ts + INTERVAL '1 hour'
    )
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN label = 'up' THEN 1 ELSE 0 END) as wins
    FROM with_context
    """
    
    result = conn.execute(query).fetchone()
    
    if not result or result[0] == 0:
        return 0.0
    
    return result[1] / result[0]


def compute_stability(
    conn: DuckDBPyConnection,
    base_tf: str,
    base_pattern: str,
    ctx_tf: str,
    ctx_pattern: str,
    window_days: int = 60,
) -> float:
    """
    Compute stability metric using rolling windows.
    Returns a value between 0 and 1.
    """
    # Calculate returns for two consecutive windows
    current_end = datetime.now()
    current_start = current_end - timedelta(days=window_days)
    prev_end = current_start
    prev_start = prev_end - timedelta(days=window_days)
    
    # Get returns for both windows
    returns_query = f"""
    WITH base_events AS (
        SELECT 
            e.symbol,
            e.event_ts,
            e.fwd_ret,
            CASE 
                WHEN e.event_ts >= '{current_start}' THEN 'current'
                WHEN e.event_ts >= '{prev_start}' THEN 'previous'
            END as window
        FROM events e
        INNER JOIN contexts c ON 
            c.symbol = e.symbol
            AND c.timeframe = '{ctx_tf}'
            AND c.context = '{ctx_pattern}'
            AND c.event_ts BETWEEN e.event_ts - INTERVAL '1 hour' AND e.event_ts + INTERVAL '1 hour'
        WHERE e.pattern = '{base_pattern}'
        AND e.timeframe = '{base_tf}'
        AND e.fwd_ret IS NOT NULL
        AND e.event_ts >= '{prev_start}'
    )
    SELECT 
        window,
        AVG(fwd_ret) as mean_ret,
        STDDEV(fwd_ret) as std_ret,
        COUNT(*) as n
    FROM base_events
    GROUP BY window
    """
    
    results = conn.execute(returns_query).fetchall()
    
    if len(results) < 2:
        return 0.5  # Default stability if insufficient data
    
    # Simple stability: 1 - abs(mean_diff) / (std_pool)
    stats = {row[0]: {"mean": row[1], "std": row[2], "n": row[3]} for row in results}
    
    if "current" not in stats or "previous" not in stats:
        return 0.5
    
    mean_diff = abs(stats["current"]["mean"] - stats["previous"]["mean"])
    pooled_std = ((stats["current"]["std"] + stats["previous"]["std"]) / 2) or 0.01
    
    stability = max(0, min(1, 1 - (mean_diff / pooled_std)))
    
    return stability


def calculate_consistency_index(
    exp_with: float,
    exp_without: float,
    hit_rate: float,
    stability: float,
) -> float:
    """
    Calculate the Consistency Index (CI).
    CI = max(exp_with - exp_without, 0) * hit_rate * stability
    """
    lift = max(exp_with - exp_without, 0)
    ci = lift * hit_rate * stability
    return ci


def update_consistency_matrix(
    conn: DuckDBPyConnection,
    base_tf: str = "5m",
    base_pattern: str = "double_bottom_macd_bull",
    context_configs: Optional[List[Tuple[str, str]]] = None,
):
    """
    Update the consistency matrix for given base pattern and contexts.
    """
    if context_configs is None:
        context_configs = [
            ("15m", "trend_up_ema20"),
            ("60m", "regime_trend"),
            ("1d", "regime_trend"),
        ]
    
    for ctx_tf, ctx_pattern in context_configs:
        # Compute metrics
        exp_with, exp_without, n = compute_expectancy_with_context(
            conn, base_tf, base_pattern, ctx_tf, ctx_pattern
        )
        
        if n < 30:  # Minimum sample size
            continue
        
        hit_rate = compute_hit_rate_with_context(
            conn, base_tf, base_pattern, ctx_tf, ctx_pattern
        )
        
        stability = compute_stability(
            conn, base_tf, base_pattern, ctx_tf, ctx_pattern
        )
        
        ci = calculate_consistency_index(exp_with, exp_without, hit_rate, stability)
        
        # Update matrix
        insert_query = """
        INSERT OR REPLACE INTO consistency_matrix (
            base_tf, base_pattern, ctx_tf, ctx_pattern,
            n, exp_lift, hit_rate, stability, ci, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        
        conn.execute(insert_query, [
            base_tf, base_pattern, ctx_tf, ctx_pattern,
            n, exp_with - exp_without, hit_rate, stability, ci
        ])