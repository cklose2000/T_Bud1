"""
Statistical stability guards for pattern validation.
Includes Benjamini-Hochberg for multiple testing and KS/PSI for drift.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
from duckdb import DuckDBPyConnection


def benjamini_hochberg_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[bool], float]:
    """
    Apply Benjamini-Hochberg correction for multiple hypothesis testing.
    
    Args:
        p_values: List of p-values from individual tests
        alpha: Family-wise error rate (default 0.05)
        
    Returns:
        Tuple of (reject_null list, adjusted alpha threshold)
    """
    if not p_values:
        return [], 0.0
    
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Calculate BH thresholds
    bh_thresholds = alpha * (np.arange(1, n + 1) / n)
    
    # Find largest i where P(i) <= threshold(i)
    reject = sorted_p <= bh_thresholds
    
    # If any are rejected, find the threshold
    if np.any(reject):
        max_i = np.max(np.where(reject)[0])
        threshold = bh_thresholds[max_i]
    else:
        threshold = 0.0
    
    # Map back to original order
    reject_original = np.zeros(n, dtype=bool)
    reject_original[sorted_indices] = reject
    
    return reject_original.tolist(), threshold


def calculate_ks_statistic(
    historical: List[float],
    recent: List[float]
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic between two samples.
    
    Args:
        historical: Historical returns/values
        recent: Recent returns/values
        
    Returns:
        Tuple of (ks_statistic, p_value)
    """
    if len(historical) < 20 or len(recent) < 10:
        # Not enough data
        return 0.0, 1.0
    
    # Run two-sample KS test
    ks_stat, p_value = stats.ks_2samp(historical, recent)
    
    return ks_stat, p_value


def calculate_psi(
    historical: List[float],
    recent: List[float],
    n_bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    PSI interpretation:
    - < 0.1: No significant change
    - 0.1-0.2: Moderate change
    - > 0.2: Significant change
    
    Args:
        historical: Historical values
        recent: Recent values
        n_bins: Number of bins for discretization
        
    Returns:
        PSI value
    """
    if len(historical) < 30 or len(recent) < 10:
        return 0.0
    
    # Create bins based on historical data
    _, bin_edges = np.histogram(historical, bins=n_bins)
    
    # Calculate frequencies
    hist_freq, _ = np.histogram(historical, bins=bin_edges)
    recent_freq, _ = np.histogram(recent, bins=bin_edges)
    
    # Convert to proportions
    hist_prop = (hist_freq + 1) / (len(historical) + n_bins)  # Add 1 to avoid log(0)
    recent_prop = (recent_freq + 1) / (len(recent) + n_bins)
    
    # Calculate PSI
    psi = np.sum((recent_prop - hist_prop) * np.log(recent_prop / hist_prop))
    
    return psi


def calculate_rolling_stability(
    conn: DuckDBPyConnection,
    symbol: str,
    pattern: str,
    timeframe: str,
    window_days: int = 30,
    recent_days: int = 7
) -> Dict[str, float]:
    """
    Calculate stability metrics for a pattern.
    
    Returns:
        Dict with stability metrics
    """
    # Get historical and recent returns
    query = """
    WITH pattern_returns AS (
        SELECT 
            event_ts,
            fwd_ret,
            CASE 
                WHEN event_ts >= CURRENT_DATE - INTERVAL ? DAY THEN 'recent'
                ELSE 'historical'
            END as period
        FROM events
        WHERE symbol = ?
        AND pattern = ?
        AND timeframe = ?
        AND event_ts >= CURRENT_DATE - INTERVAL ? DAY
        AND fwd_ret IS NOT NULL
        AND oos_split IN ('test', 'live')  -- Only OOS data
    )
    SELECT period, fwd_ret
    FROM pattern_returns
    ORDER BY event_ts
    """
    
    # Format query with parameters
    formatted_query = f"""
    WITH pattern_returns AS (
        SELECT 
            event_ts,
            fwd_ret,
            CASE 
                WHEN event_ts >= CURRENT_DATE - INTERVAL {recent_days} DAY THEN 'recent'
                ELSE 'historical'
            END as period
        FROM events
        WHERE symbol = '{symbol}'
        AND pattern = '{pattern}'
        AND timeframe = '{timeframe}'
        AND event_ts >= CURRENT_DATE - INTERVAL {window_days} DAY
        AND fwd_ret IS NOT NULL
        AND oos_split IN ('test', 'live')  -- Only OOS data
    )
    SELECT period, fwd_ret
    FROM pattern_returns
    ORDER BY event_ts
    """
    results = conn.execute(formatted_query).fetchall()
    
    if not results:
        return {
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "psi": 0.0,
            "stability_score": 1.0,
            "n_historical": 0,
            "n_recent": 0
        }
    
    # Separate historical and recent
    historical = [r[1] for r in results if r[0] == 'historical']
    recent = [r[1] for r in results if r[0] == 'recent']
    
    # Calculate metrics
    ks_stat, ks_pval = calculate_ks_statistic(historical, recent)
    psi = calculate_psi(historical, recent)
    
    # Calculate stability score (0 to 1, higher is more stable)
    # Combine KS p-value and PSI into single score
    ks_stability = ks_pval  # High p-value = stable
    psi_stability = max(0, 1 - psi / 0.2)  # PSI < 0.2 is acceptable
    
    stability_score = 0.7 * ks_stability + 0.3 * psi_stability
    
    return {
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_pval, 4),
        "psi": round(psi, 4),
        "stability_score": round(stability_score, 4),
        "n_historical": len(historical),
        "n_recent": len(recent)
    }


def validate_pattern_candidates(
    conn: DuckDBPyConnection,
    candidates: List[Dict[str, str]],
    alpha: float = 0.05
) -> List[Dict]:
    """
    Validate pattern candidates with multiple testing correction.
    
    Args:
        conn: Database connection
        candidates: List of pattern candidates with symbol, pattern, timeframe
        alpha: Significance level
        
    Returns:
        List of validated candidates with statistics
    """
    # Calculate p-values for each candidate
    results = []
    p_values = []
    
    for candidate in candidates:
        # Test if pattern returns are significantly different from zero
        query = """
        SELECT fwd_ret
        FROM events
        WHERE symbol = ?
        AND pattern = ?
        AND timeframe = ?
        AND fwd_ret IS NOT NULL
        AND oos_split IN ('test', 'live')
        """
        
        # Format query
        formatted_query = f"""
        SELECT fwd_ret
        FROM events
        WHERE symbol = '{candidate['symbol']}'
        AND pattern = '{candidate['pattern']}'
        AND timeframe = '{candidate['timeframe']}'
        AND fwd_ret IS NOT NULL
        AND oos_split IN ('test', 'live')
        """
        returns = [r[0] for r in conn.execute(formatted_query).fetchall()]
        
        if len(returns) >= 30:
            # One-sample t-test against zero
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            p_values.append(p_value)
            
            results.append({
                **candidate,
                "n": len(returns),
                "mean_return": np.mean(returns),
                "t_stat": t_stat,
                "p_value": p_value
            })
        else:
            # Not enough data
            p_values.append(1.0)
            results.append({
                **candidate,
                "n": len(returns),
                "mean_return": np.mean(returns) if returns else 0,
                "t_stat": 0,
                "p_value": 1.0
            })
    
    # Apply Benjamini-Hochberg correction
    reject_null, threshold = benjamini_hochberg_correction(p_values, alpha)
    
    # Add validation results
    for i, result in enumerate(results):
        result["significant"] = reject_null[i]
        result["bh_threshold"] = threshold
        
        # Also calculate stability
        stability = calculate_rolling_stability(
            conn,
            result['symbol'],
            result['pattern'],
            result['timeframe']
        )
        result["stability_score"] = stability["stability_score"]
        result["psi"] = stability["psi"]
    
    return results


def apply_stability_adjustment(
    base_ci: float,
    stability_score: float,
    min_stability: float = 0.4
) -> float:
    """
    Adjust CI based on stability score.
    
    Args:
        base_ci: Original consistency index
        stability_score: Stability score (0-1)
        min_stability: Minimum acceptable stability
        
    Returns:
        Adjusted CI
    """
    if stability_score < min_stability:
        # Heavily penalize unstable patterns
        adjustment = stability_score / min_stability
        return base_ci * adjustment * 0.5  # Extra penalty
    else:
        # Mild adjustment for stable patterns
        adjustment = 0.8 + 0.2 * stability_score
        return base_ci * adjustment


def check_pattern_drift(
    conn: DuckDBPyConnection,
    symbol: str,
    pattern: str,
    timeframe: str,
    alert_threshold: float = 0.2
) -> Dict[str, any]:
    """
    Check if a pattern is experiencing drift.
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        pattern: Pattern name
        timeframe: Timeframe
        alert_threshold: PSI threshold for drift alert
        
    Returns:
        Dict with drift status and metrics
    """
    stability = calculate_rolling_stability(conn, symbol, pattern, timeframe)
    
    is_drifting = stability["psi"] > alert_threshold
    
    # Get recent performance change
    perf_query = """
    SELECT 
        AVG(CASE WHEN event_ts >= CURRENT_DATE - INTERVAL 7 DAY THEN fwd_ret END) as recent_mean,
        AVG(CASE WHEN event_ts < CURRENT_DATE - INTERVAL 7 DAY THEN fwd_ret END) as historical_mean
    FROM events
    WHERE symbol = ?
    AND pattern = ?
    AND timeframe = ?
    AND event_ts >= CURRENT_DATE - INTERVAL 30 DAY
    AND fwd_ret IS NOT NULL
    AND oos_split IN ('test', 'live')
    """
    
    # Format query
    formatted_perf_query = f"""
    SELECT 
        AVG(CASE WHEN event_ts >= CURRENT_DATE - INTERVAL 7 DAY THEN fwd_ret END) as recent_mean,
        AVG(CASE WHEN event_ts < CURRENT_DATE - INTERVAL 7 DAY THEN fwd_ret END) as historical_mean
    FROM events
    WHERE symbol = '{symbol}'
    AND pattern = '{pattern}'
    AND timeframe = '{timeframe}'
    AND event_ts >= CURRENT_DATE - INTERVAL 30 DAY
    AND fwd_ret IS NOT NULL
    AND oos_split IN ('test', 'live')
    """
    perf_result = conn.execute(formatted_perf_query).fetchone()
    
    recent_mean = perf_result[0] if perf_result[0] else 0
    historical_mean = perf_result[1] if perf_result[1] else 0
    
    return {
        "is_drifting": is_drifting,
        "psi": stability["psi"],
        "stability_score": stability["stability_score"],
        "recent_mean": round(recent_mean, 4),
        "historical_mean": round(historical_mean, 4),
        "performance_change": round(recent_mean - historical_mean, 4),
        "recommendation": "pause_trading" if is_drifting else "continue"
    }