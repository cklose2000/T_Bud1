"""
Build compressed daily reports for ticker memory.
Compresses raw events into 2-5KB Markdown + JSON bundles.
"""
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import logging

from duckdb import DuckDBPyConnection
from trading_buddy.council.stability import (
    calculate_rolling_stability,
    check_pattern_drift
)

logger = logging.getLogger(__name__)


def get_top_edges(
    conn: DuckDBPyConnection,
    symbol: str,
    as_of: date,
    lookback_days: int = 60,
    top_n: int = 5
) -> List[Dict]:
    """Get top performing edges/patterns for a symbol."""
    query = """
    WITH pattern_stats AS (
        SELECT 
            pattern,
            timeframe,
            COUNT(*) as n,
            MEDIAN(fwd_ret) as median_r,
            AVG(CASE WHEN fwd_ret > 0 THEN 1 ELSE 0 END) as hit_rate,
            STDDEV(fwd_ret) as mae,  -- Using stddev as proxy for MAE
            MAX(event_ts) as last_seen
        FROM events
        WHERE symbol = ?
        AND event_ts >= ? - INTERVAL ? DAY
        AND event_ts < ?
        AND oos_split IN ('test', 'live')  -- Only OOS data
        GROUP BY pattern, timeframe
        HAVING COUNT(*) >= 10  -- Minimum sample size
    ),
    ci_data AS (
        SELECT 
            base_pattern as pattern,
            base_tf as timeframe,
            AVG(ci) as avg_ci
        FROM consistency_matrix
        WHERE base_pattern IN (SELECT pattern FROM pattern_stats)
        GROUP BY base_pattern, base_tf
    )
    SELECT 
        ps.pattern,
        ps.timeframe,
        ps.n,
        ps.median_r,
        ps.hit_rate,
        ps.mae,
        COALESCE(cd.avg_ci, 0.5) as ci,
        ps.last_seen
    FROM pattern_stats ps
    LEFT JOIN ci_data cd ON ps.pattern = cd.pattern AND ps.timeframe = cd.timeframe
    ORDER BY ps.median_r * ps.hit_rate * COALESCE(cd.avg_ci, 0.5) DESC
    LIMIT ?
    """
    
    # Execute with properly formatted parameters
    results = conn.execute(f"""
    WITH pattern_stats AS (
        SELECT 
            pattern,
            timeframe,
            COUNT(*) as n,
            MEDIAN(fwd_ret) as median_r,
            AVG(CASE WHEN fwd_ret > 0 THEN 1 ELSE 0 END) as hit_rate,
            STDDEV(fwd_ret) as mae,  -- Using stddev as proxy for MAE
            MAX(event_ts) as last_seen
        FROM events
        WHERE symbol = '{symbol}'
        AND event_ts >= DATE '{as_of}' - INTERVAL {lookback_days} DAY
        AND event_ts < DATE '{as_of}'
        AND oos_split IN ('test', 'live')  -- Only OOS data
        GROUP BY pattern, timeframe
        HAVING COUNT(*) >= 10  -- Minimum sample size
    ),
    ci_data AS (
        SELECT 
            base_pattern as pattern,
            base_tf as timeframe,
            AVG(ci) as avg_ci
        FROM consistency_matrix
        WHERE base_pattern IN (SELECT pattern FROM pattern_stats)
        GROUP BY base_pattern, base_tf
    )
    SELECT 
        ps.pattern,
        ps.timeframe,
        ps.n,
        ps.median_r,
        ps.hit_rate,
        ps.mae,
        COALESCE(cd.avg_ci, 0.5) as ci,
        ps.last_seen
    FROM pattern_stats ps
    LEFT JOIN ci_data cd ON ps.pattern = cd.pattern AND ps.timeframe = cd.timeframe
    ORDER BY ps.median_r * ps.hit_rate * COALESCE(cd.avg_ci, 0.5) DESC
    LIMIT {top_n}
    """).fetchall()
    
    return [
        {
            "pattern": r[0],
            "timeframe": r[1],
            "n": r[2],
            "median_r": round(r[3], 4),
            "hit_rate": round(r[4], 3),
            "mae": round(r[5], 4),
            "ci": round(r[6], 3),
            "last_seen": r[7].isoformat() if r[7] else None
        }
        for r in results
    ]


def detect_pattern_drift(
    conn: DuckDBPyConnection,
    symbol: str,
    as_of: date,
    window_days: int = 30
) -> Dict:
    """
    Detect patterns that have decayed or show drift using stability module.
    """
    # Get patterns that exist for this symbol  
    pattern_query = f"""
    SELECT DISTINCT pattern, timeframe
    FROM events
    WHERE symbol = '{symbol}'
    AND event_ts >= DATE '{as_of}' - INTERVAL {window_days} DAY
    AND event_ts < DATE '{as_of}'
    AND oos_split IN ('test', 'live')  -- Only OOS data
    GROUP BY pattern, timeframe
    HAVING COUNT(*) >= 10  -- Minimum for drift analysis
    """
    
    patterns = conn.execute(pattern_query).fetchall()
    
    drifted = []
    retired = []
    stable = []
    
    for pattern, timeframe in patterns:
        try:
            # Check drift using stability module
            drift_info = check_pattern_drift(
                conn, symbol, pattern, timeframe, alert_threshold=0.2
            )
            
            # Get stability score
            stability_info = calculate_rolling_stability(
                conn, symbol, pattern, timeframe, window_days=window_days
            )
            
            pattern_info = {
                "pattern": pattern,
                "timeframe": timeframe,
                "recent_mean": drift_info["recent_mean"],
                "historical_mean": drift_info["historical_mean"],
                "performance_change": drift_info["performance_change"],
                "psi": drift_info["psi"],
                "stability_score": drift_info["stability_score"],
                "is_drifting": drift_info["is_drifting"],
                "recommendation": drift_info["recommendation"],
                "n_recent": stability_info["n_recent"],
                "n_historical": stability_info["n_historical"]
            }
            
            # Classify patterns
            if drift_info["is_drifting"]:
                if drift_info["recent_mean"] < 0 and drift_info["historical_mean"] > 0:
                    # Performance degraded significantly - retire
                    retired.append(pattern_info)
                else:
                    # Just drifting - monitor
                    drifted.append(pattern_info)
            else:
                stable.append(pattern_info)
                
        except Exception as e:
            logger.warning(f"Failed to analyze drift for {symbol}:{pattern}:{timeframe}: {e}")
            continue
    
    # Sort by severity (PSI for drifted, performance change for retired)
    drifted.sort(key=lambda x: x["psi"], reverse=True)
    retired.sort(key=lambda x: abs(x["performance_change"]), reverse=True)
    
    return {
        "drifted": drifted[:3],  # Top 3 drifted patterns
        "retired": retired[:3],  # Top 3 retired patterns
        "stable": stable[:2],    # Include a few stable ones for reference
        "total_patterns": len(patterns),
        "drift_summary": {
            "drifting": len(drifted),
            "retired": len(retired), 
            "stable": len(stable)
        }
    }


def get_counterfactual_lifts(
    conn: DuckDBPyConnection,
    symbol: str,
    as_of: date
) -> List[Dict]:
    """Calculate counterfactual lifts for common scenarios."""
    # Simplified version - in practice would use more sophisticated analysis
    counterfactuals = []
    
    # Scenario 1: If 60m trend aligns
    query_60m = """
    WITH base_patterns AS (
        SELECT DISTINCT pattern, timeframe
        FROM events
        WHERE symbol = ?
        AND event_ts >= ? - INTERVAL 30 DAY
        AND event_ts < ?
        AND timeframe IN ('5min', '15min')
    )
    SELECT 
        bp.pattern,
        bp.timeframe,
        cm.exp_lift as lift_with_60m_trend,
        cm.hit_rate as hit_rate_with_60m,
        cm.n as sample_size
    FROM base_patterns bp
    LEFT JOIN consistency_matrix cm ON 
        cm.base_pattern = bp.pattern 
        AND cm.base_tf = bp.timeframe
        AND cm.ctx_tf = '60min'
        AND cm.ctx_pattern = 'trend_up'
    WHERE cm.exp_lift > 0.001
    ORDER BY cm.exp_lift DESC
    LIMIT 3
    """
    
    # Format query with parameters, fixing date types
    formatted_query = query_60m.replace('DATE ?', "DATE '{}'").replace('?', '{}').format(
        symbol, as_of, as_of
    )
    results = conn.execute(formatted_query).fetchall()
    
    for r in results:
        counterfactuals.append({
            "scenario": "60m_trend_alignment",
            "pattern": r[0],
            "timeframe": r[1],
            "lift": round(r[2], 4) if r[2] else 0,
            "hit_rate": round(r[3], 3) if r[3] else 0,
            "n": r[4] if r[4] else 0
        })
    
    return counterfactuals


def get_annotated_examples(
    conn: DuckDBPyConnection,
    symbol: str,
    as_of: date,
    n_examples: int = 5
) -> List[Dict]:
    """Get annotated examples of recent pattern occurrences."""
    query = """
    SELECT 
        e.event_ts,
        e.pattern,
        e.timeframe,
        e.attrs,
        e.fwd_ret,
        e.label,
        -- Get price context
        b.open,
        b.high,
        b.low,
        b.close
    FROM events e
    LEFT JOIN bars b ON 
        b.symbol = e.symbol 
        AND b.timeframe = e.timeframe 
        AND b.ts = e.event_ts
    WHERE e.symbol = ?
    AND e.event_ts >= DATE ? - INTERVAL 7 DAY
    AND e.event_ts < ?
    AND e.fwd_ret IS NOT NULL
    ORDER BY ABS(e.fwd_ret) DESC
    LIMIT ?
    """
    
    # Format query with parameters, fixing date types
    formatted_query = query.replace('DATE ?', "DATE '{}'").replace('?', '{}').format(
        symbol, as_of, as_of, n_examples
    )
    results = conn.execute(formatted_query).fetchall()
    
    examples = []
    for r in results:
        example = {
            "ts": r[0].isoformat() if r[0] else None,
            "pattern": r[1],
            "timeframe": r[2],
            "outcome": "win" if r[4] > 0 else "loss",
            "return": round(r[4], 4) if r[4] else 0,
            "price": {
                "open": r[6],
                "high": r[7],
                "low": r[8],
                "close": r[9]
            }
        }
        
        # Add pattern-specific details
        if r[3]:  # attrs JSON
            if r[1] == "double_bottom" or r[1] == "w_pattern":
                example["details"] = f"Bottoms at ${r[3].get('low1_price', 0):.2f} and ${r[3].get('low2_price', 0):.2f}"
            elif r[1] == "macd_bull_cross":
                example["details"] = f"MACD crossed at {r[3].get('macd', 0):.4f}"
        
        examples.append(example)
    
    return examples


def build_summary_markdown(
    symbol: str,
    as_of: date,
    top_edges: List[Dict],
    drift_data: Dict,
    counterfactuals: List[Dict],
    examples: List[Dict]
) -> str:
    """Build a concise markdown summary."""
    lines = [
        f"# {symbol} Daily Report - {as_of}",
        "",
        "## Top Edges (OOS)",
        ""
    ]
    
    # Top edges table
    if top_edges:
        lines.extend([
            "| Pattern | TF | N | Med R | Hit | MAE | CI |",
            "|---------|----|----|-------|-----|-----|-----|"
        ])
        for edge in top_edges[:3]:
            lines.append(
                f"| {edge['pattern'][:20]} | {edge['timeframe']} | "
                f"{edge['n']} | {edge['median_r']:.3f} | "
                f"{edge['hit_rate']:.1%} | {edge['mae']:.3f} | "
                f"{edge['ci']:.2f} |"
            )
    else:
        lines.append("*No significant edges found*")
    
    lines.extend(["", "## Pattern Changes", ""])
    
    # Drift summary first
    if 'drift_summary' in drift_data:
        summary = drift_data['drift_summary']
        lines.append(f"**Status:** {summary['stable']} stable, {summary['drifting']} drifting, {summary['retired']} retired")
        lines.append("")
    
    # Retired patterns (highest priority)
    if drift_data.get('retired'):
        lines.append("**🚫 Retired Patterns (avoid):**")
        for p in drift_data['retired']:
            perf_change = p.get('performance_change', 0)
            lines.append(f"- {p['pattern']} ({p['timeframe']}): {perf_change:+.3f} change (PSI: {p.get('psi', 0):.2f})")
        lines.append("")
    
    # Drifting patterns (monitor closely)
    if drift_data.get('drifted'):
        lines.append("**⚠️ Drifting Patterns (monitor):**")
        for p in drift_data['drifted']:
            psi = p.get('psi', 0)
            stability = p.get('stability_score', 0)
            lines.append(f"- {p['pattern']} ({p['timeframe']}): PSI {psi:.2f}, stability {stability:.2f}")
        lines.append("")
    
    # Stable patterns (for reference)
    if drift_data.get('stable'):
        lines.append("**✅ Stable Patterns:**")
        for p in drift_data['stable'][:2]:  # Just show top 2
            stability = p.get('stability_score', 0)
            lines.append(f"- {p['pattern']} ({p['timeframe']}): stability {stability:.2f}")
    
    # Counterfactuals
    if counterfactuals:
        lines.extend(["", "## What-If Scenarios", ""])
        for cf in counterfactuals[:2]:
            lines.append(f"- If 60m trend up: {cf['pattern']} gets +{cf['lift']:.3f} lift")
    
    # Examples
    if examples:
        lines.extend(["", "## Recent Examples", ""])
        for ex in examples[:3]:
            lines.append(
                f"- {ex['ts']}: {ex['pattern']} → {ex['outcome']} "
                f"({ex['return']:+.2%})"
            )
    
    return "\n".join(lines)


def compress_report(
    summary_md: str,
    top_edges: List[Dict],
    drift_json: Dict,
    council_ci_json: List[Dict],
    examples_json: List[Dict]
) -> Tuple[str, str, str, str, str]:
    """Compress report components to ensure < 5KB total."""
    # Truncate if needed to stay under size limits
    max_summary_len = 2000
    if len(summary_md) > max_summary_len:
        summary_md = summary_md[:max_summary_len] + "\n..."
    
    # Limit JSON sizes
    top_edges = top_edges[:5]
    examples_json = examples_json[:5]
    
    return (
        summary_md,
        json.dumps(top_edges, separators=(',', ':')),
        json.dumps(drift_json, separators=(',', ':')),
        json.dumps(council_ci_json, separators=(',', ':')),
        json.dumps(examples_json, separators=(',', ':'))
    )


def build_daily_report(
    conn: DuckDBPyConnection,
    symbol: str,
    as_of: Optional[date] = None
) -> Dict:
    """Build and store daily report for a symbol."""
    if as_of is None:
        as_of = date.today()
    
    logger.info(f"Building daily report for {symbol} as of {as_of}")
    
    # Gather data
    top_edges = get_top_edges(conn, symbol, as_of)
    drift_data = detect_pattern_drift(conn, symbol, as_of)
    counterfactuals = get_counterfactual_lifts(conn, symbol, as_of)
    examples = get_annotated_examples(conn, symbol, as_of)
    
    # Get council CI data
    council_ci_query = """
    SELECT DISTINCT
        base_pattern,
        base_tf,
        ctx_pattern,
        ctx_tf,
        ci,
        exp_lift
    FROM consistency_matrix
    WHERE base_pattern IN (
        SELECT DISTINCT pattern 
        FROM events 
        WHERE symbol = ? 
        AND event_ts >= ? - INTERVAL 30 DAY
    )
    AND ci > 0.3
    ORDER BY ci DESC
    LIMIT 10
    """
    
    # Format query with parameters, fixing date types
    formatted_query = council_ci_query.replace('DATE ?', "DATE '{}'").replace('?', '{}').format(
        symbol, as_of
    )
    ci_results = conn.execute(formatted_query).fetchall()
    council_ci_json = [
        {
            "base": f"{r[0]}_{r[1]}",
            "context": f"{r[2]}_{r[3]}",
            "ci": round(r[4], 3),
            "lift": round(r[5], 4) if r[5] else 0
        }
        for r in ci_results
    ]
    
    # Build summary
    summary_md = build_summary_markdown(
        symbol, as_of, top_edges, drift_data, counterfactuals, examples
    )
    
    # Compress
    summary_md, top_edges_json, drift_json, council_ci_json_str, examples_json = compress_report(
        summary_md, top_edges, drift_data, council_ci_json, examples
    )
    
    # Store in database
    conn.execute("""
        INSERT OR REPLACE INTO reports_daily 
        (symbol, as_of, summary_md, top_edges_json, drift_json, council_ci_json, examples_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        symbol, as_of, summary_md, top_edges_json, 
        drift_json, council_ci_json_str, examples_json
    ])
    
    # Calculate total size
    total_size = (
        len(summary_md) + len(top_edges_json) + len(drift_json) + 
        len(council_ci_json_str) + len(examples_json)
    )
    
    logger.info(f"Report for {symbol} compressed to {total_size} bytes")
    
    return {
        "symbol": symbol,
        "as_of": as_of.isoformat(),
        "size_bytes": total_size,
        "sections": {
            "summary": len(summary_md),
            "top_edges": len(top_edges_json),
            "drift": len(drift_json),
            "council_ci": len(council_ci_json_str),
            "examples": len(examples_json)
        }
    }


def build_all_reports(conn: DuckDBPyConnection, as_of: Optional[date] = None):
    """Build daily reports for all active symbols."""
    # Get symbols with recent activity
    symbols_query = """
    SELECT DISTINCT symbol
    FROM events
    WHERE event_ts >= CURRENT_DATE - INTERVAL 7 DAY
    """
    
    symbols = [r[0] for r in conn.execute(symbols_query).fetchall()]
    
    results = []
    for symbol in symbols:
        try:
            result = build_daily_report(conn, symbol, as_of)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to build report for {symbol}: {e}")
    
    return results