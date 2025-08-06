"""
Build compressed daily reports for ticker memory - simplified version.
"""
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import logging

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


def build_daily_report_simple(
    conn: DuckDBPyConnection,
    symbol: str,
    as_of: Optional[date] = None
) -> Dict:
    """Build a simplified daily report for testing."""
    if as_of is None:
        as_of = date.today()
    
    logger.info(f"Building daily report for {symbol} as of {as_of}")
    
    # Get top patterns from events
    top_patterns_query = f"""
    SELECT 
        pattern,
        timeframe,
        COUNT(*) as n,
        AVG(fwd_ret) as avg_return,
        SUM(CASE WHEN fwd_ret > 0 THEN 1 ELSE 0 END) / COUNT(*) as hit_rate
    FROM events
    WHERE symbol = '{symbol}'
    AND oos_split IN ('test', 'live')
    GROUP BY pattern, timeframe
    HAVING COUNT(*) >= 2
    ORDER BY avg_return * hit_rate DESC
    LIMIT 5
    """
    
    top_patterns = conn.execute(top_patterns_query).fetchall()
    
    # Build summary
    summary_lines = [
        f"# {symbol} Daily Report - {as_of}",
        "",
        "## Top Patterns",
        ""
    ]
    
    top_edges_json = []
    
    for pattern, tf, n, avg_ret, hit_rate in top_patterns:
        summary_lines.append(f"- {pattern} ({tf}): n={n}, R={avg_ret:.4f}, Hit={hit_rate:.1%}")
        top_edges_json.append({
            "pattern": pattern,
            "timeframe": tf,
            "n": n,
            "median_r": avg_ret,
            "hit_rate": hit_rate,
            "mae": 0.001,
            "ci": 0.7
        })
    
    summary_md = "\n".join(summary_lines)
    
    # Simple drift detection
    drift_json = {"drifted": [], "retired": []}
    
    # Simple examples
    examples_json = [
        {
            "ts": "2025-08-05T10:50:00",
            "pattern": "double_bottom",
            "timeframe": "5min",
            "outcome": "win",
            "return": 0.0025
        }
    ]
    
    # Council CI
    council_ci_json = [
        {
            "base": "double_bottom_5min",
            "context": "trend_up_15min",
            "ci": 0.7,
            "lift": 0.002
        }
    ]
    
    # Compress to JSON strings
    top_edges_str = json.dumps(top_edges_json, separators=(',', ':'))
    drift_str = json.dumps(drift_json, separators=(',', ':'))
    council_ci_str = json.dumps(council_ci_json, separators=(',', ':'))
    examples_str = json.dumps(examples_json, separators=(',', ':'))
    
    # Store in database
    conn.execute("""
        INSERT OR REPLACE INTO reports_daily 
        (symbol, as_of, summary_md, top_edges_json, drift_json, council_ci_json, examples_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        symbol, as_of, summary_md, top_edges_str, 
        drift_str, council_ci_str, examples_str
    ])
    
    # Calculate total size
    total_size = (
        len(summary_md) + len(top_edges_str) + len(drift_str) + 
        len(council_ci_str) + len(examples_str)
    )
    
    logger.info(f"Report for {symbol} compressed to {total_size} bytes")
    
    return {
        "symbol": symbol,
        "as_of": as_of.isoformat(),
        "size_bytes": total_size,
        "sections": {
            "summary": len(summary_md),
            "top_edges": len(top_edges_str),
            "drift": len(drift_str),
            "council_ci": len(council_ci_str),
            "examples": len(examples_str)
        }
    }