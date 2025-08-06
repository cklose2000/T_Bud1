"""
Report retrieval functions for LLM agent context.
"""
import json
from datetime import date, timedelta
from typing import Dict, List, Optional

from duckdb import DuckDBPyConnection


def get_reports(
    conn: DuckDBPyConnection,
    symbol: str,
    days: int = 7,
    as_of: Optional[date] = None
) -> List[Dict]:
    """
    Retrieve the latest N days of reports for a symbol.
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        days: Number of days to retrieve
        as_of: End date (defaults to today)
    
    Returns:
        List of report dictionaries, newest first
    """
    if as_of is None:
        as_of = date.today()
    
    start_date = as_of - timedelta(days=days)
    
    query = """
    SELECT 
        symbol,
        as_of,
        summary_md,
        top_edges_json,
        drift_json,
        council_ci_json,
        examples_json,
        created_at
    FROM reports_daily
    WHERE symbol = ?
    AND as_of > ?
    AND as_of <= ?
    ORDER BY as_of DESC
    """
    
    results = conn.execute(query, [symbol, start_date, as_of]).fetchall()
    
    reports = []
    for r in results:
        reports.append({
            "symbol": r[0],
            "as_of": r[1].isoformat() if r[1] else None,
            "summary_md": r[2],
            "top_edges": json.loads(r[3]) if r[3] else [],
            "drift": json.loads(r[4]) if r[4] else {},
            "council_ci": json.loads(r[5]) if r[5] else [],
            "examples": json.loads(r[6]) if r[6] else [],
            "created_at": r[7].isoformat() if r[7] else None
        })
    
    return reports


def get_latest_report(
    conn: DuckDBPyConnection,
    symbol: str,
    as_of: Optional[date] = None
) -> Optional[Dict]:
    """Get the most recent report for a symbol."""
    reports = get_reports(conn, symbol, days=1, as_of=as_of)
    return reports[0] if reports else None


def get_report_summary(
    conn: DuckDBPyConnection,
    symbol: str,
    days: int = 7
) -> Dict:
    """
    Get a compressed summary across multiple days of reports.
    Useful for providing broader context to LLM.
    """
    reports = get_reports(conn, symbol, days)
    
    if not reports:
        return {
            "symbol": symbol,
            "days_available": 0,
            "summary": f"No reports available for {symbol}"
        }
    
    # Aggregate top patterns across days
    all_patterns = {}
    for report in reports:
        for edge in report.get("top_edges", []):
            key = f"{edge['pattern']}_{edge['timeframe']}"
            if key not in all_patterns:
                all_patterns[key] = {
                    "pattern": edge["pattern"],
                    "timeframe": edge["timeframe"],
                    "occurrences": 0,
                    "avg_median_r": 0,
                    "avg_hit_rate": 0,
                    "avg_ci": 0
                }
            
            p = all_patterns[key]
            p["occurrences"] += 1
            p["avg_median_r"] += edge.get("median_r", 0)
            p["avg_hit_rate"] += edge.get("hit_rate", 0)
            p["avg_ci"] += edge.get("ci", 0)
    
    # Calculate averages
    for p in all_patterns.values():
        if p["occurrences"] > 0:
            p["avg_median_r"] /= p["occurrences"]
            p["avg_hit_rate"] /= p["occurrences"]
            p["avg_ci"] /= p["occurrences"]
    
    # Get persistent patterns (appear in multiple days)
    persistent_patterns = [
        p for p in all_patterns.values() 
        if p["occurrences"] >= min(3, len(reports))
    ]
    persistent_patterns.sort(key=lambda x: x["avg_median_r"] * x["avg_hit_rate"], reverse=True)
    
    # Check for any retired patterns
    retired_patterns = set()
    for report in reports:
        for p in report.get("drift", {}).get("retired", []):
            retired_patterns.add(f"{p['pattern']}_{p['timeframe']}")
    
    # Build summary
    summary_lines = [
        f"## {symbol} {days}-Day Summary",
        f"Reports available: {len(reports)} days",
        ""
    ]
    
    if persistent_patterns:
        summary_lines.append("**Persistent Edges:**")
        for p in persistent_patterns[:3]:
            summary_lines.append(
                f"- {p['pattern']} ({p['timeframe']}): "
                f"R={p['avg_median_r']:.3f}, "
                f"Hit={p['avg_hit_rate']:.1%}, "
                f"CI={p['avg_ci']:.2f}"
            )
    
    if retired_patterns:
        summary_lines.append("\n**Recently Retired:**")
        for rp in list(retired_patterns)[:3]:
            summary_lines.append(f"- {rp}")
    
    return {
        "symbol": symbol,
        "days_available": len(reports),
        "date_range": {
            "start": reports[-1]["as_of"],
            "end": reports[0]["as_of"]
        },
        "summary": "\n".join(summary_lines),
        "persistent_patterns": persistent_patterns[:5],
        "retired_count": len(retired_patterns),
        "total_examples": sum(len(r.get("examples", [])) for r in reports)
    }


def format_report_for_llm(report: Dict) -> str:
    """
    Format a report for LLM consumption.
    Emphasizes actionable information and reduces noise.
    """
    lines = [report["summary_md"], ""]
    
    # Add quick stats
    if report.get("top_edges"):
        lines.append("**Quick Stats:**")
        for edge in report["top_edges"][:2]:
            lines.append(
                f"- {edge['pattern']}: n={edge['n']}, "
                f"R={edge['median_r']:.3f}, Hit={edge['hit_rate']:.1%}"
            )
    
    return "\n".join(lines)


def get_multi_symbol_context(
    conn: DuckDBPyConnection,
    symbols: List[str],
    days: int = 3
) -> Dict[str, Dict]:
    """
    Get context for multiple symbols at once.
    Useful for correlation and regime analysis.
    """
    context = {}
    
    for symbol in symbols:
        summary = get_report_summary(conn, symbol, days)
        context[symbol] = {
            "summary": summary["summary"],
            "top_pattern": summary["persistent_patterns"][0] if summary["persistent_patterns"] else None,
            "days_available": summary["days_available"]
        }
    
    return context