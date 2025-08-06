"""
Tool executor that runs the actual tool functions.
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

from duckdb import DuckDBPyConnection

from trading_buddy.agents.llm.tools import (
    QueryResult,
    CouncilVoteResult,
    WhatIfResult,
    BarContext,
    DuckDBQuery,
    CouncilVoteRequest,
    WhatIfRequest,
    SelfReportRequest,
    ExplainBarRequest
)
from trading_buddy.council.vote import council_vote, council_vote_whatif
from trading_buddy.reports.get_reports import get_reports, format_report_for_llm
from trading_buddy.schemas.hypothesis import Hypothesis


def execute_duckdb_sql(conn: DuckDBPyConnection, request: DuckDBQuery) -> QueryResult:
    """Execute a read-only SQL query."""
    try:
        result = conn.execute(request.query).fetchall()
        
        # Get column names
        columns = [desc[0] for desc in conn.description] if conn.description else []
        
        return QueryResult(
            columns=columns,
            rows=result,
            row_count=len(result)
        )
    except Exception as e:
        # Return error as empty result with error in columns
        return QueryResult(
            columns=["error"],
            rows=[[str(e)]],
            row_count=1
        )


def execute_council_vote(conn: DuckDBPyConnection, request: CouncilVoteRequest) -> CouncilVoteResult:
    """Execute council vote on hypothesis."""
    try:
        # Convert dict to Hypothesis object
        hypothesis = Hypothesis(**request.hypothesis_json)
        
        # Run council vote
        result = council_vote(conn, hypothesis)
        
        # Convert to result format
        return CouncilVoteResult(
            n=result.n,
            median_fwd_ret=result.median_fwd_ret,
            mean_fwd_ret=result.mean_fwd_ret,
            hit_rate=result.hit_rate,
            mae=result.mae,
            council_score=result.council_score,
            plan={
                "entry": result.plan.entry,
                "stop": result.plan.stop,
                "time_stop": result.plan.time_stop,
                "size": result.plan.size
            },
            ci_breakdown=[
                {
                    "timeframe": item.timeframe,
                    "pattern": item.pattern,
                    "ci": item.ci,
                    "present": item.present,
                    "exp_lift": item.exp_lift
                }
                for item in result.ci_breakdown
            ],
            follow_ups=result.follow_ups
        )
    except Exception as e:
        # Return minimal result on error
        return CouncilVoteResult(
            n=0,
            median_fwd_ret=0,
            mean_fwd_ret=0,
            hit_rate=0,
            mae=0,
            council_score=0,
            plan={"entry": "Error", "stop": "Error", "time_stop": 0, "size": 0},
            ci_breakdown=[],
            follow_ups=[f"Error: {str(e)}"]
        )


def execute_whatif(conn: DuckDBPyConnection, request: WhatIfRequest) -> WhatIfResult:
    """Execute what-if analysis."""
    try:
        # Run base case
        hypothesis = Hypothesis(**request.hypothesis_json)
        base_result = council_vote(conn, hypothesis)
        
        # Run what-if case
        whatif_result = council_vote_whatif(conn, hypothesis, request.overrides)
        
        # Calculate deltas
        deltas = {
            "median_return": whatif_result.median_fwd_ret - base_result.median_fwd_ret,
            "hit_rate": whatif_result.hit_rate - base_result.hit_rate,
            "council_score": whatif_result.council_score - base_result.council_score
        }
        
        return WhatIfResult(
            base_score=base_result.council_score,
            whatif_score=whatif_result.council_score,
            deltas=deltas
        )
    except Exception as e:
        return WhatIfResult(
            base_score=0,
            whatif_score=0,
            deltas={"error": str(e)}
        )


def execute_self_report(conn: DuckDBPyConnection, request: SelfReportRequest) -> str:
    """Get daily reports for a symbol."""
    try:
        reports = get_reports(conn, request.symbol, request.days)
        
        if not reports:
            return f"No reports available for {request.symbol}"
        
        # Format the most recent report
        latest = reports[0]
        formatted = format_report_for_llm(latest)
        
        # Add summary of older reports if multiple days requested
        if len(reports) > 1:
            formatted += f"\n\n**Previous {len(reports)-1} days summary:**\n"
            
            # Aggregate key metrics
            total_patterns = sum(len(r.get("top_edges", [])) for r in reports[1:])
            retired_count = sum(len(r.get("drift", {}).get("retired", [])) for r in reports[1:])
            
            formatted += f"- Total unique patterns: {total_patterns}\n"
            formatted += f"- Patterns retired: {retired_count}\n"
        
        return formatted
    except Exception as e:
        return f"Error retrieving reports: {str(e)}"


def execute_explain_bar(conn: DuckDBPyConnection, request: ExplainBarRequest) -> BarContext:
    """Get context around a specific bar."""
    try:
        # Get target bar
        target_query = """
        SELECT ts, open, high, low, close, volume
        FROM bars
        WHERE symbol = ? AND timeframe = ? AND ts = ?
        """
        target_result = conn.execute(target_query, [
            request.symbol, request.timeframe, request.ts
        ]).fetchone()
        
        if not target_result:
            return BarContext(
                symbol=request.symbol,
                timeframe=request.timeframe,
                target_bar={"error": "Bar not found"},
                before_bars=[],
                after_bars=[],
                patterns_detected=[]
            )
        
        target_bar = {
            "ts": target_result[0].isoformat(),
            "open": target_result[1],
            "high": target_result[2],
            "low": target_result[3],
            "close": target_result[4],
            "volume": target_result[5]
        }
        
        # Get surrounding bars
        window_start = request.ts - timedelta(minutes=5 * request.window_bars)
        window_end = request.ts + timedelta(minutes=5 * request.window_bars)
        
        context_query = """
        SELECT ts, open, high, low, close, volume
        FROM bars
        WHERE symbol = ? AND timeframe = ?
        AND ts >= ? AND ts <= ?
        ORDER BY ts
        """
        
        context_results = conn.execute(context_query, [
            request.symbol, request.timeframe, window_start, window_end
        ]).fetchall()
        
        before_bars = []
        after_bars = []
        
        for r in context_results:
            bar = {
                "ts": r[0].isoformat(),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": r[5]
            }
            
            if r[0] < request.ts:
                before_bars.append(bar)
            elif r[0] > request.ts:
                after_bars.append(bar)
        
        # Check for patterns at this timestamp
        pattern_query = """
        SELECT DISTINCT pattern
        FROM events
        WHERE symbol = ? AND timeframe = ?
        AND event_ts = ?
        """
        
        pattern_results = conn.execute(pattern_query, [
            request.symbol, request.timeframe, request.ts
        ]).fetchall()
        
        patterns = [r[0] for r in pattern_results]
        
        return BarContext(
            symbol=request.symbol,
            timeframe=request.timeframe,
            target_bar=target_bar,
            before_bars=before_bars[-request.window_bars:],  # Limit to requested window
            after_bars=after_bars[:request.window_bars],
            patterns_detected=patterns
        )
        
    except Exception as e:
        return BarContext(
            symbol=request.symbol,
            timeframe=request.timeframe,
            target_bar={"error": str(e)},
            before_bars=[],
            after_bars=[],
            patterns_detected=[]
        )


def execute_tool(
    conn: DuckDBPyConnection,
    tool_name: str,
    request: Any
) -> Any:
    """
    Main tool executor dispatcher.
    
    Args:
        conn: Database connection
        tool_name: Name of tool to execute
        request: Validated request object
        
    Returns:
        Tool-specific result object
    """
    tool_map = {
        "duckdb_sql": execute_duckdb_sql,
        "council_vote": execute_council_vote,
        "whatif": execute_whatif,
        "self_report": execute_self_report,
        "explain_bar": execute_explain_bar
    }
    
    if tool_name not in tool_map:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    executor = tool_map[tool_name]
    return executor(conn, request)