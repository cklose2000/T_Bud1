"""
Agentic API endpoints for LLM-powered trading agents.
"""
from typing import Dict, List, Optional
from datetime import date

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from trading_buddy.agents.llm.run_agent import run_agent, run_multi_agent_query
from trading_buddy.core.duck import DuckDBManager
from trading_buddy.middleware.referee_enhanced import EnhancedReferee
from trading_buddy.reports.build_daily_report import build_daily_report
from trading_buddy.reports.get_reports import get_report_summary

router = APIRouter()


class AgentQueryRequest(BaseModel):
    """Request for agent query."""
    query: str = Field(..., description="Natural language query")
    mode: str = Field(
        default="analysis",
        description="Agent mode: analysis, whatif, or explanation"
    )


class AgentQueryResponse(BaseModel):
    """Response from agent query."""
    symbol: str
    query: str
    final_answer: str
    tool_calls: List[Dict]
    verified: bool
    metadata: Dict


class ReportResponse(BaseModel):
    """Daily report response."""
    symbol: str
    days_available: int
    date_range: Dict[str, str]
    summary: str
    persistent_patterns: List[Dict]


class PrecursorArmRequest(BaseModel):
    """Request to arm precursor alerts."""
    pattern: str = Field(..., description="Pattern to monitor")
    timeframe: str = Field(default="5m", description="Timeframe")
    threshold: float = Field(
        default=0.7,
        description="Probability threshold for alert",
        ge=0.5,
        le=0.95
    )


class MultiAgentQueryRequest(BaseModel):
    """Request for multi-agent comparison."""
    symbols: List[str] = Field(..., description="Symbols to compare")
    query: str = Field(..., description="Comparison query")


@router.get("/agent/{symbol}/report")
async def get_agent_report(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30)
) -> ReportResponse:
    """
    Get daily report summary for a symbol.
    
    This is the agent's "memory" - compressed daily insights.
    """
    try:
        with DuckDBManager() as db:
            summary = get_report_summary(db.conn, symbol.upper(), days)
            
            if summary["days_available"] == 0:
                # No reports exist, try to build one
                build_daily_report(db.conn, symbol.upper())
                summary = get_report_summary(db.conn, symbol.upper(), days)
            
            return ReportResponse(
                symbol=symbol.upper(),
                days_available=summary["days_available"],
                date_range=summary.get("date_range", {"start": "", "end": ""}),
                summary=summary["summary"],
                persistent_patterns=summary.get("persistent_patterns", [])
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/{symbol}/query")
async def query_agent(
    symbol: str,
    request: AgentQueryRequest,
    referee_mode: bool = Query(False, description="Enable referee validation")
) -> AgentQueryResponse:
    """
    Query a ticker-specific agent.
    
    The agent will use tools to gather data and provide evidence-based answers.
    """
    try:
        result = run_agent(
            symbol.upper(),
            request.query,
            request.mode
        )
        
        # Apply referee middleware if requested
        if referee_mode:
            with DuckDBManager() as db:
                referee = EnhancedReferee(db.conn)
                
                final_answer = result.get("final_answer", "")
                tool_results = result.get("tool_results", {})
                
                # Extract and verify numeric claims
                message_id = f"agent_query_{hash(final_answer)}"
                claims = referee.extract_numeric_claims(final_answer, message_id)
                verified_claims = referee.verify_claims_against_tools(claims, tool_results, message_id)
                
                # Check for rejected claims
                rejected_claims = [c for c in verified_claims if c.verdict == "rejected"]
                
                # Log final response
                referee.log_final_response(message_id, "/agent/query", symbol.upper(), final_answer, verified_claims)
                
                if rejected_claims:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "message": "Agent response rejected by referee - unverified claims detected",
                            "rejected_claims": [{"claim": c.claim_text, "value": c.value_numeric} for c in rejected_claims],
                            "suggestion": "All numeric claims must be traceable to tool outputs"
                        }
                    )
                
                # Add validation to metadata
                result["metadata"] = result.get("metadata", {})
                result["metadata"]["referee_validation"] = {
                    "total_claims": len(verified_claims),
                    "verified_claims": len([c for c in verified_claims if c.verdict in ["verified", "corrected"]]),
                    "rejected_claims": len(rejected_claims),
                    "status": "approved" if not rejected_claims else "rejected"
                }
                result["verified"] = not rejected_claims
        
        return AgentQueryResponse(
            symbol=symbol.upper(),
            query=request.query,
            final_answer=result.get("final_answer", ""),
            tool_calls=result.get("tool_calls", []),
            verified=result.get("verified", False),
            metadata=result.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/{symbol}/precursor/arm")
async def arm_precursor(
    symbol: str,
    request: PrecursorArmRequest
) -> Dict:
    """
    Arm precursor alerts for a pattern.
    
    The agent will monitor for patterns likely to fire within 2-3 bars.
    """
    # This would integrate with the alert service
    # For now, return confirmation
    return {
        "status": "armed",
        "symbol": symbol.upper(),
        "pattern": request.pattern,
        "timeframe": request.timeframe,
        "threshold": request.threshold,
        "message": f"Monitoring {symbol} for {request.pattern} precursors"
    }


@router.post("/agent/compare")
async def compare_agents(
    request: MultiAgentQueryRequest
) -> Dict[str, AgentQueryResponse]:
    """
    Run the same query across multiple ticker agents.
    
    Useful for finding the best opportunity across symbols.
    """
    try:
        if len(request.symbols) > 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 symbols for comparison"
            )
        
        results = run_multi_agent_query(
            [s.upper() for s in request.symbols],
            request.query
        )
        
        # Convert to response format
        responses = {}
        for symbol, result in results.items():
            if "error" not in result:
                responses[symbol] = AgentQueryResponse(
                    symbol=symbol,
                    query=request.query,
                    final_answer=result.get("final_answer", ""),
                    tool_calls=result.get("tool_calls", []),
                    verified=result.get("verified", False),
                    metadata=result.get("metadata", {})
                )
        
        return responses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/build-reports")
async def build_all_reports(
    as_of: Optional[date] = Body(default=None)
) -> Dict:
    """
    Build daily reports for all active symbols.
    
    This would typically run as a nightly job.
    """
    try:
        with DuckDBManager() as db:
            from trading_buddy.reports.build_daily_report import build_all_reports
            
            results = build_all_reports(db.conn, as_of)
            
            return {
                "status": "success",
                "reports_built": len(results),
                "symbols": [r["symbol"] for r in results],
                "total_size_bytes": sum(r["size_bytes"] for r in results)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example queries for testing
EXAMPLE_QUERIES = {
    "setup_check": "SPY showing 5m double bottom with MACD about to cross - thoughts?",
    "performance": "How have my edges performed this week?",
    "comparison": "Compare SPY vs QQQ for mean reversion setups today",
    "whatif": "What if 60m trend aligns with this setup?",
    "explanation": "Explain why we took that last SPY trade"
}


@router.get("/agent/examples")
async def get_example_queries() -> Dict:
    """Get example queries for different agent modes."""
    return {
        "examples": EXAMPLE_QUERIES,
        "modes": ["analysis", "whatif", "explanation"],
        "symbols": ["SPY", "QQQ", "NVDA", "AAPL", "TSLA"]
    }