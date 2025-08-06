"""
Tool definitions for LLM function calling.
Strict schemas to prevent hallucination.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class DuckDBQuery(BaseModel):
    """Read-only DuckDB query tool."""
    query: str = Field(
        ...,
        description="SQL query to execute. Must be SELECT only.",
        example="SELECT COUNT(*) FROM events WHERE symbol = 'SPY'"
    )
    
    @validator('query')
    def validate_read_only(cls, v):
        # Ensure query is read-only
        query_upper = v.upper().strip()
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']
        if any(word in query_upper for word in forbidden):
            raise ValueError("Only SELECT queries are allowed")
        if not query_upper.startswith('SELECT') and not query_upper.startswith('WITH'):
            raise ValueError("Query must start with SELECT or WITH")
        return v


class CouncilVoteRequest(BaseModel):
    """Council vote hypothesis."""
    hypothesis_json: Dict[str, Any] = Field(
        ...,
        description="Hypothesis in JSON format with symbol, patterns, timeframes",
        example={
            "symbol": "SPY",
            "base_timeframe": "5m",
            "patterns": [{"name": "double_bottom", "params": {}}],
            "context": [],
            "cross_timeframe_votes": []
        }
    )


class WhatIfRequest(BaseModel):
    """What-if scenario with overrides."""
    hypothesis_json: Dict[str, Any] = Field(
        ...,
        description="Base hypothesis JSON"
    )
    overrides: Dict[str, Any] = Field(
        ...,
        description="Override parameters for what-if analysis",
        example={
            "contexts": [{"tf": "60m", "pattern": "trend_up", "present": True}],
            "horizon": 20
        }
    )


class SelfReportRequest(BaseModel):
    """Retrieve daily reports for a symbol."""
    symbol: str = Field(..., description="Stock symbol", example="SPY")
    days: int = Field(
        default=7,
        description="Number of days to retrieve",
        ge=1,
        le=30
    )


class ExplainBarRequest(BaseModel):
    """Get context for a specific bar."""
    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field(..., description="Timeframe (5m, 15m, etc)")
    ts: datetime = Field(..., description="Bar timestamp")
    window_bars: int = Field(
        default=10,
        description="Number of bars before/after to include",
        ge=1,
        le=50
    )


# Tool response models

class QueryResult(BaseModel):
    """DuckDB query result."""
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    
    def to_markdown_table(self) -> str:
        """Convert to markdown table for LLM display."""
        if not self.rows:
            return "No results"
        
        # Header
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "| " + " | ".join(["---"] * len(self.columns)) + " |"
        ]
        
        # Rows (limit to 10 for readability)
        for row in self.rows[:10]:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        
        if len(self.rows) > 10:
            lines.append(f"... {len(self.rows) - 10} more rows")
        
        return "\n".join(lines)


class CouncilVoteResult(BaseModel):
    """Council vote response."""
    n: int
    median_fwd_ret: float
    mean_fwd_ret: float
    hit_rate: float
    mae: float
    council_score: float
    plan: Dict[str, Any]
    ci_breakdown: List[Dict[str, Any]]
    follow_ups: List[str]
    
    def to_summary(self) -> str:
        """Convert to concise summary for LLM."""
        lines = [
            f"**Council Vote Result**",
            f"- Sample size: {self.n} events",
            f"- Median return: {self.median_fwd_ret:.3f}",
            f"- Hit rate: {self.hit_rate:.1%}",
            f"- Council score: {self.council_score:.2f}",
            ""
        ]
        
        if self.council_score >= 0.6:
            lines.extend([
                f"**Trading Plan:**",
                f"- Entry: {self.plan['entry']}",
                f"- Stop: {self.plan['stop']}",
                f"- Size: {self.plan['size']:.1%} of capital"
            ])
        else:
            lines.append("**Decision:** No trade (low council score)")
        
        return "\n".join(lines)


class WhatIfResult(BaseModel):
    """What-if analysis result."""
    base_score: float
    whatif_score: float
    deltas: Dict[str, float]
    
    def to_summary(self) -> str:
        """Convert to summary showing deltas."""
        lines = [
            f"**What-If Analysis**",
            f"- Base score: {self.base_score:.2f}",
            f"- What-if score: {self.whatif_score:.2f}",
            f"- Change: {self.whatif_score - self.base_score:+.2f}",
            ""
        ]
        
        if self.deltas:
            lines.append("**Metric Changes:**")
            for k, v in self.deltas.items():
                lines.append(f"- {k}: {v:+.3f}")
        
        return "\n".join(lines)


class BarContext(BaseModel):
    """Context around a specific bar."""
    symbol: str
    timeframe: str
    target_bar: Dict[str, Any]
    before_bars: List[Dict[str, Any]]
    after_bars: List[Dict[str, Any]]
    patterns_detected: List[str]
    
    def to_summary(self) -> str:
        """Convert to readable summary."""
        lines = [
            f"**Bar Context: {self.symbol} {self.timeframe}**",
            f"Target: {self.target_bar['ts']} O:{self.target_bar['open']:.2f} "
            f"H:{self.target_bar['high']:.2f} L:{self.target_bar['low']:.2f} "
            f"C:{self.target_bar['close']:.2f}",
            ""
        ]
        
        if self.patterns_detected:
            lines.append(f"Patterns: {', '.join(self.patterns_detected)}")
        
        # Price action summary
        if self.before_bars:
            prev_close = self.before_bars[-1]['close']
            change = (self.target_bar['close'] - prev_close) / prev_close
            lines.append(f"Change from prev: {change:.2%}")
        
        return "\n".join(lines)


# Tool function signatures for LLM

TOOL_DEFINITIONS = [
    {
        "name": "duckdb_sql",
        "description": "Execute a read-only SQL query on the trading database",
        "parameters": DuckDBQuery.schema()
    },
    {
        "name": "council_vote", 
        "description": "Get council vote on a trading hypothesis with statistical validation",
        "parameters": CouncilVoteRequest.schema()
    },
    {
        "name": "whatif",
        "description": "Run what-if analysis with hypothetical overrides",
        "parameters": WhatIfRequest.schema()
    },
    {
        "name": "self_report",
        "description": "Retrieve daily performance reports for a symbol",
        "parameters": SelfReportRequest.schema()
    },
    {
        "name": "explain_bar",
        "description": "Get detailed context around a specific price bar",
        "parameters": ExplainBarRequest.schema()
    }
]


def validate_tool_call(tool_name: str, parameters: Dict[str, Any]) -> Any:
    """Validate tool call parameters."""
    tool_map = {
        "duckdb_sql": DuckDBQuery,
        "council_vote": CouncilVoteRequest,
        "whatif": WhatIfRequest,
        "self_report": SelfReportRequest,
        "explain_bar": ExplainBarRequest
    }
    
    if tool_name not in tool_map:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    model_class = tool_map[tool_name]
    return model_class(**parameters)