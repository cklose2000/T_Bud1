"""
Agent runner that executes LLM queries with tool enforcement.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from trading_buddy.agents.llm.persona import (
    get_ticker_persona, 
    format_user_query,
    get_referee_persona
)
from trading_buddy.agents.llm.tools import (
    TOOL_DEFINITIONS,
    validate_tool_call,
    QueryResult,
    CouncilVoteResult,
    WhatIfResult,
    BarContext
)
from trading_buddy.agents.llm.tool_executor import execute_tool
from trading_buddy.core.duck import DuckDBManager

logger = logging.getLogger(__name__)


class AgentResponse:
    """Structured response from agent."""
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.final_answer: str = ""
        self.metadata: Dict[str, Any] = {}
        self.verified: bool = False
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def add_tool_call(self, tool: str, params: Dict, result: Any):
        self.tool_calls.append({
            "tool": tool,
            "params": params,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict:
        return {
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "final_answer": self.final_answer,
            "metadata": self.metadata,
            "verified": self.verified
        }


class TickerAgent:
    """Agent for a specific ticker."""
    
    def __init__(self, symbol: str, mode: str = "analysis"):
        self.symbol = symbol
        self.mode = mode
        self.persona = get_ticker_persona(symbol, mode)
        self.db_manager = DuckDBManager()
    
    def _parse_llm_response(self, response: str) -> Tuple[str, List[Dict]]:
        """
        Parse LLM response to extract text and tool calls.
        
        Returns:
            Tuple of (text_response, tool_calls)
        """
        # This is a simplified parser. In production, would use
        # the LLM's native function calling format
        tool_calls = []
        text_parts = []
        
        lines = response.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for tool call markers
            if line.startswith("TOOL_CALL:"):
                # Parse tool call
                try:
                    tool_name = line.split(":", 1)[1].strip()
                    i += 1
                    
                    # Collect JSON parameters
                    json_lines = []
                    while i < len(lines) and not lines[i].strip().startswith("TOOL_RESULT:"):
                        json_lines.append(lines[i])
                        i += 1
                    
                    params = json.loads('\n'.join(json_lines))
                    tool_calls.append({"name": tool_name, "params": params})
                except Exception as e:
                    logger.error(f"Failed to parse tool call: {e}")
            else:
                text_parts.append(line)
            
            i += 1
        
        return '\n'.join(text_parts), tool_calls
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        results = []
        
        with self.db_manager as db:
            for call in tool_calls:
                try:
                    # Validate parameters
                    validated = validate_tool_call(call["name"], call["params"])
                    
                    # Execute tool
                    result = execute_tool(
                        db.conn,
                        call["name"],
                        validated
                    )
                    
                    results.append({
                        "tool": call["name"],
                        "params": call["params"],
                        "result": result,
                        "success": True
                    })
                    
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    results.append({
                        "tool": call["name"],
                        "params": call["params"],
                        "error": str(e),
                        "success": False
                    })
        
        return results
    
    def _format_tool_results(self, results: List[Dict]) -> str:
        """Format tool results for LLM consumption."""
        formatted = []
        
        for r in results:
            if r["success"]:
                result_obj = r["result"]
                
                # Format based on result type
                if hasattr(result_obj, 'to_summary'):
                    formatted.append(f"**{r['tool']} result:**\n{result_obj.to_summary()}")
                elif hasattr(result_obj, 'to_markdown_table'):
                    formatted.append(f"**{r['tool']} result:**\n{result_obj.to_markdown_table()}")
                else:
                    formatted.append(f"**{r['tool']} result:**\n{str(result_obj)}")
            else:
                formatted.append(f"**{r['tool']} error:** {r['error']}")
        
        return "\n\n".join(formatted)
    
    def run(self, query: str, max_iterations: int = 3) -> AgentResponse:
        """
        Run agent on a query with tool execution loops.
        
        Args:
            query: User query
            max_iterations: Maximum LLM/tool iterations
            
        Returns:
            AgentResponse with results
        """
        response = AgentResponse()
        
        # Format query with context
        formatted_query = format_user_query(query, self.symbol)
        
        # Initial messages
        messages = [
            {"role": "system", "content": self.persona},
            {"role": "user", "content": formatted_query}
        ]
        
        response.add_message("system", self.persona)
        response.add_message("user", formatted_query)
        
        # Tool execution loop
        for iteration in range(max_iterations):
            # Call LLM (simplified - in production would use actual LLM API)
            llm_response = self._mock_llm_call(messages)
            
            # Parse response
            text, tool_calls = self._parse_llm_response(llm_response)
            
            if tool_calls:
                # Execute tools
                results = self._execute_tool_calls(tool_calls)
                
                # Add to response
                for r in results:
                    if r["success"]:
                        response.add_tool_call(
                            r["tool"],
                            r["params"],
                            r["result"]
                        )
                
                # Format results and add to context
                tool_results = self._format_tool_results(results)
                messages.append({
                    "role": "assistant",
                    "content": f"{text}\n\nTool Results:\n{tool_results}"
                })
                response.add_message("assistant", f"{text}\n\nTool Results:\n{tool_results}")
            else:
                # No more tool calls, we have final answer
                response.final_answer = text
                response.add_message("assistant", text)
                break
        
        # Run referee check
        response.verified = self._verify_response(response)
        
        # Add metadata
        response.metadata = {
            "symbol": self.symbol,
            "mode": self.mode,
            "iterations": iteration + 1,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def _mock_llm_call(self, messages: List[Dict]) -> str:
        """
        Mock LLM call for testing.
        In production, would call actual LLM API.
        """
        # Simulate LLM response based on last user message
        last_message = messages[-1]["content"]
        
        if "double bottom" in last_message.lower():
            return """I'll analyze the double bottom pattern for SPY.

TOOL_CALL: self_report
{"symbol": "SPY", "days": 7}

Let me check the current pattern statistics.

TOOL_CALL: council_vote
{
    "hypothesis_json": {
        "symbol": "SPY",
        "base_timeframe": "5m",
        "patterns": [{"name": "double_bottom", "params": {}}],
        "context": [],
        "cross_timeframe_votes": []
    }
}

Based on the analysis, I'll provide a recommendation."""
        
        # Default response
        return "I need to analyze the current market data to provide a recommendation."
    
    def _verify_response(self, response: AgentResponse) -> bool:
        """
        Run referee verification on response.
        Check that all numbers are backed by tool calls.
        """
        # Extract all numbers from final answer
        import re
        numbers = re.findall(r'\b\d+\.?\d*%?\b', response.final_answer)
        
        # Check if numbers appear in tool results
        tool_output_text = str(response.tool_calls)
        
        verified = True
        for num in numbers:
            if num not in tool_output_text:
                logger.warning(f"Unverified number in response: {num}")
                verified = False
        
        return verified


def run_agent(
    symbol: str,
    query: str,
    mode: str = "analysis"
) -> Dict:
    """
    Convenience function to run an agent query.
    
    Args:
        symbol: Stock symbol
        query: User query
        mode: Agent mode (analysis, whatif, explanation)
        
    Returns:
        Response dictionary
    """
    agent = TickerAgent(symbol, mode)
    response = agent.run(query)
    return response.to_dict()


def run_multi_agent_query(
    symbols: List[str],
    query: str
) -> Dict[str, Dict]:
    """
    Run query across multiple ticker agents.
    Useful for comparison queries.
    """
    results = {}
    
    for symbol in symbols:
        try:
            results[symbol] = run_agent(symbol, query)
        except Exception as e:
            logger.error(f"Failed to run agent for {symbol}: {e}")
            results[symbol] = {"error": str(e)}
    
    return results