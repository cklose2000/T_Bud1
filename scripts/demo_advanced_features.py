#!/usr/bin/env python3
"""
Demo script showcasing the advanced Trading Buddy features:
- Daily report generation
- LLM agent with tools
- Statistical validation
- What-if analysis
"""
import sys
from pathlib import Path
from datetime import date
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.reports.build_daily_report_v2 import build_daily_report_simple
from trading_buddy.reports.get_reports import get_report_summary
from trading_buddy.council.stability import (
    calculate_rolling_stability,
    check_pattern_drift,
    validate_pattern_candidates
)
from trading_buddy.agents.llm.tools import (
    DuckDBQuery,
    CouncilVoteRequest,
    SelfReportRequest,
    validate_tool_call
)
from trading_buddy.agents.llm.persona import get_ticker_persona


def demo_daily_reports():
    """Demo the daily report system."""
    print("\n" + "="*80)
    print("DEMO: Daily Report Generation")
    print("="*80)
    
    with DuckDBManager() as db:
        # Build report
        print("\n1. Building daily report for SPY...")
        result = build_daily_report_simple(db.conn, "SPY")
        print(f"   ✓ Report built: {result['size_bytes']} bytes")
        
        # Get report summary
        print("\n2. Retrieving 7-day summary...")
        summary = get_report_summary(db.conn, "SPY", days=7)
        print(f"   ✓ {summary['days_available']} days available")
        print("\n   Summary:")
        print("   " + summary['summary'].replace('\n', '\n   '))


def demo_statistical_guards():
    """Demo statistical validation."""
    print("\n" + "="*80)
    print("DEMO: Statistical Guards")
    print("="*80)
    
    with DuckDBManager() as db:
        # Check pattern stability
        print("\n1. Checking pattern stability...")
        stability = calculate_rolling_stability(
            db.conn, "SPY", "double_bottom", "5min"
        )
        print(f"   Stability Score: {stability['stability_score']:.2f}")
        print(f"   PSI: {stability['psi']:.3f}")
        print(f"   KS p-value: {stability['ks_pvalue']:.3f}")
        
        # Check drift
        print("\n2. Checking for pattern drift...")
        drift = check_pattern_drift(
            db.conn, "SPY", "double_bottom", "5min"
        )
        print(f"   Is Drifting: {drift['is_drifting']}")
        print(f"   Performance Change: {drift['performance_change']:.4f}")
        print(f"   Recommendation: {drift['recommendation']}")
        
        # Validate candidates
        print("\n3. Validating pattern candidates with Benjamini-Hochberg...")
        candidates = [
            {"symbol": "SPY", "pattern": "double_bottom", "timeframe": "5min"},
            {"symbol": "SPY", "pattern": "w_pattern", "timeframe": "5min"},
        ]
        
        validated = validate_pattern_candidates(db.conn, candidates)
        for v in validated:
            print(f"   {v['pattern']} ({v['timeframe']}): "
                  f"p={v['p_value']:.3f}, significant={v['significant']}, "
                  f"stability={v['stability_score']:.2f}")


def demo_llm_tools():
    """Demo LLM tool system."""
    print("\n" + "="*80)
    print("DEMO: LLM Agent Tools")
    print("="*80)
    
    # Show persona
    print("\n1. SPY Agent Persona:")
    persona = get_ticker_persona("SPY", "analysis")
    print("   " + persona[:200] + "...")
    
    # Demo tool validation
    print("\n2. Tool Validation Examples:")
    
    # Valid query
    try:
        query = DuckDBQuery(query="SELECT COUNT(*) FROM events WHERE symbol = 'SPY'")
        print("   ✓ Valid SQL query accepted")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Invalid query (write operation)
    try:
        query = DuckDBQuery(query="INSERT INTO events VALUES (...)")
        print("   ✓ Insert query accepted (shouldn't happen!)")
    except Exception as e:
        print(f"   ✓ Write query rejected: {e}")
    
    # Council vote request
    print("\n3. Council Vote Request:")
    vote_request = CouncilVoteRequest(
        hypothesis_json={
            "symbol": "SPY",
            "base_timeframe": "5m",
            "patterns": [{"name": "double_bottom", "params": {}}],
            "context": [],
            "cross_timeframe_votes": []
        }
    )
    print(f"   ✓ Hypothesis validated: {vote_request.hypothesis_json['symbol']} "
          f"{vote_request.hypothesis_json['patterns'][0]['name']}")


def demo_mock_agent_interaction():
    """Demo a mock agent interaction."""
    print("\n" + "="*80)
    print("DEMO: Mock Agent Interaction")
    print("="*80)
    
    print("\nUser: SPY showing 5m double bottom with MACD about to cross - thoughts?")
    print("\nSPY Agent: I'll analyze the double bottom pattern for SPY.")
    print("\n[TOOL CALL: self_report]")
    print("Parameters: {\"symbol\": \"SPY\", \"days\": 7}")
    print("\n[TOOL RESULT]")
    print("# SPY Daily Report - 2025-08-06")
    print("No significant edges in recent data")
    
    print("\n[TOOL CALL: council_vote]")
    print("Parameters: {")
    print("  \"hypothesis_json\": {")
    print("    \"symbol\": \"SPY\",")
    print("    \"base_timeframe\": \"5m\",")
    print("    \"patterns\": [{\"name\": \"double_bottom\", \"params\": {}}]")
    print("  }")
    print("}")
    
    print("\n[TOOL RESULT]")
    print("**Council Vote Result**")
    print("- Sample size: 4 events")
    print("- Median return: 0.0025")
    print("- Hit rate: 75.0%")
    print("- Council score: 0.45")
    print("")
    print("**Decision:** No trade (low council score)")
    
    print("\nSPY Agent: The double bottom pattern shows promise with 75% hit rate,")
    print("but with only 4 samples and council score of 0.45 (below 0.6 threshold),")
    print("I recommend standing down. Wait for stronger context alignment or more data.")


def main():
    print("\n" + "="*80)
    print("TRADING BUDDY ADVANCED FEATURES DEMO")
    print("="*80)
    
    # Run all demos
    demo_daily_reports()
    demo_statistical_guards()
    demo_llm_tools()
    demo_mock_agent_interaction()
    
    print("\n" + "="*80)
    print("Demo completed! Key features demonstrated:")
    print("- Daily report generation < 5KB")
    print("- Statistical validation with drift detection")
    print("- LLM tool validation and execution")
    print("- Evidence-based agent responses")
    print("="*80)


if __name__ == "__main__":
    main()