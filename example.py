#!/usr/bin/env python3
"""
Example usage of Trading Buddy.
"""
from trading_buddy.core.duck import DuckDBManager
from trading_buddy.nlp.parser import NLPParser
from trading_buddy.council.vote import council_vote
from trading_buddy.schemas.hypothesis import Hypothesis
import json


def main():
    # Example query from a trader
    trader_text = "Watching the MACD on SPY — about to cross on the 5 after a double tap following a big drop. thoughts?"
    
    print(f"Trader query: {trader_text}")
    print("-" * 80)
    
    # Step 1: Parse the natural language
    parser = NLPParser()
    hypothesis = parser.parse(trader_text)
    
    print("\nParsed Hypothesis:")
    print(f"Symbol: {hypothesis.symbol}")
    print(f"Timeframe: {hypothesis.base_timeframe}")
    print(f"Patterns: {[p.name for p in hypothesis.patterns]}")
    print(f"Context: {[c.name for c in hypothesis.context]}")
    print("-" * 80)
    
    # Step 2: Get council vote
    try:
        with DuckDBManager() as db:
            response = council_vote(db.conn, hypothesis)
            
            print("\nCouncil Response:")
            print(f"Historical samples: {response.n}")
            print(f"Median forward return: {response.median_fwd_ret:.2%}")
            print(f"Hit rate: {response.hit_rate:.1%}")
            print(f"MAE: {response.mae:.2%}")
            print(f"Council score: {response.council_score:.2f}")
            
            print("\nCI Breakdown:")
            for ci in response.ci_breakdown:
                status = "✓" if ci.present else "✗"
                print(f"  {status} {ci.pattern} ({ci.timeframe}): CI={ci.ci:.3f}")
            
            print("\nTrading Plan:")
            print(f"Entry: {response.plan.entry}")
            print(f"Stop: {response.plan.stop}")
            print(f"Time stop: {response.plan.time_stop} bars")
            print(f"Size: {response.plan.size:.1%} of risk budget")
            
            if response.follow_ups:
                print("\nFollow-ups:")
                for fu in response.follow_ups:
                    print(f"  - {fu}")
                    
    except Exception as e:
        print(f"\nError getting council vote: {e}")
        print("Make sure to run the following first:")
        print("  1. python scripts/load_sample_data.py")
        print("  2. make init-db")
        print("  3. make backfill")
        print("  4. make ci")


if __name__ == "__main__":
    main()