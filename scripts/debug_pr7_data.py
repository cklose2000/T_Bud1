#!/usr/bin/env python3
"""
Debug PR7 test data to understand why metrics are failing.
"""
import sys
sys.path.append('.')

from trading_buddy.core.duck import DuckDBManager

def debug_test_data():
    """Debug what data is actually in the database."""
    print("🔍 Debugging PR7 test data...")
    
    with DuckDBManager() as db:
        # Check what data exists
        result = db.conn.execute("""
            SELECT 
                detector_name,
                COUNT(*) as count,
                AVG(probability_score) as avg_prob,
                AVG(CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE)) as avg_outcome,
                MIN(probability_score) as min_prob,
                MAX(probability_score) as max_prob,
                SUM(CASE WHEN CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE) > 0 THEN 1 ELSE 0 END) as positive_outcomes
            FROM pattern_events 
            WHERE symbol = 'TEST'
            GROUP BY detector_name
        """).fetchall()
        
        for row in result:
            detector, count, avg_prob, avg_outcome, min_prob, max_prob, positive = row
            print(f"\n📊 {detector}:")
            print(f"  Count: {count}")
            print(f"  Probability range: {min_prob:.3f} - {max_prob:.3f} (avg: {avg_prob:.3f})")
            print(f"  Outcome range: avg={avg_outcome:.4f}")
            print(f"  Positive outcomes: {positive}/{count} ({positive/count*100:.1f}%)")
        
        # Test a simple threshold
        print(f"\n🎯 Testing threshold 0.5:")
        threshold_test = db.conn.execute("""
            SELECT 
                detector_name,
                SUM(CASE WHEN probability_score >= 0.5 THEN 1 ELSE 0 END) as predictions_above_threshold,
                SUM(CASE WHEN probability_score >= 0.5 AND CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE) > 0 THEN 1 ELSE 0 END) as true_positives,
                SUM(CASE WHEN CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE) > 0 THEN 1 ELSE 0 END) as actual_positives
            FROM pattern_events 
            WHERE symbol = 'TEST'
            GROUP BY detector_name
        """).fetchall()
        
        for row in threshold_test:
            detector, predictions, tp, actual_pos = row
            precision = tp / predictions if predictions > 0 else 0.0
            recall = tp / actual_pos if actual_pos > 0 else 0.0
            print(f"  {detector}: predictions={predictions}, TP={tp}, precision={precision:.3f}, recall={recall:.3f}")

if __name__ == "__main__":
    debug_test_data()