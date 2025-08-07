#!/usr/bin/env python3
"""
PR7 Metrics Infrastructure Test

Comprehensive test of calibration metrics, threshold tuning, and trade gate system.
"""
import sys
sys.path.append('.')

import json
import logging
import time
from datetime import datetime, timedelta

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.core.metrics import MetricsComputer
from trading_buddy.core.threshold_tuner import ThresholdTuner
from trading_buddy.core.trade_gate import TradeGate

logger = logging.getLogger(__name__)


def setup_test_data(conn):
    """Create test pattern data for metrics computation."""
    print("🔧 Setting up test data...")
    
    # Ensure pattern_events table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pattern_events (
            id TEXT PRIMARY KEY DEFAULT (uuid()),
            event_ts TIMESTAMP NOT NULL,
            detector_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            probability_score DOUBLE,
            outcome_data JSON,
            metadata JSON
        )
    """)
    
    # Clean existing test data
    try:
        conn.execute("DELETE FROM pattern_events WHERE symbol = 'TEST' AND detector_name LIKE 'test_%'")
        conn.execute("DELETE FROM calibration_metrics WHERE detector_name LIKE 'test_%'")
        conn.execute("DELETE FROM detector_thresholds WHERE detector_name LIKE 'test_%'")
        conn.execute("DELETE FROM trade_gate_decisions WHERE detector_name LIKE 'test_%'")
    except Exception:
        # Tables might not exist yet, that's fine
        pass
    
    # Create test patterns with known outcomes
    test_patterns = []
    
    # Good detector: well-calibrated (prob_score ≈ actual success rate)
    import random
    random.seed(42)  # Reproducible results
    
    for i in range(80):
        event_ts = datetime.now() - timedelta(days=30-i//3)
        prob_score = 0.55 + (i % 40) * 0.01  # 0.55 to 0.94
        
        # Well-calibrated: actual success rate close to predicted probability
        # Add some noise but keep it realistic
        actual_success_rate = prob_score + random.uniform(-0.05, 0.05)
        actual_success_rate = max(0.0, min(1.0, actual_success_rate))  # Clamp to [0,1]
        
        is_success = random.random() < actual_success_rate
        outcome = 0.05 if is_success else -0.025
        
        test_patterns.append((
            event_ts, 'test_good_detector', 'TEST', '5m',
            prob_score, json.dumps({'forward_return_1h': outcome})
        ))
    
    # Poor detector: poorly calibrated (overconfident)
    for i in range(60):
        event_ts = datetime.now() - timedelta(days=25-i//3)
        prob_score = 0.40 + (i % 50) * 0.012  # 0.40 to 0.988
        
        # Poorly calibrated: actual success rate much lower than predicted
        # This detector is overconfident
        actual_success_rate = max(0.25, prob_score * 0.5)  # Half the confidence, minimum 25%
        
        is_success = random.random() < actual_success_rate
        outcome = 0.03 if is_success else -0.025
        
        test_patterns.append((
            event_ts, 'test_poor_detector', 'TEST', '5m',
            prob_score, json.dumps({'forward_return_1h': outcome})
        ))
    
    # Insert test data
    for pattern in test_patterns:
        conn.execute("""
            INSERT INTO pattern_events 
            (event_ts, detector_name, symbol, timeframe, probability_score, outcome_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, pattern)
    
    print(f"✅ Created {len(test_patterns)} test patterns")


def test_calibration_metrics():
    """Test Brier score and ECE computation."""
    print("\n🧪 Testing Calibration Metrics...")
    
    with DuckDBManager() as db:
        metrics_computer = MetricsComputer(db.conn)
        
        results = {}
        
        # Test good detector
        print("  📊 Testing good detector calibration...")
        brier_good = metrics_computer.compute_brier_score('test_good_detector', 'TEST', '5m')
        ece_good = metrics_computer.compute_expected_calibration_error('test_good_detector', 'TEST', '5m')
        pr_good = metrics_computer.compute_precision_recall('test_good_detector', 'TEST', '5m')
        
        if brier_good and ece_good and pr_good:
            results['good_detector'] = {
                'brier': brier_good['brier_score'],
                'ece': ece_good['ece'],
                'precision': pr_good['precision'],
                'calibration_quality': ece_good['calibration_quality']
            }
            print(f"    Brier: {brier_good['brier_score']:.4f} (baseline: {brier_good['baseline_brier']:.4f})")
            print(f"    ECE: {ece_good['ece']:.4f} ({ece_good['calibration_quality']})")
            print(f"    Precision: {pr_good['precision']:.3f}, Recall: {pr_good['recall']:.3f}")
        else:
            print("    ❌ Failed to compute metrics for good detector")
        
        # Test poor detector
        print("  📊 Testing poor detector calibration...")
        brier_poor = metrics_computer.compute_brier_score('test_poor_detector', 'TEST', '5m')
        ece_poor = metrics_computer.compute_expected_calibration_error('test_poor_detector', 'TEST', '5m')
        pr_poor = metrics_computer.compute_precision_recall('test_poor_detector', 'TEST', '5m')
        
        if brier_poor and ece_poor and pr_poor:
            results['poor_detector'] = {
                'brier': brier_poor['brier_score'],
                'ece': ece_poor['ece'],
                'precision': pr_poor['precision'],
                'calibration_quality': ece_poor['calibration_quality']
            }
            print(f"    Brier: {brier_poor['brier_score']:.4f} (baseline: {brier_poor['baseline_brier']:.4f})")
            print(f"    ECE: {ece_poor['ece']:.4f} ({ece_poor['calibration_quality']})")
            print(f"    Precision: {pr_poor['precision']:.3f}, Recall: {pr_poor['recall']:.3f}")
        else:
            print("    ❌ Failed to compute metrics for poor detector")
        
        # Validate results
        success = True
        
        if results.get('good_detector'):
            good = results['good_detector']
            if good['brier'] > 0.25 or good['precision'] < 0.50:
                print("    ❌ Good detector has unexpectedly poor metrics")
                success = False
            else:
                print("    ✅ Good detector metrics look reasonable")
        
        if results.get('poor_detector'):
            poor = results['poor_detector']
            if poor['brier'] < good['brier'] or poor['precision'] > good['precision']:
                print("    ❌ Poor detector should have worse metrics than good detector")
                success = False
            else:
                print("    ✅ Poor detector correctly shows worse calibration")
        
        return success


def test_threshold_tuning():
    """Test automated threshold tuning."""
    print("\n🎯 Testing Threshold Tuning...")
    
    with DuckDBManager() as db:
        tuner = ThresholdTuner(db.conn)
        
        # Test threshold finding for good detector
        print("  🔍 Finding optimal threshold for good detector...")
        optimal = tuner.find_optimal_threshold(
            'test_good_detector', 'TEST', '5m',
            precision_target=0.60, recall_target=0.50
        )
        
        if optimal:
            print(f"    Optimal threshold: {optimal.threshold:.2f}")
            print(f"    Performance: P={optimal.precision:.3f}, R={optimal.recall:.3f}, F1={optimal.f1_score:.3f}")
            print(f"    Predictions: {optimal.n_positive_predictions}/{optimal.n_total_samples}")
            
            # Update threshold
            success = tuner.update_detector_threshold(
                'test_good_detector', 'TEST', '5m',
                precision_target=0.60, force_update=True
            )
            
            if success:
                print("    ✅ Threshold updated successfully")
                
                # Test threshold retrieval
                current_threshold = tuner.get_detector_threshold('test_good_detector', 'TEST', '5m')
                if abs(current_threshold - optimal.threshold) < 0.01:
                    print(f"    ✅ Threshold retrieval works: {current_threshold:.2f}")
                    return True
                else:
                    print(f"    ❌ Threshold mismatch: {current_threshold:.2f} vs {optimal.threshold:.2f}")
                    return False
            else:
                print("    ❌ Failed to update threshold")
                return False
        else:
            print("    ❌ Failed to find optimal threshold")
            return False


def test_trade_gate():
    """Test trade gate system."""
    print("\n🚪 Testing Trade Gate System...")
    
    with DuckDBManager() as db:
        trade_gate = TradeGate(db.conn)
        
        test_cases = [
            # (detector_name, prob_score, expected_status, description)
            ('test_good_detector', 0.85, 'allowed', 'High confidence good detector'),
            ('test_good_detector', 0.05, 'watch_only', 'Low confidence good detector (below threshold=0.1)'),
            ('test_poor_detector', 0.90, 'allowed', 'High confidence poor detector (sufficient data)'),
            ('unknown_detector', 0.95, 'blocked', 'Unknown detector with no data'),
        ]
        
        results = []
        
        for detector_name, prob_score, expected_status, description in test_cases:
            print(f"  🧪 {description}...")
            
            should_surface, decision = trade_gate.should_surface_pattern(
                detector_name, 'TEST', '5m', prob_score
            )
            
            print(f"    Decision: {decision.summary}")
            print(f"    Confidence: {decision.confidence_score:.2f}")
            
            if decision.status.value == expected_status:
                print(f"    ✅ Expected {expected_status}, got {decision.status.value}")
                results.append(True)
            else:
                print(f"    ❌ Expected {expected_status}, got {decision.status.value}")
                results.append(False)
        
        # Test gate summary
        print("  📊 Testing gate summary...")
        summary = trade_gate.get_gate_summary(hours_back=1)
        
        if 'total_decisions' in summary and summary['total_decisions'] > 0:
            print(f"    Total decisions: {summary['total_decisions']}")
            print(f"    Allowed: {summary.get('allowed', 0)}")
            print(f"    Blocked: {summary.get('blocked', 0)}")
            print(f"    Watch-only: {summary.get('watch_only', 0)}")
            print("    ✅ Gate summary working")
            results.append(True)
        else:
            print("    ❌ Gate summary failed")
            results.append(False)
        
        return all(results)


def test_metrics_endpoint():
    """Test the /metrics/summary endpoint functionality."""
    print("\n📈 Testing Metrics Summary...")
    
    with DuckDBManager() as db:
        metrics_computer = MetricsComputer(db.conn)
        
        # Test comprehensive metrics summary
        print("  🔍 Getting comprehensive metrics summary...")
        summary = metrics_computer.get_metrics_summary()
        
        required_sections = ['council', 'calibration', 'referee', 'cache', 'ops', 'latency_slo', 'trade_gate']
        results = []
        
        for section in required_sections:
            if section in summary:
                print(f"    ✅ {section} section present")
                results.append(True)
            else:
                print(f"    ❌ {section} section missing")
                results.append(False)
        
        # Check for metrics_ref_id
        if 'metrics_ref_id' in summary and summary['metrics_ref_id']:
            print(f"    ✅ metrics_ref_id present: {summary['metrics_ref_id'][:8]}...")
            results.append(True)
        else:
            print("    ❌ metrics_ref_id missing")
            results.append(False)
        
        # Test specific sections
        if 'trade_gate' in summary and 'total_decisions' in summary['trade_gate']:
            print(f"    ✅ Trade gate metrics: {summary['trade_gate']['total_decisions']} decisions")
            results.append(True)
        else:
            print("    ❌ Trade gate metrics incomplete")
            results.append(False)
        
        return all(results)


def run_pr7_tests():
    """Run comprehensive PR7 metrics infrastructure tests."""
    print("🛡️  PR7 Metrics Infrastructure Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    # Setup test data
    with DuckDBManager() as db:
        setup_test_data(db.conn)
    
    # Run tests
    test_functions = [
        ("Calibration Metrics", test_calibration_metrics),
        ("Threshold Tuning", test_threshold_tuning), 
        ("Trade Gate System", test_trade_gate),
        ("Metrics Endpoint", test_metrics_endpoint)
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{'✅ PASS' if result else '❌ FAIL'} {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    duration = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"📊 PR7 Test Summary:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed in {duration:.1f} seconds")
    
    if passed == total:
        print("\n🎉 PR7 METRICS INFRASTRUCTURE VALIDATED!")
        print("✅ Calibration, threshold tuning, and trade gate systems working correctly")
        return True
    else:
        print(f"\n💥 PR7 VALIDATION FAILED!")
        print(f"❌ {total - passed} tests failed - metrics infrastructure has issues")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = run_pr7_tests()
    exit(0 if success else 1)