#!/usr/bin/env python3
"""
PR8 Pre-flight Checks - Verify PR7 foundation is solid before starting PR8.
"""
import sys
sys.path.append('.')

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.core.metrics import MetricsComputer
from trading_buddy.core.trade_gate import TradeGate

def run_preflight_checks():
    """Run all pre-flight checks for PR8."""
    print('🔍 PR8 Pre-flight Checks')
    print('='*50)
    
    all_pass = True
    
    # 1. Metrics sanity check
    print('\n1️⃣ Metrics Sanity Check:')
    try:
        with DuckDBManager() as db:
            mc = MetricsComputer(db.conn)
            summary = mc.get_metrics_summary()
            
            # Check latency metrics
            latency = summary.get('latency_slo', {})
            p95 = latency.get('overall_p95_latency_ms', 0)
            slo_compliant = latency.get('slo_compliant', False)
            
            print(f'  P95 latency: {p95} ms')
            print(f'  SLO compliant: {slo_compliant}')
            
            if not slo_compliant and p95 > 0:
                print('  ❌ SLO not compliant')
                all_pass = False
            else:
                print('  ✅ Latency within SLO')
            
            # Check trade gate
            gate = summary.get('trade_gate', {})
            decisions = gate.get('total_decisions', 0)
            print(f'  Trade gate decisions: {decisions}')
            
    except Exception as e:
        print(f'  ❌ Metrics check failed: {e}')
        all_pass = False
    
    # 2. Gate actually gates test
    print('\n2️⃣ Gate Actually Gates Test:')
    try:
        with DuckDBManager() as db:
            tg = TradeGate(db.conn)
            
            # Test with n_oos=41 (below 60 threshold)
            decision1 = tg.evaluate_signal('test_detector', 'TEST', '5m', 0.8, n_oos=41, stability=0.8, ci_final=0.5)
            if decision1.status.value != 'allowed':
                print(f'  n_oos=41: {decision1.status.value} ✅')
            else:
                print(f'  n_oos=41: {decision1.status.value} ❌ (should be blocked/watch-only)')
                all_pass = False
            
            # Test with stability=0.31 (below 0.4 threshold)
            decision2 = tg.evaluate_signal('test_detector', 'TEST', '5m', 0.8, n_oos=100, stability=0.31, ci_final=0.5)
            if decision2.status.value != 'allowed':
                print(f'  stability=0.31: {decision2.status.value} ✅')
            else:
                print(f'  stability=0.31: {decision2.status.value} ❌ (should be blocked/watch-only)')
                all_pass = False
                
    except Exception as e:
        print(f'  ❌ Gate test failed: {e}')
        all_pass = False
    
    # 3. Calibration baseline check
    print('\n3️⃣ Calibration Baseline Check:')
    try:
        with DuckDBManager() as db:
            # Check if we have calibration metrics
            result = db.conn.execute('''
                SELECT metric_type, AVG(value) as avg_value, COUNT(*) as count
                FROM calibration_metrics
                WHERE ts >= CURRENT_TIMESTAMP - INTERVAL '7 days'
                GROUP BY metric_type
            ''').fetchall()
            
            if not result:
                print('  ⚠️  No recent calibration metrics found')
                print('  Running test detector calibration...')
                
                # Use test data from PR7
                mc = MetricsComputer(db.conn)
                brier = mc.compute_brier_score('test_good_detector', 'TEST', '5m')
                if brier:
                    baseline = 0.25  # Random baseline
                    if brier['brier_score'] < baseline:
                        print(f'  Brier: {brier["brier_score"]:.4f} < {baseline} (baseline) ✅')
                    else:
                        print(f'  Brier: {brier["brier_score"]:.4f} >= {baseline} (baseline) ❌')
                        all_pass = False
            else:
                for metric_type, avg_value, count in result:
                    baseline = 0.25 if metric_type == 'brier' else 0.1  # Baseline for random
                    if avg_value < baseline:
                        print(f'  {metric_type}: {avg_value:.4f} < {baseline} (baseline) ✅')
                    else:
                        print(f'  {metric_type}: {avg_value:.4f} >= {baseline} (baseline) ❌')
                        all_pass = False
                        
    except Exception as e:
        print(f'  ❌ Calibration check failed: {e}')
        all_pass = False
    
    print('\n' + '='*50)
    if all_pass:
        print('✅ All pre-flight checks PASS. Ready for PR8!')
    else:
        print('❌ Some pre-flight checks FAILED. Fix issues before proceeding.')
    
    return all_pass

if __name__ == "__main__":
    success = run_preflight_checks()
    exit(0 if success else 1)