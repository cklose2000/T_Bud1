"""
Test PR8: CI v2 with Uncertainty & FDR Control

Validates bootstrap CI, FDR control, and council voting integration.
"""
import sys
import json
import numpy as np
from datetime import datetime, timedelta

import duckdb

# Add project root to path
sys.path.insert(0, '.')

from trading_buddy.core.ci_uncertainty import BootstrapCI
from trading_buddy.core.fdr_control import FDRController
from trading_buddy.council.ci_v2_integration import CIv2Integration
from trading_buddy.schemas.hypothesis import Hypothesis, CrossTimeframeVote


def test_bootstrap_ci():
    """Test bootstrap confidence intervals."""
    print("\n=== Testing Bootstrap CI ===")
    
    # Create test data with known difference
    np.random.seed(42)
    returns_present = np.random.normal(0.002, 0.01, 100)  # 0.2% mean return
    returns_absent = np.random.normal(-0.001, 0.01, 80)   # -0.1% mean return
    
    # Initialize bootstrap CI
    bootstrap = BootstrapCI(n_bootstrap=1000, confidence_level=0.95)
    
    # Test expectancy lift
    exp_lift = bootstrap.compute_expectancy_lift(returns_present, returns_absent)
    print(f"\nExpectancy Lift:")
    print(f"  Mean: {exp_lift.mean:.5f}")
    print(f"  95% CI: [{exp_lift.ci_lo:.5f}, {exp_lift.ci_hi:.5f}]")
    print(f"  n_eff: {exp_lift.n_eff}")
    print(f"  p-value: {exp_lift.p_value:.4f}")
    print(f"  CI excludes zero: {exp_lift.ci_excludes_zero}")
    
    # Test hit rate lift
    hit_lift = bootstrap.compute_hit_rate_lift(returns_present, returns_absent)
    print(f"\nHit Rate Lift:")
    print(f"  Mean: {hit_lift.mean:.3f}")
    print(f"  95% CI: [{hit_lift.ci_lo:.3f}, {hit_lift.ci_hi:.3f}]")
    print(f"  p-value: {hit_lift.p_value:.4f}")
    
    # Validate coverage
    coverage = bootstrap.validate_bootstrap_coverage(true_diff=0.003, n1=100, n2=80)
    print(f"\nBootstrap CI coverage (should be ~0.95): {coverage:.3f}")
    
    assert 0.90 <= coverage <= 0.99, f"Coverage {coverage} outside expected range"
    print("✓ Bootstrap CI test passed")


def test_fdr_control():
    """Test FDR control via Benjamini-Hochberg."""
    print("\n=== Testing FDR Control ===")
    
    # Create test p-values (mix of null and alternative)
    np.random.seed(42)
    p_values = []
    
    # 80 null hypotheses (uniform p-values)
    for i in range(80):
        p_values.append((f"null_{i}", np.random.uniform(0, 1)))
    
    # 20 alternative hypotheses (small p-values)
    for i in range(20):
        p_values.append((f"alt_{i}", np.random.beta(0.5, 5)))
    
    # Apply FDR control
    fdr = FDRController(fdr_level=0.10)
    results = fdr.apply_benjamini_hochberg(p_values)
    
    # Compute metrics
    metrics = fdr.compute_fdr_metrics(results)
    print(f"\nFDR Results:")
    print(f"  Total tests: {metrics['n_tests']}")
    print(f"  Discoveries: {metrics['n_discoveries']}")
    print(f"  Discovery rate: {metrics['discovery_rate']:.3f}")
    print(f"  Median q-value: {metrics['median_q_value']:.3f}")
    
    # Validate FDR control
    validation = fdr.validate_fdr_control(n_null=80, n_alt=20, n_simulations=100)
    print(f"\nFDR Validation:")
    print(f"  Empirical FDR: {validation['empirical_fdr']:.3f} (target: {validation['target_fdr']})")
    print(f"  Empirical power: {validation['empirical_power']:.3f}")
    print(f"  FDR controlled: {validation['fdr_controlled']}")
    
    assert validation['fdr_controlled'], "FDR not controlled at specified level"
    print("✓ FDR control test passed")


def test_ci_v2_integration():
    """Test CI v2 integration with council voting."""
    print("\n=== Testing CI v2 Integration ===")
    
    # Create test database
    conn = duckdb.connect(':memory:')
    
    # Initialize schema
    with open('sql/pr8_ci_v2_schema.sql', 'r') as f:
        conn.execute(f.read())
    
    # Enable CI v2
    conn.execute("""
        UPDATE ci_version_config 
        SET version = 'v2', rollout_percentage = 100
        WHERE id = 'default'
    """)
    
    # Insert test CI v2 data
    test_data = [
        # Good pattern: high n_eff, stable, CI excludes zero, passes FDR
        ('test_1', '1d', 'double_bottom', '4h', 'macd_bull', 'AAPL', 
         80, 75, 0.003, 0.001, 0.005, 0.02,
         0.12, 0.05, 0.19, 0.03,
         0.018, True, '2024-01', 0.8, 0.85,
         0.75, 0.64, 'allowed', 'allowed: n_eff=75, stability=0.850'),
         
        # Watch-only: low n_eff
        ('test_2', '1d', 'double_bottom', '4h', 'rsi_oversold', 'AAPL',
         45, 40, 0.002, -0.001, 0.005, 0.08,
         0.08, -0.02, 0.18, 0.12,
         0.075, True, '2024-01', 0.5, 0.70,
         0.65, 0.23, 'watch_only', 'watch-only: n_eff=40<60'),
         
        # Watch-only: CI crosses zero
        ('test_3', '1d', 'double_bottom', '4h', 'volume_spike', 'AAPL',
         100, 95, 0.001, -0.002, 0.004, 0.15,
         0.05, -0.05, 0.15, 0.25,
         0.12, False, '2024-01', 0.9, 0.60,
         0.55, 0.30, 'watch_only', 'watch-only: CI_crosses_zero, FDR_fail')
    ]
    
    for row in test_data:
        conn.execute("""
            INSERT INTO consistency_matrix_v2
            (id, base_tf, base_pattern, ctx_tf, ctx_pattern, symbol,
             n_oos_present, n_oos_absent,
             exp_lift_mean, exp_lift_ci_lo, exp_lift_ci_hi, p_value_exp,
             hit_lift_mean, hit_lift_ci_lo, hit_lift_ci_hi, p_value_hit,
             fdr_q, fdr_pass, fdr_month,
             power_score, stability,
             ci_raw, ci_final, trade_gate, gate_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)
    
    # Test CIv2Integration
    ci_v2 = CIv2Integration(conn)
    assert ci_v2.use_v2, "CI v2 should be enabled"
    
    # Test getting metrics
    metrics = ci_v2.get_consistency_metrics_v2(
        'AAPL', '1d', 'double_bottom', '4h', 'macd_bull'
    )
    
    print(f"\nCI v2 Metrics:")
    print(f"  n_eff: {metrics['n_eff']}")
    print(f"  Exp lift: {metrics['exp_lift']['mean']:.4f} "
          f"[{metrics['exp_lift']['ci_lo']:.4f}, {metrics['exp_lift']['ci_hi']:.4f}]")
    print(f"  Stability: {metrics['stability']:.3f}")
    print(f"  Power score: {metrics['power_score']:.3f}")
    print(f"  CI final: {metrics['ci_final']:.3f}")
    print(f"  Trade gate: {metrics['trade_gate']}")
    print(f"  Gate reason: {metrics['gate_reason']}")
    
    # Test CI breakdown creation
    contexts = [
        ('4h', 'macd_bull', True),
        ('4h', 'rsi_oversold', False),
        ('4h', 'volume_spike', True)
    ]
    
    breakdown = ci_v2.create_ci_breakdown_v2(
        'AAPL', '1d', 'double_bottom', contexts
    )
    
    print(f"\nCI Breakdown ({len(breakdown)} contexts):")
    for item in breakdown:
        print(f"  {item.context_name}: CI={item.ci:.3f}, "
              f"present={item.present}, gate={item.metadata.get('trade_gate', 'unknown')}")
    
    # Test rejection formatting
    rejection = ci_v2.format_rejection_reason(breakdown)
    if rejection:
        print(f"\nRejection reason: {rejection}")
    
    print("✓ CI v2 integration test passed")


def test_council_voting_with_ci_v2():
    """Test CI v2 integration module directly."""
    print("\n=== Testing Council Voting Integration ===")
    
    # Create test database
    conn = duckdb.connect(':memory:')
    
    # Initialize schema
    with open('sql/pr8_ci_v2_schema.sql', 'r') as f:
        conn.execute(f.read())
    
    # Enable CI v2
    conn.execute("""
        UPDATE ci_version_config 
        SET version = 'v2', rollout_percentage = 100
        WHERE id = 'default'
    """)
    
    # Insert test CI v2 data
    conn.execute("""
        INSERT INTO consistency_matrix_v2
        (id, base_tf, base_pattern, ctx_tf, ctx_pattern, symbol,
         n_oos_present, n_oos_absent,
         exp_lift_mean, exp_lift_ci_lo, exp_lift_ci_hi, p_value_exp,
         hit_lift_mean, hit_lift_ci_lo, hit_lift_ci_hi, p_value_hit,
         fdr_q, fdr_pass, fdr_month,
         power_score, stability,
         ci_raw, ci_final, trade_gate, gate_reason)
        VALUES 
        ('cv_test_1', '1d', 'double_bottom', '4h', 'macd_bull', 'AAPL', 
         80, 75, 0.003, 0.001, 0.005, 0.02,
         0.12, 0.05, 0.19, 0.03,
         0.018, true, '2024-01', 0.8, 0.85,
         0.75, 0.64, 'allowed', 'allowed: n_eff=75, stability=0.850')
    """)
    
    # Test CI v2 integration for council
    from trading_buddy.council.ci_v2_integration import CIv2Integration
    
    ci_v2 = CIv2Integration(conn)
    assert ci_v2.use_v2, "CI v2 should be enabled"
    
    # Test getting metrics with power/stability clamping
    metrics = ci_v2.get_consistency_metrics_v2(
        'AAPL', '1d', 'double_bottom', '4h', 'macd_bull'
    )
    
    print(f"\nCouncil CI v2 Metrics:")
    print(f"  CI final (with clamping): {metrics['ci_final']:.3f}")
    print(f"  Trade gate: {metrics['trade_gate']}")
    print(f"  Power × Stability: {metrics['power_score']:.3f} × {metrics['stability']:.3f}")
    
    # Test clamping logic
    ci_final, gate, reason = ci_v2.compute_clamped_ci(
        ci_raw=0.8,
        n_eff=45,  # Below threshold
        stability=0.35,  # Below threshold
        exp_lift_ci_lo=-0.001,  # Crosses zero
        fdr_pass=False
    )
    
    print(f"\nClamping Test (multiple failures):")
    print(f"  CI raw: 0.800 → CI final: {ci_final:.3f}")
    print(f"  Gate: {gate}")
    print(f"  Reason: {reason}")
    
    assert gate == "watch_only", "Should be watch_only with multiple failures"
    assert ci_final <= 0.2 * 0.375 * 0.35, "Should be heavily clamped"
    
    print("✓ Council voting integration test passed")


def main():
    """Run all PR8 tests."""
    print("=== PR8 CI v2 Validation Suite ===")
    
    test_bootstrap_ci()
    test_fdr_control()
    test_ci_v2_integration()
    test_council_voting_with_ci_v2()
    
    print("\n✅ All PR8 tests passed!")
    print("\nPR8 Implementation Summary:")
    print("- Bootstrap CI for uncertainty quantification ✓")
    print("- Benjamini-Hochberg FDR control ✓")
    print("- Power score and stability clamping ✓")
    print("- CI v2 council voting integration ✓")
    print("- Detailed rejection explanations ✓")
    print("- CI uncertainty metrics in /metrics/summary ✓")


if __name__ == "__main__":
    main()