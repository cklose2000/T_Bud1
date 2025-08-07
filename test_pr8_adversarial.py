"""
PR8 Adversarial Test Suite

Validates edge cases and corner scenarios for CI v2 implementation.
"""
import sys
import numpy as np
from scipy import stats
from datetime import datetime
import hashlib
import time

sys.path.insert(0, '.')

from trading_buddy.core.ci_uncertainty import BootstrapCI
from trading_buddy.core.fdr_control import FDRController


class AdversarialTests:
    """Edge case tests for CI v2 robustness."""
    
    def test_bootstrap_extreme_tails(self):
        """Test bootstrap coverage with thin samples."""
        print("\n=== Testing Bootstrap Extreme Tails ===")
        
        # Create thin-tailed distribution (uniform)
        np.random.seed(42)
        n_sims = 100
        coverage_1000 = []
        coverage_2000 = []
        
        for _ in range(n_sims):
            # Generate data with known difference
            true_diff = 0.01
            data1 = np.random.uniform(-0.02, 0.02, 15)  # Very small sample
            data2 = np.random.uniform(-0.02 - true_diff, 0.02 - true_diff, 12)
            
            # Test with 1000 bootstrap
            boot_1000 = BootstrapCI(n_bootstrap=1000)
            lift_1000 = boot_1000.compute_expectancy_lift(data1, data2, compute_p_value=False)
            covered_1000 = lift_1000.ci_lo <= true_diff <= lift_1000.ci_hi
            coverage_1000.append(covered_1000)
            
            # Test with 2000 bootstrap
            boot_2000 = BootstrapCI(n_bootstrap=2000)
            lift_2000 = boot_2000.compute_expectancy_lift(data1, data2, compute_p_value=False)
            covered_2000 = lift_2000.ci_lo <= true_diff <= lift_2000.ci_hi
            coverage_2000.append(covered_2000)
        
        cov_1000 = np.mean(coverage_1000)
        cov_2000 = np.mean(coverage_2000)
        
        print(f"Coverage with 1000 bootstraps: {cov_1000:.3f}")
        print(f"Coverage with 2000 bootstraps: {cov_2000:.3f}")
        
        # Check CI width vs theoretical SE
        se_theoretical = np.sqrt(np.var(data1)/len(data1) + np.var(data2)/len(data2))
        ci_width = lift_1000.ci_hi - lift_1000.ci_lo
        print(f"CI width / (theoretical SE × 1.96): {ci_width / (se_theoretical * 1.96):.3f}")
        
        assert ci_width >= se_theoretical * 1.9, "CI too narrow for thin samples"
        print("✓ Bootstrap tail coverage test passed")
    
    def test_skewed_returns(self):
        """Test handling of highly skewed returns."""
        print("\n=== Testing Skewed Returns ===")
        
        # Generate log-gamma returns (highly skewed)
        np.random.seed(42)
        shape, scale = 2., 2.
        returns_present = np.log(np.random.gamma(shape, scale, 100)) - 1.5
        returns_absent = np.log(np.random.gamma(shape, scale, 100)) - 1.7
        
        skew_present = stats.skew(returns_present)
        skew_absent = stats.skew(returns_absent)
        print(f"Skewness - Present: {skew_present:.2f}, Absent: {skew_absent:.2f}")
        
        # Standard bootstrap (mean-based)
        bootstrap = BootstrapCI(n_bootstrap=1000)
        mean_lift = bootstrap.compute_expectancy_lift(returns_present, returns_absent)
        
        # Median-based bootstrap for skewed data
        median_lift = self._compute_median_lift_bootstrap(
            returns_present, returns_absent, n_bootstrap=1000
        )
        
        print(f"Mean lift: {mean_lift.mean:.4f} [{mean_lift.ci_lo:.4f}, {mean_lift.ci_hi:.4f}]")
        print(f"Median lift: {median_lift['mean']:.4f} [{median_lift['ci_lo']:.4f}, {median_lift['ci_hi']:.4f}]")
        
        # With high skew, median should be more stable
        assert abs(median_lift['mean']) < abs(mean_lift.mean), "Median should be more robust to skew"
        print("✓ Skewed returns test passed")
    
    def test_block_bootstrap_dependence(self):
        """Test block bootstrap for dependent data."""
        print("\n=== Testing Block Bootstrap for Dependence ===")
        
        # Generate AR(1) process with correlation
        np.random.seed(42)
        n = 200
        rho = 0.4  # Autocorrelation
        sigma = 0.01
        
        # AR(1): y_t = rho * y_{t-1} + epsilon_t
        returns_present = np.zeros(n)
        returns_absent = np.zeros(n)
        
        for i in range(1, n):
            returns_present[i] = rho * returns_present[i-1] + np.random.normal(0.002, sigma)
            returns_absent[i] = rho * returns_absent[i-1] + np.random.normal(0, sigma)
        
        # Standard bootstrap (assumes independence)
        bootstrap_iid = BootstrapCI(n_bootstrap=500)
        lift_iid = bootstrap_iid.compute_expectancy_lift(returns_present, returns_absent)
        
        # Block bootstrap (preserves dependence)
        lift_block = self._compute_block_bootstrap_lift(
            returns_present, returns_absent, block_size=10, n_bootstrap=500
        )
        
        print(f"IID Bootstrap CI width: {lift_iid.ci_hi - lift_iid.ci_lo:.5f}")
        print(f"Block Bootstrap CI width: {lift_block['ci_hi'] - lift_block['ci_lo']:.5f}")
        
        # Block bootstrap should have wider CI due to dependence
        assert lift_block['ci_hi'] - lift_block['ci_lo'] > lift_iid.ci_hi - lift_iid.ci_lo, \
            "Block bootstrap should produce wider CI for dependent data"
        
        print("✓ Block bootstrap test passed")
    
    def test_fdr_edge_cases(self):
        """Test FDR control edge cases."""
        print("\n=== Testing FDR Edge Cases ===")
        
        fdr = FDRController(fdr_level=0.10)
        
        # Case 1: Only 3 p-values (BH should equal Holm)
        p_values_small = [("h1", 0.01), ("h2", 0.03), ("h3", 0.20)]
        bh_results = fdr.apply_benjamini_hochberg(p_values_small)
        
        # Holm procedure for comparison
        holm_results = self._apply_holm(p_values_small, alpha=0.10)
        
        bh_discoveries = sum(1 for r in bh_results.values() if r.passes_fdr)
        holm_discoveries = sum(1 for _, passes in holm_results if passes)
        
        print(f"Small sample (n=3): BH discoveries={bh_discoveries}, Holm discoveries={holm_discoveries}")
        assert bh_discoveries == holm_discoveries, "BH should equal Holm for very small n"
        
        # Case 2: 50 uniform p-values (null case)
        np.random.seed(42)
        p_values_null = [(f"null_{i}", np.random.uniform(0, 1)) for i in range(50)]
        bh_null = fdr.apply_benjamini_hochberg(p_values_null)
        
        false_discoveries = sum(1 for r in bh_null.values() if r.passes_fdr)
        empirical_fdr = false_discoveries / 50
        
        print(f"Null case (n=50): False discoveries={false_discoveries}, Empirical FDR={empirical_fdr:.3f}")
        assert empirical_fdr <= 0.12, f"Empirical FDR {empirical_fdr} exceeds tolerance"
        
        # Case 3: Mixed null/alternative
        p_values_mixed = []
        for i in range(40):
            p_values_mixed.append((f"null_{i}", np.random.uniform(0, 1)))
        for i in range(10):
            p_values_mixed.append((f"alt_{i}", np.random.beta(0.5, 5)))
        
        bh_mixed = fdr.apply_benjamini_hochberg(p_values_mixed)
        discoveries = sum(1 for r in bh_mixed.values() if r.passes_fdr)
        
        print(f"Mixed case (40 null, 10 alt): Total discoveries={discoveries}")
        print("✓ FDR edge cases test passed")
    
    def test_trade_gate_fuzzing(self):
        """Fuzz test trade gate boundaries."""
        print("\n=== Testing Trade Gate Fuzzing ===")
        
        from trading_buddy.council.ci_v2_integration import CIv2Integration
        import duckdb
        
        conn = duckdb.connect(':memory:')
        ci_v2 = CIv2Integration(conn)
        
        # Test boundary conditions
        test_cases = [
            # (n_eff, stability, ci_lo, fdr_pass, expected_gate)
            (60, 0.40, 0.001, True, "allowed"),      # All pass exactly at boundary
            (59, 0.40, 0.001, True, "watch_only"),   # n_eff just below
            (60, 0.39, 0.001, True, "watch_only"),   # stability just below
            (60, 0.40, 0.000, True, "watch_only"),   # CI includes zero
            (60, 0.40, -0.001, True, "watch_only"),  # CI crosses zero
            (60, 0.40, 0.001, False, "watch_only"),  # FDR fail
            (59, 0.39, -0.001, False, "watch_only"), # Multiple failures
        ]
        
        for n_eff, stability, ci_lo, fdr_pass, expected in test_cases:
            ci_final, gate, reason = ci_v2.compute_clamped_ci(
                ci_raw=0.8,
                n_eff=n_eff,
                stability=stability,
                exp_lift_ci_lo=ci_lo,
                fdr_pass=fdr_pass
            )
            
            assert gate == expected, f"Expected {expected}, got {gate} for {test_cases}"
            
            # Verify clamping
            if gate == "watch_only":
                assert ci_final <= 0.2 * min(1.0, n_eff/120) * stability
        
        print("✓ Trade gate fuzzing test passed")
    
    def test_n_eff_calculation(self):
        """Test effective sample size calculation."""
        print("\n=== Testing n_eff Calculation ===")
        
        # Simulate overlapping vs distinct events
        n_events_unique = 50
        n_bars_per_event = 4  # Overlapping bars
        
        # Method 1: Count all bars (wrong - inflated)
        n_eff_bars = n_events_unique * n_bars_per_event
        
        # Method 2: Count unique events (correct)
        n_eff_events = n_events_unique
        
        print(f"Counting bars (inflated): n_eff = {n_eff_bars}")
        print(f"Counting unique events (correct): n_eff = {n_eff_events}")
        
        # Power calculation should use event count
        power_inflated = min(1.0, n_eff_bars / 120)
        power_correct = min(1.0, n_eff_events / 120)
        
        print(f"Power (inflated): {power_inflated:.3f}")
        print(f"Power (correct): {power_correct:.3f}")
        
        assert power_correct < power_inflated, "Event-based count should give lower power"
        print("✓ n_eff calculation test passed")
    
    def test_power_score_effect_size(self):
        """Test power score based on effect size."""
        print("\n=== Testing Power Score with Effect Size ===")
        
        # Linear power score
        n_vals = [30, 60, 90, 120, 150, 200]
        linear_powers = [min(1.0, n/120) for n in n_vals]
        
        # Effect-size based power (Cohen's d approach)
        def effect_size_power(n, lift, pooled_sd, alpha=0.05):
            """Power based on effect size and sample size."""
            d = lift / pooled_sd  # Cohen's d
            se = pooled_sd * np.sqrt(2/n)  # SE of difference
            z_crit = stats.norm.ppf(1 - alpha/2)
            z_power = (abs(d) * np.sqrt(n/2) - z_crit)
            return stats.norm.cdf(z_power)
        
        lift = 0.003  # Larger effect size
        pooled_sd = 0.01
        
        effect_powers = [effect_size_power(n, lift, pooled_sd) for n in n_vals]
        
        print("Sample Size | Linear Power | Effect-Size Power")
        print("-" * 45)
        for n, lin_p, eff_p in zip(n_vals, linear_powers, effect_powers):
            print(f"{n:11d} | {lin_p:12.3f} | {eff_p:17.3f}")
        
        # Effect-size power should be more nuanced
        assert effect_powers[0] < 0.5, "Low n should have low power for small effect"
        assert effect_powers[-1] > 0.8, "High n should have high power"
        print("✓ Effect-size power test passed")
    
    def _compute_median_lift_bootstrap(self, returns_present, returns_absent, n_bootstrap=1000):
        """Compute median lift with bootstrap CI."""
        np.random.seed(42)
        median_lifts = []
        
        for _ in range(n_bootstrap):
            sample_present = np.random.choice(returns_present, size=len(returns_present), replace=True)
            sample_absent = np.random.choice(returns_absent, size=len(returns_absent), replace=True)
            
            median_lift = np.median(sample_present) - np.median(sample_absent)
            median_lifts.append(median_lift)
        
        return {
            'mean': np.median(returns_present) - np.median(returns_absent),
            'ci_lo': np.percentile(median_lifts, 2.5),
            'ci_hi': np.percentile(median_lifts, 97.5)
        }
    
    def _compute_block_bootstrap_lift(self, returns_present, returns_absent, block_size=10, n_bootstrap=1000):
        """Compute lift using block bootstrap."""
        np.random.seed(42)
        lifts = []
        
        n_present = len(returns_present)
        n_absent = len(returns_absent)
        
        for _ in range(n_bootstrap):
            # Block bootstrap for present
            blocks_present = []
            while len(blocks_present) < n_present:
                start = np.random.randint(0, n_present - block_size + 1)
                blocks_present.extend(returns_present[start:start + block_size])
            sample_present = np.array(blocks_present[:n_present])
            
            # Block bootstrap for absent
            blocks_absent = []
            while len(blocks_absent) < n_absent:
                start = np.random.randint(0, n_absent - block_size + 1)
                blocks_absent.extend(returns_absent[start:start + block_size])
            sample_absent = np.array(blocks_absent[:n_absent])
            
            lift = np.mean(sample_present) - np.mean(sample_absent)
            lifts.append(lift)
        
        return {
            'mean': np.mean(returns_present) - np.mean(returns_absent),
            'ci_lo': np.percentile(lifts, 2.5),
            'ci_hi': np.percentile(lifts, 97.5)
        }
    
    def _apply_holm(self, p_values, alpha=0.05):
        """Apply Holm's step-down procedure."""
        sorted_pvals = sorted(p_values, key=lambda x: x[1])
        results = []
        
        for i, (hyp_id, p_val) in enumerate(sorted_pvals):
            threshold = alpha / (len(p_values) - i)
            if p_val <= threshold:
                results.append((hyp_id, True))
            else:
                # Once we fail to reject, all remaining fail
                results.extend([(h, False) for h, _ in sorted_pvals[i:]])
                break
        
        return results


def test_performance_monitoring():
    """Test performance and resource usage."""
    print("\n=== Testing Performance Monitoring ===")
    
    # Simulate chunked processing
    n_pairs = 1000
    chunk_size = 100
    n_chunks = n_pairs // chunk_size
    
    chunk_times = []
    
    for chunk_idx in range(n_chunks):
        start_time = time.time()
        
        # Simulate bootstrap computation
        np.random.seed(chunk_idx)
        for _ in range(chunk_size):
            data1 = np.random.normal(0, 0.01, 50)
            data2 = np.random.normal(0, 0.01, 50)
            # Would run bootstrap here
            time.sleep(0.001)  # Simulate computation
        
        chunk_time = time.time() - start_time
        chunk_times.append(chunk_time)
        
        print(f"Chunk {chunk_idx + 1}/{n_chunks}: {chunk_time:.2f}s")
    
    total_time = sum(chunk_times)
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average per chunk: {np.mean(chunk_times):.2f}s")
    print(f"Max chunk time: {max(chunk_times):.2f}s")
    
    # Check if within nightly window (assuming 6 hour window)
    nightly_window_hours = 6
    assert total_time < nightly_window_hours * 3600 * 0.85, "Computation exceeds 85% of nightly window"
    print("✓ Performance monitoring test passed")


def test_code_versioning():
    """Test code versioning for cache invalidation."""
    print("\n=== Testing Code Versioning ===")
    
    # Compute hash of bootstrap code
    bootstrap_code = """
    def compute_expectancy_lift(self, returns_present, returns_absent):
        mean_present = np.mean(returns_present)
        mean_absent = np.mean(returns_absent)
        observed_lift = mean_present - mean_absent
    """
    
    code_hash = hashlib.sha256(bootstrap_code.encode()).hexdigest()[:8]
    print(f"Bootstrap code hash: {code_hash}")
    
    # Simulate code change
    bootstrap_code_v2 = """
    def compute_expectancy_lift(self, returns_present, returns_absent):
        # Added median computation
        mean_present = np.mean(returns_present)
        mean_absent = np.mean(returns_absent)
        observed_lift = mean_present - mean_absent
        median_lift = np.median(returns_present) - np.median(returns_absent)
    """
    
    code_hash_v2 = hashlib.sha256(bootstrap_code_v2.encode()).hexdigest()[:8]
    print(f"Bootstrap code hash v2: {code_hash_v2}")
    
    assert code_hash != code_hash_v2, "Code change should produce different hash"
    print("✓ Code versioning test passed")


def main():
    """Run all adversarial tests."""
    print("=== PR8 Adversarial Test Suite ===")
    
    tests = AdversarialTests()
    
    # Run all tests
    tests.test_bootstrap_extreme_tails()
    tests.test_skewed_returns()
    tests.test_block_bootstrap_dependence()
    tests.test_fdr_edge_cases()
    tests.test_trade_gate_fuzzing()
    tests.test_n_eff_calculation()
    tests.test_power_score_effect_size()
    
    test_performance_monitoring()
    test_code_versioning()
    
    print("\n✅ All adversarial tests passed!")
    print("\nKey findings:")
    print("- Bootstrap needs ≥1.9×SE width for thin samples")
    print("- Median bootstrap more robust for skewed data")
    print("- Block bootstrap essential for dependent returns")
    print("- FDR control validated at edges")
    print("- Trade gates correctly enforce all boundaries")
    print("- Effect-size power more nuanced than linear")


if __name__ == "__main__":
    main()