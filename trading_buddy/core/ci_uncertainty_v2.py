"""
PR8 v2: Enhanced Bootstrap CI with Robustness Features

Implements block bootstrap, median bootstrap, and effect-size power calculations
based on adversarial testing feedback.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

# Version hash for cache invalidation
BOOTSTRAP_CODE_VERSION = hashlib.sha256(b"ci_uncertainty_v2_2024_01").hexdigest()[:8]


@dataclass
class RobustLiftCI:
    """Enhanced lift estimate with additional robustness metrics."""
    mean: float
    median: float
    ci_lo: float
    ci_hi: float
    n_present: int
    n_absent: int
    n_eff: int  # Based on unique events, not bars
    skewness_present: float
    skewness_absent: float
    method: str  # "standard", "median", "block"
    p_value: Optional[float] = None
    power: Optional[float] = None  # Effect-size based power
    code_version: str = BOOTSTRAP_CODE_VERSION


class RobustBootstrapCI:
    """Enhanced bootstrap CI with robustness features."""
    
    def __init__(
        self, 
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
        min_ci_width_factor: float = 1.9,  # Minimum CI width as multiple of SE
        skew_threshold: float = 1.0,  # Switch to median if |skew| > threshold
        block_size: Optional[int] = None,  # For block bootstrap
        effect_size_power: bool = True  # Use effect-size based power
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_state = np.random.RandomState(random_seed)
        self.min_ci_width_factor = min_ci_width_factor
        self.skew_threshold = skew_threshold
        self.block_size = block_size
        self.effect_size_power = effect_size_power
    
    def compute_robust_lift(
        self,
        returns_present: np.ndarray,
        returns_absent: np.ndarray,
        unique_event_ids_present: Optional[List[str]] = None,
        unique_event_ids_absent: Optional[List[str]] = None,
        horizon_bars: int = 10
    ) -> RobustLiftCI:
        """
        Compute lift with automatic method selection based on data characteristics.
        
        Args:
            returns_present: Returns when context is present
            returns_absent: Returns when context is absent
            unique_event_ids_present: Unique event IDs to calculate true n_eff
            unique_event_ids_absent: Unique event IDs to calculate true n_eff
            horizon_bars: Trading horizon for block size selection
        """
        # Calculate true effective sample size
        if unique_event_ids_present and unique_event_ids_absent:
            n_eff = min(len(set(unique_event_ids_present)), len(set(unique_event_ids_absent)))
        else:
            n_eff = min(len(returns_present), len(returns_absent))
        
        # Check for insufficient data
        if n_eff < 10:
            logger.warning(f"Insufficient effective sample size: {n_eff}")
            return self._create_null_result(len(returns_present), len(returns_absent), n_eff)
        
        # Calculate skewness
        skew_present = stats.skew(returns_present)
        skew_absent = stats.skew(returns_absent)
        max_skew = max(abs(skew_present), abs(skew_absent))
        
        # Detect autocorrelation for block bootstrap decision
        acf_present = self._compute_acf(returns_present, nlags=min(10, len(returns_present)//4))
        acf_absent = self._compute_acf(returns_absent, nlags=min(10, len(returns_absent)//4))
        max_acf = max(np.max(np.abs(acf_present[1:])), np.max(np.abs(acf_absent[1:])))
        
        # Select method based on data characteristics
        if max_acf > 0.2 and self.block_size is None:
            # Use block bootstrap for dependent data
            method = "block"
            block_size = horizon_bars
            result = self._compute_block_bootstrap_lift(
                returns_present, returns_absent, block_size
            )
        elif max_skew > self.skew_threshold:
            # Use median bootstrap for skewed data
            method = "median"
            result = self._compute_median_bootstrap_lift(
                returns_present, returns_absent
            )
        else:
            # Use standard mean bootstrap
            method = "standard"
            result = self._compute_mean_bootstrap_lift(
                returns_present, returns_absent
            )
        
        # Enforce minimum CI width
        result = self._enforce_minimum_ci_width(result, returns_present, returns_absent)
        
        # Calculate effect-size based power if requested
        if self.effect_size_power:
            power = self._calculate_effect_size_power(
                result['mean'], returns_present, returns_absent, n_eff
            )
        else:
            power = min(1.0, n_eff / 120.0)  # Linear power
        
        return RobustLiftCI(
            mean=result['mean'],
            median=result['median'],
            ci_lo=result['ci_lo'],
            ci_hi=result['ci_hi'],
            n_present=len(returns_present),
            n_absent=len(returns_absent),
            n_eff=n_eff,
            skewness_present=skew_present,
            skewness_absent=skew_absent,
            method=method,
            p_value=result.get('p_value'),
            power=power
        )
    
    def _compute_mean_bootstrap_lift(self, returns_present, returns_absent):
        """Standard mean-based bootstrap."""
        observed_mean_lift = np.mean(returns_present) - np.mean(returns_absent)
        observed_median_lift = np.median(returns_present) - np.median(returns_absent)
        
        bootstrap_lifts = []
        for _ in range(self.n_bootstrap):
            sample_present = self.random_state.choice(returns_present, size=len(returns_present), replace=True)
            sample_absent = self.random_state.choice(returns_absent, size=len(returns_absent), replace=True)
            
            boot_lift = np.mean(sample_present) - np.mean(sample_absent)
            bootstrap_lifts.append(boot_lift)
        
        ci_lo, ci_hi = np.percentile(bootstrap_lifts, [self.alpha/2 * 100, (1 - self.alpha/2) * 100])
        
        # Compute p-value
        try:
            _, p_value = stats.ttest_ind(returns_present, returns_absent, equal_var=False)
        except:
            p_value = None
        
        return {
            'mean': observed_mean_lift,
            'median': observed_median_lift,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_value': p_value
        }
    
    def _compute_median_bootstrap_lift(self, returns_present, returns_absent):
        """Median-based bootstrap for skewed data."""
        observed_mean_lift = np.mean(returns_present) - np.mean(returns_absent)
        observed_median_lift = np.median(returns_present) - np.median(returns_absent)
        
        bootstrap_lifts = []
        for _ in range(self.n_bootstrap):
            sample_present = self.random_state.choice(returns_present, size=len(returns_present), replace=True)
            sample_absent = self.random_state.choice(returns_absent, size=len(returns_absent), replace=True)
            
            boot_lift = np.median(sample_present) - np.median(sample_absent)
            bootstrap_lifts.append(boot_lift)
        
        ci_lo, ci_hi = np.percentile(bootstrap_lifts, [self.alpha/2 * 100, (1 - self.alpha/2) * 100])
        
        # Use Mann-Whitney U test for p-value
        try:
            _, p_value = stats.mannwhitneyu(returns_present, returns_absent, alternative='two-sided')
        except:
            p_value = None
        
        return {
            'mean': observed_mean_lift,
            'median': observed_median_lift,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_value': p_value
        }
    
    def _compute_block_bootstrap_lift(self, returns_present, returns_absent, block_size):
        """Block bootstrap for dependent data."""
        observed_mean_lift = np.mean(returns_present) - np.mean(returns_absent)
        observed_median_lift = np.median(returns_present) - np.median(returns_absent)
        
        n_present = len(returns_present)
        n_absent = len(returns_absent)
        
        bootstrap_lifts = []
        for _ in range(self.n_bootstrap):
            # Block bootstrap for present
            sample_present = self._block_resample(returns_present, block_size)
            
            # Block bootstrap for absent
            sample_absent = self._block_resample(returns_absent, block_size)
            
            boot_lift = np.mean(sample_present) - np.mean(sample_absent)
            bootstrap_lifts.append(boot_lift)
        
        ci_lo, ci_hi = np.percentile(bootstrap_lifts, [self.alpha/2 * 100, (1 - self.alpha/2) * 100])
        
        return {
            'mean': observed_mean_lift,
            'median': observed_median_lift,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_value': None  # p-value not meaningful for dependent data
        }
    
    def _block_resample(self, data, block_size):
        """Perform block resampling."""
        n = len(data)
        blocks = []
        
        while len(blocks) < n:
            # Circular block bootstrap
            start = self.random_state.randint(0, n)
            for i in range(block_size):
                blocks.append(data[(start + i) % n])
        
        return np.array(blocks[:n])
    
    def _enforce_minimum_ci_width(self, result, returns_present, returns_absent):
        """Ensure CI width is at least min_factor × theoretical SE."""
        # Estimate theoretical SE
        var_present = np.var(returns_present, ddof=1)
        var_absent = np.var(returns_absent, ddof=1)
        se_theoretical = np.sqrt(var_present/len(returns_present) + var_absent/len(returns_absent))
        
        min_width = self.min_ci_width_factor * se_theoretical
        current_width = result['ci_hi'] - result['ci_lo']
        
        if current_width < min_width:
            # Expand CI symmetrically
            expansion = (min_width - current_width) / 2
            result['ci_lo'] -= expansion
            result['ci_hi'] += expansion
            logger.warning(f"CI width expanded from {current_width:.5f} to {min_width:.5f}")
        
        return result
    
    def _calculate_effect_size_power(self, lift, returns_present, returns_absent, n_eff):
        """Calculate power based on effect size (Cohen's d approach)."""
        # Pooled standard deviation
        var_present = np.var(returns_present, ddof=1)
        var_absent = np.var(returns_absent, ddof=1)
        n1, n2 = len(returns_present), len(returns_absent)
        
        pooled_var = ((n1 - 1) * var_present + (n2 - 1) * var_absent) / (n1 + n2 - 2)
        pooled_sd = np.sqrt(pooled_var)
        
        if pooled_sd == 0:
            return 1.0 if lift != 0 else 0.0
        
        # Cohen's d
        d = lift / pooled_sd
        
        # Power calculation for two-sample t-test
        alpha = 0.05
        z_alpha = stats.norm.ppf(1 - alpha/2)
        
        # Non-centrality parameter
        delta = abs(d) * np.sqrt(n_eff / 2)
        
        # Power = P(|T| > z_alpha | H1 true)
        power = 1 - stats.norm.cdf(z_alpha - delta) + stats.norm.cdf(-z_alpha - delta)
        
        return min(max(power, 0.0), 1.0)
    
    def _compute_acf(self, data, nlags):
        """Compute autocorrelation function."""
        if len(data) < nlags + 1:
            return np.zeros(nlags + 1)
        
        acf = np.zeros(nlags + 1)
        data_centered = data - np.mean(data)
        c0 = np.dot(data_centered, data_centered) / len(data)
        
        acf[0] = 1.0
        for k in range(1, nlags + 1):
            c_k = np.dot(data_centered[:-k], data_centered[k:]) / len(data)
            acf[k] = c_k / c0 if c0 > 0 else 0
        
        return acf
    
    def _create_null_result(self, n_present, n_absent, n_eff):
        """Create result for insufficient data."""
        return RobustLiftCI(
            mean=0.0,
            median=0.0,
            ci_lo=0.0,
            ci_hi=0.0,
            n_present=n_present,
            n_absent=n_absent,
            n_eff=n_eff,
            skewness_present=0.0,
            skewness_absent=0.0,
            method="insufficient_data",
            p_value=1.0,
            power=0.0
        )


class MonthlyFDRProcessor:
    """Enhanced FDR processor with small-sample handling."""
    
    def __init__(self, fdr_controller):
        self.fdr_controller = fdr_controller
    
    def process_month_adaptive(
        self,
        year: int,
        month: int,
        symbol: str,
        p_values: List[Tuple[str, float]],
        min_tests_for_bh: int = 20
    ) -> Dict[str, Any]:
        """
        Process FDR with adaptive method selection based on sample size.
        
        Uses Holm for small samples, BH for larger samples.
        """
        n_tests = len(p_values)
        
        if n_tests == 0:
            return {}
        
        if n_tests < min_tests_for_bh:
            # Use Holm's procedure for small samples
            logger.info(f"Using Holm procedure for {symbol} {year}-{month:02d} (n={n_tests})")
            return self._apply_holm_procedure(p_values, alpha=self.fdr_controller.fdr_level)
        else:
            # Use Benjamini-Hochberg for larger samples
            logger.info(f"Using BH procedure for {symbol} {year}-{month:02d} (n={n_tests})")
            return self.fdr_controller.apply_benjamini_hochberg(p_values)
    
    def _apply_holm_procedure(self, p_values: List[Tuple[str, float]], alpha: float):
        """Apply Holm's step-down procedure for small samples."""
        sorted_pvals = sorted(p_values, key=lambda x: x[1])
        results = {}
        
        for i, (hyp_id, p_val) in enumerate(sorted_pvals):
            threshold = alpha / (len(p_values) - i)
            
            if p_val <= threshold:
                results[hyp_id] = {
                    'passes_fdr': True,
                    'adjusted_p': p_val * (len(p_values) - i),
                    'method': 'holm'
                }
            else:
                # Once we fail to reject, all remaining fail
                for j in range(i, len(sorted_pvals)):
                    hyp_id_fail = sorted_pvals[j][0]
                    results[hyp_id_fail] = {
                        'passes_fdr': False,
                        'adjusted_p': min(sorted_pvals[j][1] * (len(p_values) - j), 1.0),
                        'method': 'holm'
                    }
                break
        
        return results