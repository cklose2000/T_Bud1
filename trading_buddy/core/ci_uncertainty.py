"""
PR8: Bootstrap Confidence Intervals for Lift Uncertainty

Computes expectancy and hit rate lifts with bootstrap confidence intervals
to quantify uncertainty in trading edge estimates.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LiftWithCI:
    """Lift estimate with confidence interval."""
    mean: float
    ci_lo: float
    ci_hi: float
    n_present: int
    n_absent: int
    p_value: Optional[float] = None
    
    @property
    def n_eff(self) -> int:
        """Effective sample size for power calculations."""
        return min(self.n_present, self.n_absent)
    
    @property
    def ci_excludes_zero(self) -> bool:
        """Whether confidence interval excludes zero (significant lift)."""
        return self.ci_lo > 0 or self.ci_hi < 0


class BootstrapCI:
    """Computes bootstrap confidence intervals for lift metrics."""
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95, random_seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_state = np.random.RandomState(random_seed)
    
    def compute_expectancy_lift(
        self, 
        returns_present: np.ndarray, 
        returns_absent: np.ndarray,
        compute_p_value: bool = True
    ) -> LiftWithCI:
        """
        Compute expectancy lift with bootstrap CI.
        
        Expectancy lift = mean(returns_present) - mean(returns_absent)
        """
        if len(returns_present) == 0 or len(returns_absent) == 0:
            logger.warning("Empty returns array provided")
            return LiftWithCI(0.0, 0.0, 0.0, len(returns_present), len(returns_absent))
        
        # Observed lift
        mean_present = np.mean(returns_present)
        mean_absent = np.mean(returns_absent)
        observed_lift = mean_present - mean_absent
        
        # Bootstrap
        bootstrap_lifts = []
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            sample_present = self.random_state.choice(returns_present, size=len(returns_present), replace=True)
            sample_absent = self.random_state.choice(returns_absent, size=len(returns_absent), replace=True)
            
            # Compute lift for this bootstrap sample
            boot_lift = np.mean(sample_present) - np.mean(sample_absent)
            bootstrap_lifts.append(boot_lift)
        
        # Compute percentile CI
        ci_lo, ci_hi = np.percentile(bootstrap_lifts, [self.alpha/2 * 100, (1 - self.alpha/2) * 100])
        
        # Optional p-value via Welch's t-test
        p_value = None
        if compute_p_value:
            try:
                _, p_value = stats.ttest_ind(returns_present, returns_absent, equal_var=False)
            except Exception as e:
                logger.warning(f"Failed to compute p-value: {e}")
        
        return LiftWithCI(
            mean=observed_lift,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            n_present=len(returns_present),
            n_absent=len(returns_absent),
            p_value=p_value
        )
    
    def compute_hit_rate_lift(
        self,
        returns_present: np.ndarray,
        returns_absent: np.ndarray,
        compute_p_value: bool = True
    ) -> LiftWithCI:
        """
        Compute hit rate lift with bootstrap CI.
        
        Hit rate = P(return > 0)
        Hit rate lift = hit_rate_present - hit_rate_absent
        """
        if len(returns_present) == 0 or len(returns_absent) == 0:
            logger.warning("Empty returns array provided")
            return LiftWithCI(0.0, 0.0, 0.0, len(returns_present), len(returns_absent))
        
        # Observed hit rates
        hit_rate_present = np.mean(returns_present > 0)
        hit_rate_absent = np.mean(returns_absent > 0)
        observed_lift = hit_rate_present - hit_rate_absent
        
        # Bootstrap
        bootstrap_lifts = []
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            sample_present = self.random_state.choice(returns_present, size=len(returns_present), replace=True)
            sample_absent = self.random_state.choice(returns_absent, size=len(returns_absent), replace=True)
            
            # Compute hit rate lift for this bootstrap sample
            boot_hit_present = np.mean(sample_present > 0)
            boot_hit_absent = np.mean(sample_absent > 0)
            boot_lift = boot_hit_present - boot_hit_absent
            bootstrap_lifts.append(boot_lift)
        
        # Compute percentile CI
        ci_lo, ci_hi = np.percentile(bootstrap_lifts, [self.alpha/2 * 100, (1 - self.alpha/2) * 100])
        
        # Optional p-value via two-proportion z-test
        p_value = None
        if compute_p_value:
            try:
                n1 = len(returns_present)
                n2 = len(returns_absent)
                x1 = np.sum(returns_present > 0)
                x2 = np.sum(returns_absent > 0)
                
                # Pooled proportion
                p_pool = (x1 + x2) / (n1 + n2)
                
                # Standard error
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
                
                # Z-statistic
                if se > 0:
                    z = (hit_rate_present - hit_rate_absent) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                else:
                    p_value = 1.0
                    
            except Exception as e:
                logger.warning(f"Failed to compute p-value: {e}")
        
        return LiftWithCI(
            mean=observed_lift,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            n_present=len(returns_present),
            n_absent=len(returns_absent),
            p_value=p_value
        )
    
    def compute_sharpe_lift(
        self,
        returns_present: np.ndarray,
        returns_absent: np.ndarray
    ) -> LiftWithCI:
        """
        Compute Sharpe ratio lift with bootstrap CI.
        
        Sharpe = mean(returns) / std(returns)
        Sharpe lift = sharpe_present - sharpe_absent
        """
        if len(returns_present) == 0 or len(returns_absent) == 0:
            logger.warning("Empty returns array provided")
            return LiftWithCI(0.0, 0.0, 0.0, len(returns_present), len(returns_absent))
        
        # Helper to compute Sharpe safely
        def safe_sharpe(returns):
            mean_ret = np.mean(returns)
            std_ret = np.std(returns, ddof=1)
            if std_ret > 0:
                return mean_ret / std_ret
            else:
                return 0.0 if mean_ret == 0 else np.sign(mean_ret) * 10.0  # Cap at ±10
        
        # Observed lift
        sharpe_present = safe_sharpe(returns_present)
        sharpe_absent = safe_sharpe(returns_absent)
        observed_lift = sharpe_present - sharpe_absent
        
        # Bootstrap
        bootstrap_lifts = []
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            sample_present = self.random_state.choice(returns_present, size=len(returns_present), replace=True)
            sample_absent = self.random_state.choice(returns_absent, size=len(returns_absent), replace=True)
            
            # Compute Sharpe lift for this bootstrap sample
            boot_sharpe_present = safe_sharpe(sample_present)
            boot_sharpe_absent = safe_sharpe(sample_absent)
            boot_lift = boot_sharpe_present - boot_sharpe_absent
            bootstrap_lifts.append(boot_lift)
        
        # Compute percentile CI
        ci_lo, ci_hi = np.percentile(bootstrap_lifts, [self.alpha/2 * 100, (1 - self.alpha/2) * 100])
        
        return LiftWithCI(
            mean=observed_lift,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            n_present=len(returns_present),
            n_absent=len(returns_absent)
        )
    
    def validate_bootstrap_coverage(
        self, 
        true_diff: float, 
        n1: int, 
        n2: int, 
        n_simulations: int = 100
    ) -> float:
        """
        Validate that bootstrap CIs have correct coverage.
        Used for testing - should cover true parameter ~95% of time.
        """
        covered = 0
        
        for _ in range(n_simulations):
            # Generate data with known difference
            data1 = self.random_state.normal(loc=true_diff/2, scale=1.0, size=n1)
            data2 = self.random_state.normal(loc=-true_diff/2, scale=1.0, size=n2)
            
            # Compute CI
            lift_ci = self.compute_expectancy_lift(data1, data2, compute_p_value=False)
            
            # Check if true difference is in CI
            if lift_ci.ci_lo <= true_diff <= lift_ci.ci_hi:
                covered += 1
        
        return covered / n_simulations