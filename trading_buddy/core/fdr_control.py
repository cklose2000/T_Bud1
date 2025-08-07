"""
PR8: False Discovery Rate (FDR) Control via Benjamini-Hochberg

Controls the false discovery rate when testing multiple hypotheses
(e.g., many pattern-context pairs) to avoid spurious discoveries.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FDRResult:
    """Result of FDR control procedure."""
    q_value: float  # FDR-adjusted p-value
    passes_fdr: bool  # Whether hypothesis passes FDR threshold
    rank: int  # Rank in sorted p-values
    threshold: float  # BH threshold for this rank
    n_tests: int  # Total number of tests
    n_discoveries: int  # Number of discoveries at this FDR level


class FDRController:
    """Implements Benjamini-Hochberg FDR control for multiple hypothesis testing."""
    
    def __init__(self, fdr_level: float = 0.10):
        """
        Initialize FDR controller.
        
        Args:
            fdr_level: Target false discovery rate (default 0.10 = 10%)
        """
        self.fdr_level = fdr_level
    
    def apply_benjamini_hochberg(
        self, 
        p_values: List[Tuple[str, float]]
    ) -> Dict[str, FDRResult]:
        """
        Apply Benjamini-Hochberg procedure to control FDR.
        
        Args:
            p_values: List of (hypothesis_id, p_value) tuples
            
        Returns:
            Dict mapping hypothesis_id to FDRResult
        """
        if not p_values:
            return {}
        
        # Sort by p-value
        sorted_pvals = sorted(p_values, key=lambda x: x[1])
        n_tests = len(sorted_pvals)
        
        # Find the largest k such that P(k) <= (k/m) * q
        k_star = 0
        for k, (_, p_val) in enumerate(sorted_pvals, 1):
            threshold = (k / n_tests) * self.fdr_level
            if p_val <= threshold:
                k_star = k
        
        # Compute q-values (FDR-adjusted p-values)
        results = {}
        
        for rank, (hyp_id, p_val) in enumerate(sorted_pvals, 1):
            # BH threshold for this rank
            threshold = (rank / n_tests) * self.fdr_level
            
            # q-value is the minimum FDR at which this hypothesis would be called significant
            q_value = p_val * n_tests / rank
            q_value = min(q_value, 1.0)  # Cap at 1
            
            # Enforce monotonicity (q-values should be non-decreasing)
            if rank > 1:
                prev_q = results[sorted_pvals[rank-2][0]].q_value
                q_value = max(q_value, prev_q)
            
            # Passes if rank <= k*
            passes_fdr = rank <= k_star
            
            results[hyp_id] = FDRResult(
                q_value=q_value,
                passes_fdr=passes_fdr,
                rank=rank,
                threshold=threshold,
                n_tests=n_tests,
                n_discoveries=k_star
            )
        
        logger.info(f"BH procedure: {k_star}/{n_tests} discoveries at FDR={self.fdr_level}")
        
        return results
    
    def apply_by_group(
        self,
        p_values_by_group: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, Dict[str, FDRResult]]:
        """
        Apply FDR control separately within each group (e.g., by month or symbol).
        
        Args:
            p_values_by_group: Dict mapping group_id to list of (hypothesis_id, p_value)
            
        Returns:
            Nested dict: group_id -> hypothesis_id -> FDRResult
        """
        results_by_group = {}
        
        for group_id, p_values in p_values_by_group.items():
            if p_values:
                results_by_group[group_id] = self.apply_benjamini_hochberg(p_values)
                logger.info(f"FDR control for group {group_id}: "
                          f"{sum(1 for r in results_by_group[group_id].values() if r.passes_fdr)}"
                          f"/{len(p_values)} pass")
            else:
                results_by_group[group_id] = {}
        
        return results_by_group
    
    def compute_fdr_metrics(self, fdr_results: Dict[str, FDRResult]) -> Dict[str, Any]:
        """Compute summary metrics for FDR results."""
        if not fdr_results:
            return {
                'n_tests': 0,
                'n_discoveries': 0,
                'discovery_rate': 0.0,
                'median_q_value': None,
                'min_q_value': None
            }
        
        n_tests = len(fdr_results)
        n_discoveries = sum(1 for r in fdr_results.values() if r.passes_fdr)
        q_values = [r.q_value for r in fdr_results.values()]
        
        return {
            'n_tests': n_tests,
            'n_discoveries': n_discoveries,
            'discovery_rate': n_discoveries / n_tests if n_tests > 0 else 0.0,
            'median_q_value': np.median(q_values),
            'min_q_value': np.min(q_values),
            'max_q_value': np.max(q_values),
            'fdr_level': self.fdr_level
        }
    
    def validate_fdr_control(
        self, 
        n_null: int, 
        n_alt: int, 
        null_p_dist='uniform',
        alt_p_dist='beta',
        n_simulations: int = 100
    ) -> Dict[str, float]:
        """
        Validate that FDR is controlled at specified level via simulation.
        
        Args:
            n_null: Number of true null hypotheses
            n_alt: Number of true alternative hypotheses  
            null_p_dist: Distribution of p-values under null ('uniform')
            alt_p_dist: Distribution of p-values under alternative ('beta')
            n_simulations: Number of simulations to run
            
        Returns:
            Dict with empirical FDR and power
        """
        false_discoveries = []
        true_discoveries = []
        
        for _ in range(n_simulations):
            p_values = []
            true_status = []  # 0 for null, 1 for alternative
            
            # Generate p-values for null hypotheses
            for i in range(n_null):
                if null_p_dist == 'uniform':
                    p = np.random.uniform(0, 1)
                else:
                    raise ValueError(f"Unknown null distribution: {null_p_dist}")
                p_values.append((f"null_{i}", p))
                true_status.append(0)
            
            # Generate p-values for alternative hypotheses
            for i in range(n_alt):
                if alt_p_dist == 'beta':
                    # Beta(0.5, 5) gives p-values concentrated near 0
                    p = np.random.beta(0.5, 5)
                else:
                    raise ValueError(f"Unknown alternative distribution: {alt_p_dist}")
                p_values.append((f"alt_{i}", p))
                true_status.append(1)
            
            # Apply BH procedure
            fdr_results = self.apply_benjamini_hochberg(p_values)
            
            # Count false and true discoveries
            sim_false_disc = 0
            sim_true_disc = 0
            
            for (hyp_id, _), status in zip(p_values, true_status):
                if fdr_results[hyp_id].passes_fdr:
                    if status == 0:  # Null hypothesis
                        sim_false_disc += 1
                    else:  # Alternative hypothesis
                        sim_true_disc += 1
            
            # Record FDR and power for this simulation
            total_disc = sim_false_disc + sim_true_disc
            if total_disc > 0:
                false_discoveries.append(sim_false_disc / total_disc)
            else:
                false_discoveries.append(0.0)
            
            if n_alt > 0:
                true_discoveries.append(sim_true_disc / n_alt)
            else:
                true_discoveries.append(0.0)
        
        return {
            'empirical_fdr': np.mean(false_discoveries),
            'empirical_power': np.mean(true_discoveries),
            'fdr_controlled': np.mean(false_discoveries) <= self.fdr_level,
            'target_fdr': self.fdr_level
        }


class MonthlyFDRProcessor:
    """Process FDR control on a monthly basis for pattern discoveries."""
    
    def __init__(self, fdr_controller: FDRController):
        self.fdr_controller = fdr_controller
    
    def process_month(
        self,
        year: int,
        month: int,
        symbol: str,
        p_values: List[Tuple[str, float]]
    ) -> Dict[str, FDRResult]:
        """
        Process FDR for a specific month and symbol.
        
        Args:
            year: Year to process
            month: Month to process (1-12)
            symbol: Trading symbol
            p_values: List of (pattern_pair_id, p_value)
            
        Returns:
            FDR results for each pattern pair
        """
        logger.info(f"Processing FDR for {symbol} {year}-{month:02d} with {len(p_values)} tests")
        
        if not p_values:
            return {}
        
        # Apply BH procedure
        results = self.fdr_controller.apply_benjamini_hochberg(p_values)
        
        # Log summary
        n_discoveries = sum(1 for r in results.values() if r.passes_fdr)
        logger.info(f"FDR results for {symbol} {year}-{month:02d}: "
                   f"{n_discoveries}/{len(p_values)} discoveries at FDR={self.fdr_controller.fdr_level}")
        
        return results