"""
PR8: CI v2 Integration with Power & Stability Clamping

Integrates bootstrap confidence intervals and FDR control into council voting
with proper power score and stability adjustments.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from duckdb import DuckDBPyConnection
from trading_buddy.core.config import settings
from trading_buddy.schemas.council import CIBreakdown

logger = logging.getLogger(__name__)


class CIv2Integration:
    """Handles CI v2 lookups with uncertainty and clamping."""
    
    # Thresholds for gating
    MIN_N_EFF = 60
    MIN_STABILITY = 0.4
    N0_POWER = 120  # Reference sample size for power calculation
    
    def __init__(self, conn: DuckDBPyConnection):
        self.conn = conn
        self.use_v2 = self._check_ci_version()
    
    def _check_ci_version(self) -> bool:
        """Check if CI v2 is enabled based on configuration."""
        try:
            result = self.conn.execute("""
                SELECT version, rollout_percentage 
                FROM ci_version_config 
                WHERE id = 'default'
            """).fetchone()
            
            if result:
                version, rollout_pct = result
                if version == 'v2' and rollout_pct > 0:
                    # Simple rollout: hash request and check if in rollout %
                    # In production, use request ID or user ID for consistent rollout
                    import random
                    return random.randint(1, 100) <= rollout_pct
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to check CI version config: {e}")
            return False
    
    def get_consistency_metrics_v2(
        self,
        symbol: str,
        base_tf: str,
        base_pattern: str,
        ctx_tf: str,
        ctx_pattern: str,
    ) -> Optional[Dict]:
        """
        Get CI v2 metrics with uncertainty and clamping.
        
        Returns dict with:
        - n_eff: Effective sample size
        - exp_lift: {mean, ci_lo, ci_hi}
        - hit_lift: {mean, ci_lo, ci_hi}
        - stability: Stability score
        - power_score: Power adjustment
        - fdr_q: FDR q-value
        - fdr_pass: Whether passes FDR
        - ci_raw: Raw CI before clamping
        - ci_final: Final CI after adjustments
        - trade_gate: 'allowed', 'watch_only', or 'blocked'
        - gate_reason: Explanation for gating decision
        """
        try:
            query = """
            SELECT 
                LEAST(n_oos_present, n_oos_absent) as n_eff,
                n_oos_present,
                n_oos_absent,
                exp_lift_mean,
                exp_lift_ci_lo,
                exp_lift_ci_hi,
                hit_lift_mean,
                hit_lift_ci_lo,
                hit_lift_ci_hi,
                stability,
                power_score,
                fdr_q,
                fdr_pass,
                ci_raw,
                ci_final,
                trade_gate,
                gate_reason
            FROM consistency_matrix_latest
            WHERE symbol = ?
            AND base_tf = ?
            AND base_pattern = ?
            AND ctx_tf = ?
            AND ctx_pattern = ?
            """
            
            result = self.conn.execute(query, [
                symbol, base_tf, base_pattern, ctx_tf, ctx_pattern
            ]).fetchone()
            
            if not result:
                return None
            
            return {
                'n_eff': result[0],
                'n_oos_present': result[1],
                'n_oos_absent': result[2],
                'exp_lift': {
                    'mean': result[3],
                    'ci_lo': result[4],
                    'ci_hi': result[5]
                },
                'hit_lift': {
                    'mean': result[6],
                    'ci_lo': result[7],
                    'ci_hi': result[8]
                },
                'stability': result[9],
                'power_score': result[10],
                'fdr_q': result[11],
                'fdr_pass': result[12],
                'ci_raw': result[13],
                'ci_final': result[14],
                'trade_gate': result[15],
                'gate_reason': result[16]
            }
            
        except Exception as e:
            logger.error(f"Failed to get CI v2 metrics: {e}")
            return None
    
    def compute_clamped_ci(
        self,
        ci_raw: float,
        n_eff: int,
        stability: float,
        exp_lift_ci_lo: float,
        fdr_pass: bool
    ) -> Tuple[float, str, str]:
        """
        Apply power and stability clamping to raw CI.
        
        Returns:
            (ci_final, trade_gate, gate_reason)
        """
        # Compute power score
        power_score = min(1.0, n_eff / self.N0_POWER)
        
        # Check gate conditions
        gate_failures = []
        
        if n_eff < self.MIN_N_EFF:
            gate_failures.append(f"n_eff={n_eff}<{self.MIN_N_EFF}")
        
        if stability < self.MIN_STABILITY:
            gate_failures.append(f"stability={stability:.3f}<{self.MIN_STABILITY}")
        
        if exp_lift_ci_lo <= 0:
            gate_failures.append("CI_crosses_zero")
        
        if not fdr_pass:
            gate_failures.append("FDR_fail")
        
        # Apply clamping based on gate status
        if gate_failures:
            # Clamp to max 0.2 when any gate fails
            ci_final = min(ci_raw, 0.2) * power_score * stability
            trade_gate = "watch_only"
            gate_reason = "watch-only: " + ", ".join(gate_failures)
        else:
            # Normal adjustment when all gates pass
            ci_final = ci_raw * power_score * stability
            trade_gate = "allowed"
            gate_reason = f"allowed: n_eff={n_eff}, stability={stability:.3f}"
        
        return ci_final, trade_gate, gate_reason
    
    def create_ci_breakdown_v2(
        self,
        symbol: str,
        base_tf: str,
        base_pattern: str,
        contexts: List[Tuple[str, str, bool]],  # [(ctx_tf, ctx_pattern, present)]
    ) -> List[CIBreakdown]:
        """
        Create CI breakdown using v2 metrics with uncertainty.
        """
        breakdown = []
        
        for ctx_tf, ctx_pattern, present in contexts:
            metrics = self.get_consistency_metrics_v2(
                symbol, base_tf, base_pattern, ctx_tf, ctx_pattern
            )
            
            if metrics:
                # Format confidence intervals for display
                exp_lift_str = (f"{metrics['exp_lift']['mean']:.4f} "
                              f"[{metrics['exp_lift']['ci_lo']:.4f}, "
                              f"{metrics['exp_lift']['ci_hi']:.4f}]")
                
                hit_lift_str = (f"{metrics['hit_lift']['mean']:.3f} "
                              f"[{metrics['hit_lift']['ci_lo']:.3f}, "
                              f"{metrics['hit_lift']['ci_hi']:.3f}]")
                
                breakdown.append(CIBreakdown(
                    context_tf=ctx_tf,
                    context_name=ctx_pattern,
                    present=present,
                    ci=metrics['ci_final'],
                    n=metrics['n_eff'],
                    exp_lift=metrics['exp_lift']['mean'],
                    hit_rate=metrics['hit_lift']['mean'],  # Approximation
                    stability=metrics['stability'],
                    metadata={
                        'exp_lift_ci': exp_lift_str,
                        'hit_lift_ci': hit_lift_str,
                        'power_score': metrics['power_score'],
                        'fdr_q': metrics['fdr_q'],
                        'fdr_pass': metrics['fdr_pass'],
                        'trade_gate': metrics['trade_gate'],
                        'gate_reason': metrics['gate_reason']
                    }
                ))
            else:
                # No data - use neutral values
                breakdown.append(CIBreakdown(
                    context_tf=ctx_tf,
                    context_name=ctx_pattern,
                    present=present,
                    ci=0.0,
                    n=0,
                    exp_lift=0.0,
                    hit_rate=0.5,
                    stability=0.0,
                    metadata={'trade_gate': 'blocked', 'gate_reason': 'no_data'}
                ))
        
        return breakdown
    
    def log_v2_decision(
        self,
        request_id: str,
        symbol: str,
        base_pattern: str,
        breakdown: List[CIBreakdown]
    ):
        """Log CI v2 decision for monitoring and debugging."""
        try:
            for item in breakdown:
                if item.metadata and 'trade_gate' in item.metadata:
                    self.conn.execute("""
                        INSERT INTO ci_v2_decision_log
                        (request_id, symbol, base_pattern, ctx_pattern,
                         n_eff, exp_lift_mean, stability, power_score,
                         fdr_q, fdr_pass, ci_raw, ci_final, 
                         trade_gate, gate_reason, ci_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v2')
                    """, [
                        request_id,
                        symbol,
                        base_pattern,
                        item.context_name,
                        item.n,
                        item.exp_lift,
                        item.stability,
                        item.metadata.get('power_score', 0),
                        item.metadata.get('fdr_q'),
                        item.metadata.get('fdr_pass', False),
                        item.ci / (item.stability * item.metadata.get('power_score', 1)),  # Approx ci_raw
                        item.ci,
                        item.metadata['trade_gate'],
                        item.metadata.get('gate_reason', '')
                    ])
                    
        except Exception as e:
            logger.error(f"Failed to log CI v2 decision: {e}")
    
    def format_rejection_reason(self, breakdown: List[CIBreakdown]) -> Optional[str]:
        """Format human-readable rejection reason from CI breakdown."""
        blocked_contexts = [
            item for item in breakdown 
            if item.metadata and item.metadata.get('trade_gate') in ['watch_only', 'blocked']
        ]
        
        if not blocked_contexts:
            return None
        
        reasons = []
        for ctx in blocked_contexts:
            reason = ctx.metadata.get('gate_reason', 'unknown')
            reasons.append(f"{ctx.context_name}: {reason}")
        
        return "CI v2 restrictions - " + "; ".join(reasons)