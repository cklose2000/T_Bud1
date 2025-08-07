"""
PR8: Nightly CI v2 Computation Job

Computes bootstrap confidence intervals and FDR control for all
pattern-context pairs to populate consistency_matrix_v2.
"""
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time
from collections import defaultdict

import duckdb
import numpy as np
from trading_buddy.core.ci_uncertainty import BootstrapCI, LiftWithCI
from trading_buddy.core.fdr_control import FDRController, MonthlyFDRProcessor
from trading_buddy.core.config import settings

logger = logging.getLogger(__name__)


class CIv2Computer:
    """Computes CI v2 metrics with uncertainty and FDR control."""
    
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.bootstrap_ci = BootstrapCI(n_bootstrap=1000, confidence_level=0.95)
        self.fdr_controller = FDRController(fdr_level=0.10)
        self.fdr_processor = MonthlyFDRProcessor(self.fdr_controller)
        
    def get_pattern_pairs(self, symbol: str, lookback_days: int = 90) -> List[Tuple[str, str, str, str]]:
        """Get all pattern-context pairs to compute CI for."""
        query = """
        WITH recent_patterns AS (
            SELECT DISTINCT base_tf, base_pattern, ctx_tf, ctx_pattern
            FROM consistency_matrix
            WHERE symbol = ?
            AND updated_at >= CURRENT_DATE - INTERVAL ? DAY
        ),
        recent_events AS (
            SELECT DISTINCT timeframe, pattern
            FROM events
            WHERE symbol = ?
            AND event_ts >= CURRENT_DATE - INTERVAL ? DAY
        )
        SELECT DISTINCT
            rp.base_tf,
            rp.base_pattern,
            rp.ctx_tf, 
            rp.ctx_pattern
        FROM recent_patterns rp
        WHERE EXISTS (
            SELECT 1 FROM recent_events re
            WHERE re.timeframe = rp.base_tf
            AND re.pattern = rp.base_pattern
        )
        """
        
        results = self.conn.execute(query, [symbol, lookback_days, symbol, lookback_days]).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in results]
    
    def get_pattern_returns(
        self,
        symbol: str,
        base_tf: str,
        base_pattern: str,
        ctx_tf: str,
        ctx_pattern: str,
        oos_start: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get OOS returns for when context is present vs absent."""
        # Query to get returns with context presence
        query = """
        WITH base_events AS (
            SELECT 
                e.event_ts,
                e.attrs->>'neckline_price' as neckline,
                o.fwd_ret_10d
            FROM events e
            JOIN outcomes o ON e.event_id = o.event_id
            WHERE e.symbol = ?
            AND e.timeframe = ?
            AND e.pattern = ?
            AND e.event_ts >= ?
            AND o.fwd_ret_10d IS NOT NULL
        ),
        context_presence AS (
            SELECT 
                be.event_ts,
                be.neckline::DOUBLE as neckline,
                be.fwd_ret_10d,
                CASE 
                    WHEN EXISTS (
                        SELECT 1 FROM contexts c
                        WHERE c.symbol = ?
                        AND c.timeframe = ?
                        AND c.context = ?
                        AND c.event_ts BETWEEN be.event_ts - INTERVAL '1 hour' 
                                          AND be.event_ts + INTERVAL '1 hour'
                    ) THEN 1 ELSE 0
                END as context_present
            FROM base_events be
        )
        SELECT 
            fwd_ret_10d,
            context_present
        FROM context_presence
        ORDER BY event_ts
        """
        
        results = self.conn.execute(query, [
            symbol, base_tf, base_pattern, oos_start,
            symbol, ctx_tf, ctx_pattern
        ]).fetchall()
        
        if not results:
            return np.array([]), np.array([])
        
        # Split returns by context presence
        returns_present = np.array([r[0] for r in results if r[1] == 1])
        returns_absent = np.array([r[0] for r in results if r[1] == 0])
        
        return returns_present, returns_absent
    
    def compute_stability(
        self,
        returns_present: np.ndarray,
        returns_absent: np.ndarray,
        window_size: int = 20
    ) -> float:
        """Compute stability metric using rolling window analysis."""
        if len(returns_present) < window_size or len(returns_absent) < window_size:
            return 0.0
        
        # Compute rolling lift stability
        n_windows = min(len(returns_present), len(returns_absent)) - window_size + 1
        if n_windows < 3:
            return 0.0
        
        rolling_lifts = []
        for i in range(n_windows):
            window_present = returns_present[i:i+window_size]
            window_absent = returns_absent[i:i+window_size]
            lift = np.mean(window_present) - np.mean(window_absent)
            rolling_lifts.append(lift)
        
        # Stability = 1 - CV (coefficient of variation)
        mean_lift = np.mean(rolling_lifts)
        std_lift = np.std(rolling_lifts)
        
        if abs(mean_lift) < 1e-6:
            return 0.0
        
        cv = abs(std_lift / mean_lift)
        stability = max(0.0, min(1.0, 1.0 - cv))
        
        return stability
    
    def compute_pattern_ci_v2(
        self,
        symbol: str,
        base_tf: str,
        base_pattern: str, 
        ctx_tf: str,
        ctx_pattern: str,
        oos_start: datetime
    ) -> Optional[Dict]:
        """Compute CI v2 metrics for a single pattern pair."""
        # Get returns
        returns_present, returns_absent = self.get_pattern_returns(
            symbol, base_tf, base_pattern, ctx_tf, ctx_pattern, oos_start
        )
        
        # Need sufficient data
        if len(returns_present) < 10 or len(returns_absent) < 10:
            logger.warning(f"Insufficient data for {base_pattern} x {ctx_pattern}")
            return None
        
        # Compute lifts with bootstrap CI
        exp_lift = self.bootstrap_ci.compute_expectancy_lift(returns_present, returns_absent)
        hit_lift = self.bootstrap_ci.compute_hit_rate_lift(returns_present, returns_absent)
        
        # Compute stability
        stability = self.compute_stability(returns_present, returns_absent)
        
        # Compute power score
        n_eff = min(len(returns_present), len(returns_absent))
        power_score = min(1.0, n_eff / 120.0)
        
        # Raw CI (before adjustments)
        if exp_lift.ci_lo > 0:
            # Positive lift with significance
            ci_raw = 1.0 - np.exp(-exp_lift.mean * 100)  # Sigmoid-like [0, 1]
        else:
            ci_raw = 0.5  # Neutral if not significant
        
        # Apply power and stability adjustments
        ci_final = ci_raw * power_score * stability
        
        # Determine trade gate
        if exp_lift.ci_lo <= 0:
            trade_gate = "watch_only"
            gate_reason = "CI crosses zero"
        elif n_eff < 60:
            ci_final = min(ci_final, 0.2)  # Clamp
            trade_gate = "watch_only"
            gate_reason = f"n_eff={n_eff}<60"
        elif stability < 0.4:
            ci_final = min(ci_final, 0.2)  # Clamp
            trade_gate = "watch_only"
            gate_reason = f"stability={stability:.3f}<0.4"
        else:
            trade_gate = "allowed"
            gate_reason = f"allowed: n_eff={n_eff}, stability={stability:.3f}"
        
        return {
            'n_oos_present': len(returns_present),
            'n_oos_absent': len(returns_absent),
            'exp_lift_mean': exp_lift.mean,
            'exp_lift_ci_lo': exp_lift.ci_lo,
            'exp_lift_ci_hi': exp_lift.ci_hi,
            'p_value_exp': exp_lift.p_value,
            'hit_lift_mean': hit_lift.mean,
            'hit_lift_ci_lo': hit_lift.ci_lo,
            'hit_lift_ci_hi': hit_lift.ci_hi,
            'p_value_hit': hit_lift.p_value,
            'power_score': power_score,
            'stability': stability,
            'ci_raw': ci_raw,
            'ci_final': ci_final,
            'trade_gate': trade_gate,
            'gate_reason': gate_reason
        }
    
    def apply_fdr_control(
        self,
        symbol: str,
        results_by_pair: Dict[Tuple, Dict],
        fdr_month: str
    ):
        """Apply FDR control to all pattern pairs for a month."""
        # Collect p-values
        p_values = []
        pair_keys = []
        
        for pair_key, metrics in results_by_pair.items():
            if metrics and metrics.get('p_value_exp') is not None:
                # Use expectancy p-value as primary test
                p_values.append((str(pair_key), metrics['p_value_exp']))
                pair_keys.append(pair_key)
        
        if not p_values:
            return
        
        # Apply FDR control
        fdr_results = self.fdr_processor.process_month(
            int(fdr_month[:4]), int(fdr_month[5:7]), symbol, p_values
        )
        
        # Update results with FDR info
        for pair_key, fdr_result in zip(pair_keys, fdr_results.values()):
            if pair_key in results_by_pair:
                results_by_pair[pair_key]['fdr_q'] = fdr_result.q_value
                results_by_pair[pair_key]['fdr_pass'] = fdr_result.passes_fdr
                results_by_pair[pair_key]['fdr_month'] = fdr_month
                
                # Update trade gate based on FDR
                if not fdr_result.passes_fdr:
                    results_by_pair[pair_key]['trade_gate'] = 'watch_only'
                    results_by_pair[pair_key]['gate_reason'] += f", FDR q={fdr_result.q_value:.3f}"
    
    def compute_all_ci_v2(
        self,
        symbol: str,
        lookback_days: int = 90,
        oos_start: Optional[datetime] = None
    ):
        """Compute CI v2 for all pattern pairs."""
        if oos_start is None:
            oos_start = datetime.now() - timedelta(days=lookback_days)
        
        # Get all pattern pairs
        pattern_pairs = self.get_pattern_pairs(symbol, lookback_days)
        logger.info(f"Computing CI v2 for {len(pattern_pairs)} pattern pairs in {symbol}")
        
        # Group results by month for FDR
        results_by_month = defaultdict(dict)
        
        # Compute CI for each pair
        start_time = time.time()
        computed = 0
        
        for base_tf, base_pattern, ctx_tf, ctx_pattern in pattern_pairs:
            pair_key = (base_tf, base_pattern, ctx_tf, ctx_pattern)
            
            # Compute metrics
            metrics = self.compute_pattern_ci_v2(
                symbol, base_tf, base_pattern, ctx_tf, ctx_pattern, oos_start
            )
            
            if metrics:
                # Determine month for FDR grouping
                fdr_month = datetime.now().strftime("%Y-%m")
                results_by_month[fdr_month][pair_key] = metrics
                computed += 1
                
                if computed % 10 == 0:
                    logger.info(f"Computed {computed}/{len(pattern_pairs)} pairs...")
        
        # Apply FDR control by month
        for fdr_month, month_results in results_by_month.items():
            self.apply_fdr_control(symbol, month_results, fdr_month)
        
        # Insert into database
        inserted = 0
        for fdr_month, month_results in results_by_month.items():
            for (base_tf, base_pattern, ctx_tf, ctx_pattern), metrics in month_results.items():
                try:
                    computation_time_ms = int((time.time() - start_time) * 1000 / len(pattern_pairs))
                    
                    self.conn.execute("""
                        INSERT INTO consistency_matrix_v2
                        (base_tf, base_pattern, ctx_tf, ctx_pattern, symbol,
                         n_oos_present, n_oos_absent,
                         exp_lift_mean, exp_lift_ci_lo, exp_lift_ci_hi, p_value_exp,
                         hit_lift_mean, hit_lift_ci_lo, hit_lift_ci_hi, p_value_hit,
                         fdr_q, fdr_pass, fdr_month,
                         power_score, stability,
                         ci_raw, ci_final, trade_gate, gate_reason,
                         computation_time_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        base_tf, base_pattern, ctx_tf, ctx_pattern, symbol,
                        metrics['n_oos_present'], metrics['n_oos_absent'],
                        metrics['exp_lift_mean'], metrics['exp_lift_ci_lo'], metrics['exp_lift_ci_hi'],
                        metrics.get('p_value_exp'),
                        metrics['hit_lift_mean'], metrics['hit_lift_ci_lo'], metrics['hit_lift_ci_hi'],
                        metrics.get('p_value_hit'),
                        metrics.get('fdr_q'), metrics.get('fdr_pass', False), metrics.get('fdr_month'),
                        metrics['power_score'], metrics['stability'],
                        metrics['ci_raw'], metrics['ci_final'], metrics['trade_gate'], metrics['gate_reason'],
                        computation_time_ms
                    ])
                    inserted += 1
                    
                except Exception as e:
                    logger.error(f"Failed to insert CI v2 for {base_pattern} x {ctx_pattern}: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"CI v2 computation complete: {inserted}/{len(pattern_pairs)} pairs "
                   f"inserted in {elapsed:.1f}s")
        
        # Update summary statistics
        self._update_summary_stats(symbol)
    
    def _update_summary_stats(self, symbol: str):
        """Update CI v2 summary statistics."""
        try:
            # Summary is maintained via the view, just log current state
            summary = self.conn.execute("""
                SELECT * FROM ci_v2_summary WHERE symbol = ?
            """, [symbol]).fetchone()
            
            if summary:
                logger.info(f"CI v2 summary for {symbol}: "
                           f"{summary[2]} allowed, {summary[3]} watch_only, "
                           f"{summary[4]} blocked, {summary[7]} FDR discoveries")
        except Exception as e:
            logger.error(f"Failed to get CI v2 summary: {e}")


def main():
    """Run CI v2 computation job."""
    import argparse
    parser = argparse.ArgumentParser(description="Compute CI v2 metrics")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument("--lookback-days", type=int, default=90,
                       help="Days of history to consider")
    parser.add_argument("--db-path", default="trading_buddy.db",
                       help="Path to database")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Connect to database
    conn = duckdb.connect(args.db_path)
    
    try:
        # Initialize tables if needed
        with open("sql/pr8_ci_v2_schema.sql", "r") as f:
            conn.execute(f.read())
        
        # Run computation
        computer = CIv2Computer(conn)
        computer.compute_all_ci_v2(
            args.symbol,
            lookback_days=args.lookback_days
        )
        
        conn.commit()
        logger.info("CI v2 computation job completed successfully")
        
    except Exception as e:
        logger.error(f"CI v2 computation failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()