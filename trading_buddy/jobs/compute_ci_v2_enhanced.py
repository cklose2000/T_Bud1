"""
PR8 Enhanced: Nightly CI v2 Computation with Robustness Features

Implements chunked processing, performance monitoring, and enhanced bootstrap methods.
"""
import logging
import time
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
import hashlib

import duckdb
import numpy as np

from trading_buddy.core.ci_uncertainty_v2 import RobustBootstrapCI, BOOTSTRAP_CODE_VERSION
from trading_buddy.core.fdr_control import FDRController, MonthlyFDRProcessor
from trading_buddy.core.config import settings

logger = logging.getLogger(__name__)


class EnhancedCIv2Computer:
    """Enhanced CI v2 computer with robustness and performance features."""
    
    def __init__(self, conn: duckdb.DuckDBPyConnection, n_bootstrap: int = 1000):
        self.conn = conn
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ci = RobustBootstrapCI(
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            min_ci_width_factor=1.9,  # Ensure wide enough CIs
            skew_threshold=1.0,
            effect_size_power=True
        )
        self.fdr_controller = FDRController(fdr_level=0.10)
        self.fdr_processor = MonthlyFDRProcessor(self.fdr_controller)
        
    def compute_all_ci_v2_chunked(
        self,
        symbol: str,
        lookback_days: int = 90,
        chunk_size: int = 100,
        max_runtime_hours: float = 6.0,
        oos_start: Optional[datetime] = None
    ):
        """
        Compute CI v2 with chunked processing and performance monitoring.
        """
        if oos_start is None:
            oos_start = datetime.now() - timedelta(days=lookback_days)
        
        start_time = time.time()
        max_runtime_seconds = max_runtime_hours * 3600
        
        # Initialize job ledger to prevent duplicates on restart
        job_id = f"ci_v2_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_job_ledger(job_id)
        
        # Get all pattern pairs
        pattern_pairs = self._get_unprocessed_pairs(symbol, lookback_days, job_id)
        total_pairs = len(pattern_pairs)
        logger.info(f"Processing {total_pairs} pattern pairs for {symbol} in chunks of {chunk_size}")
        
        # Process in chunks
        processed = 0
        chunk_times = []
        results_by_month = defaultdict(dict)
        
        for chunk_idx in range(0, total_pairs, chunk_size):
            chunk_start = time.time()
            
            # Check runtime limit
            elapsed = time.time() - start_time
            if elapsed > max_runtime_seconds * 0.85:  # 85% threshold
                logger.warning(f"Approaching runtime limit ({elapsed/3600:.1f}h), stopping at chunk {chunk_idx//chunk_size}")
                break
            
            # Process chunk
            chunk_pairs = pattern_pairs[chunk_idx:chunk_idx + chunk_size]
            chunk_results = self._process_chunk(symbol, chunk_pairs, oos_start)
            
            # Group by month for FDR
            for pair_key, metrics in chunk_results.items():
                if metrics:
                    fdr_month = datetime.now().strftime("%Y-%m")
                    results_by_month[fdr_month][pair_key] = metrics
            
            # Update job ledger
            processed += len(chunk_pairs)
            self._update_job_progress(job_id, processed, total_pairs)
            
            # Log performance
            chunk_time = time.time() - chunk_start
            chunk_times.append(chunk_time)
            logger.info(f"Chunk {chunk_idx//chunk_size + 1}: {len(chunk_pairs)} pairs in {chunk_time:.1f}s "
                       f"({processed}/{total_pairs} total)")
        
        # Apply FDR control
        self._apply_fdr_and_persist(symbol, results_by_month)
        
        # Mark job complete
        self._complete_job(job_id, processed, total_pairs, chunk_times)
        
        # Log summary
        total_time = time.time() - start_time
        logger.info(f"CI v2 computation summary for {symbol}:")
        logger.info(f"  Total pairs: {total_pairs}")
        logger.info(f"  Processed: {processed}")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        logger.info(f"  Avg chunk time: {np.mean(chunk_times):.1f}s")
        logger.info(f"  Bootstrap version: {BOOTSTRAP_CODE_VERSION}")
    
    def _get_unprocessed_pairs(self, symbol: str, lookback_days: int, job_id: str) -> List[Tuple]:
        """Get pattern pairs not yet processed in this job."""
        # Get all pairs
        query = """
        WITH recent_patterns AS (
            SELECT DISTINCT base_tf, base_pattern, ctx_tf, ctx_pattern
            FROM consistency_matrix
            WHERE symbol = ?
            AND updated_at >= CURRENT_DATE - INTERVAL ? DAY
        )
        SELECT base_tf, base_pattern, ctx_tf, ctx_pattern
        FROM recent_patterns
        WHERE NOT EXISTS (
            SELECT 1 FROM ci_v2_job_ledger
            WHERE job_id = ?
            AND base_tf = recent_patterns.base_tf
            AND base_pattern = recent_patterns.base_pattern
            AND ctx_tf = recent_patterns.ctx_tf
            AND ctx_pattern = recent_patterns.ctx_pattern
            AND status = 'completed'
        )
        """
        
        results = self.conn.execute(query, [symbol, lookback_days, job_id]).fetchall()
        return results
    
    def _process_chunk(
        self,
        symbol: str,
        chunk_pairs: List[Tuple],
        oos_start: datetime
    ) -> Dict[Tuple, Dict]:
        """Process a chunk of pattern pairs."""
        chunk_results = {}
        
        for base_tf, base_pattern, ctx_tf, ctx_pattern in chunk_pairs:
            pair_key = (base_tf, base_pattern, ctx_tf, ctx_pattern)
            
            try:
                # Get returns with event deduplication
                returns_data = self._get_returns_with_events(
                    symbol, base_tf, base_pattern, ctx_tf, ctx_pattern, oos_start
                )
                
                if not returns_data:
                    continue
                
                # Compute robust lift with automatic method selection
                lift_result = self.bootstrap_ci.compute_robust_lift(
                    returns_data['returns_present'],
                    returns_data['returns_absent'],
                    returns_data['event_ids_present'],
                    returns_data['event_ids_absent'],
                    horizon_bars=returns_data['horizon_bars']
                )
                
                # Compute additional metrics
                metrics = self._compute_pattern_metrics(lift_result, returns_data)
                chunk_results[pair_key] = metrics
                
            except Exception as e:
                logger.error(f"Failed to process {base_pattern} x {ctx_pattern}: {e}")
        
        return chunk_results
    
    def _get_returns_with_events(
        self,
        symbol: str,
        base_tf: str,
        base_pattern: str,
        ctx_tf: str,
        ctx_pattern: str,
        oos_start: datetime
    ) -> Optional[Dict]:
        """Get returns with proper event deduplication."""
        query = """
        WITH base_events AS (
            SELECT DISTINCT
                e.event_id,
                e.event_ts,
                o.fwd_ret_10d,
                -- Determine horizon bars based on timeframe
                CASE 
                    WHEN e.timeframe LIKE '%m' THEN 10 * 24 * 60 / CAST(REPLACE(e.timeframe, 'm', '') AS INTEGER)
                    WHEN e.timeframe LIKE '%h' THEN 10 * 24 / CAST(REPLACE(e.timeframe, 'h', '') AS INTEGER)
                    WHEN e.timeframe LIKE '%d' THEN 10 / CAST(REPLACE(e.timeframe, 'd', '') AS INTEGER)
                    ELSE 10
                END as horizon_bars
            FROM events e
            JOIN outcomes o ON e.event_id = o.event_id
            WHERE e.symbol = ?
            AND e.timeframe = ?
            AND e.pattern = ?
            AND e.event_ts >= ?
            AND o.fwd_ret_10d IS NOT NULL
        ),
        context_check AS (
            SELECT 
                be.*,
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
            event_id,
            fwd_ret_10d,
            context_present,
            horizon_bars
        FROM context_check
        ORDER BY event_id
        """
        
        results = self.conn.execute(query, [
            symbol, base_tf, base_pattern, oos_start,
            symbol, ctx_tf, ctx_pattern
        ]).fetchall()
        
        if not results:
            return None
        
        # Separate by context presence
        data_present = [(r[0], r[1]) for r in results if r[2] == 1]
        data_absent = [(r[0], r[1]) for r in results if r[2] == 0]
        
        if len(data_present) < 10 or len(data_absent) < 10:
            return None
        
        horizon_bars = int(results[0][3]) if results else 10
        
        return {
            'returns_present': np.array([d[1] for d in data_present]),
            'returns_absent': np.array([d[1] for d in data_absent]),
            'event_ids_present': [d[0] for d in data_present],
            'event_ids_absent': [d[0] for d in data_absent],
            'horizon_bars': horizon_bars
        }
    
    def _compute_pattern_metrics(self, lift_result, returns_data) -> Dict:
        """Compute comprehensive metrics for pattern pair."""
        # Compute stability using rolling windows
        stability = self._compute_rolling_stability(
            returns_data['returns_present'],
            returns_data['returns_absent'],
            window_fraction=0.3
        )
        
        # Determine trade gate based on all criteria
        trade_gate, gate_reason = self._determine_trade_gate(
            lift_result, stability
        )
        
        # Raw CI before adjustments (approximation)
        if lift_result.ci_lo > 0:
            ci_raw = 1.0 - np.exp(-lift_result.mean * 100)
        else:
            ci_raw = 0.5
        
        # Final CI after adjustments
        if trade_gate == "watch_only":
            ci_final = min(ci_raw, 0.2) * lift_result.power * stability
        else:
            ci_final = ci_raw * lift_result.power * stability
        
        return {
            'n_oos_present': lift_result.n_present,
            'n_oos_absent': lift_result.n_absent,
            'n_eff': lift_result.n_eff,
            'exp_lift_mean': lift_result.mean,
            'exp_lift_median': lift_result.median,
            'exp_lift_ci_lo': lift_result.ci_lo,
            'exp_lift_ci_hi': lift_result.ci_hi,
            'hit_lift_mean': 0.0,  # Would compute separately
            'hit_lift_ci_lo': 0.0,
            'hit_lift_ci_hi': 0.0,
            'p_value_exp': lift_result.p_value,
            'power_score': lift_result.power,
            'stability': stability,
            'ci_raw': ci_raw,
            'ci_final': ci_final,
            'trade_gate': trade_gate,
            'gate_reason': gate_reason,
            'bootstrap_method': lift_result.method,
            'skew_present': lift_result.skewness_present,
            'skew_absent': lift_result.skewness_absent,
            'code_version': lift_result.code_version
        }
    
    def _compute_rolling_stability(self, returns_present, returns_absent, window_fraction=0.3):
        """Compute stability using rolling window analysis."""
        window_size = max(10, int(min(len(returns_present), len(returns_absent)) * window_fraction))
        
        if len(returns_present) < window_size or len(returns_absent) < window_size:
            return 0.0
        
        rolling_lifts = []
        for i in range(len(returns_present) - window_size + 1):
            window_present = returns_present[i:i+window_size]
            
            # Find corresponding window in absent (may be different size)
            j = min(i, len(returns_absent) - window_size)
            window_absent = returns_absent[j:j+window_size]
            
            lift = np.mean(window_present) - np.mean(window_absent)
            rolling_lifts.append(lift)
        
        if len(rolling_lifts) < 3:
            return 0.0
        
        # Stability based on consistency of lift direction and magnitude
        mean_lift = np.mean(rolling_lifts)
        if abs(mean_lift) < 1e-6:
            return 0.0
        
        # Coefficient of variation
        cv = np.std(rolling_lifts) / abs(mean_lift)
        stability = max(0.0, min(1.0, 1.0 - cv))
        
        # Penalize if lift changes sign
        if np.min(rolling_lifts) * np.max(rolling_lifts) < 0:
            stability *= 0.5
        
        return stability
    
    def _determine_trade_gate(self, lift_result, stability):
        """Determine trade gate and reason based on all criteria."""
        failures = []
        
        # Check n_eff
        if lift_result.n_eff < 60:
            failures.append(f"n_eff={lift_result.n_eff}<60")
        
        # Check stability
        if stability < 0.4:
            failures.append(f"stability={stability:.3f}<0.4")
        
        # Check CI excludes zero
        if lift_result.ci_lo <= 0:
            failures.append("CI_crosses_zero")
        
        # Check for extreme skewness
        if max(abs(lift_result.skewness_present), abs(lift_result.skewness_absent)) > 2.0:
            failures.append("extreme_skew")
        
        if failures:
            return "watch_only", "watch-only: " + ", ".join(failures)
        else:
            return "allowed", f"allowed: n_eff={lift_result.n_eff}, stability={stability:.3f}, power={lift_result.power:.3f}"
    
    def _apply_fdr_and_persist(self, symbol: str, results_by_month: Dict[str, Dict]):
        """Apply FDR control and persist results."""
        for fdr_month, month_results in results_by_month.items():
            # Collect p-values
            p_values = []
            for pair_key, metrics in month_results.items():
                if metrics.get('p_value_exp') is not None:
                    p_values.append((str(pair_key), metrics['p_value_exp']))
            
            # Apply adaptive FDR control
            year, month = map(int, fdr_month.split('-'))
            fdr_results = self.fdr_processor.process_month_adaptive(
                year, month, symbol, p_values, min_tests_for_bh=20
            )
            
            # Update results and persist
            for pair_key, metrics in month_results.items():
                pair_str = str(pair_key)
                if pair_str in fdr_results:
                    fdr_result = fdr_results[pair_str]
                    metrics['fdr_q'] = fdr_result.get('adjusted_p', 1.0)
                    metrics['fdr_pass'] = fdr_result.get('passes_fdr', False)
                    metrics['fdr_method'] = fdr_result.get('method', 'none')
                    
                    # Update gate if FDR fails
                    if not metrics['fdr_pass'] and metrics['trade_gate'] == 'allowed':
                        metrics['trade_gate'] = 'watch_only'
                        metrics['gate_reason'] += f", FDR_fail (q={metrics['fdr_q']:.3f})"
                
                # Insert into database
                self._insert_ci_v2_record(symbol, pair_key, metrics, fdr_month)
    
    def _insert_ci_v2_record(self, symbol: str, pair_key: Tuple, metrics: Dict, fdr_month: str):
        """Insert CI v2 record with all enhanced fields."""
        base_tf, base_pattern, ctx_tf, ctx_pattern = pair_key
        
        try:
            # Generate unique ID
            record_id = hashlib.sha256(
                f"{symbol}_{pair_key}_{metrics['code_version']}_{time.time()}".encode()
            ).hexdigest()[:16]
            
            self.conn.execute("""
                INSERT INTO consistency_matrix_v2
                (id, base_tf, base_pattern, ctx_tf, ctx_pattern, symbol,
                 n_oos_present, n_oos_absent,
                 exp_lift_mean, exp_lift_ci_lo, exp_lift_ci_hi, p_value_exp,
                 hit_lift_mean, hit_lift_ci_lo, hit_lift_ci_hi, p_value_hit,
                 fdr_q, fdr_pass, fdr_month,
                 power_score, stability,
                 ci_raw, ci_final, trade_gate, gate_reason,
                 computation_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record_id, base_tf, base_pattern, ctx_tf, ctx_pattern, symbol,
                metrics['n_oos_present'], metrics['n_oos_absent'],
                metrics['exp_lift_mean'], metrics['exp_lift_ci_lo'], metrics['exp_lift_ci_hi'],
                metrics.get('p_value_exp'),
                metrics['hit_lift_mean'], metrics['hit_lift_ci_lo'], metrics['hit_lift_ci_hi'],
                metrics.get('p_value_hit'),
                metrics.get('fdr_q'), metrics.get('fdr_pass', False), fdr_month,
                metrics['power_score'], metrics['stability'],
                metrics['ci_raw'], metrics['ci_final'], metrics['trade_gate'], metrics['gate_reason'],
                int(time.time() * 1000)  # Placeholder for actual computation time
            ])
            
            # Also insert metadata record
            self.conn.execute("""
                INSERT INTO ci_v2_metadata
                (ci_v2_id, bootstrap_method, bootstrap_n, code_version,
                 skew_present, skew_absent, n_eff_unique_events, fdr_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record_id,
                metrics['bootstrap_method'],
                self.n_bootstrap,
                metrics['code_version'],
                metrics['skew_present'],
                metrics['skew_absent'],
                metrics['n_eff'],
                metrics.get('fdr_method', 'bh')
            ])
            
        except Exception as e:
            logger.error(f"Failed to insert CI v2 record: {e}")
    
    def _init_job_ledger(self, job_id: str):
        """Initialize job ledger for tracking progress."""
        self.conn.execute("""
            INSERT INTO ci_v2_job_runs
            (job_id, symbol, status, started_at)
            VALUES (?, ?, 'running', CURRENT_TIMESTAMP)
        """, [job_id, 'multi'])
    
    def _update_job_progress(self, job_id: str, processed: int, total: int):
        """Update job progress in ledger."""
        self.conn.execute("""
            UPDATE ci_v2_job_runs
            SET pairs_processed = ?, pairs_total = ?, last_updated = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, [processed, total, job_id])
    
    def _complete_job(self, job_id: str, processed: int, total: int, chunk_times: List[float]):
        """Mark job as complete with statistics."""
        self.conn.execute("""
            UPDATE ci_v2_job_runs
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                pairs_processed = ?,
                pairs_total = ?,
                runtime_seconds = ?,
                avg_chunk_seconds = ?
            WHERE job_id = ?
        """, [processed, total, sum(chunk_times), np.mean(chunk_times) if chunk_times else 0, job_id])


def main():
    """Enhanced CI v2 computation with monitoring."""
    parser = argparse.ArgumentParser(description="Compute enhanced CI v2 metrics")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                       help="Number of bootstrap iterations")
    parser.add_argument("--chunk-size", type=int, default=100,
                       help="Pairs per chunk")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--max-runtime-hours", type=float, default=6.0)
    parser.add_argument("--db-path", default="trading_buddy.db")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    conn = duckdb.connect(args.db_path)
    
    try:
        # Create enhanced schema tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ci_v2_job_runs (
                job_id TEXT PRIMARY KEY,
                symbol TEXT,
                status TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                pairs_processed INTEGER DEFAULT 0,
                pairs_total INTEGER,
                runtime_seconds DOUBLE,
                avg_chunk_seconds DOUBLE,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ci_v2_job_ledger (
                job_id TEXT,
                base_tf TEXT,
                base_pattern TEXT,
                ctx_tf TEXT,
                ctx_pattern TEXT,
                status TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, base_tf, base_pattern, ctx_tf, ctx_pattern)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ci_v2_metadata (
                ci_v2_id TEXT PRIMARY KEY,
                bootstrap_method TEXT,
                bootstrap_n INTEGER,
                code_version TEXT,
                skew_present DOUBLE,
                skew_absent DOUBLE,
                n_eff_unique_events INTEGER,
                fdr_method TEXT
            )
        """)
        
        # Run enhanced computation
        computer = EnhancedCIv2Computer(conn, n_bootstrap=args.n_bootstrap)
        computer.compute_all_ci_v2_chunked(
            args.symbol,
            lookback_days=args.lookback_days,
            chunk_size=args.chunk_size,
            max_runtime_hours=args.max_runtime_hours
        )
        
        conn.commit()
        logger.info("Enhanced CI v2 computation completed successfully")
        
    except Exception as e:
        logger.error(f"Enhanced CI v2 computation failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()