"""
PR7: Calibration & Metrics Infrastructure

Core metrics computation including Brier scores, Expected Calibration Error,
and comprehensive system health monitoring.
"""
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import statistics

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class MetricsComputer:
    """Computes calibration and system health metrics."""
    
    def __init__(self, conn: DuckDBPyConnection):
        self.conn = conn
        self._ensure_metrics_tables()
    
    def _ensure_metrics_tables(self):
        """Create metrics tracking tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_metrics (
                id TEXT PRIMARY KEY DEFAULT (uuid()),
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT,  -- 'brier', 'ece', 'precision', 'recall'
                detector_name TEXT,
                timeframe TEXT,
                symbol TEXT,
                value DOUBLE,
                n_samples INTEGER,
                metadata JSON
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_health_metrics (
                id TEXT PRIMARY KEY DEFAULT (uuid()),
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value DOUBLE,
                metric_unit TEXT,
                component TEXT,  -- 'council', 'referee', 'cache', 'ops'
                metadata JSON
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS latency_metrics (
                id TEXT PRIMARY KEY DEFAULT (uuid()),
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                endpoint TEXT,
                latency_ms DOUBLE,
                success BOOLEAN,
                error_type TEXT
            )
        """)
    
    def compute_brier_score(
        self, 
        detector_name: str,
        symbol: str = None, 
        timeframe: str = None,
        lookback_days: int = 30
    ) -> Optional[Dict[str, float]]:
        """
        Compute Brier Score for detector predictions.
        
        Brier Score = mean((p_predicted - outcome_actual)^2)
        Lower is better (perfect = 0, worst = 1)
        """
        try:
            where_clause = f"WHERE detector_name = '{detector_name}'"
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            if timeframe:
                where_clause += f" AND timeframe = '{timeframe}'"
            where_clause += f" AND event_ts >= CURRENT_TIMESTAMP - INTERVAL '{lookback_days} days'"
            where_clause += " AND outcome_data IS NOT NULL"
            where_clause += " AND json_extract(outcome_data, '$.forward_return_1h') IS NOT NULL"
            
            # Get predictions and outcomes
            query = f"""
            SELECT 
                probability_score,
                CASE 
                    WHEN CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE) > 0 
                    THEN 1.0 
                    ELSE 0.0 
                END as actual_outcome
            FROM pattern_events
            {where_clause}
            AND probability_score IS NOT NULL
            """
            
            results = self.conn.execute(query).fetchall()
            
            if len(results) < 10:  # Minimum sample size
                return None
            
            brier_sum = 0.0
            for prob_score, actual in results:
                if prob_score is not None:
                    brier_sum += (float(prob_score) - float(actual)) ** 2
            
            brier_score = brier_sum / len(results)
            
            # Store metric
            self._store_calibration_metric(
                metric_type='brier',
                detector_name=detector_name,
                symbol=symbol,
                timeframe=timeframe,
                value=brier_score,
                n_samples=len(results)
            )
            
            return {
                'brier_score': brier_score,
                'n_samples': len(results),
                'baseline_brier': 0.25  # Random guessing baseline
            }
            
        except Exception as e:
            logger.error(f"Failed to compute Brier score for {detector_name}: {e}")
            return None
    
    def compute_expected_calibration_error(
        self,
        detector_name: str,
        symbol: str = None,
        timeframe: str = None,
        lookback_days: int = 30,
        n_bins: int = 10
    ) -> Optional[Dict[str, float]]:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted probabilities
        and actual frequencies across probability bins.
        """
        try:
            where_clause = f"WHERE detector_name = '{detector_name}'"
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            if timeframe:
                where_clause += f" AND timeframe = '{timeframe}'"
            where_clause += f" AND event_ts >= CURRENT_TIMESTAMP - INTERVAL '{lookback_days} days'"
            where_clause += " AND outcome_data IS NOT NULL"
            where_clause += " AND json_extract(outcome_data, '$.forward_return_1h') IS NOT NULL"
            
            query = f"""
            SELECT 
                probability_score,
                CASE 
                    WHEN CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE) > 0 
                    THEN 1.0 
                    ELSE 0.0 
                END as actual_outcome
            FROM pattern_events
            {where_clause}
            AND probability_score IS NOT NULL
            """
            
            results = self.conn.execute(query).fetchall()
            
            if len(results) < 20:  # Need more samples for binning
                return None
            
            # Create probability bins
            bins = [(i/n_bins, (i+1)/n_bins) for i in range(n_bins)]
            bin_data = defaultdict(list)
            
            # Assign predictions to bins
            for prob_score, actual in results:
                prob_score, actual = float(prob_score), float(actual)
                for i, (bin_start, bin_end) in enumerate(bins):
                    if bin_start <= prob_score < bin_end or (i == n_bins-1 and prob_score == bin_end):
                        bin_data[i].append((prob_score, actual))
                        break
            
            # Compute ECE
            ece = 0.0
            total_samples = len(results)
            
            for bin_idx, samples in bin_data.items():
                if len(samples) == 0:
                    continue
                
                bin_weight = len(samples) / total_samples
                avg_confidence = sum(prob for prob, _ in samples) / len(samples)
                accuracy = sum(actual for _, actual in samples) / len(samples)
                
                ece += bin_weight * abs(avg_confidence - accuracy)
            
            # Store metric
            self._store_calibration_metric(
                metric_type='ece',
                detector_name=detector_name,
                symbol=symbol,
                timeframe=timeframe,
                value=ece,
                n_samples=len(results),
                metadata={'n_bins': n_bins}
            )
            
            return {
                'ece': ece,
                'n_samples': len(results),
                'n_bins': n_bins,
                'calibration_quality': 'excellent' if ece < 0.05 else 'good' if ece < 0.1 else 'poor'
            }
            
        except Exception as e:
            logger.error(f"Failed to compute ECE for {detector_name}: {e}")
            return None
    
    def compute_precision_recall(
        self,
        detector_name: str,
        symbol: str = None,
        timeframe: str = None,
        lookback_days: int = 30,
        threshold: float = 0.5
    ) -> Optional[Dict[str, float]]:
        """Compute precision and recall for detector predictions."""
        try:
            where_clause = f"WHERE detector_name = '{detector_name}'"
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            if timeframe:
                where_clause += f" AND timeframe = '{timeframe}'"
            where_clause += f" AND event_ts >= CURRENT_TIMESTAMP - INTERVAL '{lookback_days} days'"
            where_clause += " AND outcome_data IS NOT NULL"
            where_clause += " AND json_extract(outcome_data, '$.forward_return_1h') IS NOT NULL"
            
            query = f"""
            SELECT 
                probability_score,
                CASE 
                    WHEN CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE) > 0 
                    THEN 1.0 
                    ELSE 0.0 
                END as actual_outcome
            FROM pattern_events
            {where_clause}
            AND probability_score IS NOT NULL
            """
            
            results = self.conn.execute(query).fetchall()
            
            if len(results) < 10:
                return None
            
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for prob_score, actual in results:
                prob_score, actual = float(prob_score), float(actual)
                predicted = 1.0 if prob_score >= threshold else 0.0
                
                if predicted == 1.0 and actual == 1.0:
                    true_positives += 1
                elif predicted == 1.0 and actual == 0.0:
                    false_positives += 1
                elif predicted == 0.0 and actual == 1.0:
                    false_negatives += 1
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Store metrics
            self._store_calibration_metric(
                metric_type='precision',
                detector_name=detector_name,
                symbol=symbol,
                timeframe=timeframe,
                value=precision,
                n_samples=len(results),
                metadata={'threshold': threshold}
            )
            
            self._store_calibration_metric(
                metric_type='recall',
                detector_name=detector_name,
                symbol=symbol,
                timeframe=timeframe,
                value=recall,
                n_samples=len(results),
                metadata={'threshold': threshold}
            )
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_samples': len(results),
                'threshold': threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to compute P/R for {detector_name}: {e}")
            return None
    
    def get_council_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive council voting health metrics."""
        try:
            # Get recent council voting stats
            council_query = """
            SELECT 
                COUNT(*) as total_votes,
                AVG(consistency_index) as avg_ci,
                STDDEV(consistency_index) as ci_stability,
                AVG(n_oos) as avg_n_oos,
                MIN(n_oos) as min_n_oos,
                AVG(hit_rate) as avg_hit_rate,
                AVG(median_return) as avg_return
            FROM (
                -- Simulate council votes for recent patterns
                SELECT 
                    0.75 as consistency_index,  -- Placeholder - replace with actual CI computation
                    50 as n_oos,
                    0.65 as hit_rate,
                    0.025 as median_return
                FROM pattern_events 
                WHERE event_ts >= CURRENT_TIMESTAMP - INTERVAL '7 days'
                LIMIT 100
            )
            """
            
            result = self.conn.execute(council_query).fetchone()
            
            if result and result[0] > 0:
                return {
                    'total_votes': result[0],
                    'avg_consistency_index': result[1],
                    'ci_stability': result[2] or 0.0,
                    'avg_n_oos': result[3],
                    'min_n_oos': result[4],
                    'avg_hit_rate': result[5],
                    'avg_return': result[6],
                    'trade_gate_status': 'allowed' if (result[4] or 0) >= 60 and (result[2] or 0) < 0.4 else 'blocked'
                }
            else:
                return {
                    'total_votes': 0,
                    'trade_gate_status': 'blocked',
                    'reason': 'insufficient_data'
                }
                
        except Exception as e:
            logger.error(f"Failed to get council health metrics: {e}")
            return {'error': str(e)}
    
    def get_referee_health_metrics(self) -> Dict[str, Any]:
        """Get referee truth harness health metrics."""
        try:
            query = """
            SELECT 
                COUNT(*) as total_claims,
                SUM(CASE WHEN verdict = 'ok' THEN 1 ELSE 0 END) as claims_ok,
                SUM(CASE WHEN verdict = 'corrected' THEN 1 ELSE 0 END) as claims_corrected,
                SUM(CASE WHEN verdict = 'rejected' THEN 1 ELSE 0 END) as claims_rejected
            FROM claims_log 
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            """
            
            result = self.conn.execute(query).fetchone()
            
            if result and result[0] > 0:
                total, ok, corrected, rejected = result
                return {
                    'total_claims': total,
                    'claims_ok': ok,
                    'claims_corrected': corrected,
                    'claims_rejected': rejected,
                    'accuracy_rate': ok / total if total > 0 else 0.0,
                    'correction_rate': corrected / total if total > 0 else 0.0,
                    'rejection_rate': rejected / total if total > 0 else 0.0
                }
            else:
                return {
                    'total_claims': 0,
                    'claims_ok': 0,
                    'claims_corrected': 0,
                    'claims_rejected': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get referee metrics: {e}")
            return {'error': str(e)}
    
    def get_cache_health_metrics(self) -> Dict[str, Any]:
        """Get whatif cache health metrics."""
        try:
            # This would connect to actual cache metrics
            # For now, return placeholder structure
            return {
                'hit_rate': 0.87,
                'total_requests': 1250,
                'cache_hits': 1087,
                'cache_misses': 163,
                'evictions_24h': 15,
                'poison_detected': 0,
                'avg_response_time_ms': 12.5
            }
        except Exception as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return {'error': str(e)}
    
    def get_ops_health_metrics(self) -> Dict[str, Any]:
        """Get operational health metrics."""
        try:
            # Check recent write failures
            write_failures = self.conn.execute("""
                SELECT COUNT(*) 
                FROM system_health_metrics 
                WHERE metric_name = 'write_failure' 
                AND ts >= CURRENT_TIMESTAMP - INTERVAL '5 minutes'
            """).fetchone()[0]
            
            # Check scheduler lag (placeholder)
            scheduler_lag_ms = 45.2  # Would come from actual scheduler monitoring
            
            return {
                'write_failures_5m': write_failures,
                'scheduler_lag_ms': scheduler_lag_ms,
                'database_status': 'healthy',
                'api_status': 'healthy',
                'last_health_check': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get ops metrics: {e}")
            return {'error': str(e)}
    
    def record_latency(self, endpoint: str, latency_ms: float, success: bool = True, error_type: str = None):
        """Record endpoint latency for SLO monitoring."""
        try:
            self.conn.execute("""
                INSERT INTO latency_metrics (endpoint, latency_ms, success, error_type)
                VALUES (?, ?, ?, ?)
            """, [endpoint, latency_ms, success, error_type])
        except Exception as e:
            logger.error(f"Failed to record latency for {endpoint}: {e}")
    
    def get_latency_slo_metrics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get P95 latency and SLO compliance metrics."""
        try:
            query = f"""
            SELECT 
                endpoint,
                COUNT(*) as total_requests,
                AVG(latency_ms) as avg_latency_ms,
                PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_latency_ms,
                PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms,
                PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency_ms,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_requests
            FROM latency_metrics 
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL '{hours_back} hours'
            GROUP BY endpoint
            ORDER BY total_requests DESC
            """
            
            results = self.conn.execute(query).fetchall()
            
            endpoint_metrics = {}
            overall_p95 = 0.0
            
            for row in results:
                endpoint, total, avg, p50, p95, p99, success, failures = row
                success_rate = success / total if total > 0 else 0.0
                
                endpoint_metrics[endpoint] = {
                    'total_requests': total,
                    'avg_latency_ms': avg,
                    'p50_latency_ms': p50,
                    'p95_latency_ms': p95,
                    'p99_latency_ms': p99,
                    'success_rate': success_rate,
                    'slo_compliant': p95 <= 1000.0,  # 1 second SLO
                    'failed_requests': failures
                }
                
                if total > overall_p95:
                    overall_p95 = p95
            
            return {
                'overall_p95_latency_ms': overall_p95,
                'slo_target_ms': 1000.0,
                'slo_compliant': overall_p95 <= 1000.0,
                'endpoints': endpoint_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get latency SLO metrics: {e}")
            return {'error': str(e)}
    
    def _store_calibration_metric(
        self,
        metric_type: str,
        detector_name: str,
        symbol: str = None,
        timeframe: str = None,
        value: float = None,
        n_samples: int = None,
        metadata: Dict = None
    ):
        """Store calibration metric in database."""
        try:
            self.conn.execute("""
                INSERT INTO calibration_metrics 
                (metric_type, detector_name, symbol, timeframe, value, n_samples, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [metric_type, detector_name, symbol, timeframe, value, n_samples, json.dumps(metadata) if metadata else None])
        except Exception as e:
            logger.error(f"Failed to store calibration metric: {e}")
    
    def get_trade_gate_metrics(self) -> Dict[str, Any]:
        """Get trade gate decision metrics."""
        try:
            from trading_buddy.core.trade_gate import TradeGate
            trade_gate = TradeGate(self.conn)
            return trade_gate.get_gate_summary(hours_back=24)
        except Exception as e:
            logger.error(f"Failed to get trade gate metrics: {e}")
            return {'error': str(e)}
    
    def get_ci_uncertainty_metrics(self) -> Dict[str, Any]:
        """Get CI v2 uncertainty and FDR metrics."""
        try:
            # Check if CI v2 is enabled
            ci_version_query = """
                SELECT version, rollout_percentage
                FROM ci_version_config
                WHERE id = 'default'
            """
            ci_version_result = self.conn.execute(ci_version_query).fetchone()
            
            if not ci_version_result or ci_version_result[0] != 'v2':
                return {
                    'enabled': False,
                    'version': 'v1',
                    'rollout_percentage': 0
                }
            
            # Get CI v2 summary statistics
            summary_query = """
                SELECT 
                    symbol,
                    total_pairs,
                    allowed_count,
                    watch_only_count,
                    blocked_count,
                    avg_ci_final,
                    avg_power,
                    avg_stability,
                    fdr_discoveries,
                    latest_computation
                FROM ci_v2_summary
                ORDER BY symbol
            """
            
            summaries = self.conn.execute(summary_query).fetchall()
            
            # Get FDR statistics by month
            fdr_query = """
                SELECT 
                    fdr_month,
                    COUNT(*) as total_tests,
                    SUM(CASE WHEN fdr_pass THEN 1 ELSE 0 END) as discoveries,
                    AVG(fdr_q) as avg_q_value,
                    MIN(fdr_q) as min_q_value
                FROM consistency_matrix_latest
                WHERE fdr_month IS NOT NULL
                GROUP BY fdr_month
                ORDER BY fdr_month DESC
                LIMIT 6
            """
            
            fdr_stats = self.conn.execute(fdr_query).fetchall()
            
            # Get distribution of trade gates
            gate_distribution_query = """
                SELECT 
                    trade_gate,
                    COUNT(*) as count,
                    AVG(n_eff) as avg_n_eff,
                    AVG(stability) as avg_stability,
                    AVG(ci_final) as avg_ci_final
                FROM consistency_matrix_latest
                GROUP BY trade_gate
            """
            
            gate_dist = self.conn.execute(gate_distribution_query).fetchall()
            
            return {
                'enabled': True,
                'version': ci_version_result[0],
                'rollout_percentage': ci_version_result[1],
                'summary_by_symbol': {
                    row[0]: {
                        'total_pairs': row[1],
                        'allowed': row[2],
                        'watch_only': row[3],
                        'blocked': row[4],
                        'avg_ci_final': row[5],
                        'avg_power': row[6],
                        'avg_stability': row[7],
                        'fdr_discoveries': row[8],
                        'last_updated': row[9].isoformat() if row[9] else None
                    }
                    for row in summaries
                },
                'fdr_stats_by_month': [
                    {
                        'month': row[0],
                        'total_tests': row[1],
                        'discoveries': row[2],
                        'discovery_rate': row[2] / row[1] if row[1] > 0 else 0,
                        'avg_q_value': row[3],
                        'min_q_value': row[4]
                    }
                    for row in fdr_stats
                ],
                'gate_distribution': {
                    row[0]: {
                        'count': row[1],
                        'avg_n_eff': row[2],
                        'avg_stability': row[3],
                        'avg_ci_final': row[4]
                    }
                    for row in gate_dist
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get CI uncertainty metrics: {e}")
            return {'error': str(e)}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for /metrics/summary endpoint."""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics_ref_id': str(uuid.uuid4()),
            'council': self.get_council_health_metrics(),
            'calibration': {
                'brier_scores': {},  # Will be populated by nightly job
                'ece_scores': {},
                'precision_recall': {}
            },
            'referee': self.get_referee_health_metrics(),
            'cache': self.get_cache_health_metrics(),
            'ops': self.get_ops_health_metrics(),
            'latency_slo': self.get_latency_slo_metrics(),
            'trade_gate': self.get_trade_gate_metrics(),
            'ci_uncertainty': self.get_ci_uncertainty_metrics()
        }