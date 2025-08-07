"""
PR7: Automated Threshold Tuning

Automatically tunes probability thresholds for each detector based on OOS performance.
Ensures we only surface signals that meet calibrated precision/recall targets.
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from duckdb import DuckDBPyConnection
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdCandidate:
    """A candidate threshold with its performance metrics."""
    threshold: float
    precision: float
    recall: float
    f1_score: float
    n_positive_predictions: int
    n_total_samples: int
    
    @property
    def prediction_rate(self) -> float:
        """Fraction of samples predicted as positive."""
        return self.n_positive_predictions / self.n_total_samples if self.n_total_samples > 0 else 0.0


class ThresholdTuner:
    """Tunes probability thresholds for detectors based on OOS performance."""
    
    def __init__(self, conn: DuckDBPyConnection):
        self.conn = conn
        self._ensure_threshold_tables()
    
    def _ensure_threshold_tables(self):
        """Create threshold tuning tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detector_thresholds (
                id TEXT PRIMARY KEY DEFAULT (uuid()),
                detector_name TEXT NOT NULL,
                symbol TEXT,
                timeframe TEXT,
                threshold_value DOUBLE NOT NULL,
                precision_target DOUBLE NOT NULL,
                recall_target DOUBLE NOT NULL,
                actual_precision DOUBLE,
                actual_recall DOUBLE,
                f1_score DOUBLE,
                n_oos_samples INTEGER,
                tuned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',  -- 'active', 'deprecated', 'testing'
                metadata JSON,
                
                UNIQUE (detector_name, symbol, timeframe, status)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS threshold_tuning_log (
                id TEXT PRIMARY KEY DEFAULT (uuid()),
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detector_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                old_threshold DOUBLE,
                new_threshold DOUBLE,
                precision_improvement DOUBLE,
                recall_change DOUBLE,
                reason TEXT,
                n_samples INTEGER
            )
        """)
    
    def find_optimal_threshold(
        self,
        detector_name: str,
        symbol: str = None,
        timeframe: str = None,
        precision_target: float = 0.60,
        recall_target: float = 0.50,
        lookback_days: int = 30,
        min_samples: int = 20
    ) -> Optional[ThresholdCandidate]:
        """
        Find optimal threshold for a detector based on OOS performance.
        
        Uses grid search across probability thresholds to find the best
        precision/recall trade-off that meets our targets.
        """
        try:
            # Get OOS data for this detector
            where_clause = f"WHERE detector_name = '{detector_name}'"
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            if timeframe:
                where_clause += f" AND timeframe = '{timeframe}'"
            where_clause += f" AND event_ts >= CURRENT_TIMESTAMP - INTERVAL '{lookback_days} days'"
            where_clause += " AND outcome_data IS NOT NULL"
            where_clause += " AND json_extract(outcome_data, '$.forward_return_1h') IS NOT NULL"
            where_clause += " AND probability_score IS NOT NULL"
            
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
            ORDER BY event_ts
            """
            
            results = self.conn.execute(query).fetchall()
            
            if len(results) < min_samples:
                logger.warning(f"Insufficient samples for {detector_name}:{symbol}:{timeframe} ({len(results)} < {min_samples})")
                return None
            
            # Convert to numpy arrays for efficient computation
            probabilities = np.array([float(row[0]) for row in results])
            outcomes = np.array([float(row[1]) for row in results])
            
            # Grid search over threshold candidates
            threshold_candidates = np.arange(0.1, 1.0, 0.05)  # 0.10, 0.15, ..., 0.95
            best_candidate = None
            best_score = -1.0
            
            candidates = []
            
            for threshold in threshold_candidates:
                predictions = (probabilities >= threshold).astype(float)
                
                # Calculate confusion matrix components
                true_positives = np.sum((predictions == 1) & (outcomes == 1))
                false_positives = np.sum((predictions == 1) & (outcomes == 0))
                false_negatives = np.sum((predictions == 0) & (outcomes == 1))
                
                # Calculate metrics
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                n_positive_predictions = int(np.sum(predictions))
                
                candidate = ThresholdCandidate(
                    threshold=threshold,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    n_positive_predictions=n_positive_predictions,
                    n_total_samples=len(results)
                )
                
                candidates.append(candidate)
                
                # Score candidate based on how well it meets our targets
                score = self._score_threshold_candidate(
                    candidate, precision_target, recall_target
                )
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            # Log all candidates for debugging
            logger.debug(f"Threshold candidates for {detector_name}:")
            for c in sorted(candidates, key=lambda x: x.threshold):
                logger.debug(f"  {c.threshold:.2f}: P={c.precision:.3f}, R={c.recall:.3f}, F1={c.f1_score:.3f}, predictions={c.n_positive_predictions}")
            
            if best_candidate:
                logger.info(f"Optimal threshold for {detector_name}:{symbol}:{timeframe}: "
                          f"{best_candidate.threshold:.2f} (P={best_candidate.precision:.3f}, "
                          f"R={best_candidate.recall:.3f}, F1={best_candidate.f1_score:.3f})")
                
                # Check if threshold meets minimum quality standards
                if best_candidate.precision < 0.40 or best_candidate.recall < 0.25:
                    logger.warning(f"Detector {detector_name} has poor performance even with optimal threshold - consider disabling")
            
            return best_candidate
            
        except Exception as e:
            logger.error(f"Failed to find optimal threshold for {detector_name}: {e}")
            return None
    
    def _score_threshold_candidate(
        self, 
        candidate: ThresholdCandidate, 
        precision_target: float, 
        recall_target: float
    ) -> float:
        """
        Score a threshold candidate based on how well it meets our targets.
        
        Priority:
        1. Meet minimum precision target (precision >= precision_target)
        2. Maximize F1 score among candidates that meet precision target
        3. Penalize candidates with very low prediction rates (< 0.05)
        """
        score = 0.0
        
        # Penalty for not meeting precision target
        if candidate.precision < precision_target:
            score -= 100.0 * (precision_target - candidate.precision)
        
        # Bonus for meeting precision target
        if candidate.precision >= precision_target:
            score += 50.0
        
        # F1 score contribution (0-100)
        score += 100.0 * candidate.f1_score
        
        # Penalty for very low prediction rates (unusable thresholds)
        if candidate.prediction_rate < 0.05:
            score -= 25.0
        
        # Bonus for meeting both precision and recall targets
        if candidate.precision >= precision_target and candidate.recall >= recall_target:
            score += 25.0
        
        return score
    
    def update_detector_threshold(
        self,
        detector_name: str,
        symbol: str = None,
        timeframe: str = None,
        precision_target: float = 0.60,
        recall_target: float = 0.50,
        force_update: bool = False
    ) -> bool:
        """Update threshold for a detector if improvement is significant."""
        try:
            # Get current threshold
            current_threshold = self._get_current_threshold(detector_name, symbol, timeframe)
            
            # Find optimal threshold
            optimal_candidate = self.find_optimal_threshold(
                detector_name, symbol, timeframe, precision_target, recall_target
            )
            
            if not optimal_candidate:
                logger.warning(f"Could not find optimal threshold for {detector_name}")
                return False
            
            # Check if update is warranted
            should_update = force_update
            reason = "forced_update" if force_update else None
            
            if not should_update and current_threshold:
                # Compare performance
                current_perf = self._evaluate_threshold_performance(
                    detector_name, symbol, timeframe, current_threshold['threshold_value']
                )
                
                if current_perf:
                    precision_improvement = optimal_candidate.precision - current_perf.get('precision', 0.0)
                    recall_change = optimal_candidate.recall - current_perf.get('recall', 0.0)
                    
                    # Update if precision improves significantly or if current precision is below target
                    if (precision_improvement > 0.05 or  # 5% precision improvement
                        current_perf.get('precision', 0.0) < precision_target or  # Below target
                        abs(recall_change) > 0.10):  # Significant recall change
                        should_update = True
                        reason = f"precision_improvement_{precision_improvement:.3f}"
            else:
                # No current threshold - set initial one
                should_update = True
                reason = "initial_threshold"
            
            if should_update:
                # Store new threshold
                self._store_threshold(
                    detector_name, symbol, timeframe,
                    threshold_value=optimal_candidate.threshold,
                    precision_target=precision_target,
                    recall_target=recall_target,
                    actual_precision=optimal_candidate.precision,
                    actual_recall=optimal_candidate.recall,
                    f1_score=optimal_candidate.f1_score,
                    n_oos_samples=optimal_candidate.n_total_samples
                )
                
                # Log the update
                old_threshold = current_threshold['threshold_value'] if current_threshold else None
                self._log_threshold_change(
                    detector_name, symbol, timeframe,
                    old_threshold=old_threshold,
                    new_threshold=optimal_candidate.threshold,
                    precision_improvement=optimal_candidate.precision - (current_threshold.get('actual_precision', 0.0) if current_threshold else 0.0),
                    recall_change=optimal_candidate.recall - (current_threshold.get('actual_recall', 0.0) if current_threshold else 0.0),
                    reason=reason,
                    n_samples=optimal_candidate.n_total_samples
                )
                
                logger.info(f"Updated threshold for {detector_name}:{symbol}:{timeframe}: "
                          f"{old_threshold} → {optimal_candidate.threshold:.2f} ({reason})")
                return True
            else:
                logger.info(f"No threshold update needed for {detector_name}:{symbol}:{timeframe}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update threshold for {detector_name}: {e}")
            return False
    
    def _get_current_threshold(self, detector_name: str, symbol: str = None, timeframe: str = None) -> Optional[Dict]:
        """Get current active threshold for detector."""
        try:
            where_clause = f"WHERE detector_name = '{detector_name}' AND status = 'active'"
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            if timeframe:
                where_clause += f" AND timeframe = '{timeframe}'"
            
            result = self.conn.execute(f"""
                SELECT threshold_value, precision_target, recall_target, 
                       actual_precision, actual_recall, f1_score, n_oos_samples
                FROM detector_thresholds 
                {where_clause}
                ORDER BY tuned_at DESC
                LIMIT 1
            """).fetchone()
            
            if result:
                return {
                    'threshold_value': result[0],
                    'precision_target': result[1],
                    'recall_target': result[2],
                    'actual_precision': result[3],
                    'actual_recall': result[4],
                    'f1_score': result[5],
                    'n_oos_samples': result[6]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get current threshold: {e}")
            return None
    
    def _evaluate_threshold_performance(
        self, 
        detector_name: str, 
        symbol: str, 
        timeframe: str, 
        threshold: float
    ) -> Optional[Dict]:
        """Evaluate performance of a specific threshold."""
        try:
            where_clause = f"WHERE detector_name = '{detector_name}'"
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            if timeframe:
                where_clause += f" AND timeframe = '{timeframe}'"
            where_clause += " AND event_ts >= CURRENT_TIMESTAMP - INTERVAL '30 days'"
            where_clause += " AND outcome_data IS NOT NULL"
            where_clause += " AND probability_score IS NOT NULL"
            
            query = f"""
            SELECT 
                AVG(CASE WHEN probability_score >= {threshold} THEN 1.0 ELSE 0.0 END) as prediction_rate,
                AVG(CASE 
                    WHEN probability_score >= {threshold} AND json_extract(outcome_data, '$.forward_return_1h') > 0
                    THEN 1.0 ELSE 0.0 
                END) / NULLIF(AVG(CASE WHEN probability_score >= {threshold} THEN 1.0 ELSE 0.0 END), 0) as precision,
                AVG(CASE 
                    WHEN probability_score >= {threshold} AND json_extract(outcome_data, '$.forward_return_1h') > 0
                    THEN 1.0 ELSE 0.0 
                END) / NULLIF(AVG(CASE WHEN json_extract(outcome_data, '$.forward_return_1h') > 0 THEN 1.0 ELSE 0.0 END), 0) as recall,
                COUNT(*) as n_samples
            FROM pattern_events
            {where_clause}
            """
            
            result = self.conn.execute(query).fetchone()
            
            if result and result[3] > 0:
                precision = result[1] or 0.0
                recall = result[2] or 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                return {
                    'prediction_rate': result[0] or 0.0,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'n_samples': result[3]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to evaluate threshold performance: {e}")
            return None
    
    def _store_threshold(
        self,
        detector_name: str,
        symbol: str,
        timeframe: str,
        threshold_value: float,
        precision_target: float,
        recall_target: float,
        actual_precision: float,
        actual_recall: float,
        f1_score: float,
        n_oos_samples: int
    ):
        """Store new threshold, deactivating old one."""
        try:
            # Deactivate old threshold
            where_clause = f"detector_name = '{detector_name}' AND status = 'active'"
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            if timeframe:
                where_clause += f" AND timeframe = '{timeframe}'"
            
            self.conn.execute(f"""
                UPDATE detector_thresholds 
                SET status = 'deprecated'
                WHERE {where_clause}
            """)
            
            # Insert new threshold
            self.conn.execute("""
                INSERT INTO detector_thresholds 
                (detector_name, symbol, timeframe, threshold_value, precision_target, recall_target,
                 actual_precision, actual_recall, f1_score, n_oos_samples, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """, [detector_name, symbol, timeframe, threshold_value, precision_target, recall_target,
                  actual_precision, actual_recall, f1_score, n_oos_samples])
            
        except Exception as e:
            logger.error(f"Failed to store threshold: {e}")
    
    def _log_threshold_change(
        self,
        detector_name: str,
        symbol: str,
        timeframe: str,
        old_threshold: float,
        new_threshold: float,
        precision_improvement: float,
        recall_change: float,
        reason: str,
        n_samples: int
    ):
        """Log threshold change for audit trail."""
        try:
            self.conn.execute("""
                INSERT INTO threshold_tuning_log
                (detector_name, symbol, timeframe, old_threshold, new_threshold,
                 precision_improvement, recall_change, reason, n_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [detector_name, symbol, timeframe, old_threshold, new_threshold,
                  precision_improvement, recall_change, reason, n_samples])
        except Exception as e:
            logger.error(f"Failed to log threshold change: {e}")
    
    def get_detector_threshold(self, detector_name: str, symbol: str = None, timeframe: str = None) -> float:
        """Get current threshold for detector, defaulting to 0.5 if none set."""
        current = self._get_current_threshold(detector_name, symbol, timeframe)
        return current['threshold_value'] if current else 0.5
    
    def should_surface_signal(
        self, 
        detector_name: str, 
        probability_score: float, 
        symbol: str = None, 
        timeframe: str = None
    ) -> bool:
        """Determine if a signal should be surfaced based on tuned threshold."""
        if probability_score is None:
            return False
        
        threshold = self.get_detector_threshold(detector_name, symbol, timeframe)
        return probability_score >= threshold
    
    def tune_all_detectors(self, min_samples: int = 20) -> Dict[str, Any]:
        """Tune thresholds for all detectors with sufficient data."""
        logger.info("Starting threshold tuning for all detectors")
        
        # Get active detectors
        detectors = self._get_active_detectors(min_samples)
        
        results = {
            'detectors_processed': 0,
            'thresholds_updated': 0,
            'failed': 0,
            'insufficient_data': 0
        }
        
        for detector_config in detectors:
            detector_name = detector_config['detector_name']
            symbol = detector_config['symbol']
            timeframe = detector_config['timeframe']
            
            try:
                updated = self.update_detector_threshold(
                    detector_name, symbol, timeframe
                )
                
                results['detectors_processed'] += 1
                if updated:
                    results['thresholds_updated'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to tune {detector_name}:{symbol}:{timeframe}: {e}")
                results['failed'] += 1
        
        logger.info(f"Threshold tuning complete: {results}")
        return results
    
    def _get_active_detectors(self, min_samples: int) -> List[Dict[str, str]]:
        """Get list of detectors with sufficient data for threshold tuning."""
        try:
            query = """
            SELECT DISTINCT 
                detector_name,
                symbol,
                timeframe,
                COUNT(*) as pattern_count
            FROM pattern_events 
            WHERE event_ts >= CURRENT_TIMESTAMP - INTERVAL '30 days'
            AND outcome_data IS NOT NULL
            AND probability_score IS NOT NULL
            GROUP BY detector_name, symbol, timeframe
            HAVING COUNT(*) >= ?
            ORDER BY pattern_count DESC
            """
            
            results = self.conn.execute(query, [min_samples]).fetchall()
            
            return [
                {
                    'detector_name': row[0],
                    'symbol': row[1],
                    'timeframe': row[2],
                    'pattern_count': row[3]
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Failed to get active detectors: {e}")
            return []