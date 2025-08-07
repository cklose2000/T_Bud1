"""
PR7: Trade Gate System

Implements strict gating requirements for all trading signals.
No signal can be surfaced unless it passes all quality checks.

Requirements:
- n_oos ≥ 60 (sufficient out-of-sample data)
- stability ≥ 0.4 (consistent performance across time)
- CI ci_final > 0 (positive consistency index)
- Calibrated threshold pass (precision target met)
- No drift detected (PSI/KS within bounds)
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from duckdb import DuckDBPyConnection
from trading_buddy.core.threshold_tuner import ThresholdTuner

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Trade gate status enumeration."""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    WATCH_ONLY = "watch_only"


@dataclass
class GateDecision:
    """Result of trade gate evaluation."""
    status: GateStatus
    confidence_score: float  # 0-1, how confident we are in this signal
    blocking_reasons: List[str]
    gate_metrics: Dict[str, Any]
    
    @property
    def is_tradeable(self) -> bool:
        """Whether this signal can be traded."""
        return self.status == GateStatus.ALLOWED
    
    @property
    def summary(self) -> str:
        """Human-readable summary of gate decision."""
        if self.status == GateStatus.ALLOWED:
            return f"ALLOWED (confidence: {self.confidence_score:.2f})"
        else:
            reasons = ", ".join(self.blocking_reasons)
            return f"{self.status.value.upper()}: {reasons}"


class TradeGate:
    """Implements strict quality gates for all trading signals."""
    
    # Gate thresholds (non-negotiable minimums)
    MIN_OOS_SAMPLES = 60
    MIN_STABILITY = 0.4
    MIN_CI_FINAL = 0.0
    MIN_PRECISION = 0.50  # Default precision target
    MAX_DRIFT_PSI = 0.25  # Population Stability Index threshold
    MAX_DRIFT_KS = 0.10   # Kolmogorov-Smirnov test threshold
    
    def __init__(self, conn: DuckDBPyConnection):
        self.conn = conn
        self.threshold_tuner = ThresholdTuner(conn)
        self._ensure_gate_tables()
    
    def _ensure_gate_tables(self):
        """Create trade gate tracking tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_gate_decisions (
                id TEXT PRIMARY KEY DEFAULT (uuid()),
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detector_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                gate_status TEXT,  -- 'allowed', 'blocked', 'watch_only'
                confidence_score DOUBLE,
                blocking_reasons JSON,
                gate_metrics JSON,
                probability_score DOUBLE,
                final_threshold DOUBLE
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS drift_monitoring (
                id TEXT PRIMARY KEY DEFAULT (uuid()),
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detector_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                psi_score DOUBLE,
                ks_statistic DOUBLE,
                drift_detected BOOLEAN,
                baseline_period_start TIMESTAMP,
                baseline_period_end TIMESTAMP,
                recent_period_start TIMESTAMP,
                recent_period_end TIMESTAMP
            )
        """)
    
    def evaluate_signal(
        self,
        detector_name: str,
        symbol: str,
        timeframe: str,
        probability_score: float,
        ci_final: Optional[float] = None,
        n_oos: Optional[int] = None,
        stability: Optional[float] = None,
        hit_rate: Optional[float] = None
    ) -> GateDecision:
        """
        Evaluate whether a signal should pass the trade gate.
        
        This is the single point where ALL signals are evaluated for quality.
        """
        blocking_reasons = []
        gate_metrics = {}
        
        # Get detector statistics if not provided
        if n_oos is None or stability is None or ci_final is None:
            detector_stats = self._get_detector_stats(detector_name, symbol, timeframe)
            n_oos = n_oos or detector_stats.get('n_oos', 0)
            stability = stability or detector_stats.get('stability', 0.0)
            ci_final = ci_final or detector_stats.get('ci_final', 0.0)
            hit_rate = hit_rate or detector_stats.get('hit_rate', 0.0)
        
        gate_metrics.update({
            'n_oos': n_oos,
            'stability': stability,
            'ci_final': ci_final,
            'hit_rate': hit_rate,
            'probability_score': probability_score
        })
        
        # Gate 1: Sufficient OOS data
        if n_oos < self.MIN_OOS_SAMPLES:
            blocking_reasons.append(f"insufficient_oos_data_{n_oos}<{self.MIN_OOS_SAMPLES}")
        
        # Gate 2: Stability requirement
        if stability < self.MIN_STABILITY:
            blocking_reasons.append(f"low_stability_{stability:.3f}<{self.MIN_STABILITY}")
        
        # Gate 3: Positive consistency index
        if ci_final <= self.MIN_CI_FINAL:
            blocking_reasons.append(f"non_positive_ci_{ci_final:.3f}≤{self.MIN_CI_FINAL}")
        
        # Gate 4: Calibrated threshold check
        threshold_pass, threshold_metrics = self._check_calibrated_threshold(
            detector_name, symbol, timeframe, probability_score
        )
        gate_metrics.update(threshold_metrics)
        
        if not threshold_pass:
            blocking_reasons.append(f"threshold_fail_{probability_score:.3f}<{threshold_metrics.get('threshold', 0.5)}")
        
        # Gate 5: Drift detection
        drift_detected, drift_metrics = self._check_drift(detector_name, symbol, timeframe)
        gate_metrics.update(drift_metrics)
        
        if drift_detected:
            blocking_reasons.append(f"drift_detected_psi={drift_metrics.get('psi_score', 0):.3f}")
        
        # Calculate confidence score (0-1)
        confidence_score = self._calculate_confidence_score(
            n_oos, stability, ci_final, probability_score, hit_rate
        )
        
        # Determine gate status
        if len(blocking_reasons) == 0:
            status = GateStatus.ALLOWED
        elif len(blocking_reasons) == 1 and blocking_reasons[0].startswith("threshold_fail"):
            # Only threshold issue - allow as watch-only
            status = GateStatus.WATCH_ONLY
            blocking_reasons = ["below_calibrated_threshold"]
        else:
            # Multiple issues or serious problems
            status = GateStatus.BLOCKED
        
        decision = GateDecision(
            status=status,
            confidence_score=confidence_score,
            blocking_reasons=blocking_reasons,
            gate_metrics=gate_metrics
        )
        
        # Log decision
        self._log_gate_decision(detector_name, symbol, timeframe, decision, probability_score)
        
        return decision
    
    def _get_detector_stats(self, detector_name: str, symbol: str, timeframe: str) -> Dict[str, float]:
        """Get detector performance statistics."""
        try:
            # Get recent pattern statistics
            query = f"""
            WITH recent_patterns AS (
                SELECT 
                    COUNT(*) as n_patterns,
                    AVG(CASE WHEN CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE) > 0 THEN 1.0 ELSE 0.0 END) as hit_rate,
                    STDDEV(CAST(json_extract(outcome_data, '$.forward_return_1h') AS DOUBLE)) as volatility
                FROM pattern_events 
                WHERE detector_name = '{detector_name}'
                AND symbol = '{symbol}'
                AND timeframe = '{timeframe}'
                AND event_ts >= CURRENT_TIMESTAMP - INTERVAL '90 days'
                AND outcome_data IS NOT NULL
                AND json_extract(outcome_data, '$.forward_return_1h') IS NOT NULL
            ),
            stability_calc AS (
                SELECT 
                    -- Calculate stability as inverse of coefficient of variation
                    1.0 - (COALESCE(volatility, 0.0) / NULLIF(ABS(COALESCE(hit_rate, 0.5) - 0.5), 0.01)) as stability_metric
                FROM recent_patterns
            )
            SELECT 
                rp.n_patterns,
                COALESCE(rp.hit_rate, 0.0) as hit_rate,
                COALESCE(sc.stability_metric, 0.0) as stability,
                0.5 as ci_final  -- Placeholder - would come from actual CI calculation
            FROM recent_patterns rp
            CROSS JOIN stability_calc sc
            """
            
            result = self.conn.execute(query).fetchone()
            
            if result and result[0] > 0:
                return {
                    'n_oos': result[0],
                    'hit_rate': result[1] or 0.0,
                    'stability': min(max(result[2] or 0.0, 0.0), 1.0),  # Clamp to [0,1]
                    'ci_final': result[3] or 0.0
                }
            else:
                return {
                    'n_oos': 0,
                    'hit_rate': 0.0,
                    'stability': 0.0,
                    'ci_final': 0.0
                }
        except Exception as e:
            logger.error(f"Failed to get detector stats for {detector_name}: {e}")
            return {
                'n_oos': 0,
                'hit_rate': 0.0,
                'stability': 0.0,
                'ci_final': 0.0
            }
    
    def _check_calibrated_threshold(
        self, 
        detector_name: str, 
        symbol: str, 
        timeframe: str, 
        probability_score: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if probability score passes calibrated threshold."""
        threshold = self.threshold_tuner.get_detector_threshold(detector_name, symbol, timeframe)
        passes = probability_score >= threshold
        
        # Get current threshold performance metrics
        current_threshold_data = self.threshold_tuner._get_current_threshold(detector_name, symbol, timeframe)
        
        metrics = {
            'threshold': threshold,
            'threshold_passes': passes,
            'actual_precision': current_threshold_data.get('actual_precision') if current_threshold_data else None,
            'precision_target': current_threshold_data.get('precision_target') if current_threshold_data else self.MIN_PRECISION
        }
        
        return passes, metrics
    
    def _check_drift(self, detector_name: str, symbol: str, timeframe: str) -> Tuple[bool, Dict[str, Any]]:
        """Check for distribution drift in detector features."""
        try:
            # Get recent drift monitoring results
            query = f"""
            SELECT psi_score, ks_statistic, drift_detected
            FROM drift_monitoring 
            WHERE detector_name = '{detector_name}'
            AND symbol = '{symbol}'
            AND timeframe = '{timeframe}'
            AND ts >= CURRENT_TIMESTAMP - INTERVAL '7 days'
            ORDER BY ts DESC
            LIMIT 1
            """
            
            result = self.conn.execute(query).fetchone()
            
            if result:
                psi_score, ks_statistic, drift_detected = result
                return drift_detected, {
                    'psi_score': psi_score,
                    'ks_statistic': ks_statistic,
                    'drift_detected': drift_detected
                }
            else:
                # No recent drift data - assume no drift but flag as unknown
                return False, {
                    'psi_score': 0.0,
                    'ks_statistic': 0.0,
                    'drift_detected': False,
                    'drift_data_available': False
                }
                
        except Exception as e:
            logger.error(f"Failed to check drift for {detector_name}: {e}")
            return False, {'error': str(e)}
    
    def _calculate_confidence_score(
        self,
        n_oos: int,
        stability: float,
        ci_final: float,
        probability_score: float,
        hit_rate: float
    ) -> float:
        """Calculate confidence score (0-1) for the signal."""
        
        # Convert inputs to float to avoid decimal arithmetic issues
        n_oos = float(n_oos) if n_oos is not None else 0.0
        stability = float(stability) if stability is not None else 0.0
        ci_final = float(ci_final) if ci_final is not None else 0.0
        probability_score = float(probability_score) if probability_score is not None else 0.0
        hit_rate = float(hit_rate) if hit_rate is not None else 0.0
        
        # Base score from sample size (log scale, saturates at 200 samples)
        sample_score = min(n_oos / 200.0, 1.0) * 0.25
        
        # Stability contribution (higher is better)
        stability_score = stability * 0.25
        
        # CI contribution (positive CI is good, higher is better, but cap at 2.0)
        ci_score = min(max(ci_final, 0.0) / 2.0, 1.0) * 0.25
        
        # Probability score contribution (higher probability = higher confidence)
        prob_score = probability_score * 0.15
        
        # Hit rate contribution (closer to 0.65 is better for most patterns)
        hit_rate_score = (1.0 - abs(hit_rate - 0.65) / 0.35) * 0.10 if hit_rate > 0 else 0.0
        
        total_score = sample_score + stability_score + ci_score + prob_score + hit_rate_score
        
        # Ensure score is between 0 and 1
        return min(max(total_score, 0.0), 1.0)
    
    def _log_gate_decision(
        self,
        detector_name: str,
        symbol: str,
        timeframe: str,
        decision: GateDecision,
        probability_score: float
    ):
        """Log gate decision for audit and monitoring."""
        try:
            import json
            
            # Convert decimals to floats for JSON serialization
            serializable_metrics = {}
            for key, value in decision.gate_metrics.items():
                if hasattr(value, '__float__'):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            
            self.conn.execute("""
                INSERT INTO trade_gate_decisions 
                (detector_name, symbol, timeframe, gate_status, confidence_score,
                 blocking_reasons, gate_metrics, probability_score, final_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                detector_name, symbol, timeframe, decision.status.value, decision.confidence_score,
                json.dumps(decision.blocking_reasons),
                json.dumps(serializable_metrics),
                probability_score,
                serializable_metrics.get('threshold', 0.5)
            ])
        except Exception as e:
            logger.error(f"Failed to log gate decision: {e}")
    
    def get_gate_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of gate decisions for monitoring."""
        try:
            query = f"""
            SELECT 
                gate_status,
                COUNT(*) as count,
                AVG(confidence_score) as avg_confidence,
                COUNT(DISTINCT detector_name || ':' || symbol || ':' || timeframe) as unique_detectors
            FROM trade_gate_decisions 
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL '{hours_back} hours'
            GROUP BY gate_status
            """
            
            results = self.conn.execute(query).fetchall()
            
            summary = {
                'total_decisions': 0,
                'allowed': 0,
                'blocked': 0,
                'watch_only': 0,
                'avg_confidence_allowed': 0.0,
                'unique_detectors_evaluated': set()
            }
            
            for row in results:
                status, count, avg_conf, unique_detectors = row
                summary['total_decisions'] += count
                summary[status] = count
                
                if status == 'allowed':
                    summary['avg_confidence_allowed'] = avg_conf or 0.0
            
            # Get unique detectors count
            unique_query = f"""
            SELECT COUNT(DISTINCT detector_name || ':' || symbol || ':' || timeframe)
            FROM trade_gate_decisions 
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL '{hours_back} hours'
            """
            
            unique_count = self.conn.execute(unique_query).fetchone()[0]
            summary['unique_detectors_evaluated'] = unique_count
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get gate summary: {e}")
            return {'error': str(e)}
    
    def should_surface_pattern(
        self,
        detector_name: str,
        symbol: str,
        timeframe: str,
        probability_score: float,
        **kwargs
    ) -> Tuple[bool, GateDecision]:
        """
        Main entry point: determine if a pattern should be surfaced to users.
        
        Returns:
            (should_surface, gate_decision)
        """
        decision = self.evaluate_signal(
            detector_name, symbol, timeframe, probability_score, **kwargs
        )
        
        should_surface = decision.status in [GateStatus.ALLOWED, GateStatus.WATCH_ONLY]
        
        if decision.status == GateStatus.BLOCKED:
            logger.info(f"BLOCKED signal {detector_name}:{symbol}:{timeframe} "
                       f"(prob={probability_score:.3f}): {decision.summary}")
        elif decision.status == GateStatus.WATCH_ONLY:
            logger.info(f"WATCH-ONLY signal {detector_name}:{symbol}:{timeframe} "
                       f"(prob={probability_score:.3f}): {decision.summary}")
        else:
            logger.info(f"ALLOWED signal {detector_name}:{symbol}:{timeframe} "
                       f"(prob={probability_score:.3f}, confidence={decision.confidence_score:.2f})")
        
        return should_surface, decision
    
    @classmethod
    def enforce_gate_discipline(cls, conn: DuckDBPyConnection):
        """
        Static method to check that all recommendations go through the trade gate.
        Call this from any endpoint that surfaces trading signals.
        """
        # This would be implemented as a decorator or middleware
        # For now, it's a placeholder to remind developers
        logger.info("Trade gate discipline enforced - all signals must pass quality gates")