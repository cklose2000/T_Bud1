"""
PR7: Metrics Summary API Endpoint

Single source of truth for all system health and calibration metrics.
"""
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.core.metrics import MetricsComputer

router = APIRouter()


class MetricsSummaryResponse(BaseModel):
    """Response model for metrics summary."""
    timestamp: str
    metrics_ref_id: str
    council: Dict[str, Any]
    calibration: Dict[str, Any]
    referee: Dict[str, Any] 
    cache: Dict[str, Any]
    ops: Dict[str, Any]
    latency_slo: Dict[str, Any]
    trade_gate: Dict[str, Any]
    ci_uncertainty: Dict[str, Any]


@router.get("/summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary():
    """
    Get comprehensive system health and calibration metrics.
    
    This is the single source of truth for:
    - Council voting performance and stability
    - Calibration metrics (Brier score, ECE, Precision/Recall)
    - Referee truth harness compliance
    - Cache performance and poisoning detection
    - Operational health (write failures, scheduler lag)
    - Latency SLO compliance
    """
    start_time = time.time()
    
    try:
        with DuckDBManager() as db:
            metrics_computer = MetricsComputer(db.conn)
            summary = metrics_computer.get_metrics_summary()
            
            # Record our own latency
            latency_ms = (time.time() - start_time) * 1000
            metrics_computer.record_latency("/metrics/summary", latency_ms, success=True)
            
            return MetricsSummaryResponse(**summary)
            
    except Exception as e:
        # Record failure latency
        latency_ms = (time.time() - start_time) * 1000
        try:
            with DuckDBManager() as db:
                metrics_computer = MetricsComputer(db.conn)
                metrics_computer.record_latency(
                    "/metrics/summary", 
                    latency_ms, 
                    success=False, 
                    error_type=type(e).__name__
                )
        except:
            pass  # Don't fail on metric recording failure
            
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics summary: {str(e)}")


@router.get("/calibration/{detector_name}")
async def get_detector_calibration(
    detector_name: str,
    symbol: str = None,
    timeframe: str = None,
    lookback_days: int = 30
):
    """
    Get detailed calibration metrics for a specific detector.
    
    Returns Brier score, ECE, and precision/recall for the detector.
    """
    start_time = time.time()
    
    try:
        with DuckDBManager() as db:
            metrics_computer = MetricsComputer(db.conn)
            
            brier_result = metrics_computer.compute_brier_score(
                detector_name, symbol, timeframe, lookback_days
            )
            
            ece_result = metrics_computer.compute_expected_calibration_error(
                detector_name, symbol, timeframe, lookback_days
            )
            
            pr_result = metrics_computer.compute_precision_recall(
                detector_name, symbol, timeframe, lookback_days
            )
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            metrics_computer.record_latency(f"/metrics/calibration/{detector_name}", latency_ms)
            
            return {
                'detector_name': detector_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'lookback_days': lookback_days,
                'brier_score': brier_result,
                'expected_calibration_error': ece_result,
                'precision_recall': pr_result,
                'calibration_quality': _assess_calibration_quality(brier_result, ece_result, pr_result)
            }
            
    except Exception as e:
        # Record failure
        latency_ms = (time.time() - start_time) * 1000
        try:
            with DuckDBManager() as db:
                metrics_computer = MetricsComputer(db.conn)
                metrics_computer.record_latency(
                    f"/metrics/calibration/{detector_name}", 
                    latency_ms, 
                    success=False,
                    error_type=type(e).__name__
                )
        except:
            pass
            
        raise HTTPException(status_code=500, detail=f"Failed to compute calibration for {detector_name}: {str(e)}")


@router.get("/health")
async def get_health_check():
    """
    Basic health check endpoint for operational monitoring.
    
    Returns system status and key operational metrics.
    """
    start_time = time.time()
    
    try:
        with DuckDBManager() as db:
            metrics_computer = MetricsComputer(db.conn)
            
            # Quick health checks
            ops_metrics = metrics_computer.get_ops_health_metrics()
            council_metrics = metrics_computer.get_council_health_metrics()
            
            latency_ms = (time.time() - start_time) * 1000
            metrics_computer.record_latency("/metrics/health", latency_ms)
            
            # Determine overall health
            is_healthy = (
                ops_metrics.get('database_status') == 'healthy' and
                ops_metrics.get('api_status') == 'healthy' and
                ops_metrics.get('write_failures_5m', 0) < 10 and
                council_metrics.get('trade_gate_status') != 'blocked' and
                latency_ms < 500  # 500ms health check SLO
            )
            
            return {
                'status': 'healthy' if is_healthy else 'degraded',
                'response_time_ms': latency_ms,
                'database_status': ops_metrics.get('database_status', 'unknown'),
                'api_status': ops_metrics.get('api_status', 'unknown'),
                'trade_gate_status': council_metrics.get('trade_gate_status', 'unknown'),
                'write_failures_5m': ops_metrics.get('write_failures_5m', 0),
                'timestamp': time.time()
            }
            
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'status': 'unhealthy',
            'error': str(e),
            'response_time_ms': latency_ms,
            'timestamp': time.time()
        }


def _assess_calibration_quality(
    brier_result: Dict = None, 
    ece_result: Dict = None, 
    pr_result: Dict = None
) -> str:
    """Assess overall calibration quality based on metrics."""
    
    if not brier_result or not ece_result:
        return 'insufficient_data'
    
    brier_score = brier_result.get('brier_score', 1.0)
    ece = ece_result.get('ece', 1.0)
    
    # Quality assessment thresholds
    if brier_score <= 0.15 and ece <= 0.05:
        return 'excellent'
    elif brier_score <= 0.20 and ece <= 0.10:
        return 'good'
    elif brier_score <= 0.25 and ece <= 0.15:
        return 'acceptable'
    else:
        return 'poor'