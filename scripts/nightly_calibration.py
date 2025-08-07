#!/usr/bin/env python3
"""
PR7: Nightly Calibration Job

Computes Brier scores, Expected Calibration Error, and precision/recall
for all detectors with sufficient OOS data. Runs daily at 2 AM.
"""
import sys
sys.path.append('.')

import logging
from datetime import datetime
from typing import Dict, List

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.core.metrics import MetricsComputer
from trading_buddy.core.threshold_tuner import ThresholdTuner

logger = logging.getLogger(__name__)


def get_active_detectors(conn) -> List[Dict[str, str]]:
    """Get list of active detectors with recent patterns."""
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
        GROUP BY detector_name, symbol, timeframe
        HAVING COUNT(*) >= 10  -- Minimum sample size for meaningful calibration
        ORDER BY pattern_count DESC
        """
        
        results = conn.execute(query).fetchall()
        
        detectors = []
        for row in results:
            detector_name, symbol, timeframe, count = row
            detectors.append({
                'detector_name': detector_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'pattern_count': count
            })
        
        logger.info(f"Found {len(detectors)} active detector configurations")
        return detectors
        
    except Exception as e:
        logger.error(f"Failed to get active detectors: {e}")
        return []


def compute_all_calibration_metrics():
    """Main function to compute calibration metrics for all active detectors."""
    logger.info("Starting nightly calibration computation")
    
    with DuckDBManager() as db:
        metrics_computer = MetricsComputer(db.conn)
        
        # Get active detectors
        detectors = get_active_detectors(db.conn)
        
        if not detectors:
            logger.warning("No active detectors found - skipping calibration")
            return
        
        results = {
            'brier_computed': 0,
            'ece_computed': 0,
            'pr_computed': 0,
            'failed': 0
        }
        
        for detector_config in detectors:
            detector_name = detector_config['detector_name']
            symbol = detector_config['symbol']
            timeframe = detector_config['timeframe']
            
            logger.info(f"Computing calibration for {detector_name}:{symbol}:{timeframe}")
            
            try:
                # Compute Brier Score
                brier_result = metrics_computer.compute_brier_score(
                    detector_name=detector_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=30
                )
                
                if brier_result:
                    results['brier_computed'] += 1
                    logger.info(f"  Brier Score: {brier_result['brier_score']:.4f} (n={brier_result['n_samples']})")
                
                # Compute Expected Calibration Error
                ece_result = metrics_computer.compute_expected_calibration_error(
                    detector_name=detector_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=30
                )
                
                if ece_result:
                    results['ece_computed'] += 1
                    logger.info(f"  ECE: {ece_result['ece']:.4f} ({ece_result['calibration_quality']})")
                
                # Compute Precision/Recall
                pr_result = metrics_computer.compute_precision_recall(
                    detector_name=detector_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=30
                )
                
                if pr_result:
                    results['pr_computed'] += 1
                    logger.info(f"  P/R: {pr_result['precision']:.3f}/{pr_result['recall']:.3f} (F1: {pr_result['f1_score']:.3f})")
                
                # Check for concerning calibration issues
                if brier_result and brier_result['brier_score'] > 0.30:
                    logger.warning(f"HIGH BRIER SCORE: {detector_name} has Brier {brier_result['brier_score']:.4f} > 0.30")
                
                if ece_result and ece_result['ece'] > 0.15:
                    logger.warning(f"HIGH ECE: {detector_name} has ECE {ece_result['ece']:.4f} > 0.15 (poor calibration)")
                
            except Exception as e:
                logger.error(f"Failed to compute calibration for {detector_name}:{symbol}:{timeframe}: {e}")
                results['failed'] += 1
        
        # Log summary
        logger.info(f"Nightly calibration complete: {results}")
        
        # Store nightly job completion
        try:
            db.conn.execute("""
                INSERT INTO system_health_metrics 
                (metric_name, metric_value, component, metadata)
                VALUES ('nightly_calibration_complete', 1.0, 'calibration', ?)
            """, [str(results).replace("'", '"')])  # JSON-like format
        except Exception as e:
            logger.error(f"Failed to record nightly job completion: {e}")


def check_calibration_alerts():
    """Check for calibration issues that need attention."""
    logger.info("Checking for calibration alerts")
    
    with DuckDBManager() as db:
        # Check for detectors with concerning metrics in the last 24 hours
        alerts = []
        
        try:
            # High Brier scores
            high_brier_query = """
            SELECT detector_name, symbol, timeframe, value, n_samples
            FROM calibration_metrics 
            WHERE metric_type = 'brier' 
            AND ts >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            AND value > 0.30
            ORDER BY value DESC
            """
            
            high_brier = db.conn.execute(high_brier_query).fetchall()
            for row in high_brier:
                alerts.append(f"HIGH BRIER: {row[0]}:{row[1]}:{row[2]} = {row[3]:.4f} (n={row[4]})")
            
            # Poor calibration (high ECE)
            high_ece_query = """
            SELECT detector_name, symbol, timeframe, value, n_samples
            FROM calibration_metrics 
            WHERE metric_type = 'ece' 
            AND ts >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            AND value > 0.15
            ORDER BY value DESC
            """
            
            high_ece = db.conn.execute(high_ece_query).fetchall()
            for row in high_ece:
                alerts.append(f"POOR CALIBRATION: {row[0]}:{row[1]}:{row[2]} ECE = {row[3]:.4f} (n={row[4]})")
            
            # Low precision detectors
            low_precision_query = """
            SELECT detector_name, symbol, timeframe, value, n_samples
            FROM calibration_metrics 
            WHERE metric_type = 'precision' 
            AND ts >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            AND value < 0.40
            ORDER BY value ASC
            """
            
            low_precision = db.conn.execute(low_precision_query).fetchall()
            for row in low_precision:
                alerts.append(f"LOW PRECISION: {row[0]}:{row[1]}:{row[2]} = {row[3]:.3f} (n={row[4]})")
            
            if alerts:
                logger.warning("CALIBRATION ALERTS:")
                for alert in alerts:
                    logger.warning(f"  {alert}")
            else:
                logger.info("No calibration alerts - all detectors within acceptable ranges")
                
        except Exception as e:
            logger.error(f"Failed to check calibration alerts: {e}")


def tune_all_thresholds():
    """Tune thresholds for all detectors based on recent OOS performance."""
    logger.info("Starting threshold tuning for all detectors")
    
    with DuckDBManager() as db:
        threshold_tuner = ThresholdTuner(db.conn)
        results = threshold_tuner.tune_all_detectors(min_samples=20)
        
        logger.info(f"Threshold tuning results: {results}")
        
        # Alert on detectors that couldn't be tuned
        if results['insufficient_data'] > 0:
            logger.warning(f"{results['insufficient_data']} detectors have insufficient data for threshold tuning")
        
        if results['failed'] > 0:
            logger.error(f"{results['failed']} detectors failed threshold tuning")
        
        return results


def run_nightly_calibration():
    """Main entry point for nightly calibration job."""
    start_time = datetime.now()
    logger.info(f"Starting nightly calibration job at {start_time}")
    
    try:
        # Step 1: Tune thresholds based on recent performance
        logger.info("=== Step 1: Threshold Tuning ===")
        threshold_results = tune_all_thresholds()
        
        # Step 2: Compute all calibration metrics
        logger.info("=== Step 2: Calibration Metrics ===")
        compute_all_calibration_metrics()
        
        # Step 3: Check for alerts
        logger.info("=== Step 3: Calibration Alerts ===")
        check_calibration_alerts()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Nightly calibration completed in {duration:.1f} seconds")
        
        # Log combined results
        logger.info(f"Summary - Thresholds: {threshold_results['thresholds_updated']} updated, "
                   f"Calibration: complete")
        
        return True
        
    except Exception as e:
        logger.error(f"Nightly calibration job failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = run_nightly_calibration()
    exit(0 if success else 1)