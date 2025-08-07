"""
Precursor detection job that runs periodically to check for pattern precursors.
Uses APScheduler for scheduling within the FastAPI process.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict

from duckdb import DuckDBPyConnection

from trading_buddy.precursors.detector import PrecursorDetector
from trading_buddy.precursors.webhook import WebhookSender

logger = logging.getLogger(__name__)


class PrecursorJob:
    """Manages precursor detection and alerting."""
    
    def __init__(self, conn: DuckDBPyConnection):
        self.conn = conn
        self.detector = PrecursorDetector(conn)
        self.webhook_sender = WebhookSender()
        self._is_running = False
    
    def run_detection_cycle(self) -> Dict[str, any]:
        """
        Run one cycle of precursor detection.
        
        Returns:
            Dict with cycle results and statistics
        """
        if self._is_running:
            logger.warning("Precursor detection already running, skipping cycle")
            return {"status": "skipped", "reason": "already_running"}
        
        self._is_running = True
        cycle_start = datetime.now()
        
        try:
            logger.info("Starting precursor detection cycle")
            
            # Get active subscriptions
            subscriptions = self._get_active_subscriptions()
            
            if not subscriptions:
                logger.info("No active alert subscriptions")
                return {
                    "status": "completed",
                    "subscriptions": 0,
                    "alerts_sent": 0,
                    "duration_seconds": (datetime.now() - cycle_start).total_seconds()
                }
            
            # Get unique symbols from subscriptions
            symbols = list(set(sub['symbol'] for sub in subscriptions))
            
            logger.info(f"Checking {len(symbols)} symbols for {len(subscriptions)} subscriptions")
            
            # Detect precursors
            all_precursors = self.detector.detect_all_precursors(symbols)
            
            logger.info(f"Found {len(all_precursors)} potential precursors")
            
            # Process alerts
            alerts_sent = 0
            alerts_failed = 0
            
            for precursor in all_precursors:
                try:
                    # Find matching subscriptions
                    matching_subs = self._find_matching_subscriptions(
                        subscriptions, precursor
                    )
                    
                    for subscription in matching_subs:
                        # Check if we should send this alert
                        should_send = self._should_send_alert(subscription, precursor)
                        
                        if should_send:
                            # Send webhook
                            result = self._send_alert(subscription, precursor)
                            
                            if result['success']:
                                alerts_sent += 1
                            else:
                                alerts_failed += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process precursor alert: {e}")
                    alerts_failed += 1
                    continue
            
            # Cleanup old alerts
            cleaned_count = self._cleanup_old_alerts()
            
            duration = (datetime.now() - cycle_start).total_seconds()
            
            result = {
                "status": "completed",
                "subscriptions": len(subscriptions),
                "symbols_checked": len(symbols),
                "precursors_found": len(all_precursors),
                "alerts_sent": alerts_sent,
                "alerts_failed": alerts_failed,
                "cleaned_up": cleaned_count,
                "duration_seconds": round(duration, 2)
            }
            
            logger.info(f"Precursor cycle completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Precursor detection cycle failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "duration_seconds": (datetime.now() - cycle_start).total_seconds()
            }
            
        finally:
            self._is_running = False
    
    def _get_active_subscriptions(self) -> List[Dict]:
        """Get all active alert subscriptions."""
        query = """
        SELECT 
            user_id,
            symbol,
            pattern,
            timeframe,
            min_probability,
            quiet_hours_start,
            quiet_hours_end,
            webhook_url
        FROM alert_subscriptions
        WHERE active = TRUE
        """
        
        results = self.conn.execute(query).fetchall()
        
        return [
            {
                'user_id': r[0],
                'symbol': r[1],
                'pattern': r[2],
                'timeframe': r[3],
                'min_probability': r[4],
                'quiet_hours_start': r[5],
                'quiet_hours_end': r[6],
                'webhook_url': r[7]
            }
            for r in results
        ]
    
    def _find_matching_subscriptions(
        self, 
        subscriptions: List[Dict], 
        precursor: Dict
    ) -> List[Dict]:
        """Find subscriptions that match a precursor."""
        matching = []
        
        for sub in subscriptions:
            if (sub['symbol'] == precursor['symbol'] and
                sub['pattern'] == precursor['pattern'] and
                sub['timeframe'] == precursor['timeframe']):
                matching.append(sub)
        
        return matching
    
    def _should_send_alert(self, subscription: Dict, precursor: Dict) -> bool:
        """Check if we should send an alert for this precursor."""
        
        # Check probability threshold
        if precursor['probability'] < subscription['min_probability']:
            return False
        
        # Check quiet hours
        current_hour = datetime.now().hour
        quiet_start = subscription.get('quiet_hours_start')
        quiet_end = subscription.get('quiet_hours_end')
        
        if quiet_start is not None and quiet_end is not None:
            if quiet_start <= quiet_end:
                # Normal range (e.g., 22:00 to 06:00 next day)
                if quiet_start <= current_hour <= quiet_end:
                    logger.debug(f"Skipping alert in quiet hours: {current_hour}")
                    return False
            else:
                # Overnight range (e.g., 22:00 to 06:00 next day)
                if current_hour >= quiet_start or current_hour <= quiet_end:
                    logger.debug(f"Skipping alert in quiet hours: {current_hour}")
                    return False
        
        # Check for recent duplicate alerts (deduplication)
        return self._check_duplicate_alert(subscription, precursor)
    
    def _check_duplicate_alert(self, subscription: Dict, precursor: Dict) -> bool:
        """Check if we recently sent a similar alert (deduplication)."""
        
        # Look for alerts in the last 15 minutes for the same pattern
        query = """
        SELECT COUNT(*) 
        FROM precursor_alerts
        WHERE symbol = ?
        AND pattern = ?
        AND timeframe = ?
        AND user_id = ?
        AND sent_at IS NOT NULL
        AND detected_at >= CURRENT_TIMESTAMP - INTERVAL 15 MINUTE
        """
        
        count = self.conn.execute(query, [
            precursor['symbol'],
            precursor['pattern'], 
            precursor['timeframe'],
            subscription['user_id']
        ]).fetchone()[0]
        
        return count == 0  # Send only if no recent alerts
    
    def _send_alert(self, subscription: Dict, precursor: Dict) -> Dict:
        """Send alert webhook and record in database."""
        
        # Send webhook
        webhook_result = self.webhook_sender.send_precursor_alert(
            subscription['webhook_url'],
            precursor,
            subscription['user_id']
        )
        
        # Record in database
        try:
            self.conn.execute("""
                INSERT INTO precursor_alerts 
                (symbol, pattern, timeframe, probability, sent_at, webhook_response_code, user_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                precursor['symbol'],
                precursor['pattern'],
                precursor['timeframe'],
                precursor['probability'],
                datetime.now() if webhook_result['success'] else None,
                webhook_result.get('status_code'),
                subscription['user_id'],
                precursor.get('metadata', {})
            ])
            
        except Exception as e:
            logger.error(f"Failed to record alert in database: {e}")
        
        return webhook_result
    
    def _cleanup_old_alerts(self, keep_days: int = 7) -> int:
        """Clean up old alert records."""
        try:
            result = self.conn.execute("""
                DELETE FROM precursor_alerts
                WHERE detected_at < CURRENT_TIMESTAMP - INTERVAL ? DAY
            """, [keep_days])
            
            # Get count from result
            count = result.fetchone()
            return count[0] if count else 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")
            return 0
    
    def get_alert_stats(self, days: int = 7) -> Dict[str, any]:
        """Get statistics about recent alerts."""
        try:
            stats_query = """
            SELECT 
                COUNT(*) as total_alerts,
                COUNT(CASE WHEN sent_at IS NOT NULL THEN 1 END) as sent_alerts,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(probability) as avg_probability
            FROM precursor_alerts
            WHERE detected_at >= CURRENT_TIMESTAMP - INTERVAL ? DAY
            """
            
            stats = self.conn.execute(stats_query, [days]).fetchone()
            
            # Get top patterns
            pattern_query = """
            SELECT 
                pattern,
                COUNT(*) as count,
                AVG(probability) as avg_prob
            FROM precursor_alerts
            WHERE detected_at >= CURRENT_TIMESTAMP - INTERVAL ? DAY
            GROUP BY pattern
            ORDER BY count DESC
            LIMIT 5
            """
            
            patterns = self.conn.execute(pattern_query, [days]).fetchall()
            
            return {
                "total_alerts": stats[0],
                "sent_alerts": stats[1],
                "send_rate": stats[1] / max(1, stats[0]),
                "unique_symbols": stats[2],
                "unique_users": stats[3],
                "avg_probability": round(stats[4], 3) if stats[4] else 0,
                "top_patterns": [
                    {"pattern": p[0], "count": p[1], "avg_probability": round(p[2], 3)}
                    for p in patterns
                ],
                "period_days": days
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert stats: {e}")
            return {"error": str(e)}


# Singleton instance for the scheduler
_precursor_job_instance = None


def get_precursor_job(conn: DuckDBPyConnection) -> PrecursorJob:
    """Get precursor job singleton."""
    global _precursor_job_instance
    
    if _precursor_job_instance is None:
        _precursor_job_instance = PrecursorJob(conn)
    
    return _precursor_job_instance