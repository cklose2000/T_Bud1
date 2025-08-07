"""
API endpoints for precursor detection and alert management.
"""
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field, HttpUrl

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.precursors.job import get_precursor_job
from trading_buddy.precursors.webhook import WebhookSender

router = APIRouter()


class AlertSubscription(BaseModel):
    """Alert subscription configuration."""
    user_id: str = Field(..., description="User identifier")
    symbol: str = Field(..., description="Stock symbol")
    pattern: str = Field(..., description="Pattern to monitor")
    timeframe: str = Field(default="5m", description="Timeframe")
    min_probability: float = Field(
        default=0.7, 
        description="Minimum probability threshold",
        ge=0.5, 
        le=0.95
    )
    quiet_hours_start: Optional[int] = Field(
        default=None, 
        description="Quiet hours start (0-23)",
        ge=0,
        le=23
    )
    quiet_hours_end: Optional[int] = Field(
        default=None,
        description="Quiet hours end (0-23)",
        ge=0, 
        le=23
    )
    webhook_url: HttpUrl = Field(..., description="Webhook URL for notifications")


class AlertSubscriptionResponse(BaseModel):
    """Alert subscription response."""
    user_id: str
    symbol: str
    pattern: str
    timeframe: str
    min_probability: float
    quiet_hours_start: Optional[int]
    quiet_hours_end: Optional[int]
    webhook_url: str
    active: bool
    created_at: datetime


@router.post("/alerts/subscribe", response_model=AlertSubscriptionResponse)
async def subscribe_to_alerts(subscription: AlertSubscription):
    """Subscribe to precursor alerts for a pattern."""
    try:
        with DuckDBManager() as db:
            # Test webhook first
            webhook_sender = WebhookSender()
            test_result = webhook_sender.test_webhook(str(subscription.webhook_url))
            
            if not test_result['success']:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Webhook test failed",
                        "error": test_result.get('error', 'Unknown error'),
                        "status_code": test_result.get('status_code')
                    }
                )
            
            # Insert subscription
            db.conn.execute("""
                INSERT OR REPLACE INTO alert_subscriptions
                (user_id, symbol, pattern, timeframe, min_probability, 
                 quiet_hours_start, quiet_hours_end, webhook_url, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            """, [
                subscription.user_id,
                subscription.symbol.upper(),
                subscription.pattern,
                subscription.timeframe,
                subscription.min_probability,
                subscription.quiet_hours_start,
                subscription.quiet_hours_end,
                str(subscription.webhook_url)
            ])
            
            # Fetch the created subscription
            result = db.conn.execute("""
                SELECT user_id, symbol, pattern, timeframe, min_probability,
                       quiet_hours_start, quiet_hours_end, webhook_url, 
                       active, created_at
                FROM alert_subscriptions
                WHERE user_id = ? AND symbol = ? AND pattern = ? AND timeframe = ?
            """, [
                subscription.user_id,
                subscription.symbol.upper(), 
                subscription.pattern,
                subscription.timeframe
            ]).fetchone()
            
            return AlertSubscriptionResponse(
                user_id=result[0],
                symbol=result[1],
                pattern=result[2],
                timeframe=result[3],
                min_probability=result[4],
                quiet_hours_start=result[5],
                quiet_hours_end=result[6],
                webhook_url=result[7],
                active=result[8],
                created_at=result[9]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription failed: {str(e)}")


@router.get("/alerts/subscriptions", response_model=List[AlertSubscriptionResponse])
async def get_subscriptions(
    user_id: str = Query(..., description="User identifier"),
    active_only: bool = Query(True, description="Only return active subscriptions")
):
    """Get alert subscriptions for a user."""
    try:
        with DuckDBManager() as db:
            query = """
            SELECT user_id, symbol, pattern, timeframe, min_probability,
                   quiet_hours_start, quiet_hours_end, webhook_url, 
                   active, created_at
            FROM alert_subscriptions
            WHERE user_id = ?
            """
            
            params = [user_id]
            
            if active_only:
                query += " AND active = TRUE"
            
            query += " ORDER BY created_at DESC"
            
            results = db.conn.execute(query, params).fetchall()
            
            return [
                AlertSubscriptionResponse(
                    user_id=r[0],
                    symbol=r[1], 
                    pattern=r[2],
                    timeframe=r[3],
                    min_probability=r[4],
                    quiet_hours_start=r[5],
                    quiet_hours_end=r[6],
                    webhook_url=r[7],
                    active=r[8],
                    created_at=r[9]
                )
                for r in results
            ]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get subscriptions: {str(e)}")


@router.delete("/alerts/unsubscribe")
async def unsubscribe_from_alerts(
    user_id: str = Query(..., description="User identifier"),
    symbol: str = Query(..., description="Stock symbol"),
    pattern: str = Query(..., description="Pattern name"),
    timeframe: str = Query(default="5m", description="Timeframe")
):
    """Unsubscribe from alerts."""
    try:
        with DuckDBManager() as db:
            result = db.conn.execute("""
                UPDATE alert_subscriptions
                SET active = FALSE
                WHERE user_id = ? AND symbol = ? AND pattern = ? AND timeframe = ?
            """, [user_id, symbol.upper(), pattern, timeframe])
            
            return {
                "status": "success",
                "message": f"Unsubscribed from {symbol} {pattern} alerts"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unsubscribe failed: {str(e)}")


@router.post("/alerts/test-webhook")
async def test_webhook(
    webhook_url: HttpUrl = Body(..., description="Webhook URL to test")
):
    """Test a webhook URL."""
    try:
        webhook_sender = WebhookSender()
        result = webhook_sender.test_webhook(str(webhook_url))
        
        if result['success']:
            return {
                "status": "success",
                "message": "Webhook test successful",
                "response_code": result.get('status_code'),
                "latency_ms": result.get('latency_ms')
            }
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "failed",
                    "error": result.get('error', 'Unknown error'),
                    "response_code": result.get('status_code')
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook test failed: {str(e)}")


@router.get("/alerts/history")
async def get_alert_history(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    pattern: Optional[str] = Query(None, description="Filter by pattern"),
    days: int = Query(7, description="Days to look back", ge=1, le=30),
    limit: int = Query(50, description="Max results", ge=1, le=200)
):
    """Get alert history."""
    try:
        with DuckDBManager() as db:
            query = """
            SELECT 
                symbol, pattern, timeframe, probability,
                detected_at, sent_at, webhook_response_code, user_id
            FROM precursor_alerts
            WHERE detected_at >= CURRENT_TIMESTAMP - INTERVAL ? DAY
            """
            
            params = [days]
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.upper())
            
            if pattern:
                query += " AND pattern = ?"
                params.append(pattern)
            
            query += " ORDER BY detected_at DESC LIMIT ?"
            params.append(limit)
            
            results = db.conn.execute(query, params).fetchall()
            
            return {
                "alerts": [
                    {
                        "symbol": r[0],
                        "pattern": r[1],
                        "timeframe": r[2],
                        "probability": r[3],
                        "detected_at": r[4],
                        "sent_at": r[5],
                        "webhook_response_code": r[6],
                        "user_id": r[7],
                        "was_sent": r[5] is not None
                    }
                    for r in results
                ],
                "total": len(results),
                "period_days": days
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert history: {str(e)}")


@router.get("/alerts/stats")
async def get_alert_stats(
    days: int = Query(7, description="Days to analyze", ge=1, le=30)
):
    """Get alert statistics."""
    try:
        with DuckDBManager() as db:
            job = get_precursor_job(db.conn)
            stats = job.get_alert_stats(days)
            return stats
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/detection/run-now")
async def run_detection_now():
    """Manually trigger precursor detection."""
    try:
        with DuckDBManager() as db:
            job = get_precursor_job(db.conn)
            result = job.run_detection_cycle()
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.get("/patterns/supported")
async def get_supported_patterns():
    """Get list of supported precursor patterns."""
    return {
        "patterns": [
            {
                "name": "double_bottom",
                "description": "Double bottom reversal pattern",
                "timeframes": ["5m", "15m", "30m", "1h"],
                "typical_probability_range": [0.3, 0.9]
            },
            {
                "name": "macd_bull_cross", 
                "description": "MACD bullish crossover",
                "timeframes": ["5m", "15m", "30m"],
                "typical_probability_range": [0.4, 0.8]
            }
        ],
        "timeframes": ["5m", "15m", "30m", "1h"],
        "min_probability": 0.5,
        "max_probability": 0.95
    }