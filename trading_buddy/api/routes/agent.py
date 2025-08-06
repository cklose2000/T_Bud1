from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trading_buddy.core.duck import DuckDBManager

router = APIRouter()


class Alert(BaseModel):
    id: str
    symbol: str
    timeframe: str
    alert_type: str
    message: str
    created_at: datetime
    snoozed_until: Optional[datetime] = None


class PrecursorConfig(BaseModel):
    symbol: str
    pattern: str
    threshold: float = 0.7


# In-memory alert storage (in production, use a proper queue/database)
alerts: Dict[str, Alert] = {}


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(active_only: bool = True):
    """
    Get pending alerts.
    """
    now = datetime.now()
    
    if active_only:
        return [
            alert for alert in alerts.values()
            if alert.snoozed_until is None or alert.snoozed_until <= now
        ]
    
    return list(alerts.values())


@router.post("/snooze/{alert_id}")
async def snooze_alert(alert_id: str, minutes: int = 60):
    """
    Snooze an alert for specified minutes.
    """
    if alert_id not in alerts:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alerts[alert_id].snoozed_until = datetime.now() + timedelta(minutes=minutes)
    
    return {"status": "snoozed", "until": alerts[alert_id].snoozed_until}


@router.post("/arm_precursor")
async def arm_precursor(config: PrecursorConfig):
    """
    Configure precursor detection for a pattern.
    """
    # In a real implementation, this would:
    # 1. Start monitoring for the pattern precursors
    # 2. Calculate probability of pattern completion
    # 3. Generate alerts when threshold is met
    
    # For now, create a mock alert
    alert_id = f"{config.symbol}_{config.pattern}_{datetime.now().timestamp()}"
    
    alert = Alert(
        id=alert_id,
        symbol=config.symbol,
        timeframe="5m",
        alert_type="precursor",
        message=f"Likely {config.pattern} forming on {config.symbol} - prepare entry",
        created_at=datetime.now(),
    )
    
    alerts[alert_id] = alert
    
    return {
        "status": "armed",
        "config": config,
        "alert_id": alert_id,
    }


@router.delete("/alerts/{alert_id}")
async def dismiss_alert(alert_id: str):
    """
    Dismiss an alert permanently.
    """
    if alert_id not in alerts:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    del alerts[alert_id]
    
    return {"status": "dismissed"}