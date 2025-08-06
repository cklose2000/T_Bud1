from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.detectors.contexts import (
    detect_large_drop,
    detect_regime,
    detect_trend_context,
)
from trading_buddy.detectors.outcomes import update_event_outcomes
from trading_buddy.detectors.patterns import (
    detect_compound_pattern,
    detect_double_bottom,
    detect_macd_bull_cross,
)

router = APIRouter()


class BackfillRequest(BaseModel):
    symbols: List[str]
    timeframes: List[str] = ["5m", "15m", "60m"]
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    days: Optional[int] = 365


async def backfill_events_task(
    symbols: List[str],
    timeframes: List[str],
    start_date: datetime,
    end_date: datetime,
):
    """Background task to backfill events."""
    with DuckDBManager() as db:
        conn = db.conn
        
        for symbol in symbols:
            # Detect patterns for 5m
            if "5m" in timeframes:
                # Double bottoms
                patterns = detect_double_bottom(
                    conn, symbol, "5m",
                    start_ts=start_date.isoformat(),
                    end_ts=end_date.isoformat()
                )
                
                # Insert into events table
                for p in patterns:
                    conn.execute("""
                        INSERT OR IGNORE INTO events 
                        (symbol, timeframe, event_ts, pattern, attrs)
                        VALUES (?, ?, ?, ?, ?)
                    """, [symbol, "5m", p["event_ts"], p["pattern"], p["attrs"]])
                
                # MACD crosses
                crosses = detect_macd_bull_cross(
                    conn, symbol, "5m",
                    start_ts=start_date.isoformat(),
                    end_ts=end_date.isoformat()
                )
                
                for c in crosses:
                    conn.execute("""
                        INSERT OR IGNORE INTO events 
                        (symbol, timeframe, event_ts, pattern, attrs)
                        VALUES (?, ?, ?, ?, ?)
                    """, [symbol, "5m", c["event_ts"], c["pattern"], c["attrs"]])
                
                # Compound patterns
                compounds = detect_compound_pattern(
                    conn, symbol, "5m",
                    start_ts=start_date.isoformat(),
                    end_ts=end_date.isoformat()
                )
                
                for comp in compounds:
                    conn.execute("""
                        INSERT OR IGNORE INTO events 
                        (symbol, timeframe, event_ts, pattern, attrs)
                        VALUES (?, ?, ?, ?, ?)
                    """, [symbol, "5m", comp["event_ts"], comp["pattern"], comp["attrs"]])
            
            # Detect contexts
            if "15m" in timeframes:
                contexts = detect_trend_context(
                    conn, symbol, "15m",
                    start_ts=start_date.isoformat(),
                    end_ts=end_date.isoformat()
                )
                
                for ctx in contexts:
                    conn.execute("""
                        INSERT OR IGNORE INTO contexts
                        (symbol, timeframe, event_ts, context, attrs)
                        VALUES (?, ?, ?, ?, ?)
                    """, [symbol, "15m", ctx["event_ts"], ctx["context"], ctx["attrs"]])
            
            if "60m" in timeframes:
                regimes = detect_regime(
                    conn, symbol, "60m",
                    start_ts=start_date.isoformat(),
                    end_ts=end_date.isoformat()
                )
                
                for r in regimes:
                    conn.execute("""
                        INSERT OR IGNORE INTO contexts
                        (symbol, timeframe, event_ts, context, attrs)
                        VALUES (?, ?, ?, ?, ?)
                    """, [symbol, "60m", r["event_ts"], r["context"], r["attrs"]])
        
        # Update outcomes for all events
        events = conn.execute("""
            SELECT symbol, timeframe, event_ts, pattern
            FROM events
            WHERE fwd_ret IS NULL
        """).fetchall()
        
        event_dicts = [
            {
                "symbol": e[0],
                "timeframe": e[1],
                "event_ts": e[2],
                "pattern": e[3],
            }
            for e in events
        ]
        
        update_event_outcomes(conn, event_dicts)
        
        conn.commit()


@router.post("/backfill")
async def backfill_signals(
    request: BackfillRequest,
    background_tasks: BackgroundTasks,
):
    """
    Backfill historical events and contexts for specified symbols.
    Runs as a background task.
    """
    # Calculate date range
    end_date = request.to_date or datetime.now()
    if request.from_date:
        start_date = request.from_date
    else:
        days = request.days or 365
        start_date = end_date - timedelta(days=days)
    
    # Start background task
    background_tasks.add_task(
        backfill_events_task,
        request.symbols,
        request.timeframes,
        start_date,
        end_date,
    )
    
    return {
        "status": "backfill started",
        "symbols": request.symbols,
        "timeframes": request.timeframes,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }