from contextlib import asynccontextmanager
from typing import Dict
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from trading_buddy.api.routes import agent, council, nlp, signal
from trading_buddy.core.config import settings
from trading_buddy.core.ddl import initialize_database
from trading_buddy.core.duck import DuckDBManager, initialize_schema_discovery
from trading_buddy.precursors.job import get_precursor_job

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app resources on startup."""
    # Initialize database
    with DuckDBManager() as db:
        # Create tables
        initialize_database(db.conn)
        
        # Auto-discover schema
        if not initialize_schema_discovery(db.conn):
            print("Warning: Could not auto-discover OHLCV table schema")
    
    # Start precursor detection scheduler
    try:
        def run_precursor_cycle():
            """Sync wrapper for precursor job."""
            try:
                with DuckDBManager() as db:
                    job = get_precursor_job(db.conn)
                    result = job.run_detection_cycle()
                    logger.info(f"Precursor cycle: {result}")
            except Exception as e:
                logger.error(f"Precursor cycle failed: {e}")
        
        # Schedule precursor job every 60 seconds
        scheduler.add_job(
            run_precursor_cycle,
            'interval', 
            seconds=60,
            id='precursor_detection',
            max_instances=1  # Prevent overlapping runs
        )
        
        # Add daily report generation job (runs at 12:30 AM)
        def run_daily_reports():
            """Generate daily reports for active symbols."""
            try:
                with DuckDBManager() as db:
                    from trading_buddy.reports.build_daily_report import build_all_reports
                    results = build_all_reports(db.conn)
                    logger.info(f"Built {len(results)} daily reports")
            except Exception as e:
                logger.error(f"Daily reports failed: {e}")
        
        scheduler.add_job(
            run_daily_reports,
            'cron',
            hour=0,
            minute=30,
            id='daily_reports'
        )
        
        # Add nightly calibration job (runs at 2:00 AM)
        def run_nightly_calibration():
            """Run nightly calibration computation for all detectors."""
            try:
                from scripts.nightly_calibration import run_nightly_calibration
                success = run_nightly_calibration()
                logger.info(f"Nightly calibration completed successfully: {success}")
            except Exception as e:
                logger.error(f"Nightly calibration failed: {e}")
        
        scheduler.add_job(
            run_nightly_calibration,
            'cron',
            hour=2,
            minute=0,
            id='nightly_calibration'
        )
        
        # Start scheduler
        scheduler.start()
        logger.info("Started background scheduler with precursor detection, daily reports, and nightly calibration")
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
    
    yield
    
    # Cleanup
    try:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")
    except Exception as e:
        logger.error(f"Error shutting down scheduler: {e}")


app = FastAPI(
    title="Trading Buddy API",
    description="NLP-fluent trading buddy with council-based signal validation",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Latency tracking middleware for SLO monitoring
from trading_buddy.middleware.latency_tracker import LatencyTrackingMiddleware
app.add_middleware(LatencyTrackingMiddleware)

# Include routers
app.include_router(nlp.router, prefix="/nlp", tags=["NLP"])
app.include_router(council.router, prefix="/council", tags=["Council"])
app.include_router(signal.router, prefix="/signal", tags=["Signal"])
app.include_router(agent.router, prefix="/agent", tags=["Agent"])

# Import and include additional routers
from trading_buddy.api.routes import agentic, precursor, metrics
app.include_router(agentic.router, prefix="/agentic", tags=["Agentic"])
app.include_router(precursor.router, prefix="/precursor", tags=["Precursor"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Trading Buddy API",
        "version": "0.1.0",
        "endpoints": [
            "/nlp/parse",
            "/council/vote", 
            "/council/whatif",
            "/signal/backfill",
            "/agent/alerts",
            "/metrics/summary",
            "/metrics/health",
            "/metrics/calibration/{detector_name}",
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        with DuckDBManager() as db:
            # Check database connection
            result = db.conn.execute("SELECT 1").fetchone()
            
            # Check if views exist
            tables = db.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                AND table_name LIKE 'bars%'
            """).fetchall()
            
            return {
                "status": "healthy",
                "database": "connected",
                "views": [t[0] for t in tables],
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }