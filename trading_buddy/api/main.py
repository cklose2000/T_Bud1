from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from trading_buddy.api.routes import agent, council, nlp, signal
from trading_buddy.core.config import settings
from trading_buddy.core.ddl import initialize_database
from trading_buddy.core.duck import DuckDBManager, initialize_schema_discovery


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
    
    yield
    
    # Cleanup (if needed)


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

# Include routers
app.include_router(nlp.router, prefix="/nlp", tags=["NLP"])
app.include_router(council.router, prefix="/council", tags=["Council"])
app.include_router(signal.router, prefix="/signal", tags=["Signal"])
app.include_router(agent.router, prefix="/agent", tags=["Agent"])

# Import and include agentic router
from trading_buddy.api.routes import agentic
app.include_router(agentic.router, prefix="/agentic", tags=["Agentic"])


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