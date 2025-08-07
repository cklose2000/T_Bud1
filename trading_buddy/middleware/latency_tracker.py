"""
PR7: Latency Tracking Middleware

Automatically tracks P95 latency for all API endpoints to ensure SLO compliance.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.core.metrics import MetricsComputer


class LatencyTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track endpoint latency and add metrics_ref_id to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request latency and add metrics reference ID."""
        start_time = time.time()
        
        # Generate unique metrics reference ID for this request
        metrics_ref_id = str(uuid.uuid4())
        request.state.metrics_ref_id = metrics_ref_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Add metrics reference to response headers
            response.headers["X-Metrics-Ref-ID"] = metrics_ref_id
            
            # Record latency (async, don't block response)
            self._record_latency_async(
                endpoint=f"{request.method} {request.url.path}",
                latency_ms=latency_ms,
                success=200 <= response.status_code < 400
            )
            
            return response
            
        except Exception as e:
            # Calculate latency for failed requests too
            latency_ms = (time.time() - start_time) * 1000
            
            # Record failure latency
            self._record_latency_async(
                endpoint=f"{request.method} {request.url.path}",
                latency_ms=latency_ms,
                success=False,
                error_type=type(e).__name__
            )
            
            raise
    
    def _record_latency_async(
        self, 
        endpoint: str, 
        latency_ms: float, 
        success: bool = True, 
        error_type: str = None
    ):
        """Record latency asynchronously (fire and forget)."""
        try:
            # Use a separate connection to avoid interfering with request processing
            with DuckDBManager() as db:
                metrics_computer = MetricsComputer(db.conn)
                metrics_computer.record_latency(endpoint, latency_ms, success, error_type)
        except Exception:
            # Silently fail - don't let metrics recording break the API
            pass


def add_metrics_ref_to_response(response_data: dict, request: Request) -> dict:
    """Helper function to add metrics_ref_id to JSON responses."""
    if hasattr(request.state, 'metrics_ref_id'):
        if isinstance(response_data, dict):
            response_data['metrics_ref_id'] = request.state.metrics_ref_id
    return response_data