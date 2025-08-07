from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from trading_buddy.council.vote import council_vote, council_vote_whatif
from trading_buddy.core.duck import DuckDBManager
from trading_buddy.middleware.referee_enhanced import EnhancedReferee
from trading_buddy.schemas.council import CouncilResponse
from trading_buddy.schemas.hypothesis import Hypothesis
from trading_buddy.whatif.cache import get_cache_instance

router = APIRouter()


@router.post("/vote", response_model=CouncilResponse)
async def get_council_vote(
    hypothesis: Hypothesis,
    referee_mode: bool = Query(False, description="Enable referee validation")
):
    """
    Get council vote for a trading hypothesis.
    """
    try:
        with DuckDBManager() as db:
            response = council_vote(db.conn, hypothesis)
            
            # Apply referee middleware if requested
            if referee_mode:
                referee = EnhancedReferee(db.conn)
                
                # Convert response to text for validation
                response_text = f"""
                Sample size: {response.n} events
                Median return: {response.median_return:.4f}
                Hit rate: {response.hit_rate:.1%}
                Council score: {response.council_score:.2f}
                """
                
                # Extract numeric claims
                claims = referee.extract_numeric_claims(response_text, message_id=f"council_vote_{hash(response_text)}")
                
                # Verify against tool results
                tool_results = {
                    "council_vote": {
                        "n": response.n,
                        "median_return": response.median_return,
                        "hit_rate": response.hit_rate,
                        "council_score": response.council_score
                    }
                }
                
                verified_claims = referee.verify_claims_against_tools(claims, tool_results)
                
                # Check if any claims were rejected
                rejected_claims = [c for c in verified_claims if c.verdict == "rejected"]
                
                # Log final response
                message_id = f"council_vote_{hash(response_text)}"
                referee.log_final_response(message_id, "/council/vote", hypothesis.symbol, response_text, verified_claims)
                
                # Add validation metadata to response
                response.metadata = response.metadata or {}
                response.metadata["referee_validation"] = {
                    "total_claims": len(verified_claims),
                    "verified_claims": len([c for c in verified_claims if c.verdict in ["verified", "corrected"]]),
                    "rejected_claims": len(rejected_claims),
                    "status": "rejected" if rejected_claims else "approved"
                }
                
                if rejected_claims:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "message": "Response rejected by referee - unverified claims detected",
                            "rejected_claims": [{"claim": c.claim_text, "value": c.value_numeric} for c in rejected_claims],
                            "suggestion": "All numeric claims must be traceable to tool outputs"
                        }
                    )
            
            return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Council vote failed: {str(e)}")


class WhatIfRequest(BaseModel):
    hypothesis: Hypothesis
    overrides: Dict[str, Any]


@router.post("/whatif", response_model=CouncilResponse)
async def get_whatif_vote(
    request: WhatIfRequest,
    use_cache: bool = Query(True, description="Use what-if cache if available"),
    cache_ttl_hours: int = Query(8, description="Cache TTL in hours", ge=1, le=24)
):
    """
    Get council vote with what-if scenario overrides.
    
    Example overrides:
    {
        "contexts": [{"tf": "60m", "pattern": "regime_trend", "present": true}],
        "horizon": 15
    }
    """
    try:
        with DuckDBManager() as db:
            if use_cache:
                cache = get_cache_instance(db.conn)
                response, was_cached = cache.get_or_compute(
                    request.hypothesis,
                    request.overrides,
                    ttl_hours=cache_ttl_hours
                )
                
                # Add cache metadata
                response.metadata = response.metadata or {}
                response.metadata["cached"] = was_cached
                response.metadata["cache_ttl_hours"] = cache_ttl_hours
                
            else:
                response = council_vote_whatif(db.conn, request.hypothesis, request.overrides)
                response.metadata = response.metadata or {}
                response.metadata["cached"] = False
            
            return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-if vote failed: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get what-if cache statistics."""
    try:
        with DuckDBManager() as db:
            cache = get_cache_instance(db.conn)
            stats = cache.get_stats()
            return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")


@router.post("/cache/cleanup")
async def cleanup_cache():
    """Clean up expired cache entries."""
    try:
        with DuckDBManager() as db:
            cache = get_cache_instance(db.conn)
            removed_count = cache.cleanup_expired()
            return {
                "status": "success",
                "entries_removed": removed_count,
                "message": f"Cleaned up {removed_count} expired cache entries"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache cleanup failed: {str(e)}")


@router.post("/cache/precompute")
async def precompute_scenarios(
    symbols: List[str] = Query(..., description="Symbols to precompute scenarios for")
):
    """Precompute common what-if scenarios for symbols."""
    try:
        if len(symbols) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
        
        with DuckDBManager() as db:
            cache = get_cache_instance(db.conn)
            results = cache.precompute_common_scenarios([s.upper() for s in symbols])
            
            total_precomputed = sum(results.values())
            
            return {
                "status": "success",
                "total_scenarios": total_precomputed,
                "by_symbol": results,
                "message": f"Precomputed {total_precomputed} scenarios across {len(symbols)} symbols"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Precomputation failed: {str(e)}")