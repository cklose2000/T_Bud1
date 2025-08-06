from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trading_buddy.council.vote import council_vote, council_vote_whatif
from trading_buddy.core.duck import DuckDBManager
from trading_buddy.schemas.council import CouncilResponse
from trading_buddy.schemas.hypothesis import Hypothesis

router = APIRouter()


@router.post("/vote", response_model=CouncilResponse)
async def get_council_vote(hypothesis: Hypothesis):
    """
    Get council vote for a trading hypothesis.
    """
    try:
        with DuckDBManager() as db:
            response = council_vote(db.conn, hypothesis)
            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Council vote failed: {str(e)}")


class WhatIfRequest(BaseModel):
    hypothesis: Hypothesis
    overrides: Dict[str, Any]


@router.post("/whatif", response_model=CouncilResponse)
async def get_whatif_vote(request: WhatIfRequest):
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
            response = council_vote_whatif(db.conn, request.hypothesis, request.overrides)
            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-if vote failed: {str(e)}")