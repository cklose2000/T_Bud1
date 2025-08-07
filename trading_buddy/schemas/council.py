from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TradingPlan(BaseModel):
    entry: str
    stop: str
    time_stop: int
    size: float


class CIBreakdown(BaseModel):
    context_tf: str  # Renamed for clarity
    context_name: str  # Renamed for clarity
    ci: float
    present: bool
    n: int = 0  # Sample size
    exp_lift: float = 0.0
    hit_rate: float = 0.5
    stability: float = 0.0
    metadata: Optional[Dict] = None  # For CI v2 additional info


class CouncilResponse(BaseModel):
    n: int
    median_fwd_ret: float
    mean_fwd_ret: float
    hit_rate: float
    mae: float
    council_score: float
    ci_breakdown: List[CIBreakdown]
    plan: TradingPlan
    counterfactuals: Optional[Dict[str, float]] = None
    follow_ups: List[str] = Field(default_factory=list)
    metadata: Optional[Dict] = None  # For CI version tracking, request ID, etc.