from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TradingPlan(BaseModel):
    entry: str
    stop: str
    time_stop: int
    size: float


class CIBreakdown(BaseModel):
    timeframe: str
    pattern: str
    ci: float
    present: bool
    exp_lift: Optional[float] = None


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