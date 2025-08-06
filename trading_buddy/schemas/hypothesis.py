from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PatternConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ContextConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class EstimationTarget(BaseModel):
    horizon_bars: int = Field(default=10)
    targets: List[str] = Field(default=["fwd_ret", "hit_rate", "mae"])


class CrossTimeframeVote(BaseModel):
    tf: str
    validators: List[Dict[str, Any]]
    lead_window: str


class Hypothesis(BaseModel):
    symbol: str
    base_timeframe: str = Field(default="5m")
    patterns: List[PatternConfig]
    context: List[ContextConfig] = Field(default_factory=list)
    what_to_estimate: EstimationTarget = Field(default_factory=EstimationTarget)
    cross_timeframe_votes: List[CrossTimeframeVote] = Field(default_factory=list)
    compare_peer_symbols: List[str] = Field(default_factory=list)
    now_ts: Optional[datetime] = Field(default=None)