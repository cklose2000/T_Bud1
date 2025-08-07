from datetime import datetime
from typing import Dict, List, Optional

from duckdb import DuckDBPyConnection

from trading_buddy.council.consistency import calculate_consistency_index
from trading_buddy.council.ci_v2_integration import CIv2Integration
from trading_buddy.core.config import settings
# from trading_buddy.detectors.indicators import compute_atr
from trading_buddy.detectors.outcomes import compute_pattern_stats
from trading_buddy.schemas.council import CIBreakdown, CouncilResponse, TradingPlan
from trading_buddy.schemas.hypothesis import Hypothesis


def check_current_context(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    context_name: str,
    event_ts: datetime,
) -> bool:
    """
    Check if a context is currently active near the event timestamp.
    """
    query = f"""
    SELECT COUNT(*) > 0
    FROM contexts
    WHERE symbol = '{symbol}'
    AND timeframe = '{timeframe}'
    AND context = '{context_name}'
    AND event_ts BETWEEN '{event_ts}' - INTERVAL '1 hour' AND '{event_ts}' + INTERVAL '1 hour'
    """
    
    result = conn.execute(query).fetchone()
    return result[0] if result else False


def get_consistency_metrics(
    conn: DuckDBPyConnection,
    base_tf: str,
    base_pattern: str,
    ctx_tf: str,
    ctx_pattern: str,
) -> Optional[Dict]:
    """
    Get consistency metrics from the matrix.
    """
    query = """
    SELECT n, exp_lift, hit_rate, stability, ci
    FROM consistency_matrix
    WHERE base_tf = ?
    AND base_pattern = ?
    AND ctx_tf = ?
    AND ctx_pattern = ?
    ORDER BY updated_at DESC
    LIMIT 1
    """
    
    result = conn.execute(query, [base_tf, base_pattern, ctx_tf, ctx_pattern]).fetchone()
    
    if result:
        return {
            "n": result[0],
            "exp_lift": result[1],
            "hit_rate": result[2],
            "stability": result[3],
            "ci": result[4],
        }
    
    return None


def calculate_council_score(
    ci_breakdown: List[CIBreakdown],
) -> float:
    """
    Calculate weighted council score based on CI values.
    """
    if not ci_breakdown:
        return 0.0
    
    # Sum of (present * ci) / sum of ci
    weighted_sum = sum(item.ci * (1.0 if item.present else 0.0) for item in ci_breakdown)
    ci_sum = sum(item.ci for item in ci_breakdown)
    
    if ci_sum == 0:
        return 0.5  # Neutral score if no CI data
    
    return weighted_sum / ci_sum


def create_trading_plan(
    conn: DuckDBPyConnection,
    symbol: str,
    timeframe: str,
    event_ts: datetime,
    council_score: float,
    pattern_attrs: Dict,
) -> TradingPlan:
    """
    Create a trading plan based on council score and pattern attributes.
    """
    # Simplified ATR calculation
    atr_query = f"""
    WITH tr_calc AS (
        SELECT 
            ts,
            GREATEST(
                high - low,
                ABS(high - LAG(close) OVER (ORDER BY ts)),
                ABS(low - LAG(close) OVER (ORDER BY ts))
            ) as true_range
        FROM bars
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        ORDER BY ts DESC
        LIMIT 15
    )
    SELECT AVG(true_range) as atr
    FROM tr_calc
    """
    
    atr_result = conn.execute(atr_query).fetchone()
    current_atr = atr_result[0] if atr_result and atr_result[0] else 0.01
    
    # Get current price (use neckline price from pattern)
    neckline = pattern_attrs.get("neckline_price", 100)
    
    # Calculate VWAP (simplified - just use current price)
    vwap = neckline
    
    # Entry logic
    entry = f"VWAP−0.1σ pullback ({vwap * 0.999:.2f}) or next close"
    
    # Stop: 1.2 × ATR below last swing low
    last_swing_low = pattern_attrs.get("low2_price", neckline * 0.98)
    stop_price = last_swing_low - (1.2 * current_atr)
    stop = f"{stop_price:.2f} (1.2×ATR below swing)"
    
    # Time stop: default lead bars
    time_stop = settings.default_horizon_bars
    
    # Size: risk budget × council score
    size = settings.default_risk_budget * council_score
    
    return TradingPlan(
        entry=entry,
        stop=stop,
        time_stop=time_stop,
        size=round(size, 4),
    )


def council_vote(
    conn: DuckDBPyConnection,
    hypothesis: Hypothesis,
) -> CouncilResponse:
    """
    Main council voting function with CI v2 integration.
    """
    # Get base pattern (assuming compound pattern for now)
    base_pattern = "double_bottom_macd_bull"
    base_tf = hypothesis.base_timeframe
    
    # Get historical stats
    stats = compute_pattern_stats(conn, base_pattern, base_tf)
    
    if not stats:
        # No historical data
        return CouncilResponse(
            n=0,
            median_fwd_ret=0.0,
            mean_fwd_ret=0.0,
            hit_rate=0.0,
            mae=0.0,
            council_score=0.0,
            ci_breakdown=[],
            plan=TradingPlan(
                entry="No data",
                stop="No data",
                time_stop=0,
                size=0.0,
            ),
            follow_ups=["Insufficient historical data for this pattern"],
        )
    
    # Initialize CI v2 integration
    ci_v2 = CIv2Integration(conn)
    use_v2 = ci_v2.use_v2
    
    # Generate request ID for logging
    import uuid
    request_id = str(uuid.uuid4())
    
    # Check current contexts and build CI breakdown
    event_ts = hypothesis.now_ts or datetime.now()
    ci_breakdown = []
    
    if use_v2:
        # Use CI v2 with uncertainty and clamping
        contexts = []
        for vote in hypothesis.cross_timeframe_votes:
            for validator in vote.validators:
                ctx_name = validator["name"]
                ctx_tf = vote.tf
                
                # Check if context is present
                present = check_current_context(
                    conn, hypothesis.symbol, ctx_tf, ctx_name, event_ts
                )
                contexts.append((ctx_tf, ctx_name, present))
        
        # Get CI v2 breakdown with uncertainty
        ci_breakdown = ci_v2.create_ci_breakdown_v2(
            hypothesis.symbol, base_tf, base_pattern, contexts
        )
        
        # Log decision for monitoring
        ci_v2.log_v2_decision(request_id, hypothesis.symbol, base_pattern, ci_breakdown)
        
    else:
        # Use legacy CI v1 (existing logic)
        for vote in hypothesis.cross_timeframe_votes:
            for validator in vote.validators:
                ctx_name = validator["name"]
                
                # Check if context is present
                present = check_current_context(
                    conn, hypothesis.symbol, vote.tf, ctx_name, event_ts
                )
                
                # Get CI metrics
                metrics = get_consistency_metrics(
                    conn, base_tf, base_pattern, vote.tf, ctx_name
                )
                
                if metrics:
                    ci_breakdown.append(CIBreakdown(
                        context_tf=vote.tf,
                        context_name=ctx_name,
                        ci=metrics["ci"],
                        present=present,
                        n=metrics["n"],
                        exp_lift=metrics["exp_lift"],
                        hit_rate=metrics["hit_rate"],
                        stability=metrics["stability"],
                    ))
    
    # Calculate council score
    council_score = calculate_council_score(ci_breakdown)
    
    # Get recent event for pattern attributes
    pattern_attrs_query = f"""
    SELECT attrs
    FROM events
    WHERE symbol = '{hypothesis.symbol}'
    AND pattern = '{base_pattern}'
    AND timeframe = '{base_tf}'
    ORDER BY event_ts DESC
    LIMIT 1
    """
    
    attrs_result = conn.execute(pattern_attrs_query).fetchone()
    pattern_attrs = attrs_result[0] if attrs_result else {}
    
    # Create trading plan
    plan = create_trading_plan(
        conn, hypothesis.symbol, base_tf, event_ts, council_score, pattern_attrs
    )
    
    # Build response
    response = CouncilResponse(
        n=stats["n"],
        median_fwd_ret=stats["median_fwd_ret"],
        mean_fwd_ret=stats["mean_fwd_ret"],
        hit_rate=stats["hit_rate"],
        mae=stats["mae"],
        council_score=council_score,
        ci_breakdown=ci_breakdown,
        plan=plan,
        follow_ups=[],
        metadata={"ci_version": "v2" if use_v2 else "v1", "request_id": request_id}
    )
    
    # Add follow-up suggestions
    if council_score < 0.5:
        response.follow_ups.append("Low council score - consider waiting for better context alignment")
    
    if not any(item.present for item in ci_breakdown):
        response.follow_ups.append("No supporting contexts detected - higher risk setup")
    
    # Add CI v2 specific rejection reasons if applicable
    if use_v2:
        rejection_reason = ci_v2.format_rejection_reason(ci_breakdown)
        if rejection_reason:
            response.follow_ups.append(rejection_reason)
    
    return response


def council_vote_whatif(
    conn: DuckDBPyConnection,
    hypothesis: Hypothesis,
    overrides: Dict[str, any],
) -> CouncilResponse:
    """
    Council vote with what-if overrides.
    """
    # Apply overrides to hypothesis
    modified_hypothesis = hypothesis.model_copy()
    
    # Override contexts if specified
    if "contexts" in overrides:
        for ctx_override in overrides["contexts"]:
            tf = ctx_override.get("tf")
            pattern = ctx_override.get("pattern")
            present = ctx_override.get("present", True)
            
            # Modify the hypothesis to simulate the override
            # This is simplified - in practice would need more sophisticated override logic
            
    # Override horizon if specified
    if "horizon" in overrides:
        modified_hypothesis.what_to_estimate.horizon_bars = overrides["horizon"]
    
    # Run normal council vote with modified hypothesis
    response = council_vote(conn, modified_hypothesis)
    
    # Add note about overrides
    response.follow_ups.append(f"What-if scenario with overrides: {overrides}")
    
    return response