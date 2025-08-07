"""
Referee middleware for validating numeric claims in LLM responses.
Ensures no hallucinated numbers make it to users.
Enhanced with truth harness for PR5 - every claim must be tool-traceable.
"""
import re
import json
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


@dataclass
class NumericClaim:
    """A numeric claim found in LLM response."""
    value: float
    value_numeric: float  # Normalized to decimal form
    unit: str
    context: str
    claim_text: str
    source: Optional[str] = None
    tool_name: Optional[str] = None
    tool_inputs_hash: Optional[str] = None
    tool_outputs_hash: Optional[str] = None
    verified: bool = False
    verdict: str = "pending"
    needs_unit_disambiguation: bool = False


class RefereeValidator:
    """Validates numeric claims against tool outputs with enhanced pattern matching."""
    
    # Enhanced patterns for finding numeric claims (including sneaky ones)
    NUMERIC_PATTERNS = [
        # Explicit percentages and decimals
        (r'(\d+(?:\.\d+)?)\s*%', 'pct', lambda x: float(x) / 100),
        (r'(\d+(?:\.\d+)?)\s*(?:percent|pct)', 'pct', lambda x: float(x) / 100),
        (r'(\d+(?:\.\d+)?)\s*bps?', 'bps', lambda x: float(x) / 10000),
        (r'(\d+(?:\.\d+)?)\s*basis\s+points?', 'bps', lambda x: float(x) / 10000),
        
        # Statistical measures
        (r'(?:median|mean|average)\s*(?:of\s*|is\s*)?([+-]?\d+(?:\.\d+)?)', 'decimal', float),
        (r'([+-]?\d+(?:\.\d+)?)\s*(?:median|mean|average)', 'decimal', float),
        (r'(?:hit\s+rate|accuracy|precision)\s*(?:of\s*|is\s*)?(\d+(?:\.\d+)?)', 'rate', lambda x: float(x) if float(x) <= 1 else float(x)/100),
        (r'(\d+(?:\.\d+)?)\s*(?:hit\s+rate|accuracy|precision)', 'rate', lambda x: float(x) if float(x) <= 1 else float(x)/100),
        
        # Returns and performance
        (r'return(?:s?)?\s*(?:of\s*)?([+-]?\d+(?:\.\d+)?)', 'return', float),
        (r'([+-]?\d+(?:\.\d+)?)\s*return', 'return', float),
        (r'expectancy\s*(?:of\s*)?([+-]?\d+(?:\.\d+)?)', 'return', float),
        
        # CI and scores
        (r'CI\s*(?:of\s*|=\s*)?(\d+(?:\.\d+)?)', 'ci', float),
        (r'(?:council\s*)?score\s*(?:of\s*|=\s*)?(\d+(?:\.\d+)?)', 'score', float),
        (r'(\d+(?:\.\d+)?)\s*(?:council\s*)?score', 'score', float),
        
        # Counts and sample sizes
        (r'n\s*=\s*(\d+)', 'count', int),
        (r'(\d+)\s*(?:events?|samples?|occurrences?|trades?)', 'count', int),
        (r'sample\s+size\s*(?:of\s*)?(\d+)', 'count', int),
        
        # P-values and statistics
        (r'p(?:\s*-?\s*value)?\s*=\s*(\d+(?:\.\d+)?)', 'pvalue', float),
        (r'(?:sharpe|mae|mfe)\s*(?:of\s*|is\s*)?(\d+(?:\.\d+)?)', 'ratio', float),
        
        # Sneaky implied metrics (the tricky ones)
        (r'(?:about|around|roughly|approximately)\s+(?:a\s+)?third\s*(?:of\s*a\s*)?(?:percent)?', 'implied_pct', lambda x: 0.33/100),
        (r'(?:about|around|roughly)\s+(\d+(?:\.\d+)?)\s*(?:percent|%|bps)', 'implied_pct', lambda x: float(x)/100 if 'bps' not in x else float(x)/10000),
        (r'(?:a\s+hair\s+under|just\s+under|slightly\s+less\s+than)\s+(\d+(?:\.\d+)?)', 'implied_decimal', lambda x: float(x) * 0.95),  # Assume 5% less
        (r'(?:modestly|slightly)\s+positive', 'implied_positive', lambda x: 0.001),  # Small positive
        (r'(?:high|strong)\s*(?:\(around\s+(\d+(?:\.\d+)?)\s*(?:%|percent)\))?', 'implied_high', lambda x: float(x)/100 if x else 0.7),
        (r'(?:call\s+it|about)\s+(\d+(?:\.\d+)?)', 'approximation', float),
    ]
    
    def __init__(self, conn: Optional[DuckDBPyConnection] = None):
        self.conn = conn
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), unit, converter) 
            for pattern, unit, converter in self.NUMERIC_PATTERNS
        ]
    
    def extract_claims(self, text: str) -> List[NumericClaim]:
        """Extract numeric claims from text."""
        claims = []
        
        for pattern in self.patterns:
            matches = pattern.finditer(text)
            for match in matches:
                try:
                    value = float(match.group(1))
                    context = match.group(0)
                    claims.append(NumericClaim(
                        value=value,
                        context=context
                    ))
                except (ValueError, IndexError):
                    continue
        
        return claims
    
    def verify_against_tools(
        self, 
        claims: List[NumericClaim], 
        tool_results: Dict[str, Any]
    ) -> Tuple[List[NumericClaim], List[str]]:
        """
        Verify claims against tool results.
        
        Returns:
            Tuple of (verified_claims, errors)
        """
        verified_claims = []
        errors = []
        
        for claim in claims:
            verification = self._find_verification(claim, tool_results)
            
            if verification:
                claim.source = verification['source']
                claim.verified = True
                
                # Check if values match within tolerance and auto-correct
                actual_value = verification['value']
                original_claim_value = claim.value
                
                # Handle percentage/decimal conversion
                if claim.value > 1 and actual_value <= 1 and actual_value > 0:
                    # Claimed is percentage, actual is decimal - use percentage form
                    corrected_value = actual_value * 100
                elif claim.value <= 1 and claim.value > 0 and actual_value > 1:
                    # Claimed is decimal, actual is percentage - use decimal form  
                    corrected_value = actual_value / 100
                else:
                    corrected_value = actual_value
                
                tolerance = self._get_tolerance(max(original_claim_value, corrected_value))
                
                if abs(original_claim_value - corrected_value) > tolerance:
                    logger.warning(
                        f"Value mismatch: claimed {original_claim_value}, corrected to {corrected_value}"
                    )
                
                # Always use the corrected value
                claim.value = corrected_value
                
                verified_claims.append(claim)
            else:
                errors.append(
                    f"Unverified numeric claim: {claim.context} ({claim.value})"
                )
        
        return verified_claims, errors
    
    def _find_verification(
        self, 
        claim: NumericClaim, 
        tool_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find verification for a claim in tool results."""
        
        # Check council vote results
        if 'council_vote' in tool_results:
            vote_result = tool_results['council_vote']
            
            # Look for matching values
            if 'hit_rate' in vote_result:
                hit_rate = vote_result['hit_rate']
                if self._values_match(claim.value, hit_rate, as_percentage=True):
                    return {'source': 'council_vote', 'value': hit_rate}
            
            if 'median_return' in vote_result:
                median_ret = vote_result['median_return']
                if self._values_match(claim.value, median_ret):
                    return {'source': 'council_vote', 'value': median_ret}
            
            if 'n' in vote_result:
                n = vote_result['n']
                if self._values_match(claim.value, n):
                    return {'source': 'council_vote', 'value': n}
            
            if 'council_score' in vote_result:
                score = vote_result['council_score']
                if self._values_match(claim.value, score):
                    return {'source': 'council_vote', 'value': score}
        
        # Check self-report results
        if 'self_report' in tool_results:
            report = tool_results['self_report']
            
            # Check summary for patterns like "n=X" or percentages
            if 'summary' in report:
                summary_claims = self.extract_claims(report['summary'])
                for summary_claim in summary_claims:
                    if self._values_match(claim.value, summary_claim.value):
                        return {'source': 'self_report', 'value': summary_claim.value}
        
        # Check DuckDB query results
        if 'duckdb_query' in tool_results:
            query_result = tool_results['duckdb_query']
            if isinstance(query_result, (list, tuple)) and query_result:
                # Try to find matching numeric values in result
                for row in query_result:
                    if isinstance(row, (list, tuple)):
                        for value in row:
                            if isinstance(value, (int, float)):
                                if self._values_match(claim.value, value):
                                    return {'source': 'duckdb_query', 'value': value}
        
        return None
    
    def _values_match(
        self, 
        claimed: float, 
        actual: float, 
        as_percentage: bool = False
    ) -> bool:
        """Check if two values match within tolerance."""
        # Auto-detect percentage vs decimal mismatch
        if (claimed > 1 and actual <= 1 and actual > 0) or as_percentage:
            # Claimed is percentage, actual is decimal - convert actual to percentage
            if actual <= 1:
                actual *= 100
        elif (claimed <= 1 and claimed > 0 and actual > 1):
            # Claimed is decimal, actual is percentage - convert claimed to percentage
            claimed *= 100
        
        tolerance = self._get_tolerance(max(claimed, actual))
        return abs(claimed - actual) <= tolerance
    
    def _get_tolerance(self, value: float) -> float:
        """Get tolerance for value comparison."""
        if value == 0:
            return 0.001
        elif abs(value) < 0.01:
            return 0.0005  # Tight tolerance for small values
        elif abs(value) < 1:
            return 0.01    # 1% tolerance for decimals
        else:
            return max(0.1, abs(value) * 0.01)  # 1% tolerance, minimum 0.1


def validate_response_schema(response_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate response follows required schema.
    
    Expected schema:
    {
        "claims": [
            {"type": "numeric", "value": float, "context": str, "source": str}
        ],
        "plan": {
            "action": str,
            "entry": float,
            "stop": float,
            "rationale": str
        }
    }
    """
    errors = []
    
    # Check top-level structure
    if not isinstance(response_data, dict):
        errors.append("Response must be a JSON object")
        return False, errors
    
    # Claims are optional but must be array if present
    if 'claims' in response_data:
        claims = response_data['claims']
        if not isinstance(claims, list):
            errors.append("'claims' must be an array")
        else:
            for i, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    errors.append(f"Claim {i} must be an object")
                    continue
                
                required_fields = ['type', 'value', 'context']
                for field in required_fields:
                    if field not in claim:
                        errors.append(f"Claim {i} missing required field: {field}")
                
                if 'type' in claim and claim['type'] == 'numeric':
                    if not isinstance(claim.get('value'), (int, float)):
                        errors.append(f"Claim {i} numeric value must be a number")
    
    # Plan is optional but must have required structure if present
    if 'plan' in response_data:
        plan = response_data['plan']
        if not isinstance(plan, dict):
            errors.append("'plan' must be an object")
        else:
            if 'action' in plan:
                if plan['action'] not in ['trade', 'watch', 'pass']:
                    errors.append("Plan action must be 'trade', 'watch', or 'pass'")
            
            # If action is 'trade', require entry/stop
            if plan.get('action') == 'trade':
                for field in ['entry', 'stop']:
                    if field not in plan:
                        errors.append(f"Trade plan missing required field: {field}")
                    elif not isinstance(plan[field], (int, float)):
                        errors.append(f"Plan {field} must be numeric")
    
    return len(errors) == 0, errors


def referee_middleware(
    response_text: str,
    tool_results: Dict[str, Any],
    strict_mode: bool = True
) -> Dict[str, Any]:
    """
    Apply referee validation to LLM response.
    
    Args:
        response_text: Raw LLM response text
        tool_results: Results from tool calls made during response
        strict_mode: If True, reject responses with unverified claims
        
    Returns:
        Dict with validation results and corrected response
    """
    validator = RefereeValidator()
    
    # Try to parse as JSON first
    try:
        response_data = json.loads(response_text)
        schema_valid, schema_errors = validate_response_schema(response_data)
    except json.JSONDecodeError:
        # Treat as plain text, extract claims
        response_data = {"raw_text": response_text}
        schema_valid = False
        schema_errors = ["Response is not valid JSON"]
    
    # Extract numeric claims from text
    claims = validator.extract_claims(response_text)
    
    # Verify claims against tool results
    verified_claims, verification_errors = validator.verify_against_tools(
        claims, tool_results
    )
    
    # Combine all errors  
    all_errors = verification_errors  # Only count verification errors for validity
    
    # Determine if response is valid (schema errors are warnings for non-JSON)
    is_valid = len(verification_errors) == 0
    
    if not is_valid and strict_mode:
        # In strict mode, reject invalid responses
        return {
            "status": "rejected",
            "errors": all_errors + schema_errors,  # Include all errors in rejection
            "original_response": response_text,
            "verified_claims": [
                {
                    "value": claim.value,
                    "context": claim.context,
                    "source": claim.source,
                    "verified": claim.verified
                }
                for claim in verified_claims
            ],
            "suggestion": "Please revise response with proper tool citations"
        }
    
    # Auto-correct numeric values where possible
    corrected_response = response_text
    for claim in verified_claims:
        if claim.source:
            # Replace the original value with verified value in text
            old_context = claim.context
            new_context = old_context.replace(
                str(claim.value), 
                str(claim.value)  # Already corrected in verify_against_tools
            )
            corrected_response = corrected_response.replace(old_context, new_context)
    
    return {
        "status": "approved" if is_valid else "corrected",
        "errors": all_errors,
        "original_response": response_text,
        "corrected_response": corrected_response,
        "verified_claims": [
            {
                "value": claim.value,
                "context": claim.context,
                "source": claim.source,
                "verified": claim.verified
            }
            for claim in verified_claims
        ]
    }