"""
Enhanced referee implementation for PR5 truth harness.
Implements extract_numeric_claims and verify_claims_against_tools with tool fingerprinting.
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


def canonical_hash(obj: Any) -> str:
    """Create canonical hash of any object."""
    if isinstance(obj, dict):
        # Sort keys recursively
        sorted_obj = {k: canonical_hash(v) for k, v in sorted(obj.items())}
        json_str = json.dumps(sorted_obj, separators=(',', ':'), sort_keys=True)
    elif isinstance(obj, list):
        json_str = json.dumps([canonical_hash(item) for item in obj], separators=(',', ':'))
    elif isinstance(obj, float):
        # Round to avoid floating point precision issues
        json_str = json.dumps(round(obj, 8))
    else:
        json_str = json.dumps(obj, separators=(',', ':'))
    
    return hashlib.md5(json_str.encode()).hexdigest()


@dataclass
class NumericClaim:
    """A numeric claim found in LLM response with tool traceability."""
    claim_text: str
    value: float  # Original value as found
    value_numeric: float  # Normalized to decimal form
    unit: str
    context: str
    tool_name: Optional[str] = None
    tool_inputs_hash: Optional[str] = None
    tool_outputs_hash: Optional[str] = None
    verdict: str = "pending"
    needs_unit_disambiguation: bool = False


class EnhancedReferee:
    """Enhanced referee with comprehensive numeric claim extraction."""
    
    # Pattern definitions: (regex, unit_type, value_converter)
    PATTERNS = [
        # Explicit percentages and decimals
        (r'(\d+(?:\.\d+)?)\s*%', 'pct', lambda x: float(x) / 100),
        (r'(\d+(?:\.\d+)?)\s*(?:percent|pct)', 'pct', lambda x: float(x) / 100),
        (r'(\d+(?:\.\d+)?)\s*bps?', 'bps', lambda x: float(x) / 10000),
        
        # Statistical measures with context
        (r'(?:median|mean|average)\s*(?:of\s*|is\s*|:)?\s*([+-]?\d+(?:\.\d+)?)', 'decimal', float),
        (r'([+-]?\d+(?:\.\d+)?)\s*(?:median|mean|average)', 'decimal', float),
        (r'(?:hit\s+rate|accuracy|precision)\s*(?:of\s*|is\s*|:)?\s*(\d+(?:\.\d+)?)', 'rate', lambda x: float(x) if float(x) <= 1 else float(x)/100),
        
        # Returns and expectancy
        (r'return(?:s?)?\s*(?:of\s*)?([+-]?\d+(?:\.\d+)?)', 'return', float),
        (r'expectancy\s*(?:of\s*)?([+-]?\d+(?:\.\d+)?)', 'return', float),
        
        # Council and CI scores
        (r'(?:CI|council\s*score)\s*(?:of\s*|=\s*)?(\d+(?:\.\d+)?)', 'score', float),
        (r'(\d+(?:\.\d+)?)\s*(?:CI|council\s*score)', 'score', float),
        
        # Sample sizes
        (r'n\s*=\s*(\d+)', 'count', int),
        (r'(\d+)\s*(?:events?|samples?|occurrences?|trades?)', 'count', int),
        (r'sample\s+size\s*(?:of\s*)?(\d+)', 'count', int),
        
        # Sneaky implied metrics (PR5 focus)
        (r'(?:about|around|roughly|approximately)\s+(?:a\s+)?third(?:\s+of\s+a\s+percent)?', 'implied_pct', lambda x: 0.33/100),
        (r'(?:about|around|roughly|approximately)\s+(\d+(?:\.\d+)?)\s*(?:percent|%)', 'implied_pct', lambda x: float(x)/100),
        (r'(?:about|around|roughly|approximately)\s+(\d+(?:\.\d+)?)', 'approximation', float),
        (r'(?:about|around|roughly)\s+(\d+)\s*bps', 'implied_bps', lambda x: float(x)/10000),
        (r'(?:a\s+hair\s+under|just\s+under|slightly\s+less\s+than)\s+(\d+(?:\.\d+)?)', 'implied_under', lambda x: float(x) * 0.95),
        (r'(?:modestly|slightly)\s+positive', 'implied_small_pos', lambda x: 0.001),
        (r'(?:call\s+it|about)\s+(\d+(?:\.\d+)?)', 'approximation', float),
        (r'(?:high|strong)(?:\s*\(around\s+(\d+(?:\.\d+)?)\s*%\))?', 'implied_high', lambda x: float(x)/100 if x else 0.7),
        (r'(?:around\s+)?(?:sixty|70|seventy)\s+(?:percent|%)', 'implied_word_pct', lambda x: 0.65),  # Average of common ranges
    ]
    
    def __init__(self, conn: Optional[DuckDBPyConnection] = None):
        self.conn = conn
        self.patterns = [(re.compile(p, re.IGNORECASE), unit, converter) for p, unit, converter in self.PATTERNS]
    
    def extract_numeric_claims(self, text: str, message_id: Optional[str] = None) -> List[NumericClaim]:
        """Extract all numeric claims from text using enhanced patterns."""
        claims = []
        
        for pattern, unit_type, converter in self.patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                try:
                    # Extract the matched text and context
                    full_match = match.group(0)
                    claim_text = full_match.strip()
                    
                    # Get surrounding context (10 chars before/after)
                    start = max(0, match.start() - 10)
                    end = min(len(text), match.end() + 10)
                    context = text[start:end].strip()
                    
                    # Extract numeric value
                    if match.groups():
                        raw_value = match.group(1)
                        if raw_value is None:  # Pattern with no capture group (implied values)
                            value = converter(None)
                            value_numeric = value
                        else:
                            value = float(raw_value) if raw_value else 0
                            value_numeric = converter(raw_value)
                    else:
                        # No capture group, use converter with None
                        value = converter(None)
                        value_numeric = value
                    
                    # Check for unit disambiguation needs
                    needs_disambiguation = (
                        unit_type == 'decimal' and 0.1 < value < 10 and 
                        '%' not in claim_text and 'percent' not in claim_text.lower()
                    )
                    
                    claim = NumericClaim(
                        claim_text=claim_text,
                        value=value,
                        value_numeric=value_numeric,
                        unit=unit_type,
                        context=context,
                        needs_unit_disambiguation=needs_disambiguation
                    )
                    
                    claims.append(claim)
                    
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Failed to parse numeric claim '{full_match}': {e}")
                    continue
        
        # Deduplicate claims by claim_text
        seen = set()
        unique_claims = []
        for claim in claims:
            if claim.claim_text not in seen:
                seen.add(claim.claim_text)
                unique_claims.append(claim)
        
        logger.info(f"Extracted {len(unique_claims)} unique numeric claims from text")
        return unique_claims
    
    def verify_claims_against_tools(
        self, 
        claims: List[NumericClaim], 
        tool_results: Dict[str, Any],
        message_id: Optional[str] = None
    ) -> List[NumericClaim]:
        """Verify claims against tool outputs with fingerprinting."""
        
        verified_claims = []
        
        for claim in claims:
            # Try to find verification in tool results
            verification = self._find_tool_verification(claim, tool_results)
            
            if verification:
                claim.tool_name = verification['tool_name']
                claim.tool_inputs_hash = verification['inputs_hash']  
                claim.tool_outputs_hash = verification['outputs_hash']
                claim.verdict = "verified"
                
                # Check if auto-correction is needed
                expected_value = verification['value']
                if abs(claim.value_numeric - expected_value) > self._get_tolerance(expected_value):
                    logger.info(f"Auto-correcting claim: {claim.value_numeric} -> {expected_value}")
                    claim.value_numeric = expected_value
                    claim.verdict = "corrected"
                
            else:
                claim.verdict = "rejected"
                logger.warning(f"Unverified numeric claim: {claim.claim_text}")
            
            verified_claims.append(claim)
            
            # Log to database if available
            if self.conn and message_id:
                self._log_claim_to_db(claim, message_id)
        
        return verified_claims
    
    def _find_tool_verification(self, claim: NumericClaim, tool_results: Dict[str, Any]) -> Optional[Dict]:
        """Find matching tool output for a claim."""
        
        # Check council_vote results
        if 'council_vote' in tool_results:
            vote_data = tool_results['council_vote']
            inputs_hash = canonical_hash({k: v for k, v in vote_data.items() if k.startswith('input_')})
            outputs_hash = canonical_hash({k: v for k, v in vote_data.items() if not k.startswith('input_')})
            
            # Match different claim types
            if claim.unit in ['count', 'n'] and 'n' in vote_data:
                return {
                    'tool_name': 'council_vote',
                    'value': float(vote_data['n']),
                    'inputs_hash': inputs_hash,
                    'outputs_hash': outputs_hash
                }
            
            if claim.unit in ['rate', 'pct', 'implied_pct'] and 'hit_rate' in vote_data:
                hit_rate = vote_data['hit_rate']
                # Normalize to decimal form
                normalized = hit_rate if hit_rate <= 1 else hit_rate / 100
                return {
                    'tool_name': 'council_vote',
                    'value': normalized,
                    'inputs_hash': inputs_hash,
                    'outputs_hash': outputs_hash
                }
            
            if claim.unit in ['return', 'decimal'] and 'median_return' in vote_data:
                return {
                    'tool_name': 'council_vote', 
                    'value': float(vote_data['median_return']),
                    'inputs_hash': inputs_hash,
                    'outputs_hash': outputs_hash
                }
                
            if claim.unit in ['score', 'ci'] and 'council_score' in vote_data:
                return {
                    'tool_name': 'council_vote',
                    'value': float(vote_data['council_score']),
                    'inputs_hash': inputs_hash,
                    'outputs_hash': outputs_hash
                }
        
        # Check self_report results
        if 'self_report' in tool_results:
            report_data = tool_results['self_report']
            # Extract numbers from report text and cross-reference
            if 'summary' in report_data:
                report_claims = self.extract_numeric_claims(report_data['summary'])
                for report_claim in report_claims:
                    if self._claims_match(claim, report_claim):
                        return {
                            'tool_name': 'self_report',
                            'value': report_claim.value_numeric,
                            'inputs_hash': canonical_hash(report_data),
                            'outputs_hash': canonical_hash(report_data['summary'])
                        }
        
        # Check duckdb_query results
        if 'duckdb_query' in tool_results:
            query_data = tool_results['duckdb_query']
            if isinstance(query_data, (list, tuple)) and query_data:
                # Look for matching numeric values in query results
                for row in query_data:
                    if isinstance(row, (list, tuple)):
                        for value in row:
                            if isinstance(value, (int, float)):
                                if abs(claim.value_numeric - float(value)) <= self._get_tolerance(float(value)):
                                    return {
                                        'tool_name': 'duckdb_query',
                                        'value': float(value),
                                        'inputs_hash': canonical_hash(query_data),
                                        'outputs_hash': canonical_hash(row)
                                    }
        
        return None
    
    def _claims_match(self, claim1: NumericClaim, claim2: NumericClaim) -> bool:
        """Check if two claims refer to the same metric."""
        tolerance = self._get_tolerance(max(claim1.value_numeric, claim2.value_numeric))
        return abs(claim1.value_numeric - claim2.value_numeric) <= tolerance
    
    def _get_tolerance(self, value: float) -> float:
        """Get tolerance for value matching."""
        if abs(value) < 0.001:
            return 0.0001
        elif abs(value) < 0.01:
            return 0.001
        elif abs(value) < 1:
            return 0.01
        else:
            return max(0.1, abs(value) * 0.02)
    
    def _log_claim_to_db(self, claim: NumericClaim, message_id: str, route: str = "", symbol: str = ""):
        """Log claim to claims_log table."""
        if not self.conn:
            return
        
        try:
            self.conn.execute("""
                INSERT INTO claims_log 
                (ts, route, symbol, message_id, claim_text, value_numeric, unit,
                 tool_name, tool_inputs_hash, tool_outputs_hash, verdict)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                datetime.now(),
                route,
                symbol,
                message_id,
                claim.claim_text,
                claim.value_numeric,
                claim.unit,
                claim.tool_name,
                claim.tool_inputs_hash,
                claim.tool_outputs_hash,
                claim.verdict
            ])
        except Exception as e:
            logger.error(f"Failed to log claim to database: {e}")
    
    def log_final_response(
        self, 
        message_id: str, 
        route: str, 
        symbol: str, 
        body: str, 
        claims: List[NumericClaim]
    ):
        """Log final response with claim statistics."""
        if not self.conn:
            return
            
        try:
            verified_count = sum(1 for c in claims if c.verdict in ['verified', 'corrected'])
            unverified_count = len(claims) - verified_count
            
            self.conn.execute("""
                INSERT OR REPLACE INTO final_responses
                (message_id, ts, route, symbol, body, claims_count, verified_count, unverified_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                message_id,
                datetime.now(),
                route,
                symbol,
                body,
                len(claims),
                verified_count,
                unverified_count
            ])
        except Exception as e:
            logger.error(f"Failed to log final response: {e}")


# Public interface functions for the pytest tests
def extract_numeric_claims(text: str) -> List[Dict]:
    """Extract numeric claims - interface for tests."""
    referee = EnhancedReferee()
    claims = referee.extract_numeric_claims(text)
    
    return [
        {
            "claim_text": claim.claim_text,
            "value_numeric": claim.value_numeric,
            "unit": claim.unit,
            "context": claim.context,
            "needs_unit_disambiguation": claim.needs_unit_disambiguation
        }
        for claim in claims
    ]


def verify_claims_against_tools(claims: List[Dict], tool_hits: List[Dict]) -> List[Dict]:
    """Verify claims against tools - interface for tests."""
    referee = EnhancedReferee()
    
    # Convert test format back to NumericClaim objects
    claim_objects = [
        NumericClaim(
            claim_text=c['claim_text'],
            value=c.get('value', c['value_numeric']),
            value_numeric=c['value_numeric'],
            unit=c['unit'],
            context=c['context'],
            needs_unit_disambiguation=c.get('needs_unit_disambiguation', False)
        )
        for c in claims
    ]
    
    # Convert tool_hits to expected format
    tool_results = {}
    for hit in tool_hits:
        tool_results[hit.get('tool_name', 'unknown')] = hit.get('data', {})
    
    verified = referee.verify_claims_against_tools(claim_objects, tool_results)
    
    return [
        {
            "claim_text": claim.claim_text,
            "value_numeric": claim.value_numeric,
            "verdict": claim.verdict,
            "tool_name": claim.tool_name
        }
        for claim in verified
    ]