"""
What-if cache system for fast counterfactual responses.
Caches council vote results with overrides for common scenarios.
"""
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from duckdb import DuckDBPyConnection

from trading_buddy.council.vote import council_vote_whatif
from trading_buddy.schemas.hypothesis import Hypothesis
from trading_buddy.schemas.council import CouncilResponse

logger = logging.getLogger(__name__)


class WhatIfCache:
    """Cache for what-if scenario results."""
    
    def __init__(self, conn: DuckDBPyConnection, default_ttl_hours: int = 8):
        """
        Initialize cache.
        
        Args:
            conn: Database connection
            default_ttl_hours: Default time-to-live for cache entries
        """
        self.conn = conn
        self.default_ttl_hours = default_ttl_hours
    
    def _hash_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Generate stable hash for hypothesis."""
        # Convert to dict and sort keys for consistent hashing
        hyp_dict = hypothesis.dict()
        
        # Remove volatile fields that don't affect the core hypothesis
        hyp_dict.pop('created_at', None)
        hyp_dict.pop('id', None)
        
        # Sort patterns and contexts for consistent ordering
        if 'patterns' in hyp_dict:
            hyp_dict['patterns'] = sorted(
                hyp_dict['patterns'], 
                key=lambda p: (p.get('name', ''), str(p.get('params', {})))
            )
        
        if 'context' in hyp_dict:
            hyp_dict['context'] = sorted(
                hyp_dict['context'],
                key=lambda c: (c.get('tf', ''), c.get('pattern', ''))
            )
        
        # Create stable JSON representation
        json_str = json.dumps(hyp_dict, sort_keys=True, separators=(',', ':'))
        
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _hash_overrides(self, overrides: Dict[str, Any]) -> str:
        """Generate stable hash for overrides."""
        # Sort all nested structures for consistency
        sorted_overrides = self._sort_dict_recursive(overrides)
        json_str = json.dumps(sorted_overrides, sort_keys=True, separators=(',', ':'))
        
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _sort_dict_recursive(self, obj: Any) -> Any:
        """Recursively sort dictionaries and lists for consistent hashing."""
        if isinstance(obj, dict):
            return {k: self._sort_dict_recursive(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            # Sort lists if they contain dicts with keys we can sort by
            if obj and isinstance(obj[0], dict):
                # Try to sort by common keys like 'tf', 'pattern', 'name'
                sort_keys = ['tf', 'pattern', 'name', 'type']
                for key in sort_keys:
                    if key in obj[0]:
                        try:
                            obj = sorted(obj, key=lambda x: x.get(key, ''))
                            break
                        except (TypeError, KeyError):
                            continue
            return [self._sort_dict_recursive(item) for item in obj]
        else:
            return obj
    
    def get(self, hypothesis: Hypothesis, overrides: Dict[str, Any]) -> Optional[CouncilResponse]:
        """
        Get cached what-if result.
        
        Args:
            hypothesis: Base trading hypothesis
            overrides: What-if scenario overrides
            
        Returns:
            Cached result or None if not found/expired
        """
        hyp_hash = self._hash_hypothesis(hypothesis)
        override_hash = self._hash_overrides(overrides)
        
        try:
            result = self.conn.execute("""
                SELECT response_json, metadata, hit_count
                FROM whatif_cache
                WHERE hypothesis_hash = ?
                AND overrides_hash = ?
                AND expires_at > CURRENT_TIMESTAMP
            """, [hyp_hash, override_hash]).fetchone()
            
            if result:
                response_data = result[0]  # JSON column
                metadata = result[1]
                hit_count = result[2]
                
                # Update hit count
                self.conn.execute("""
                    UPDATE whatif_cache
                    SET hit_count = hit_count + 1
                    WHERE hypothesis_hash = ? AND overrides_hash = ?
                """, [hyp_hash, override_hash])
                
                logger.info(f"Cache hit for {hyp_hash[:8]}+{override_hash[:8]} (hits: {hit_count + 1})")
                
                # Convert back to CouncilResponse
                # Handle case where response_data might be a JSON string
                if isinstance(response_data, str):
                    response_data = json.loads(response_data)
                
                return CouncilResponse(**response_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            return None
    
    def put(
        self, 
        hypothesis: Hypothesis, 
        overrides: Dict[str, Any], 
        response: CouncilResponse,
        ttl_hours: Optional[int] = None
    ) -> bool:
        """
        Cache what-if result.
        
        Args:
            hypothesis: Base trading hypothesis
            overrides: What-if scenario overrides
            response: Council response to cache
            ttl_hours: Custom TTL, uses default if None
            
        Returns:
            True if cached successfully
        """
        hyp_hash = self._hash_hypothesis(hypothesis)
        override_hash = self._hash_overrides(overrides)
        
        if ttl_hours is None:
            ttl_hours = self.default_ttl_hours
        
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        try:
            # Store response as JSON
            response_json = response.dict()
            
            # Metadata for debugging/analytics
            metadata = {
                "symbol": hypothesis.symbol,
                "base_timeframe": hypothesis.base_timeframe,
                "pattern_count": len(hypothesis.patterns),
                "override_keys": list(overrides.keys()),
                "cached_at": datetime.now().isoformat()
            }
            
            self.conn.execute("""
                INSERT OR REPLACE INTO whatif_cache
                (hypothesis_hash, overrides_hash, response_json, metadata, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """, [
                hyp_hash,
                override_hash,
                response_json,  # DuckDB JSON column handles this
                metadata,       # DuckDB JSON column handles this  
                expires_at
            ])
            
            logger.info(f"Cached what-if result for {hyp_hash[:8]}+{override_hash[:8]} (expires: {expires_at})")
            return True
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            return False
    
    def get_or_compute(
        self, 
        hypothesis: Hypothesis, 
        overrides: Dict[str, Any],
        ttl_hours: Optional[int] = None
    ) -> Tuple[CouncilResponse, bool]:
        """
        Get cached result or compute if not found.
        
        Args:
            hypothesis: Base trading hypothesis
            overrides: What-if scenario overrides
            ttl_hours: Custom TTL for new cache entries
            
        Returns:
            Tuple of (result, was_cached)
        """
        # Try cache first
        cached_result = self.get(hypothesis, overrides)
        if cached_result:
            return cached_result, True
        
        # Compute fresh result
        try:
            result = council_vote_whatif(self.conn, hypothesis, overrides)
            
            # Cache the result
            self.put(hypothesis, overrides, result, ttl_hours)
            
            return result, False
            
        except Exception as e:
            logger.error(f"What-if computation failed: {e}")
            # Return a default/error response
            error_response = CouncilResponse(
                n=0,
                median_return=0.0,
                hit_rate=0.0,
                council_score=0.0,
                metadata={"error": str(e), "cached": False}
            )
            return error_response, False
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries removed
        """
        try:
            result = self.conn.execute("""
                DELETE FROM whatif_cache
                WHERE expires_at <= CURRENT_TIMESTAMP
            """)
            
            deleted = result.fetchone()
            count = deleted[0] if deleted else 0
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired cache entries")
            
            return count
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats_result = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN expires_at > CURRENT_TIMESTAMP THEN 1 END) as active_entries,
                    COUNT(CASE WHEN expires_at <= CURRENT_TIMESTAMP THEN 1 END) as expired_entries,
                    AVG(hit_count) as avg_hits,
                    MAX(hit_count) as max_hits,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry
                FROM whatif_cache
            """).fetchone()
            
            if stats_result:
                return {
                    "total_entries": stats_result[0],
                    "active_entries": stats_result[1],
                    "expired_entries": stats_result[2],
                    "avg_hits": round(stats_result[3], 2) if stats_result[3] else 0,
                    "max_hits": stats_result[4],
                    "oldest_entry": stats_result[5],
                    "newest_entry": stats_result[6]
                }
            else:
                return {"total_entries": 0}
                
        except Exception as e:
            logger.error(f"Stats retrieval error: {e}")
            return {"error": str(e)}
    
    def precompute_common_scenarios(self, symbols: List[str]) -> Dict[str, int]:
        """
        Precompute common what-if scenarios for symbols.
        
        Args:
            symbols: List of symbols to precompute for
            
        Returns:
            Dict with counts of scenarios precomputed per symbol
        """
        results = {}
        
        # Common override scenarios
        common_scenarios = [
            # Timeframe alignment scenarios
            {
                "name": "60m_trend_up",
                "overrides": {
                    "contexts": [{"tf": "60m", "pattern": "trend_up", "present": True}]
                }
            },
            {
                "name": "15m_trend_down", 
                "overrides": {
                    "contexts": [{"tf": "15m", "pattern": "trend_down", "present": True}]
                }
            },
            # Volume scenarios
            {
                "name": "high_volume",
                "overrides": {
                    "contexts": [{"tf": "5m", "pattern": "volume_spike", "present": True}]
                }
            },
            # Market regime scenarios
            {
                "name": "risk_off_regime",
                "overrides": {
                    "contexts": [{"tf": "1d", "pattern": "regime_risk_off", "present": True}]
                }
            }
        ]
        
        for symbol in symbols:
            symbol_count = 0
            
            # Get recent patterns for this symbol
            recent_patterns = self.conn.execute("""
                SELECT DISTINCT pattern, timeframe
                FROM events
                WHERE symbol = ?
                AND event_ts >= CURRENT_DATE - INTERVAL 30 DAY
                LIMIT 10
            """, [symbol]).fetchall()
            
            for pattern, timeframe in recent_patterns:
                # Create base hypothesis
                try:
                    hypothesis = Hypothesis(
                        symbol=symbol,
                        base_timeframe=timeframe,
                        patterns=[{"name": pattern, "params": {}}],
                        context=[],
                        cross_timeframe_votes=[]
                    )
                    
                    # Precompute common scenarios for this pattern
                    for scenario in common_scenarios:
                        try:
                            _, was_cached = self.get_or_compute(
                                hypothesis, 
                                scenario["overrides"],
                                ttl_hours=12  # Shorter TTL for precomputed
                            )
                            
                            if not was_cached:
                                symbol_count += 1
                                
                        except Exception as e:
                            logger.warning(f"Failed to precompute {scenario['name']} for {symbol}:{pattern}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Failed to create hypothesis for {symbol}:{pattern}: {e}")
                    continue
            
            results[symbol] = symbol_count
            logger.info(f"Precomputed {symbol_count} scenarios for {symbol}")
        
        return results


def get_cache_instance(conn: DuckDBPyConnection) -> WhatIfCache:
    """Get a what-if cache instance."""
    return WhatIfCache(conn)