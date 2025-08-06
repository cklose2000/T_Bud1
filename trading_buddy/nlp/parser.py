import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from trading_buddy.schemas.hypothesis import (
    ContextConfig,
    CrossTimeframeVote,
    EstimationTarget,
    Hypothesis,
    PatternConfig,
)


class NLPParser:
    """
    Rule-based NLP parser for converting trader text to hypothesis JSON.
    """
    
    def __init__(self):
        # Pattern synonyms
        self.pattern_map = {
            "double tap": "double_bottom",
            "double bottom": "double_bottom",
            "db": "double_bottom",
            "macd cross": "macd_bull_cross",
            "macd bull": "macd_bull_cross",
            "macd bullish": "macd_bull_cross",
        }
        
        # Timeframe synonyms
        self.timeframe_map = {
            "1": "1m", "1m": "1m", "1min": "1m", "one minute": "1m",
            "5": "5m", "5m": "5m", "5min": "5m", "five minute": "5m",
            "15": "15m", "15m": "15m", "15min": "15m", "fifteen minute": "15m",
            "60": "60m", "60m": "60m", "1h": "60m", "hour": "60m", "hourly": "60m",
            "1d": "1d", "d": "1d", "day": "1d", "daily": "1d",
        }
        
        # Context patterns
        self.context_map = {
            "big drop": "large_drop",
            "large drop": "large_drop",
            "dropped": "large_drop",
            "fell": "large_drop",
            "trending": "trend_up_ema20",
            "trend": "trend_up_ema20",
            "uptrend": "trend_up_ema20",
        }
        
        # Symbol patterns
        self.symbol_pattern = re.compile(r'\b([A-Z]{1,5})\b')
        
        # What-if patterns
        self.whatif_patterns = [
            (r'what if (\w+) not (\w+)', 'negate_context'),
            (r'assume (\w+) regime=(\w+)', 'set_regime'),
            (r'horizon=(\d+)', 'set_horizon'),
        ]
    
    def extract_symbol(self, text: str) -> Optional[str]:
        """Extract ticker symbol from text."""
        matches = self.symbol_pattern.findall(text.upper())
        # Common tickers
        common_tickers = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "AMD", "MSFT"]
        
        for match in matches:
            if match in common_tickers:
                return match
        
        # Return first match if no common ticker found
        return matches[0] if matches else "SPY"
    
    def extract_timeframe(self, text: str) -> str:
        """Extract primary timeframe from text."""
        text_lower = text.lower()
        
        for key, tf in self.timeframe_map.items():
            if f" {key} " in f" {text_lower} " or f" {key}m" in text_lower:
                return tf
        
        return "5m"  # Default
    
    def extract_patterns(self, text: str) -> List[PatternConfig]:
        """Extract patterns from text."""
        text_lower = text.lower()
        patterns = []
        
        # Check for double bottom
        for synonym in ["double tap", "double bottom", "db"]:
            if synonym in text_lower:
                patterns.append(PatternConfig(
                    name="double_bottom",
                    params={"lookback": "2h", "tolerance_bps": 50}
                ))
                break
        
        # Check for MACD
        if any(word in text_lower for word in ["macd", "cross"]):
            patterns.append(PatternConfig(
                name="macd_bull_cross",
                params={"fast": 12, "slow": 26, "signal": 9, "within_bars": 3}
            ))
        
        # If no patterns found, assume compound pattern
        if not patterns:
            patterns = [
                PatternConfig(name="double_bottom", params={"lookback": "2h", "tolerance_bps": 50}),
                PatternConfig(name="macd_bull_cross", params={"fast": 12, "slow": 26, "signal": 9, "within_bars": 3}),
            ]
        
        return patterns
    
    def extract_contexts(self, text: str) -> List[ContextConfig]:
        """Extract context conditions from text."""
        text_lower = text.lower()
        contexts = []
        
        # Check for large drop
        if any(word in text_lower for word in ["big drop", "large drop", "dropped", "fell"]):
            # Try to extract percentage
            pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text_lower)
            threshold = -float(pct_match.group(1)) if pct_match else -1.5
            
            contexts.append(ContextConfig(
                name="large_drop",
                params={"lookback": "1d", "threshold_pct": threshold}
            ))
        
        return contexts
    
    def extract_cross_timeframe_votes(self, text: str) -> List[CrossTimeframeVote]:
        """Extract cross-timeframe validation requirements."""
        # Default cross-timeframe votes
        votes = [
            CrossTimeframeVote(
                tf="15m",
                validators=[{"name": "trend_up_ema20"}],
                lead_window="45m"
            ),
            CrossTimeframeVote(
                tf="60m",
                validators=[{"name": "regime_trend"}],
                lead_window="4h"
            ),
        ]
        
        # Could enhance this to extract from text
        text_lower = text.lower()
        if "no trend" in text_lower:
            votes = []  # Skip trend validation
        
        return votes
    
    def extract_peer_symbols(self, text: str) -> List[str]:
        """Extract peer symbols for comparison."""
        symbols = self.symbol_pattern.findall(text.upper())
        
        # Remove the primary symbol and return others
        if len(symbols) > 1:
            return symbols[1:]
        
        # Default peers
        return ["QQQ", "NVDA"]
    
    def parse_whatif(self, text: str) -> Dict[str, Any]:
        """Parse what-if conditions."""
        overrides = {}
        
        for pattern, action in self.whatif_patterns:
            match = re.search(pattern, text.lower())
            if match:
                if action == 'negate_context':
                    tf, context = match.groups()
                    overrides['contexts'] = [{
                        'tf': self.timeframe_map.get(tf, tf),
                        'pattern': context,
                        'present': False
                    }]
                elif action == 'set_regime':
                    tf, regime = match.groups()
                    overrides['contexts'] = [{
                        'tf': self.timeframe_map.get(tf, tf),
                        'pattern': f'regime_{regime}',
                        'present': True
                    }]
                elif action == 'set_horizon':
                    overrides['horizon'] = int(match.group(1))
        
        return overrides
    
    def parse(self, text: str) -> Hypothesis:
        """
        Parse trader text into Hypothesis object.
        """
        # Extract components
        symbol = self.extract_symbol(text)
        base_timeframe = self.extract_timeframe(text)
        patterns = self.extract_patterns(text)
        contexts = self.extract_contexts(text)
        cross_timeframe_votes = self.extract_cross_timeframe_votes(text)
        peer_symbols = self.extract_peer_symbols(text)
        
        # Build hypothesis
        hypothesis = Hypothesis(
            symbol=symbol,
            base_timeframe=base_timeframe,
            patterns=patterns,
            context=contexts,
            what_to_estimate=EstimationTarget(
                horizon_bars=10,
                targets=["fwd_ret", "hit_rate", "mae"]
            ),
            cross_timeframe_votes=cross_timeframe_votes,
            compare_peer_symbols=peer_symbols,
            now_ts=datetime.now(),
        )
        
        return hypothesis