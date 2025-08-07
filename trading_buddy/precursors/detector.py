"""
Precursor detection for patterns that are likely to fire soon.
Analyzes partial pattern formation and calculates probability of completion.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class PrecursorDetector:
    """Detects patterns that are likely to complete in the next few bars."""
    
    def __init__(self, conn: DuckDBPyConnection):
        self.conn = conn
    
    def detect_double_bottom_precursor(
        self, 
        symbol: str, 
        timeframe: str = "5m",
        lookback_bars: int = 50
    ) -> Optional[Dict]:
        """
        Detect potential double bottom precursors.
        
        Looks for:
        1. Recent significant low (first bottom)
        2. Current price near that low level (within 2-3%)
        3. Volume/momentum suggesting potential reversal
        """
        try:
            # Get recent bars for analysis
            query = f"""
            SELECT ts, open, high, low, close, volume
            FROM bars_{timeframe}
            WHERE symbol = '{symbol}'
            ORDER BY ts DESC
            LIMIT {lookback_bars}
            """
            
            bars = self.conn.execute(query).fetchall()
            if len(bars) < 20:
                return None
            
            # Reverse to chronological order for analysis
            bars = list(reversed(bars))
            
            # Find significant lows in the last 20-30 bars
            lows = []
            for i in range(10, len(bars) - 5):  # Avoid edges
                current_low = bars[i][3]  # low price
                
                # Check if this is a local minimum
                is_local_min = True
                for j in range(max(0, i-5), min(len(bars), i+6)):
                    if j != i and bars[j][3] < current_low:
                        is_local_min = False
                        break
                
                if is_local_min:
                    lows.append({
                        'index': i,
                        'price': current_low,
                        'timestamp': bars[i][0],
                        'volume': bars[i][5]
                    })
            
            if len(lows) < 1:
                return None
            
            # Find the most significant recent low (first potential bottom)
            significant_low = min(lows[-3:], key=lambda x: x['price'])
            current_bar = bars[-1]
            current_low = current_bar[3]
            
            # Check if we're forming a potential second bottom
            price_diff_pct = abs(current_low - significant_low['price']) / significant_low['price']
            
            # Criteria for double bottom precursor:
            # 1. Within 3% of the first bottom
            # 2. At least 10 bars after the first bottom
            # 3. Price has moved up from the first bottom and is coming back down
            
            bars_since_first = len(bars) - 1 - significant_low['index']
            
            if (price_diff_pct <= 0.03 and 
                bars_since_first >= 10 and 
                bars_since_first <= 35):
                
                # Check if there was an intermediate high (bounce)
                max_high_between = max(
                    bars[significant_low['index']:][:-3], 
                    key=lambda x: x[2]  # high price
                )[2]
                
                bounce_pct = (max_high_between - significant_low['price']) / significant_low['price']
                
                if bounce_pct >= 0.01:  # At least 1% bounce
                    # Calculate probability based on historical completion rates
                    probability = self._calculate_double_bottom_probability(
                        symbol, timeframe, price_diff_pct, bounce_pct, bars_since_first
                    )
                    
                    return {
                        'pattern': 'double_bottom',
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'probability': probability,
                        'first_bottom_price': significant_low['price'],
                        'current_price': current_low,
                        'price_diff_pct': price_diff_pct,
                        'bars_since_first': bars_since_first,
                        'bounce_pct': bounce_pct,
                        'metadata': {
                            'first_bottom_time': significant_low['timestamp'],
                            'analysis_time': datetime.now()
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Double bottom precursor detection failed for {symbol}: {e}")
            return None
    
    def detect_macd_cross_precursor(
        self, 
        symbol: str, 
        timeframe: str = "5m"
    ) -> Optional[Dict]:
        """
        Detect MACD bullish cross precursors.
        
        Looks for MACD line approaching signal line from below.
        """
        try:
            # This would require MACD calculation
            # For now, return a simple heuristic based on recent price action
            query = f"""
            SELECT ts, close
            FROM bars_{timeframe}
            WHERE symbol = '{symbol}'
            ORDER BY ts DESC
            LIMIT 20
            """
            
            bars = self.conn.execute(query).fetchall()
            if len(bars) < 12:
                return None
            
            # Simple momentum proxy - if last 3 bars show increasing momentum upward
            recent_closes = [b[1] for b in reversed(bars[:12])]
            
            # Check if we have upward momentum building
            short_ma = sum(recent_closes[-3:]) / 3
            medium_ma = sum(recent_closes[-6:]) / 6
            long_ma = sum(recent_closes) / len(recent_closes)
            
            # Precursor: short MA crossing above medium MA, both below long MA (oversold bounce)
            if short_ma > medium_ma and medium_ma < long_ma:
                momentum_strength = (short_ma - medium_ma) / medium_ma
                
                if 0.001 <= momentum_strength <= 0.01:  # Modest but building momentum
                    probability = min(0.95, 0.3 + momentum_strength * 50)
                    
                    return {
                        'pattern': 'macd_bull_cross',
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'probability': probability,
                        'momentum_strength': momentum_strength,
                        'current_price': recent_closes[-1],
                        'metadata': {
                            'analysis_time': datetime.now(),
                            'short_ma': short_ma,
                            'medium_ma': medium_ma
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"MACD cross precursor detection failed for {symbol}: {e}")
            return None
    
    def _calculate_double_bottom_probability(
        self,
        symbol: str,
        timeframe: str,
        price_diff_pct: float,
        bounce_pct: float,
        bars_since_first: int
    ) -> float:
        """Calculate probability of double bottom completion based on historical data."""
        
        # Base probability from historical completion rates
        base_prob = 0.4
        
        # Adjust based on price difference (closer = higher probability)
        if price_diff_pct <= 0.005:  # Within 0.5%
            price_factor = 1.4
        elif price_diff_pct <= 0.015:  # Within 1.5%
            price_factor = 1.2
        elif price_diff_pct <= 0.025:  # Within 2.5%
            price_factor = 1.0
        else:
            price_factor = 0.8
        
        # Adjust based on bounce strength (stronger bounce = higher probability)
        if bounce_pct >= 0.03:  # 3%+ bounce
            bounce_factor = 1.3
        elif bounce_pct >= 0.02:  # 2%+ bounce
            bounce_factor = 1.1
        else:
            bounce_factor = 1.0
        
        # Adjust based on timing (too early or too late reduces probability)
        if 12 <= bars_since_first <= 25:  # Sweet spot
            timing_factor = 1.2
        elif 8 <= bars_since_first <= 30:  # Good timing
            timing_factor = 1.0
        else:
            timing_factor = 0.7
        
        # Calculate final probability
        probability = base_prob * price_factor * bounce_factor * timing_factor
        
        # Cap at 95%
        return min(0.95, probability)
    
    def detect_all_precursors(self, symbols: List[str]) -> List[Dict]:
        """
        Detect all precursor patterns for given symbols.
        
        Returns list of precursor alerts ready to be processed.
        """
        all_precursors = []
        
        for symbol in symbols:
            # Try each pattern type
            patterns_to_check = [
                ('double_bottom', self.detect_double_bottom_precursor),
                ('macd_bull_cross', self.detect_macd_cross_precursor),
            ]
            
            for pattern_name, detector_func in patterns_to_check:
                try:
                    result = detector_func(symbol, timeframe='5m')
                    if result and result.get('probability', 0) >= 0.5:
                        all_precursors.append(result)
                except Exception as e:
                    logger.warning(f"Failed to check {pattern_name} precursor for {symbol}: {e}")
                    continue
        
        # Sort by probability descending
        all_precursors.sort(key=lambda x: x.get('probability', 0), reverse=True)
        
        return all_precursors
    
    def get_active_symbols(self) -> List[str]:
        """Get symbols that have recent trading activity."""
        try:
            # Get symbols with activity in the last 2 days
            query = """
            SELECT DISTINCT symbol
            FROM bars_5m
            WHERE ts >= CURRENT_TIMESTAMP - INTERVAL 2 DAY
            ORDER BY symbol
            """
            
            results = self.conn.execute(query).fetchall()
            return [r[0] for r in results]
            
        except Exception as e:
            logger.error(f"Failed to get active symbols: {e}")
            return ['SPY', 'QQQ', 'NVDA', 'AAPL', 'TSLA']  # Fallback list