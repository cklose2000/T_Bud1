import pytest

from trading_buddy.nlp.parser import NLPParser
from trading_buddy.schemas.hypothesis import Hypothesis


class TestNLPParser:
    def setup_method(self):
        self.parser = NLPParser()
    
    def test_parse_basic_query(self):
        text = "Watching the MACD on SPY — about to cross on the 5 after a double tap following a big drop. thoughts?"
        hypothesis = self.parser.parse(text)
        
        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.symbol == "SPY"
        assert hypothesis.base_timeframe == "5m"
        assert len(hypothesis.patterns) >= 2
        assert any(p.name == "double_bottom" for p in hypothesis.patterns)
        assert any(p.name == "macd_bull_cross" for p in hypothesis.patterns)
        assert len(hypothesis.context) > 0
        assert hypothesis.context[0].name == "large_drop"
    
    def test_extract_symbol(self):
        assert self.parser.extract_symbol("SPY is looking good") == "SPY"
        assert self.parser.extract_symbol("watching NVDA and QQQ") == "NVDA"
        assert self.parser.extract_symbol("no symbol here") == "SPY"  # default
    
    def test_extract_timeframe(self):
        assert self.parser.extract_timeframe("on the 5 minute") == "5m"
        assert self.parser.extract_timeframe("15m chart") == "15m"
        assert self.parser.extract_timeframe("hourly timeframe") == "60m"
        assert self.parser.extract_timeframe("no timeframe") == "5m"  # default
    
    def test_extract_patterns(self):
        patterns = self.parser.extract_patterns("double bottom with MACD cross")
        assert len(patterns) == 2
        assert patterns[0].name == "double_bottom"
        assert patterns[1].name == "macd_bull_cross"
        
        patterns = self.parser.extract_patterns("double tap forming")
        assert len(patterns) == 1
        assert patterns[0].name == "double_bottom"
    
    def test_extract_contexts(self):
        contexts = self.parser.extract_contexts("after a big drop of 2.5%")
        assert len(contexts) == 1
        assert contexts[0].name == "large_drop"
        assert contexts[0].params["threshold_pct"] == -2.5
        
        contexts = self.parser.extract_contexts("following a large drop")
        assert len(contexts) == 1
        assert contexts[0].params["threshold_pct"] == -1.5  # default
    
    def test_whatif_parsing(self):
        overrides = self.parser.parse_whatif("what if 15m not trending?")
        assert "contexts" in overrides
        assert overrides["contexts"][0]["present"] is False
        
        overrides = self.parser.parse_whatif("assume 60m regime=trend")
        assert overrides["contexts"][0]["pattern"] == "regime_trend"
        
        overrides = self.parser.parse_whatif("horizon=15")
        assert overrides["horizon"] == 15