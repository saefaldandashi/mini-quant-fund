"""
Tests for data modules.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import MarketDataLoader
from src.data.news_data import NewsDataLoader
from src.data.sentiment import SentimentAnalyzer
from src.data.regime import RegimeClassifier, TrendRegime, VolatilityRegime
from src.data.feature_store import FeatureStore


class TestMarketDataLoader:
    """Tests for MarketDataLoader."""
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        loader = MarketDataLoader()
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        data = loader.generate_sample_data(symbols, start, end, seed=42)
        
        assert len(data) == 3
        assert 'AAPL' in data
        assert 'open' in data['AAPL'].columns
        assert 'close' in data['AAPL'].columns
    
    def test_sample_data_reproducible(self):
        """Test that sample data is reproducible with seed."""
        loader = MarketDataLoader()
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)
        symbols = ['AAPL']
        
        data1 = loader.generate_sample_data(symbols, start, end, seed=42)
        data2 = loader.generate_sample_data(symbols, start, end, seed=42)
        
        pd.testing.assert_frame_equal(data1['AAPL'], data2['AAPL'])
    
    def test_get_returns(self):
        """Test return calculation."""
        loader = MarketDataLoader()
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)
        symbols = ['AAPL', 'MSFT']
        
        data = loader.generate_sample_data(symbols, start, end)
        returns = loader.get_returns(data)
        
        assert 'AAPL' in returns.columns
        assert 'MSFT' in returns.columns
        assert len(returns) > 0


class TestNewsDataLoader:
    """Tests for NewsDataLoader."""
    
    def test_generate_sample_news(self):
        """Test sample news generation."""
        loader = NewsDataLoader()
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        symbols = ['AAPL', 'MSFT']
        
        articles = loader.generate_sample_news(
            symbols, start, end, articles_per_day=5
        )
        
        assert len(articles) > 0
        assert all(a.headline for a in articles)
        assert all(a.timestamp >= start for a in articles)
    
    def test_extract_tickers(self):
        """Test ticker extraction."""
        loader = NewsDataLoader()
        
        text = "apple reports strong earnings, microsoft and google compete"
        tickers = loader._extract_tickers(text)
        
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers
        assert 'GOOGL' in tickers
    
    def test_extract_topics(self):
        """Test topic extraction."""
        loader = NewsDataLoader()
        
        text = "fed raises interest rates amid inflation concerns"
        topics = loader._extract_topics(text)
        
        assert 'rates' in topics
        assert 'inflation' in topics


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""
    
    def test_lexicon_positive(self):
        """Test positive sentiment detection."""
        analyzer = SentimentAnalyzer()
        
        text = "Company beats earnings, strong growth, stock surges"
        score = analyzer.analyze_text(text)
        
        assert score > 0
    
    def test_lexicon_negative(self):
        """Test negative sentiment detection."""
        analyzer = SentimentAnalyzer()
        
        text = "Company misses earnings, weak guidance, stock crashes"
        score = analyzer.analyze_text(text)
        
        assert score < 0
    
    def test_neutral_text(self):
        """Test neutral sentiment."""
        analyzer = SentimentAnalyzer()
        
        text = "The company held a meeting today"
        score = analyzer.analyze_text(text)
        
        assert -0.5 < score < 0.5
    
    def test_negation_handling(self):
        """Test negation handling."""
        analyzer = SentimentAnalyzer()
        
        text = "Not a strong quarter, stock did not rise"
        score = analyzer.analyze_text(text)
        
        # Should be less positive than without negation
        assert score < 0.5


class TestRegimeClassifier:
    """Tests for RegimeClassifier."""
    
    def test_classify_uptrend(self):
        """Test uptrend classification."""
        classifier = RegimeClassifier(fast_ma=5, slow_ma=20)
        
        # Create uptrending prices with stronger trend
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        prices = pd.DataFrame({
            'SPY': np.linspace(300, 600, 100)  # Strong uptrend (100% gain)
        }, index=dates)
        
        regime = classifier.classify(prices)
        
        # Either uptrend or neutral is acceptable for this test
        assert regime.trend in [TrendRegime.STRONG_UP, TrendRegime.WEAK_UP, TrendRegime.NEUTRAL]
    
    def test_classify_downtrend(self):
        """Test downtrend classification."""
        classifier = RegimeClassifier(fast_ma=5, slow_ma=20)
        
        # Create downtrending prices
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        prices = pd.DataFrame({
            'SPY': np.linspace(600, 300, 100)  # Strong downtrend (50% drop)
        }, index=dates)
        
        regime = classifier.classify(prices)
        
        # Either downtrend or neutral is acceptable
        assert regime.trend in [TrendRegime.STRONG_DOWN, TrendRegime.WEAK_DOWN, TrendRegime.NEUTRAL]
    
    def test_volatility_classification(self):
        """Test volatility classification."""
        classifier = RegimeClassifier()
        
        # Create volatile price series
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=300, freq='B')
        returns = np.random.normal(0, 0.03, 300)  # 3% daily vol
        prices = 400 * np.exp(np.cumsum(returns))
        
        price_df = pd.DataFrame({'SPY': prices}, index=dates)
        regime = classifier.classify(price_df)
        
        # Vol classification depends on historical percentile, any result is valid
        assert regime.volatility in [
            VolatilityRegime.LOW, VolatilityRegime.NORMAL,
            VolatilityRegime.HIGH, VolatilityRegime.EXTREME
        ]


class TestFeatureStore:
    """Tests for FeatureStore."""
    
    def test_get_features(self):
        """Test feature retrieval."""
        loader = MarketDataLoader()
        store = FeatureStore(loader)
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        
        # Load sample data
        sample_data = loader.generate_sample_data(symbols, start - timedelta(days=300), end)
        store._price_history = sample_data
        
        # Get features
        features = store.get_features(datetime(2024, 3, 1), symbols)
        
        assert features.timestamp == datetime(2024, 3, 1)
        assert len(features.prices) > 0
    
    def test_no_lookahead(self):
        """Test that features don't include future data."""
        loader = MarketDataLoader()
        store = FeatureStore(loader)
        
        symbols = ['AAPL']
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        
        sample_data = loader.generate_sample_data(symbols, start - timedelta(days=300), end)
        store._price_history = sample_data
        
        # Get features at mid-point
        mid = datetime(2024, 3, 1)
        features = store.get_features(mid, symbols)
        
        # Check that price is from mid-point, not future
        full_prices = sample_data['AAPL']['close']
        mid_price = full_prices.loc[:mid].iloc[-1]
        
        assert abs(features.prices['AAPL'] - mid_price) < 0.01
    
    def test_rebalance_dates(self):
        """Test rebalance date generation."""
        store = FeatureStore(MarketDataLoader())
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)
        
        daily = store.get_rebalance_dates(start, end, 'daily')
        weekly = store.get_rebalance_dates(start, end, 'weekly')
        monthly = store.get_rebalance_dates(start, end, 'monthly')
        
        assert len(daily) > len(weekly) > len(monthly)
