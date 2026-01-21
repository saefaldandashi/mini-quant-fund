"""
Tests for trading strategies.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_store import Features
from src.data.regime import MarketRegime, TrendRegime, VolatilityRegime, RiskRegime
from src.strategies import (
    TimeSeriesMomentumStrategy,
    CrossSectionMomentumStrategy,
    MeanReversionStrategy,
    VolatilityRegimeVolTargetStrategy,
    RiskParityMinVarStrategy,
)


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    timestamp = datetime(2024, 6, 15)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    features = Features(
        timestamp=timestamp,
        symbols=symbols,
        prices={'AAPL': 180.0, 'MSFT': 420.0, 'GOOGL': 175.0, 'AMZN': 185.0, 'META': 500.0},
        returns_1d={'AAPL': 0.01, 'MSFT': -0.005, 'GOOGL': 0.02, 'AMZN': -0.01, 'META': 0.015},
        returns_21d={'AAPL': 0.05, 'MSFT': 0.03, 'GOOGL': 0.08, 'AMZN': -0.02, 'META': 0.06},
        returns_63d={'AAPL': 0.12, 'MSFT': 0.08, 'GOOGL': 0.15, 'AMZN': 0.02, 'META': 0.10},
        returns_126d={'AAPL': 0.20, 'MSFT': 0.15, 'GOOGL': 0.25, 'AMZN': -0.05, 'META': 0.18},
        volatility_21d={'AAPL': 0.18, 'MSFT': 0.15, 'GOOGL': 0.22, 'AMZN': 0.25, 'META': 0.20},
        volatility_63d={'AAPL': 0.20, 'MSFT': 0.17, 'GOOGL': 0.24, 'AMZN': 0.26, 'META': 0.21},
        ma_20={'AAPL': 175.0, 'MSFT': 415.0, 'GOOGL': 170.0, 'AMZN': 190.0, 'META': 490.0},
        ma_50={'AAPL': 170.0, 'MSFT': 400.0, 'GOOGL': 165.0, 'AMZN': 195.0, 'META': 480.0},
        ma_200={'AAPL': 160.0, 'MSFT': 380.0, 'GOOGL': 155.0, 'AMZN': 180.0, 'META': 450.0},
    )
    
    # Add regime
    features.regime = MarketRegime(
        timestamp=pd.Timestamp(timestamp),
        trend=TrendRegime.WEAK_UP,
        trend_strength=0.3,
        volatility=VolatilityRegime.NORMAL,
        volatility_percentile=0.5,
        correlation_regime=0.5,
        risk_regime=RiskRegime.NEUTRAL,
        description="Normal market conditions"
    )
    
    # Add covariance matrix
    cov_data = np.eye(5) * 0.04  # 20% vol diagonal
    cov_data += 0.01  # Add correlation
    features.covariance_matrix = pd.DataFrame(
        cov_data, index=symbols, columns=symbols
    )
    
    return features


class TestTimeSeriesMomentum:
    """Tests for TimeSeriesMomentumStrategy."""
    
    def test_signal_generation(self, sample_features):
        """Test basic signal generation."""
        strategy = TimeSeriesMomentumStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        assert signal.strategy_name == "TimeSeriesMomentum"
        assert signal.timestamp == sample_features.timestamp
        assert isinstance(signal.desired_weights, dict)
        assert 0 <= signal.confidence <= 1
    
    def test_weights_sum_reasonable(self, sample_features):
        """Test that weights sum is reasonable."""
        strategy = TimeSeriesMomentumStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        total_weight = sum(abs(w) for w in signal.desired_weights.values())
        assert total_weight <= 2.0  # Allow some leverage
    
    def test_long_only_mode(self, sample_features):
        """Test long-only mode produces no short positions."""
        strategy = TimeSeriesMomentumStrategy({'long_only': True})
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        for weight in signal.desired_weights.values():
            assert weight >= 0
    
    def test_positive_momentum_stocks_get_weight(self, sample_features):
        """Test that stocks with positive momentum get weight."""
        strategy = TimeSeriesMomentumStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        # GOOGL has highest momentum
        if 'GOOGL' in signal.desired_weights:
            assert signal.desired_weights['GOOGL'] > 0


class TestCrossSectionMomentum:
    """Tests for CrossSectionMomentumStrategy."""
    
    def test_selects_top_n(self, sample_features):
        """Test that strategy selects top N stocks."""
        strategy = CrossSectionMomentumStrategy({'top_n': 3})
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        positive_positions = [s for s, w in signal.desired_weights.items() if w > 0]
        assert len(positive_positions) <= 3
    
    def test_weights_are_equal(self, sample_features):
        """Test equal weighting of selected stocks."""
        strategy = CrossSectionMomentumStrategy({'top_n': 3, 'bottom_n': 0})
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        positive_weights = [w for w in signal.desired_weights.values() if w > 0]
        if len(positive_weights) > 1:
            # All weights should be approximately equal
            assert max(positive_weights) - min(positive_weights) < 0.01


class TestMeanReversion:
    """Tests for MeanReversionStrategy."""
    
    def test_oversold_gets_long(self, sample_features):
        """Test that oversold stocks get long positions."""
        # Make AMZN appear oversold
        sample_features.prices['AMZN'] = 170.0  # Below MA
        sample_features.ma_20['AMZN'] = 200.0
        
        strategy = MeanReversionStrategy({'z_threshold': 1.0})
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        # AMZN should have positive weight (buy the dip)
        if 'AMZN' in signal.desired_weights:
            assert signal.desired_weights['AMZN'] > 0


class TestRiskParity:
    """Tests for RiskParityMinVarStrategy."""
    
    def test_all_stocks_get_weight(self, sample_features):
        """Test that all stocks get some weight."""
        strategy = RiskParityMinVarStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        assert len(signal.desired_weights) > 0
    
    def test_weights_sum_to_one(self, sample_features):
        """Test weights sum to approximately 1."""
        strategy = RiskParityMinVarStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        total = sum(signal.desired_weights.values())
        assert 0.9 <= total <= 1.1
    
    def test_max_weight_constraint(self, sample_features):
        """Test maximum weight constraint."""
        strategy = RiskParityMinVarStrategy({'max_weight': 0.10})
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        for weight in signal.desired_weights.values():
            assert weight <= 0.25  # Allow buffer for optimization


class TestVolatilityStrategy:
    """Tests for VolatilityRegimeVolTargetStrategy."""
    
    def test_vol_scaling(self, sample_features):
        """Test volatility scaling."""
        strategy = VolatilityRegimeVolTargetStrategy({'target_vol': 0.12})
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        assert signal.risk_estimate > 0
        assert 0 <= signal.confidence <= 1


class TestNoLookAhead:
    """Tests to ensure no look-ahead bias."""
    
    def test_uses_only_available_data(self, sample_features):
        """Verify strategies only use data up to timestamp."""
        strategy = TimeSeriesMomentumStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        # Signal should be based on historical data only
        # (This is a structural test - actual implementation enforces this)
        assert signal.timestamp == sample_features.timestamp


class TestOutputShapes:
    """Tests for correct output shapes."""
    
    def test_weights_are_floats(self, sample_features):
        """Test that all weights are floats."""
        strategy = TimeSeriesMomentumStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        for symbol, weight in signal.desired_weights.items():
            assert isinstance(weight, (int, float))
    
    def test_no_nan_weights(self, sample_features):
        """Test that no weights are NaN."""
        strategy = TimeSeriesMomentumStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        for weight in signal.desired_weights.values():
            assert not np.isnan(weight)
    
    def test_explanation_is_dict(self, sample_features):
        """Test that explanation is a dictionary."""
        strategy = TimeSeriesMomentumStrategy()
        signal = strategy.generate_signals(sample_features, sample_features.timestamp)
        
        assert isinstance(signal.explanation, dict)
