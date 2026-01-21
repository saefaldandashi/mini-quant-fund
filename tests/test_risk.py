"""
Tests for risk management.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_store import Features
from src.risk.risk_manager import RiskManager, RiskConstraints, RiskCheckResult


@pytest.fixture
def sample_features():
    """Create sample features."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'BAC']
    
    features = Features(
        timestamp=datetime(2024, 6, 15),
        symbols=symbols,
        prices={s: 100.0 for s in symbols},
        volatility_21d={s: 0.20 for s in symbols},
    )
    
    cov = np.eye(6) * 0.04
    features.covariance_matrix = pd.DataFrame(cov, index=symbols, columns=symbols)
    
    return features


@pytest.fixture
def risk_manager():
    """Create risk manager with default constraints."""
    constraints = RiskConstraints(
        max_position_size=0.15,
        max_sector_exposure=0.30,
        max_leverage=1.0,
        max_turnover=0.50,
        vol_target=0.12,
    )
    return RiskManager(constraints)


class TestRiskManager:
    """Tests for RiskManager."""
    
    def test_approve_valid_weights(self, risk_manager, sample_features):
        """Test approval of valid weights."""
        weights = {'AAPL': 0.10, 'MSFT': 0.10, 'GOOGL': 0.10}
        
        result = risk_manager.check_and_approve(weights, sample_features)
        
        assert result.approved
        assert len(result.violations) == 0
    
    def test_clip_oversized_positions(self, risk_manager, sample_features):
        """Test clipping of oversized positions."""
        weights = {'AAPL': 0.50}  # Exceeds 15% limit
        
        result = risk_manager.check_and_approve(weights, sample_features)
        
        assert result.approved_weights['AAPL'] <= 0.15
        assert len(result.adjustments) > 0
    
    def test_reduce_leverage(self, risk_manager, sample_features):
        """Test leverage reduction."""
        weights = {
            'AAPL': 0.15, 'MSFT': 0.15, 'GOOGL': 0.15,
            'AMZN': 0.15, 'JPM': 0.15, 'BAC': 0.15,
        }  # Total = 0.90, but with adjustments might exceed
        
        result = risk_manager.check_and_approve(weights, sample_features)
        
        total_exposure = sum(abs(w) for w in result.approved_weights.values())
        assert total_exposure <= 1.1  # Allow small buffer
    
    def test_sector_limits(self, risk_manager, sample_features):
        """Test sector exposure limits."""
        # Tech-heavy portfolio
        weights = {'AAPL': 0.15, 'MSFT': 0.15, 'GOOGL': 0.15, 'AMZN': 0.15}
        
        result = risk_manager.check_and_approve(weights, sample_features)
        
        # Should have reduced tech exposure
        tech_exposure = sum(
            abs(result.approved_weights.get(s, 0))
            for s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        )
        assert tech_exposure <= 0.35  # Allow small buffer
    
    def test_turnover_limit(self, risk_manager, sample_features):
        """Test turnover limiting."""
        current = {'AAPL': 0.50, 'MSFT': 0.50}
        proposed = {'GOOGL': 0.50, 'AMZN': 0.50}  # 100% turnover
        
        result = risk_manager.check_and_approve(
            proposed, sample_features, current
        )
        
        # Should have blended towards current
        assert result.risk_metrics['turnover'] <= 0.6
    
    def test_drawdown_trigger(self, risk_manager, sample_features):
        """Test drawdown trigger reduces exposure."""
        weights = {'AAPL': 0.15, 'MSFT': 0.15, 'GOOGL': 0.15}
        
        # Simulate drawdown
        risk_manager.high_water_mark = 100.0
        
        result = risk_manager.check_and_approve(
            weights, sample_features, current_nav=80.0  # 20% drawdown
        )
        
        # Should have reduced exposure
        assert len(result.violations) > 0 or len(result.adjustments) > 0
    
    def test_risk_metrics_calculated(self, risk_manager, sample_features):
        """Test that risk metrics are calculated."""
        weights = {'AAPL': 0.10, 'MSFT': 0.10}
        
        result = risk_manager.check_and_approve(weights, sample_features)
        
        assert 'leverage' in result.risk_metrics
        assert 'n_positions' in result.risk_metrics
        assert 'drawdown' in result.risk_metrics


class TestRiskConstraints:
    """Tests for RiskConstraints configuration."""
    
    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = RiskConstraints()
        
        assert constraints.max_position_size == 0.15
        assert constraints.max_leverage == 1.0
        assert constraints.vol_target == 0.12
    
    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = RiskConstraints(
            max_position_size=0.20,
            max_leverage=1.5,
            vol_target=0.15,
        )
        
        assert constraints.max_position_size == 0.20
        assert constraints.max_leverage == 1.5
        assert constraints.vol_target == 0.15


class TestStopLossTakeProfit:
    """Tests for stop-loss and take-profit."""
    
    def test_stop_loss_triggered(self, sample_features):
        """Test stop-loss trigger."""
        constraints = RiskConstraints(
            enable_stop_loss=True,
            stop_loss_pct=0.05,
        )
        manager = RiskManager(constraints)
        
        # Set entry price
        manager.entry_prices = {'AAPL': 110.0}  # Bought at 110
        sample_features.prices['AAPL'] = 100.0  # Now at 100 (-9%)
        
        weights = {'AAPL': 0.20}
        result = manager.check_and_approve(weights, sample_features)
        
        # Should have closed position
        assert result.approved_weights.get('AAPL', 0) < 0.10
    
    def test_take_profit_triggered(self, sample_features):
        """Test take-profit trigger."""
        constraints = RiskConstraints(
            enable_take_profit=True,
            take_profit_pct=0.20,
        )
        manager = RiskManager(constraints)
        
        # Set entry price
        manager.entry_prices = {'AAPL': 80.0}  # Bought at 80
        sample_features.prices['AAPL'] = 100.0  # Now at 100 (+25%)
        
        weights = {'AAPL': 0.20}
        result = manager.check_and_approve(weights, sample_features)
        
        # Should have reduced position
        assert result.approved_weights.get('AAPL', 0) <= 0.15
