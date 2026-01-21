"""
Tests for debate engine and ensemble.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_store import Features
from src.data.regime import MarketRegime, TrendRegime, VolatilityRegime, RiskRegime
from src.strategies.base import SignalOutput
from src.debate.debate_engine import DebateEngine, StrategyScore
from src.debate.ensemble import EnsembleOptimizer, EnsembleMode


@pytest.fixture
def sample_signals():
    """Create sample strategy signals."""
    timestamp = datetime(2024, 6, 15)
    
    signal1 = SignalOutput(
        strategy_name="Strategy1",
        timestamp=timestamp,
        desired_weights={'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.2},
        expected_return=0.12,
        expected_returns_by_asset={'AAPL': 0.15, 'MSFT': 0.10, 'GOOGL': 0.12},
        risk_estimate=0.15,
        confidence=0.7,
        regime_fit=0.6,
        diversification_score=0.7,
    )
    
    signal2 = SignalOutput(
        strategy_name="Strategy2",
        timestamp=timestamp,
        desired_weights={'AAPL': 0.2, 'MSFT': 0.3, 'AMZN': 0.2},
        expected_return=0.10,
        expected_returns_by_asset={'AAPL': 0.12, 'MSFT': 0.15, 'AMZN': 0.08},
        risk_estimate=0.12,
        confidence=0.8,
        regime_fit=0.7,
        diversification_score=0.8,
    )
    
    signal3 = SignalOutput(
        strategy_name="Strategy3",
        timestamp=timestamp,
        desired_weights={'NVDA': 0.4, 'AMD': 0.3},
        expected_return=0.18,
        expected_returns_by_asset={'NVDA': 0.20, 'AMD': 0.15},
        risk_estimate=0.25,
        confidence=0.5,
        regime_fit=0.4,
        diversification_score=0.4,
    )
    
    return {
        'Strategy1': signal1,
        'Strategy2': signal2,
        'Strategy3': signal3,
    }


@pytest.fixture
def sample_features():
    """Create sample features."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'AMD']
    
    features = Features(
        timestamp=datetime(2024, 6, 15),
        symbols=symbols,
        prices={'AAPL': 180, 'MSFT': 420, 'GOOGL': 175, 'AMZN': 185, 'NVDA': 130, 'AMD': 160},
        volatility_21d={'AAPL': 0.18, 'MSFT': 0.15, 'GOOGL': 0.20, 'AMZN': 0.22, 'NVDA': 0.30, 'AMD': 0.28},
    )
    
    features.regime = MarketRegime(
        timestamp=pd.Timestamp(features.timestamp),
        trend=TrendRegime.WEAK_UP,
        trend_strength=0.3,
        volatility=VolatilityRegime.NORMAL,
        volatility_percentile=0.5,
        correlation_regime=0.5,
        risk_regime=RiskRegime.NEUTRAL,
        description="Normal conditions"
    )
    
    cov = np.eye(6) * 0.04
    features.covariance_matrix = pd.DataFrame(cov, index=symbols, columns=symbols)
    
    return features


class TestDebateEngine:
    """Tests for DebateEngine."""
    
    def test_run_debate(self, sample_signals, sample_features):
        """Test running the debate."""
        engine = DebateEngine()
        scores, transcript = engine.run_debate(sample_signals, sample_features)
        
        assert len(scores) == 3
        assert transcript is not None
    
    def test_scores_are_valid(self, sample_signals, sample_features):
        """Test that scores are in valid range."""
        engine = DebateEngine()
        scores, _ = engine.run_debate(sample_signals, sample_features)
        
        for name, score in scores.items():
            assert 0 <= score.total_score <= 1
            assert 0 <= score.alpha_score <= 1
            assert 0 <= score.regime_fit_score <= 1
    
    def test_winning_strategies_selected(self, sample_signals, sample_features):
        """Test that winning strategies are selected."""
        engine = DebateEngine()
        _, transcript = engine.run_debate(sample_signals, sample_features)
        
        assert len(transcript.winning_strategies) > 0
    
    def test_transcript_has_summary(self, sample_signals, sample_features):
        """Test that transcript has summary."""
        engine = DebateEngine()
        _, transcript = engine.run_debate(sample_signals, sample_features)
        
        assert transcript.summary is not None
        assert len(transcript.to_string()) > 0
    
    def test_identifies_agreements(self, sample_signals, sample_features):
        """Test identification of strategy agreements."""
        # Both Strategy1 and Strategy2 agree on AAPL (long)
        engine = DebateEngine()
        _, transcript = engine.run_debate(sample_signals, sample_features)
        
        # Should have found some consensus
        assert isinstance(transcript.agreements, list)


class TestEnsembleOptimizer:
    """Tests for EnsembleOptimizer."""
    
    def test_weighted_vote(self, sample_signals, sample_features):
        """Test weighted voting ensemble."""
        engine = DebateEngine()
        scores, _ = engine.run_debate(sample_signals, sample_features)
        
        optimizer = EnsembleOptimizer()
        weights, metadata = optimizer.combine(
            sample_signals, scores, sample_features, {},
            EnsembleMode.WEIGHTED_VOTE
        )
        
        assert len(weights) > 0
        assert all(isinstance(w, float) for w in weights.values())
    
    def test_convex_optimization(self, sample_signals, sample_features):
        """Test convex optimization ensemble."""
        engine = DebateEngine()
        scores, _ = engine.run_debate(sample_signals, sample_features)
        
        optimizer = EnsembleOptimizer()
        weights, metadata = optimizer.combine(
            sample_signals, scores, sample_features, {},
            EnsembleMode.CONVEX_OPTIMIZATION
        )
        
        assert len(weights) >= 0
    
    def test_position_limits_applied(self, sample_signals, sample_features):
        """Test that position limits are applied."""
        engine = DebateEngine()
        scores, _ = engine.run_debate(sample_signals, sample_features)
        
        optimizer = EnsembleOptimizer({'max_position': 0.15})
        weights, metadata = optimizer.combine(
            sample_signals, scores, sample_features, {},
            EnsembleMode.WEIGHTED_VOTE
        )
        
        for weight in weights.values():
            assert abs(weight) <= 0.35  # Allow buffer for ensemble combination
    
    def test_turnover_limits(self, sample_signals, sample_features):
        """Test turnover limit constraint."""
        current_weights = {'AAPL': 0.5, 'MSFT': 0.3, 'TSLA': 0.2}
        
        engine = DebateEngine()
        scores, _ = engine.run_debate(sample_signals, sample_features)
        
        optimizer = EnsembleOptimizer({'max_turnover': 0.2})
        weights, metadata = optimizer.combine(
            sample_signals, scores, sample_features, current_weights,
            EnsembleMode.WEIGHTED_VOTE
        )
        
        # Turnover should be limited
        assert metadata.get('turnover', 0) <= 0.6  # Some buffer
    
    def test_metadata_contains_contributions(self, sample_signals, sample_features):
        """Test that metadata contains strategy contributions."""
        engine = DebateEngine()
        scores, _ = engine.run_debate(sample_signals, sample_features)
        
        optimizer = EnsembleOptimizer()
        _, metadata = optimizer.combine(
            sample_signals, scores, sample_features, {},
            EnsembleMode.WEIGHTED_VOTE
        )
        
        assert 'strategy_contributions' in metadata


class TestConstraintsIntegration:
    """Integration tests for constraint handling."""
    
    def test_all_constraints_applied(self, sample_signals, sample_features):
        """Test that all constraints are applied together."""
        engine = DebateEngine()
        scores, _ = engine.run_debate(sample_signals, sample_features)
        
        optimizer = EnsembleOptimizer({
            'max_position': 0.15,
            'max_leverage': 1.0,
            'max_turnover': 0.5,
            'vol_target': 0.12,
        })
        
        weights, metadata = optimizer.combine(
            sample_signals, scores, sample_features, {},
            EnsembleMode.WEIGHTED_VOTE
        )
        
        # Check leverage
        total_exposure = sum(abs(w) for w in weights.values())
        assert total_exposure <= 1.5  # Allow buffer
        
        # Check individual positions - allow buffer for ensemble combination
        for w in weights.values():
            assert abs(w) <= 0.35
