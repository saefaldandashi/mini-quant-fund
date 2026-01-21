"""
Value and Quality Tilt Strategy.
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from .base import Strategy, SignalOutput
from src.data.feature_store import Features


class ValueQualityTiltStrategy(Strategy):
    """
    Value and Quality Tilt Strategy.
    Tilts towards value (low P/E) and quality (high ROE) stocks.
    Uses sector ETF proxies or hardcoded scores as stub.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ValueQualityTilt", config)
        self._required_features = ['prices']
        
        # Hardcoded value/quality scores (would be computed from fundamentals in production)
        # Higher = more value + quality
        self.value_quality_scores = {
            # High value + quality
            'JPM': 0.8,
            'JNJ': 0.75,
            'PG': 0.7,
            'KO': 0.7,
            'XOM': 0.65,
            'CVX': 0.65,
            'WMT': 0.6,
            'BAC': 0.6,
            # Medium
            'AAPL': 0.5,
            'MSFT': 0.5,
            'GOOGL': 0.45,
            'META': 0.4,
            'HD': 0.55,
            'COST': 0.55,
            # Low value (growth)
            'TSLA': 0.2,
            'NVDA': 0.25,
            'AMZN': 0.3,
            'NFLX': 0.25,
            'AMD': 0.2,
        }
        
        self.top_n = config.get('top_n', 10) if config else 10
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate value/quality signals."""
        weights = {}
        expected_returns = {}
        
        # Get scores for available symbols
        scored_assets = []
        for symbol in features.symbols:
            if symbol in self.value_quality_scores:
                score = self.value_quality_scores[symbol]
                scored_assets.append((symbol, score))
        
        if not scored_assets:
            return self._empty_signal(t)
        
        # Sort by score descending
        scored_assets.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_assets = scored_assets[:self.top_n]
        
        # Score-weighted allocation
        total_score = sum(s for _, s in top_assets)
        
        for symbol, score in top_assets:
            weight = score / total_score if total_score > 0 else 1.0 / len(top_assets)
            weights[symbol] = weight
            expected_returns[symbol] = 0.03 + score * 0.05  # Higher score = higher expected return
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        # Value tends to underperform in strong bull markets
        regime_fit = 0.6
        if features.regime:
            from src.data.regime import TrendRegime
            if features.regime.trend == TrendRegime.STRONG_UP:
                regime_fit = 0.4  # Growth outperforms in strong bull
            elif features.regime.trend in [TrendRegime.WEAK_DOWN, TrendRegime.STRONG_DOWN]:
                regime_fit = 0.7  # Value more defensive
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=0.5,
            explanation={
                'top_picks': [s for s, _ in top_assets],
                'avg_score': np.mean([s for _, s in top_assets]),
                'note': 'Stub implementation using hardcoded scores',
            },
            regime_fit=regime_fit,
            diversification_score=0.6,
        )
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'note': 'No value/quality data available'},
        )
