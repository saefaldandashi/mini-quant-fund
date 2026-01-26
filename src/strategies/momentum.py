"""
Momentum-based strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import Strategy, SignalOutput
from src.data.feature_store import Features
from src.data.regime import TrendRegime, RiskRegime


class TimeSeriesMomentumStrategy(Strategy):
    """
    Time-Series Momentum Strategy.
    Goes long assets with positive momentum, short assets with negative momentum.
    
    NOW WITH MACRO AWARENESS:
    - Reduces positions in high geopolitical risk environments
    - Adjusts based on overall risk sentiment
    - Considers peer consensus on positions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TimeSeriesMomentum", config)
        self._required_features = ['returns_126d', 'volatility_21d', 'regime']
        
        # Config
        self.lookback = config.get('lookback', 126) if config else 126
        self.vol_target = config.get('vol_target', 0.10) if config else 0.10
        self.long_only = config.get('long_only', True) if config else True
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate time-series momentum signals with macro awareness."""
        weights = {}
        expected_returns = {}
        explanations = {}
        
        # USE MACRO FEATURES FOR POSITION SIZING
        macro_multiplier = 1.0
        macro_reason = None
        
        if self.macro_features is not None:
            # Reduce momentum exposure in high-risk environments
            geo_risk = getattr(self.macro_features, 'geopolitical_risk_index', 0)
            fin_stress = getattr(self.macro_features, 'financial_stress_index', 0)
            
            if geo_risk > 0.5 or fin_stress > 0.5:
                macro_multiplier = 0.5  # Cut exposure in half
                macro_reason = f"High risk (geo={geo_risk:.2f}, stress={fin_stress:.2f})"
            elif geo_risk < -0.3 and fin_stress < -0.3:
                macro_multiplier = 1.2  # Increase exposure
                macro_reason = "Low risk environment"
        
        if self.risk_sentiment is not None:
            # Adjust based on overall risk sentiment
            if hasattr(self.risk_sentiment, 'equity_bias'):
                if self.risk_sentiment.equity_bias < -0.3:
                    macro_multiplier *= 0.7  # Risk-off, reduce
                    macro_reason = f"Risk-off sentiment (equity_bias={self.risk_sentiment.equity_bias:.2f})"
        
        # Use appropriate return horizon
        if self.lookback <= 21:
            returns = features.returns_21d
        elif self.lookback <= 63:
            returns = features.returns_63d
        else:
            returns = features.returns_126d
        
        vol = features.volatility_21d
        
        for symbol in features.symbols:
            if symbol not in returns or symbol not in vol:
                continue
            
            ret = returns[symbol]
            asset_vol = vol.get(symbol, 0.20)
            
            if asset_vol <= 0:
                continue
            
            # Signal: sign of momentum, scaled by inverse vol
            if ret > 0:
                raw_weight = 1.0
            elif ret < 0 and not self.long_only:
                raw_weight = -1.0
            else:
                raw_weight = 0.0
            
            # Vol-scale
            if asset_vol > 0:
                weight = raw_weight * (self.vol_target / asset_vol)
            else:
                weight = 0.0
            
            # APPLY MACRO ADJUSTMENT
            weight *= macro_multiplier
            
            # CHECK PEER CONSENSUS (if available)
            peer_avg, peer_agreement = self.get_peer_consensus(symbol)
            consensus_adj = None
            if peer_agreement > 0.6 and abs(weight) > 0.01:
                # Strong peer agreement - consider following
                if (peer_avg > 0 and weight > 0) or (peer_avg < 0 and weight < 0):
                    # Agreement - slightly boost
                    weight *= 1.1
                    consensus_adj = "boosted (peer agreement)"
            
            weights[symbol] = weight
            # Annualize expected return but CAP at realistic levels
            # Past momentum doesn't linearly predict future returns
            raw_exp_ret = ret * 252 / self.lookback
            # Apply decay factor (momentum has diminishing predictive power)
            discounted = raw_exp_ret * 0.3  # 30% decay factor
            # Cap at +/- 50% annualized
            expected_returns[symbol] = max(-0.50, min(0.50, discounted))
            explanations[symbol] = {
                'momentum': ret,
                'volatility': asset_vol,
                'raw_signal': raw_weight,
                'macro_multiplier': macro_multiplier,
                'macro_reason': macro_reason,
                'consensus_adj': consensus_adj,
            }
        
        # Normalize
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 1.0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Regime fit
        regime_fit = 0.5
        if features.regime:
            if features.regime.trend in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
                regime_fit = 0.8
            elif features.regime.trend == TrendRegime.NEUTRAL:
                regime_fit = 0.3
        
        # Calculate portfolio metrics
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=0.6,
            explanation={
                'top_picks': sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5],
                'regime_trend': features.regime.trend.value if features.regime else 'unknown',
                'details': explanations,
            },
            regime_fit=regime_fit,
            diversification_score=0.6,
        )


class CrossSectionMomentumStrategy(Strategy):
    """
    Cross-Sectional Momentum Strategy.
    Ranks assets by momentum and goes long winners, short losers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CrossSectionMomentum", config)
        self._required_features = ['returns_126d', 'returns_21d', 'volatility_21d']
        
        # Config
        self.lookback = config.get('lookback', 126) if config else 126
        self.top_n = config.get('top_n', 5) if config else 5
        self.bottom_n = config.get('bottom_n', 0) if config else 0  # 0 = long only
        self.skip_recent = config.get('skip_recent', 21) if config else 21  # Skip last month
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate cross-sectional momentum signals."""
        # Calculate momentum score: 12-1 month momentum (skip last month)
        momentum_scores = {}
        
        for symbol in features.symbols:
            ret_long = features.returns_126d.get(symbol)
            ret_short = features.returns_21d.get(symbol)
            
            if ret_long is not None and ret_short is not None:
                # 12-1 month momentum
                momentum = ret_long - ret_short
                momentum_scores[symbol] = momentum
        
        if len(momentum_scores) == 0:
            return self._empty_signal(t)
        
        # Rank
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top and bottom
        winners = ranked[:self.top_n]
        losers = ranked[-self.bottom_n:] if self.bottom_n > 0 else []
        
        # Equal weight
        weight_per_stock = 1.0 / (self.top_n + self.bottom_n) if self.bottom_n > 0 else 1.0 / self.top_n
        
        weights = {}
        expected_returns = {}
        
        for symbol, score in winners:
            weights[symbol] = weight_per_stock
            # Cap expected return at realistic levels (30% discount + 50% cap)
            expected_returns[symbol] = max(-0.50, min(0.50, score * 0.3))
        
        for symbol, score in losers:
            weights[symbol] = -weight_per_stock
            # For shorts, negative score → positive expected return (capped)
            expected_returns[symbol] = max(-0.50, min(0.50, -score * 0.3))
        
        # Regime fit
        regime_fit = 0.6
        if features.regime:
            if features.regime.risk_regime == RiskRegime.RISK_ON:
                regime_fit = 0.7
            elif features.regime.risk_regime == RiskRegime.RISK_OFF:
                regime_fit = 0.4
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=0.65,
            explanation={
                'winners': [s for s, _ in winners],
                'losers': [s for s, _ in losers],
                'momentum_spread': winners[0][1] - ranked[-1][1] if len(ranked) > 1 else 0,
            },
            regime_fit=regime_fit,
            diversification_score=0.7,
        )
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'error': 'Insufficient data'},
        )
