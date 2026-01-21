"""
Volatility-based strategies.
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from .base import Strategy, SignalOutput
from src.data.feature_store import Features
from src.data.regime import VolatilityRegime


class VolatilityRegimeVolTargetStrategy(Strategy):
    """
    Volatility Regime and Vol Targeting Strategy.
    Adjusts portfolio exposure based on realized volatility to maintain constant risk.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("VolatilityRegimeVolTarget", config)
        self._required_features = ['volatility_21d', 'volatility_63d', 'regime']
        
        # Config
        self.target_vol = config.get('target_vol', 0.12) if config else 0.12
        self.max_leverage = config.get('max_leverage', 1.5) if config else 1.5
        self.min_leverage = config.get('min_leverage', 0.5) if config else 0.5
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate vol-targeting signals."""
        weights = {}
        expected_returns = {}
        
        # Calculate current portfolio vol estimate (using simple average)
        vols = list(features.volatility_21d.values())
        if not vols:
            return self._empty_signal(t)
        
        avg_vol = np.mean(vols)
        
        # Vol scaling factor
        if avg_vol > 0:
            scale = self.target_vol / avg_vol
            scale = np.clip(scale, self.min_leverage, self.max_leverage)
        else:
            scale = 1.0
        
        # Equal weight all assets, scaled by vol target
        n_assets = len(features.symbols)
        base_weight = 1.0 / n_assets
        
        for symbol in features.symbols:
            symbol_vol = features.volatility_21d.get(symbol, avg_vol)
            
            # Inverse vol weighting within the scaled portfolio
            if symbol_vol > 0:
                vol_weight = (1.0 / symbol_vol)
            else:
                vol_weight = 1.0
            
            weights[symbol] = base_weight * scale
            expected_returns[symbol] = 0.05  # Neutral expectation
        
        # Normalize to sum to scale
        total = sum(weights.values())
        if total > 0:
            weights = {k: v * scale / total for k, v in weights.items()}
        
        # Regime adjustment
        regime_fit = 0.7
        leverage_explanation = "normal"
        
        if features.regime:
            if features.regime.volatility == VolatilityRegime.EXTREME:
                # Reduce exposure in extreme vol
                weights = {k: v * 0.5 for k, v in weights.items()}
                regime_fit = 0.5
                leverage_explanation = "reduced due to extreme volatility"
            elif features.regime.volatility == VolatilityRegime.LOW:
                regime_fit = 0.8
                leverage_explanation = "increased due to low volatility"
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        realized_vol = self._calculate_risk(weights, features.covariance_matrix)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=realized_vol,
            confidence=0.7,
            explanation={
                'avg_vol': avg_vol,
                'target_vol': self.target_vol,
                'scale_factor': scale,
                'leverage': leverage_explanation,
                'regime_vol': features.regime.volatility.value if features.regime else 'unknown',
            },
            regime_fit=regime_fit,
            diversification_score=0.8,
        )
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'error': 'No volatility data'},
        )
