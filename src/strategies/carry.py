"""
Carry Strategy (stub implementation for equities).
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from .base import Strategy, SignalOutput
from src.data.feature_store import Features


class CarryStrategy(Strategy):
    """
    Carry Strategy.
    For equities, uses dividend yield as carry proxy.
    This is a stub that can be extended with proper dividend data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Carry", config)
        self._required_features = ['prices', 'returns_252d']
        
        # Hardcoded dividend yields as proxy (would be loaded from data in production)
        self.dividend_yields = {
            'T': 0.065,
            'VZ': 0.065,
            'XOM': 0.035,
            'CVX': 0.04,
            'KO': 0.03,
            'PEP': 0.028,
            'JNJ': 0.03,
            'PG': 0.025,
            'MO': 0.08,
            'PM': 0.055,
        }
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate carry signals based on dividend yield."""
        weights = {}
        expected_returns = {}
        
        # Filter to symbols with carry data
        carry_assets = []
        for symbol in features.symbols:
            if symbol in self.dividend_yields:
                carry = self.dividend_yields[symbol]
                carry_assets.append((symbol, carry))
        
        if not carry_assets:
            return self._empty_signal(t)
        
        # Rank by carry and go long high carry
        carry_assets.sort(key=lambda x: x[1], reverse=True)
        
        # Top half by carry
        n_long = max(1, len(carry_assets) // 2)
        long_assets = carry_assets[:n_long]
        
        # Equal weight
        weight = 1.0 / n_long
        
        for symbol, carry in long_assets:
            weights[symbol] = weight
            expected_returns[symbol] = carry  # Carry is expected return
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=0.4,  # Low confidence as this is a stub
            explanation={
                'carry_assets': [s for s, _ in long_assets],
                'avg_carry': np.mean([c for _, c in long_assets]),
                'note': 'Stub implementation using hardcoded dividend yields',
            },
            regime_fit=0.6,
            diversification_score=0.5,
        )
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'note': 'No carry data available'},
        )
