"""
Risk Parity and Minimum Variance Strategy.
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
from scipy.optimize import minimize

from .base import Strategy, SignalOutput
from src.data.feature_store import Features


class RiskParityMinVarStrategy(Strategy):
    """
    Risk Parity / Minimum Variance Strategy.
    Allocates such that each asset contributes equally to portfolio risk.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("RiskParityMinVar", config)
        self._required_features = ['volatility_21d', 'covariance_matrix']
        
        self.mode = config.get('mode', 'risk_parity') if config else 'risk_parity'
        self.max_weight = config.get('max_weight', 0.15) if config else 0.15
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate risk parity or min variance weights."""
        
        # Get covariance matrix
        cov = features.covariance_matrix
        if cov is None or cov.empty:
            return self._simple_inverse_vol(features, t)
        
        # Filter to symbols with data
        symbols = [s for s in features.symbols if s in cov.columns]
        if len(symbols) < 2:
            return self._simple_inverse_vol(features, t)
        
        cov_subset = cov.loc[symbols, symbols].values
        n = len(symbols)
        
        if self.mode == 'risk_parity':
            weights = self._risk_parity_weights(cov_subset, n)
        else:
            weights = self._min_var_weights(cov_subset, n)
        
        # Build weight dict
        weight_dict = {}
        expected_returns = {}
        
        for i, symbol in enumerate(symbols):
            w = np.clip(weights[i], 0, self.max_weight)
            if w > 0.001:
                weight_dict[symbol] = w
                expected_returns[symbol] = 0.05  # Neutral expectation
        
        # Normalize
        total = sum(weight_dict.values())
        if total > 0:
            weight_dict = {k: v / total for k, v in weight_dict.items()}
        
        exp_ret = self._calculate_expected_return(weight_dict, expected_returns)
        risk = self._calculate_risk(weight_dict, cov)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weight_dict,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=0.7,
            explanation={
                'mode': self.mode,
                'n_assets': len(weight_dict),
                'max_weight': max(weight_dict.values()) if weight_dict else 0,
                'min_weight': min(weight_dict.values()) if weight_dict else 0,
            },
            regime_fit=0.7,  # Works in most regimes
            diversification_score=0.9,  # High diversification
        )
    
    def _risk_parity_weights(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Compute risk parity weights."""
        # Initial guess: equal weight
        x0 = np.ones(n) / n
        
        def risk_contribution(w):
            port_var = w @ cov @ w
            if port_var <= 0:
                return np.zeros(n)
            marginal = cov @ w
            rc = w * marginal / np.sqrt(port_var)
            return rc
        
        def objective(w):
            rc = risk_contribution(w)
            target = np.mean(rc)
            return np.sum((rc - target) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.01, self.max_weight) for _ in range(n)]
        
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 100}
            )
            if result.success:
                return result.x
        except:
            pass
        
        # Fallback to inverse vol
        vols = np.sqrt(np.diag(cov))
        inv_vol = 1.0 / np.maximum(vols, 0.01)
        return inv_vol / inv_vol.sum()
    
    def _min_var_weights(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Compute minimum variance weights."""
        x0 = np.ones(n) / n
        
        def objective(w):
            return w @ cov @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, self.max_weight) for _ in range(n)]
        
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 100}
            )
            if result.success:
                return result.x
        except:
            pass
        
        # Fallback
        vols = np.sqrt(np.diag(cov))
        inv_vol = 1.0 / np.maximum(vols, 0.01)
        return inv_vol / inv_vol.sum()
    
    def _simple_inverse_vol(self, features: Features, t: datetime) -> SignalOutput:
        """Simple inverse volatility weighting."""
        weights = {}
        vols = features.volatility_21d
        
        for symbol in features.symbols:
            vol = vols.get(symbol, 0.20)
            if vol > 0:
                weights[symbol] = 1.0 / vol
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=0.05,
            risk_estimate=0.10,
            confidence=0.5,
            explanation={'mode': 'inverse_vol_fallback'},
            regime_fit=0.6,
            diversification_score=0.8,
        )
