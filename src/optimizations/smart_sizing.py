"""
Smart Position Sizing - Dynamic sizing based on Kelly Criterion and volatility.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    symbol: str
    base_weight: float
    adjusted_weight: float
    kelly_fraction: float
    vol_scalar: float
    conviction_scalar: float
    reason: str


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    
    Kelly formula: f* = (bp - q) / b
    where:
        f* = fraction of capital to bet
        b = odds received on the bet (win/loss ratio)
        p = probability of winning
        q = probability of losing (1 - p)
    
    In trading terms:
        f* = (expected_return / variance) * scaling_factor
    """
    
    def __init__(
        self,
        fractional_kelly: float = 0.25,  # Use 25% Kelly (more conservative)
        max_kelly: float = 0.30,
        min_kelly: float = 0.02,
    ):
        """
        Initialize Kelly calculator.
        
        Args:
            fractional_kelly: Fraction of full Kelly to use (0.25 = quarter Kelly)
            max_kelly: Maximum Kelly fraction allowed
            min_kelly: Minimum Kelly fraction
        """
        self.fractional_kelly = fractional_kelly
        self.max_kelly = max_kelly
        self.min_kelly = min_kelly
    
    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly fraction for a strategy/symbol.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
        
        Returns:
            Optimal position fraction
        """
        if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
            return self.min_kelly
        
        # Win/loss ratio
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        # Kelly formula
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly *= self.fractional_kelly
        
        # Clip to bounds
        kelly = max(self.min_kelly, min(self.max_kelly, kelly))
        
        return kelly
    
    def calculate_from_returns(
        self,
        returns: List[float],
    ) -> float:
        """
        Calculate Kelly from a list of returns.
        
        Args:
            returns: List of historical returns
        
        Returns:
            Optimal position fraction
        """
        if not returns or len(returns) < 5:
            return self.min_kelly
        
        wins = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        
        if not wins or not losses:
            return self.min_kelly
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        return self.calculate(win_rate, avg_win, avg_loss)


class SmartPositionSizer:
    """
    Dynamic position sizing that considers:
    - Kelly Criterion for edge-based sizing
    - Volatility scaling (reduce size in high vol)
    - Conviction weighting (higher confidence = larger size)
    - Correlation penalty (reduce for correlated positions)
    - Drawdown reduction (scale down during drawdowns)
    """
    
    def __init__(
        self,
        target_vol: float = 0.12,
        max_position: float = 0.15,
        vol_lookback: int = 21,
        use_kelly: bool = True,
        drawdown_scaling: bool = True,
    ):
        """
        Initialize smart sizer.
        
        Args:
            target_vol: Target portfolio volatility
            max_position: Maximum position size
            vol_lookback: Lookback for volatility calculation
            use_kelly: Whether to use Kelly sizing
            drawdown_scaling: Whether to reduce size during drawdowns
        """
        self.target_vol = target_vol
        self.max_position = max_position
        self.vol_lookback = vol_lookback
        self.use_kelly = use_kelly
        self.drawdown_scaling = drawdown_scaling
        
        self.kelly = KellyCriterion()
        
        # Track historical returns for Kelly calculation
        self._returns_history: Dict[str, List[float]] = {}
        self._current_drawdown: float = 0.0
    
    def size_positions(
        self,
        base_weights: Dict[str, float],
        confidences: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Tuple[Dict[str, float], List[SizingResult]]:
        """
        Calculate optimal position sizes.
        
        Args:
            base_weights: Base weights from strategy/ensemble
            confidences: Confidence level per symbol (0-1)
            volatilities: Volatility per symbol (annualized)
            correlations: Pairwise correlations (optional)
        
        Returns:
            Tuple of (adjusted_weights, sizing_details)
        """
        adjusted = {}
        details = []
        
        for symbol, base_weight in base_weights.items():
            if base_weight < 0.001:
                continue
            
            vol = volatilities.get(symbol, 0.20)
            confidence = confidences.get(symbol, 0.5)
            
            # 1. Volatility scaling
            vol_scalar = self._vol_scalar(vol)
            
            # 2. Conviction scaling
            conviction_scalar = self._conviction_scalar(confidence)
            
            # 3. Kelly sizing
            kelly_fraction = 1.0
            if self.use_kelly and symbol in self._returns_history:
                kelly_fraction = self.kelly.calculate_from_returns(
                    self._returns_history[symbol]
                )
                kelly_fraction = kelly_fraction / 0.25  # Normalize around 1.0
            
            # 4. Drawdown scaling
            dd_scalar = 1.0
            if self.drawdown_scaling and self._current_drawdown > 0.05:
                dd_scalar = max(0.5, 1.0 - self._current_drawdown)
            
            # Combine scalars
            total_scalar = vol_scalar * conviction_scalar * kelly_fraction * dd_scalar
            adjusted_weight = base_weight * total_scalar
            
            # Apply max position limit
            adjusted_weight = min(adjusted_weight, self.max_position)
            
            adjusted[symbol] = adjusted_weight
            
            details.append(SizingResult(
                symbol=symbol,
                base_weight=base_weight,
                adjusted_weight=adjusted_weight,
                kelly_fraction=kelly_fraction,
                vol_scalar=vol_scalar,
                conviction_scalar=conviction_scalar,
                reason=self._sizing_reason(vol_scalar, conviction_scalar, kelly_fraction),
            ))
        
        # Normalize to sum to 1 (or less if cash buffer)
        total = sum(adjusted.values())
        if total > 1.0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted, details
    
    def _vol_scalar(self, vol: float) -> float:
        """
        Calculate volatility scaling factor.
        Higher vol -> smaller position.
        """
        if vol <= 0:
            return 1.0
        
        # Target vol / actual vol
        scalar = self.target_vol / vol
        
        # Clip to reasonable range
        return max(0.3, min(2.0, scalar))
    
    def _conviction_scalar(self, confidence: float) -> float:
        """
        Scale position based on conviction.
        Higher confidence -> larger position.
        """
        # Map confidence to 0.7-1.3 range
        return 0.7 + 0.6 * confidence
    
    def _sizing_reason(
        self,
        vol_scalar: float,
        conviction_scalar: float,
        kelly_fraction: float,
    ) -> str:
        """Generate human-readable sizing reason."""
        reasons = []
        
        if vol_scalar < 0.8:
            reasons.append("reduced for high vol")
        elif vol_scalar > 1.2:
            reasons.append("increased for low vol")
        
        if conviction_scalar > 1.1:
            reasons.append("boosted by high confidence")
        elif conviction_scalar < 0.9:
            reasons.append("reduced for low confidence")
        
        if kelly_fraction > 1.2:
            reasons.append("Kelly suggests larger size")
        elif kelly_fraction < 0.8:
            reasons.append("Kelly suggests smaller size")
        
        return "; ".join(reasons) if reasons else "standard sizing"
    
    def update_returns(self, symbol: str, return_value: float):
        """Update returns history for Kelly calculation."""
        if symbol not in self._returns_history:
            self._returns_history[symbol] = []
        
        self._returns_history[symbol].append(return_value)
        
        # Keep last 50 returns
        if len(self._returns_history[symbol]) > 50:
            self._returns_history[symbol] = self._returns_history[symbol][-50:]
    
    def set_drawdown(self, drawdown: float):
        """Update current drawdown for scaling."""
        self._current_drawdown = max(0, drawdown)
    
    def get_sizing_stats(self) -> Dict[str, any]:
        """Get sizing statistics."""
        kelly_fractions = {}
        
        for symbol, returns in self._returns_history.items():
            if len(returns) >= 5:
                kelly_fractions[symbol] = self.kelly.calculate_from_returns(returns)
        
        return {
            'symbols_tracked': len(self._returns_history),
            'kelly_fractions': kelly_fractions,
            'current_drawdown': self._current_drawdown,
        }
