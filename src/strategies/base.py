"""
Base strategy class and signal output types.
All strategies inherit from this base class.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '..')
from src.data.feature_store import Features


@dataclass
class SignalOutput:
    """
    Output from a strategy's signal generation.
    """
    # Strategy identification
    strategy_name: str
    timestamp: datetime
    
    # Portfolio weights (sum should be <= 1.0 for long-only, can sum to 0 for dollar-neutral)
    desired_weights: Dict[str, float] = field(default_factory=dict)
    
    # Expected return (annualized)
    expected_return: float = 0.0
    expected_returns_by_asset: Dict[str, float] = field(default_factory=dict)
    
    # Risk estimate (annualized volatility or other risk metric)
    risk_estimate: float = 0.0
    
    # Confidence in the signal (0 to 1)
    confidence: float = 0.5
    
    # Explanation of the signal
    explanation: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    regime_fit: float = 0.5  # How well this strategy fits current regime
    diversification_score: float = 0.5  # Contribution to diversification
    
    # Holding period for intraday strategies (0 = no limit / position trade)
    holding_period_minutes: int = 0  # Auto-exit after this many minutes
    
    def normalize_weights(self, target_sum: float = 1.0) -> None:
        """Normalize weights to sum to target."""
        total = sum(abs(w) for w in self.desired_weights.values())
        if total > 0:
            factor = target_sum / total
            self.desired_weights = {k: v * factor for k, v in self.desired_weights.items()}
    
    def clip_weights(self, max_weight: float = 0.2) -> None:
        """Clip individual weights to maximum."""
        self.desired_weights = {
            k: np.clip(v, -max_weight, max_weight)
            for k, v in self.desired_weights.items()
        }


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    NOW SUPPORTS INTER-STRATEGY COMMUNICATION:
    - Strategies can see what others proposed
    - Strategies can adjust based on competitor signals
    - Enables genuine "debate" between strategies
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._required_features: List[str] = []
        
        # NEW: Storage for other strategies' proposals
        self.peer_signals: Dict[str, 'SignalOutput'] = {}
        
        # NEW: Macro context (set externally)
        self.macro_features: Optional[Any] = None
        self.risk_sentiment: Optional[Any] = None
        
        # NEW: Ticker sentiment data (set externally)
        # Dict of symbol -> sentiment score (-1 to 1)
        self.ticker_sentiments: Dict[str, float] = {}
        self.sentiment_confidence: Dict[str, float] = {}
    
    @property
    def required_features(self) -> List[str]:
        """List of required feature names."""
        return self._required_features
    
    def set_peer_signals(self, signals: Dict[str, 'SignalOutput']) -> None:
        """
        Set signals from other strategies.
        Called before generate_signals to enable inter-strategy awareness.
        """
        self.peer_signals = {k: v for k, v in signals.items() if k != self.name}
    
    def set_macro_context(self, macro_features: Any, risk_sentiment: Any) -> None:
        """
        Set macro context for the strategy.
        Ensures ALL strategies have access to macro data.
        """
        self.macro_features = macro_features
        self.risk_sentiment = risk_sentiment
    
    def set_sentiment_data(
        self, 
        sentiments: Dict[str, float],
        confidence: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Set ticker sentiment data for the strategy.
        
        Args:
            sentiments: Dict of symbol -> sentiment score (-1 to 1)
            confidence: Optional dict of symbol -> confidence (0 to 1)
        """
        self.ticker_sentiments = sentiments or {}
        self.sentiment_confidence = confidence or {}
    
    def get_sentiment_adjustment(self, symbol: str, base_weight: float) -> tuple:
        """
        Get sentiment-adjusted weight for a symbol.
        
        Returns:
            (adjusted_weight, adjustment_reason)
        """
        if not self.ticker_sentiments or symbol not in self.ticker_sentiments:
            return (base_weight, None)
        
        sentiment = self.ticker_sentiments[symbol]
        confidence = self.sentiment_confidence.get(symbol, 0.5)
        
        # Only adjust if sentiment is strong and confident
        if abs(sentiment) < 0.3 or confidence < 0.4:
            return (base_weight, None)
        
        # Calculate adjustment factor
        # Strong positive sentiment = boost longs, reduce shorts
        # Strong negative sentiment = reduce longs, boost shorts
        adjustment_factor = 1.0 + (sentiment * confidence * 0.2)  # Max 20% adjustment
        
        adjusted = base_weight * adjustment_factor
        
        # Cap the adjustment
        max_single = 0.15
        adjusted = max(-max_single, min(max_single, adjusted))
        
        reason = f"Sentiment {sentiment:+.2f} (conf: {confidence:.0%}) → {adjustment_factor:.2f}x"
        
        return (adjusted, reason)
    
    def get_peer_consensus(self, symbol: str) -> tuple:
        """
        Get consensus view on a symbol from peer strategies.
        Returns (avg_weight, agreement_score) where:
        - avg_weight: average weight proposed by peers
        - agreement_score: 0-1 how much peers agree
        """
        if not self.peer_signals:
            return (0.0, 0.0)
        
        weights = []
        for signal in self.peer_signals.values():
            if symbol in signal.desired_weights:
                weights.append(signal.desired_weights[symbol])
        
        if not weights:
            return (0.0, 0.0)
        
        avg_weight = np.mean(weights)
        
        # Agreement: inverse of variance (higher = more agreement)
        if len(weights) > 1:
            variance = np.var(weights)
            agreement = 1.0 / (1.0 + variance * 100)  # Scale for 0-1
        else:
            agreement = 0.5
        
        return (avg_weight, agreement)
    
    def should_follow_consensus(self, symbol: str, my_weight: float) -> tuple:
        """
        Determine if strategy should adjust to follow/oppose consensus.
        Returns (adjusted_weight, reason)
        """
        avg_weight, agreement = self.get_peer_consensus(symbol)
        
        # If strong agreement and I'm contrarian, consider adjusting
        if agreement > 0.7 and abs(my_weight - avg_weight) > 0.1:
            # Strong consensus, I'm different
            if abs(avg_weight) > abs(my_weight):
                # Peers are more confident - slightly adjust toward consensus
                adjusted = my_weight * 0.7 + avg_weight * 0.3
                return (adjusted, f"Adjusted toward peer consensus ({agreement:.0%} agreement)")
        
        # If low agreement and I'm confident, can maintain position
        if agreement < 0.3 and abs(my_weight) > 0.05:
            return (my_weight, "Maintaining contrarian position (low peer agreement)")
        
        return (my_weight, None)
    
    # =========================================================================
    # CROSS-ASSET SIGNAL HELPERS
    # =========================================================================
    
    def get_cross_asset_signal(self, features: Features, signal_type: str) -> float:
        """
        Get a cross-asset signal from features.
        
        Signal types: 'oil', 'gold', 'dxy', 'europe', 'credit', 'china'
        Returns: Signal strength (-1 to 1, negative=bearish, positive=bullish)
        """
        if not hasattr(features, 'cross_asset_signals') or not features.cross_asset_signals:
            return 0.0
        
        signal_map = {
            'oil': 'oil_signal',
            'gold': 'gold_signal',
            'dxy': 'dxy_signal',
            'dollar': 'dxy_signal',
            'europe': 'europe_lead',
            'credit': 'credit_signal',
            'china': 'china_signal',
        }
        
        key = signal_map.get(signal_type.lower(), signal_type)
        return features.cross_asset_signals.get(key, 0.0)
    
    def get_oil_signal(self, features: Features) -> float:
        """Get oil price signal. Positive = oil rising, negative = falling."""
        return self.get_cross_asset_signal(features, 'oil')
    
    def get_dxy_signal(self, features: Features) -> float:
        """Get dollar strength signal. Positive = strong dollar, negative = weak."""
        return self.get_cross_asset_signal(features, 'dxy')
    
    def get_europe_lead_signal(self, features: Features) -> float:
        """Get Europe premarket lead signal."""
        return self.get_cross_asset_signal(features, 'europe')
    
    def get_credit_signal(self, features: Features) -> float:
        """Get credit stress signal. Negative = credit stress."""
        return self.get_cross_asset_signal(features, 'credit')
    
    def get_cross_asset_return(self, features: Features, symbol: str) -> float:
        """Get the 1-day return of a cross-asset symbol."""
        if not hasattr(features, 'cross_asset_returns') or not features.cross_asset_returns:
            return 0.0
        return features.cross_asset_returns.get(symbol, 0.0)
    
    def adjust_weight_for_cross_asset(
        self,
        symbol: str,
        base_weight: float,
        features: Features,
    ) -> tuple:
        """
        Adjust a position weight based on cross-asset signals.
        
        Returns (adjusted_weight, reason)
        """
        adjustment = 1.0
        reasons = []
        
        # Energy stocks affected by oil
        energy_stocks = ['XOM', 'CVX', 'SLB', 'OXY', 'COP', 'XLE']
        if symbol in energy_stocks:
            oil_signal = self.get_oil_signal(features)
            if abs(oil_signal) > 0.1:
                adjustment *= (1 + oil_signal * 0.3)  # Up to 30% adjustment
                reasons.append(f"Oil signal {oil_signal:+.2f}")
        
        # Multinational stocks affected by DXY
        multinational_stocks = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'META', 'NVDA', 'AMZN']
        if symbol in multinational_stocks:
            dxy_signal = self.get_dxy_signal(features)
            if abs(dxy_signal) > 0.1:
                # Strong dollar hurts multinationals
                adjustment *= (1 - dxy_signal * 0.2)  # Up to 20% adjustment
                reasons.append(f"DXY signal {dxy_signal:+.2f}")
        
        # Financial stocks affected by credit
        financial_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'XLF']
        if symbol in financial_stocks:
            credit_signal = self.get_credit_signal(features)
            if abs(credit_signal) > 0.1:
                adjustment *= (1 + credit_signal * 0.25)  # Up to 25% adjustment
                reasons.append(f"Credit signal {credit_signal:+.2f}")
        
        adjusted_weight = base_weight * adjustment
        reason = ", ".join(reasons) if reasons else None
        
        return (adjusted_weight, reason)
    
    def should_short_based_on_cross_asset(
        self,
        symbol: str,
        features: Features,
    ) -> tuple:
        """
        Check if cross-asset signals suggest shorting this symbol.
        
        Returns (should_short: bool, strength: float, reason: str)
        """
        # Energy shorts on oil decline
        energy_stocks = ['XOM', 'CVX', 'SLB', 'OXY', 'COP']
        if symbol in energy_stocks:
            oil_signal = self.get_oil_signal(features)
            if oil_signal < -0.2:
                return (True, abs(oil_signal), f"Oil declining ({oil_signal:.2f})")
        
        # Multinational shorts on strong dollar
        multinational_stocks = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'META', 'AMZN']
        if symbol in multinational_stocks:
            dxy_signal = self.get_dxy_signal(features)
            if dxy_signal > 0.2:
                return (True, dxy_signal, f"Strong dollar ({dxy_signal:.2f})")
        
        # Financial shorts on credit stress
        financial_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS']
        if symbol in financial_stocks:
            credit_signal = self.get_credit_signal(features)
            if credit_signal < -0.3:
                return (True, abs(credit_signal), f"Credit stress ({credit_signal:.2f})")
        
        return (False, 0.0, None)
    
    @abstractmethod
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """
        Generate trading signals based on features.
        
        Args:
            features: Feature object with all available data
            t: Current timestamp
            
        Returns:
            SignalOutput with desired weights and metadata
        """
        pass
    
    def validate_features(self, features: Features) -> bool:
        """Check if required features are available."""
        for feat in self.required_features:
            if not hasattr(features, feat):
                return False
            val = getattr(features, feat)
            if val is None or (isinstance(val, dict) and len(val) == 0):
                return False
        return True
    
    def _calculate_expected_return(
        self,
        weights: Dict[str, float],
        expected_returns: Dict[str, float]
    ) -> float:
        """Calculate portfolio expected return."""
        total = 0.0
        for symbol, weight in weights.items():
            if symbol in expected_returns:
                total += weight * expected_returns[symbol]
        return total
    
    def _calculate_risk(
        self,
        weights: Dict[str, float],
        cov_matrix: Optional[pd.DataFrame]
    ) -> float:
        """Calculate portfolio risk (volatility)."""
        if cov_matrix is None:
            return 0.15  # Default assumption
        
        symbols = [s for s in weights.keys() if s in cov_matrix.columns]
        if len(symbols) == 0:
            return 0.15
        
        w = np.array([weights.get(s, 0) for s in symbols])
        cov = cov_matrix.loc[symbols, symbols].values
        
        try:
            var = w @ cov @ w
            return np.sqrt(max(0, var))
        except:
            return 0.15
