"""
ML Meta-Ensemble Strategy.
Learns optimal weights over other strategies.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque

from .base import Strategy, SignalOutput
from src.data.feature_store import Features


class MLMetaEnsembleStrategy(Strategy):
    """
    ML Meta-Ensemble Strategy.
    Learns to weight other strategies based on their recent performance.
    Uses a simple online learning approach.
    """
    
    def __init__(
        self,
        strategies: List[Strategy],
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__("MLMetaEnsemble", config)
        self._required_features = []  # Uses other strategies' outputs
        
        self.strategies = strategies
        self.strategy_names = [s.name for s in strategies]
        
        # Config
        self.learning_rate = config.get('learning_rate', 0.1) if config else 0.1
        self.lookback = config.get('lookback', 20) if config else 20
        self.min_weight = config.get('min_weight', 0.05) if config else 0.05
        
        # State
        self.strategy_weights = {name: 1.0 / len(strategies) for name in self.strategy_names}
        self.performance_history: Dict[str, deque] = {
            name: deque(maxlen=self.lookback) for name in self.strategy_names
        }
        self.last_signals: Dict[str, SignalOutput] = {}
        self.last_prices: Dict[str, float] = {}
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate ensemble signals from underlying strategies."""
        
        # Update performance based on realized returns (if we have history)
        if self.last_prices and features.prices:
            self._update_performance(features)
        
        # Get signals from all strategies
        strategy_signals: Dict[str, SignalOutput] = {}
        
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signals(features, t)
                strategy_signals[strategy.name] = signal
            except Exception as e:
                # Skip failed strategies
                continue
        
        if not strategy_signals:
            return self._empty_signal(t)
        
        # Update weights based on recent performance
        self._update_weights()
        
        # Combine signals
        combined_weights = {}
        combined_returns = {}
        
        for symbol in features.symbols:
            symbol_weight = 0.0
            symbol_return = 0.0
            weight_sum = 0.0
            
            for strat_name, signal in strategy_signals.items():
                strat_weight = self.strategy_weights.get(strat_name, 0)
                asset_weight = signal.desired_weights.get(symbol, 0)
                asset_return = signal.expected_returns_by_asset.get(symbol, 0)
                
                symbol_weight += strat_weight * asset_weight
                symbol_return += strat_weight * asset_return * signal.confidence
                weight_sum += strat_weight * signal.confidence
            
            if abs(symbol_weight) > 0.001:
                combined_weights[symbol] = symbol_weight
                combined_returns[symbol] = symbol_return / weight_sum if weight_sum > 0 else 0
        
        # Normalize
        total = sum(abs(w) for w in combined_weights.values())
        if total > 1.0:
            combined_weights = {k: v / total for k, v in combined_weights.items()}
        
        # Store for next update
        self.last_signals = strategy_signals
        self.last_prices = dict(features.prices)
        
        # Calculate metrics
        exp_ret = self._calculate_expected_return(combined_weights, combined_returns)
        risk = self._calculate_risk(combined_weights, features.covariance_matrix)
        
        # Confidence: weighted average of strategy confidences
        avg_confidence = np.mean([
            s.confidence * self.strategy_weights.get(s.strategy_name, 0)
            for s in strategy_signals.values()
        ]) if strategy_signals else 0.5
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=combined_weights,
            expected_return=exp_ret,
            expected_returns_by_asset=combined_returns,
            risk_estimate=risk,
            confidence=avg_confidence,
            explanation={
                'strategy_weights': dict(self.strategy_weights),
                'active_strategies': list(strategy_signals.keys()),
                'strategy_signals': {
                    name: {
                        'confidence': s.confidence,
                        'expected_return': s.expected_return,
                        'regime_fit': s.regime_fit,
                    }
                    for name, s in strategy_signals.items()
                },
            },
            regime_fit=0.7,
            diversification_score=0.9,  # High diversification through combination
        )
    
    def _update_performance(self, features: Features) -> None:
        """Update strategy performance based on realized returns."""
        for strat_name, signal in self.last_signals.items():
            # Calculate realized P&L for this strategy's positions
            pnl = 0.0
            for symbol, weight in signal.desired_weights.items():
                last_price = self.last_prices.get(symbol)
                curr_price = features.prices.get(symbol)
                
                if last_price and curr_price and last_price > 0:
                    ret = (curr_price - last_price) / last_price
                    pnl += weight * ret
            
            self.performance_history[strat_name].append(pnl)
    
    def _update_weights(self) -> None:
        """Update strategy weights based on recent performance."""
        # Calculate recent Sharpe-like metric for each strategy
        scores = {}
        
        for name, history in self.performance_history.items():
            if len(history) < 5:
                scores[name] = 0.0
                continue
            
            returns = np.array(history)
            mean_ret = np.mean(returns)
            std_ret = np.std(returns) + 1e-6
            
            # Simple Sharpe proxy
            scores[name] = mean_ret / std_ret
        
        # Softmax to convert scores to weights
        if scores:
            max_score = max(scores.values())
            exp_scores = {k: np.exp((v - max_score) / 0.1) for k, v in scores.items()}
            total = sum(exp_scores.values())
            
            if total > 0:
                new_weights = {k: max(self.min_weight, v / total) for k, v in exp_scores.items()}
                
                # Blend with current weights (momentum)
                for name in self.strategy_weights:
                    if name in new_weights:
                        self.strategy_weights[name] = (
                            (1 - self.learning_rate) * self.strategy_weights[name] +
                            self.learning_rate * new_weights[name]
                        )
                
                # Re-normalize
                total = sum(self.strategy_weights.values())
                self.strategy_weights = {k: v / total for k, v in self.strategy_weights.items()}
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'error': 'No strategy signals available'},
        )
