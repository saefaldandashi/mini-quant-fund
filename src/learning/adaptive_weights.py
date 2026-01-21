"""
Adaptive Weight Learner - Learns optimal strategy weights over time.

Uses multiple learning algorithms:
- Exponential Moving Performance (EMP): Recent performance weighted more heavily
- Multi-Armed Bandit (UCB1): Exploration vs exploitation balance
- Online Gradient Descent: Continuous weight updates
- Regime-Conditional Weighting: Different weights for different regimes
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class WeightState:
    """Current learned weights for a strategy."""
    strategy_name: str
    base_weight: float = 0.111  # Default: equal weight among 9 strategies
    
    # Exponential moving average of performance
    ema_performance: float = 0.0
    ema_alpha: float = 0.1  # Decay factor
    
    # UCB1 parameters
    times_selected: int = 0
    cumulative_reward: float = 0.0
    ucb_bonus: float = 0.0
    
    # Regime-specific weights
    regime_weights: Dict[str, float] = field(default_factory=dict)
    
    # Confidence in this strategy
    learned_confidence: float = 0.5
    
    # Last update
    last_updated: str = ''
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WeightState':
        return cls(**data)


class AdaptiveWeightLearner:
    """
    Learns optimal strategy weights using online learning algorithms.
    
    Features:
    - EMA-based weight adjustment (recent performance matters more)
    - UCB1 exploration bonus (try underexplored strategies)
    - Regime-conditional weights (different weights for different market conditions)
    - Smooth weight transitions (prevents sudden changes)
    """
    
    def __init__(
        self,
        strategy_names: List[str],
        storage_path: str = "outputs/learned_weights.json",
        learning_rate: float = 0.05,
        exploration_factor: float = 1.0,
        min_weight: float = 0.02,
        max_weight: float = 0.40,
    ):
        """
        Initialize the adaptive weight learner.
        
        Args:
            strategy_names: List of strategy names to track
            storage_path: Path to persist learned weights
            learning_rate: How quickly to adapt (0-1)
            exploration_factor: UCB1 exploration parameter
            min_weight: Minimum weight for any strategy
            max_weight: Maximum weight for any strategy
        """
        self.strategy_names = strategy_names
        self.storage_path = Path(storage_path)
        self.learning_rate = learning_rate
        self.exploration_factor = exploration_factor
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        self.weights: Dict[str, WeightState] = {}
        self.total_rounds: int = 0
        self.reward_history: List[Dict] = []
        
        self._load()
        self._initialize_missing_strategies()
    
    def _initialize_missing_strategies(self):
        """Initialize weights for any new strategies."""
        for name in self.strategy_names:
            if name not in self.weights:
                self.weights[name] = WeightState(
                    strategy_name=name,
                    base_weight=1.0 / len(self.strategy_names),
                )
    
    def _load(self):
        """Load learned weights from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for name, weight_dict in data.get('weights', {}).items():
                    self.weights[name] = WeightState.from_dict(weight_dict)
                
                self.total_rounds = data.get('total_rounds', 0)
                self.reward_history = data.get('reward_history', [])[-100:]  # Keep last 100
                
                logging.info(f"Loaded learned weights for {len(self.weights)} strategies")
            except Exception as e:
                logging.warning(f"Could not load learned weights: {e}")
    
    def _save(self):
        """Persist learned weights to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'weights': {name: w.to_dict() for name, w in self.weights.items()},
                    'total_rounds': self.total_rounds,
                    'reward_history': self.reward_history[-100:],
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save learned weights: {e}")
    
    def get_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get current learned weights for all strategies.
        
        Args:
            regime: Optional market regime for regime-specific weights
        
        Returns:
            Dict of strategy_name -> weight
        """
        raw_weights = {}
        
        for name, state in self.weights.items():
            if regime and regime in state.regime_weights:
                # Use regime-specific weight
                weight = state.regime_weights[regime]
            else:
                # Use base weight adjusted by EMA performance
                weight = state.base_weight * (1 + state.ema_performance)
            
            # Add UCB exploration bonus
            if self.total_rounds > 0 and state.times_selected > 0:
                ucb_bonus = self.exploration_factor * math.sqrt(
                    2 * math.log(self.total_rounds) / state.times_selected
                )
                weight += ucb_bonus * 0.1  # Small exploration bonus
            
            # Apply confidence scaling
            weight *= (0.5 + 0.5 * state.learned_confidence)
            
            raw_weights[name] = weight
        
        # Normalize and clip weights
        return self._normalize_weights(raw_weights)
    
    def _normalize_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1 and apply min/max constraints."""
        # Clip to min/max
        clipped = {
            name: max(self.min_weight, min(self.max_weight, w))
            for name, w in raw_weights.items()
        }
        
        # Normalize to sum to 1
        total = sum(clipped.values())
        if total > 0:
            normalized = {name: w / total for name, w in clipped.items()}
        else:
            # Fallback to equal weights
            n = len(clipped)
            normalized = {name: 1.0 / n for name in clipped}
        
        return normalized
    
    def update_from_outcomes(
        self,
        strategy_returns: Dict[str, float],
        regime: Optional[str] = None,
    ):
        """
        Update weights based on observed strategy returns.
        
        Args:
            strategy_returns: Dict of strategy_name -> return achieved
            regime: Current market regime (optional)
        """
        self.total_rounds += 1
        
        # Record history
        self.reward_history.append({
            'timestamp': datetime.now().isoformat(),
            'returns': strategy_returns,
            'regime': regime,
        })
        
        for name, ret in strategy_returns.items():
            if name not in self.weights:
                continue
            
            state = self.weights[name]
            
            # Update EMA performance
            state.ema_performance = (
                state.ema_alpha * ret +
                (1 - state.ema_alpha) * state.ema_performance
            )
            
            # Update UCB1 stats
            state.times_selected += 1
            state.cumulative_reward += ret
            
            # Update regime-specific weight
            if regime:
                if regime not in state.regime_weights:
                    state.regime_weights[regime] = state.base_weight
                
                # Gradient update for regime weight
                current_regime_weight = state.regime_weights[regime]
                gradient = ret  # Simple: positive return -> increase weight
                new_regime_weight = current_regime_weight + self.learning_rate * gradient
                state.regime_weights[regime] = max(
                    self.min_weight,
                    min(self.max_weight, new_regime_weight)
                )
            
            # Update learned confidence based on consistency
            if len(self.reward_history) >= 5:
                recent_returns = [
                    h['returns'].get(name, 0) 
                    for h in self.reward_history[-5:]
                ]
                consistency = 1.0 - np.std(recent_returns) / (abs(np.mean(recent_returns)) + 0.01)
                state.learned_confidence = max(0.2, min(0.95, consistency))
            
            state.last_updated = datetime.now().isoformat()
        
        # Perform online gradient descent on base weights
        self._gradient_descent_update(strategy_returns)
        
        self._save()
        
        logging.info(
            f"Updated weights from outcomes: "
            f"avg_return={np.mean(list(strategy_returns.values())):.4f}"
        )
    
    def _gradient_descent_update(self, strategy_returns: Dict[str, float]):
        """
        Update base weights using online gradient descent.
        
        Objective: Maximize weighted return (reward)
        """
        # Current weights
        current_weights = {
            name: state.base_weight 
            for name, state in self.weights.items()
        }
        
        # Compute gradients (derivative of reward w.r.t. weights)
        # Simple gradient: return * (1 if positive, -1 if negative)
        gradients = {}
        for name, ret in strategy_returns.items():
            if name in self.weights:
                # Gradient encourages higher weight for positive returns
                gradients[name] = ret
        
        # Update weights
        for name, gradient in gradients.items():
            state = self.weights[name]
            new_weight = state.base_weight + self.learning_rate * gradient
            state.base_weight = max(self.min_weight, min(self.max_weight, new_weight))
        
        # Renormalize base weights
        total = sum(s.base_weight for s in self.weights.values())
        if total > 0:
            for state in self.weights.values():
                state.base_weight /= total
    
    def get_exploration_recommendations(self) -> List[str]:
        """
        Get strategies that should be explored more.
        
        Returns:
            List of strategy names that are underexplored
        """
        if self.total_rounds < 10:
            return []
        
        recommendations = []
        avg_selections = self.total_rounds / len(self.weights) if self.weights else 0
        
        for name, state in self.weights.items():
            if state.times_selected < avg_selections * 0.5:
                recommendations.append(name)
        
        return recommendations
    
    def get_learning_summary(self) -> Dict[str, any]:
        """Get a summary of what has been learned."""
        current_weights = self.get_weights()
        
        # Find best and worst strategies by learned weight
        sorted_by_weight = sorted(
            current_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Find regime specialists
        regime_specialists = defaultdict(list)
        for name, state in self.weights.items():
            for regime, weight in state.regime_weights.items():
                regime_specialists[regime].append((name, weight))
        
        for regime in regime_specialists:
            regime_specialists[regime].sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_learning_rounds': self.total_rounds,
            'current_weights': current_weights,
            'top_strategies': sorted_by_weight[:3],
            'bottom_strategies': sorted_by_weight[-3:],
            'regime_specialists': dict(regime_specialists),
            'exploration_needed': self.get_exploration_recommendations(),
            'avg_confidence': np.mean([s.learned_confidence for s in self.weights.values()]),
        }
    
    def blend_with_debate_scores(
        self,
        debate_scores: Dict[str, float],
        regime: Optional[str] = None,
        learned_weight_influence: float = 0.3,
    ) -> Dict[str, float]:
        """
        Blend learned weights with debate engine scores.
        
        This allows the learning system to influence but not override
        the debate engine's real-time analysis.
        
        Args:
            debate_scores: Scores from the debate engine
            regime: Current market regime
            learned_weight_influence: How much to weight learned vs debate (0-1)
        
        Returns:
            Blended weights
        """
        learned = self.get_weights(regime)
        
        # Normalize debate scores to weights
        total_debate = sum(debate_scores.values()) or 1
        debate_weights = {
            name: score / total_debate 
            for name, score in debate_scores.items()
        }
        
        # Blend
        blended = {}
        for name in set(learned.keys()) | set(debate_weights.keys()):
            lw = learned.get(name, 0)
            dw = debate_weights.get(name, 0)
            blended[name] = (
                learned_weight_influence * lw +
                (1 - learned_weight_influence) * dw
            )
        
        # Normalize
        total = sum(blended.values()) or 1
        return {name: w / total for name, w in blended.items()}
