"""
Thompson Sampling - Smart exploration vs exploitation for strategy selection.

Thompson Sampling is a Bayesian approach that:
- Maintains probability distributions over strategy performance
- Samples from these distributions to decide weights
- Naturally balances exploration (trying uncertain strategies) 
  with exploitation (using proven strategies)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BetaDistribution:
    """
    Beta distribution for strategy performance.
    
    Beta(alpha, beta) where:
    - alpha = successes + 1
    - beta = failures + 1
    
    Mean = alpha / (alpha + beta)
    """
    alpha: float = 1.0  # Prior successes
    beta: float = 1.0   # Prior failures
    
    def sample(self) -> float:
        """Sample from the distribution."""
        return np.random.beta(self.alpha, self.beta)
    
    def mean(self) -> float:
        """Get the mean of the distribution."""
        return self.alpha / (self.alpha + self.beta)
    
    def variance(self) -> float:
        """Get the variance (uncertainty)."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    def update(self, success: bool, weight: float = 1.0):
        """Update distribution based on outcome."""
        if success:
            self.alpha += weight
        else:
            self.beta += weight
    
    def update_continuous(self, return_value: float):
        """
        Update based on continuous return.
        Positive return -> success, negative -> failure.
        Magnitude determines weight.
        """
        if return_value >= 0:
            # Success - weight proportional to return
            self.alpha += 1 + min(return_value * 10, 5)
        else:
            # Failure - weight proportional to loss
            self.beta += 1 + min(abs(return_value) * 10, 5)


@dataclass
class StrategyBelief:
    """Complete belief state for a strategy."""
    strategy_name: str
    distribution: BetaDistribution = field(default_factory=BetaDistribution)
    
    # Performance by regime
    regime_distributions: Dict[str, BetaDistribution] = field(default_factory=dict)
    
    # Tracking
    total_selections: int = 0
    total_successes: int = 0
    last_sampled_value: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'strategy_name': self.strategy_name,
            'alpha': self.distribution.alpha,
            'beta': self.distribution.beta,
            'regime_distributions': {
                regime: {'alpha': d.alpha, 'beta': d.beta}
                for regime, d in self.regime_distributions.items()
            },
            'total_selections': self.total_selections,
            'total_successes': self.total_successes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyBelief':
        belief = cls(strategy_name=data['strategy_name'])
        belief.distribution = BetaDistribution(
            alpha=data.get('alpha', 1.0),
            beta=data.get('beta', 1.0)
        )
        belief.total_selections = data.get('total_selections', 0)
        belief.total_successes = data.get('total_successes', 0)
        
        for regime, rd in data.get('regime_distributions', {}).items():
            belief.regime_distributions[regime] = BetaDistribution(
                alpha=rd['alpha'],
                beta=rd['beta']
            )
        
        return belief


class ThompsonSamplingWeights:
    """
    Thompson Sampling for adaptive strategy weighting.
    
    Features:
    - Bayesian probability model for each strategy
    - Automatic exploration of uncertain strategies
    - Regime-conditional sampling
    - Convergence to optimal weights over time
    """
    
    def __init__(
        self,
        strategy_names: List[str],
        storage_path: str = "outputs/thompson_beliefs.json",
        exploration_bonus: float = 0.1,
        min_weight: float = 0.02,
    ):
        """
        Initialize Thompson Sampling.
        
        Args:
            strategy_names: List of strategy names
            storage_path: Path for persisting beliefs
            exploration_bonus: Bonus for uncertain strategies
            min_weight: Minimum weight for any strategy
        """
        self.strategy_names = strategy_names
        self.storage_path = Path(storage_path)
        self.exploration_bonus = exploration_bonus
        self.min_weight = min_weight
        
        self.beliefs: Dict[str, StrategyBelief] = {}
        self._load()
        self._initialize_missing()
    
    def _initialize_missing(self):
        """Initialize beliefs for new strategies."""
        for name in self.strategy_names:
            if name not in self.beliefs:
                self.beliefs[name] = StrategyBelief(strategy_name=name)
    
    def _load(self):
        """Load beliefs from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for name, belief_data in data.get('beliefs', {}).items():
                    self.beliefs[name] = StrategyBelief.from_dict(belief_data)
                
                logger.info(f"Loaded Thompson beliefs for {len(self.beliefs)} strategies")
            except Exception as e:
                logger.warning(f"Could not load Thompson beliefs: {e}")
    
    def _save(self):
        """Persist beliefs to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'beliefs': {name: b.to_dict() for name, b in self.beliefs.items()},
                    'last_updated': str(np.datetime64('now')),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save Thompson beliefs: {e}")
    
    def sample_weights(
        self,
        regime: Optional[str] = None,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Sample strategy weights using Thompson Sampling.
        
        Args:
            regime: Current market regime (optional)
            n_samples: Number of samples for averaging
        
        Returns:
            Dict of strategy_name -> weight
        """
        # Collect samples
        all_samples = {name: [] for name in self.strategy_names}
        
        for _ in range(n_samples):
            for name in self.strategy_names:
                belief = self.beliefs.get(name)
                if not belief:
                    all_samples[name].append(0.5)
                    continue
                
                # Use regime-specific distribution if available
                if regime and regime in belief.regime_distributions:
                    sample = belief.regime_distributions[regime].sample()
                else:
                    sample = belief.distribution.sample()
                
                # Add exploration bonus for uncertain strategies
                uncertainty = belief.distribution.variance()
                sample += self.exploration_bonus * np.sqrt(uncertainty)
                
                all_samples[name].append(sample)
                belief.last_sampled_value = sample
        
        # Average samples and normalize
        raw_weights = {
            name: np.mean(samples)
            for name, samples in all_samples.items()
        }
        
        # Ensure minimum weight
        for name in raw_weights:
            raw_weights[name] = max(self.min_weight, raw_weights[name])
        
        # Normalize
        total = sum(raw_weights.values())
        weights = {name: w / total for name, w in raw_weights.items()}
        
        # Update selection counts
        for name in self.strategy_names:
            if name in self.beliefs:
                self.beliefs[name].total_selections += 1
        
        return weights
    
    def update_from_outcome(
        self,
        strategy_returns: Dict[str, float],
        regime: Optional[str] = None,
    ):
        """
        Update beliefs based on observed outcomes.
        
        Args:
            strategy_returns: Dict of strategy_name -> return
            regime: Market regime during this period
        """
        for name, return_val in strategy_returns.items():
            if name not in self.beliefs:
                continue
            
            belief = self.beliefs[name]
            
            # Update overall distribution
            belief.distribution.update_continuous(return_val)
            
            # Update regime-specific distribution
            if regime:
                if regime not in belief.regime_distributions:
                    belief.regime_distributions[regime] = BetaDistribution()
                belief.regime_distributions[regime].update_continuous(return_val)
            
            # Track successes
            if return_val > 0:
                belief.total_successes += 1
        
        self._save()
    
    def get_exploration_priorities(self) -> List[Tuple[str, float]]:
        """
        Get strategies ranked by exploration priority.
        
        Returns:
            List of (strategy_name, uncertainty_score)
        """
        priorities = []
        
        for name, belief in self.beliefs.items():
            uncertainty = belief.distribution.variance()
            n_selections = belief.total_selections
            
            # UCB-like priority: uncertainty + inverse of selections
            priority = uncertainty + 1.0 / (n_selections + 1)
            priorities.append((name, priority))
        
        return sorted(priorities, key=lambda x: x[1], reverse=True)
    
    def get_regime_specialists(self, regime: str) -> List[Tuple[str, float]]:
        """
        Get strategies ranked by performance in a regime.
        
        Args:
            regime: Market regime
        
        Returns:
            List of (strategy_name, expected_performance)
        """
        specialists = []
        
        for name, belief in self.beliefs.items():
            if regime in belief.regime_distributions:
                mean = belief.regime_distributions[regime].mean()
            else:
                mean = belief.distribution.mean()
            
            specialists.append((name, mean))
        
        return sorted(specialists, key=lambda x: x[1], reverse=True)
    
    def blend_with_debate(
        self,
        debate_scores: Dict[str, float],
        regime: Optional[str] = None,
        thompson_influence: float = 0.4,
    ) -> Dict[str, float]:
        """
        Blend Thompson weights with debate scores.
        
        Args:
            debate_scores: Scores from debate engine
            regime: Current market regime
            thompson_influence: Weight given to Thompson (0-1)
        
        Returns:
            Blended strategy weights
        """
        thompson_weights = self.sample_weights(regime)
        
        # Normalize debate scores
        total_debate = sum(debate_scores.values()) or 1
        debate_weights = {
            name: score / total_debate
            for name, score in debate_scores.items()
        }
        
        # Blend
        blended = {}
        all_names = set(thompson_weights.keys()) | set(debate_weights.keys())
        
        for name in all_names:
            tw = thompson_weights.get(name, 0)
            dw = debate_weights.get(name, 0)
            blended[name] = thompson_influence * tw + (1 - thompson_influence) * dw
        
        # Normalize
        total = sum(blended.values()) or 1
        return {name: w / total for name, w in blended.items()}
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of Thompson Sampling state."""
        return {
            'strategies': len(self.beliefs),
            'total_updates': sum(b.total_selections for b in self.beliefs.values()),
            'beliefs': {
                name: {
                    'mean': b.distribution.mean(),
                    'uncertainty': b.distribution.variance(),
                    'selections': b.total_selections,
                    'success_rate': b.total_successes / max(1, b.total_selections),
                }
                for name, b in self.beliefs.items()
            },
            'exploration_priorities': self.get_exploration_priorities()[:3],
        }
