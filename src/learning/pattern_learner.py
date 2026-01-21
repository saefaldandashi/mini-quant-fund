"""
Pattern Learner - Learns regime-specific patterns and trading rules.

Discovers:
- Which market conditions favor which strategies
- Common patterns before winning/losing trades
- Optimal holding periods for different setups
- Risk signals that precede drawdowns
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np


@dataclass
class Pattern:
    """A learned pattern with its associated outcome."""
    pattern_id: str
    description: str
    
    # Conditions that define this pattern
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Outcome statistics
    times_observed: int = 0
    times_profitable: int = 0
    avg_return: float = 0.0
    confidence: float = 0.0
    
    # Best action for this pattern
    recommended_action: str = ''  # 'increase_exposure', 'reduce_exposure', 'hold'
    recommended_strategies: List[str] = field(default_factory=list)
    strategies_to_avoid: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Pattern':
        return cls(**data)


@dataclass
class TradingRule:
    """A learned trading rule."""
    rule_id: str
    rule_text: str
    
    # Rule conditions (if-then)
    if_conditions: Dict[str, Any] = field(default_factory=dict)
    then_action: str = ''
    
    # Effectiveness
    times_triggered: int = 0
    times_correct: int = 0
    accuracy: float = 0.0
    expected_value: float = 0.0
    
    # Is this rule active?
    active: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingRule':
        return cls(**data)


class PatternLearner:
    """
    Learns patterns from trading history and market conditions.
    
    Features:
    - Automatic pattern discovery from trade outcomes
    - Regime-strategy affinity learning
    - Holding period optimization
    - Risk signal detection
    """
    
    def __init__(self, storage_path: str = "outputs/patterns.json"):
        self.storage_path = Path(storage_path)
        self.patterns: Dict[str, Pattern] = {}
        self.rules: Dict[str, TradingRule] = {}
        self.observations: List[Dict] = []
        self._load()
        self._initialize_base_patterns()
    
    def _load(self):
        """Load learned patterns from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for pid, pattern_dict in data.get('patterns', {}).items():
                    self.patterns[pid] = Pattern.from_dict(pattern_dict)
                
                for rid, rule_dict in data.get('rules', {}).items():
                    self.rules[rid] = TradingRule.from_dict(rule_dict)
                
                self.observations = data.get('observations', [])[-500:]
                
                logging.info(f"Loaded {len(self.patterns)} patterns, {len(self.rules)} rules")
            except Exception as e:
                logging.warning(f"Could not load patterns: {e}")
    
    def _save(self):
        """Persist learned patterns to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'patterns': {pid: p.to_dict() for pid, p in self.patterns.items()},
                    'rules': {rid: r.to_dict() for rid, r in self.rules.items()},
                    'observations': self.observations[-500:],
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save patterns: {e}")
    
    def _initialize_base_patterns(self):
        """Initialize common market patterns if not already present."""
        base_patterns = [
            Pattern(
                pattern_id='high_vol_risk_off',
                description='High volatility with risk-off regime',
                conditions={'volatility_regime': 'high', 'regime': 'risk_off'},
                recommended_action='reduce_exposure',
                recommended_strategies=['TailRiskOverlay', 'RiskParityMinVar'],
                strategies_to_avoid=['TimeSeriesMomentum', 'CrossSectionMomentum'],
            ),
            Pattern(
                pattern_id='low_vol_risk_on',
                description='Low volatility with risk-on regime',
                conditions={'volatility_regime': 'low', 'regime': 'risk_on'},
                recommended_action='increase_exposure',
                recommended_strategies=['TimeSeriesMomentum', 'CrossSectionMomentum', 'MeanReversion'],
                strategies_to_avoid=['TailRiskOverlay'],
            ),
            Pattern(
                pattern_id='high_vol_trending',
                description='High volatility with strong trend',
                conditions={'volatility_regime': 'high', 'trend_strength': 'strong'},
                recommended_action='hold',
                recommended_strategies=['TimeSeriesMomentum', 'TailRiskOverlay'],
                strategies_to_avoid=['MeanReversion'],
            ),
            Pattern(
                pattern_id='mean_reversion_setup',
                description='Low volatility, weak trend, oversold conditions',
                conditions={'volatility_regime': 'low', 'trend_strength': 'weak'},
                recommended_action='increase_exposure',
                recommended_strategies=['MeanReversion', 'ValueQualityTilt'],
                strategies_to_avoid=['TimeSeriesMomentum'],
            ),
            Pattern(
                pattern_id='sentiment_divergence',
                description='Positive sentiment but negative price action',
                conditions={'sentiment': 'positive', 'price_direction': 'down'},
                recommended_action='hold',
                recommended_strategies=['NewsSentimentEvent', 'ValueQualityTilt'],
                strategies_to_avoid=['TimeSeriesMomentum'],
            ),
        ]
        
        for pattern in base_patterns:
            if pattern.pattern_id not in self.patterns:
                self.patterns[pattern.pattern_id] = pattern
        
        self._save()
    
    def record_observation(
        self,
        market_context: Dict[str, Any],
        strategy_signals: Dict[str, float],
        outcome_return: float,
        winning_strategies: List[str],
        losing_strategies: List[str],
    ):
        """
        Record an observation for pattern learning.
        
        Args:
            market_context: Current market conditions
            strategy_signals: Strategy name -> signal strength
            outcome_return: Actual portfolio return
            winning_strategies: Strategies that performed well
            losing_strategies: Strategies that performed poorly
        """
        observation = {
            'timestamp': datetime.now().isoformat(),
            'market_context': market_context,
            'strategy_signals': strategy_signals,
            'outcome_return': outcome_return,
            'winning_strategies': winning_strategies,
            'losing_strategies': losing_strategies,
        }
        
        self.observations.append(observation)
        
        # Update pattern statistics
        self._update_patterns(observation)
        
        # Try to discover new patterns
        if len(self.observations) >= 20 and len(self.observations) % 10 == 0:
            self._discover_patterns()
        
        self._save()
    
    def _update_patterns(self, observation: Dict):
        """Update existing patterns with new observation."""
        market_ctx = observation['market_context']
        outcome = observation['outcome_return']
        
        for pattern in self.patterns.values():
            # Check if pattern conditions match
            matches = True
            for key, expected_value in pattern.conditions.items():
                actual_value = market_ctx.get(key)
                
                if isinstance(expected_value, str):
                    if actual_value != expected_value:
                        matches = False
                        break
                elif isinstance(expected_value, (int, float)):
                    if actual_value is None or abs(actual_value - expected_value) > 0.1:
                        matches = False
                        break
            
            if matches:
                # Update pattern statistics
                pattern.times_observed += 1
                if outcome > 0:
                    pattern.times_profitable += 1
                
                # Update rolling average return
                alpha = 0.1
                pattern.avg_return = (
                    alpha * outcome + (1 - alpha) * pattern.avg_return
                )
                
                # Update confidence
                if pattern.times_observed >= 5:
                    pattern.confidence = pattern.times_profitable / pattern.times_observed
                
                # Update recommended strategies based on what worked
                for strategy in observation['winning_strategies']:
                    if strategy not in pattern.recommended_strategies:
                        if pattern.times_observed >= 3:
                            pattern.recommended_strategies.append(strategy)
                
                for strategy in observation['losing_strategies']:
                    if strategy not in pattern.strategies_to_avoid:
                        if pattern.times_observed >= 3:
                            pattern.strategies_to_avoid.append(strategy)
    
    def _discover_patterns(self):
        """
        Attempt to discover new patterns from observations.
        
        Uses simple association rule mining approach.
        """
        if len(self.observations) < 20:
            return
        
        # Group observations by regime and volatility
        regime_vol_groups = defaultdict(list)
        
        for obs in self.observations[-100:]:
            ctx = obs['market_context']
            regime = ctx.get('regime', 'unknown')
            vol = ctx.get('volatility_regime', 'unknown')
            key = f"{regime}_{vol}"
            regime_vol_groups[key].append(obs)
        
        # Look for patterns in each group
        for group_key, group_obs in regime_vol_groups.items():
            if len(group_obs) < 5:
                continue
            
            # Find consistently winning strategies in this group
            strategy_wins = defaultdict(int)
            strategy_losses = defaultdict(int)
            
            for obs in group_obs:
                for s in obs['winning_strategies']:
                    strategy_wins[s] += 1
                for s in obs['losing_strategies']:
                    strategy_losses[s] += 1
            
            # Create pattern for strategies that consistently win
            for strategy, wins in strategy_wins.items():
                total = wins + strategy_losses.get(strategy, 0)
                if total >= 3 and wins / total >= 0.7:
                    pattern_id = f"discovered_{group_key}_{strategy}"
                    
                    if pattern_id not in self.patterns:
                        regime, vol = group_key.split('_')
                        self.patterns[pattern_id] = Pattern(
                            pattern_id=pattern_id,
                            description=f"{strategy} excels in {regime} {vol}-vol conditions",
                            conditions={'regime': regime, 'volatility_regime': vol},
                            recommended_strategies=[strategy],
                            times_observed=total,
                            times_profitable=wins,
                            confidence=wins / total,
                            recommended_action='increase_exposure',
                        )
                        logging.info(f"Discovered new pattern: {pattern_id}")
    
    def get_active_patterns(self, market_context: Dict[str, Any]) -> List[Pattern]:
        """
        Get patterns that match current market conditions.
        
        Args:
            market_context: Current market conditions
        
        Returns:
            List of matching patterns, sorted by confidence
        """
        matching = []
        
        for pattern in self.patterns.values():
            # Check if pattern conditions match
            matches = True
            for key, expected_value in pattern.conditions.items():
                actual_value = market_context.get(key)
                
                if actual_value is None:
                    continue
                
                if isinstance(expected_value, str):
                    if actual_value != expected_value:
                        matches = False
                        break
                elif isinstance(expected_value, (int, float)):
                    if abs(actual_value - expected_value) > 0.2:
                        matches = False
                        break
            
            if matches and pattern.times_observed >= 3:
                matching.append(pattern)
        
        # Sort by confidence
        return sorted(matching, key=lambda p: p.confidence, reverse=True)
    
    def get_strategy_recommendations(
        self, market_context: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Get strategy recommendations based on active patterns.
        
        Args:
            market_context: Current market conditions
        
        Returns:
            Tuple of (recommended_strategies, strategies_to_avoid)
        """
        active = self.get_active_patterns(market_context)
        
        if not active:
            return [], []
        
        # Aggregate recommendations weighted by confidence
        recommend_scores = defaultdict(float)
        avoid_scores = defaultdict(float)
        
        for pattern in active:
            for strategy in pattern.recommended_strategies:
                recommend_scores[strategy] += pattern.confidence
            for strategy in pattern.strategies_to_avoid:
                avoid_scores[strategy] += pattern.confidence
        
        # Get top recommendations
        recommended = sorted(
            recommend_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        to_avoid = sorted(
            avoid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return (
            [s for s, _ in recommended],
            [s for s, _ in to_avoid],
        )
    
    def get_risk_signals(self, market_context: Dict[str, Any]) -> List[str]:
        """
        Get risk signals based on learned patterns.
        
        Args:
            market_context: Current market conditions
        
        Returns:
            List of risk warning messages
        """
        warnings = []
        active = self.get_active_patterns(market_context)
        
        for pattern in active:
            if pattern.recommended_action == 'reduce_exposure':
                warnings.append(
                    f"Pattern '{pattern.description}' suggests reducing exposure "
                    f"(confidence: {pattern.confidence:.0%})"
                )
            
            if pattern.avg_return < -0.02 and pattern.times_observed >= 5:
                warnings.append(
                    f"Pattern '{pattern.description}' historically negative "
                    f"(avg return: {pattern.avg_return:.1%})"
                )
        
        return warnings
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of learned patterns."""
        high_confidence = [
            p for p in self.patterns.values()
            if p.confidence >= 0.6 and p.times_observed >= 5
        ]
        
        discovered = [
            p for p in self.patterns.values()
            if p.pattern_id.startswith('discovered_')
        ]
        
        return {
            'total_patterns': len(self.patterns),
            'total_observations': len(self.observations),
            'high_confidence_patterns': len(high_confidence),
            'discovered_patterns': len(discovered),
            'top_patterns': [
                {
                    'id': p.pattern_id,
                    'description': p.description,
                    'confidence': p.confidence,
                    'observations': p.times_observed,
                }
                for p in sorted(
                    self.patterns.values(),
                    key=lambda x: x.confidence,
                    reverse=True
                )[:5]
            ],
        }
