"""
Debate Learning System - Learns from debate outcomes to improve future debates.

Tracks:
1. Which attacks succeeded in which market regimes
2. Which strategies defend well vs. poorly
3. Debate-to-outcome correlations (did the debate winner actually perform well?)
4. Successful argument patterns
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DebateRecord:
    """Record of a single debate outcome."""
    timestamp: str
    regime: str
    volatility_regime: str
    
    # Debate participants
    strategies: List[str]
    
    # Initial vs final scores
    initial_scores: Dict[str, float]
    final_scores: Dict[str, float]
    
    # Attack records
    attacks: List[Dict[str, Any]]  # {attacker, defender, claim, strength, succeeded}
    
    # Rebuttal records
    rebuttals: List[Dict[str, Any]]  # {defender, attacker, claim, strength, succeeded}
    
    # Winners
    debate_winners: List[str]
    
    # Post-debate outcome (filled in later)
    actual_performance: Optional[Dict[str, float]] = None  # strategy -> actual return
    debate_accuracy: Optional[float] = None  # Did winners actually perform better?


@dataclass
class AttackPattern:
    """A learned attack pattern."""
    attacker: str
    defender: str
    regime: str
    claim_type: str  # e.g., "whipsaw", "concentration", "regime_mismatch"
    
    # Success metrics
    times_used: int = 0
    times_succeeded: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.times_succeeded / max(1, self.times_used)


@dataclass
class StrategyDebateProfile:
    """Learned profile of how a strategy performs in debates."""
    strategy_name: str
    
    # Attack effectiveness by regime
    attack_success_by_regime: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # regime -> (wins, total)
    
    # Defense effectiveness by regime
    defense_success_by_regime: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    # Overall debate performance
    debates_won: int = 0
    debates_lost: int = 0
    debates_participated: int = 0
    
    # Accuracy: Did winning debates lead to good performance?
    accurate_wins: int = 0  # Won debate AND performed well
    inaccurate_wins: int = 0  # Won debate but performed poorly
    accurate_losses: int = 0  # Lost debate AND performed poorly
    inaccurate_losses: int = 0  # Lost debate but actually performed well
    
    @property
    def debate_win_rate(self) -> float:
        return self.debates_won / max(1, self.debates_participated)
    
    @property
    def debate_accuracy(self) -> float:
        """How often does debate outcome match actual performance?"""
        total = self.accurate_wins + self.inaccurate_wins + self.accurate_losses + self.inaccurate_losses
        if total == 0:
            return 0.5
        return (self.accurate_wins + self.accurate_losses) / total
    
    def get_attack_success_rate(self, regime: str) -> float:
        wins, total = self.attack_success_by_regime.get(regime, (0, 0))
        return wins / max(1, total)
    
    def get_defense_success_rate(self, regime: str) -> float:
        wins, total = self.defense_success_by_regime.get(regime, (0, 0))
        return wins / max(1, total)


class DebateLearner:
    """
    Learns from debate outcomes to improve future debates.
    
    Key learnings:
    1. Attack effectiveness: Which attacks work in which regimes
    2. Defense patterns: Which strategies defend well
    3. Debate accuracy: Do debate winners actually outperform?
    4. Argument quality: Which types of arguments are most convincing
    """
    
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.debates_dir = self.outputs_dir / "debate_history"
        self.debates_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory state
        self.debate_history: List[DebateRecord] = []
        self.strategy_profiles: Dict[str, StrategyDebateProfile] = {}
        self.attack_patterns: Dict[str, AttackPattern] = {}  # key: "attacker_defender_regime"
        
        # Load existing data
        self._load_history()
    
    def _load_history(self):
        """Load debate history from disk."""
        history_file = self.debates_dir / "debate_history.json"
        profiles_file = self.debates_dir / "strategy_profiles.json"
        patterns_file = self.debates_dir / "attack_patterns.json"
        
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.debate_history = [DebateRecord(**d) for d in data[-100:]]  # Keep last 100
                logger.info(f"Loaded {len(self.debate_history)} debate records")
            
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    data = json.load(f)
                    for name, profile_data in data.items():
                        self.strategy_profiles[name] = StrategyDebateProfile(
                            strategy_name=name,
                            attack_success_by_regime=profile_data.get('attack_success_by_regime', {}),
                            defense_success_by_regime=profile_data.get('defense_success_by_regime', {}),
                            debates_won=profile_data.get('debates_won', 0),
                            debates_lost=profile_data.get('debates_lost', 0),
                            debates_participated=profile_data.get('debates_participated', 0),
                            accurate_wins=profile_data.get('accurate_wins', 0),
                            inaccurate_wins=profile_data.get('inaccurate_wins', 0),
                            accurate_losses=profile_data.get('accurate_losses', 0),
                            inaccurate_losses=profile_data.get('inaccurate_losses', 0),
                        )
                logger.info(f"Loaded {len(self.strategy_profiles)} strategy debate profiles")
            
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    for key, pattern_data in data.items():
                        self.attack_patterns[key] = AttackPattern(**pattern_data)
                logger.info(f"Loaded {len(self.attack_patterns)} attack patterns")
                
        except Exception as e:
            logger.warning(f"Error loading debate history: {e}")
    
    def _save_history(self):
        """Persist debate learning to disk."""
        try:
            # Save debate history
            history_file = self.debates_dir / "debate_history.json"
            with open(history_file, 'w') as f:
                json.dump([asdict(d) for d in self.debate_history[-100:]], f, indent=2, default=str)
            
            # Save strategy profiles
            profiles_file = self.debates_dir / "strategy_profiles.json"
            profiles_data = {}
            for name, profile in self.strategy_profiles.items():
                profiles_data[name] = {
                    'strategy_name': profile.strategy_name,
                    'attack_success_by_regime': profile.attack_success_by_regime,
                    'defense_success_by_regime': profile.defense_success_by_regime,
                    'debates_won': profile.debates_won,
                    'debates_lost': profile.debates_lost,
                    'debates_participated': profile.debates_participated,
                    'accurate_wins': profile.accurate_wins,
                    'inaccurate_wins': profile.inaccurate_wins,
                    'accurate_losses': profile.accurate_losses,
                    'inaccurate_losses': profile.inaccurate_losses,
                }
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            # Save attack patterns
            patterns_file = self.debates_dir / "attack_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.attack_patterns.items()}, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error saving debate history: {e}")
    
    def record_debate(
        self,
        timestamp: datetime,
        regime: str,
        volatility_regime: str,
        strategies: List[str],
        initial_scores: Dict[str, float],
        final_scores: Dict[str, float],
        attacks: List[Dict[str, Any]],
        rebuttals: List[Dict[str, Any]],
        debate_winners: List[str],
    ) -> None:
        """Record a debate outcome for learning."""
        
        record = DebateRecord(
            timestamp=timestamp.isoformat(),
            regime=regime,
            volatility_regime=volatility_regime,
            strategies=strategies,
            initial_scores=initial_scores,
            final_scores=final_scores,
            attacks=attacks,
            rebuttals=rebuttals,
            debate_winners=debate_winners,
        )
        
        self.debate_history.append(record)
        
        # Update strategy profiles
        for strat in strategies:
            if strat not in self.strategy_profiles:
                self.strategy_profiles[strat] = StrategyDebateProfile(strategy_name=strat)
            
            profile = self.strategy_profiles[strat]
            profile.debates_participated += 1
            
            if strat in debate_winners:
                profile.debates_won += 1
            else:
                profile.debates_lost += 1
        
        # Update attack patterns
        for attack in attacks:
            attacker = attack.get('attacker')
            defender = attack.get('defender')
            strength = attack.get('strength', 0)
            
            if not attacker or not defender:
                continue
            
            key = f"{attacker}_{defender}_{regime}"
            
            if key not in self.attack_patterns:
                self.attack_patterns[key] = AttackPattern(
                    attacker=attacker,
                    defender=defender,
                    regime=regime,
                    claim_type=attack.get('claim_type', 'general'),
                )
            
            pattern = self.attack_patterns[key]
            pattern.times_used += 1
            
            # Attack succeeded if defender's score dropped significantly
            initial = initial_scores.get(defender, 0)
            final = final_scores.get(defender, 0)
            if final < initial - 0.03:  # Dropped by at least 3%
                pattern.times_succeeded += 1
            
            # Update attacker's profile
            attacker_profile = self.strategy_profiles.get(attacker)
            if attacker_profile:
                wins, total = attacker_profile.attack_success_by_regime.get(regime, (0, 0))
                total += 1
                if final < initial - 0.03:
                    wins += 1
                attacker_profile.attack_success_by_regime[regime] = (wins, total)
        
        # Update rebuttal patterns
        for rebuttal in rebuttals:
            defender = rebuttal.get('defender')
            
            if not defender:
                continue
            
            defender_profile = self.strategy_profiles.get(defender)
            if defender_profile:
                wins, total = defender_profile.defense_success_by_regime.get(regime, (0, 0))
                total += 1
                
                # Defense succeeded if score didn't drop much despite being attacked
                initial = initial_scores.get(defender, 0)
                final = final_scores.get(defender, 0)
                if final >= initial - 0.02:  # Held ground
                    wins += 1
                defender_profile.defense_success_by_regime[regime] = (wins, total)
        
        self._save_history()
        logger.info(f"Recorded debate outcome with {len(attacks)} attacks, winners: {debate_winners}")
    
    def record_actual_performance(
        self,
        timestamp: datetime,
        strategy_returns: Dict[str, float],  # strategy_name -> actual return
    ) -> None:
        """
        Record actual performance to validate debate accuracy.
        Called after positions have been held for some time.
        """
        # Find the closest prior debate
        target_time = timestamp.isoformat()
        
        for record in reversed(self.debate_history):
            if record.actual_performance is not None:
                continue  # Already filled
            
            # Check if this debate is within 24 hours before the performance measurement
            debate_time = datetime.fromisoformat(record.timestamp)
            if debate_time <= timestamp and (timestamp - debate_time) < timedelta(days=1):
                record.actual_performance = strategy_returns
                
                # Calculate debate accuracy
                winners = record.debate_winners
                non_winners = [s for s in record.strategies if s not in winners]
                
                winner_avg_return = sum(strategy_returns.get(w, 0) for w in winners) / max(1, len(winners))
                non_winner_avg_return = sum(strategy_returns.get(w, 0) for w in non_winners) / max(1, len(non_winners))
                
                # Debate was accurate if winners outperformed non-winners
                record.debate_accuracy = 1.0 if winner_avg_return > non_winner_avg_return else 0.0
                
                # Update strategy profiles
                for strat in record.strategies:
                    profile = self.strategy_profiles.get(strat)
                    if not profile:
                        continue
                    
                    strat_return = strategy_returns.get(strat, 0)
                    median_return = sorted(strategy_returns.values())[len(strategy_returns) // 2] if strategy_returns else 0
                    performed_well = strat_return > median_return
                    
                    if strat in winners:
                        if performed_well:
                            profile.accurate_wins += 1
                        else:
                            profile.inaccurate_wins += 1
                    else:
                        if performed_well:
                            profile.inaccurate_losses += 1
                        else:
                            profile.accurate_losses += 1
                
                self._save_history()
                logger.info(f"Updated debate accuracy: {record.debate_accuracy:.0%}")
                break
    
    def get_attack_boost(
        self,
        attacker: str,
        defender: str,
        regime: str,
    ) -> float:
        """
        Get a boost/penalty for an attack based on historical success.
        
        Returns:
            Multiplier (e.g., 1.2 means 20% more effective, 0.8 means 20% less)
        """
        key = f"{attacker}_{defender}_{regime}"
        pattern = self.attack_patterns.get(key)
        
        if pattern and pattern.times_used >= 3:  # Need at least 3 samples
            success_rate = pattern.success_rate
            # Map success rate to multiplier: 0% -> 0.7, 50% -> 1.0, 100% -> 1.3
            return 0.7 + (success_rate * 0.6)
        
        # Check attacker's general attack success in this regime
        attacker_profile = self.strategy_profiles.get(attacker)
        if attacker_profile:
            attack_rate = attacker_profile.get_attack_success_rate(regime)
            if attack_rate > 0:
                return 0.8 + (attack_rate * 0.4)
        
        return 1.0  # No adjustment
    
    def get_defense_boost(
        self,
        defender: str,
        regime: str,
    ) -> float:
        """
        Get a boost/penalty for defense based on historical success.
        """
        defender_profile = self.strategy_profiles.get(defender)
        if defender_profile:
            defense_rate = defender_profile.get_defense_success_rate(regime)
            if defense_rate > 0:
                # Good defenders get stronger rebuttals
                return 0.8 + (defense_rate * 0.4)
        
        return 1.0
    
    def get_debate_credibility(self, strategy: str) -> float:
        """
        Get overall credibility of a strategy's debate performance.
        
        Combines:
        - Win rate in debates
        - Accuracy of debate wins (do wins translate to performance?)
        """
        profile = self.strategy_profiles.get(strategy)
        if not profile or profile.debates_participated < 3:
            return 0.5  # Neutral
        
        # Weight win rate and accuracy
        win_rate = profile.debate_win_rate
        accuracy = profile.debate_accuracy
        
        # Accuracy matters more - we want strategies that win AND perform
        credibility = 0.3 * win_rate + 0.7 * accuracy
        
        return credibility
    
    def get_learned_debate_weights(
        self,
        strategies: List[str],
        regime: str,
        base_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Adjust debate scores based on learned credibility.
        
        Strategies with high debate accuracy get boosted.
        Strategies that win debates but underperform get penalized.
        """
        adjusted = {}
        
        for strat in strategies:
            base = base_scores.get(strat, 0.5)
            credibility = self.get_debate_credibility(strat)
            
            # Blend: 70% base score, 30% adjusted by credibility
            # credibility of 0.5 = no change
            # credibility of 1.0 = 15% boost
            # credibility of 0.0 = 15% penalty
            adjustment = (credibility - 0.5) * 0.3
            adjusted[strat] = max(0.1, min(1.0, base + adjustment))
        
        return adjusted
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of debate learning."""
        summary = {
            'total_debates_recorded': len(self.debate_history),
            'total_attack_patterns': len(self.attack_patterns),
            'strategies_tracked': len(self.strategy_profiles),
            'strategy_credibility': {},
            'top_attack_patterns': [],
            'debate_accuracy': 0.0,
        }
        
        # Strategy credibility
        for name, profile in self.strategy_profiles.items():
            summary['strategy_credibility'][name] = {
                'win_rate': f"{profile.debate_win_rate:.0%}",
                'accuracy': f"{profile.debate_accuracy:.0%}",
                'credibility': f"{self.get_debate_credibility(name):.0%}",
                'debates': profile.debates_participated,
            }
        
        # Top attack patterns
        sorted_patterns = sorted(
            self.attack_patterns.values(),
            key=lambda p: p.success_rate * p.times_used,
            reverse=True
        )
        for pattern in sorted_patterns[:5]:
            summary['top_attack_patterns'].append({
                'attacker': pattern.attacker,
                'defender': pattern.defender,
                'regime': pattern.regime,
                'success_rate': f"{pattern.success_rate:.0%}",
                'times_used': pattern.times_used,
            })
        
        # Overall debate accuracy
        accurate = sum(1 for d in self.debate_history if d.debate_accuracy == 1.0)
        total_with_accuracy = sum(1 for d in self.debate_history if d.debate_accuracy is not None)
        if total_with_accuracy > 0:
            summary['debate_accuracy'] = f"{accurate / total_with_accuracy:.0%}"
        
        return summary
    
    def get_insights_for_regime(self, regime: str) -> List[str]:
        """Get learned insights specific to a regime."""
        insights = []
        
        # Find successful attack patterns in this regime
        regime_patterns = [p for p in self.attack_patterns.values() 
                          if p.regime == regime and p.times_used >= 2]
        
        best_attacks = sorted(regime_patterns, key=lambda p: p.success_rate, reverse=True)[:3]
        for pattern in best_attacks:
            if pattern.success_rate > 0.6:
                insights.append(
                    f"In {regime}: {pattern.attacker} attacks on {pattern.defender} "
                    f"succeed {pattern.success_rate:.0%} of the time"
                )
        
        # Find best defenders in this regime
        best_defenders = []
        for name, profile in self.strategy_profiles.items():
            defense_rate = profile.get_defense_success_rate(regime)
            if defense_rate > 0.5:
                best_defenders.append((name, defense_rate))
        
        best_defenders.sort(key=lambda x: x[1], reverse=True)
        for name, rate in best_defenders[:2]:
            insights.append(f"In {regime}: {name} defends well ({rate:.0%} success)")
        
        return insights
