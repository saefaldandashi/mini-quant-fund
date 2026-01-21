"""
Feedback Loop - Connects outcomes to weight adjustments.

This is the critical component that makes the system LEARN from trades.
It tracks what worked, what didn't, and adjusts strategy weights accordingly.
"""
import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import math

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    strategy_name: str
    
    # Signal performance
    total_signals: int = 0
    correct_signals: int = 0
    
    # Return metrics
    total_return: float = 0.0
    avg_return: float = 0.0
    return_when_correct: float = 0.0
    return_when_wrong: float = 0.0
    
    # Regime-specific performance
    performance_by_regime: Dict[str, float] = field(default_factory=dict)
    
    # Confidence metrics
    calibration_error: float = 0.0  # How well calibrated are confidence scores?
    
    # Time
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def accuracy(self) -> float:
        if self.total_signals == 0:
            return 0.5
        return self.correct_signals / self.total_signals
    
    @property
    def edge(self) -> float:
        """Calculate edge (excess return over random)."""
        return self.accuracy - 0.5


@dataclass
class FeedbackLoop:
    """
    Tracks performance and adjusts strategy weights based on outcomes.
    
    Key Features:
    1. Tracks per-strategy performance
    2. Calculates Bayesian weight adjustments
    3. Regime-conditional adjustments
    4. Decay for old data
    """
    
    storage_path: str = "outputs/feedback_loop.json"
    
    # Performance tracking
    strategy_performance: Dict[str, StrategyPerformance] = field(default_factory=dict)
    
    # Weight adjustments
    weight_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Config
    learning_rate: float = 0.1
    decay_half_life_days: int = 30
    min_samples_for_adjustment: int = 10
    
    def __post_init__(self):
        self.strategy_performance = {}
        self.weight_adjustments = {}
        self._load()
    
    def _load(self):
        """Load feedback data from storage."""
        path = Path(self.storage_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                for name, perf in data.get('strategy_performance', {}).items():
                    self.strategy_performance[name] = StrategyPerformance(
                        strategy_name=name,
                        total_signals=perf.get('total_signals', 0),
                        correct_signals=perf.get('correct_signals', 0),
                        total_return=perf.get('total_return', 0.0),
                        avg_return=perf.get('avg_return', 0.0),
                        return_when_correct=perf.get('return_when_correct', 0.0),
                        return_when_wrong=perf.get('return_when_wrong', 0.0),
                        performance_by_regime=perf.get('performance_by_regime', {}),
                        calibration_error=perf.get('calibration_error', 0.0),
                    )
                
                self.weight_adjustments = data.get('weight_adjustments', {})
                
                logger.info(f"Loaded feedback data for {len(self.strategy_performance)} strategies")
                
            except Exception as e:
                logger.warning(f"Could not load feedback data: {e}")
    
    def _save(self):
        """Save feedback data to storage."""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            perf_data = {}
            for name, perf in self.strategy_performance.items():
                perf_data[name] = {
                    'strategy_name': name,
                    'total_signals': perf.total_signals,
                    'correct_signals': perf.correct_signals,
                    'total_return': perf.total_return,
                    'avg_return': perf.avg_return,
                    'return_when_correct': perf.return_when_correct,
                    'return_when_wrong': perf.return_when_wrong,
                    'performance_by_regime': perf.performance_by_regime,
                    'calibration_error': perf.calibration_error,
                    'last_updated': datetime.now().isoformat(),
                }
            
            with open(path, 'w') as f:
                json.dump({
                    'strategy_performance': perf_data,
                    'weight_adjustments': self.weight_adjustments,
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save feedback data: {e}")
    
    def record_outcome(
        self,
        strategy_name: str,
        was_correct: bool,
        signal_return: float,
        regime: str = "unknown",
        confidence: float = 0.5,
    ):
        """Record the outcome of a signal."""
        
        # Initialize if needed
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )
        
        perf = self.strategy_performance[strategy_name]
        
        # Update counts
        perf.total_signals += 1
        if was_correct:
            perf.correct_signals += 1
        
        # Update returns
        perf.total_return += signal_return
        perf.avg_return = perf.total_return / perf.total_signals
        
        if was_correct:
            # Exponential moving average
            alpha = 0.1
            perf.return_when_correct = (1 - alpha) * perf.return_when_correct + alpha * signal_return
        else:
            alpha = 0.1
            perf.return_when_wrong = (1 - alpha) * perf.return_when_wrong + alpha * abs(signal_return)
        
        # Update regime-specific performance
        if regime not in perf.performance_by_regime:
            perf.performance_by_regime[regime] = 0.0
        
        # Track regime performance as EMA
        regime_contribution = 1.0 if was_correct else -1.0
        perf.performance_by_regime[regime] = (
            0.9 * perf.performance_by_regime[regime] + 0.1 * regime_contribution
        )
        
        # Update calibration error (how well calibrated is confidence?)
        # Error = |confidence - actual_accuracy|
        actual = 1.0 if was_correct else 0.0
        perf.calibration_error = (
            0.9 * perf.calibration_error + 0.1 * abs(confidence - actual)
        )
        
        perf.last_updated = datetime.now()
        
        # Recalculate weight adjustments
        self._update_weight_adjustments()
        
        self._save()
    
    def record_batch_outcomes(
        self,
        outcomes: List[Dict],
    ):
        """Record multiple outcomes at once."""
        for outcome in outcomes:
            self.record_outcome(
                strategy_name=outcome['strategy_name'],
                was_correct=outcome['was_correct'],
                signal_return=outcome.get('signal_return', 0.0),
                regime=outcome.get('regime', 'unknown'),
                confidence=outcome.get('confidence', 0.5),
            )
    
    def _update_weight_adjustments(self):
        """Recalculate weight adjustments based on performance."""
        
        for name, perf in self.strategy_performance.items():
            # Need minimum samples
            if perf.total_signals < self.min_samples_for_adjustment:
                self.weight_adjustments[name] = 0.0
                continue
            
            # Calculate adjustment based on edge
            edge = perf.edge
            
            # Sharpe-like adjustment
            if perf.return_when_correct + perf.return_when_wrong > 0:
                risk_adjusted_edge = edge * (
                    perf.return_when_correct / 
                    max(0.01, perf.return_when_correct + perf.return_when_wrong)
                )
            else:
                risk_adjusted_edge = edge
            
            # Adjust by calibration quality
            # Well-calibrated strategies get more weight
            calibration_factor = max(0.5, 1.0 - perf.calibration_error)
            
            # Final adjustment
            adjustment = risk_adjusted_edge * calibration_factor * self.learning_rate
            
            # Clip to reasonable range
            self.weight_adjustments[name] = max(-0.3, min(0.3, adjustment))
    
    def get_weight_adjustment(
        self,
        strategy_name: str,
        regime: Optional[str] = None,
    ) -> float:
        """
        Get the weight adjustment for a strategy.
        
        Returns a multiplier (e.g., 1.1 = 10% increase, 0.9 = 10% decrease).
        """
        base_adjustment = self.weight_adjustments.get(strategy_name, 0.0)
        
        # Add regime-specific adjustment
        if regime and strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            regime_adj = perf.performance_by_regime.get(regime, 0.0)
            
            # Blend base and regime-specific
            total_adjustment = base_adjustment + 0.5 * regime_adj * self.learning_rate
        else:
            total_adjustment = base_adjustment
        
        # Convert to multiplier
        return 1.0 + total_adjustment
    
    def get_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Apply feedback-based adjustments to base weights.
        
        Returns adjusted weights that sum to 1.
        """
        adjusted = {}
        
        for strategy, weight in base_weights.items():
            multiplier = self.get_weight_adjustment(strategy, regime)
            adjusted[strategy] = weight * multiplier
        
        # Normalize to sum to 1
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all strategy performance."""
        summary = {}
        
        for name, perf in self.strategy_performance.items():
            summary[name] = {
                'accuracy': perf.accuracy,
                'edge': perf.edge,
                'total_signals': perf.total_signals,
                'avg_return': perf.avg_return,
                'weight_adjustment': self.weight_adjustments.get(name, 0.0),
                'is_profitable': perf.edge > 0 and perf.avg_return > 0,
            }
        
        # Sort by edge
        summary = dict(sorted(summary.items(), key=lambda x: x[1]['edge'], reverse=True))
        
        return summary
    
    def get_best_strategies(self, n: int = 3) -> List[str]:
        """Get top N performing strategies."""
        ranked = sorted(
            self.strategy_performance.items(),
            key=lambda x: x[1].edge,
            reverse=True
        )
        return [name for name, _ in ranked[:n]]
    
    def get_worst_strategies(self, n: int = 3) -> List[str]:
        """Get bottom N performing strategies."""
        ranked = sorted(
            self.strategy_performance.items(),
            key=lambda x: x[1].edge,
        )
        return [name for name, _ in ranked[:n]]
    
    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on performance."""
        recommendations = []
        
        # Check for strategies with significant edge
        for name, perf in self.strategy_performance.items():
            if perf.total_signals >= self.min_samples_for_adjustment:
                if perf.edge > 0.1:
                    recommendations.append(
                        f"✅ {name} has strong edge ({perf.edge:.1%}). Consider increasing weight."
                    )
                elif perf.edge < -0.1:
                    recommendations.append(
                        f"⚠️ {name} has negative edge ({perf.edge:.1%}). Consider reducing weight."
                    )
        
        # Check calibration
        poorly_calibrated = [
            name for name, perf in self.strategy_performance.items()
            if perf.calibration_error > 0.3
        ]
        if poorly_calibrated:
            recommendations.append(
                f"🎯 Strategies with poor calibration: {', '.join(poorly_calibrated)}"
            )
        
        # Check regime performance
        for name, perf in self.strategy_performance.items():
            for regime, score in perf.performance_by_regime.items():
                if score > 0.3:
                    recommendations.append(
                        f"📈 {name} performs well in {regime} regime"
                    )
                elif score < -0.3:
                    recommendations.append(
                        f"📉 {name} performs poorly in {regime} regime"
                    )
        
        return recommendations[:10]  # Limit to 10
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "=" * 60)
        print("FEEDBACK LOOP - PERFORMANCE SUMMARY")
        print("=" * 60)
        
        if not self.strategy_performance:
            print("No performance data yet. Run some trades first.")
            return
        
        summary = self.get_performance_summary()
        
        print("\n📊 STRATEGY PERFORMANCE:")
        for name, stats in summary.items():
            status = "✅" if stats['is_profitable'] else "❌"
            print(f"  {status} {name}:")
            print(f"      Accuracy: {stats['accuracy']:.1%}")
            print(f"      Edge: {stats['edge']:+.1%}")
            print(f"      Signals: {stats['total_signals']}")
            print(f"      Weight Adj: {stats['weight_adjustment']:+.1%}")
        
        print("\n🎯 RECOMMENDATIONS:")
        for rec in self.get_recommendations()[:5]:
            print(f"  {rec}")
        
        print("=" * 60)
