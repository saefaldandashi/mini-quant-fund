"""
Strategy Performance Tracker - Rolling metrics for each strategy.

Tracks:
- Prediction accuracy (did the strategy's direction prediction come true?)
- Signal-weighted returns (did high-confidence signals perform better?)
- Regime-specific performance (which strategies work in which conditions?)
- Drawdown and risk metrics
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy."""
    strategy_name: str
    
    # Accuracy metrics
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    
    # Return metrics
    cumulative_return: float = 0.0
    avg_return_per_trade: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0  # gross profits / gross losses
    
    # Regime-specific
    performance_by_regime: Dict[str, float] = field(default_factory=dict)
    best_regime: str = ''
    worst_regime: str = ''
    
    # Confidence calibration
    avg_confidence_when_right: float = 0.0
    avg_confidence_when_wrong: float = 0.0
    confidence_calibration: float = 0.0  # Should be close to accuracy
    
    # Rolling metrics (last N trades)
    rolling_accuracy_20: float = 0.0
    rolling_return_20: float = 0.0
    
    # Trend
    improving: bool = False
    momentum_score: float = 0.0  # Positive if getting better
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyMetrics':
        return cls(**data)


class PerformanceTracker:
    """
    Tracks and analyzes strategy performance over time.
    
    Features:
    - Rolling metrics with configurable windows
    - Regime-specific performance breakdown
    - Confidence calibration analysis
    - Trend detection (is strategy improving or degrading?)
    """
    
    def __init__(self, storage_path: str = "outputs/strategy_performance.json"):
        self.storage_path = Path(storage_path)
        self.metrics: Dict[str, StrategyMetrics] = {}
        self.prediction_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Per-symbol performance tracking
        # Maps strategy -> symbol -> {wins, losses, total_pnl}
        self.symbol_performance: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})
        )
        
        self._load()
    
    def _load(self):
        """Load performance history from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for name, metrics_dict in data.get('metrics', {}).items():
                    self.metrics[name] = StrategyMetrics.from_dict(metrics_dict)
                
                self.prediction_history = defaultdict(list, data.get('history', {}))
                
                logging.info(f"Loaded performance data for {len(self.metrics)} strategies")
            except Exception as e:
                logging.warning(f"Could not load performance tracker: {e}")
    
    def _save(self):
        """Persist performance data to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'metrics': {name: m.to_dict() for name, m in self.metrics.items()},
                    'history': dict(self.prediction_history),
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save performance tracker: {e}")
    
    def record_prediction(
        self,
        strategy_name: str,
        symbol: str,
        predicted_direction: str,  # 'long', 'short', 'neutral'
        confidence: float,
        expected_return: float,
        regime: str,
    ):
        """
        Record a strategy's prediction for later evaluation.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Stock symbol
            predicted_direction: 'long', 'short', or 'neutral'
            confidence: Confidence level (0-1)
            expected_return: Expected return from strategy
            regime: Current market regime
        """
        self.prediction_history[strategy_name].append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': predicted_direction,
            'confidence': confidence,
            'expected_return': expected_return,
            'regime': regime,
            'outcome': None,  # To be filled later
            'actual_return': None,
        })
        
        # Keep only last 500 predictions per strategy
        if len(self.prediction_history[strategy_name]) > 500:
            self.prediction_history[strategy_name] = self.prediction_history[strategy_name][-500:]
        
        self._save()
    
    def record_outcome(
        self,
        strategy_name: str,
        symbol: str,
        actual_return: float,
        prediction_timestamp: Optional[str] = None,
    ):
        """
        Record the actual outcome for a prediction.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Stock symbol
            actual_return: Actual return achieved
            prediction_timestamp: Timestamp of original prediction (optional)
        """
        history = self.prediction_history.get(strategy_name, [])
        
        # Find the most recent prediction for this symbol
        for pred in reversed(history):
            if pred['symbol'] == symbol and pred['outcome'] is None:
                pred['actual_return'] = actual_return
                
                # Determine if prediction was correct
                if pred['direction'] == 'long':
                    pred['outcome'] = 'correct' if actual_return > 0 else 'incorrect'
                elif pred['direction'] == 'short':
                    pred['outcome'] = 'correct' if actual_return < 0 else 'incorrect'
                else:
                    pred['outcome'] = 'correct' if abs(actual_return) < 0.02 else 'incorrect'
                
                # Update per-symbol performance
                sym_perf = self.symbol_performance[strategy_name][symbol]
                if pred['outcome'] == 'correct':
                    sym_perf['wins'] += 1
                else:
                    sym_perf['losses'] += 1
                sym_perf['total_pnl'] += actual_return
                
                break
        
        # Recalculate metrics
        self._recalculate_metrics(strategy_name)
        self._save()
    
    def _recalculate_metrics(self, strategy_name: str):
        """Recalculate all metrics for a strategy."""
        history = self.prediction_history.get(strategy_name, [])
        evaluated = [p for p in history if p['outcome'] is not None]
        
        if not evaluated:
            return
        
        if strategy_name not in self.metrics:
            self.metrics[strategy_name] = StrategyMetrics(strategy_name=strategy_name)
        
        m = self.metrics[strategy_name]
        
        # Basic accuracy
        m.total_predictions = len(evaluated)
        m.correct_predictions = len([p for p in evaluated if p['outcome'] == 'correct'])
        m.accuracy = m.correct_predictions / m.total_predictions if m.total_predictions > 0 else 0
        
        # Return metrics
        returns = [p['actual_return'] for p in evaluated if p['actual_return'] is not None]
        if returns:
            m.cumulative_return = sum(returns)
            m.avg_return_per_trade = np.mean(returns)
            m.volatility = np.std(returns) if len(returns) > 1 else 0
            m.sharpe_ratio = m.avg_return_per_trade / m.volatility if m.volatility > 0 else 0
            
            # Win rate and profit factor
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            m.win_rate = len(wins) / len(returns) if returns else 0
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0.001
            m.profit_factor = gross_profit / gross_loss
            
            # Max drawdown (simplified)
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            m.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Regime-specific performance
        regime_returns = defaultdict(list)
        for p in evaluated:
            if p['actual_return'] is not None:
                regime_returns[p['regime']].append(p['actual_return'])
        
        m.performance_by_regime = {
            regime: np.mean(rets) for regime, rets in regime_returns.items()
        }
        
        if m.performance_by_regime:
            m.best_regime = max(m.performance_by_regime, key=m.performance_by_regime.get)
            m.worst_regime = min(m.performance_by_regime, key=m.performance_by_regime.get)
        
        # Confidence calibration
        correct = [p for p in evaluated if p['outcome'] == 'correct']
        incorrect = [p for p in evaluated if p['outcome'] == 'incorrect']
        
        m.avg_confidence_when_right = np.mean([p['confidence'] for p in correct]) if correct else 0
        m.avg_confidence_when_wrong = np.mean([p['confidence'] for p in incorrect]) if incorrect else 0
        
        # Good calibration: confident when right, less confident when wrong
        m.confidence_calibration = m.avg_confidence_when_right - m.avg_confidence_when_wrong
        
        # Rolling metrics (last 20)
        recent = evaluated[-20:]
        if recent:
            m.rolling_accuracy_20 = len([p for p in recent if p['outcome'] == 'correct']) / len(recent)
            recent_returns = [p['actual_return'] for p in recent if p['actual_return'] is not None]
            m.rolling_return_20 = sum(recent_returns) if recent_returns else 0
        
        # Trend detection
        if len(evaluated) >= 40:
            old_accuracy = len([p for p in evaluated[-40:-20] if p['outcome'] == 'correct']) / 20
            new_accuracy = m.rolling_accuracy_20
            m.momentum_score = new_accuracy - old_accuracy
            m.improving = m.momentum_score > 0.05
    
    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """
        Get strategies ranked by overall performance score.
        
        Returns:
            List of (strategy_name, score) tuples, sorted by score descending
        """
        scores = []
        
        for name, m in self.metrics.items():
            # Composite score weighing multiple factors
            score = (
                m.accuracy * 0.25 +
                m.sharpe_ratio * 0.20 +
                m.win_rate * 0.15 +
                min(m.profit_factor, 3.0) / 3.0 * 0.15 +  # Cap profit factor contribution
                m.confidence_calibration * 0.10 +
                m.momentum_score * 0.15  # Recent improvement matters
            )
            scores.append((name, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def get_regime_specialists(self, regime: str) -> List[Tuple[str, float]]:
        """
        Get strategies ranked by performance in a specific regime.
        
        Args:
            regime: Market regime to check
        
        Returns:
            List of (strategy_name, performance) tuples
        """
        specialists = []
        
        for name, m in self.metrics.items():
            perf = m.performance_by_regime.get(regime, 0.0)
            specialists.append((name, perf))
        
        return sorted(specialists, key=lambda x: x[1], reverse=True)
    
    def get_learning_insights(self) -> Dict[str, any]:
        """
        Generate insights about what the system has learned.
        
        Returns:
            Dict with learning insights and recommendations
        """
        insights = {
            'top_strategies': [],
            'struggling_strategies': [],
            'regime_insights': {},
            'calibration_issues': [],
            'improving_strategies': [],
            'recommendations': [],
        }
        
        ranking = self.get_strategy_ranking()
        
        if ranking:
            insights['top_strategies'] = ranking[:3]
            insights['struggling_strategies'] = ranking[-3:]
        
        # Find calibration issues
        for name, m in self.metrics.items():
            if m.avg_confidence_when_wrong > m.avg_confidence_when_right:
                insights['calibration_issues'].append({
                    'strategy': name,
                    'issue': 'Overconfident when wrong',
                    'recommendation': f"Reduce weight for {name} when confidence is high"
                })
            
            if m.improving:
                insights['improving_strategies'].append(name)
        
        # Regime insights
        regimes = set()
        for m in self.metrics.values():
            regimes.update(m.performance_by_regime.keys())
        
        for regime in regimes:
            specialists = self.get_regime_specialists(regime)
            if specialists:
                best = specialists[0]
                worst = specialists[-1]
                insights['regime_insights'][regime] = {
                    'best_strategy': best[0],
                    'best_performance': best[1],
                    'worst_strategy': worst[0],
                    'avoid_strategy': worst[0] if worst[1] < -0.01 else None,
                }
        
        # Generate recommendations
        if insights['calibration_issues']:
            insights['recommendations'].append(
                "Some strategies are overconfident when wrong - consider confidence discounting"
            )
        
        if insights['improving_strategies']:
            insights['recommendations'].append(
                f"Strategies improving recently: {', '.join(insights['improving_strategies'][:3])}"
            )
        
        return insights
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of all strategy performance."""
        return {
            'strategies_tracked': len(self.metrics),
            'total_predictions': sum(m.total_predictions for m in self.metrics.values()),
            'overall_accuracy': np.mean([m.accuracy for m in self.metrics.values()]) if self.metrics else 0,
            'best_strategy': self.get_strategy_ranking()[0] if self.get_strategy_ranking() else None,
            'insights': self.get_learning_insights(),
        }
