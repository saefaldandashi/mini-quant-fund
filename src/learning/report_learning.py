"""
Report Learning - Persistent learning from report insights.

This module:
1. Persists report summaries and performance history
2. Tracks trends over time
3. Learns regime-strategy correlations
4. Provides historical analysis for better decisions

This is the KEY integration between reporting and learning.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class ReportSummary:
    """Summary of a single report for persistence."""
    report_date: str
    report_type: str
    
    # Portfolio metrics
    portfolio_value: float
    return_1d: float
    return_1m: float
    alpha_1d: float
    volatility: float
    sharpe_ratio: float
    drawdown: float
    
    # Macro context
    regime_label: str
    vix: float
    risk_sentiment: str
    
    # Strategy performance
    strategy_weights: Dict[str, float]
    strategy_win_rates: Dict[str, float]
    top_strategies: List[str]
    
    # Holdings
    position_count: int
    total_pnl: float
    
    # Insights extracted
    insight_count: int
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RegimePerformance:
    """Track strategy performance per regime."""
    regime: str
    strategy: str
    observations: int
    avg_return: float
    win_rate: float
    last_updated: str


class ReportLearningStore:
    """
    Persistent store for report-based learning.
    
    Maintains:
    1. Report history - All past report summaries
    2. Performance trends - Rolling averages and trends
    3. Regime correlations - Strategy performance by regime
    4. Learned patterns - Recurring insights
    """
    
    def __init__(self, data_dir: str = "outputs/learning"):
        """
        Initialize the store.
        
        Args:
            data_dir: Directory for persistence files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.history_file = self.data_dir / "report_history.json"
        self.regime_file = self.data_dir / "regime_correlations.json"
        self.trends_file = self.data_dir / "performance_trends.json"
        self.patterns_file = self.data_dir / "learned_patterns.json"
        
        # Load existing data
        self.report_history: List[Dict] = self._load_json(self.history_file, [])
        self.regime_performance: Dict[str, Dict] = self._load_json(self.regime_file, {})
        self.performance_trends: Dict = self._load_json(self.trends_file, {})
        self.learned_patterns: Dict[str, Any] = self._load_json(self.patterns_file, {})
        
        logging.info(f"ReportLearningStore initialized with {len(self.report_history)} historical reports")
    
    def _load_json(self, path: Path, default: Any) -> Any:
        """Load JSON file or return default."""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load {path}: {e}")
        return default
    
    def _save_json(self, path: Path, data: Any):
        """Save data to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save {path}: {e}")
    
    def record_report(self, summary: ReportSummary):
        """
        Record a new report summary.
        
        This is called after each report generation to persist learning.
        """
        # Add to history
        self.report_history.append(summary.to_dict())
        
        # Keep last 365 reports (1 year of daily)
        self.report_history = self.report_history[-365:]
        
        # Update regime correlations
        self._update_regime_correlations(summary)
        
        # Update trends
        self._update_performance_trends(summary)
        
        # Extract patterns
        self._extract_patterns(summary)
        
        # Persist all data
        self._save_all()
        
        logging.info(f"Recorded report for {summary.report_date}, total history: {len(self.report_history)}")
    
    def _update_regime_correlations(self, summary: ReportSummary):
        """Update regime-strategy correlations from report."""
        regime = summary.regime_label.lower()
        
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {}
        
        for strategy, win_rate in summary.strategy_win_rates.items():
            if strategy not in self.regime_performance[regime]:
                self.regime_performance[regime][strategy] = {
                    'observations': 0,
                    'total_win_rate': 0.0,
                    'returns': [],
                }
            
            perf = self.regime_performance[regime][strategy]
            perf['observations'] += 1
            perf['total_win_rate'] += win_rate
            perf['avg_win_rate'] = perf['total_win_rate'] / perf['observations']
            perf['last_updated'] = datetime.now().isoformat()
    
    def _update_performance_trends(self, summary: ReportSummary):
        """Update rolling performance trends."""
        now = summary.report_date
        
        # Track key metrics over time
        metrics = ['return_1d', 'alpha_1d', 'volatility', 'sharpe_ratio', 'drawdown']
        
        for metric in metrics:
            if metric not in self.performance_trends:
                self.performance_trends[metric] = []
            
            value = getattr(summary, metric, 0)
            self.performance_trends[metric].append({
                'date': now,
                'value': value,
            })
            
            # Keep last 90 days
            self.performance_trends[metric] = self.performance_trends[metric][-90:]
        
        # Calculate rolling averages
        self.performance_trends['rolling_metrics'] = {
            'avg_daily_return': self._calc_rolling_avg('return_1d', 20),
            'avg_alpha': self._calc_rolling_avg('alpha_1d', 20),
            'avg_sharpe': self._calc_rolling_avg('sharpe_ratio', 20),
            'trend_return': self._calc_trend('return_1d'),
            'trend_sharpe': self._calc_trend('sharpe_ratio'),
        }
    
    def _calc_rolling_avg(self, metric: str, window: int) -> float:
        """Calculate rolling average for a metric."""
        if metric not in self.performance_trends:
            return 0.0
        
        values = [d['value'] for d in self.performance_trends[metric][-window:]]
        return sum(values) / len(values) if values else 0.0
    
    def _calc_trend(self, metric: str) -> str:
        """Calculate trend direction for a metric."""
        if metric not in self.performance_trends:
            return 'unknown'
        
        values = [d['value'] for d in self.performance_trends[metric][-10:]]
        if len(values) < 3:
            return 'insufficient_data'
        
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if second_half > first_half * 1.05:
            return 'improving'
        elif second_half < first_half * 0.95:
            return 'declining'
        return 'stable'
    
    def _extract_patterns(self, summary: ReportSummary):
        """Extract recurring patterns from report."""
        # Pattern: High VIX + certain strategies
        if summary.vix > 25:
            pattern_key = 'high_vix_performance'
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    'observations': 0,
                    'best_strategies': {},
                    'avg_alpha': 0.0,
                }
            
            self.learned_patterns[pattern_key]['observations'] += 1
            for strategy, wr in summary.strategy_win_rates.items():
                # Handle both defaultdict and regular dict (from JSON load)
                if strategy not in self.learned_patterns[pattern_key]['best_strategies']:
                    self.learned_patterns[pattern_key]['best_strategies'][strategy] = 0.0
                self.learned_patterns[pattern_key]['best_strategies'][strategy] += wr
        
        # Pattern: Bull regime best strategies
        if 'bull' in summary.regime_label.lower():
            pattern_key = 'bull_regime_winners'
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    'observations': 0,
                    'strategy_performance': {},
                }
            
            self.learned_patterns[pattern_key]['observations'] += 1
            for strategy, wr in summary.strategy_win_rates.items():
                # Handle both defaultdict and regular dict (from JSON load)
                if strategy not in self.learned_patterns[pattern_key]['strategy_performance']:
                    self.learned_patterns[pattern_key]['strategy_performance'][strategy] = []
                self.learned_patterns[pattern_key]['strategy_performance'][strategy].append(wr)
        
        # Pattern: Bear regime best strategies
        if 'bear' in summary.regime_label.lower():
            pattern_key = 'bear_regime_winners'
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    'observations': 0,
                    'strategy_performance': {},
                }
            
            self.learned_patterns[pattern_key]['observations'] += 1
            for strategy, wr in summary.strategy_win_rates.items():
                # Handle both defaultdict and regular dict (from JSON load)
                if strategy not in self.learned_patterns[pattern_key]['strategy_performance']:
                    self.learned_patterns[pattern_key]['strategy_performance'][strategy] = []
                self.learned_patterns[pattern_key]['strategy_performance'][strategy].append(wr)
    
    def _save_all(self):
        """Persist all data."""
        self._save_json(self.history_file, self.report_history)
        self._save_json(self.regime_file, self.regime_performance)
        self._save_json(self.trends_file, self.performance_trends)
        
        # Convert defaultdicts to regular dicts for JSON
        patterns_to_save = {}
        for key, value in self.learned_patterns.items():
            patterns_to_save[key] = {}
            for k, v in value.items():
                if isinstance(v, defaultdict):
                    patterns_to_save[key][k] = dict(v)
                else:
                    patterns_to_save[key][k] = v
        self._save_json(self.patterns_file, patterns_to_save)
    
    # === QUERY METHODS FOR LEARNING ===
    
    def get_best_strategies_for_regime(self, regime: str) -> List[Tuple[str, float]]:
        """
        Get best performing strategies for a given regime.
        
        Returns:
            List of (strategy_name, avg_win_rate) tuples, sorted by performance
        """
        regime = regime.lower()
        
        if regime not in self.regime_performance:
            return []
        
        strategies = []
        for strategy, perf in self.regime_performance[regime].items():
            if perf.get('observations', 0) >= 3:  # Minimum observations
                strategies.append((strategy, perf.get('avg_win_rate', 0.5)))
        
        strategies.sort(key=lambda x: x[1], reverse=True)
        return strategies
    
    def get_strategy_weight_recommendation(self, regime: str) -> Dict[str, float]:
        """
        Get recommended strategy weights based on regime and historical performance.
        
        This is the KEY learning output - it tells the system what weights to use
        based on what worked in similar conditions in the past.
        """
        best_strategies = self.get_best_strategies_for_regime(regime)
        
        if not best_strategies:
            return {}
        
        # Calculate recommended weights based on historical win rates
        total_score = sum(wr for _, wr in best_strategies)
        
        if total_score == 0:
            return {}
        
        recommendations = {}
        for strategy, win_rate in best_strategies:
            # Weight = normalized win rate
            recommendations[strategy] = win_rate / total_score
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary for display."""
        rolling = self.performance_trends.get('rolling_metrics', {})
        
        return {
            'total_reports': len(self.report_history),
            'regimes_tracked': list(self.regime_performance.keys()),
            'patterns_found': len(self.learned_patterns),
            'rolling_avg_return': rolling.get('avg_daily_return', 0),
            'rolling_avg_alpha': rolling.get('avg_alpha', 0),
            'rolling_avg_sharpe': rolling.get('avg_sharpe', 0),
            'return_trend': rolling.get('trend_return', 'unknown'),
            'sharpe_trend': rolling.get('trend_sharpe', 'unknown'),
        }
    
    def get_learned_insights(self) -> List[str]:
        """Get human-readable insights from learned patterns."""
        insights = []
        
        # Regime-specific insights
        for regime, strategies in self.regime_performance.items():
            if len(strategies) >= 3:
                best = max(strategies.items(), key=lambda x: x[1].get('avg_win_rate', 0))
                if best[1].get('avg_win_rate', 0) > 0.6:
                    insights.append(
                        f"In {regime.upper()} regime, {best[0]} has historically performed best "
                        f"(avg win rate: {best[1]['avg_win_rate']*100:.0f}%)"
                    )
        
        # Pattern insights
        for pattern_key, pattern_data in self.learned_patterns.items():
            obs = pattern_data.get('observations', 0)
            if obs >= 5:
                if pattern_key == 'high_vix_performance':
                    best_strat = max(
                        pattern_data.get('best_strategies', {}).items(),
                        key=lambda x: x[1],
                        default=(None, 0)
                    )
                    if best_strat[0]:
                        insights.append(
                            f"In high VIX environments (25+), {best_strat[0]} has performed well "
                            f"(based on {obs} observations)"
                        )
        
        # Trend insights
        rolling = self.performance_trends.get('rolling_metrics', {})
        if rolling.get('trend_return') == 'improving':
            insights.append("Portfolio returns are trending upward over the last 10 reports")
        elif rolling.get('trend_return') == 'declining':
            insights.append("Portfolio returns are trending downward - review strategy allocation")
        
        return insights


# === INTEGRATION HELPER ===

def create_report_summary_from_data(data) -> ReportSummary:
    """
    Create a ReportSummary from ReportData.
    
    This is the bridge between report generation and learning persistence.
    """
    from src.reporting.live_collectors import ReportData
    
    # Extract strategy info
    strategy_weights = {}
    strategy_win_rates = {}
    top_strategies = []
    
    for s in data.strategy_performance[:5]:  # Top 5
        strategy_weights[s.name] = s.weight
        strategy_win_rates[s.name] = s.win_rate
    
    if data.strategy_performance:
        top_strategies = [s.name for s in sorted(
            data.strategy_performance, 
            key=lambda x: x.debate_score, 
            reverse=True
        )[:3]]
    
    # Calculate total P/L
    total_pnl = sum(p.unrealized_pnl for p in data.positions)
    
    return ReportSummary(
        report_date=data.report_date.strftime('%Y-%m-%d'),
        report_type=data.report_type,
        portfolio_value=data.portfolio.portfolio_value,
        return_1d=data.portfolio.return_1d,
        return_1m=data.portfolio.return_1m,
        alpha_1d=data.portfolio.alpha_1d,
        volatility=data.portfolio.volatility_20d,
        sharpe_ratio=data.portfolio.sharpe_ratio,
        drawdown=data.portfolio.current_drawdown,
        regime_label=data.macro.regime_label,
        vix=data.macro.vix,
        risk_sentiment=data.macro.risk_sentiment,
        strategy_weights=strategy_weights,
        strategy_win_rates=strategy_win_rates,
        top_strategies=top_strategies,
        position_count=len(data.positions),
        total_pnl=total_pnl,
        insight_count=data.patterns_found,
        recommendations=data.recommendations,
    )
