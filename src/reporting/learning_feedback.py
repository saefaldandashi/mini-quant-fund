"""
Learning Feedback - Extracts insights from reports and feeds them to the learning system.

This closes the loop: Reports → Learning → Better Decisions
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .live_collectors import ReportData, Position, StrategyPerformance


@dataclass
class TrendInsight:
    """Extracted trend from report data."""
    category: str  # 'strategy', 'regime', 'position', 'execution'
    description: str
    confidence: float
    action_suggested: Optional[str] = None
    data: Dict = None


class ReportLearningFeedback:
    """
    Extracts insights from reports and feeds them to the learning system.
    
    This enables:
    1. Strategy weight adjustments based on historical performance
    2. Pattern detection from trends
    3. Regime-strategy correlation learning
    4. Execution pattern optimization
    """
    
    def __init__(self, learning_engine=None):
        """
        Initialize feedback loop.
        
        Args:
            learning_engine: LearningEngine instance to update
        """
        self.learning_engine = learning_engine
        self.insight_history: List[TrendInsight] = []
    
    def extract_and_learn(self, data: ReportData) -> List[TrendInsight]:
        """
        Extract insights from report data and update learning system.
        
        Args:
            data: ReportData from report generation
        
        Returns:
            List of extracted insights
        """
        insights = []
        
        # 1. Strategy performance trends
        strategy_insights = self._analyze_strategy_performance(data)
        insights.extend(strategy_insights)
        
        # 2. Regime correlation insights
        regime_insights = self._analyze_regime_correlation(data)
        insights.extend(regime_insights)
        
        # 3. Position concentration insights
        position_insights = self._analyze_position_patterns(data)
        insights.extend(position_insights)
        
        # 4. Performance attribution insights
        attribution_insights = self._analyze_attribution(data)
        insights.extend(attribution_insights)
        
        # Store insights
        self.insight_history.extend(insights)
        
        # Update learning system
        if self.learning_engine:
            self._update_learning_system(insights, data)
        
        return insights
    
    def _analyze_strategy_performance(self, data: ReportData) -> List[TrendInsight]:
        """Analyze strategy performance and extract trends."""
        insights = []
        
        for strategy in data.strategy_performance:
            # High performing strategy
            if strategy.win_rate >= 0.6 and strategy.confidence >= 0.7:
                insights.append(TrendInsight(
                    category='strategy',
                    description=f"{strategy.name} is performing well (win rate: {strategy.win_rate*100:.0f}%)",
                    confidence=strategy.confidence,
                    action_suggested=f"Consider increasing weight for {strategy.name}",
                    data={'strategy': strategy.name, 'win_rate': strategy.win_rate},
                ))
            
            # Underperforming strategy
            elif strategy.win_rate < 0.4:
                insights.append(TrendInsight(
                    category='strategy',
                    description=f"{strategy.name} is underperforming (win rate: {strategy.win_rate*100:.0f}%)",
                    confidence=0.7,
                    action_suggested=f"Consider reducing weight for {strategy.name}",
                    data={'strategy': strategy.name, 'win_rate': strategy.win_rate},
                ))
            
            # High debate score but low weight (missed opportunity)
            if strategy.debate_score > 0.7 and strategy.weight < 0.1:
                insights.append(TrendInsight(
                    category='strategy',
                    description=f"{strategy.name} has strong debate support but low weight",
                    confidence=0.6,
                    action_suggested=f"Review weight allocation for {strategy.name}",
                    data={'strategy': strategy.name, 'debate_score': strategy.debate_score},
                ))
        
        return insights
    
    def _analyze_regime_correlation(self, data: ReportData) -> List[TrendInsight]:
        """Analyze regime-strategy correlations."""
        insights = []
        
        regime = data.macro.regime_label.lower()
        
        # Check which strategies perform well in current regime
        for strategy in data.strategy_performance:
            # In risk-off regime, defensive strategies should outperform
            if 'bear' in regime or 'risk_off' in regime:
                if 'momentum' in strategy.name.lower() and strategy.win_rate < 0.4:
                    insights.append(TrendInsight(
                        category='regime',
                        description=f"Momentum strategies underperforming in risk-off regime as expected",
                        confidence=0.8,
                        action_suggested="Reduce momentum exposure in risk-off regimes",
                        data={'regime': regime, 'strategy': strategy.name},
                    ))
                
                if ('risk' in strategy.name.lower() or 'overlay' in strategy.name.lower()) and strategy.win_rate > 0.6:
                    insights.append(TrendInsight(
                        category='regime',
                        description=f"Risk overlays performing well in current regime",
                        confidence=0.8,
                        action_suggested="Maintain defensive positioning",
                        data={'regime': regime, 'strategy': strategy.name},
                    ))
            
            # In risk-on regime, momentum should do well
            elif 'bull' in regime or 'risk_on' in regime:
                if 'momentum' in strategy.name.lower() and strategy.win_rate > 0.6:
                    insights.append(TrendInsight(
                        category='regime',
                        description=f"Momentum strategies performing well in risk-on regime",
                        confidence=0.8,
                        data={'regime': regime, 'strategy': strategy.name},
                    ))
        
        # High VIX environment
        if data.macro.vix > 25:
            insights.append(TrendInsight(
                category='regime',
                description=f"Elevated VIX ({data.macro.vix:.1f}) suggests increased volatility",
                confidence=0.9,
                action_suggested="Consider reducing position sizes and increasing hedges",
                data={'vix': data.macro.vix},
            ))
        
        return insights
    
    def _analyze_position_patterns(self, data: ReportData) -> List[TrendInsight]:
        """Analyze position concentration and patterns."""
        insights = []
        
        if not data.positions:
            return insights
        
        # Check concentration
        top_3_weight = sum(p.weight for p in data.positions[:3])
        if top_3_weight > 0.5:
            insights.append(TrendInsight(
                category='position',
                description=f"Portfolio concentrated: top 3 positions = {top_3_weight*100:.0f}%",
                confidence=0.9,
                action_suggested="Consider diversifying to reduce concentration risk",
                data={'top_3_weight': top_3_weight},
            ))
        
        # Check for large unrealized losses
        for pos in data.positions:
            if pos.unrealized_pnl_pct < -20:
                insights.append(TrendInsight(
                    category='position',
                    description=f"{pos.symbol} has significant unrealized loss ({pos.unrealized_pnl_pct:.0f}%)",
                    confidence=0.8,
                    action_suggested=f"Review thesis for {pos.symbol} - consider stop-loss",
                    data={'symbol': pos.symbol, 'pnl_pct': pos.unrealized_pnl_pct},
                ))
        
        # Check for large gains (consider trimming)
        for pos in data.positions:
            if pos.unrealized_pnl_pct > 50:
                insights.append(TrendInsight(
                    category='position',
                    description=f"{pos.symbol} has large unrealized gain ({pos.unrealized_pnl_pct:.0f}%)",
                    confidence=0.6,
                    action_suggested=f"Consider trimming {pos.symbol} to lock in gains",
                    data={'symbol': pos.symbol, 'pnl_pct': pos.unrealized_pnl_pct},
                ))
        
        return insights
    
    def _analyze_attribution(self, data: ReportData) -> List[TrendInsight]:
        """Analyze performance attribution."""
        insights = []
        
        # Portfolio vs benchmark
        if data.portfolio.alpha_1d > 0.01:  # 1% outperformance
            insights.append(TrendInsight(
                category='performance',
                description=f"Portfolio outperformed benchmark by {data.portfolio.alpha_1d*100:.1f}% today",
                confidence=0.9,
                data={'alpha': data.portfolio.alpha_1d},
            ))
        elif data.portfolio.alpha_1d < -0.01:  # 1% underperformance
            insights.append(TrendInsight(
                category='performance',
                description=f"Portfolio underperformed benchmark by {abs(data.portfolio.alpha_1d)*100:.1f}% today",
                confidence=0.9,
                action_suggested="Review position selection and timing",
                data={'alpha': data.portfolio.alpha_1d},
            ))
        
        # Drawdown analysis
        if data.portfolio.current_drawdown < -0.05:  # 5% drawdown
            insights.append(TrendInsight(
                category='risk',
                description=f"Portfolio in drawdown: {data.portfolio.current_drawdown*100:.1f}%",
                confidence=0.9,
                action_suggested="Consider reducing exposure until drawdown recovers",
                data={'drawdown': data.portfolio.current_drawdown},
            ))
        
        # Sharpe ratio
        if data.portfolio.sharpe_ratio < 0.5:
            insights.append(TrendInsight(
                category='risk',
                description=f"Low risk-adjusted returns (Sharpe: {data.portfolio.sharpe_ratio:.2f})",
                confidence=0.7,
                action_suggested="Review portfolio construction for better risk-adjusted returns",
                data={'sharpe': data.portfolio.sharpe_ratio},
            ))
        elif data.portfolio.sharpe_ratio > 2.0:
            insights.append(TrendInsight(
                category='performance',
                description=f"Excellent risk-adjusted returns (Sharpe: {data.portfolio.sharpe_ratio:.2f})",
                confidence=0.8,
                data={'sharpe': data.portfolio.sharpe_ratio},
            ))
        
        return insights
    
    def _update_learning_system(self, insights: List[TrendInsight], data: ReportData):
        """Feed insights back to learning system."""
        if not self.learning_engine:
            return
        
        try:
            # Update strategy weights based on performance
            for strategy in data.strategy_performance:
                if hasattr(self.learning_engine, 'update_strategy_performance'):
                    self.learning_engine.update_strategy_performance(
                        strategy.name,
                        win_rate=strategy.win_rate,
                        regime=data.macro.regime_label,
                    )
            
            # Record regime-strategy correlation
            if hasattr(self.learning_engine, 'record_regime_correlation'):
                for strategy in data.strategy_performance:
                    self.learning_engine.record_regime_correlation(
                        regime=data.macro.regime_label,
                        strategy=strategy.name,
                        performance=strategy.contribution,
                    )
            
            # Record pattern observations
            if hasattr(self.learning_engine, 'record_pattern'):
                for insight in insights:
                    if insight.confidence >= 0.7:
                        self.learning_engine.record_pattern(
                            category=insight.category,
                            description=insight.description,
                            data=insight.data,
                        )
            
            logging.info(f"Fed {len(insights)} insights back to learning system")
        
        except Exception as e:
            logging.warning(f"Could not update learning system: {e}")
    
    def get_recent_insights(self, limit: int = 20) -> List[TrendInsight]:
        """Get recent insights."""
        return self.insight_history[-limit:]
    
    def get_actionable_recommendations(self) -> List[str]:
        """Get actionable recommendations from recent insights."""
        recommendations = []
        
        for insight in self.insight_history[-50:]:
            if insight.action_suggested and insight.confidence >= 0.6:
                recommendations.append(insight.action_suggested)
        
        # Deduplicate while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]
