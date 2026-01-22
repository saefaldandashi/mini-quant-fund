"""
Learning Engine - Central coordinator for all learning components.

Brings together:
- Trade Memory: Stores all trades with context
- Performance Tracker: Tracks strategy accuracy
- Adaptive Weights: Learns optimal strategy weights
- Pattern Learner: Discovers market patterns

Provides a unified interface for the trading system to:
- Record trades and outcomes
- Get learning-adjusted weights
- Receive insights and recommendations
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .trade_memory import TradeMemory, TradeRecord
from .performance_tracker import PerformanceTracker, StrategyMetrics
from .adaptive_weights import AdaptiveWeightLearner
from .pattern_learner import PatternLearner


class LearningEngine:
    """
    Central coordinator for all learning components.
    
    This is the main interface the trading system uses to:
    1. Record trades and their outcomes
    2. Get learning-influenced weights
    3. Receive recommendations and insights
    4. Track what the system has learned
    """
    
    def __init__(
        self,
        strategy_names: List[str],
        outputs_dir: str = "outputs",
        learning_influence: float = 0.3,
    ):
        """
        Initialize the learning engine.
        
        Args:
            strategy_names: List of strategy names to track
            outputs_dir: Directory for persisting learning data
            learning_influence: Base learning influence (0-1), will scale up with data
        """
        self.strategy_names = strategy_names
        self.base_learning_influence = learning_influence
        self.learning_influence = learning_influence  # Will be updated dynamically
        
        # Initialize components
        self.trade_memory = TradeMemory(f"{outputs_dir}/trade_memory.json")
        self.performance_tracker = PerformanceTracker(f"{outputs_dir}/strategy_performance.json")
        self.adaptive_weights = AdaptiveWeightLearner(
            strategy_names, f"{outputs_dir}/learned_weights.json"
        )
        self.pattern_learner = PatternLearner(f"{outputs_dir}/patterns.json")
        
        # Learning state
        self.last_signals: Dict[str, Dict] = {}
        self.last_regime: Optional[str] = None
        self.pending_predictions: Dict[str, List[Dict]] = {}
        
        logging.info(f"Learning engine initialized for {len(strategy_names)} strategies")
    
    def get_adaptive_learning_influence(self) -> float:
        """
        Dynamically adjust learning influence based on data collected.
        
        The more data we have, the more we trust the learning system.
        This allows the system to gradually increase influence as it gains experience.
        
        Returns:
            Learning influence factor (0.2 to 0.7)
        """
        try:
            stats = self.trade_memory.get_statistics()
            total_trades = stats.get('total_trades', 0)
            win_rate = stats.get('win_rate', 0.5)
        except Exception:
            total_trades = 0
            win_rate = 0.5
        
        # Scale influence based on number of trades
        if total_trades < 10:
            influence = 0.2  # Low influence, still learning
        elif total_trades < 30:
            influence = 0.35  # Moderate influence
        elif total_trades < 100:
            influence = 0.5  # High influence
        else:
            influence = 0.65  # Strong influence - trust the learning
        
        # Boost further if win rate is good
        if win_rate > 0.55 and total_trades >= 30:
            influence = min(0.7, influence * 1.15)
        
        # Reduce if win rate is poor
        if win_rate < 0.45 and total_trades >= 30:
            influence = max(0.2, influence * 0.85)
        
        self.learning_influence = influence
        return influence
    
    def record_signals(
        self,
        strategy_signals: Dict[str, Dict],
        debate_scores: Dict[str, float],
        regime: str,
        market_context: Dict[str, Any],
    ):
        """
        Record strategy signals before trade execution.
        
        This is called before trades are placed to record what each
        strategy predicted. The outcomes will be matched later.
        
        Args:
            strategy_signals: Dict of strategy_name -> signal info
            debate_scores: Dict of strategy_name -> debate score
            regime: Current market regime
            market_context: Current market conditions
        """
        self.last_signals = strategy_signals
        self.last_regime = regime
        
        # Record predictions for each strategy
        for strategy_name, signal in strategy_signals.items():
            weights = signal.get('weights', {})
            confidence = signal.get('confidence', 0.5)
            expected_return = signal.get('expected_return', 0.0)
            
            for symbol, weight in weights.items():
                if abs(weight) > 0.01:
                    direction = 'long' if weight > 0 else 'short'
                    
                    self.performance_tracker.record_prediction(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        predicted_direction=direction,
                        confidence=confidence,
                        expected_return=expected_return,
                        regime=regime,
                    )
                    
                    # Track pending predictions for outcome matching
                    if strategy_name not in self.pending_predictions:
                        self.pending_predictions[strategy_name] = []
                    
                    self.pending_predictions[strategy_name].append({
                        'symbol': symbol,
                        'direction': direction,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat(),
                    })
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        ensemble_weight: float,
        ensemble_mode: str,
        leverage_used: float = 1.0,
        margin_cost_daily: float = 0.0,
        leverage_state: str = 'healthy',
        intended_holding_minutes: int = 0,
        entry_strategy: str = '',
    ):
        """
        Record an executed trade.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            price: Execution price
            ensemble_weight: Final weight from ensemble
            ensemble_mode: Mode used by ensemble
            leverage_used: Leverage ratio at trade time (1.0 = no leverage)
            margin_cost_daily: Daily margin interest cost for leveraged trades
            leverage_state: State of leverage manager at trade time
            intended_holding_minutes: Expected holding period (0 = no limit)
            entry_strategy: Primary strategy that triggered this trade
        """
        # Store leverage info for trade memory
        self._leverage_info = {
            'leverage_used': leverage_used,
            'margin_cost_daily': margin_cost_daily,
            'leverage_state': leverage_state,
        }
        # Build strategy signals list from last recorded signals
        strategy_signals = []
        for strategy_name, signal in self.last_signals.items():
            weights = signal.get('weights', {})
            strategy_signals.append({
                'name': strategy_name,
                'weight': weights.get(symbol, 0.0),
                'confidence': signal.get('confidence', 0.5),
                'expected_return': signal.get('expected_return', 0.0),
                'debate_score': signal.get('debate_score', 0.0),
                'explanation': signal.get('explanation', {}),
            })
        
        # Build market context
        market_context = {
            'regime': self.last_regime or 'unknown',
            'volatility_regime': 'medium',  # Will be filled in by caller
            'trend_strength': 0.0,
            'correlation_regime': 'medium',
            'spy_return_1d': 0.0,
            'spy_return_5d': 0.0,
        }
        
        self.trade_memory.record_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            strategy_signals=strategy_signals,
            ensemble_weight=ensemble_weight,
            ensemble_mode=ensemble_mode,
            market_context=market_context,
            leverage_used=leverage_used,
            margin_cost_daily=margin_cost_daily,
            leverage_state=leverage_state,
            intended_holding_minutes=intended_holding_minutes,
            entry_strategy=entry_strategy,
        )
    
    def record_outcomes(
        self,
        symbol_returns: Dict[str, float],
        market_context: Dict[str, Any],
    ):
        """
        Record trade outcomes and update learning.
        
        This should be called periodically (e.g., daily) to update
        the learning system with actual returns.
        
        Args:
            symbol_returns: Dict of symbol -> actual return
            market_context: Market conditions during this period
        """
        # Update trade memory with current prices
        current_prices = {}
        for symbol, ret in symbol_returns.items():
            # Estimate current price from return (simplified)
            trades = [t for t in self.trade_memory.trades if t.symbol == symbol]
            if trades:
                last_trade = trades[-1]
                current_prices[symbol] = last_trade.entry_price * (1 + ret)
        
        self.trade_memory.update_position_prices(current_prices)
        
        # Calculate strategy-level returns
        strategy_returns = {}
        winning_strategies = []
        losing_strategies = []
        
        for strategy_name, predictions in self.pending_predictions.items():
            total_return = 0.0
            count = 0
            
            for pred in predictions:
                symbol = pred['symbol']
                if symbol in symbol_returns:
                    actual_ret = symbol_returns[symbol]
                    
                    # Record outcome in performance tracker
                    self.performance_tracker.record_outcome(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        actual_return=actual_ret,
                    )
                    
                    # Accumulate strategy return
                    if pred['direction'] == 'long':
                        total_return += actual_ret
                    else:
                        total_return -= actual_ret
                    count += 1
            
            if count > 0:
                avg_return = total_return / count
                strategy_returns[strategy_name] = avg_return
                
                if avg_return > 0.005:
                    winning_strategies.append(strategy_name)
                elif avg_return < -0.005:
                    losing_strategies.append(strategy_name)
        
        # Update adaptive weights
        if strategy_returns:
            regime = market_context.get('regime', 'unknown')
            self.adaptive_weights.update_from_outcomes(strategy_returns, regime)
        
        # Record observation for pattern learning
        strategy_signals = {
            name: sum(p['confidence'] for p in preds) / len(preds) if preds else 0
            for name, preds in self.pending_predictions.items()
        }
        
        total_return = np.mean(list(symbol_returns.values())) if symbol_returns else 0
        
        self.pattern_learner.record_observation(
            market_context=market_context,
            strategy_signals=strategy_signals,
            outcome_return=total_return,
            winning_strategies=winning_strategies,
            losing_strategies=losing_strategies,
        )
        
        # Clear pending predictions
        self.pending_predictions = {}
        
        logging.info(
            f"Recorded outcomes: {len(symbol_returns)} symbols, "
            f"{len(winning_strategies)} winning strategies, "
            f"{len(losing_strategies)} losing strategies"
        )
    
    def get_learned_weights(
        self,
        debate_scores: Dict[str, float],
        market_context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Get strategy weights that incorporate learning.
        
        Blends debate engine scores with learned weights based on
        historical performance.
        
        The learning influence is DYNAMIC - it increases as we collect more data
        and have higher confidence in what we've learned.
        
        Args:
            debate_scores: Scores from the debate engine
            market_context: Current market conditions
        
        Returns:
            Blended strategy weights
        """
        regime = market_context.get('regime', 'unknown')
        
        # Get ADAPTIVE learning influence based on data collected
        current_influence = self.get_adaptive_learning_influence()
        
        # Get base learned weights with adaptive influence
        learned_weights = self.adaptive_weights.blend_with_debate_scores(
            debate_scores=debate_scores,
            regime=regime,
            learned_weight_influence=current_influence,
        )
        
        logging.debug(f"Learning influence: {current_influence:.1%} (adaptive)")
        
        # ============================================================
        # PATTERN-BASED ADJUSTMENTS (More aggressive)
        # ============================================================
        # Get strategy recommendations from pattern learner
        recommended, to_avoid = self.pattern_learner.get_strategy_recommendations(
            market_context
        )
        
        # Get active patterns for even stronger adjustments
        active_patterns = self.pattern_learner.get_active_patterns(market_context)
        
        # Track applied adjustments for logging
        adjustments_applied = []
        
        # Apply pattern-based boosts/penalties
        for pattern in active_patterns:
            if pattern.confidence >= 0.6 and pattern.times_observed >= 5:
                # This is a high-confidence pattern - apply stronger adjustments
                
                # Boost recommended strategies
                for strategy in pattern.recommended_strategies:
                    if strategy in learned_weights:
                        boost = 1.3 if pattern.confidence > 0.75 else 1.2
                        learned_weights[strategy] *= boost
                        adjustments_applied.append(f"+{(boost-1)*100:.0f}% {strategy} (pattern: {pattern.description[:30]})")
                
                # Penalize strategies to avoid
                for strategy in pattern.strategies_to_avoid:
                    if strategy in learned_weights:
                        penalty = 0.7 if pattern.confidence > 0.75 else 0.8
                        learned_weights[strategy] *= penalty
                        adjustments_applied.append(f"-{(1-penalty)*100:.0f}% {strategy} (avoid: {pattern.description[:30]})")
        
        # Apply general recommendations with moderate adjustments
        for strategy in recommended:
            if strategy in learned_weights and strategy not in [
                s for p in active_patterns for s in p.recommended_strategies
            ]:
                learned_weights[strategy] *= 1.15  # 15% boost
        
        for strategy in to_avoid:
            if strategy in learned_weights and strategy not in [
                s for p in active_patterns for s in p.strategies_to_avoid
            ]:
                learned_weights[strategy] *= 0.85  # 15% penalty
        
        # Log adjustments if any were made
        if adjustments_applied:
            logging.info(f"Pattern-based adjustments: {len(adjustments_applied)} applied")
            for adj in adjustments_applied[:3]:  # Log top 3
                logging.debug(f"  {adj}")
        
        # Renormalize
        total = sum(learned_weights.values())
        if total > 0:
            learned_weights = {k: v / total for k, v in learned_weights.items()}
        
        return learned_weights
    
    def get_recommendations(
        self, market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get trading recommendations based on learning.
        
        Args:
            market_context: Current market conditions
        
        Returns:
            Dict with recommendations and insights
        """
        # Get pattern recommendations
        recommended, to_avoid = self.pattern_learner.get_strategy_recommendations(
            market_context
        )
        
        # Get risk signals
        risk_signals = self.pattern_learner.get_risk_signals(market_context)
        
        # Get performance insights
        perf_insights = self.performance_tracker.get_learning_insights()
        
        # Get exploration recommendations
        explore = self.adaptive_weights.get_exploration_recommendations()
        
        # Get active patterns
        active_patterns = self.pattern_learner.get_active_patterns(market_context)
        
        return {
            'recommended_strategies': recommended,
            'strategies_to_avoid': to_avoid,
            'risk_warnings': risk_signals,
            'exploration_suggestions': explore,
            'active_patterns': [
                {
                    'description': p.description,
                    'confidence': p.confidence,
                    'action': p.recommended_action,
                }
                for p in active_patterns[:3]
            ],
            'performance_insights': perf_insights,
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of what the system has learned.
        
        Returns:
            Dict with all learning statistics and insights
        """
        trade_stats = self.trade_memory.get_statistics()
        perf_summary = self.performance_tracker.get_summary()
        weight_summary = self.adaptive_weights.get_learning_summary()
        pattern_summary = self.pattern_learner.get_learning_summary()
        
        # Include report-based learning if available
        report_learning = {}
        try:
            from .report_learning import ReportLearningStore
            report_store = ReportLearningStore()
            report_learning = report_store.get_performance_summary()
        except Exception:
            pass
        
        return {
            'trade_history': {
                'total_trades': trade_stats.get('total_trades', 0),
                'win_rate': trade_stats.get('win_rate', 0),
                'total_pnl': trade_stats.get('total_pnl', 0),
                'avg_pnl_percent': trade_stats.get('avg_pnl_percent', 0),
            },
            'strategy_performance': {
                'strategies_tracked': perf_summary.get('strategies_tracked', 0),
                'total_predictions': perf_summary.get('total_predictions', 0),
                'overall_accuracy': perf_summary.get('overall_accuracy', 0),
                'best_strategy': perf_summary.get('best_strategy'),
            },
            'learned_weights': {
                'learning_rounds': weight_summary.get('total_learning_rounds', 0),
                'top_strategies': weight_summary.get('top_strategies', []),
                'current_weights': weight_summary.get('current_weights', {}),
            },
            'patterns': {
                'total_patterns': pattern_summary.get('total_patterns', 0),
                'discovered_patterns': pattern_summary.get('discovered_patterns', 0),
                'top_patterns': pattern_summary.get('top_patterns', []),
            },
            'report_learning': report_learning,
            'patterns_found': report_learning.get('patterns_found', 0) + pattern_summary.get('discovered_patterns', 0),
            'recommendations': self._get_all_recommendations(),
            'learning_influence': self.learning_influence,
            'is_learning': trade_stats.get('total_trades', 0) >= 5 or report_learning.get('total_reports', 0) >= 3,
        }
    
    def _get_all_recommendations(self) -> List[str]:
        """Aggregate recommendations from all learning sources."""
        recommendations = []
        
        # From pattern learner
        try:
            pattern_summary = self.pattern_learner.get_learning_summary()
            for pattern in pattern_summary.get('top_patterns', [])[:3]:
                if isinstance(pattern, dict):
                    recommendations.append(pattern.get('description', str(pattern)))
                else:
                    recommendations.append(str(pattern))
        except Exception:
            pass
        
        # From report learning store
        try:
            from .report_learning import ReportLearningStore
            report_store = ReportLearningStore()
            learned = report_store.get_learned_insights()
            recommendations.extend(learned[:3])
        except Exception:
            pass
        
        return recommendations[:5]
    
    def get_regime_adjusted_weights(self, base_weights: Dict[str, float], regime: str) -> Dict[str, float]:
        """
        Adjust strategy weights based on regime-specific learned performance.
        
        This is the KEY method that applies report-based learning to trading decisions.
        
        Args:
            base_weights: The base strategy weights from debate/ensemble
            regime: Current market regime
        
        Returns:
            Adjusted weights based on what worked in similar conditions
        """
        try:
            from .report_learning import ReportLearningStore
            report_store = ReportLearningStore()
            
            # Get recommended weights for this regime
            regime_weights = report_store.get_strategy_weight_recommendation(regime)
            
            if not regime_weights:
                return base_weights
            
            # Blend base weights with learned weights
            adjusted = {}
            blend_factor = 0.3  # 30% influence from learning
            
            for strategy, base_weight in base_weights.items():
                learned_weight = regime_weights.get(strategy, base_weight)
                adjusted[strategy] = (1 - blend_factor) * base_weight + blend_factor * learned_weight
            
            # Normalize to sum to 1
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v/total for k, v in adjusted.items()}
            
            logging.info(f"Applied regime-adjusted weights for {regime} regime")
            return adjusted
            
        except Exception as e:
            logging.warning(f"Could not get regime-adjusted weights: {e}")
            return base_weights
    
    def get_mistake_analysis(self) -> Dict[str, Any]:
        """
        Analyze mistakes and what can be learned from them.
        
        Returns:
            Dict with mistake analysis and lessons
        """
        losing_trades = self.trade_memory.get_losing_trades()
        
        if not losing_trades:
            return {
                'total_losing_trades': 0,
                'common_patterns': [],
                'lessons': ["Not enough data to analyze mistakes yet"],
            }
        
        # Analyze common patterns in losing trades
        regime_counts = {}
        strategy_counts = {}
        
        for trade in losing_trades:
            if trade.market_context:
                regime = trade.market_context.regime
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            for sig in trade.strategy_signals:
                if sig.debate_score > 0.3:
                    strategy_counts[sig.strategy_name] = strategy_counts.get(
                        sig.strategy_name, 0
                    ) + 1
        
        # Find most common losing conditions
        worst_regime = max(regime_counts, key=regime_counts.get) if regime_counts else None
        worst_strategy = max(strategy_counts, key=strategy_counts.get) if strategy_counts else None
        
        lessons = []
        
        if worst_regime:
            lessons.append(
                f"Most losses occur in {worst_regime} regime - "
                f"consider reducing exposure in this condition"
            )
        
        if worst_strategy:
            lessons.append(
                f"{worst_strategy} is associated with most losing trades - "
                f"consider reducing its weight or improving its signals"
            )
        
        # Calculate average loss by holding period
        by_holding_period = {}
        for trade in losing_trades:
            if trade.holding_period_days is not None:
                bucket = 'short' if trade.holding_period_days <= 5 else 'long'
                if bucket not in by_holding_period:
                    by_holding_period[bucket] = []
                by_holding_period[bucket].append(trade.pnl_percent or 0)
        
        if by_holding_period:
            for bucket, losses in by_holding_period.items():
                avg_loss = np.mean(losses)
                if abs(avg_loss) > 2:
                    lessons.append(
                        f"{bucket.capitalize()}-term trades have avg loss of {avg_loss:.1f}% - "
                        f"review {bucket}-term position sizing"
                    )
        
        return {
            'total_losing_trades': len(losing_trades),
            'worst_regime': worst_regime,
            'worst_strategy': worst_strategy,
            'regime_distribution': regime_counts,
            'strategy_distribution': strategy_counts,
            'lessons': lessons or ["Continue building data for better analysis"],
        }
    
    def should_override_decision(
        self,
        proposed_action: str,
        market_context: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if learning suggests overriding a proposed decision.
        
        This provides a safety check based on learned patterns.
        
        Args:
            proposed_action: The action being considered
            market_context: Current market conditions
        
        Returns:
            Tuple of (should_override, reason)
        """
        risk_signals = self.pattern_learner.get_risk_signals(market_context)
        
        if proposed_action == 'increase_exposure' and risk_signals:
            return True, f"Risk signals active: {risk_signals[0]}"
        
        # Check if we're repeating a known mistake
        active_patterns = self.pattern_learner.get_active_patterns(market_context)
        
        for pattern in active_patterns:
            if (pattern.avg_return < -0.03 and 
                pattern.times_observed >= 5 and
                pattern.confidence < 0.4):
                return True, f"Pattern '{pattern.description}' has poor history"
        
        return False, None
