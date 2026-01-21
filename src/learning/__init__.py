"""
Learning system for continuous improvement of trading strategies.

This module provides:
- TradeMemory: Persistent storage of trades with full context
- PerformanceTracker: Rolling metrics for each strategy
- AdaptiveWeightLearner: Learns optimal strategy weights over time
- PatternLearner: Learns regime-specific patterns
- MistakeAnalyzer: Identifies and learns from losing trades
- DebateLearner: Learns from debate outcomes to improve future debates
- OutcomeTracker: Tracks signals and their outcomes
- SignalValidator: Validates signals before trading
- FeedbackLoop: Connects outcomes to weight adjustments
"""

from .trade_memory import TradeMemory, TradeRecord
from .performance_tracker import PerformanceTracker, StrategyMetrics
from .adaptive_weights import AdaptiveWeightLearner
from .pattern_learner import PatternLearner
from .learning_engine import LearningEngine
from .debate_learner import DebateLearner, DebateRecord, StrategyDebateProfile
from .outcome_tracker import OutcomeTracker
from .signal_validator import SignalValidator
from .feedback_loop import FeedbackLoop

__all__ = [
    'TradeMemory',
    'TradeRecord', 
    'PerformanceTracker',
    'StrategyMetrics',
    'AdaptiveWeightLearner',
    'PatternLearner',
    'LearningEngine',
    'DebateLearner',
    'DebateRecord',
    'StrategyDebateProfile',
    'OutcomeTracker',
    'SignalValidator',
    'FeedbackLoop',
]
