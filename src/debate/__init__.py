"""Debate and ensemble modules."""
from .debate_engine import DebateEngine, StrategyScore, DebateTranscript
from .ensemble import EnsembleOptimizer, EnsembleMode
from .adversarial import AdversarialDebateEngine, AdversarialTranscript, Argument, ArgumentType

__all__ = [
    'DebateEngine',
    'StrategyScore',
    'DebateTranscript',
    'EnsembleOptimizer',
    'EnsembleMode',
    'AdversarialDebateEngine',
    'AdversarialTranscript',
    'Argument',
    'ArgumentType',
]
