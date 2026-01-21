"""
LLM Module - Uses Large Language Models for complex reasoning tasks.

Key principle: LLM for reasoning, not for routine tasks.
Use LLM only where it adds value over simple rules.
"""
from .event_extractor import LLMEventExtractor, MacroEvent
from .theme_synthesizer import ThemeSynthesizer
from .llm_service import LLMService
from .debate_arguments import LLMDebateArgumentGenerator, DebateArgument

__all__ = [
    'LLMEventExtractor',
    'MacroEvent',
    'ThemeSynthesizer',
    'LLMService',
    'LLMDebateArgumentGenerator',
    'DebateArgument',
]
