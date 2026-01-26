"""
LLM Module - Uses Large Language Models for complex reasoning tasks.

Key principle: LLM for reasoning, not for routine tasks.
Use LLM only where it adds value over simple rules.

Providers:
- Gemini (primary, free tier)
- OpenAI (fallback)
- Anthropic (fallback)

Features:
- Auto-selection of best available provider
- Smart fallback when provider fails
- Multi-provider parallel execution for consensus
"""
from .event_extractor import LLMEventExtractor, MacroEvent
from .theme_synthesizer import ThemeSynthesizer
from .llm_service import LLMService
from .debate_arguments import LLMDebateArgumentGenerator, DebateArgument
from .multi_provider import MultiProviderLLM, MultiProviderResponse, get_multi_llm

__all__ = [
    'LLMEventExtractor',
    'MacroEvent',
    'ThemeSynthesizer',
    'LLMService',
    'LLMDebateArgumentGenerator',
    'DebateArgument',
    'MultiProviderLLM',
    'MultiProviderResponse',
    'get_multi_llm',
]
