"""
News Intelligence Layer - Macro, Geopolitics, and Financial News Analysis.

High signal, low noise news processing for market-moving events.
"""

from .taxonomy import MacroTaxonomy, TaxonomyTag
from .sources import SourceCredibility, SourceTier
from .relevance import RelevanceGate
from .events import EventExtractor, MacroEvent
from .impact import ImpactScorer
from .sentiment import RiskSentimentAnalyzer
from .aggregator import MacroFeatureAggregator, DailyMacroFeatures
from .pipeline import NewsIntelligencePipeline

__all__ = [
    'MacroTaxonomy',
    'TaxonomyTag',
    'SourceCredibility',
    'SourceTier',
    'RelevanceGate',
    'EventExtractor',
    'MacroEvent',
    'ImpactScorer',
    'RiskSentimentAnalyzer',
    'MacroFeatureAggregator',
    'DailyMacroFeatures',
    'NewsIntelligencePipeline',
]
