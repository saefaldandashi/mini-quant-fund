"""
Relevance Gate - Filter out irrelevant news before processing.

Critical for preventing noise from contaminating signals.
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import re
import logging

from .taxonomy import MacroTaxonomy, TaxonomyTag, TaxonomyMatch
from .sources import SourceCredibility, SourceTier

logger = logging.getLogger(__name__)


@dataclass
class RelevanceResult:
    """Result of relevance filtering."""
    is_relevant: bool
    relevance_score: float  # 0-1
    taxonomy_matches: List[TaxonomyMatch]
    rejection_reason: Optional[str] = None


class RelevanceGate:
    """
    Filter news articles for market relevance.
    
    Applies multi-stage filtering:
    1. Hard reject patterns (sports, entertainment, local news)
    2. Taxonomy matching (must match macro/geo/finance tags)
    3. Source credibility (unknown sources get lower scores)
    4. Confidence threshold
    """
    
    # Additional hard reject patterns (very aggressive filtering)
    HARD_REJECT_PATTERNS = [
        # Sports
        r'\b(super bowl|world series|stanley cup|wimbledon|olympics)\b',
        r'\b(coach|player|team|season|game|match|score|win|lose|defeat)\b(?!.*\b(fed|market|stock|economy)\b)',
        
        # Entertainment
        r'\b(netflix|disney\+|hulu|streaming show|tv series|blockbuster)\b(?!.*\b(stock|earnings|revenue)\b)',
        r'\b(grammy|oscar|emmy|golden globe|award show)\b',
        
        # Local/State news
        r'\b(local news|community|neighborhood|town hall)\b',
        r'\b(high school|elementary school|middle school)\b(?!.*\beducation policy\b)',
        r'\b(county|municipality|city council)\b(?!.*\b(budget|fiscal|debt)\b)',
        
        # Lifestyle
        r'\b(recipe|cooking|restaurant review|food blog)\b',
        r'\b(travel guide|vacation|tourist|hotel review)\b(?!.*\b(travel spending|tourism gdp)\b)',
        r'\b(fashion week|runway|style tip)\b',
        
        # Non-market
        r'\b(obituary|funeral|memorial service)\b',
        r'\b(weather forecast|storm warning|temperature)\b(?!.*\b(commodit|crop|energy)\b)',
    ]
    
    # Patterns that REQUIRE market context
    CONTEXT_REQUIRED_PATTERNS = [
        # Generic words that need financial context
        (r'\bchina\b', r'\b(trade|tariff|economy|market|yuan|currency|growth|gdp|export|import)\b'),
        (r'\brussia\b', r'\b(sanction|oil|gas|energy|economy|war|conflict|trade)\b'),
        (r'\belection\b', r'\b(market|economy|policy|fiscal|tax|regulation|investor)\b'),
    ]
    
    def __init__(
        self,
        taxonomy: Optional[MacroTaxonomy] = None,
        source_credibility: Optional[SourceCredibility] = None,
        min_taxonomy_confidence: float = 0.3,
        min_source_weight: float = 0.1,
        require_tier_1_for_low_confidence: bool = True,
    ):
        """
        Initialize relevance gate.
        
        Args:
            taxonomy: MacroTaxonomy classifier
            source_credibility: SourceCredibility system
            min_taxonomy_confidence: Minimum confidence for taxonomy match
            min_source_weight: Minimum source weight to consider
            require_tier_1_for_low_confidence: Require tier 1 source for borderline articles
        """
        self.taxonomy = taxonomy or MacroTaxonomy()
        self.source_credibility = source_credibility or SourceCredibility()
        self.min_taxonomy_confidence = min_taxonomy_confidence
        self.min_source_weight = min_source_weight
        self.require_tier_1_for_low_confidence = require_tier_1_for_low_confidence
        
        # Compile patterns
        self._hard_reject_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.HARD_REJECT_PATTERNS
        ]
        self._context_required_compiled = [
            (re.compile(trigger, re.IGNORECASE), re.compile(context, re.IGNORECASE))
            for trigger, context in self.CONTEXT_REQUIRED_PATTERNS
        ]
    
    def filter(
        self,
        title: str,
        body: str = "",
        source: str = "",
    ) -> RelevanceResult:
        """
        Apply relevance filtering to an article.
        
        Args:
            title: Article title/headline
            body: Article body text
            source: Source name
            
        Returns:
            RelevanceResult with decision and details
        """
        text = f"{title} {body}"
        
        # Stage 1: Hard reject patterns
        for pattern in self._hard_reject_compiled:
            if pattern.search(text):
                return RelevanceResult(
                    is_relevant=False,
                    relevance_score=0.0,
                    taxonomy_matches=[],
                    rejection_reason=f"Hard reject pattern: {pattern.pattern[:50]}",
                )
        
        # Stage 2: Context-required patterns
        for trigger, context in self._context_required_compiled:
            if trigger.search(text) and not context.search(text):
                return RelevanceResult(
                    is_relevant=False,
                    relevance_score=0.1,
                    taxonomy_matches=[],
                    rejection_reason=f"Missing required context for: {trigger.pattern}",
                )
        
        # Stage 3: Taxonomy matching
        taxonomy_matches = self.taxonomy.classify(title, body)
        
        if not taxonomy_matches:
            return RelevanceResult(
                is_relevant=False,
                relevance_score=0.1,
                taxonomy_matches=[],
                rejection_reason="No taxonomy match",
            )
        
        max_confidence = taxonomy_matches[0].confidence
        
        # Stage 4: Source credibility
        source_info = self.source_credibility.get_source_info(source)
        source_weight = source_info.weight
        
        if source_weight < self.min_source_weight:
            return RelevanceResult(
                is_relevant=False,
                relevance_score=max_confidence * source_weight,
                taxonomy_matches=taxonomy_matches,
                rejection_reason=f"Source too low credibility: {source} ({source_info.tier.value})",
            )
        
        # Stage 5: Confidence threshold with source adjustment
        adjusted_confidence = max_confidence * (0.5 + 0.5 * source_weight)
        
        # Borderline cases: require high-tier source
        if adjusted_confidence < self.min_taxonomy_confidence:
            if self.require_tier_1_for_low_confidence:
                if source_info.tier not in [SourceTier.TIER_1, SourceTier.TIER_2]:
                    return RelevanceResult(
                        is_relevant=False,
                        relevance_score=adjusted_confidence,
                        taxonomy_matches=taxonomy_matches,
                        rejection_reason="Low confidence + non-tier-1 source",
                    )
        
        # Passed all filters
        relevance_score = min(1.0, adjusted_confidence + source_weight * 0.2)
        
        return RelevanceResult(
            is_relevant=True,
            relevance_score=relevance_score,
            taxonomy_matches=taxonomy_matches,
            rejection_reason=None,
        )
    
    def is_relevant(self, title: str, body: str = "", source: str = "") -> bool:
        """Simple boolean check for relevance."""
        return self.filter(title, body, source).is_relevant
    
    def batch_filter(
        self,
        articles: List[dict],
        title_key: str = "title",
        body_key: str = "body",
        source_key: str = "source",
    ) -> Tuple[List[dict], List[dict]]:
        """
        Filter a batch of articles.
        
        Returns:
            Tuple of (relevant_articles, rejected_articles)
        """
        relevant = []
        rejected = []
        
        for article in articles:
            result = self.filter(
                title=article.get(title_key, ""),
                body=article.get(body_key, ""),
                source=article.get(source_key, ""),
            )
            
            if result.is_relevant:
                article['_relevance_score'] = result.relevance_score
                article['_taxonomy_matches'] = result.taxonomy_matches
                relevant.append(article)
            else:
                article['_rejection_reason'] = result.rejection_reason
                rejected.append(article)
        
        logger.info(
            f"Relevance filter: {len(relevant)} relevant, {len(rejected)} rejected "
            f"({len(relevant)/(len(articles) or 1)*100:.1f}% pass rate)"
        )
        
        return relevant, rejected
