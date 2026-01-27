"""
Item Selector for Daily Digest.

Selects top N items per category using:
- impact_score (primary)
- source_tier (secondary)  
- timestamp (recency tie-break)

Applies configurable thresholds to filter noise.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pytz

from .schema import (
    FeedItem, 
    DigestCategory, 
    CATEGORY_MAPPING,
    SOURCE_CREDIBILITY,
)

logger = logging.getLogger(__name__)


class SelectionConfig:
    """Configuration for item selection."""
    
    def __init__(
        self,
        items_per_category: int = 20,  # Very high - comprehensive coverage
        impact_threshold: float = 0.30,  # Reasonable threshold for quality
        novelty_threshold: float = 0.0,  # Disabled - handled by dedup
        min_credibility: float = 0.0,  # Disabled - include all sources
        max_age_hours: int = 72,  # 3 days for comprehensive coverage
        boost_tier1_sources: bool = True,
        dedupe_by_headline_similarity: float = 0.90,  # Very high - only exact dupes
    ):
        self.items_per_category = items_per_category
        self.impact_threshold = impact_threshold
        self.novelty_threshold = novelty_threshold
        self.min_credibility = min_credibility
        self.max_age_hours = max_age_hours
        self.boost_tier1_sources = boost_tier1_sources
        self.dedupe_by_headline_similarity = dedupe_by_headline_similarity


class SelectionResult:
    """Result of selection for one category."""
    
    def __init__(
        self,
        category: DigestCategory,
        selected: List[FeedItem],
        total_considered: int,
        total_passed_threshold: int,
        reasons: List[str],
    ):
        self.category = category
        self.selected = selected
        self.total_considered = total_considered
        self.total_passed_threshold = total_passed_threshold
        self.reasons = reasons
        
    @property
    def avg_impact(self) -> float:
        if not self.selected:
            return 0.0
        return sum(i.impact_score for i in self.selected) / len(self.selected)
    
    @property
    def max_impact(self) -> float:
        if not self.selected:
            return 0.0
        return max(i.impact_score for i in self.selected)


class ItemSelector:
    """Selects top items per category for digest."""
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        
    def select(
        self, 
        items: List[FeedItem],
        target_date: Optional[datetime] = None,
    ) -> Dict[DigestCategory, SelectionResult]:
        """
        Select top N items per category.
        
        Args:
            items: All feed items to consider
            target_date: Date to filter by (uses today if not specified)
            
        Returns:
            Dict mapping categories to selection results
        """
        if target_date is None:
            target_date = datetime.now(pytz.UTC)
        elif target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=pytz.UTC)
            
        # Group items by digest category
        by_category = self._group_by_category(items)
        
        results = {}
        for category, category_items in by_category.items():
            result = self._select_for_category(category, category_items, target_date)
            results[category] = result
            
        return results
    
    def _group_by_category(
        self, 
        items: List[FeedItem]
    ) -> Dict[DigestCategory, List[FeedItem]]:
        """Group items by their digest category."""
        grouped = defaultdict(list)
        
        for item in items:
            # Map feed category to digest category
            digest_cat = CATEGORY_MAPPING.get(item.category)
            if digest_cat is None:
                # Try uppercase
                digest_cat = CATEGORY_MAPPING.get(item.category.upper())
            if digest_cat is None:
                # Try lowercase
                digest_cat = CATEGORY_MAPPING.get(item.category.lower())
                
            if digest_cat is not None:
                item.digest_category = digest_cat
                item.source_tier = self._get_source_tier(item.source)
                grouped[digest_cat].append(item)
            else:
                logger.debug(f"No mapping for category: {item.category}")
                
        return grouped
    
    def _get_source_tier(self, source: str) -> int:
        """Get source credibility tier (1=best, 5=default)."""
        source_lower = source.lower()
        for key, tier in SOURCE_CREDIBILITY.items():
            if key in source_lower:
                return tier
        return SOURCE_CREDIBILITY.get("default", 5)
    
    def _select_for_category(
        self,
        category: DigestCategory,
        items: List[FeedItem],
        target_date: datetime,
    ) -> SelectionResult:
        """Select top N items for a single category."""
        
        reasons = []
        total_considered = len(items)
        
        # Step 1: Apply thresholds
        cutoff = target_date - timedelta(hours=self.config.max_age_hours)
        filtered = []
        
        for item in items:
            # Ensure timestamp has timezone
            item_ts = item.timestamp
            if item_ts.tzinfo is None:
                item_ts = item_ts.replace(tzinfo=pytz.UTC)
            
            # Check age
            if item_ts < cutoff:
                continue
                
            # Check impact threshold
            if item.impact_score < self.config.impact_threshold:
                continue
                
            # Check novelty threshold
            if item.novelty_score < self.config.novelty_threshold:
                continue
                
            # Check credibility
            if item.credibility_score < self.config.min_credibility:
                continue
                
            filtered.append(item)
            
        total_passed = len(filtered)
        reasons.append(f"{total_passed}/{total_considered} passed thresholds")
        
        if not filtered:
            return SelectionResult(
                category=category,
                selected=[],
                total_considered=total_considered,
                total_passed_threshold=0,
                reasons=reasons,
            )
        
        # Step 2: Score and rank
        scored = []
        for item in filtered:
            # Composite score: impact (60%) + source tier (25%) + recency (15%)
            impact_component = item.impact_score * 0.60
            
            # Source tier: tier 1 = 1.0, tier 5 = 0.2
            tier_score = 1.0 - (item.source_tier - 1) * 0.2
            tier_component = tier_score * 0.25
            
            # Recency: items from last 6 hours get boost
            item_ts = item.timestamp
            if item_ts.tzinfo is None:
                item_ts = item_ts.replace(tzinfo=pytz.UTC)
            age_hours = (target_date - item_ts).total_seconds() / 3600
            recency_score = max(0, 1.0 - age_hours / 48)  # Linear decay over 48h
            recency_component = recency_score * 0.15
            
            composite = impact_component + tier_component + recency_component
            scored.append((composite, item))
            
        # Sort by composite score descending
        scored.sort(key=lambda x: (-x[0], -x[1].impact_score))
        
        # Step 3: Deduplicate by headline similarity
        selected = []
        seen_headlines = []
        
        for score, item in scored:
            if len(selected) >= self.config.items_per_category:
                break
                
            # Check headline similarity
            if self._is_duplicate(item.headline, seen_headlines):
                continue
                
            # Add selection reason
            item.selection_reason = (
                f"Impact: {item.impact_score:.2f}, "
                f"Tier: {item.source_tier}, "
                f"Score: {score:.3f}"
            )
            
            selected.append(item)
            seen_headlines.append(item.headline.lower())
            
        reasons.append(f"Selected top {len(selected)} after ranking")
        
        return SelectionResult(
            category=category,
            selected=selected,
            total_considered=total_considered,
            total_passed_threshold=total_passed,
            reasons=reasons,
        )
    
    def _is_duplicate(self, headline: str, seen: List[str]) -> bool:
        """Check if headline is too similar to already seen headlines."""
        headline_lower = headline.lower()
        
        for seen_headline in seen:
            similarity = self._headline_similarity(headline_lower, seen_headline)
            if similarity >= self.config.dedupe_by_headline_similarity:
                return True
                
        return False
    
    def _headline_similarity(self, h1: str, h2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(h1.split())
        words2 = set(h2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)


def select_items(
    items: List[FeedItem],
    items_per_category: int = 20,  # Comprehensive coverage
    impact_threshold: float = 0.30,  # All notable events
    novelty_threshold: float = 0.0,  # Disabled - handled by dedup
) -> Dict[DigestCategory, SelectionResult]:
    """
    Convenience function to select items.
    
    Args:
        items: Feed items to select from
        items_per_category: Number of items per category (default: 20)
        impact_threshold: Minimum impact score (default: 0.30)
        novelty_threshold: Minimum novelty score (default: 0 - disabled)
        
    Returns:
        Dict of selection results by category
    """
    config = SelectionConfig(
        items_per_category=items_per_category,
        impact_threshold=impact_threshold,
        novelty_threshold=novelty_threshold,
    )
    selector = ItemSelector(config)
    return selector.select(items)
