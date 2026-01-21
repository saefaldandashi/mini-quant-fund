"""
Impact Scoring - Score news articles by expected market impact.

Combines source credibility, novelty, severity, and timing.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import pytz

from .sources import SourceCredibility, SourceTier
from .events import MacroEvent, EventDirection
from .taxonomy import TaxonomyTag

logger = logging.getLogger(__name__)


@dataclass
class ImpactBreakdown:
    """Breakdown of impact score components."""
    source_weight: float
    novelty_score: float
    severity_score: float
    timing_score: float
    systemic_score: float
    total_impact: float


class ImpactScorer:
    """
    Calculate market impact scores for news events.
    
    Factors:
    - Source credibility weight
    - Novelty (duplicate detection, decay)
    - Severity language
    - Proximity to scheduled releases
    - Systemic keywords
    """
    
    # High-impact systemic keywords
    SYSTEMIC_KEYWORDS = [
        'systemic', 'contagion', 'cascade', 'emergency',
        'bailout', 'rescue', 'intervention', 'liquidity injection',
        'rate surprise', 'unexpected', 'shock', 'unprecedented',
        'sanctions expanded', 'shipping disruption', 'supply chain',
        'banking stress', 'credit crisis', 'default risk',
        'nuclear', 'escalation', 'invasion',
    ]
    
    # Timing windows for scheduled releases (hour of day, tag)
    SCHEDULED_RELEASE_WINDOWS = {
        TaxonomyTag.MACRO_INFLATION: [8, 9, 10],  # CPI usually 8:30 ET
        TaxonomyTag.MACRO_LABOR: [8, 9],  # Jobs report 8:30 ET
        TaxonomyTag.MACRO_GROWTH: [8, 9, 10],
        TaxonomyTag.CENTRAL_BANK: [14, 15],  # FOMC 2pm ET
    }
    
    # Tag impact weights (some events more market-moving)
    TAG_IMPACT_WEIGHTS = {
        TaxonomyTag.CENTRAL_BANK: 1.2,
        TaxonomyTag.FINANCIAL_STRESS: 1.3,
        TaxonomyTag.GEOPOLITICS: 1.1,
        TaxonomyTag.MACRO_INFLATION: 1.1,
        TaxonomyTag.MACRO_LABOR: 1.0,
        TaxonomyTag.MACRO_GROWTH: 1.0,
        TaxonomyTag.FISCAL_POLICY: 0.9,
        TaxonomyTag.COMMODITIES_ENERGY: 0.9,
        TaxonomyTag.FX_TRADE: 0.8,
        TaxonomyTag.REGULATION: 0.8,
        TaxonomyTag.MARKET_GENERAL: 0.7,
    }
    
    def __init__(
        self,
        source_credibility: Optional[SourceCredibility] = None,
        novelty_halflife_hours: float = 6.0,
    ):
        """
        Initialize impact scorer.
        
        Args:
            source_credibility: Source credibility system
            novelty_halflife_hours: Half-life for novelty decay
        """
        self.source_credibility = source_credibility or SourceCredibility()
        self.novelty_halflife = timedelta(hours=novelty_halflife_hours)
        
        # Track seen events for novelty calculation
        self._seen_events: Dict[str, datetime] = {}
    
    def score(
        self,
        event: MacroEvent,
        current_time: Optional[datetime] = None,
    ) -> ImpactBreakdown:
        """
        Calculate impact score for an event.
        
        Args:
            event: MacroEvent to score
            current_time: Current time for decay calculations
            
        Returns:
            ImpactBreakdown with component scores
        """
        current_time = current_time or datetime.now(pytz.UTC)
        
        # 1. Source credibility weight
        source_weight = self.source_credibility.get_weight(event.source)
        
        # 2. Novelty score (decays for repeated news)
        novelty_score = self._calculate_novelty(event, current_time)
        
        # 3. Severity score (from event)
        severity_score = event.severity_score
        
        # 4. Timing score (proximity to scheduled releases)
        timing_score = self._calculate_timing_score(event)
        
        # 5. Systemic score (presence of systemic keywords)
        systemic_score = self._calculate_systemic_score(event)
        
        # Combine scores
        # Formula: weighted combination with tag multiplier
        base_impact = (
            source_weight * 0.25 +
            novelty_score * 0.25 +
            severity_score * 0.20 +
            timing_score * 0.15 +
            systemic_score * 0.15
        )
        
        # Apply tag weight multiplier
        tag_weight = 1.0
        if event.tags:
            tag_weights = [
                self.TAG_IMPACT_WEIGHTS.get(tag, 1.0)
                for tag in event.tags
            ]
            tag_weight = max(tag_weights)
        
        total_impact = min(1.0, base_impact * tag_weight)
        
        return ImpactBreakdown(
            source_weight=source_weight,
            novelty_score=novelty_score,
            severity_score=severity_score,
            timing_score=timing_score,
            systemic_score=systemic_score,
            total_impact=total_impact,
        )
    
    def _calculate_novelty(
        self,
        event: MacroEvent,
        current_time: datetime,
    ) -> float:
        """Calculate novelty score with decay for repeated news."""
        # Use event's pre-calculated novelty
        base_novelty = event.novelty_score
        
        # Apply time decay if we've seen similar events
        event_key = self._make_event_key(event)
        
        if event_key in self._seen_events:
            last_seen = self._seen_events[event_key]
            time_since = current_time - last_seen
            
            # Exponential decay
            decay_factor = 0.5 ** (time_since.total_seconds() / 
                                   self.novelty_halflife.total_seconds())
            base_novelty *= (1.0 - decay_factor * 0.5)
        
        # Store this event
        self._seen_events[event_key] = event.event_time
        
        # Clean old entries
        cutoff = current_time - timedelta(days=1)
        self._seen_events = {
            k: v for k, v in self._seen_events.items()
            if v > cutoff
        }
        
        return base_novelty
    
    def _make_event_key(self, event: MacroEvent) -> str:
        """Create a key for deduplication."""
        # Combine tags and entities for a unique signature
        tags_str = "_".join(t.value for t in event.tags[:2])
        entities_str = "_".join(sorted(event.entities[:3]))
        return f"{tags_str}_{entities_str}_{event.direction.value}"
    
    def _calculate_timing_score(self, event: MacroEvent) -> float:
        """Score based on timing relative to scheduled releases."""
        if not event.tags:
            return 0.5
        
        event_hour = event.event_time.hour
        
        for tag in event.tags:
            if tag in self.SCHEDULED_RELEASE_WINDOWS:
                expected_hours = self.SCHEDULED_RELEASE_WINDOWS[tag]
                if event_hour in expected_hours:
                    # Event during expected release window = higher impact
                    return 0.9
                elif abs(event_hour - min(expected_hours)) <= 2:
                    return 0.7
        
        return 0.5
    
    def _calculate_systemic_score(self, event: MacroEvent) -> float:
        """Score based on presence of systemic risk keywords."""
        text = f"{event.title} {event.rationale}".lower()
        
        matches = 0
        for keyword in self.SYSTEMIC_KEYWORDS:
            if keyword in text:
                matches += 1
        
        # Scale: 0 matches = 0.3, 1+ = higher
        if matches == 0:
            return 0.3
        elif matches == 1:
            return 0.6
        elif matches == 2:
            return 0.8
        else:
            return 1.0
    
    def batch_score(
        self,
        events: List[MacroEvent],
        current_time: Optional[datetime] = None,
    ) -> List[ImpactBreakdown]:
        """Score multiple events."""
        return [self.score(event, current_time) for event in events]
    
    def get_high_impact_events(
        self,
        events: List[MacroEvent],
        threshold: float = 0.6,
        current_time: Optional[datetime] = None,
    ) -> List[tuple]:
        """
        Filter to high-impact events.
        
        Returns:
            List of (event, impact_breakdown) tuples above threshold
        """
        results = []
        
        for event in events:
            breakdown = self.score(event, current_time)
            if breakdown.total_impact >= threshold:
                results.append((event, breakdown))
        
        # Sort by impact descending
        results.sort(key=lambda x: x[1].total_impact, reverse=True)
        
        return results
