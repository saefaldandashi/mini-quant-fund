"""
Macro Feature Aggregator - Produce daily macro features from events.

Aggregates events into trading-system-ready features.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import logging
import pytz

from .taxonomy import TaxonomyTag
from .events import MacroEvent, EventDirection
from .sentiment import RiskSentiment

logger = logging.getLogger(__name__)


@dataclass
class DailyMacroFeatures:
    """Daily aggregated macro features for trading."""
    date: datetime
    
    # Core indices (all in range [-1, 1] or [0, 1])
    macro_inflation_pressure_index: float = 0.0  # Higher = more inflation pressure
    labor_strength_index: float = 0.0  # Higher = stronger labor market
    growth_momentum_index: float = 0.0  # Higher = stronger growth
    central_bank_hawkishness_index: float = 0.0  # Higher = more hawkish
    geopolitical_risk_index: float = 0.0  # Higher = more risk
    financial_stress_index: float = 0.0  # Higher = more stress
    commodities_supply_risk_index: float = 0.0  # Higher = supply concerns
    
    # Overall risk sentiment
    overall_risk_sentiment_index: float = 0.0  # [-1, 1]: risk-off to risk-on
    
    # Directional biases
    equity_bias: float = 0.0
    rates_bias: float = 0.0
    dollar_bias: float = 0.0
    
    # Metadata
    event_count: int = 0
    high_impact_event_count: int = 0
    data_quality_score: float = 0.0  # 0-1: confidence in features
    
    # Top events for explanation
    top_events: List[str] = field(default_factory=list)


class MacroFeatureAggregator:
    """
    Aggregate macro events into daily features.
    
    Uses decayed sums with source weighting and duplicate handling.
    All features are timestamp-aligned and safe for backtests.
    """
    
    # Tag to feature index mapping
    TAG_TO_FEATURE: Dict[TaxonomyTag, str] = {
        TaxonomyTag.MACRO_INFLATION: 'macro_inflation_pressure_index',
        TaxonomyTag.MACRO_LABOR: 'labor_strength_index',
        TaxonomyTag.MACRO_GROWTH: 'growth_momentum_index',
        TaxonomyTag.CENTRAL_BANK: 'central_bank_hawkishness_index',
        TaxonomyTag.GEOPOLITICS: 'geopolitical_risk_index',
        TaxonomyTag.FINANCIAL_STRESS: 'financial_stress_index',
        TaxonomyTag.COMMODITIES_ENERGY: 'commodities_supply_risk_index',
    }
    
    # Direction to sign mapping for each feature
    DIRECTION_SIGNS: Dict[str, Dict[EventDirection, float]] = {
        'macro_inflation_pressure_index': {
            EventDirection.INFLATION_UP: 1.0,
            EventDirection.INFLATION_DOWN: -1.0,
            EventDirection.HAWKISH: 0.3,  # Hawkish implies inflation concern
        },
        'labor_strength_index': {
            EventDirection.LABOR_STRONG: 1.0,
            EventDirection.LABOR_WEAK: -1.0,
        },
        'growth_momentum_index': {
            EventDirection.GROWTH_UP: 1.0,
            EventDirection.GROWTH_DOWN: -1.0,
            EventDirection.RISK_ON: 0.3,
            EventDirection.RISK_OFF: -0.3,
        },
        'central_bank_hawkishness_index': {
            EventDirection.HAWKISH: 1.0,
            EventDirection.DOVISH: -1.0,
        },
        'geopolitical_risk_index': {
            EventDirection.RISK_OFF: 0.8,  # Geopolitical risk is risk-off
            EventDirection.STRESS_UP: 0.5,
        },
        'financial_stress_index': {
            EventDirection.STRESS_UP: 1.0,
            EventDirection.STRESS_DOWN: -1.0,
            EventDirection.RISK_OFF: 0.3,
        },
        'commodities_supply_risk_index': {
            EventDirection.RISK_OFF: 0.5,  # Supply concerns often with risk-off
        },
    }
    
    def __init__(
        self,
        lookback_days: int = 7,
        decay_halflife_days: float = 3.0,
        high_impact_threshold: float = 0.6,
    ):
        """
        Initialize aggregator.
        
        Args:
            lookback_days: Days of events to consider
            decay_halflife_days: Half-life for exponential decay
            high_impact_threshold: Threshold for high-impact events
        """
        self.lookback_days = lookback_days
        self.decay_halflife = timedelta(days=decay_halflife_days)
        self.high_impact_threshold = high_impact_threshold
        
        # Cache of daily features
        self._feature_cache: Dict[str, DailyMacroFeatures] = {}
    
    def aggregate(
        self,
        events: List[MacroEvent],
        as_of: datetime,
        risk_sentiment: Optional[RiskSentiment] = None,
    ) -> DailyMacroFeatures:
        """
        Aggregate events into daily features.
        
        Args:
            events: List of MacroEvents
            as_of: Point-in-time (only uses events with timestamp <= as_of)
            risk_sentiment: Optional pre-computed risk sentiment
            
        Returns:
            DailyMacroFeatures for the date
        """
        # Handle timezone compatibility
        def make_comparable(dt):
            """Ensure datetime is comparable (remove timezone if needed)."""
            if dt.tzinfo is not None and as_of.tzinfo is None:
                return dt.replace(tzinfo=None)
            elif dt.tzinfo is None and as_of.tzinfo is not None:
                return pytz.UTC.localize(dt)
            return dt
        
        # Filter to events before as_of (no future leakage!)
        valid_events = [
            e for e in events
            if make_comparable(e.event_time) <= as_of
        ]
        
        # Filter to lookback window
        cutoff = as_of - timedelta(days=self.lookback_days)
        recent_events = [
            e for e in valid_events
            if make_comparable(e.event_time) > cutoff
        ]
        
        # Initialize features
        features = DailyMacroFeatures(date=as_of)
        features.event_count = len(recent_events)
        
        if not recent_events:
            return features
        
        # Aggregate by feature index
        feature_values: Dict[str, List[tuple]] = defaultdict(list)
        
        for event in recent_events:
            # Time decay weight
            days_ago = (as_of - event.event_time).total_seconds() / 86400
            time_weight = 0.5 ** (days_ago / self.decay_halflife.total_seconds() * 86400)
            
            # Total weight
            weight = time_weight * event.impact_score * event.novelty_score
            
            # Skip duplicates with low novelty
            if event.is_duplicate and event.novelty_score < 0.3:
                continue
            
            # Count high impact
            if event.impact_score >= self.high_impact_threshold:
                features.high_impact_event_count += 1
            
            # Map to features
            for tag in event.tags:
                if tag in self.TAG_TO_FEATURE:
                    feature_name = self.TAG_TO_FEATURE[tag]
                    
                    # Get direction sign
                    direction_signs = self.DIRECTION_SIGNS.get(feature_name, {})
                    sign = direction_signs.get(event.direction, 0.0)
                    
                    if sign != 0:
                        value = sign * event.severity_score
                        feature_values[feature_name].append((value, weight))
        
        # Compute weighted sums for each feature
        for feature_name, values in feature_values.items():
            if values:
                total_weight = sum(w for _, w in values)
                if total_weight > 0:
                    weighted_sum = sum(v * w for v, w in values) / total_weight
                    # Clip to [-1, 1]
                    setattr(features, feature_name, np.clip(weighted_sum, -1, 1))
        
        # Add risk sentiment if provided
        if risk_sentiment:
            features.overall_risk_sentiment_index = risk_sentiment.risk_sentiment
            features.equity_bias = risk_sentiment.equity_bias
            features.rates_bias = risk_sentiment.rates_bias
            features.dollar_bias = risk_sentiment.dollar_bias
        else:
            # Derive overall risk sentiment from features
            features.overall_risk_sentiment_index = self._derive_risk_sentiment(features)
        
        # Data quality score
        features.data_quality_score = self._calculate_quality_score(
            recent_events, features
        )
        
        # Top events for explanation
        features.top_events = self._get_top_events(recent_events)
        
        return features
    
    def _derive_risk_sentiment(self, features: DailyMacroFeatures) -> float:
        """Derive overall risk sentiment from individual features."""
        # Weighted combination
        sentiment = 0.0
        
        # Positive for risk: growth, labor strength
        sentiment += features.growth_momentum_index * 0.3
        sentiment += features.labor_strength_index * 0.2
        
        # Negative for risk: stress, geopolitics, hawkishness
        sentiment -= features.financial_stress_index * 0.3
        sentiment -= features.geopolitical_risk_index * 0.2
        sentiment -= features.central_bank_hawkishness_index * 0.15
        sentiment -= features.commodities_supply_risk_index * 0.1
        
        # Inflation is context-dependent but generally negative
        sentiment -= abs(features.macro_inflation_pressure_index) * 0.1
        
        return np.clip(sentiment, -1, 1)
    
    def _calculate_quality_score(
        self,
        events: List[MacroEvent],
        features: DailyMacroFeatures,
    ) -> float:
        """Calculate data quality score."""
        if not events:
            return 0.0
        
        # Factors: event count, confidence, high-impact ratio
        count_factor = min(1.0, len(events) / 20)  # 20+ events = full score
        
        avg_confidence = np.mean([e.confidence for e in events])
        
        high_impact_ratio = features.high_impact_event_count / max(1, len(events))
        
        return count_factor * 0.4 + avg_confidence * 0.4 + high_impact_ratio * 0.2
    
    def _get_top_events(
        self,
        events: List[MacroEvent],
        n: int = 5,
    ) -> List[str]:
        """Get top N events by impact for explanation."""
        sorted_events = sorted(events, key=lambda e: e.impact_score, reverse=True)
        
        return [
            f"[{e.tags[0].value if e.tags else 'unknown'}] {e.title[:60]}..."
            for e in sorted_events[:n]
        ]
    
    def get_feature_changes(
        self,
        current: DailyMacroFeatures,
        previous: DailyMacroFeatures,
    ) -> Dict[str, float]:
        """Calculate feature changes between two periods."""
        changes = {}
        
        feature_names = [
            'macro_inflation_pressure_index',
            'labor_strength_index',
            'growth_momentum_index',
            'central_bank_hawkishness_index',
            'geopolitical_risk_index',
            'financial_stress_index',
            'commodities_supply_risk_index',
            'overall_risk_sentiment_index',
        ]
        
        for name in feature_names:
            current_val = getattr(current, name, 0.0)
            previous_val = getattr(previous, name, 0.0)
            changes[name] = current_val - previous_val
        
        return changes
    
    def print_macro_brief(self, features: DailyMacroFeatures) -> str:
        """Generate a "Macro Brief" summary for debugging."""
        lines = [
            "=" * 60,
            f"MACRO BRIEF - {features.date.strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "📊 MACRO INDICES:",
            f"  Inflation Pressure:  {features.macro_inflation_pressure_index:+.2f}",
            f"  Labor Strength:      {features.labor_strength_index:+.2f}",
            f"  Growth Momentum:     {features.growth_momentum_index:+.2f}",
            f"  CB Hawkishness:      {features.central_bank_hawkishness_index:+.2f}",
            f"  Geopolitical Risk:   {features.geopolitical_risk_index:+.2f}",
            f"  Financial Stress:    {features.financial_stress_index:+.2f}",
            f"  Commodity Supply:    {features.commodities_supply_risk_index:+.2f}",
            "",
            "🎯 RISK SENTIMENT:",
            f"  Overall:   {features.overall_risk_sentiment_index:+.2f} "
            f"({'RISK-ON' if features.overall_risk_sentiment_index > 0.1 else 'RISK-OFF' if features.overall_risk_sentiment_index < -0.1 else 'NEUTRAL'})",
            f"  Equities:  {features.equity_bias:+.2f}",
            f"  Rates:     {features.rates_bias:+.2f}",
            f"  Dollar:    {features.dollar_bias:+.2f}",
            "",
            f"📈 STATS: {features.event_count} events, "
            f"{features.high_impact_event_count} high-impact, "
            f"quality: {features.data_quality_score:.0%}",
            "",
            "🔥 TOP EVENTS:",
        ]
        
        for event in features.top_events[:5]:
            lines.append(f"  • {event}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
