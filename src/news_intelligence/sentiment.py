"""
Risk Sentiment Analyzer - Context-aware sentiment for market impact.

Computes risk-on/risk-off sentiment conditioned on event type.
NOT generic emotion - market-specific risk appetite.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import logging
import pytz

from .taxonomy import TaxonomyTag
from .events import MacroEvent, EventDirection

logger = logging.getLogger(__name__)


@dataclass
class RiskSentiment:
    """Risk sentiment for a time period."""
    timestamp: datetime
    
    # Core sentiment
    risk_sentiment: float  # [-1, 1]: -1=risk-off, +1=risk-on
    sentiment_delta: float  # Change vs 7d average
    sentiment_volatility: float  # Std dev over rolling window
    
    # Directional biases (derived from sentiment + context)
    equity_bias: float  # [-1, 1]: bullish/bearish for equities
    rates_bias: float  # [-1, 1]: higher/lower rates expected
    dollar_bias: float  # [-1, 1]: stronger/weaker dollar
    commodity_bias: float  # [-1, 1]: bullish/bearish commodities
    
    # Confidence
    confidence: float  # 0-1
    event_count: int


class RiskSentimentAnalyzer:
    """
    Analyze risk sentiment from macro events.
    
    Context-aware: hawkish Fed is risk-off, war escalation is risk-off,
    but disinflation surprise may be risk-on.
    """
    
    # Direction to risk-sentiment mapping
    # Positive = risk-on, Negative = risk-off
    DIRECTION_RISK_MAPPING: Dict[EventDirection, float] = {
        EventDirection.HAWKISH: -0.6,  # Hawkish = risk-off
        EventDirection.DOVISH: 0.5,  # Dovish = risk-on
        EventDirection.RISK_ON: 0.8,
        EventDirection.RISK_OFF: -0.8,
        EventDirection.INFLATION_UP: -0.4,  # Inflation up = rates up = risk-off
        EventDirection.INFLATION_DOWN: 0.4,  # Disinflation = risk-on
        EventDirection.GROWTH_UP: 0.5,
        EventDirection.GROWTH_DOWN: -0.5,
        EventDirection.LABOR_STRONG: 0.3,  # Good for economy, but may mean hawkish Fed
        EventDirection.LABOR_WEAK: -0.3,
        EventDirection.STRESS_UP: -0.9,  # Financial stress = strong risk-off
        EventDirection.STRESS_DOWN: 0.6,
        EventDirection.NEUTRAL: 0.0,
    }
    
    # Tag-specific sentiment modifiers
    TAG_SENTIMENT_MODIFIERS: Dict[TaxonomyTag, Dict[str, float]] = {
        TaxonomyTag.CENTRAL_BANK: {
            'equity_modifier': -0.3,  # Fed news often moves opposite to rates
            'rates_modifier': 1.0,  # Direct rates impact
            'dollar_modifier': 0.5,  # Rate diff affects USD
        },
        TaxonomyTag.GEOPOLITICS: {
            'equity_modifier': -0.5,  # War/conflict = risk-off
            'commodity_modifier': 0.6,  # Geopolitics often bullish oil/gold
            'dollar_modifier': 0.3,  # Flight to safety = USD up
        },
        TaxonomyTag.FINANCIAL_STRESS: {
            'equity_modifier': -0.8,  # Credit stress = bad for stocks
            'rates_modifier': -0.4,  # Stress = flight to safety = rates down
            'dollar_modifier': 0.2,  # Mixed (flight to safety vs domestic stress)
        },
        TaxonomyTag.COMMODITIES_ENERGY: {
            'commodity_modifier': 1.0,  # Direct impact
            'equity_modifier': -0.2,  # High oil often bad for stocks
        },
    }
    
    def __init__(
        self,
        lookback_days: int = 7,
        decay_halflife_hours: float = 24.0,
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            lookback_days: Days for rolling averages
            decay_halflife_hours: Half-life for event decay
        """
        self.lookback_days = lookback_days
        self.decay_halflife = timedelta(hours=decay_halflife_hours)
        
        # Historical sentiment storage
        self._sentiment_history: List[Tuple[datetime, float, float]] = []  # (time, sentiment, weight)
    
    def analyze(
        self,
        events: List[MacroEvent],
        as_of: datetime,
    ) -> RiskSentiment:
        """
        Compute risk sentiment from a list of events.
        
        Args:
            events: List of MacroEvents
            as_of: Point-in-time reference
            
        Returns:
            RiskSentiment with directional biases
        """
        if not events:
            return self._neutral_sentiment(as_of)
        
        # Handle timezone compatibility
        def make_comparable(dt):
            """Ensure datetime is comparable."""
            if dt.tzinfo is not None and as_of.tzinfo is None:
                return dt.replace(tzinfo=None)
            elif dt.tzinfo is None and as_of.tzinfo is not None:
                return pytz.UTC.localize(dt)
            return dt
        
        # Filter to relevant time window
        cutoff = as_of - timedelta(days=self.lookback_days)
        recent_events = [
            e for e in events
            if cutoff <= make_comparable(e.event_time) <= as_of
        ]
        
        if not recent_events:
            return self._neutral_sentiment(as_of)
        
        # Calculate weighted sentiment
        weighted_sentiments = []
        equity_biases = []
        rates_biases = []
        dollar_biases = []
        commodity_biases = []
        
        for event in recent_events:
            # Time decay weight
            time_diff = (as_of - event.event_time).total_seconds() / 3600
            time_weight = 0.5 ** (time_diff / (self.decay_halflife.total_seconds() / 3600))
            
            # Impact weight
            impact_weight = event.impact_score
            
            # Combined weight
            total_weight = time_weight * impact_weight * event.confidence
            
            # Base sentiment from direction
            base_sentiment = self.DIRECTION_RISK_MAPPING.get(
                event.direction, 0.0
            )
            
            weighted_sentiments.append((base_sentiment, total_weight))
            
            # Calculate directional biases
            equity_bias, rates_bias, dollar_bias, commodity_bias = \
                self._calculate_biases(event, base_sentiment)
            
            equity_biases.append((equity_bias, total_weight))
            rates_biases.append((rates_bias, total_weight))
            dollar_biases.append((dollar_bias, total_weight))
            commodity_biases.append((commodity_bias, total_weight))
        
        # Compute weighted averages
        def weighted_avg(items: List[Tuple[float, float]]) -> float:
            total_weight = sum(w for _, w in items)
            if total_weight == 0:
                return 0.0
            return sum(v * w for v, w in items) / total_weight
        
        risk_sentiment = weighted_avg(weighted_sentiments)
        equity_bias = weighted_avg(equity_biases)
        rates_bias = weighted_avg(rates_biases)
        dollar_bias = weighted_avg(dollar_biases)
        commodity_bias = weighted_avg(commodity_biases)
        
        # Calculate sentiment delta vs 7d average
        sentiment_delta = self._calculate_delta(risk_sentiment, as_of)
        
        # Calculate sentiment volatility
        sentiment_volatility = self._calculate_volatility(as_of)
        
        # Store current sentiment
        total_weight = sum(w for _, w in weighted_sentiments)
        self._sentiment_history.append((as_of, risk_sentiment, total_weight))
        
        # Clean old history
        history_cutoff = as_of - timedelta(days=30)
        self._sentiment_history = [
            (t, s, w) for t, s, w in self._sentiment_history
            if t > history_cutoff
        ]
        
        # Confidence based on event count and weight
        confidence = min(1.0, 0.3 + len(recent_events) * 0.1 + total_weight * 0.2)
        
        return RiskSentiment(
            timestamp=as_of,
            risk_sentiment=np.clip(risk_sentiment, -1, 1),
            sentiment_delta=sentiment_delta,
            sentiment_volatility=sentiment_volatility,
            equity_bias=np.clip(equity_bias, -1, 1),
            rates_bias=np.clip(rates_bias, -1, 1),
            dollar_bias=np.clip(dollar_bias, -1, 1),
            commodity_bias=np.clip(commodity_bias, -1, 1),
            confidence=confidence,
            event_count=len(recent_events),
        )
    
    def _calculate_biases(
        self,
        event: MacroEvent,
        base_sentiment: float,
    ) -> Tuple[float, float, float, float]:
        """Calculate directional biases for an event."""
        equity_bias = base_sentiment * 0.8  # Default: sentiment drives equities
        rates_bias = -base_sentiment * 0.3  # Default: risk-off = rates down
        dollar_bias = -base_sentiment * 0.2  # Default: risk-off = dollar up
        commodity_bias = 0.0  # Default: neutral
        
        # Apply tag-specific modifiers
        for tag in event.tags:
            if tag in self.TAG_SENTIMENT_MODIFIERS:
                modifiers = self.TAG_SENTIMENT_MODIFIERS[tag]
                
                if 'equity_modifier' in modifiers:
                    equity_bias += base_sentiment * modifiers['equity_modifier']
                if 'rates_modifier' in modifiers:
                    rates_bias += base_sentiment * modifiers['rates_modifier']
                if 'dollar_modifier' in modifiers:
                    dollar_bias += base_sentiment * modifiers['dollar_modifier']
                if 'commodity_modifier' in modifiers:
                    commodity_bias += base_sentiment * modifiers['commodity_modifier']
        
        return equity_bias, rates_bias, dollar_bias, commodity_bias
    
    def _calculate_delta(self, current: float, as_of: datetime) -> float:
        """Calculate sentiment change vs 7d average."""
        if not self._sentiment_history:
            return 0.0
        
        # Get 7d average
        cutoff = as_of - timedelta(days=7)
        historical = [
            (s, w) for t, s, w in self._sentiment_history
            if cutoff <= t < as_of
        ]
        
        if not historical:
            return 0.0
        
        total_weight = sum(w for _, w in historical)
        if total_weight == 0:
            return 0.0
        
        historical_avg = sum(s * w for s, w in historical) / total_weight
        
        return current - historical_avg
    
    def _calculate_volatility(self, as_of: datetime) -> float:
        """Calculate sentiment volatility over rolling window."""
        cutoff = as_of - timedelta(days=7)
        recent = [s for t, s, _ in self._sentiment_history if t > cutoff]
        
        if len(recent) < 2:
            return 0.0
        
        return float(np.std(recent))
    
    def _neutral_sentiment(self, as_of: datetime) -> RiskSentiment:
        """Return neutral sentiment when no events."""
        return RiskSentiment(
            timestamp=as_of,
            risk_sentiment=0.0,
            sentiment_delta=0.0,
            sentiment_volatility=0.0,
            equity_bias=0.0,
            rates_bias=0.0,
            dollar_bias=0.0,
            commodity_bias=0.0,
            confidence=0.1,
            event_count=0,
        )
    
    def get_sentiment_summary(self, sentiment: RiskSentiment) -> str:
        """Generate human-readable sentiment summary."""
        # Risk stance
        if sentiment.risk_sentiment > 0.3:
            risk_stance = "RISK-ON"
        elif sentiment.risk_sentiment < -0.3:
            risk_stance = "RISK-OFF"
        else:
            risk_stance = "NEUTRAL"
        
        # Changes
        if sentiment.sentiment_delta > 0.1:
            change = "improving"
        elif sentiment.sentiment_delta < -0.1:
            change = "deteriorating"
        else:
            change = "stable"
        
        # Biases
        biases = []
        if abs(sentiment.equity_bias) > 0.2:
            biases.append(f"Equities: {'bullish' if sentiment.equity_bias > 0 else 'bearish'}")
        if abs(sentiment.rates_bias) > 0.2:
            biases.append(f"Rates: {'higher' if sentiment.rates_bias > 0 else 'lower'}")
        if abs(sentiment.commodity_bias) > 0.2:
            biases.append(f"Commodities: {'bullish' if sentiment.commodity_bias > 0 else 'bearish'}")
        
        summary = f"{risk_stance} ({change}) | {' | '.join(biases) or 'No strong biases'}"
        summary += f" | Confidence: {sentiment.confidence:.0%} ({sentiment.event_count} events)"
        
        return summary
