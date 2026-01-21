"""
News Intelligence Pipeline - Main orchestrator for news processing.

Coordinates all components: loading, filtering, classification, 
event extraction, impact scoring, sentiment analysis, and aggregation.
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import logging
import pytz

from .taxonomy import MacroTaxonomy, TaxonomyTag
from .sources import SourceCredibility, SourceTier
from .relevance import RelevanceGate, RelevanceResult
from .events import EventExtractor, MacroEvent
from .impact import ImpactScorer, ImpactBreakdown
from .sentiment import RiskSentimentAnalyzer, RiskSentiment
from .aggregator import MacroFeatureAggregator, DailyMacroFeatures

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Input article format."""
    timestamp: datetime
    source: str
    title: str
    body: str
    url: Optional[str] = None


@dataclass  
class ProcessingStats:
    """Statistics from processing run."""
    total_articles: int
    relevant_articles: int
    rejected_articles: int
    events_extracted: int
    high_impact_events: int
    pass_rate: float
    processing_time_ms: float


class NewsIntelligencePipeline:
    """
    Main pipeline for news intelligence processing.
    
    Flow:
    1. Load articles from sources
    2. Filter through relevance gate
    3. Classify with taxonomy
    4. Extract structured events
    5. Score impact
    6. Analyze sentiment
    7. Aggregate into features
    """
    
    def __init__(
        self,
        taxonomy: Optional[MacroTaxonomy] = None,
        source_credibility: Optional[SourceCredibility] = None,
        relevance_gate: Optional[RelevanceGate] = None,
        event_extractor: Optional[EventExtractor] = None,
        impact_scorer: Optional[ImpactScorer] = None,
        sentiment_analyzer: Optional[RiskSentimentAnalyzer] = None,
        feature_aggregator: Optional[MacroFeatureAggregator] = None,
        cache_dir: str = "outputs/news_intelligence",
    ):
        """
        Initialize pipeline with components.
        
        All components are optional and will use defaults if not provided.
        """
        self.taxonomy = taxonomy or MacroTaxonomy()
        self.source_credibility = source_credibility or SourceCredibility()
        self.relevance_gate = relevance_gate or RelevanceGate(
            taxonomy=self.taxonomy,
            source_credibility=self.source_credibility,
        )
        self.event_extractor = event_extractor or EventExtractor()
        self.impact_scorer = impact_scorer or ImpactScorer(
            source_credibility=self.source_credibility,
        )
        self.sentiment_analyzer = sentiment_analyzer or RiskSentimentAnalyzer()
        self.feature_aggregator = feature_aggregator or MacroFeatureAggregator()
        
        # Cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Event storage
        self._events: List[MacroEvent] = []
        self._processed_ids: set = set()
    
    def process_articles(
        self,
        articles: List[NewsArticle],
        as_of: Optional[datetime] = None,
    ) -> Tuple[List[MacroEvent], ProcessingStats]:
        """
        Process a batch of articles through the full pipeline.
        
        Args:
            articles: List of NewsArticle objects
            as_of: Point-in-time reference (defaults to now)
            
        Returns:
            Tuple of (extracted_events, processing_stats)
        """
        import time
        start_time = time.time()
        
        as_of = as_of or datetime.now(pytz.UTC)
        
        events = []
        rejected_count = 0
        
        for article in articles:
            # Skip duplicates
            article_id = f"{article.timestamp}_{hash(article.title)}"
            if article_id in self._processed_ids:
                continue
            self._processed_ids.add(article_id)
            
            # Step 1: Relevance filtering
            relevance = self.relevance_gate.filter(
                title=article.title,
                body=article.body,
                source=article.source,
            )
            
            if not relevance.is_relevant:
                rejected_count += 1
                logger.debug(f"Rejected: {article.title[:50]}... ({relevance.rejection_reason})")
                continue
            
            # Step 2: Event extraction
            event = self.event_extractor.extract(
                title=article.title,
                body=article.body,
                source=article.source,
                timestamp=article.timestamp,
                taxonomy_matches=relevance.taxonomy_matches,
            )
            
            if event:
                # Step 3: Impact scoring
                impact = self.impact_scorer.score(event, as_of)
                event.impact_score = impact.total_impact
                
                events.append(event)
                self._events.append(event)
        
        # Calculate stats
        relevant_count = len(articles) - rejected_count
        high_impact = sum(1 for e in events if e.impact_score >= 0.6)
        
        processing_time = (time.time() - start_time) * 1000
        
        stats = ProcessingStats(
            total_articles=len(articles),
            relevant_articles=relevant_count,
            rejected_articles=rejected_count,
            events_extracted=len(events),
            high_impact_events=high_impact,
            pass_rate=relevant_count / max(1, len(articles)),
            processing_time_ms=processing_time,
        )
        
        logger.info(
            f"Processed {len(articles)} articles: "
            f"{relevant_count} relevant, {len(events)} events, "
            f"{high_impact} high-impact ({processing_time:.0f}ms)"
        )
        
        return events, stats
    
    def get_daily_macro_features(
        self,
        date: datetime,
    ) -> DailyMacroFeatures:
        """
        Get aggregated macro features for a date.
        
        API for trading system.
        
        Args:
            date: Date to get features for
            
        Returns:
            DailyMacroFeatures
        """
        # Get risk sentiment first
        risk_sentiment = self.sentiment_analyzer.analyze(self._events, date)
        
        # Aggregate features
        features = self.feature_aggregator.aggregate(
            events=self._events,
            as_of=date,
            risk_sentiment=risk_sentiment,
        )
        
        return features
    
    def get_intraday_event_stream(
        self,
        start: datetime,
        end: datetime,
    ) -> List[MacroEvent]:
        """
        Get event stream for a time window.
        
        API for trading system.
        
        Args:
            start: Start time
            end: End time
            
        Returns:
            List of MacroEvents in the window
        """
        return [
            e for e in self._events
            if start <= e.event_time <= end
        ]
    
    def get_risk_sentiment(
        self,
        as_of: datetime,
    ) -> RiskSentiment:
        """
        Get current risk sentiment.
        
        Args:
            as_of: Point-in-time
            
        Returns:
            RiskSentiment
        """
        return self.sentiment_analyzer.analyze(self._events, as_of)
    
    def print_macro_brief(
        self,
        as_of: Optional[datetime] = None,
    ) -> str:
        """
        Generate and print macro brief for debugging.
        
        Args:
            as_of: Point-in-time (defaults to now)
            
        Returns:
            Macro brief string
        """
        as_of = as_of or datetime.now(pytz.UTC)
        
        features = self.get_daily_macro_features(as_of)
        brief = self.feature_aggregator.print_macro_brief(features)
        
        # Add sentiment summary
        sentiment = self.get_risk_sentiment(as_of)
        sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(sentiment)
        
        full_brief = f"{brief}\n\n💬 SENTIMENT: {sentiment_summary}"
        
        return full_brief
    
    def load_from_json(
        self,
        file_path: str,
        as_of: Optional[datetime] = None,
    ) -> Tuple[List[MacroEvent], ProcessingStats]:
        """
        Load and process articles from a JSON file.
        
        Expected format:
        [
            {"timestamp": "2024-01-15T10:30:00", "source": "Reuters", 
             "title": "...", "body": "...", "url": "..."},
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            as_of: Point-in-time reference
            
        Returns:
            Tuple of (events, stats)
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        articles = []
        for item in data:
            try:
                timestamp_str = item['timestamp'].replace('Z', '+00:00')
                timestamp = datetime.fromisoformat(timestamp_str)
                
                # Ensure timezone-aware (use UTC if naive)
                if timestamp.tzinfo is None:
                    timestamp = pytz.UTC.localize(timestamp)
                
                articles.append(NewsArticle(
                    timestamp=timestamp,
                    source=item.get('source', 'unknown'),
                    title=item.get('title', ''),
                    body=item.get('body', item.get('text', '')),
                    url=item.get('url'),
                ))
            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue
        
        return self.process_articles(articles, as_of)
    
    def save_events(self, file_path: Optional[str] = None):
        """Save extracted events to JSON."""
        file_path = file_path or str(self.cache_dir / "events.json")
        
        events_data = []
        for event in self._events:
            events_data.append({
                'event_id': event.event_id,
                'event_time': event.event_time.isoformat(),
                'source': event.source,
                'title': event.title,
                'tags': [t.value for t in event.tags],
                'entities': event.entities,
                'direction': event.direction.value,
                'severity_score': event.severity_score,
                'impact_score': event.impact_score,
                'confidence': event.confidence,
                'rationale': event.rationale,
            })
        
        with open(file_path, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        logger.info(f"Saved {len(events_data)} events to {file_path}")
    
    def load_events(self, file_path: Optional[str] = None):
        """Load events from JSON."""
        file_path = file_path or str(self.cache_dir / "events.json")
        
        if not Path(file_path).exists():
            logger.warning(f"Events file not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            try:
                event = MacroEvent(
                    event_id=item['event_id'],
                    event_time=datetime.fromisoformat(item['event_time']),
                    source=item['source'],
                    title=item['title'],
                    tags=[TaxonomyTag(t) for t in item['tags']],
                    entities=item['entities'],
                    direction=item['direction'],
                    severity_score=item['severity_score'],
                    impact_score=item['impact_score'],
                    confidence=item['confidence'],
                    rationale=item['rationale'],
                )
                self._events.append(event)
            except Exception as e:
                logger.warning(f"Error loading event: {e}")
                continue
        
        logger.info(f"Loaded {len(self._events)} events from {file_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'total_events': len(self._events),
            'events_by_tag': self._count_by_tag(),
            'events_by_source_tier': self._count_by_source_tier(),
            'high_impact_events': sum(1 for e in self._events if e.impact_score >= 0.6),
            'avg_impact': sum(e.impact_score for e in self._events) / max(1, len(self._events)),
        }
    
    def _count_by_tag(self) -> Dict[str, int]:
        """Count events by taxonomy tag."""
        counts: Dict[str, int] = {}
        for event in self._events:
            for tag in event.tags:
                counts[tag.value] = counts.get(tag.value, 0) + 1
        return counts
    
    def _count_by_source_tier(self) -> Dict[str, int]:
        """Count events by source tier."""
        counts: Dict[str, int] = {}
        for event in self._events:
            tier = self.source_credibility.get_tier(event.source).value
            counts[tier] = counts.get(tier, 0) + 1
        return counts
    
    def clear(self):
        """Clear all stored events."""
        self._events = []
        self._processed_ids = set()
        logger.info("Cleared all events")
