"""
Event Extraction - Extract structured events from news articles.

Produces market-moving events with direction, severity, and impact scores.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import logging

from .taxonomy import TaxonomyTag, TaxonomyMatch

logger = logging.getLogger(__name__)


class EventDirection(Enum):
    """Direction of event impact."""
    HAWKISH = "hawkish"  # Central bank tightening bias
    DOVISH = "dovish"  # Central bank loosening bias
    RISK_ON = "risk_on"  # Positive for risk assets
    RISK_OFF = "risk_off"  # Negative for risk assets
    INFLATION_UP = "inflation_up"
    INFLATION_DOWN = "inflation_down"
    GROWTH_UP = "growth_up"
    GROWTH_DOWN = "growth_down"
    LABOR_STRONG = "labor_strong"
    LABOR_WEAK = "labor_weak"
    STRESS_UP = "stress_up"
    STRESS_DOWN = "stress_down"
    NEUTRAL = "neutral"


@dataclass
class MacroEvent:
    """Structured macro event extracted from news."""
    event_id: str
    event_time: datetime
    source: str
    title: str
    
    # Classification
    tags: List[TaxonomyTag]
    entities: List[str]  # Fed, ECB, CPI, OPEC, countries, etc.
    
    # Direction and magnitude
    direction: EventDirection
    severity_score: float  # 0-1 how severe/significant
    impact_score: float  # 0-1 expected market impact
    confidence: float  # 0-1 confidence in extraction
    
    # Explanation
    rationale: str
    matched_phrases: List[str] = field(default_factory=list)
    
    # Metadata
    is_duplicate: bool = False
    novelty_score: float = 1.0  # Decays for repeated news


class EventExtractor:
    """
    Extract structured macro events from news articles.
    
    Identifies key entities, determines direction, and scores severity.
    """
    
    # Entity patterns
    ENTITY_PATTERNS: Dict[str, List[str]] = {
        # Central Banks
        'Fed': [r'\bfed\b', r'federal reserve', r'\bfomc\b', r'powell'],
        'ECB': [r'\becb\b', r'european central bank', r'lagarde'],
        'BOJ': [r'\bboj\b', r'bank of japan'],
        'BOE': [r'bank of england', r'\bboe\b'],
        'PBOC': [r'\bpboc\b', r'people\'?s bank of china'],
        
        # Economic indicators
        'CPI': [r'\bcpi\b', r'consumer price index'],
        'PCE': [r'\bpce\b', r'personal consumption'],
        'GDP': [r'\bgdp\b', r'gross domestic product'],
        'PMI': [r'\bpmi\b', r'purchasing manager'],
        'NFP': [r'nonfarm payroll', r'jobs report', r'employment report'],
        'Unemployment': [r'unemployment', r'jobless rate'],
        
        # Organizations
        'OPEC': [r'\bopec\b', r'opec\+'],
        'IMF': [r'\bimf\b', r'international monetary fund'],
        'Treasury': [r'treasury department', r'treasury secretary'],
        
        # Countries/Regions (geopolitical)
        'China': [r'\bchina\b', r'\bchinese\b', r'beijing'],
        'Russia': [r'\brussia\b', r'\brussian\b', r'moscow', r'kremlin'],
        'Ukraine': [r'\bukraine\b', r'\bukrainian\b', r'kyiv'],
        'Iran': [r'\biran\b', r'\biranian\b', r'tehran'],
        'Middle East': [r'middle east', r'\bisrael\b', r'\bgaza\b'],
        'EU': [r'\beu\b', r'european union', r'eurozone'],
    }
    
    # Direction patterns
    DIRECTION_PATTERNS: Dict[EventDirection, List[str]] = {
        EventDirection.HAWKISH: [
            r'rate hike', r'raise.{1,10}rate', r'tighten', r'hawkish',
            r'higher.{1,10}longer', r'restrictive', r'inflation fight',
            r'not done', r'more work', r'additional hike',
        ],
        EventDirection.DOVISH: [
            r'rate cut', r'lower.{1,10}rate', r'ease', r'dovish',
            r'pause', r'pivot', r'accommodative', r'supportive',
            r'slow.{1,10}hike', r'less restrictive',
        ],
        EventDirection.RISK_ON: [
            r'rally', r'surge', r'soar', r'jump', r'optimism',
            r'beat.{1,10}estimate', r'better.{1,10}expect', r'upside surprise',
            r'deal reached', r'agreement', r'ceasefire', r'peace',
            r'stimulus', r'support package',
        ],
        EventDirection.RISK_OFF: [
            r'selloff', r'plunge', r'crash', r'tumble', r'fear',
            r'miss.{1,10}estimate', r'worse.{1,10}expect', r'downside surprise',
            r'escalat', r'tension', r'conflict', r'strike', r'attack',
            r'default', r'crisis', r'stress', r'panic',
        ],
        EventDirection.INFLATION_UP: [
            r'inflation.{1,10}(higher|rise|surge|jump|accelerate|hot)',
            r'price.{1,10}(higher|rise|surge|jump)',
            r'cpi.{1,10}(beat|higher|hot|accelerate)',
            r'sticky inflation', r'persistent inflation',
        ],
        EventDirection.INFLATION_DOWN: [
            r'inflation.{1,10}(lower|fall|drop|cool|slow|ease)',
            r'price.{1,10}(lower|fall|drop)',
            r'cpi.{1,10}(miss|lower|cool|slow)',
            r'disinflation', r'deflation',
        ],
        EventDirection.GROWTH_UP: [
            r'gdp.{1,10}(beat|higher|stronger|accelerate|grow)',
            r'growth.{1,10}(beat|higher|stronger|accelerate)',
            r'expansion', r'robust economy', r'soft landing',
        ],
        EventDirection.GROWTH_DOWN: [
            r'gdp.{1,10}(miss|lower|weaker|slow|contract)',
            r'growth.{1,10}(miss|lower|weaker|slow)',
            r'contraction', r'recession', r'hard landing', r'slowdown',
        ],
        EventDirection.LABOR_STRONG: [
            r'(job|payroll).{1,10}(beat|higher|stronger|add|gain)',
            r'unemployment.{1,10}(fall|drop|lower|decline)',
            r'hiring.{1,10}(strong|surge|boom)',
            r'labor market.{1,10}(tight|strong|hot)',
        ],
        EventDirection.LABOR_WEAK: [
            r'(job|payroll).{1,10}(miss|lower|weaker|loss)',
            r'unemployment.{1,10}(rise|higher|jump|surge)',
            r'layoff', r'job cut', r'hiring freeze',
            r'labor market.{1,10}(weak|cool|soft)',
        ],
        EventDirection.STRESS_UP: [
            r'bank.{1,10}(fail|run|crisis|trouble|stress)',
            r'credit.{1,10}(crisis|crunch|tighten)',
            r'liquidity.{1,10}(crisis|crunch|dry)',
            r'contagion', r'systemic risk', r'bailout',
        ],
        EventDirection.STRESS_DOWN: [
            r'stress.{1,10}(ease|subside|stabiliz)',
            r'confidence.{1,10}(return|restore)',
            r'calm return', r'crisis.{1,10}(averted|over)',
        ],
    }
    
    # Severity boost patterns
    SEVERITY_PATTERNS: List[Tuple[str, float]] = [
        (r'unexpect', 0.2),
        (r'surpris', 0.2),
        (r'shock', 0.3),
        (r'historic', 0.3),
        (r'record', 0.2),
        (r'emergency', 0.3),
        (r'crisis', 0.3),
        (r'dramatic', 0.2),
        (r'significant', 0.15),
        (r'major', 0.15),
        (r'massive', 0.2),
        (r'unprecedented', 0.3),
        (r'\d+%', 0.1),  # Numbers present
        (r'\d+ basis point', 0.15),
        (r'\d+ bp\b', 0.15),
    ]
    
    def __init__(
        self,
        min_confidence: float = 0.3,
    ):
        """
        Initialize event extractor.
        
        Args:
            min_confidence: Minimum confidence to emit an event
        """
        self.min_confidence = min_confidence
        
        # Compile patterns
        self._entity_patterns = {
            name: [re.compile(p, re.IGNORECASE) for p in patterns]
            for name, patterns in self.ENTITY_PATTERNS.items()
        }
        
        self._direction_patterns = {
            direction: [re.compile(p, re.IGNORECASE) for p in patterns]
            for direction, patterns in self.DIRECTION_PATTERNS.items()
        }
        
        self._severity_patterns = [
            (re.compile(p, re.IGNORECASE), boost)
            for p, boost in self.SEVERITY_PATTERNS
        ]
        
        # Duplicate tracking
        self._seen_titles: Dict[str, datetime] = {}
    
    def extract(
        self,
        title: str,
        body: str,
        source: str,
        timestamp: datetime,
        taxonomy_matches: List[TaxonomyMatch],
    ) -> Optional[MacroEvent]:
        """
        Extract a structured event from an article.
        
        Args:
            title: Article title
            body: Article body
            source: Source name
            timestamp: Publication timestamp
            taxonomy_matches: Pre-computed taxonomy matches
            
        Returns:
            MacroEvent or None if not extractable
        """
        text = f"{title} {body}"
        
        if not taxonomy_matches:
            return None
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Determine direction
        direction, direction_confidence, matched_phrases = self._determine_direction(
            text, taxonomy_matches
        )
        
        # Calculate severity
        severity_score = self._calculate_severity(text, taxonomy_matches)
        
        # Check for duplicates
        is_duplicate, novelty_score = self._check_duplicate(title, timestamp)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            taxonomy_matches, direction_confidence, entities, source
        )
        
        if confidence < self.min_confidence:
            return None
        
        # Calculate impact score
        impact_score = self._calculate_impact(
            severity_score, novelty_score, confidence, taxonomy_matches
        )
        
        # Build rationale
        rationale = self._build_rationale(
            taxonomy_matches, entities, direction, matched_phrases
        )
        
        # Generate event ID
        event_id = f"{timestamp.strftime('%Y%m%d%H%M')}_{hash(title) % 10000:04d}"
        
        return MacroEvent(
            event_id=event_id,
            event_time=timestamp,
            source=source,
            title=title,
            tags=[m.tag for m in taxonomy_matches],
            entities=entities,
            direction=direction,
            severity_score=severity_score,
            impact_score=impact_score,
            confidence=confidence,
            rationale=rationale,
            matched_phrases=matched_phrases,
            is_duplicate=is_duplicate,
            novelty_score=novelty_score,
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        
        for entity_name, patterns in self._entity_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    entities.append(entity_name)
                    break
        
        return entities
    
    def _determine_direction(
        self,
        text: str,
        taxonomy_matches: List[TaxonomyMatch],
    ) -> Tuple[EventDirection, float, List[str]]:
        """Determine the directional bias of the event."""
        direction_scores: Dict[EventDirection, float] = {}
        matched_phrases: List[str] = []
        
        for direction, patterns in self._direction_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    score += 0.3 * len(matches)
                    matched_phrases.extend(matches[:2])
            
            if score > 0:
                direction_scores[direction] = min(1.0, score)
        
        if not direction_scores:
            return EventDirection.NEUTRAL, 0.3, []
        
        # Get highest scoring direction
        best_direction = max(direction_scores, key=direction_scores.get)
        confidence = direction_scores[best_direction]
        
        return best_direction, confidence, matched_phrases[:5]
    
    def _calculate_severity(
        self,
        text: str,
        taxonomy_matches: List[TaxonomyMatch],
    ) -> float:
        """Calculate severity score based on language intensity."""
        base_severity = 0.3
        
        # Boost from taxonomy confidence
        if taxonomy_matches:
            base_severity += taxonomy_matches[0].confidence * 0.2
        
        # Boost from severity patterns
        for pattern, boost in self._severity_patterns:
            if pattern.search(text):
                base_severity += boost
        
        return min(1.0, base_severity)
    
    def _check_duplicate(
        self,
        title: str,
        timestamp: datetime,
    ) -> Tuple[bool, float]:
        """Check if this is a duplicate story and calculate novelty."""
        # Normalize title
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        words = set(normalized.split())
        
        is_duplicate = False
        max_similarity = 0.0
        
        for seen_title, seen_time in list(self._seen_titles.items()):
            # Clean old entries (older than 24 hours)
            if (timestamp - seen_time).total_seconds() > 86400:
                del self._seen_titles[seen_title]
                continue
            
            # Calculate Jaccard similarity
            seen_words = set(seen_title.split())
            intersection = len(words & seen_words)
            union = len(words | seen_words)
            similarity = intersection / max(1, union)
            
            if similarity > max_similarity:
                max_similarity = similarity
            
            if similarity > 0.7:
                is_duplicate = True
        
        # Store this title
        self._seen_titles[normalized] = timestamp
        
        # Novelty decays with similarity
        novelty_score = 1.0 - max_similarity * 0.7
        
        return is_duplicate, novelty_score
    
    def _calculate_confidence(
        self,
        taxonomy_matches: List[TaxonomyMatch],
        direction_confidence: float,
        entities: List[str],
        source: str,
    ) -> float:
        """Calculate overall confidence in the event extraction."""
        confidence = 0.3
        
        # Taxonomy confidence
        if taxonomy_matches:
            confidence += taxonomy_matches[0].confidence * 0.3
        
        # Direction confidence
        confidence += direction_confidence * 0.2
        
        # Entity presence
        confidence += min(0.2, len(entities) * 0.05)
        
        return min(1.0, confidence)
    
    def _calculate_impact(
        self,
        severity: float,
        novelty: float,
        confidence: float,
        taxonomy_matches: List[TaxonomyMatch],
    ) -> float:
        """Calculate expected market impact score."""
        # Base impact from severity
        impact = severity * 0.4
        
        # Novelty factor
        impact += novelty * 0.3
        
        # Confidence factor
        impact += confidence * 0.2
        
        # Tag-based boost (some tags more impactful)
        high_impact_tags = {
            TaxonomyTag.CENTRAL_BANK,
            TaxonomyTag.FINANCIAL_STRESS,
            TaxonomyTag.GEOPOLITICS,
        }
        
        if taxonomy_matches:
            for match in taxonomy_matches:
                if match.tag in high_impact_tags:
                    impact += 0.15
                    break
        
        return min(1.0, impact)
    
    def _build_rationale(
        self,
        taxonomy_matches: List[TaxonomyMatch],
        entities: List[str],
        direction: EventDirection,
        matched_phrases: List[str],
    ) -> str:
        """Build human-readable rationale for the event."""
        parts = []
        
        # Tags
        if taxonomy_matches:
            tags_str = ", ".join(m.tag.value for m in taxonomy_matches[:2])
            parts.append(f"Category: {tags_str}")
        
        # Entities
        if entities:
            parts.append(f"Entities: {', '.join(entities[:3])}")
        
        # Direction
        parts.append(f"Direction: {direction.value}")
        
        # Key phrases
        if matched_phrases:
            parts.append(f"Key phrases: {', '.join(matched_phrases[:3])}")
        
        return " | ".join(parts)
