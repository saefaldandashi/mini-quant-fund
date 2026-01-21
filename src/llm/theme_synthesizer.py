"""
Theme Synthesizer - Identifies themes across multiple articles.

Uses LLM to synthesize patterns when multiple articles discuss related topics.
"""
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

from .llm_service import LLMService
from .event_extractor import MacroEvent

logger = logging.getLogger(__name__)


@dataclass
class Theme:
    """Synthesized theme from multiple articles."""
    theme_id: str
    name: str
    description: str
    
    # Theme strength
    momentum: str  # accelerating, stable, decelerating
    severity: float  # 0-1
    confidence: float  # 0-1
    
    # Evidence
    article_count: int
    supporting_events: List[str]  # Event IDs
    key_evidence: List[str]  # Key phrases from articles
    
    # Trading implications
    direction: str  # risk_on, risk_off, neutral
    affected_assets: List[str]
    trading_implication: str
    
    # Time
    first_seen: datetime
    last_updated: datetime
    is_new: bool  # First time seeing this theme


class ThemeSynthesizer:
    """
    Synthesizes themes from multiple macro events.
    
    Strategy:
    1. Group events by topic
    2. Identify patterns across groups
    3. Use LLM to synthesize narrative (if available)
    4. Track theme momentum over time
    """
    
    # Theme categories to track
    THEME_CATEGORIES = [
        'inflation_narrative',
        'fed_policy_path',
        'growth_outlook',
        'geopolitical_tension',
        'financial_stability',
        'sector_rotation',
        'earnings_outlook',
        'china_outlook',
        'energy_supply',
        'tech_sentiment',
    ]
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        min_articles_for_theme: int = 3,
    ):
        self.llm_service = llm_service
        self.min_articles_for_theme = min_articles_for_theme
        
        # Track themes over time
        self.active_themes: Dict[str, Theme] = {}
        self.theme_history: List[Theme] = []
    
    def group_events_by_topic(
        self,
        events: List[MacroEvent],
    ) -> Dict[str, List[MacroEvent]]:
        """Group events by their event type."""
        groups = defaultdict(list)
        
        for event in events:
            groups[event.event_type].append(event)
        
        return dict(groups)
    
    def _synthesize_rule_based(
        self,
        topic: str,
        events: List[MacroEvent],
    ) -> Optional[Theme]:
        """Synthesize theme using rules."""
        
        if len(events) < self.min_articles_for_theme:
            return None
        
        # Count directions
        direction_counts = defaultdict(int)
        for event in events:
            direction_counts[event.direction] += 1
        
        # Dominant direction
        dominant_direction = max(direction_counts, key=direction_counts.get)
        
        # Calculate severity
        avg_severity = sum(e.severity for e in events) / len(events)
        
        # Determine momentum
        recent_events = sorted(events, key=lambda e: e.timestamp, reverse=True)[:3]
        older_events = sorted(events, key=lambda e: e.timestamp, reverse=True)[3:]
        
        if recent_events and older_events:
            recent_severity = sum(e.severity for e in recent_events) / len(recent_events)
            older_severity = sum(e.severity for e in older_events) / len(older_events)
            
            if recent_severity > older_severity * 1.2:
                momentum = "accelerating"
            elif recent_severity < older_severity * 0.8:
                momentum = "decelerating"
            else:
                momentum = "stable"
        else:
            momentum = "stable"
        
        # Trading implications
        if topic == "central_bank":
            if dominant_direction == "hawkish":
                trading_implication = "Reduce duration, favor quality"
            else:
                trading_implication = "Add duration, favor growth"
        elif topic == "inflation":
            if dominant_direction in ["bullish", "hawkish"]:
                trading_implication = "Favor real assets, reduce bonds"
            else:
                trading_implication = "Add bonds, favor growth"
        elif topic == "geopolitics":
            if dominant_direction in ["bearish", "risk_off"]:
                trading_implication = "Reduce risk, add havens"
            else:
                trading_implication = "Add risk assets"
        else:
            trading_implication = f"Monitor {topic} developments"
        
        # Map direction to trading direction
        if dominant_direction in ["hawkish", "bearish", "risk_off"]:
            theme_direction = "risk_off"
        elif dominant_direction in ["dovish", "bullish", "risk_on"]:
            theme_direction = "risk_on"
        else:
            theme_direction = "neutral"
        
        # Check if new theme
        is_new = topic not in self.active_themes
        
        theme = Theme(
            theme_id=f"{topic}_{datetime.now().strftime('%Y%m%d')}",
            name=f"{topic.replace('_', ' ').title()} Theme",
            description=f"Based on {len(events)} articles showing {dominant_direction} signals",
            momentum=momentum,
            severity=avg_severity,
            confidence=min(0.9, 0.5 + len(events) * 0.1),  # More events = higher confidence
            article_count=len(events),
            supporting_events=[e.event_id for e in events],
            key_evidence=[e.source_headline[:50] for e in events[:3]],
            direction=theme_direction,
            affected_assets=[],
            trading_implication=trading_implication,
            first_seen=min(e.timestamp for e in events),
            last_updated=datetime.now(),
            is_new=is_new,
        )
        
        return theme
    
    def _synthesize_with_llm(
        self,
        topic: str,
        events: List[MacroEvent],
    ) -> Optional[Theme]:
        """Synthesize theme using LLM for deeper analysis."""
        
        if not self.llm_service or not self.llm_service.is_available():
            return self._synthesize_rule_based(topic, events)
        
        # Build evidence string
        evidence = []
        for event in events[:5]:  # Limit to 5 events
            evidence.append(f"- [{event.direction}] {event.source_headline}")
        
        prompt = f"""Analyze these {len(events)} related news events about "{topic}" and synthesize the market theme.

EVENTS:
{chr(10).join(evidence)}

Provide analysis in JSON format:
{{
    "theme_name": "concise theme name",
    "description": "2-3 sentence summary of what's happening",
    "momentum": "accelerating|stable|decelerating",
    "severity": 0.0-1.0,
    "direction": "risk_on|risk_off|neutral",
    "affected_assets": ["list", "of", "assets"],
    "trading_implication": "one clear actionable implication",
    "key_risks": ["risk1", "risk2"]
}}

Respond ONLY with JSON."""

        system = """You are a macro strategist synthesizing market themes from news flow.
Be concise and actionable. Focus on market implications."""

        response = self.llm_service.call(prompt, system=system, temperature=0.3, max_tokens=400)
        
        if not response:
            return self._synthesize_rule_based(topic, events)
        
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            return Theme(
                theme_id=f"{topic}_{datetime.now().strftime('%Y%m%d')}",
                name=data.get('theme_name', topic),
                description=data.get('description', f"Theme about {topic}"),
                momentum=data.get('momentum', 'stable'),
                severity=float(data.get('severity', 0.5)),
                confidence=0.85,
                article_count=len(events),
                supporting_events=[e.event_id for e in events],
                key_evidence=[e.source_headline[:50] for e in events[:3]],
                direction=data.get('direction', 'neutral'),
                affected_assets=data.get('affected_assets', []),
                trading_implication=data.get('trading_implication', 'Monitor developments'),
                first_seen=min(e.timestamp for e in events),
                last_updated=datetime.now(),
                is_new=topic not in self.active_themes,
            )
            
        except json.JSONDecodeError:
            return self._synthesize_rule_based(topic, events)
    
    def synthesize_themes(
        self,
        events: List[MacroEvent],
        use_llm: bool = True,
    ) -> List[Theme]:
        """
        Synthesize themes from a batch of events.
        """
        if not events:
            return []
        
        # Group by topic
        groups = self.group_events_by_topic(events)
        
        themes = []
        
        for topic, topic_events in groups.items():
            if len(topic_events) < self.min_articles_for_theme:
                continue
            
            if use_llm and self.llm_service:
                theme = self._synthesize_with_llm(topic, topic_events)
            else:
                theme = self._synthesize_rule_based(topic, topic_events)
            
            if theme:
                themes.append(theme)
                self.active_themes[topic] = theme
                self.theme_history.append(theme)
        
        logger.info(f"Synthesized {len(themes)} themes from {len(events)} events")
        
        return themes
    
    def get_active_themes(self) -> List[Theme]:
        """Get currently active themes."""
        return list(self.active_themes.values())
    
    def get_risk_themes(self) -> List[Theme]:
        """Get themes that are risk-off."""
        return [t for t in self.active_themes.values() if t.direction == "risk_off"]
    
    def get_overall_market_stance(self) -> Dict:
        """Get overall market stance from all themes."""
        if not self.active_themes:
            return {
                'stance': 'neutral',
                'confidence': 0.0,
                'themes': [],
            }
        
        risk_off_count = sum(1 for t in self.active_themes.values() if t.direction == "risk_off")
        risk_on_count = sum(1 for t in self.active_themes.values() if t.direction == "risk_on")
        
        if risk_off_count > risk_on_count:
            stance = 'risk_off'
        elif risk_on_count > risk_off_count:
            stance = 'risk_on'
        else:
            stance = 'neutral'
        
        avg_confidence = sum(t.confidence for t in self.active_themes.values()) / len(self.active_themes)
        
        return {
            'stance': stance,
            'confidence': avg_confidence,
            'risk_off_themes': risk_off_count,
            'risk_on_themes': risk_on_count,
            'total_themes': len(self.active_themes),
            'accelerating_themes': sum(1 for t in self.active_themes.values() if t.momentum == "accelerating"),
        }
    
    def get_stats(self) -> Dict:
        """Get synthesizer statistics."""
        return {
            'active_themes': len(self.active_themes),
            'total_themes_created': len(self.theme_history),
            'risk_off_themes': len(self.get_risk_themes()),
            'market_stance': self.get_overall_market_stance(),
        }
