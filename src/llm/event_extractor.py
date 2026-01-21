"""
LLM Event Extractor - Extracts structured macro events from news articles.

Uses LLM only for complex articles that need reasoning.
Simple articles are handled by rules.
"""
import logging
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from .llm_service import LLMService

logger = logging.getLogger(__name__)


@dataclass
class MacroEvent:
    """Structured macro event extracted from news."""
    event_id: str
    timestamp: datetime
    event_type: str  # central_bank, inflation, growth, geopolitics, etc.
    actor: str  # Fed, ECB, OPEC, etc.
    action: str  # rate_hike, hawkish_guidance, surprise, etc.
    direction: str  # bullish, bearish, hawkish, dovish, risk_on, risk_off
    severity: float  # 0-1
    confidence: float  # 0-1
    
    # Market implications
    equity_impact: str  # bullish, bearish, neutral
    bond_impact: str
    fx_impact: str
    commodity_impact: str
    
    # Affected assets
    affected_tickers: List[str]
    affected_sectors: List[str]
    
    # Explanation
    rationale: str
    key_phrases: List[str]
    
    # Metadata
    source_headline: str
    source_url: Optional[str] = None
    was_llm_extracted: bool = False


# Keywords that indicate macro-relevant content
MACRO_KEYWORDS = {
    'central_bank': ['fed', 'federal reserve', 'ecb', 'boj', 'bank of england', 'rate decision', 
                     'monetary policy', 'interest rate', 'quantitative', 'powell', 'lagarde'],
    'inflation': ['inflation', 'cpi', 'pce', 'price index', 'consumer prices', 'deflation', 
                  'disinflation', 'sticky prices'],
    'growth': ['gdp', 'economic growth', 'recession', 'expansion', 'pmi', 'retail sales',
               'industrial production', 'employment', 'jobless', 'payrolls', 'nfp'],
    'geopolitics': ['war', 'invasion', 'sanctions', 'tariff', 'trade war', 'conflict',
                    'tension', 'military', 'attack', 'crisis'],
    'financial_stress': ['bank failure', 'credit crisis', 'liquidity', 'default', 'bailout',
                         'bankruptcy', 'contagion', 'stress'],
    'energy': ['opec', 'oil', 'crude', 'natural gas', 'energy crisis', 'production cut',
               'supply disruption'],
}

# Direction indicators
DIRECTION_KEYWORDS = {
    'hawkish': ['hawkish', 'tighten', 'higher for longer', 'more hikes', 'inflation concerns'],
    'dovish': ['dovish', 'cut', 'ease', 'pause', 'pivot', 'disinflation'],
    'bullish': ['surge', 'rally', 'boom', 'strong', 'beat', 'exceed', 'outperform'],
    'bearish': ['plunge', 'crash', 'weak', 'miss', 'disappoint', 'underperform'],
    'risk_off': ['fear', 'panic', 'sell-off', 'flight to safety', 'haven'],
    'risk_on': ['optimism', 'appetite', 'rally', 'relief'],
}


class LLMEventExtractor:
    """
    Extracts structured macro events from news articles.
    
    Strategy:
    1. Pre-filter articles using keywords (no LLM needed)
    2. For macro-relevant articles, extract events
    3. Use LLM only for complex/ambiguous articles
    4. Fall back to rule-based for simple cases
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        llm_threshold: float = 0.6,  # Complexity threshold for using LLM
        api_key: Optional[str] = None,
    ):
        self.llm_service = llm_service
        self.llm_threshold = llm_threshold
        
        # Initialize LLM if API key provided
        if api_key and not llm_service:
            self.llm_service = LLMService(api_key=api_key)
        
        # Stats
        self.articles_processed = 0
        self.events_extracted = 0
        self.llm_calls = 0
        self.rule_based_extractions = 0
    
    def is_macro_relevant(self, headline: str, summary: str) -> tuple[bool, str]:
        """
        Check if article is macro-relevant.
        Returns (is_relevant, event_type).
        """
        text = (headline + " " + summary).lower()
        
        for event_type, keywords in MACRO_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return True, event_type
        
        return False, ""
    
    def extract_direction(self, text: str) -> str:
        """Extract direction/sentiment from text."""
        text_lower = text.lower()
        
        for direction, keywords in DIRECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return direction
        
        return "neutral"
    
    def estimate_complexity(self, headline: str, summary: str) -> float:
        """
        Estimate how complex the article is.
        Higher = need LLM, Lower = can use rules.
        """
        text = headline + " " + summary
        
        complexity = 0.0
        
        # Multiple event types = complex
        event_types_found = 0
        for keywords in MACRO_KEYWORDS.values():
            if any(k in text.lower() for k in keywords):
                event_types_found += 1
        
        if event_types_found > 1:
            complexity += 0.3
        
        # Conflicting signals = complex
        directions_found = []
        for direction, keywords in DIRECTION_KEYWORDS.items():
            if any(k in text.lower() for k in keywords):
                directions_found.append(direction)
        
        # Check for conflicts
        if 'hawkish' in directions_found and 'dovish' in directions_found:
            complexity += 0.4
        if 'bullish' in directions_found and 'bearish' in directions_found:
            complexity += 0.4
        
        # Long summary = complex
        if len(summary) > 500:
            complexity += 0.2
        
        # Numbers and percentages = need precision
        if re.search(r'\d+\.?\d*%', text):
            complexity += 0.1
        
        return min(1.0, complexity)
    
    def _extract_rule_based(
        self,
        headline: str,
        summary: str,
        event_type: str,
        timestamp: datetime,
        url: Optional[str] = None,
    ) -> MacroEvent:
        """Extract event using rules (no LLM)."""
        
        self.rule_based_extractions += 1
        text = headline + " " + summary
        
        # Extract direction
        direction = self.extract_direction(text)
        
        # Determine market impacts based on event type and direction
        equity_impact = "neutral"
        bond_impact = "neutral"
        fx_impact = "neutral"
        commodity_impact = "neutral"
        
        if event_type == "central_bank":
            if direction in ["hawkish"]:
                equity_impact = "bearish"
                bond_impact = "bearish"
                fx_impact = "bullish"  # USD
            elif direction in ["dovish"]:
                equity_impact = "bullish"
                bond_impact = "bullish"
                fx_impact = "bearish"
        
        elif event_type == "inflation":
            if direction in ["bullish", "hawkish"]:  # Higher inflation
                equity_impact = "bearish"
                bond_impact = "bearish"
            elif direction in ["bearish", "dovish"]:  # Lower inflation
                equity_impact = "bullish"
                bond_impact = "bullish"
        
        elif event_type == "growth":
            if direction in ["bullish"]:
                equity_impact = "bullish"
            elif direction in ["bearish"]:
                equity_impact = "bearish"
        
        elif event_type == "geopolitics":
            if direction in ["bearish", "risk_off"]:
                equity_impact = "bearish"
                commodity_impact = "bullish"  # Safe haven
        
        elif event_type == "energy":
            commodity_impact = direction
        
        # Extract actor from keywords
        actor = "Unknown"
        for keyword in ['fed', 'federal reserve']:
            if keyword in text.lower():
                actor = "Federal Reserve"
                break
        for keyword in ['ecb']:
            if keyword in text.lower():
                actor = "ECB"
                break
        
        # Extract key phrases
        key_phrases = []
        for direction_type, keywords in DIRECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text.lower():
                    key_phrases.append(keyword)
        
        return MacroEvent(
            event_id=f"{event_type}_{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            event_type=event_type,
            actor=actor,
            action=event_type,
            direction=direction,
            severity=0.5,  # Default medium
            confidence=0.6,  # Lower for rule-based
            equity_impact=equity_impact,
            bond_impact=bond_impact,
            fx_impact=fx_impact,
            commodity_impact=commodity_impact,
            affected_tickers=[],
            affected_sectors=[],
            rationale=f"Rule-based extraction: {event_type} event with {direction} signal",
            key_phrases=key_phrases[:5],
            source_headline=headline,
            source_url=url,
            was_llm_extracted=False,
        )
    
    def _extract_with_llm(
        self,
        headline: str,
        summary: str,
        event_type: str,
        timestamp: datetime,
        url: Optional[str] = None,
    ) -> Optional[MacroEvent]:
        """Extract event using LLM for complex articles."""
        
        if not self.llm_service or not self.llm_service.is_available():
            # Fall back to rule-based
            return self._extract_rule_based(headline, summary, event_type, timestamp, url)
        
        self.llm_calls += 1
        
        prompt = f"""Analyze this financial news article and extract a structured macro event.

HEADLINE: {headline}

SUMMARY: {summary}

Extract the following in JSON format:
{{
    "event_type": "central_bank|inflation|growth|geopolitics|financial_stress|energy",
    "actor": "who is taking action (e.g., Federal Reserve, ECB, OPEC)",
    "action": "what is happening (e.g., rate_hike, hawkish_guidance)",
    "direction": "hawkish|dovish|bullish|bearish|risk_on|risk_off|neutral",
    "severity": 0.0-1.0,
    "equity_impact": "bullish|bearish|neutral",
    "bond_impact": "bullish|bearish|neutral",
    "fx_impact": "bullish|bearish|neutral",
    "commodity_impact": "bullish|bearish|neutral",
    "affected_sectors": ["list", "of", "sectors"],
    "rationale": "brief explanation of why this matters for markets",
    "key_phrases": ["important", "phrases", "from", "article"]
}}

Respond ONLY with the JSON, no other text."""

        system = """You are a financial analyst expert at extracting market-relevant events from news.
Focus on macro-economic implications and market impact.
Be precise and conservative in your assessments."""

        response = self.llm_service.call(prompt, system=system, temperature=0.2, max_tokens=500)
        
        if not response:
            return self._extract_rule_based(headline, summary, event_type, timestamp, url)
        
        try:
            # Parse JSON from response
            content = response.content.strip()
            
            # Try to extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            return MacroEvent(
                event_id=f"{data.get('event_type', event_type)}_{timestamp.strftime('%Y%m%d%H%M%S')}",
                timestamp=timestamp,
                event_type=data.get('event_type', event_type),
                actor=data.get('actor', 'Unknown'),
                action=data.get('action', event_type),
                direction=data.get('direction', 'neutral'),
                severity=float(data.get('severity', 0.5)),
                confidence=0.85,  # Higher for LLM
                equity_impact=data.get('equity_impact', 'neutral'),
                bond_impact=data.get('bond_impact', 'neutral'),
                fx_impact=data.get('fx_impact', 'neutral'),
                commodity_impact=data.get('commodity_impact', 'neutral'),
                affected_tickers=[],
                affected_sectors=data.get('affected_sectors', []),
                rationale=data.get('rationale', 'LLM-extracted event'),
                key_phrases=data.get('key_phrases', []),
                source_headline=headline,
                source_url=url,
                was_llm_extracted=True,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse LLM response: {e}")
            return self._extract_rule_based(headline, summary, event_type, timestamp, url)
    
    def extract_event(
        self,
        headline: str,
        summary: str,
        timestamp: datetime,
        url: Optional[str] = None,
        overall_sentiment: float = 0,
        force_llm: bool = False,
    ) -> Optional[MacroEvent]:
        """
        Extract a structured event from an article.
        
        Returns None if article is not macro-relevant.
        """
        self.articles_processed += 1
        
        # Check relevance
        is_relevant, event_type = self.is_macro_relevant(headline, summary)
        
        if not is_relevant:
            return None
        
        # Estimate complexity
        complexity = self.estimate_complexity(headline, summary)
        
        # Decide: LLM or rules?
        if force_llm or complexity >= self.llm_threshold:
            event = self._extract_with_llm(headline, summary, event_type, timestamp, url)
        else:
            event = self._extract_rule_based(headline, summary, event_type, timestamp, url)
        
        if event:
            self.events_extracted += 1
        
        return event
    
    def extract_events_batch(
        self,
        articles: List[Dict],
    ) -> List[MacroEvent]:
        """Extract events from a batch of articles."""
        events = []
        
        for article in articles:
            event = self.extract_event(
                headline=article.get('headline', article.get('title', '')),
                summary=article.get('summary', article.get('body', '')),
                timestamp=article.get('timestamp', datetime.now()),
                url=article.get('url'),
                overall_sentiment=article.get('overall_sentiment', 0),
            )
            
            if event:
                events.append(event)
        
        logger.info(f"Extracted {len(events)} events from {len(articles)} articles")
        
        return events
    
    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        return {
            'articles_processed': self.articles_processed,
            'events_extracted': self.events_extracted,
            'extraction_rate': self.events_extracted / max(1, self.articles_processed),
            'llm_calls': self.llm_calls,
            'rule_based_extractions': self.rule_based_extractions,
            'llm_usage_rate': self.llm_calls / max(1, self.events_extracted),
        }
