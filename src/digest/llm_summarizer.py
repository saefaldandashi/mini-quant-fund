"""
LLM Summarizer for Daily Digest.

Uses structured prompts to generate investor-grade summaries
with market impact analysis for US and GCC markets.

Key features:
- Deterministic prompt contract
- Output validation and parsing
- Retry logic for invalid outputs
- Source citation enforcement
"""

import logging
import re
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .schema import (
    FeedItem,
    DigestCategory,
    CategorySummary,
    MarketImpact,
    ExecutiveBrief,
    CATEGORY_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)


# ============================================================
# PROMPT TEMPLATES
# ============================================================

CATEGORY_SYSTEM_PROMPT = """You are a senior macro strategist at a global investment bank.
You write concise, actionable market intelligence for portfolio managers.

STRICT RULES:
1. ONLY use information from the provided articles. DO NOT add external facts.
2. EVERY bullet MUST cite [Source] at the end.
3. Use conditional language: "could", "may", "suggests", "indicates".
4. NO predictions. Only conditional impact analysis.
5. Stay under 1500 characters total.
6. If articles are insufficient, say "Insufficient data" rather than hallucinate.

OUTPUT FORMAT (EXACT):
Follow this structure exactly. Use the exact headers shown.

WHAT_HAPPENED:
- [Bullet 1 describing key event] [Source]
- [Bullet 2 if relevant] [Source]
- [Bullet 3 if relevant] [Source]

WHY_IT_MATTERS:
- [Why this is significant for markets] [Source]
- [Additional context if needed] [Source]

MARKET_IMPACT_US:
- [Impact on specific US asset/rate] [Source]
- [Impact on another US asset] [Source]
- [Additional impacts if clear] [Source]

MARKET_IMPACT_GCC:
- [Impact on Brent/oil prices] [Source]
- [Impact on GCC equities or rates via USD peg] [Source]
- [Additional GCC-specific impacts] [Source]

CONFIDENCE: [Low|Medium|High]

WATCHLIST:
- [What to monitor next] [Source]
- [Additional watchlist item if relevant] [Source]"""


CATEGORY_USER_PROMPT = """Category: {category_name}

Articles to summarize:

{articles}

Generate a structured summary following the EXACT format specified.
Focus on market implications for both US and GCC markets."""


EXECUTIVE_SYSTEM_PROMPT = """You are a Chief Investment Strategist writing the executive brief for a daily intelligence report.

STRICT RULES:
1. Synthesize the MOST important points from ALL categories provided.
2. EVERY bullet MUST cite [Source].
3. Maximum 5 takeaways, each under 150 characters.
4. Themes should be 2-4 word phrases derived from the content.
5. Risk tone based on aggregate sentiment of the articles.

OUTPUT FORMAT (EXACT):

TOP_TAKEAWAYS:
- [Most important takeaway] [Source]
- [Second takeaway] [Source]
- [Third takeaway] [Source]
- [Fourth takeaway if warranted] [Source]
- [Fifth takeaway if warranted] [Source]

TODAYS_THEMES:
- [Theme 1]
- [Theme 2]
- [Theme 3]

RISK_TONE: [Risk-On|Risk-Off|Neutral]"""


EXECUTIVE_USER_PROMPT = """Digest Date: {date}

Category Summaries:
{summaries}

Generate an executive brief following the EXACT format specified."""


# ============================================================
# PARSER
# ============================================================

class SummaryParser:
    """Parses LLM output into structured objects."""
    
    @staticmethod
    def parse_category_summary(text: str) -> Tuple[Optional[CategorySummary], Optional[str]]:
        """
        Parse category summary from LLM output.
        
        Returns:
            (CategorySummary, None) on success
            (None, error_message) on failure
        """
        try:
            # Extract sections using regex
            sections = {}
            
            # Define section patterns
            patterns = {
                'what_happened': r'WHAT_HAPPENED:\s*\n((?:- .+\n?)+)',
                'why_it_matters': r'WHY_IT_MATTERS:\s*\n((?:- .+\n?)+)',
                'market_impact_us': r'MARKET_IMPACT_US:\s*\n((?:- .+\n?)+)',
                'market_impact_gcc': r'MARKET_IMPACT_GCC:\s*\n((?:- .+\n?)+)',
                'confidence': r'CONFIDENCE:\s*(\w+)',
                'watchlist': r'WATCHLIST:\s*\n((?:- .+\n?)+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    sections[key] = match.group(1).strip()
            
            # Parse bullet points
            def parse_bullets(section_text: str) -> List[str]:
                if not section_text:
                    return []
                bullets = []
                for line in section_text.split('\n'):
                    line = line.strip()
                    if line.startswith('- '):
                        bullets.append(line[2:].strip())
                return bullets
            
            # Build summary object
            what_happened = parse_bullets(sections.get('what_happened', ''))
            why_it_matters = parse_bullets(sections.get('why_it_matters', ''))
            market_us_bullets = parse_bullets(sections.get('market_impact_us', ''))
            market_gcc_bullets = parse_bullets(sections.get('market_impact_gcc', ''))
            watchlist = parse_bullets(sections.get('watchlist', ''))
            
            confidence_raw = sections.get('confidence', 'Medium').strip()
            confidence = 'Medium'
            if 'high' in confidence_raw.lower():
                confidence = 'High'
            elif 'low' in confidence_raw.lower():
                confidence = 'Low'
            
            # Validate minimum content
            if not what_happened:
                return None, "Missing WHAT_HAPPENED section"
            
            summary = CategorySummary(
                what_happened=what_happened[:3],
                why_it_matters=why_it_matters[:2],
                market_impact_us=MarketImpact(
                    bullets=market_us_bullets[:4],
                    confidence=confidence
                ),
                market_impact_gcc=MarketImpact(
                    bullets=market_gcc_bullets[:4],
                    confidence=confidence
                ),
                watchlist=watchlist[:3],
                confidence=confidence,
            )
            
            return summary, None
            
        except Exception as e:
            return None, f"Parse error: {str(e)}"
    
    @staticmethod
    def parse_executive_brief(text: str) -> Tuple[Optional[ExecutiveBrief], Optional[str]]:
        """Parse executive brief from LLM output."""
        try:
            sections = {}
            
            patterns = {
                'takeaways': r'TOP_TAKEAWAYS:\s*\n((?:- .+\n?)+)',
                'themes': r'TODAYS_THEMES:\s*\n((?:- .+\n?)+)',
                'risk_tone': r'RISK_TONE:\s*(.+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    sections[key] = match.group(1).strip()
            
            def parse_bullets(section_text: str) -> List[str]:
                if not section_text:
                    return []
                bullets = []
                for line in section_text.split('\n'):
                    line = line.strip()
                    if line.startswith('- '):
                        bullets.append(line[2:].strip())
                return bullets
            
            takeaways = parse_bullets(sections.get('takeaways', ''))
            themes = parse_bullets(sections.get('themes', ''))
            risk_tone_raw = sections.get('risk_tone', 'Neutral').strip()
            
            risk_tone = 'Neutral'
            if 'risk-on' in risk_tone_raw.lower() or 'risk on' in risk_tone_raw.lower():
                risk_tone = 'Risk-On'
            elif 'risk-off' in risk_tone_raw.lower() or 'risk off' in risk_tone_raw.lower():
                risk_tone = 'Risk-Off'
            
            if not takeaways:
                return None, "Missing TOP_TAKEAWAYS section"
            
            brief = ExecutiveBrief(
                top_takeaways=takeaways[:5],
                todays_themes=themes[:4],
                risk_tone=risk_tone,
            )
            
            return brief, None
            
        except Exception as e:
            return None, f"Parse error: {str(e)}"


# ============================================================
# SUMMARIZER
# ============================================================

class LLMSummarizer:
    """Generates structured summaries using LLM."""
    
    def __init__(
        self,
        llm_service=None,
        max_retries: int = 2,
        max_chars_per_summary: int = 2000,
    ):
        self.llm_service = llm_service
        self.max_retries = max_retries
        self.max_chars_per_summary = max_chars_per_summary
        self.parser = SummaryParser()
        
        # Initialize LLM service if not provided
        if self.llm_service is None:
            try:
                from src.llm.llm_service import LLMService
                self.llm_service = LLMService(provider="auto")
            except Exception as e:
                logger.warning(f"Could not initialize LLM service: {e}")
                self.llm_service = None
    
    def summarize_category(
        self,
        category: DigestCategory,
        items: List[FeedItem],
    ) -> Tuple[Optional[CategorySummary], Optional[str]]:
        """
        Generate summary for a category.
        
        Args:
            category: The digest category
            items: Selected feed items for this category
            
        Returns:
            (CategorySummary, None) on success
            (None, error_message) on failure
        """
        if not items:
            return None, "No items provided"
            
        if self.llm_service is None:
            return self._fallback_summary(items), None
        
        # Build article text for prompt
        articles_text = self._format_articles(items)
        category_name = CATEGORY_DISPLAY_NAMES.get(category, str(category))
        
        user_prompt = CATEGORY_USER_PROMPT.format(
            category_name=category_name,
            articles=articles_text,
        )
        
        # Call LLM with retries
        for attempt in range(self.max_retries):
            try:
                # Use LLMService.call() method
                response = self.llm_service.call(
                    prompt=user_prompt,
                    system=CATEGORY_SYSTEM_PROMPT,
                    max_tokens=1500,
                    temperature=0.3,  # Low for consistency
                )
                
                if not response or not response.content:
                    continue
                
                # Parse response (LLMService.call returns LLMResponse object)
                summary, error = self.parser.parse_category_summary(response.content)
                
                if summary:
                    return summary, None
                    
                # If parse failed, retry with stricter prompt
                logger.warning(f"Parse failed (attempt {attempt+1}): {error}")
                
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                
        # All retries failed - use fallback
        return self._fallback_summary(items), "LLM failed, using fallback"
    
    def generate_executive_brief(
        self,
        date: str,
        category_summaries: Dict[DigestCategory, CategorySummary],
    ) -> Tuple[Optional[ExecutiveBrief], Optional[str]]:
        """
        Generate executive brief from category summaries.
        
        Args:
            date: Digest date
            category_summaries: Dict of category -> summary
            
        Returns:
            (ExecutiveBrief, None) on success
            (None, error_message) on failure
        """
        if not category_summaries:
            return ExecutiveBrief(
                top_takeaways=["No significant market-moving events today."],
                todays_themes=["Quiet Markets"],
                risk_tone="Neutral",
            ), None
            
        if self.llm_service is None:
            return self._fallback_executive_brief(category_summaries), None
        
        # Format summaries for prompt
        summaries_text = self._format_summaries(category_summaries)
        
        user_prompt = EXECUTIVE_USER_PROMPT.format(
            date=date,
            summaries=summaries_text,
        )
        
        for attempt in range(self.max_retries):
            try:
                # Use LLMService.call() method
                response = self.llm_service.call(
                    prompt=user_prompt,
                    system=EXECUTIVE_SYSTEM_PROMPT,
                    max_tokens=800,
                    temperature=0.3,
                )
                
                if not response or not response.content:
                    continue
                
                # LLMService.call returns LLMResponse object
                brief, error = self.parser.parse_executive_brief(response.content)
                
                if brief:
                    return brief, None
                    
                logger.warning(f"Executive brief parse failed (attempt {attempt+1}): {error}")
                
            except Exception as e:
                logger.warning(f"Executive brief LLM failed (attempt {attempt+1}): {e}")
                
        return self._fallback_executive_brief(category_summaries), "LLM failed, using fallback"
    
    def _format_articles(self, items: List[FeedItem]) -> str:
        """Format articles for prompt."""
        articles = []
        for i, item in enumerate(items, 1):
            excerpt = item.get_excerpt(max_length=400)
            articles.append(
                f"[Article {i}]\n"
                f"Source: {item.source}\n"
                f"Time: {item.timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Headline: {item.headline}\n"
                f"Content: {excerpt}\n"
            )
        return "\n".join(articles)
    
    def _format_summaries(
        self, 
        summaries: Dict[DigestCategory, CategorySummary]
    ) -> str:
        """Format category summaries for executive brief prompt."""
        parts = []
        for category, summary in summaries.items():
            cat_name = CATEGORY_DISPLAY_NAMES.get(category, str(category))
            bullets = summary.what_happened[:2]
            parts.append(
                f"[{cat_name}]\n" +
                "\n".join(f"- {b}" for b in bullets)
            )
        return "\n\n".join(parts)
    
    def _fallback_summary(self, items: List[FeedItem]) -> CategorySummary:
        """Generate fallback summary without LLM."""
        what_happened = []
        for item in items[:3]:
            what_happened.append(f"{item.headline[:100]} [{item.source}]")
        
        return CategorySummary(
            what_happened=what_happened,
            why_it_matters=["See individual articles for details."],
            market_impact_us=MarketImpact(
                bullets=["Analysis unavailable - LLM service not configured."],
                confidence="Low"
            ),
            market_impact_gcc=MarketImpact(
                bullets=["Analysis unavailable - LLM service not configured."],
                confidence="Low"
            ),
            watchlist=["Monitor developments in this space."],
            confidence="Low",
        )
    
    def _fallback_executive_brief(
        self, 
        summaries: Dict[DigestCategory, CategorySummary]
    ) -> ExecutiveBrief:
        """Generate fallback executive brief without LLM."""
        takeaways = []
        themes = []
        
        for category, summary in list(summaries.items())[:5]:
            cat_name = CATEGORY_DISPLAY_NAMES.get(category, str(category))
            themes.append(cat_name)
            if summary.what_happened:
                takeaways.append(summary.what_happened[0][:150])
        
        return ExecutiveBrief(
            top_takeaways=takeaways[:5] or ["No significant events."],
            todays_themes=themes[:4] or ["Quiet Day"],
            risk_tone="Neutral",
        )
