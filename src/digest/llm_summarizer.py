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
    TradingSignal,
    ExecutiveBrief,
    MarketOutlook,
    RecommendedPosture,
    NoiseVsSignal,
    KeyLevel,
    StrategySignals,
    CATEGORY_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)


# ============================================================
# PROMPT TEMPLATES
# ============================================================

CATEGORY_SYSTEM_PROMPT = """You are a senior macro strategist at a global investment bank.
You write detailed, actionable market intelligence for portfolio managers.

STRICT RULES:
1. ONLY use information from the provided articles. DO NOT add external facts.
2. EVERY bullet MUST cite [Source] at the end.
3. Use conditional language: "could", "may", "suggests", "indicates".
4. NO predictions. Only conditional impact analysis based on the news.
5. Be DETAILED - explain the causal chain from event → market impact.
6. If articles are insufficient, say "Insufficient data" rather than hallucinate.

OUTPUT FORMAT (EXACT):
Follow this structure exactly. Use the exact headers shown.

WHAT_HAPPENED:
- [Detailed description of key event with context] [Source]
- [Second major development if relevant] [Source]
- [Third development if relevant] [Source]

WHY_IT_MATTERS:
- [Explain WHY this is significant - the mechanism, not just the fact] [Source]
- [Historical context or precedent if mentioned] [Source]
- [Chain of effects: event → first impact → second order effects] [Source]

MARKET_IMPACT_US:
- [SPX/NDX: Specific impact with reasoning] [Source]
- [UST 2Y/10Y yields: Direction and why] [Source]
- [USD (DXY): Strength/weakness implications] [Source]
- [Credit spreads/VIX: Risk sentiment implications] [Source]
- [Sector-specific impacts if relevant] [Source]

MARKET_IMPACT_GCC:
GCC = Saudi Arabia (Tadawul), UAE (DFM, ADX), Qatar (QSE), Kuwait, Bahrain, Oman.
- [Brent/WTI oil: Impact and reasoning - crucial for GCC fiscal balances] [Source]
- [Tadawul (Saudi): Specific impact given oil dependence and Vision 2030] [Source]
- [UAE markets (DFM/ADX): Impact on financials, real estate, trade] [Source]
- [USD peg transmission: How US rate moves affect GCC rates and liquidity] [Source]
- [GCC sovereign credit/bonds: Implications for fiscal positions] [Source]
- [Regional trade/shipping: Especially if Suez/Hormuz mentioned] [Source]

CONFIDENCE: [Low|Medium|High]

WATCHLIST:
- [Key data release or event to monitor next] [Source]
- [Price level or indicator to watch] [Source]
- [Potential escalation/de-escalation trigger] [Source]

TRADING_SIGNALS:
- Direction: [BULLISH|BEARISH|NEUTRAL]
- Conviction: [1-5]
- Timeframe: [INTRADAY|DAYS|WEEKS]
- Key assets affected: [list 3-5 tickers]"""


CATEGORY_USER_PROMPT = """Category: {category_name}

Articles to summarize:

{articles}

Generate a structured summary following the EXACT format specified.
Focus on market implications for both US and GCC markets."""


EXECUTIVE_SYSTEM_PROMPT = """You are a Chief Investment Strategist writing the executive brief for a daily intelligence report.
You advise portfolio managers on how to position given today's news flow.

STRICT RULES:
1. Synthesize the MOST important points from ALL categories provided.
2. EVERY bullet MUST cite [Source].
3. Be ACTIONABLE - tell readers what this means for their portfolios.
4. Themes should be 2-4 word phrases derived from the content.
5. Risk tone based on aggregate sentiment of the articles.

OUTPUT FORMAT (EXACT):

TOP_TAKEAWAYS:
- [Most important takeaway with market implication] [Source]
- [Second takeaway] [Source]
- [Third takeaway] [Source]
- [Fourth takeaway if warranted] [Source]
- [Fifth takeaway if warranted] [Source]

TODAYS_THEMES:
- [Theme 1]
- [Theme 2]
- [Theme 3]
- [Theme 4]

RISK_TONE: [Risk-On|Risk-Off|Neutral]

MARKET_OUTLOOK:
Overall: [1-2 sentences on what today's news means for markets globally]
US Markets: [Specific outlook for US equities, bonds, USD]
GCC Markets: [Specific outlook for Tadawul, DFM, oil prices, regional trade]

RECOMMENDED_POSTURE:
- Equity exposure: [Increase|Maintain|Reduce|Neutral]
- Duration (bonds): [Extend|Maintain|Reduce|Neutral]
- Cash: [Build|Maintain|Deploy|Neutral]
- Risk assets: [Overweight|Neutral|Underweight]

NOISE_VS_SIGNAL:
- Signal (act on): [What news is truly market-moving today]
- Noise (ignore): [What news is sensational but not actionable]
- Watch (wait for): [What developments need more clarity before acting]

KEY_LEVELS_TO_WATCH:
- [Asset 1]: [Price level and significance]
- [Asset 2]: [Price level and significance]
- [Asset 3]: [Price level and significance]

STRATEGY_SIGNALS:
For ML integration - provide structured signals:
- overall_bias: [BULLISH|BEARISH|NEUTRAL]
- conviction_score: [1-10]
- volatility_expectation: [LOW|MEDIUM|HIGH|SPIKE]
- sector_tilts: [list sectors to overweight/underweight]
- risk_events_next_24h: [list key events]"""


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
        """Parse executive brief from LLM output with enhanced market guidance."""
        try:
            sections = {}
            
            # Basic patterns
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
            
            def extract_field(pattern: str, default: str = "") -> str:
                match = re.search(pattern, text, re.IGNORECASE)
                return match.group(1).strip() if match else default
            
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
            
            # Parse Market Outlook section
            market_outlook = None
            outlook_match = re.search(r'MARKET_OUTLOOK:\s*\n(.+?)(?=\n\n|\nRECOMMENDED|\nNOISE|\nKEY_LEVELS|\nSTRATEGY|$)', text, re.IGNORECASE | re.DOTALL)
            if outlook_match:
                outlook_text = outlook_match.group(1)
                overall = extract_field(r'Overall:\s*(.+?)(?=\n|$)', "")
                us_markets = extract_field(r'US Markets?:\s*(.+?)(?=\n|$)', "")
                gcc_markets = extract_field(r'GCC Markets?:\s*(.+?)(?=\n|$)', "")
                market_outlook = MarketOutlook(
                    overall=overall,
                    us_markets=us_markets,
                    gcc_markets=gcc_markets,
                )
            
            # Parse Recommended Posture
            recommended_posture = None
            posture_match = re.search(r'RECOMMENDED_POSTURE:\s*\n(.+?)(?=\n\n|\nNOISE|\nKEY_LEVELS|\nSTRATEGY|$)', text, re.IGNORECASE | re.DOTALL)
            if posture_match:
                posture_text = posture_match.group(1)
                equity = extract_field(r'Equity exposure:\s*(\w+)', "Neutral")
                duration = extract_field(r'Duration.*?:\s*(\w+)', "Neutral")
                cash = extract_field(r'Cash:\s*(\w+)', "Neutral")
                risk_assets = extract_field(r'Risk assets:\s*(\w+)', "Neutral")
                recommended_posture = RecommendedPosture(
                    equity_exposure=equity,
                    duration_bonds=duration,
                    cash_position=cash,
                    risk_assets=risk_assets,
                )
            
            # Parse Noise vs Signal
            noise_vs_signal = None
            nvs_match = re.search(r'NOISE_VS_SIGNAL:\s*\n(.+?)(?=\n\n|\nKEY_LEVELS|\nSTRATEGY|$)', text, re.IGNORECASE | re.DOTALL)
            if nvs_match:
                nvs_text = nvs_match.group(1)
                signal_match = re.search(r'Signal.*?:\s*\[(.+?)\]', nvs_text, re.IGNORECASE)
                noise_match = re.search(r'Noise.*?:\s*\[(.+?)\]', nvs_text, re.IGNORECASE)
                watch_match = re.search(r'Watch.*?:\s*\[(.+?)\]', nvs_text, re.IGNORECASE)
                noise_vs_signal = NoiseVsSignal(
                    signal_act_on=[signal_match.group(1).strip()] if signal_match else [],
                    noise_ignore=[noise_match.group(1).strip()] if noise_match else [],
                    watch_wait_for=[watch_match.group(1).strip()] if watch_match else [],
                )
            
            # Parse Key Levels
            key_levels = []
            levels_match = re.search(r'KEY_LEVELS_TO_WATCH:\s*\n((?:- .+\n?)+)', text, re.IGNORECASE)
            if levels_match:
                for line in levels_match.group(1).split('\n'):
                    if line.strip().startswith('- '):
                        parts = line[2:].split(':')
                        if len(parts) >= 2:
                            key_levels.append(KeyLevel(
                                asset=parts[0].strip(),
                                level=parts[1].strip() if len(parts) > 1 else "",
                                significance=parts[2].strip() if len(parts) > 2 else "",
                            ))
            
            # Parse Strategy Signals (for ML integration)
            strategy_signals = None
            signals_match = re.search(r'STRATEGY_SIGNALS:\s*\n(.+?)(?=$)', text, re.IGNORECASE | re.DOTALL)
            if signals_match:
                signals_text = signals_match.group(1)
                overall_bias = extract_field(r'overall_bias:\s*(\w+)', "NEUTRAL").upper()
                conviction = int(extract_field(r'conviction_score:\s*(\d+)', "5"))
                volatility = extract_field(r'volatility_expectation:\s*(\w+)', "MEDIUM").upper()
                
                # Parse sector tilts
                sector_tilts = {}
                tilts_match = re.search(r'sector_tilts:\s*\[(.+?)\]', signals_text, re.IGNORECASE)
                if tilts_match:
                    for tilt in tilts_match.group(1).split(','):
                        if ':' in tilt:
                            sector, weight = tilt.split(':')
                            sector_tilts[sector.strip()] = weight.strip()
                
                # Parse risk events
                risk_events = []
                events_match = re.search(r'risk_events_next_24h:\s*\[(.+?)\]', signals_text, re.IGNORECASE)
                if events_match:
                    risk_events = [e.strip() for e in events_match.group(1).split(',')]
                
                strategy_signals = StrategySignals(
                    overall_bias=overall_bias,
                    conviction_score=min(10, max(1, conviction)),
                    volatility_expectation=volatility,
                    sector_tilts=sector_tilts,
                    risk_events_next_24h=risk_events,
                )
            
            brief = ExecutiveBrief(
                top_takeaways=takeaways[:5],
                todays_themes=themes[:4],
                risk_tone=risk_tone,
                market_outlook=market_outlook,
                recommended_posture=recommended_posture,
                noise_vs_signal=noise_vs_signal,
                key_levels=key_levels[:3],
                strategy_signals=strategy_signals,
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
