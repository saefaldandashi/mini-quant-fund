"""
Schema definitions for Daily Digest.

Pydantic models for:
- FeedItem: Normalized feed item from intel feed
- DigestSection: Category-level summary with market impacts
- DigestOutput: Full digest output structure
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class DigestCategory(str, Enum):
    """Standard digest categories with display names."""
    CENTRAL_BANKS = "central_banks"
    INFLATION_MACRO = "inflation_macro"
    GEOPOLITICS = "geopolitics"
    SHIPPING_SUPPLY = "shipping_supply"
    ENERGY_COMMODITIES = "energy_commodities"
    FINANCIAL_STRESS = "financial_stress"
    EQUITY_CORPORATE = "equity_corporate"
    GCC_POLICY = "gcc_policy"


# Mapping from feed categories to digest categories
CATEGORY_MAPPING = {
    # From NewsCategory enum values
    "central_bank": DigestCategory.CENTRAL_BANKS,
    "macro_data": DigestCategory.INFLATION_MACRO,
    "geopolitical": DigestCategory.GEOPOLITICS,
    "sanctions": DigestCategory.GEOPOLITICS,
    "shipping_logistics": DigestCategory.SHIPPING_SUPPLY,
    "energy_commodity": DigestCategory.ENERGY_COMMODITIES,
    "financial_stress": DigestCategory.FINANCIAL_STRESS,
    "corporate_deals": DigestCategory.EQUITY_CORPORATE,
    "business_innovation": DigestCategory.EQUITY_CORPORATE,
    "elections_policy": DigestCategory.GEOPOLITICS,
    "irrelevant": None,  # Skip
    
    # Aliases for robustness (uppercase)
    "CENTRAL_BANK": DigestCategory.CENTRAL_BANKS,
    "MACRO_DATA": DigestCategory.INFLATION_MACRO,
    "GEOPOLITICAL": DigestCategory.GEOPOLITICS,
    "SANCTIONS": DigestCategory.GEOPOLITICS,
    "SHIPPING_LOGISTICS": DigestCategory.SHIPPING_SUPPLY,
    "ENERGY_COMMODITY": DigestCategory.ENERGY_COMMODITIES,
    "FINANCIAL_STRESS": DigestCategory.FINANCIAL_STRESS,
    "CORPORATE_DEALS": DigestCategory.EQUITY_CORPORATE,
    "BUSINESS_INNOVATION": DigestCategory.EQUITY_CORPORATE,
    "ELECTIONS_POLICY": DigestCategory.GEOPOLITICS,
    
    # Additional category aliases
    "military": DigestCategory.GEOPOLITICS,
    "conflict": DigestCategory.GEOPOLITICS,
    "diplomatic": DigestCategory.GEOPOLITICS,
    "trade": DigestCategory.GEOPOLITICS,
    "tariff": DigestCategory.GEOPOLITICS,
    "oil": DigestCategory.ENERGY_COMMODITIES,
    "opec": DigestCategory.ENERGY_COMMODITIES,
    "commodities": DigestCategory.ENERGY_COMMODITIES,
    "rates": DigestCategory.CENTRAL_BANKS,
    "fed": DigestCategory.CENTRAL_BANKS,
    "inflation": DigestCategory.INFLATION_MACRO,
    "gdp": DigestCategory.INFLATION_MACRO,
    "unemployment": DigestCategory.INFLATION_MACRO,
    "shipping": DigestCategory.SHIPPING_SUPPLY,
    "ports": DigestCategory.SHIPPING_SUPPLY,
    "freight": DigestCategory.SHIPPING_SUPPLY,
    "banking": DigestCategory.FINANCIAL_STRESS,
    "credit": DigestCategory.FINANCIAL_STRESS,
}

# Display names for categories
CATEGORY_DISPLAY_NAMES = {
    DigestCategory.CENTRAL_BANKS: "Central Banks & Rates",
    DigestCategory.INFLATION_MACRO: "Inflation & Macro Data",
    DigestCategory.GEOPOLITICS: "Geopolitics & Security",
    DigestCategory.SHIPPING_SUPPLY: "Shipping & Supply Chains",
    DigestCategory.ENERGY_COMMODITIES: "Energy & Commodities",
    DigestCategory.FINANCIAL_STRESS: "Financial Stress & Credit",
    DigestCategory.EQUITY_CORPORATE: "Equity & Corporate",
    DigestCategory.GCC_POLICY: "GCC Policy & Economy",
}

# Category icons for rendering
CATEGORY_ICONS = {
    DigestCategory.CENTRAL_BANKS: "🏦",
    DigestCategory.INFLATION_MACRO: "📊",
    DigestCategory.GEOPOLITICS: "🌍",
    DigestCategory.SHIPPING_SUPPLY: "🚢",
    DigestCategory.ENERGY_COMMODITIES: "🛢️",
    DigestCategory.FINANCIAL_STRESS: "📉",
    DigestCategory.EQUITY_CORPORATE: "📈",
    DigestCategory.GCC_POLICY: "🏛️",
}

# Source credibility tiers (1 = highest)
SOURCE_CREDIBILITY = {
    # Tier 1 - Major wires and official sources
    "reuters": 1,
    "bloomberg": 1,
    "wsj": 1,
    "ft": 1,
    "fed": 1,
    "ecb": 1,
    "imf": 1,
    "bis": 1,
    
    # Tier 2 - Quality financial press
    "guardian": 2,
    "nyt": 2,
    "economist": 2,
    "cnbc": 2,
    "yahoo_finance": 2,
    "marketwatch": 2,
    "barrons": 2,
    
    # Tier 3 - General news
    "ap": 3,
    "afp": 3,
    "bbc": 3,
    "cnn": 3,
    "guardian_world": 3,
    "al_jazeera": 3,
    
    # Tier 4 - Industry/specialty
    "oilprice": 4,
    "splash247": 4,
    "gcaptain": 4,
    "defensenews": 4,
    
    # Default
    "default": 5,
}


class FeedItem(BaseModel):
    """Normalized feed item from intel feed."""
    
    event_id: str
    timestamp: datetime
    headline: str
    summary: str = ""
    source: str
    url: Optional[str] = None
    
    # Classification
    category: str
    tags: List[str] = Field(default_factory=list)
    matched_keywords: List[str] = Field(default_factory=list)
    
    # Scores (0-1)
    relevance_score: float = 0.5
    impact_score: float = 0.5
    credibility_score: float = 0.5
    novelty_score: float = 1.0
    final_score: float = 0.5
    
    # Direction
    direction: str = "neutral"
    direction_confidence: float = 0.0
    
    # Affected
    affected_assets: List[str] = Field(default_factory=list)
    affected_regions: List[str] = Field(default_factory=list)
    
    # Explainability
    rationale: str = ""
    
    # Selection metadata (added during selection)
    selection_reason: Optional[str] = None
    digest_category: Optional[DigestCategory] = None
    source_tier: int = 5
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v
    
    def get_digest_category(self) -> Optional[DigestCategory]:
        """Map feed category to digest category."""
        return CATEGORY_MAPPING.get(self.category)
    
    def get_source_tier(self) -> int:
        """Get source credibility tier."""
        source_lower = self.source.lower()
        for key, tier in SOURCE_CREDIBILITY.items():
            if key in source_lower:
                return tier
        return SOURCE_CREDIBILITY["default"]
    
    def get_excerpt(self, max_length: int = 500) -> str:
        """Get clean excerpt for LLM."""
        text = self.summary or ""
        # Strip HTML tags
        import re
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > max_length:
            text = text[:max_length] + "..."
        return text
    
    class Config:
        use_enum_values = True


class MarketImpact(BaseModel):
    """Market impact assessment."""
    
    bullets: List[str] = Field(default_factory=list)
    confidence: str = "Medium"  # Low, Medium, High
    
    @field_validator('bullets', mode='after')
    @classmethod
    def limit_bullets(cls, v):
        return v[:6]  # Increased for more detail


class TradingSignal(BaseModel):
    """ML-ready trading signal from digest analysis."""
    
    direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    conviction: int = 3  # 1-5 scale
    timeframe: str = "DAYS"  # INTRADAY, DAYS, WEEKS
    affected_assets: List[str] = Field(default_factory=list)


class CategorySummary(BaseModel):
    """LLM-generated summary for a category."""
    
    what_happened: List[str] = Field(default_factory=list, description="Detailed bullets on what happened")
    why_it_matters: List[str] = Field(default_factory=list, description="Why this is significant")
    market_impact_us: MarketImpact = Field(default_factory=MarketImpact)
    market_impact_gcc: MarketImpact = Field(default_factory=MarketImpact)
    watchlist: List[str] = Field(default_factory=list, description="Items to monitor")
    confidence: str = "Medium"
    trading_signal: Optional[TradingSignal] = None  # For ML integration
    
    @field_validator('what_happened', mode='after')
    @classmethod
    def limit_what_happened(cls, v):
        return v[:5]  # Increased for more detail
    
    @field_validator('why_it_matters', mode='after')
    @classmethod
    def limit_why_it_matters(cls, v):
        return v[:4]  # Increased for more detail
    
    @field_validator('watchlist', mode='after')
    @classmethod
    def limit_watchlist(cls, v):
        return v[:5]  # Increased


class DigestSection(BaseModel):
    """A section of the digest for one category."""
    
    category: DigestCategory
    display_name: str
    icon: str
    
    # Selected items (top N)
    items: List[FeedItem] = Field(default_factory=list)
    
    # Aggregate scores
    avg_impact_score: float = 0.0
    max_impact_score: float = 0.0
    total_items_considered: int = 0
    
    # LLM-generated summary
    summary: Optional[CategorySummary] = None
    
    # Fallback if LLM fails
    llm_failed: bool = False
    llm_error: Optional[str] = None
    
    def get_display_name(self) -> str:
        return CATEGORY_DISPLAY_NAMES.get(self.category, str(self.category))
    
    def get_icon(self) -> str:
        return CATEGORY_ICONS.get(self.category, "📋")


class MarketSnapshot(BaseModel):
    """Current market data snapshot."""
    
    # US Markets
    spx: Optional[float] = None
    spx_change: Optional[float] = None
    ndx: Optional[float] = None
    ndx_change: Optional[float] = None
    dxy: Optional[float] = None
    dxy_change: Optional[float] = None
    ust_2y: Optional[float] = None
    ust_10y: Optional[float] = None
    vix: Optional[float] = None
    
    # Commodities
    brent: Optional[float] = None
    brent_change: Optional[float] = None
    wti: Optional[float] = None
    wti_change: Optional[float] = None
    gold: Optional[float] = None
    gold_change: Optional[float] = None
    
    # GCC Proxies (if available)
    tadawul: Optional[float] = None  # Saudi index
    tadawul_change: Optional[float] = None
    dfm: Optional[float] = None  # Dubai
    dfm_change: Optional[float] = None
    
    timestamp: Optional[datetime] = None


class MarketOutlook(BaseModel):
    """Overall market outlook synthesized from all categories."""
    
    overall: str = ""  # 1-2 sentences on global market implications
    us_markets: str = ""  # US equities, bonds, USD outlook
    gcc_markets: str = ""  # Tadawul, DFM, oil, regional outlook


class RecommendedPosture(BaseModel):
    """Portfolio positioning recommendations."""
    
    equity_exposure: str = "Neutral"  # Increase, Maintain, Reduce, Neutral
    duration_bonds: str = "Neutral"  # Extend, Maintain, Reduce, Neutral
    cash_position: str = "Neutral"  # Build, Maintain, Deploy, Neutral
    risk_assets: str = "Neutral"  # Overweight, Neutral, Underweight


class NoiseVsSignal(BaseModel):
    """Distinguish actionable news from noise."""
    
    signal_act_on: List[str] = Field(default_factory=list, description="Truly market-moving news")
    noise_ignore: List[str] = Field(default_factory=list, description="Sensational but not actionable")
    watch_wait_for: List[str] = Field(default_factory=list, description="Needs more clarity")


class KeyLevel(BaseModel):
    """Price level to watch."""
    
    asset: str
    level: str
    significance: str


class StrategySignals(BaseModel):
    """ML-ready signals from digest analysis."""
    
    overall_bias: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    conviction_score: int = 5  # 1-10 scale
    volatility_expectation: str = "MEDIUM"  # LOW, MEDIUM, HIGH, SPIKE
    sector_tilts: Dict[str, str] = Field(default_factory=dict)  # sector -> overweight/underweight
    risk_events_next_24h: List[str] = Field(default_factory=list)


class ExecutiveBrief(BaseModel):
    """Executive summary at top of digest - comprehensive market guidance."""
    
    top_takeaways: List[str] = Field(default_factory=list, description="Top 5 takeaways")
    todays_themes: List[str] = Field(default_factory=list, description="Key themes")
    risk_tone: Optional[str] = None  # "Risk-On", "Risk-Off", "Neutral"
    risk_score: Optional[float] = None
    
    # NEW: Enhanced guidance
    market_outlook: Optional[MarketOutlook] = None
    recommended_posture: Optional[RecommendedPosture] = None
    noise_vs_signal: Optional[NoiseVsSignal] = None
    key_levels: List[KeyLevel] = Field(default_factory=list)
    
    # NEW: ML integration
    strategy_signals: Optional[StrategySignals] = None
    
    @field_validator('top_takeaways', mode='after')
    @classmethod
    def limit_takeaways(cls, v):
        return v[:5]


class DigestMetadata(BaseModel):
    """Metadata for the digest."""
    
    date: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"
    feed_path: str = ""
    total_items_processed: int = 0
    total_items_selected: int = 0
    categories_with_content: int = 0
    llm_model: Optional[str] = None
    generation_time_seconds: float = 0.0


class DigestOutput(BaseModel):
    """Complete digest output."""
    
    metadata: DigestMetadata
    executive_brief: ExecutiveBrief
    market_snapshot: Optional[MarketSnapshot] = None
    sections: List[DigestSection] = Field(default_factory=list)
    
    # Output paths (filled after generation)
    html_path: Optional[str] = None
    pdf_path: Optional[str] = None
    json_path: Optional[str] = None
    
    def get_sections_by_impact(self) -> List[DigestSection]:
        """Get sections sorted by max impact score."""
        return sorted(
            [s for s in self.sections if s.items],
            key=lambda s: s.max_impact_score,
            reverse=True
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.dict()


class CategoryMapping(BaseModel):
    """Configuration for category mapping."""
    
    mappings: Dict[str, DigestCategory] = Field(default_factory=lambda: CATEGORY_MAPPING.copy())
    display_names: Dict[DigestCategory, str] = Field(default_factory=lambda: CATEGORY_DISPLAY_NAMES.copy())
    icons: Dict[DigestCategory, str] = Field(default_factory=lambda: CATEGORY_ICONS.copy())
    
    def get_digest_category(self, feed_category: str) -> Optional[DigestCategory]:
        """Map feed category to digest category."""
        return self.mappings.get(feed_category)
