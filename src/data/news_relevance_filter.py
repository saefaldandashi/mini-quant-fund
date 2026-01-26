"""
NEWS RELEVANCE FILTER - Rule-Based Market-Moving News Detection

This filter ensures ONLY market-moving news enters downstream systems.
Aggressively discards noise (sports, celebrity, local news) and retains
only news that plausibly affects markets.

NO LLM CLASSIFICATION - RULE-BASED ONLY - FULLY DETERMINISTIC
"""

import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from enum import Enum
import pytz


class MarketDirection(Enum):
    """Inferred market direction from news."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    OIL_UP = "oil_up"
    OIL_DOWN = "oil_down"
    RATES_UP = "rates_up"
    RATES_DOWN = "rates_down"
    USD_UP = "usd_up"
    USD_DOWN = "usd_down"
    NEUTRAL = "neutral"


class NewsCategory(Enum):
    """Primary news categories for market impact."""
    GEOPOLITICAL = "geopolitical"
    SANCTIONS = "sanctions"
    ENERGY_COMMODITY = "energy_commodity"
    SHIPPING_LOGISTICS = "shipping_logistics"
    MACRO_DATA = "macro_data"
    CENTRAL_BANK = "central_bank"
    FINANCIAL_STRESS = "financial_stress"
    ELECTIONS_POLICY = "elections_policy"
    CORPORATE_DEALS = "corporate_deals"
    BUSINESS_INNOVATION = "business_innovation"
    IRRELEVANT = "irrelevant"


@dataclass
class NewsEvent:
    """Structured news event with scoring and classification."""
    event_id: str
    timestamp: datetime
    headline: str
    summary: str
    source: str
    url: Optional[str]
    
    # Classification
    category: NewsCategory
    tags: List[str]
    matched_keywords: List[str]
    
    # Scoring (0-1)
    relevance_score: float
    impact_score: float
    credibility_score: float
    novelty_score: float
    
    # Combined score
    final_score: float
    
    # Direction inference
    direction: MarketDirection
    direction_confidence: float
    
    # Affected assets/sectors
    affected_assets: List[str]
    affected_regions: List[str]
    
    # Explainability
    rationale: str
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "headline": self.headline,
            "summary": self.summary[:300],
            "source": self.source,
            "url": self.url,
            "category": self.category.value,
            "tags": self.tags,
            "matched_keywords": self.matched_keywords,
            "relevance_score": round(self.relevance_score, 3),
            "impact_score": round(self.impact_score, 3),
            "credibility_score": round(self.credibility_score, 3),
            "novelty_score": round(self.novelty_score, 3),
            "final_score": round(self.final_score, 3),
            "direction": self.direction.value,
            "direction_confidence": round(self.direction_confidence, 3),
            "affected_assets": self.affected_assets,
            "affected_regions": self.affected_regions,
            "rationale": self.rationale,
        }


class NewsRelevanceFilter:
    """
    Rule-based news relevance filter for market-moving events.
    
    Implements deterministic, configurable, testable filtering with
    full explainability of accept/reject decisions.
    """
    
    # ============================================================
    # I. HARD KEEP CATEGORIES (PRIMARY SIGNAL)
    # ============================================================
    
    # A. GEOPOLITICAL RISK AND CONFLICT
    GEOPOLITICAL_KEYWORDS = [
        "war", "invasion", "offensive", "counteroffensive", "missile", "missiles",
        "drone", "drones", "airstrike", "airstrikes", "naval incident", "naval clash",
        "cross-border attack", "retaliation", "retaliates", "retaliatory",
        "declares war", "launches strikes", "mobilization", "mobilizes",
        "conscription", "martial law", "state of emergency",
        "ceasefire", "truce", "peace deal", "peace talks",
        "ceasefire agreed", "ceasefire collapses", "talks break down", "negotiations suspended",
        "siege", "blockade", "territorial control", "territory seized",
        "military aid", "arms shipments", "defense pact", "security agreement",
        "jets deployed", "troops deployed", "military buildup", "escalation",
    ]
    
    GEOPOLITICAL_ENTITIES = [
        "nato", "united nations", "un security council", "icc",
        "defense ministry", "ministry of defense", "pentagon",
        "middle east", "israel", "iran", "saudi arabia", "uae", "emirates",
        "eastern europe", "south china sea", "taiwan strait",
        "red sea", "strait of hormuz", "black sea", "korean peninsula",
        "gaza", "lebanon", "syria", "yemen", "iraq", "ukraine", "russia",
    ]
    
    # B. SANCTIONS, EXPORT CONTROLS, TRADE RESTRICTIONS
    SANCTIONS_KEYWORDS = [
        "sanctions", "sanctioned", "sanctions imposed", "sanctions lifted", "sanctions expanded",
        "export controls", "export ban", "import ban",
        "tariffs", "tariff hike", "trade ban", "embargo",
        "asset freeze", "assets frozen",
        "swift restrictions", "payment restrictions",
        "secondary sanctions",
        "blacklist", "entity list", "restricted list",
        "seizes assets", "retaliatory measures",
    ]
    
    # C. ENERGY SUPPLY AND COMMODITY SHOCKS
    ENERGY_KEYWORDS = [
        "opec", "opec+", "production cut", "output cut", "production increase",
        "oil supply", "gas supply", "lng supply",
        "pipeline sabotage", "pipeline leak",
        "refinery fire", "refinery explosion", "refinery shutdown",
        "force majeure",
        "tanker attack", "shipping insurance",
        "port closure",
        "export ban", "price cap",
        "strategic reserves release",
        "mining strike", "rare earth disruption",
        "oil prices", "crude oil", "natural gas", "brent", "wti",
    ]
    
    # D. SHIPPING LANES, CHOKEPOINTS, LOGISTICS
    SHIPPING_KEYWORDS = [
        "shipping disrupted", "shipping disruption",
        "reroute", "rerouting",
        "suspend transit", "transit suspended",
        "blockade",
        "freight rates surge",
        "insurance premiums jump",
        "port closed", "port strike",
        "aviation route closed",
        "supply chain disruption", "logistics crisis",
    ]
    
    CHOKEPOINTS = [
        "suez canal", "panama canal", "strait of hormuz",
        "bab el-mandeb", "strait of malacca", "bosphorus",
    ]
    
    # E. MACROECONOMIC DATA AND SURPRISES
    MACRO_KEYWORDS = [
        "cpi", "pce", "inflation", "core inflation",
        "inflation expectations",
        "unemployment", "jobless claims",
        "nonfarm payrolls", "payrolls",
        "wages", "wage growth",
        "gdp", "pmi", "ism",
        "retail sales",
        "industrial production",
        "housing starts", "housing permits",
        "delinquencies", "defaults",
    ]
    
    SURPRISE_LANGUAGE = [
        "unexpectedly", "unexpected",
        "beats forecasts", "beats estimates",
        "misses forecasts", "misses estimates",
        "surges", "plunges",
        "highest since", "lowest since",
        "revised sharply", "surprise",
    ]
    
    # F. CENTRAL BANKS, RATES, LIQUIDITY
    CENTRAL_BANK_KEYWORDS = [
        "interest rate hike", "rate cut", "rate hold", "rate decision",
        "forward guidance",
        "dot plot",
        "minutes",
        "quantitative easing", "qe",
        "quantitative tightening", "qt",
        "balance sheet runoff",
        "liquidity facility",
        "emergency facility",
        "yield curve control",
        "currency intervention",
        "hawkish", "dovish", "pivot",
        "monetary policy",
    ]
    
    CENTRAL_BANK_ENTITIES = [
        "federal reserve", "fed", "fomc",
        "ecb", "european central bank",
        "bank of england", "boe",
        "bank of japan", "boj",
        "snb", "swiss national bank",
        "pboc", "people's bank of china",
        "rbi", "reserve bank of india",
        "rba", "reserve bank of australia",
    ]
    
    # G. SYSTEMIC FINANCIAL STRESS
    FINANCIAL_STRESS_KEYWORDS = [
        "bank run",
        "bailout", "rescue",
        "resolution",
        "emergency merger",
        "capital shortfall",
        "liquidity crisis",
        "default",
        "restructuring",
        "sovereign debt",
        "imf support",
        "downgrade", "credit downgrade",
        "distressed",
        "contagion",
        "funding stress",
        "margin calls",
        "banking crisis", "financial crisis",
    ]
    
    # H. ELECTIONS AND POLICY SHOCKS
    ELECTIONS_KEYWORDS = [
        "election", "snap election",
        "contested election",
        "coup",
        "impeachment",
        "referendum",
        "government shutdown",
        "debt ceiling",
        "fiscal stimulus",
        "capital controls",
        "nationalization",
        "tariff plan",
        "budget crisis",
        "political crisis",
    ]
    
    # I. MAJOR CORPORATE DEALS
    CORPORATE_KEYWORDS = [
        "merger", "acquisition", "acquire", "acquired",
        "takeover", "buyout",
        "strategic stake",
        "spin-off", "spinoff",
        "ipo", "secondary offering",
        "bankruptcy", "chapter 11",
        "restructuring",
        "antitrust",
        "doj", "ftc",
        "eu competition authority",
        "regulator blocks", "regulator approves",
    ]
    
    # J. BUSINESS INNOVATION (ONLY IF BIG AND REAL)
    INNOVATION_KEYWORDS = [
        "billion investment", "billion deal",
        "multi-year contract",
        "exclusive supply",
        "mass production",
        "breakthrough",
        "fda approval",
        "phase 3",
        "contract award",
        "capex",
        "foundry",
        "tape-out",
        "yield improvement",
    ]
    
    INNOVATION_VERTICALS = [
        "semiconductor", "semiconductors", "chips",
        "ai compute", "artificial intelligence",
        "defense", "aerospace",
        "biotech", "pharmaceutical",
        "cloud", "data center",
        "energy storage", "battery",
        "ev", "electric vehicle",
        "industrial automation",
        "telecom", "5g",
    ]
    
    # ============================================================
    # II. HARD DISCARD CATEGORIES (NEGATIVE FILTERS)
    # ============================================================
    
    HARD_DISCARD_KEYWORDS = [
        # Sports (use word boundaries to avoid false positives like "inflation" matching "nfl")
        "football game", "soccer match", "basketball game", "baseball game", 
        "tennis match", "golf tournament",
        "cricket match", "rugby match", "olympics 2024", "world cup 2026",
        "premier league match", " nba ", " nfl ", " mlb ", " nhl ",
        "scored a goal", "touchdown pass", "goal scored", "sports match",
        
        # Entertainment/Celebrity
        "celebrity", "kardashian", "hollywood", "movie", "film release",
        "album", "concert", "grammy", "oscar", "emmy", "red carpet",
        "dating", "married", "divorced", "pregnant",
        
        # Lifestyle
        "recipe", "cooking", "restaurant review", "travel tips",
        "fashion", "outfit", "makeup", "beauty tips",
        "diet", "workout", "yoga", "meditation",
        "home decor", "gardening",
        
        # Local news
        "local crime", "robbery", "burglary", "car accident",
        "house fire", "local police", "city council",
        "school board", "local election",
        "obituary", "funeral",
        
        # Irrelevant weather
        "weather forecast", "sunny", "cloudy",
        
        # Opinion/Editorial (unless paired with policy)
        "opinion:", "editorial:", "commentary:",
        "letter to the editor",
    ]
    
    # ============================================================
    # III. MARKET-MOVING TRIGGER PHRASES (BOOSTERS)
    # ============================================================
    
    ESCALATION_VERBS = [
        "declares", "announces", "imposes", "expands", "lifts", "blocks",
        "approves", "seizes", "deploys", "mobilizes", "retaliates",
        "strikes", "attacks", "invades", "shoots down", "intercepts",
        "blockade", "embargo", "force majeure", "suspends", "halts",
        "shutdown", "blackout",
    ]
    
    SURPRISE_MAGNITUDE = [
        "unexpected", "shock", "surge", "plunge", "soars", "tumbles",
        "record", "highest since", "lowest since", "misses estimates",
        "beats forecasts", "revised sharply", "emergency", "rare", "historic",
        "unprecedented", "crisis",
    ]
    
    FINANCIAL_STRESS_PHRASES = [
        "bailout", "rescue", "capital shortfall", "default", "restructuring",
        "bank run", "liquidity", "margin calls", "downgrade", "junk",
        "distressed", "contagion",
    ]
    
    LOGISTICS_PHRASES = [
        "reroute", "divert", "port closed", "shipping disrupted",
        "insurance premiums", "freight rates", "bottleneck",
    ]
    
    DEAL_PHRASES = [
        "acquire", "merger", "takeover", "buyout", "strategic stake",
        "spin-off", "ipo", "antitrust", "doj", "ftc",
    ]
    
    # ============================================================
    # IV. NUMERIC SIGNAL BOOSTS
    # ============================================================
    
    NUMERIC_PATTERNS = [
        r'\$\s*\d+', r'€\s*\d+', r'£\s*\d+', r'¥\s*\d+',
        r'\d+\s*billion', r'\d+\s*million', r'\d+\s*trillion',
        r'\d+\.?\d*\s*%', r'\d+\s*bps', r'\d+\s*basis points',
        r'\d+\s*barrels', r'\d+\s*bpd', r'\d+\s*mb/d',
        r'\d+\s*tons', r'\d+\s*tonnes',
        r'effective immediately',
    ]
    
    # ============================================================
    # V. SOURCE CREDIBILITY WEIGHTING
    # ============================================================
    
    TIER_1_SOURCES = [
        "reuters", "bloomberg", "financial times", "ft.com",
        "wall street journal", "wsj", "associated press", "ap news",
        "fed", "federal reserve", "ecb", "bank of england",
        "treasury", "ministry", "government", "official",
    ]
    
    TIER_2_SOURCES = [
        "cnbc", "bbc", "cnn", "nytimes", "new york times",
        "washington post", "guardian", "economist",
        "marketwatch", "yahoo finance", "barrons",
    ]
    
    TIER_3_SOURCES = [
        "blog", "substack", "medium", "reddit", "twitter", "x.com",
    ]
    
    # ============================================================
    # VI. DIRECTION INFERENCE RULES
    # ============================================================
    
    RISK_OFF_TRIGGERS = [
        "war", "invasion", "airstrike", "missile", "sanctions",
        "escalation", "crisis", "default", "bank run", "contagion",
        "hawkish", "rate hike", "tightening",
    ]
    
    RISK_ON_TRIGGERS = [
        "ceasefire", "peace", "sanctions lifted", "dovish", "rate cut",
        "stimulus", "rescue", "bailout successful", "recovery",
    ]
    
    OIL_UP_TRIGGERS = [
        "opec cut", "production cut", "pipeline attack", "refinery fire",
        "shipping disrupted", "strait of hormuz", "tanker attack",
        "force majeure", "supply disruption",
    ]
    
    OIL_DOWN_TRIGGERS = [
        "opec increase", "production increase", "demand falls",
        "recession fears", "strategic reserves",
    ]
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    RELEVANCE_THRESHOLD = 0.3
    IMPACT_THRESHOLD = 0.2
    FINAL_SCORE_THRESHOLD = 0.25
    
    def __init__(self, output_dir: str = "outputs/news_filter"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Deduplication cache
        self.seen_headlines: Set[str] = set()
        self.seen_urls: Set[str] = set()
        
        # Stats
        self.stats = {
            "total_processed": 0,
            "accepted": 0,
            "rejected": 0,
            "rejected_reasons": {},
        }
        
        # Pre-compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.numeric_patterns = [re.compile(p, re.IGNORECASE) for p in self.NUMERIC_PATTERNS]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return text.lower().strip()
    
    def _generate_id(self, headline: str, source: str) -> str:
        """Generate unique event ID."""
        content = f"{headline[:100]}_{source}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _check_hard_discard(self, text: str) -> Tuple[bool, str]:
        """
        Check if article should be hard discarded.
        Returns (should_discard, reason).
        """
        text_lower = " " + self._normalize_text(text) + " "  # Add spaces for word boundary matching
        
        for keyword in self.HARD_DISCARD_KEYWORDS:
            # For short keywords, require word boundaries
            if len(keyword) <= 4:
                pattern = f" {keyword} "
                if pattern in text_lower:
                    return True, f"Hard discard: matched '{keyword}'"
            else:
                if keyword in text_lower:
                    # Check for exceptions (e.g., "weather" with infrastructure)
                    if keyword in ["weather forecast", "sunny", "cloudy"]:
                        infrastructure_words = ["pipeline", "refinery", "port", "power grid", "flood"]
                        if any(w in text_lower for w in infrastructure_words):
                            continue  # Don't discard if infrastructure-related
                    
                    return True, f"Hard discard: matched '{keyword}'"
        
        return False, ""
    
    def _check_novelty(self, headline: str, url: Optional[str]) -> float:
        """
        Check if this is a novel (non-duplicate) article.
        Returns novelty score 0-1.
        """
        headline_key = self._normalize_text(headline)[:80]
        
        # Exact URL match
        if url and url in self.seen_urls:
            return 0.0
        
        # Near-identical headline
        if headline_key in self.seen_headlines:
            return 0.0
        
        # Check for very similar headlines (fuzzy match)
        for seen in self.seen_headlines:
            if len(headline_key) > 20 and len(seen) > 20:
                # Simple overlap check
                words1 = set(headline_key.split())
                words2 = set(seen.split())
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > 0.8:
                    return 0.3  # Partial novelty
        
        return 1.0
    
    def _calculate_relevance(self, text: str) -> Tuple[float, List[str], NewsCategory]:
        """
        Calculate relevance score and identify category.
        Returns (score, matched_keywords, category).
        """
        text_lower = self._normalize_text(text)
        matched = []
        scores = {}
        
        # Check each category with base relevance scores
        # Higher base = higher priority category
        category_keywords = [
            (NewsCategory.GEOPOLITICAL, self.GEOPOLITICAL_KEYWORDS + self.GEOPOLITICAL_ENTITIES, 0.25),
            (NewsCategory.FINANCIAL_STRESS, self.FINANCIAL_STRESS_KEYWORDS, 0.25),
            (NewsCategory.SANCTIONS, self.SANCTIONS_KEYWORDS, 0.20),
            (NewsCategory.ENERGY_COMMODITY, self.ENERGY_KEYWORDS, 0.20),
            (NewsCategory.SHIPPING_LOGISTICS, self.SHIPPING_KEYWORDS + self.CHOKEPOINTS, 0.20),
            (NewsCategory.CENTRAL_BANK, self.CENTRAL_BANK_KEYWORDS + self.CENTRAL_BANK_ENTITIES, 0.20),
            (NewsCategory.MACRO_DATA, self.MACRO_KEYWORDS + self.SURPRISE_LANGUAGE, 0.15),
            (NewsCategory.ELECTIONS_POLICY, self.ELECTIONS_KEYWORDS, 0.15),
            (NewsCategory.CORPORATE_DEALS, self.CORPORATE_KEYWORDS, 0.10),
            (NewsCategory.BUSINESS_INNOVATION, self.INNOVATION_KEYWORDS, 0.10),
        ]
        
        for category, keywords, base_score in category_keywords:
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                # Base score + bonus per keyword (diminishing returns)
                keyword_bonus = min(0.4, len(matches) * 0.1)
                scores[category] = base_score + keyword_bonus
                matched.extend(matches)
        
        # Innovation requires vertical match
        if NewsCategory.BUSINESS_INNOVATION in scores:
            has_vertical = any(v in text_lower for v in self.INNOVATION_VERTICALS)
            if not has_vertical:
                scores[NewsCategory.BUSINESS_INNOVATION] *= 0.3
        
        # Boost for chokepoints (always high impact)
        if any(cp in text_lower for cp in self.CHOKEPOINTS):
            for cat in scores:
                scores[cat] += 0.15
        
        if not scores:
            return 0.0, [], NewsCategory.IRRELEVANT
        
        # Get primary category
        primary_category = max(scores, key=scores.get)
        total_score = min(1.0, scores[primary_category])
        
        return total_score, list(set(matched)), primary_category
    
    def _calculate_impact(self, text: str, category: NewsCategory) -> float:
        """Calculate market impact score."""
        text_lower = self._normalize_text(text)
        impact = 0.0
        
        # Base impact by category
        category_impact = {
            NewsCategory.GEOPOLITICAL: 0.4,
            NewsCategory.SANCTIONS: 0.35,
            NewsCategory.ENERGY_COMMODITY: 0.35,
            NewsCategory.SHIPPING_LOGISTICS: 0.3,
            NewsCategory.MACRO_DATA: 0.3,
            NewsCategory.CENTRAL_BANK: 0.4,
            NewsCategory.FINANCIAL_STRESS: 0.5,
            NewsCategory.ELECTIONS_POLICY: 0.25,
            NewsCategory.CORPORATE_DEALS: 0.2,
            NewsCategory.BUSINESS_INNOVATION: 0.15,
        }
        impact = category_impact.get(category, 0.0)
        
        # Boost for escalation verbs
        escalation_count = sum(1 for v in self.ESCALATION_VERBS if v in text_lower)
        impact += min(0.2, escalation_count * 0.05)
        
        # Boost for surprise/magnitude language
        surprise_count = sum(1 for s in self.SURPRISE_MAGNITUDE if s in text_lower)
        impact += min(0.2, surprise_count * 0.05)
        
        # Boost for numeric signals
        for pattern in self.numeric_patterns:
            if pattern.search(text_lower):
                impact += 0.05
        
        # Boost for "billion" amounts
        if "billion" in text_lower:
            impact += 0.1
        
        # Boost for chokepoints
        if any(cp in text_lower for cp in self.CHOKEPOINTS):
            impact += 0.15
        
        return min(1.0, impact)
    
    def _calculate_credibility(self, source: str) -> float:
        """Calculate source credibility score."""
        source_lower = source.lower()
        
        for tier1 in self.TIER_1_SOURCES:
            if tier1 in source_lower:
                return 1.0
        
        for tier2 in self.TIER_2_SOURCES:
            if tier2 in source_lower:
                return 0.7
        
        for tier3 in self.TIER_3_SOURCES:
            if tier3 in source_lower:
                return 0.3
        
        return 0.5  # Unknown source
    
    def _infer_direction(self, text: str, category: NewsCategory) -> Tuple[MarketDirection, float]:
        """Infer market direction from news content."""
        text_lower = self._normalize_text(text)
        
        risk_off_count = sum(1 for t in self.RISK_OFF_TRIGGERS if t in text_lower)
        risk_on_count = sum(1 for t in self.RISK_ON_TRIGGERS if t in text_lower)
        oil_up_count = sum(1 for t in self.OIL_UP_TRIGGERS if t in text_lower)
        oil_down_count = sum(1 for t in self.OIL_DOWN_TRIGGERS if t in text_lower)
        
        # Determine primary direction
        scores = {
            MarketDirection.RISK_OFF: risk_off_count * 0.2,
            MarketDirection.RISK_ON: risk_on_count * 0.2,
            MarketDirection.OIL_UP: oil_up_count * 0.15,
            MarketDirection.OIL_DOWN: oil_down_count * 0.15,
        }
        
        # Category-based defaults
        if category == NewsCategory.GEOPOLITICAL and risk_off_count == 0:
            scores[MarketDirection.RISK_OFF] += 0.1
        
        if category == NewsCategory.CENTRAL_BANK:
            if "hawkish" in text_lower or "rate hike" in text_lower:
                scores[MarketDirection.RATES_UP] = 0.5
            elif "dovish" in text_lower or "rate cut" in text_lower:
                scores[MarketDirection.RATES_DOWN] = 0.5
        
        if max(scores.values()) == 0:
            return MarketDirection.NEUTRAL, 0.0
        
        direction = max(scores, key=scores.get)
        confidence = min(1.0, scores[direction])
        
        return direction, confidence
    
    def _identify_affected_assets(self, text: str, category: NewsCategory) -> List[str]:
        """Identify potentially affected assets/sectors."""
        text_lower = self._normalize_text(text)
        assets = []
        
        # Energy
        if any(w in text_lower for w in ["oil", "crude", "opec", "pipeline", "refinery"]):
            assets.extend(["XLE", "OIL", "USO"])
        
        # Defense
        if any(w in text_lower for w in ["military", "defense", "missile", "weapons"]):
            assets.extend(["XAR", "ITA", "LMT", "RTX"])
        
        # Financials
        if any(w in text_lower for w in ["bank", "financial", "credit"]):
            assets.extend(["XLF", "KBE", "JPM"])
        
        # Tech
        if any(w in text_lower for w in ["semiconductor", "chip", "ai", "tech"]):
            assets.extend(["XLK", "SMH", "NVDA", "AMD"])
        
        # Safe havens
        if any(w in text_lower for w in ["crisis", "war", "escalation"]):
            assets.extend(["GLD", "TLT", "VIX"])
        
        return list(set(assets))
    
    def _identify_regions(self, text: str) -> List[str]:
        """Identify affected regions."""
        text_lower = self._normalize_text(text)
        regions = []
        
        region_keywords = {
            "middle_east": ["middle east", "israel", "iran", "saudi", "uae", "gulf", "gaza", "yemen", "iraq"],
            "asia": ["china", "japan", "korea", "taiwan", "asia", "pacific"],
            "europe": ["europe", "eu", "nato", "germany", "france", "uk", "britain"],
            "russia": ["russia", "ukraine", "moscow", "putin"],
            "americas": ["us", "united states", "america", "washington", "fed"],
        }
        
        for region, keywords in region_keywords.items():
            if any(kw in text_lower for kw in keywords):
                regions.append(region)
        
        return regions if regions else ["global"]
    
    def _generate_tags(self, matched_keywords: List[str], category: NewsCategory) -> List[str]:
        """Generate descriptive tags."""
        tags = [category.value]
        
        # Add specific tags based on keywords
        if any(kw in matched_keywords for kw in ["war", "invasion", "airstrike"]):
            tags.append("conflict")
        if any(kw in matched_keywords for kw in ["sanctions", "embargo"]):
            tags.append("sanctions")
        if any(kw in matched_keywords for kw in ["oil", "opec", "pipeline"]):
            tags.append("energy")
        if any(kw in matched_keywords for kw in ["rate", "fed", "ecb"]):
            tags.append("monetary_policy")
        if any(kw in matched_keywords for kw in ["billion", "merger", "acquisition"]):
            tags.append("deal")
        
        return list(set(tags))
    
    def _build_rationale(self, matched_keywords: List[str], category: NewsCategory,
                         relevance: float, impact: float) -> str:
        """Build explainable rationale for the decision."""
        parts = [
            f"Category: {category.value}",
            f"Matched keywords: {', '.join(matched_keywords[:5])}",
            f"Relevance: {relevance:.2f}",
            f"Impact: {impact:.2f}",
        ]
        return " | ".join(parts)
    
    def filter_article(self, headline: str, summary: str, source: str,
                       timestamp: datetime, url: Optional[str] = None) -> Optional[NewsEvent]:
        """
        Filter a single article. Returns NewsEvent if accepted, None if rejected.
        """
        self.stats["total_processed"] += 1
        full_text = f"{headline} {summary}"
        
        # Step 1: Hard discard check
        should_discard, reason = self._check_hard_discard(full_text)
        if should_discard:
            self.stats["rejected"] += 1
            self.stats["rejected_reasons"][reason] = self.stats["rejected_reasons"].get(reason, 0) + 1
            logging.debug(f"REJECTED: {headline[:50]}... | {reason}")
            return None
        
        # Step 2: Novelty check
        novelty = self._check_novelty(headline, url)
        if novelty == 0.0:
            self.stats["rejected"] += 1
            reason = "Duplicate"
            self.stats["rejected_reasons"][reason] = self.stats["rejected_reasons"].get(reason, 0) + 1
            logging.debug(f"REJECTED: {headline[:50]}... | Duplicate")
            return None
        
        # Step 3: Calculate relevance
        relevance, matched_keywords, category = self._calculate_relevance(full_text)
        
        if relevance < self.RELEVANCE_THRESHOLD:
            self.stats["rejected"] += 1
            reason = f"Low relevance ({relevance:.2f})"
            self.stats["rejected_reasons"]["Low relevance"] = self.stats["rejected_reasons"].get("Low relevance", 0) + 1
            logging.debug(f"REJECTED: {headline[:50]}... | {reason}")
            return None
        
        # Step 4: Calculate impact
        impact = self._calculate_impact(full_text, category)
        
        if impact < self.IMPACT_THRESHOLD:
            self.stats["rejected"] += 1
            reason = f"Low impact ({impact:.2f})"
            self.stats["rejected_reasons"]["Low impact"] = self.stats["rejected_reasons"].get("Low impact", 0) + 1
            logging.debug(f"REJECTED: {headline[:50]}... | {reason}")
            return None
        
        # Step 5: Calculate credibility
        credibility = self._calculate_credibility(source)
        
        # Step 6: Calculate final score
        final_score = (
            relevance * 0.35 +
            impact * 0.35 +
            credibility * 0.15 +
            novelty * 0.15
        )
        
        if final_score < self.FINAL_SCORE_THRESHOLD:
            self.stats["rejected"] += 1
            reason = f"Low final score ({final_score:.2f})"
            self.stats["rejected_reasons"]["Low final score"] = self.stats["rejected_reasons"].get("Low final score", 0) + 1
            logging.debug(f"REJECTED: {headline[:50]}... | {reason}")
            return None
        
        # Step 7: Infer direction
        direction, direction_confidence = self._infer_direction(full_text, category)
        
        # Step 8: Identify affected assets and regions
        affected_assets = self._identify_affected_assets(full_text, category)
        affected_regions = self._identify_regions(full_text)
        
        # Step 9: Generate tags and rationale
        tags = self._generate_tags(matched_keywords, category)
        rationale = self._build_rationale(matched_keywords, category, relevance, impact)
        
        # Step 10: Mark as seen (deduplication)
        self.seen_headlines.add(self._normalize_text(headline)[:80])
        if url:
            self.seen_urls.add(url)
        
        # Create event
        event = NewsEvent(
            event_id=self._generate_id(headline, source),
            timestamp=timestamp,
            headline=headline,
            summary=summary[:500],
            source=source,
            url=url,
            category=category,
            tags=tags,
            matched_keywords=matched_keywords[:10],
            relevance_score=relevance,
            impact_score=impact,
            credibility_score=credibility,
            novelty_score=novelty,
            final_score=final_score,
            direction=direction,
            direction_confidence=direction_confidence,
            affected_assets=affected_assets,
            affected_regions=affected_regions,
            rationale=rationale,
        )
        
        self.stats["accepted"] += 1
        logging.info(f"ACCEPTED: [{category.value}] {headline[:50]}... | score={final_score:.2f}")
        
        return event
    
    def filter_batch(self, articles: List[dict]) -> List[NewsEvent]:
        """
        Filter a batch of articles.
        
        Args:
            articles: List of dicts with keys: headline, summary, source, timestamp, url
            
        Returns:
            List of accepted NewsEvent objects
        """
        events = []
        for article in articles:
            event = self.filter_article(
                headline=article.get("headline", ""),
                summary=article.get("summary", ""),
                source=article.get("source", "unknown"),
                timestamp=article.get("timestamp", datetime.now(pytz.UTC)),
                url=article.get("url"),
            )
            if event:
                events.append(event)
        
        return events
    
    def get_stats(self) -> dict:
        """Get filtering statistics."""
        total = self.stats["total_processed"]
        return {
            "total_processed": total,
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "acceptance_rate": self.stats["accepted"] / total if total > 0 else 0,
            "rejection_reasons": self.stats["rejected_reasons"],
        }
    
    def save_events(self, events: List[NewsEvent], filename: Optional[str] = None):
        """Save accepted events to JSON file."""
        if filename is None:
            filename = f"news_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        data = {
            "generated_at": datetime.now(pytz.UTC).isoformat(),
            "stats": self.get_stats(),
            "events": [e.to_dict() for e in events],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Saved {len(events)} events to {filepath}")
        return filepath
    
    def reset_dedup_cache(self):
        """Reset deduplication cache (call at start of each day)."""
        self.seen_headlines.clear()
        self.seen_urls.clear()


# Singleton instance
_filter_instance: Optional[NewsRelevanceFilter] = None

def get_news_filter() -> NewsRelevanceFilter:
    """Get singleton instance of NewsRelevanceFilter."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = NewsRelevanceFilter()
    return _filter_instance
