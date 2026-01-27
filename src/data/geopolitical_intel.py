"""
Geopolitical Intelligence Layer

Monitors global events that impact markets:
- Military tensions and conflicts
- Regional instability
- Flight/travel disruptions (early warning signals)
- Diplomatic crises
- Sanctions and trade tensions
- Regional market reactions

Data Sources:
1. NewsAPI - Real-time global news
2. GDELT Project - Global event database (conflicts, protests)
3. RSS feeds - Reuters, BBC, Al Jazeera
4. Regional market data - Middle East, Asia indices
5. Flight data - Disruption signals

Now with ADVANCED RELEVANCE FILTERING:
- Rule-based market-moving detection
- Hard discard for irrelevant content
- Scoring: relevance, impact, credibility, novelty
- Direction inference for market impact
"""

import os
import json
import logging
import hashlib
import requests
import feedparser
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import re

# Import the advanced relevance filter
try:
    from src.data.news_relevance_filter import get_news_filter, NewsEvent, NewsCategory, MarketDirection
    HAS_RELEVANCE_FILTER = True
except ImportError:
    HAS_RELEVANCE_FILTER = False
    logging.warning("NewsRelevanceFilter not available, using basic filtering")

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class GeopoliticalEvent:
    """Represents a geopolitical event that may impact markets."""
    event_id: str
    timestamp: datetime
    headline: str
    summary: str
    source: str
    event_type: str  # military, diplomatic, economic, civil_unrest, natural_disaster
    severity: float  # 0-1 scale
    regions: List[str]  # affected regions
    keywords: List[str]
    market_impact_score: float  # estimated market impact 0-1
    url: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "event_type": self.event_type,
            "severity": self.severity,
            "regions": self.regions,
            "keywords": self.keywords,
            "market_impact_score": self.market_impact_score,
            "url": self.url,
        }


@dataclass
class GeopoliticalRiskAssessment:
    """Overall geopolitical risk assessment."""
    timestamp: datetime
    overall_risk_score: float  # 0-1
    risk_level: str  # low, moderate, elevated, high, critical
    active_events: List[GeopoliticalEvent]
    regional_risks: Dict[str, float]  # region -> risk score
    recommended_exposure_adjustment: float  # multiplier 0.0-1.0
    key_concerns: List[str]
    safe_haven_signal: bool  # should rotate to safe havens?
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level,
            "active_events_count": len(self.active_events),
            "active_events": [e.to_dict() for e in self.active_events[:10]],
            "regional_risks": self.regional_risks,
            "recommended_exposure_adjustment": self.recommended_exposure_adjustment,
            "key_concerns": self.key_concerns,
            "safe_haven_signal": self.safe_haven_signal,
        }


class GeopoliticalIntelligence:
    """
    Comprehensive geopolitical risk monitoring system.
    
    Monitors multiple data sources to detect events that could impact markets.
    """
    
    # Keywords that indicate high-impact geopolitical events
    # COMPREHENSIVE EXPANSION for ALL market-moving categories
    HIGH_IMPACT_KEYWORDS = {
        # ======================================================
        # GEOPOLITICAL RISK & CONFLICT
        # ======================================================
        "military": [
            "airstrike", "military", "jets", "troops", "deploy", "missile", 
            "strike", "attack", "invasion", "war", "conflict", "combat",
            "bombing", "airspace", "naval", "defense", "offensive",
            "retaliation", "mobilization", "ceasefire", "peace talks",
            "drone attack", "terrorism", "insurgency", "counteroffensive",
            "military escalation", "naval incident", "territorial",
        ],
        "diplomatic": [
            "sanctions", "embargo", "diplomatic", "treaty", "alliance",
            "ambassador", "expelled", "relations", "summit", "talks",
            "tariff", "trade war", "trade deal", "export controls",
            "blacklist", "asset freeze", "swift", "secondary sanctions",
            "trade embargo", "import ban", "export ban", "retaliation",
        ],
        "civil_unrest": [
            "protest", "riot", "uprising", "revolution", "coup",
            "martial law", "curfew", "emergency", "unrest",
            "election", "elections", "referendum", "impeachment", "government",
            "regime change", "political crisis", "constitutional crisis",
        ],
        
        # ======================================================
        # CENTRAL BANKS & MONETARY POLICY (CRITICAL)
        # ======================================================
        "central_bank": [
            # Core Institutions
            "federal reserve", "fed", "fomc", "ecb", "european central bank",
            "bank of england", "boe", "bank of japan", "boj", "pboc",
            "snb", "rba", "rbnz", "boc", "bank of canada",
            "bis", "imf", "international monetary fund",
            # Key Officials
            "jerome powell", "powell", "lagarde", "christine lagarde",
            "bailey", "andrew bailey", "ueda", "kazuo ueda", "fed chair",
            # Policy Actions
            "interest rate hike", "interest rate cut", "rate hike", "rate cut",
            "rate hold", "rate decision", "policy tightening", "policy easing",
            "monetary tightening", "monetary easing", "emergency meeting",
            "inter-meeting decision", "forward guidance", "dot plot",
            "policy statement", "balance sheet policy", "minutes",
            # Liquidity Tools
            "quantitative easing", "qe", "quantitative tightening", "qt",
            "balance sheet runoff", "repo operations", "reverse repo",
            "standing facilities", "discount window", "liquidity injection",
            "liquidity withdrawal", "emergency facility", "yield curve control",
            "currency intervention",
            # Communication Tone (Sentiment)
            "hawkish", "dovish", "neutral stance", "restrictive policy",
            "accommodative policy", "data-dependent", "higher for longer",
            "policy pivot", "tightening bias", "easing bias", "monetary policy",
            "central bank", "rate decision", "fomc meeting",
        ],
        
        # ======================================================
        # MACROECONOMIC INDICATORS
        # ======================================================
        "economic": [
            # Inflation
            "cpi", "pce", "core cpi", "core pce", "inflation", "core inflation",
            "inflation expectations", "disinflation", "deflation",
            "price pressures", "sticky inflation", "inflation surprise",
            # Labor Market
            "unemployment", "unemployment rate", "jobless claims", "initial claims",
            "continuing claims", "nonfarm payrolls", "non-farm payrolls", "nfp",
            "employment growth", "wage growth", "average hourly earnings",
            "labor participation", "layoffs", "hiring freeze",
            # Growth & Demand
            "gdp", "gdp growth", "recession", "economic slowdown", "contraction",
            "economic expansion", "pmi", "ism", "retail sales",
            "consumer spending", "consumer confidence", "industrial production",
            # Housing & Credit
            "housing starts", "building permits", "home sales", "mortgage rates",
            "credit conditions", "lending standards", "delinquencies", "defaults",
            # Legacy
            "tariff", "trade war", "currency", "devaluation", "default",
            "bailout", "crisis", "collapse", "shutdown",
        ],
        
        # ======================================================
        # FINANCIAL MARKETS & STRESS
        # ======================================================
        "financial_stress": [
            # Market Structure
            "equity markets", "bond markets", "yield curve", "curve inversion",
            "term premium", "real yields", "credit spreads", "swap spreads",
            "funding markets",
            # Volatility & Risk
            "market volatility", "vix", "implied volatility", "volatility spike",
            "risk-off", "risk-on", "flight to safety", "market selloff",
            "market rally", "liquidity crunch",
            # Banking & Credit Stress (SPECIFIC - avoid generic words like "rescue")
            "bank run", "bank failure", "banking crisis", "deposit outflows",
            "liquidity stress", "capital adequacy", "stress test",
            "loan losses", "defaults", "debt restructuring", "credit downgrade",
            "bankruptcy filing", "debt distress", "bailout package", "bank rescue",
            "capital shortfall", "liquidity crisis", "sovereign debt crisis",
            "imf support", "imf bailout", "distressed debt", "contagion risk",
            "funding stress", "margin calls", "financial crisis", "junk bonds",
            "systemic risk", "too big to fail", "emergency lending",
        ],
        
        # ======================================================
        # SHIPPING, TRADE & GLOBAL SUPPLY CHAINS (VERY IMPORTANT)
        # ======================================================
        "shipping": [
            # Shipping & Logistics
            "shipping", "shipping disruption", "shipping delays",
            "port congestion", "container shortage", "vessel shortage",
            "freight rates", "charter rates", "demurrage",
            "logistics bottlenecks", "rerouting", "reroute", "maritime risk",
            "freight", "container", "port", "tanker", "cargo",
            "insurance premium", "shipping rates", "chokepoint",
            # Strategic Chokepoints
            "suez canal", "panama canal", "strait of hormuz",
            "bab el-mandeb", "strait of malacca", "bosphorus",
            "red sea", "south china sea", "black sea",
            # Trade & Transport
            "global trade", "trade flows", "trade deficit",
            "export restrictions", "customs delays", "trade sanctions",
            "reshoring", "nearshoring", "supply chain", "logistics",
            "blockade",
        ],
        "infrastructure": [
            "flight cancelled", "flights cancelled", "airspace closed",
            "port closed", "shipping disrupted", "pipeline", "embargo",
            "supply chain disruption", "logistics crisis",
        ],
        
        # ======================================================
        # ENERGY & COMMODITIES
        # ======================================================
        "energy": [
            # Energy Markets
            "opec", "opec+", "oil", "crude oil", "brent", "wti",
            "natural gas", "lng", "energy supply", "energy shortages",
            "energy sanctions", "refinery", "refinery outages",
            "pipeline disruption", "oil prices", "energy crisis",
            "fuel", "gasoline", "petroleum",
            # OPEC & Supply
            "production cut", "production cuts", "output cut",
            "production increase", "spare capacity", "inventory draw",
            "inventory build", "strategic petroleum reserve", "spr",
            "drilling activity", "rig counts",
            "tanker", "force majeure", "strategic reserves",
            # Industrial & Ag Commodities
            "gold", "silver", "copper", "aluminum", "steel",
            "rare earth", "rare earths", "commodity", "commodities",
            "mining", "precious metals", "safe haven",
            "grain exports", "wheat", "corn", "soybeans", "fertilizer",
        ],
        
        # ======================================================
        # FISCAL & REGULATORY POLICY
        # ======================================================
        "fiscal_regulatory": [
            # Fiscal Policy
            "government spending", "fiscal stimulus", "austerity",
            "budget deficit", "public debt", "debt ceiling", "treasury issuance",
            "government shutdown",
            # Regulation
            "regulatory crackdown", "financial regulation", "capital requirements",
            "banking reform", "market intervention", "price controls",
            "subsidies", "tax changes", "tax reform",
            "antitrust", "doj", "ftc", "sec",
        ],
        
        # ======================================================
        # FUTURES, RATES & DERIVATIVES
        # ======================================================
        "rates_derivatives": [
            # Futures & Term Structure
            "futures curve", "contango", "backwardation", "roll yield",
            "open interest", "contract expiry", "front month", "calendar spread",
            # Rates & Fixed Income
            "treasury yields", "bond auction", "yield spike", "duration risk",
            "curve steepening", "curve flattening", "rate volatility", "swap rates",
        ],
    }
    
    # SEVERITY MODIFIERS - Words that boost impact score
    SEVERITY_MODIFIERS = [
        "unexpectedly", "sharply", "emergency", "historic", "unprecedented",
        "sudden", "significant", "severe", "escalates", "intensifies",
        "collapses", "surges", "spikes", "plunges", "widens", "narrows",
        "crisis", "rare", "record", "shock",
    ]
    
    # HARD DISCARD - Events that should NEVER be classified as market-moving
    # These are local news, accidents, and irrelevant events
    HARD_DISCARD_KEYWORDS = [
        # Local disasters (not market-moving unless massive scale)
        "ferry sinks", "ferry sank", "ferry capsizes", "ferry accident", "boat sinks",
        "bus crash", "train derailment", "plane crash", "car accident",
        "building collapse", "house fire", "apartment fire", "factory fire",
        "people dead", "people killed", "bodies found", "bodies recovered",
        "missing persons", "search and rescue", "rescue operation", "rescue workers",
        "rescuers save", "survivors found", "survivors rescued", "death toll",
        "people onboard", "passengers aboard", "passengers rescued",
        # Crime/Local
        "arrested", "murder", "robbery", "burglary", "theft", "assault",
        "shooting", "stabbing", "drug bust", "gang violence",
        "local police", "city council", "town hall",
        # Sports/Entertainment
        "world cup", "olympics", "football", "soccer", "basketball", "baseball",
        "tennis", "golf", "celebrity", "kardashian", "hollywood", "movie",
        "concert", "grammy", "oscar", "emmy", "red carpet",
        # Lifestyle
        "recipe", "cooking", "restaurant", "travel tips", "fashion",
        "diet", "workout", "yoga", "meditation", "home decor",
        # Social media / Tech regulation (unless antitrust)
        "social media ban", "under-15", "age verification", "content moderation",
        "parental controls", "screen time", "online safety",
        # Weather (unless infrastructure impact)
        "weather forecast", "sunny", "cloudy", "chance of rain",
        # Obituaries
        "obituary", "funeral", "passed away", "dies at",
    ]
    
    # MINIMUM SEVERITY THRESHOLD - Events below this are discarded
    MIN_SEVERITY_THRESHOLD = 0.25
    
    # Regions and their market indices
    REGIONAL_INDICES = {
        "middle_east": ["TASI.SR", "ADI.AD", "DFMGI.DU", "EGX30.CA"],  # Saudi, UAE, Egypt
        "asia": ["^N225", "^HSI", "000001.SS", "^KS11", "^TWII"],  # Japan, HK, China, Korea, Taiwan
        "europe": ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E"],  # UK, Germany, France, Euro
        "emerging": ["EEM", "^BVSP", "^BSESN", "^JKSE"],  # EM ETF, Brazil, India, Indonesia
    }
    
    # RSS feeds for geopolitical news
    # General world news
    RSS_FEEDS = {
        "reuters_world": "https://feeds.reuters.com/Reuters/worldNews",
        "bbc_world": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "ap_world": "https://rsshub.app/apnews/topics/world-news",
        "guardian_world": "https://www.theguardian.com/world/rss",
    }
    
    # TARGETED RSS FEEDS - Topic-specific for higher relevance
    TARGETED_RSS_FEEDS = {
        # Middle East specific
        "aljazeera_middleeast": "https://www.aljazeera.com/xml/rss/all.xml",
        "bbc_middleeast": "http://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
        
        # Asia specific  
        "bbc_asia": "http://feeds.bbci.co.uk/news/world/asia/rss.xml",
        
        # US Politics & Foreign Policy
        "bbc_uspolitics": "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
        
        # Business/Economy (for trade wars, sanctions)
        "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
        "bbc_business": "http://feeds.bbci.co.uk/news/business/rss.xml",
        
        # Defense & Military (specialized)
        "defense_news": "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
        
        # Europe (for NATO, EU decisions)
        "bbc_europe": "http://feeds.bbci.co.uk/news/world/europe/rss.xml",
        
        # Russia/Ukraine
        "guardian_russia": "https://www.theguardian.com/world/russia/rss",
        
        # CENTRAL BANKS & MONETARY POLICY
        "fed_news": "https://www.federalreserve.gov/feeds/press_all.xml",
        "ecb_press": "https://www.ecb.europa.eu/rss/press.html",
        "ft_central_banks": "https://www.ft.com/central-banks?format=rss",
        
        # ENERGY & COMMODITIES
        "reuters_energy": "https://feeds.reuters.com/reuters/energyNews",
        "oilprice_news": "https://oilprice.com/rss/main",
        
        # FINANCIAL MARKETS & ECONOMY
        "reuters_markets": "https://feeds.reuters.com/reuters/companyNews",
        "ft_markets": "https://www.ft.com/markets?format=rss",
        "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "bloomberg_markets": "https://feeds.bloomberg.com/markets/news.rss",
        
        # TRADE & TARIFFS
        "reuters_trade": "https://feeds.reuters.com/reuters/USTradingDesk",
        
        # SHIPPING & LOGISTICS
        "splash247": "https://splash247.com/feed/",
        "gcaptain": "https://gcaptain.com/feed/",
        "lloyds_list": "https://lloydslist.maritimeintelligence.informa.com/rss",
    }
    
    def __init__(self, cache_dir: str = "outputs/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "geopolitical_cache.json"
        self.filtered_cache_file = self.cache_dir / "filtered_events_cache.json"  # PERSISTENT FILTERED EVENTS
        
        # API keys
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        
        # Event cache
        self.events_cache: List[GeopoliticalEvent] = []
        self.last_assessment: Optional[GeopoliticalRiskAssessment] = None
        self.last_update: Optional[datetime] = None
        self.last_filtered_update: Optional[datetime] = None  # Track filtered events update time
        
        # Advanced relevance filter (rule-based, deterministic)
        self.relevance_filter = get_news_filter() if HAS_RELEVANCE_FILTER else None
        self.filtered_events: List[NewsEvent] = []  # High-quality filtered events
        
        # Load caches (both raw and filtered)
        self._load_cache()
        self._load_filtered_cache()  # LOAD PERSISTENT FILTERED EVENTS
        
        logging.info(f"GeopoliticalIntelligence: Loaded {len(self.events_cache)} raw events, "
                    f"{len(self.filtered_events)} filtered events from cache")
    
    def _load_cache(self):
        """Load cached events from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Only load events from last 48 hours
                cutoff = datetime.now(pytz.UTC) - timedelta(hours=48)
                
                self.events_cache = []
                for event_data in data.get("events", []):
                    try:
                        ts = datetime.fromisoformat(event_data["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=pytz.UTC)
                        if ts > cutoff:
                            self.events_cache.append(GeopoliticalEvent(
                                event_id=event_data["event_id"],
                                timestamp=ts,
                                headline=event_data["headline"],
                                summary=event_data.get("summary", ""),
                                source=event_data["source"],
                                event_type=event_data["event_type"],
                                severity=event_data["severity"],
                                regions=event_data["regions"],
                                keywords=event_data.get("keywords", []),
                                market_impact_score=event_data["market_impact_score"],
                                url=event_data.get("url"),
                            ))
                    except Exception:
                        continue
                
                if data.get("last_update"):
                    self.last_update = datetime.fromisoformat(data["last_update"])
                    
        except Exception as e:
            logging.warning(f"Could not load geopolitical cache: {e}")
    
    def _save_cache(self):
        """Save events to disk cache."""
        try:
            data = {
                "events": [e.to_dict() for e in self.events_cache],
                "last_update": datetime.now(pytz.UTC).isoformat(),
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save geopolitical cache: {e}")
    
    def _load_filtered_cache(self):
        """
        Load high-quality filtered events from persistent cache.
        This ensures filtered news survives page refreshes and server restarts.
        """
        try:
            if not self.filtered_cache_file.exists():
                logging.info("No filtered events cache file found")
                return
                
            with open(self.filtered_cache_file, 'r') as f:
                data = json.load(f)
            
            # Only load events from last 48 hours
            cutoff = datetime.now(pytz.UTC) - timedelta(hours=48)
            
            # Import NewsEvent and enums at module level is safer
            from src.data.news_relevance_filter import NewsEvent, NewsCategory, MarketDirection
            
            # Build lookup maps for enums (case-insensitive)
            category_map = {c.name.lower(): c for c in NewsCategory}
            category_map.update({c.value.lower(): c for c in NewsCategory})
            direction_map = {d.name.lower(): d for d in MarketDirection}
            direction_map.update({d.value.lower(): d for d in MarketDirection})
            
            self.filtered_events = []
            loaded_count = 0
            skipped_count = 0
            
            for event_data in data.get("events", []):
                try:
                    # Parse timestamp
                    ts_str = event_data.get("timestamp")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=pytz.UTC)
                    else:
                        ts = datetime.now(pytz.UTC)
                    
                    # Skip old events
                    if ts < cutoff:
                        skipped_count += 1
                        continue
                    
                    # Parse category (case-insensitive)
                    cat_str = str(event_data.get("category", "irrelevant")).lower()
                    category = category_map.get(cat_str, NewsCategory.IRRELEVANT)
                    
                    # Parse direction (case-insensitive)
                    dir_str = str(event_data.get("direction", "neutral")).lower()
                    direction = direction_map.get(dir_str, MarketDirection.NEUTRAL)
                    
                    event = NewsEvent(
                        event_id=event_data.get("event_id", ""),
                        timestamp=ts,
                        headline=event_data.get("headline", ""),
                        summary=event_data.get("summary", ""),
                        source=event_data.get("source", ""),
                        url=event_data.get("url"),
                        category=category,
                        tags=event_data.get("tags", []),
                        matched_keywords=event_data.get("matched_keywords", []),
                        relevance_score=float(event_data.get("relevance_score", 0.5)),
                        impact_score=float(event_data.get("impact_score", 0.5)),
                        credibility_score=float(event_data.get("credibility_score", 0.5)),
                        novelty_score=float(event_data.get("novelty_score", 0.5)),
                        final_score=float(event_data.get("final_score", 0.5)),
                        direction=direction,
                        direction_confidence=float(event_data.get("direction_confidence", 0.5)),
                        affected_assets=event_data.get("affected_assets", []),
                        affected_regions=event_data.get("affected_regions", []),
                        rationale=event_data.get("rationale", ""),  # Include rationale field
                    )
                    self.filtered_events.append(event)
                    loaded_count += 1
                    
                except Exception as e:
                    logging.debug(f"Could not restore filtered event: {e}")
                    continue
            
            if data.get("last_update"):
                try:
                    self.last_filtered_update = datetime.fromisoformat(
                        data["last_update"].replace('Z', '+00:00')
                    )
                except:
                    self.last_filtered_update = datetime.now(pytz.UTC)
            
            logging.info(f"Loaded {loaded_count} filtered events from cache "
                        f"(skipped {skipped_count} old events)")
                    
        except Exception as e:
            logging.warning(f"Could not load filtered events cache: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_filtered_cache(self):
        """
        Save high-quality filtered events to persistent cache.
        Called after every update to ensure no data loss.
        """
        try:
            events_data = []
            for event in self.filtered_events:
                try:
                    # Convert NewsEvent to dict
                    event_dict = event.to_dict() if hasattr(event, 'to_dict') else {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp),
                        "headline": event.headline,
                        "summary": event.summary,
                        "source": event.source,
                        "url": event.url,
                        "category": event.category.name if hasattr(event.category, 'name') else str(event.category),
                        "tags": event.tags,
                        "matched_keywords": event.matched_keywords,
                        "relevance_score": event.relevance_score,
                        "impact_score": event.impact_score,
                        "credibility_score": event.credibility_score,
                        "novelty_score": event.novelty_score,
                        "final_score": event.final_score,
                        "direction": event.direction.name if hasattr(event.direction, 'name') else str(event.direction),
                        "direction_confidence": event.direction_confidence,
                        "affected_assets": event.affected_assets,
                        "affected_regions": event.affected_regions,
                    }
                    events_data.append(event_dict)
                except Exception as e:
                    logging.debug(f"Could not serialize filtered event: {e}")
                    continue
            
            data = {
                "events": events_data,
                "last_update": datetime.now(pytz.UTC).isoformat(),
                "total_count": len(events_data),
            }
            
            with open(self.filtered_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"Saved {len(events_data)} filtered events to cache")
            
        except Exception as e:
            logging.warning(f"Could not save filtered events cache: {e}")
    
    def get_cached_filtered_events_age(self) -> Optional[float]:
        """Get the age of cached filtered events in minutes."""
        if self.last_filtered_update:
            age = datetime.now(pytz.UTC) - self.last_filtered_update
            return age.total_seconds() / 60
        return None
    
    def _generate_event_id(self, headline: str, source: str) -> str:
        """Generate unique event ID from headline and source."""
        content = f"{headline[:100]}_{source}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _classify_event(self, text: str) -> Tuple[str, float, List[str]]:
        """
        Classify event type and severity from text.
        Returns: (event_type, severity, matched_keywords)
        
        IMPORTANT: Returns severity=0 for hard-discard events
        """
        text_lower = " " + text.lower() + " "  # Add spaces for word boundary matching
        
        # STEP 1: Check hard discard FIRST - reject irrelevant content immediately
        for discard_kw in self.HARD_DISCARD_KEYWORDS:
            if discard_kw in text_lower:
                # This is noise - return immediately with 0 severity
                return "irrelevant", 0.0, []
        
        matched_keywords = []
        event_type = "general"
        max_matches = 0
        all_matches = []
        
        # Short keywords that need word boundaries to avoid false positives
        # e.g., "war" should not match "warns", "ware", etc.
        short_keywords = ["war", "fed", "ecb", "boj", "qe", "qt", "oil", "lng", "gdp", "cpi", "ppi", "pmi", "ism"]
        
        # Check all categories
        for etype, keywords in self.HIGH_IMPACT_KEYWORDS.items():
            matches = []
            for kw in keywords:
                if kw in short_keywords:
                    # Require word boundaries for short keywords
                    if f" {kw} " in text_lower or f" {kw}," in text_lower or f" {kw}." in text_lower:
                        matches.append(kw)
                elif kw in text_lower:
                    matches.append(kw)
            all_matches.extend(matches)
            if len(matches) > max_matches:
                max_matches = len(matches)
                event_type = etype
                matched_keywords = matches
        
        # Calculate severity based on keyword matches and SEVERITY_MODIFIERS
        # Use class-level modifiers for consistency
        severity_count = sum(1 for w in self.SEVERITY_MODIFIERS if w in text_lower)
        
        # Also check for urgency words
        urgency_words = ["breaking", "urgent", "imminent", "immediate", "critical", "major", "massive"]
        urgency_count = sum(1 for w in urgency_words if w in text_lower)
        
        # Base severity now starts higher (0.3) so market-moving events pass
        # Each keyword match adds to severity
        base_severity = 0.3 if max_matches > 0 else 0.1
        keyword_bonus = min(0.4, max_matches * 0.12)  # Up to 0.4 from keywords
        severity_bonus = min(0.25, severity_count * 0.08)  # Up to 0.25 from severity modifiers
        urgency_bonus = min(0.15, urgency_count * 0.05)  # Up to 0.15 from urgency
        
        # CRITICAL event types get base bonus (central_bank, financial_stress)
        critical_types = ["central_bank", "financial_stress"]
        if event_type in critical_types:
            base_severity += 0.15
        
        # VERY IMPORTANT event types get base bonus
        high_impact_types = ["military", "energy", "shipping", "rates_derivatives"]
        if event_type in high_impact_types:
            base_severity += 0.1
        
        severity = min(1.0, base_severity + keyword_bonus + severity_bonus + urgency_bonus)
        
        return event_type, severity, matched_keywords
    
    def _identify_regions(self, text: str) -> List[str]:
        """Identify affected regions from text."""
        text_lower = text.lower()
        regions = []
        
        region_keywords = {
            "middle_east": ["middle east", "saudi", "uae", "emirates", "bahrain", 
                           "qatar", "iran", "iraq", "israel", "gaza", "lebanon",
                           "syria", "jordan", "kuwait", "oman", "yemen", "gulf"],
            "asia": ["china", "japan", "korea", "taiwan", "hong kong", "india",
                    "indonesia", "singapore", "vietnam", "thailand", "asia",
                    "pacific", "philippines", "malaysia"],
            "europe": ["europe", "european", "uk", "britain", "germany", "france",
                      "italy", "spain", "nato", "eu", "brussels"],
            "americas": ["us", "united states", "america", "canada", "mexico",
                        "brazil", "latin america"],
            "russia": ["russia", "russian", "moscow", "ukraine", "putin"],
            "africa": ["africa", "african", "egypt", "south africa", "nigeria"],
        }
        
        for region, keywords in region_keywords.items():
            if any(kw in text_lower for kw in keywords):
                regions.append(region)
        
        return regions if regions else ["global"]
    
    def _calculate_market_impact(self, event_type: str, severity: float, 
                                  regions: List[str]) -> float:
        """Calculate estimated market impact score."""
        # Base impact by event type
        type_impact = {
            "military": 0.9,
            "diplomatic": 0.6,
            "economic": 0.7,
            "civil_unrest": 0.5,
            "infrastructure": 0.7,
            "general": 0.3,
            # NEW event types
            "central_bank": 0.85,      # Fed/ECB decisions are highly impactful
            "energy": 0.8,             # Oil/gas prices affect everything
            "financial_stress": 0.9,   # Banking crises are critical
            "shipping": 0.7,           # Supply chain disruptions
        }
        
        base_impact = type_impact.get(event_type, 0.3)
        
        # Adjust by regions (more regions = more global impact)
        region_multiplier = min(1.5, 1.0 + (len(regions) - 1) * 0.1)
        
        # High-impact regions for US markets
        high_impact_regions = ["middle_east", "asia", "russia", "americas"]
        if any(r in regions for r in high_impact_regions):
            region_multiplier *= 1.2
        
        impact = base_impact * severity * region_multiplier
        return min(1.0, impact)
    
    def fetch_rss_news(self, hours_back: int = 24) -> List[GeopoliticalEvent]:
        """Fetch news from RSS feeds."""
        events = []
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
        
        def fetch_feed(name: str, url: str) -> List[GeopoliticalEvent]:
            feed_events = []
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:  # Limit per feed
                    try:
                        # Parse timestamp
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            ts = datetime(*entry.published_parsed[:6], tzinfo=pytz.UTC)
                        else:
                            ts = datetime.now(pytz.UTC)
                        
                        if ts < cutoff:
                            continue
                        
                        headline = entry.title
                        summary = entry.get('summary', entry.get('description', ''))[:500]
                        
                        # Classify event
                        full_text = f"{headline} {summary}"
                        event_type, severity, keywords = self._classify_event(full_text)
                        
                        # Skip low-severity or irrelevant events
                        # Uses class-level threshold for consistency
                        if severity < self.MIN_SEVERITY_THRESHOLD or event_type == "irrelevant":
                            continue
                        
                        regions = self._identify_regions(full_text)
                        market_impact = self._calculate_market_impact(event_type, severity, regions)
                        
                        event = GeopoliticalEvent(
                            event_id=self._generate_event_id(headline, name),
                            timestamp=ts,
                            headline=headline,
                            summary=summary,
                            source=name,
                            event_type=event_type,
                            severity=severity,
                            regions=regions,
                            keywords=keywords,
                            market_impact_score=market_impact,
                            url=entry.get('link'),
                        )
                        feed_events.append(event)
                        
                    except Exception as e:
                        continue
                        
            except Exception as e:
                logging.warning(f"Failed to fetch RSS feed {name}: {e}")
            
            return feed_events
        
        # Combine general and targeted feeds
        all_feeds = {**self.RSS_FEEDS, **self.TARGETED_RSS_FEEDS}
        
        # Fetch all feeds in parallel (increased workers for more feeds)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fetch_feed, name, url): name 
                for name, url in all_feeds.items()
            }
            
            for future in as_completed(futures, timeout=45):
                try:
                    feed_events = future.result()
                    events.extend(feed_events)
                except Exception:
                    continue
        
        return events
    
    def fetch_newsapi(self, hours_back: int = 24) -> List[GeopoliticalEvent]:
        """Fetch news from NewsAPI (if API key available)."""
        if not self.newsapi_key:
            return []
        
        events = []
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
        
        # Queries for geopolitical events
        queries = [
            "military strike OR airstrike OR troops deploy",
            "sanctions OR embargo OR diplomatic crisis",
            "flights cancelled conflict OR airspace closed",
            "middle east tension OR gulf crisis",
            "trade war OR tariffs escalation",
        ]
        
        try:
            for query in queries[:3]:  # Limit API calls
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "apiKey": self.newsapi_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "from": cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get("articles", []):
                        try:
                            ts_str = article.get("publishedAt", "")
                            if ts_str:
                                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            else:
                                continue
                            
                            headline = article.get("title", "")
                            summary = article.get("description", "")[:500]
                            source = article.get("source", {}).get("name", "newsapi")
                            
                            full_text = f"{headline} {summary}"
                            event_type, severity, keywords = self._classify_event(full_text)
                            
                            if severity < 0.3:
                                continue
                            
                            regions = self._identify_regions(full_text)
                            market_impact = self._calculate_market_impact(event_type, severity, regions)
                            
                            event = GeopoliticalEvent(
                                event_id=self._generate_event_id(headline, source),
                                timestamp=ts,
                                headline=headline,
                                summary=summary,
                                source=source,
                                event_type=event_type,
                                severity=severity,
                                regions=regions,
                                keywords=keywords,
                                market_impact_score=market_impact,
                                url=article.get("url"),
                            )
                            events.append(event)
                            
                        except Exception:
                            continue
                            
        except Exception as e:
            logging.warning(f"NewsAPI fetch failed: {e}")
        
        return events
    
    def fetch_regional_market_data(self) -> Dict[str, Dict]:
        """
        Fetch regional market data to detect panic selling.
        Returns dict of region -> {change_pct, is_panic}
        """
        if not yf:
            return {}
        
        regional_data = {}
        
        for region, indices in self.REGIONAL_INDICES.items():
            try:
                # Get data for first available index in region
                for index in indices[:2]:  # Try first 2
                    try:
                        ticker = yf.Ticker(index)
                        hist = ticker.history(period="5d")
                        
                        if len(hist) >= 2:
                            current = hist['Close'].iloc[-1]
                            prev = hist['Close'].iloc[-2]
                            change_pct = ((current / prev) - 1) * 100
                            
                            # Detect panic (>2% single-day drop)
                            is_panic = change_pct < -2.0
                            
                            regional_data[region] = {
                                "index": index,
                                "change_pct": round(change_pct, 2),
                                "is_panic": is_panic,
                                "current": round(current, 2),
                            }
                            break
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                logging.warning(f"Could not get {region} market data: {e}")
        
        return regional_data
    
    def update_events(self, hours_back: int = 24) -> int:
        """
        Fetch new events from all sources and apply advanced relevance filtering.
        Returns number of new events added.
        """
        all_events = []
        raw_articles = []  # For relevance filter
        
        # Fetch from multiple sources
        logging.info("Fetching geopolitical events from RSS feeds...")
        rss_events = self.fetch_rss_news(hours_back)
        all_events.extend(rss_events)
        
        # Also collect raw articles for advanced filtering
        for event in rss_events:
            raw_articles.append({
                "headline": event.headline,
                "summary": event.summary,
                "source": event.source,
                "timestamp": event.timestamp,
                "url": event.url,
            })
        
        logging.info("Fetching geopolitical events from NewsAPI...")
        newsapi_events = self.fetch_newsapi(hours_back)
        all_events.extend(newsapi_events)
        
        for event in newsapi_events:
            raw_articles.append({
                "headline": event.headline,
                "summary": event.summary,
                "source": event.source,
                "timestamp": event.timestamp,
                "url": event.url,
            })
        
        # ============================================================
        # ADVANCED RELEVANCE FILTERING (Rule-based, deterministic)
        # ============================================================
        if self.relevance_filter and raw_articles:
            logging.info(f"Applying advanced relevance filter to {len(raw_articles)} articles...")
            
            # Filter articles
            filtered = self.relevance_filter.filter_batch(raw_articles)
            
            # Merge new filtered events with existing cached ones (deduplicate by event_id)
            existing_ids = {e.event_id for e in self.filtered_events}
            new_filtered = [e for e in filtered if e.event_id not in existing_ids]
            self.filtered_events = new_filtered + self.filtered_events
            
            # Keep only last 48 hours worth and limit to 500 events max
            cutoff = datetime.now(pytz.UTC) - timedelta(hours=48)
            self.filtered_events = [
                e for e in self.filtered_events 
                if e.timestamp and (
                    e.timestamp > cutoff if hasattr(e.timestamp, '__gt__') else True
                )
            ][:500]
            
            # Log filter stats
            stats = self.relevance_filter.get_stats()
            logging.info(f"Filter results: {stats['accepted']} accepted, "
                        f"{stats['rejected']} rejected "
                        f"({stats['acceptance_rate']*100:.1f}% acceptance rate)")
            
            # Save filtered events to BOTH locations (original timestamped + persistent cache)
            if filtered:
                self.relevance_filter.save_events(filtered)
            
            # CRITICAL: Save to persistent cache for survival across restarts
            self.last_filtered_update = datetime.now(pytz.UTC)
            self._save_filtered_cache()
            logging.info(f"Persistent filtered cache: {len(self.filtered_events)} total events")
        
        # Deduplicate by event_id
        existing_ids = {e.event_id for e in self.events_cache}
        new_events = [e for e in all_events if e.event_id not in existing_ids]
        
        # Add new events
        self.events_cache.extend(new_events)
        
        # Sort by timestamp (newest first)
        self.events_cache.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Keep only last 48 hours
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=48)
        self.events_cache = [e for e in self.events_cache 
                            if e.timestamp.replace(tzinfo=pytz.UTC) > cutoff]
        
        # Limit cache size
        self.events_cache = self.events_cache[:200]
        
        self.last_update = datetime.now(pytz.UTC)
        self._save_cache()
        
        logging.info(f"Geopolitical: Found {len(new_events)} new events, "
                    f"{len(self.events_cache)} total cached, "
                    f"{len(self.filtered_events)} high-quality filtered")
        
        return len(new_events)
    
    def get_filtered_events(self, auto_refresh_if_empty: bool = True, max_age_minutes: int = 60) -> List:
        """
        Get high-quality filtered events (from advanced filter).
        
        Args:
            auto_refresh_if_empty: If True, triggers refresh when no events available
            max_age_minutes: If events are older than this, trigger refresh
            
        Returns:
            List of NewsEvent objects
        """
        # Check if we need to refresh (empty or stale)
        should_refresh = False
        
        if not self.filtered_events and auto_refresh_if_empty:
            logging.info("No filtered events in cache, triggering refresh...")
            should_refresh = True
        elif self.last_filtered_update:
            age_minutes = self.get_cached_filtered_events_age()
            if age_minutes and age_minutes > max_age_minutes:
                logging.info(f"Filtered events are {age_minutes:.1f} min old (max: {max_age_minutes}), refreshing...")
                should_refresh = True
        
        if should_refresh:
            try:
                self.update_events(hours_back=24)
            except Exception as e:
                logging.warning(f"Auto-refresh failed: {e}")
        
        return self.filtered_events
    
    def get_filtered_events_summary(self) -> dict:
        """Get a summary of filtered events for UI display."""
        events = self.filtered_events
        
        # Group by category
        by_category = {}
        for event in events:
            cat = event.category.name if hasattr(event.category, 'name') else str(event.category)
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(event)
        
        return {
            "total_events": len(events),
            "categories": {cat: len(evts) for cat, evts in by_category.items()},
            "last_update": self.last_filtered_update.isoformat() if self.last_filtered_update else None,
            "cache_age_minutes": self.get_cached_filtered_events_age(),
        }
    
    def get_filter_stats(self) -> dict:
        """Get statistics from the relevance filter."""
        if self.relevance_filter:
            return self.relevance_filter.get_stats()
        return {}
    
    def get_risk_assessment(self, refresh: bool = False) -> GeopoliticalRiskAssessment:
        """
        Get comprehensive geopolitical risk assessment.
        
        This is the main method that integrates all intelligence.
        """
        # Update events if needed
        if refresh or not self.last_update or \
           (datetime.now(pytz.UTC) - self.last_update) > timedelta(minutes=30):
            self.update_events()
        
        now = datetime.now(pytz.UTC)
        
        # Get high-impact events from last 24 hours
        cutoff_24h = now - timedelta(hours=24)
        recent_events = [e for e in self.events_cache 
                        if e.timestamp.replace(tzinfo=pytz.UTC) > cutoff_24h]
        
        # Filter to significant events
        significant_events = [e for e in recent_events if e.market_impact_score > 0.4]
        
        # Calculate regional risks
        regional_risks = {}
        for event in significant_events:
            for region in event.regions:
                current_risk = regional_risks.get(region, 0)
                regional_risks[region] = min(1.0, current_risk + event.market_impact_score * 0.2)
        
        # Get live regional market data
        market_data = self.fetch_regional_market_data()
        
        # Incorporate market panic signals
        for region, data in market_data.items():
            if data.get("is_panic"):
                regional_risks[region] = min(1.0, regional_risks.get(region, 0) + 0.3)
        
        # Calculate overall risk
        if significant_events:
            # Weight by recency (more recent = higher weight)
            weighted_scores = []
            for event in significant_events:
                age_hours = (now - event.timestamp.replace(tzinfo=pytz.UTC)).total_seconds() / 3600
                recency_weight = max(0.3, 1.0 - (age_hours / 24))  # Decay over 24h
                weighted_scores.append(event.market_impact_score * recency_weight)
            
            # Overall score: combination of max and average
            max_score = max(weighted_scores)
            avg_score = sum(weighted_scores) / len(weighted_scores)
            overall_risk = 0.6 * max_score + 0.4 * avg_score
        else:
            overall_risk = 0.1  # Low baseline
        
        # Determine risk level
        if overall_risk >= 0.8:
            risk_level = "critical"
            exposure_adj = 0.3
        elif overall_risk >= 0.6:
            risk_level = "high"
            exposure_adj = 0.5
        elif overall_risk >= 0.4:
            risk_level = "elevated"
            exposure_adj = 0.7
        elif overall_risk >= 0.2:
            risk_level = "moderate"
            exposure_adj = 0.9
        else:
            risk_level = "low"
            exposure_adj = 1.0
        
        # Determine if safe haven rotation is warranted
        safe_haven_signal = (
            overall_risk >= 0.6 or
            any(e.event_type == "military" and e.severity > 0.7 for e in significant_events) or
            sum(1 for d in market_data.values() if d.get("is_panic")) >= 2
        )
        
        # Extract key concerns
        key_concerns = []
        for event in sorted(significant_events, 
                           key=lambda x: x.market_impact_score, 
                           reverse=True)[:5]:
            key_concerns.append(f"{event.event_type.upper()}: {event.headline[:80]}")
        
        self.last_assessment = GeopoliticalRiskAssessment(
            timestamp=now,
            overall_risk_score=round(overall_risk, 3),
            risk_level=risk_level,
            active_events=significant_events[:20],
            regional_risks=regional_risks,
            recommended_exposure_adjustment=exposure_adj,
            key_concerns=key_concerns,
            safe_haven_signal=safe_haven_signal,
        )
        
        return self.last_assessment
    
    def get_context_for_llm(self) -> str:
        """
        Generate context string for LLM debate/reasoning.
        """
        assessment = self.get_risk_assessment()
        
        context_parts = [
            f"## Geopolitical Risk Assessment",
            f"Overall Risk: {assessment.risk_level.upper()} ({assessment.overall_risk_score:.0%})",
            f"Recommended Exposure Adjustment: {assessment.recommended_exposure_adjustment:.0%}",
            f"Safe Haven Rotation Signal: {'YES' if assessment.safe_haven_signal else 'No'}",
        ]
        
        if assessment.regional_risks:
            context_parts.append("\n### Regional Risks:")
            for region, risk in sorted(assessment.regional_risks.items(), 
                                       key=lambda x: x[1], reverse=True):
                context_parts.append(f"  - {region}: {risk:.0%}")
        
        if assessment.key_concerns:
            context_parts.append("\n### Key Concerns:")
            for concern in assessment.key_concerns:
                context_parts.append(f"  - {concern}")
        
        if assessment.active_events:
            context_parts.append(f"\n### Recent Events ({len(assessment.active_events)}):")
            for event in assessment.active_events[:5]:
                age_hours = (datetime.now(pytz.UTC) - 
                            event.timestamp.replace(tzinfo=pytz.UTC)).total_seconds() / 3600
                context_parts.append(
                    f"  - [{event.event_type}] {event.headline[:60]}... "
                    f"(severity: {event.severity:.0%}, {age_hours:.0f}h ago)"
                )
        
        return "\n".join(context_parts)
    
    def get_exposure_multiplier(self) -> float:
        """Get recommended exposure multiplier based on geopolitical risk."""
        assessment = self.get_risk_assessment()
        return assessment.recommended_exposure_adjustment


# Singleton instance
_geo_intel_instance: Optional[GeopoliticalIntelligence] = None

def get_geopolitical_intel() -> GeopoliticalIntelligence:
    """Get singleton instance of GeopoliticalIntelligence."""
    global _geo_intel_instance
    if _geo_intel_instance is None:
        _geo_intel_instance = GeopoliticalIntelligence()
    return _geo_intel_instance
