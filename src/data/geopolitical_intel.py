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
    # EXPANDED to capture ALL market-moving categories
    HIGH_IMPACT_KEYWORDS = {
        "military": ["airstrike", "military", "jets", "troops", "deploy", "missile", 
                     "strike", "attack", "invasion", "war", "conflict", "combat",
                     "bombing", "airspace", "naval", "defense", "offensive",
                     "retaliation", "mobilization", "ceasefire", "peace talks"],
        "diplomatic": ["sanctions", "embargo", "diplomatic", "treaty", "alliance",
                      "ambassador", "expelled", "relations", "summit", "talks",
                      "tariff", "trade war", "trade deal", "export controls",
                      "blacklist", "asset freeze", "swift"],
        "economic": ["tariff", "trade war", "currency", "devaluation", "default",
                    "bailout", "crisis", "collapse", "shutdown", "recession",
                    "gdp", "inflation", "cpi", "ppi", "unemployment", "payrolls",
                    "retail sales", "pmi", "ism", "housing"],
        "civil_unrest": ["protest", "riot", "uprising", "revolution", "coup",
                        "martial law", "curfew", "emergency", "unrest",
                        "election", "referendum", "impeachment", "government"],
        "infrastructure": ["flight cancelled", "flights cancelled", "airspace closed",
                          "port closed", "shipping disrupted", "pipeline", "embargo",
                          "suez", "panama canal", "strait of hormuz", "supply chain",
                          "freight", "logistics", "blockade", "reroute"],
        # NEW CATEGORIES - Critical for market intelligence
        "central_bank": ["federal reserve", "fed", "fomc", "ecb", "bank of england",
                        "bank of japan", "boj", "pboc", "rate hike", "rate cut",
                        "interest rate", "monetary policy", "hawkish", "dovish",
                        "quantitative easing", "qe", "quantitative tightening", "qt",
                        "balance sheet", "liquidity", "pivot", "dot plot",
                        "fed chair", "powell", "lagarde", "bailey", "ueda",
                        "central bank", "rate decision", "fomc meeting"],
        "energy": ["opec", "oil", "crude", "brent", "wti", "natural gas", "lng",
                  "refinery", "production cut", "output cut", "oil prices",
                  "energy crisis", "fuel", "gasoline", "petroleum",
                  "tanker", "force majeure", "strategic reserves",
                  "gold", "silver", "copper", "commodity", "commodities",
                  "mining", "rare earth", "precious metals", "safe haven"],
        "financial_stress": ["bank run", "banking crisis", "credit crisis",
                            "liquidity crisis", "margin calls", "contagion",
                            "downgrade", "junk", "distressed", "restructuring",
                            "imf", "bailout", "rescue", "capital shortfall",
                            "sovereign debt", "yield curve", "credit spread"],
        "shipping": ["shipping", "freight", "container", "port", "tanker",
                    "suez canal", "panama canal", "strait of hormuz", "red sea",
                    "logistics", "supply chain", "cargo", "reroute",
                    "insurance premium", "shipping rates", "chokepoint"],
    }
    
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
        
        # API keys
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        
        # Event cache
        self.events_cache: List[GeopoliticalEvent] = []
        self.last_assessment: Optional[GeopoliticalRiskAssessment] = None
        self.last_update: Optional[datetime] = None
        
        # Advanced relevance filter (rule-based, deterministic)
        self.relevance_filter = get_news_filter() if HAS_RELEVANCE_FILTER else None
        self.filtered_events: List[NewsEvent] = []  # High-quality filtered events
        
        # Load cache
        self._load_cache()
    
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
    
    def _generate_event_id(self, headline: str, source: str) -> str:
        """Generate unique event ID from headline and source."""
        content = f"{headline[:100]}_{source}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _classify_event(self, text: str) -> Tuple[str, float, List[str]]:
        """
        Classify event type and severity from text.
        Returns: (event_type, severity, matched_keywords)
        """
        text_lower = " " + text.lower() + " "  # Add spaces for word boundary matching
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
        
        # Calculate severity based on keyword matches and urgency words
        urgency_words = ["breaking", "urgent", "emergency", "imminent", "immediate", 
                        "critical", "major", "massive", "unprecedented", "surge",
                        "plunge", "crisis", "shock", "surprise", "unexpected"]
        urgency_count = sum(1 for w in urgency_words if w in text_lower)
        
        # Base severity now starts higher (0.3) so market-moving events pass
        # Each keyword match adds to severity
        base_severity = 0.3 if max_matches > 0 else 0.1
        keyword_bonus = min(0.4, max_matches * 0.12)  # Up to 0.4 from keywords
        urgency_bonus = min(0.3, urgency_count * 0.1)  # Up to 0.3 from urgency
        
        # High-impact event types get base bonus
        high_impact_types = ["military", "central_bank", "energy", "financial_stress", "shipping"]
        if event_type in high_impact_types:
            base_severity += 0.1
        
        severity = min(1.0, base_severity + keyword_bonus + urgency_bonus)
        
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
                        
                        # Skip low-severity events
                        if severity < 0.3:
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
            self.filtered_events = filtered
            
            # Log filter stats
            stats = self.relevance_filter.get_stats()
            logging.info(f"Filter results: {stats['accepted']} accepted, "
                        f"{stats['rejected']} rejected "
                        f"({stats['acceptance_rate']*100:.1f}% acceptance rate)")
            
            # Save filtered events
            if filtered:
                self.relevance_filter.save_events(filtered)
        
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
    
    def get_filtered_events(self) -> List:
        """Get high-quality filtered events (from advanced filter)."""
        return self.filtered_events
    
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
