"""
News data ingestion and entity linking.
Supports World News API and configurable news sources.
Includes intelligent caching to minimize API calls.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import re
import logging
import requests
import os
import hashlib

logger = logging.getLogger(__name__)


# World News API Configuration
WORLD_NEWS_API_KEY = "c500f1ab944c4594aac42c480db66c97"
WORLD_NEWS_API_BASE = "https://api.worldnewsapi.com"


@dataclass
class NewsArticle:
    """Represents a single news article."""
    id: str
    timestamp: datetime
    headline: str
    body: Optional[str]
    source: str
    url: Optional[str]
    tickers: List[str]  # Linked tickers
    topics: List[str]   # Topic tags
    raw_sentiment: Optional[float] = None


class NewsDataLoader:
    """
    Load and process news data with entity linking to tickers.
    Supports World News API for real news data.
    """
    
    # Ticker to company name mapping for entity linking
    TICKER_ENTITIES = {
        'AAPL': ['apple', 'iphone', 'ipad', 'tim cook', 'cupertino'],
        'MSFT': ['microsoft', 'windows', 'azure', 'satya nadella', 'xbox', 'copilot'],
        'AMZN': ['amazon', 'aws', 'bezos', 'jassy', 'prime', 'alexa'],
        'GOOGL': ['google', 'alphabet', 'youtube', 'sundar pichai', 'android', 'gemini'],
        'META': ['meta', 'facebook', 'instagram', 'zuckerberg', 'whatsapp', 'threads'],
        'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'ev maker', 'cybertruck'],
        'NVDA': ['nvidia', 'jensen huang', 'gpu', 'graphics', 'cuda', 'geforce'],
        'JPM': ['jpmorgan', 'jp morgan', 'jamie dimon', 'chase'],
        'BAC': ['bank of america', 'bofa'],
        'GS': ['goldman sachs', 'goldman'],
        'MS': ['morgan stanley'],
        'WMT': ['walmart', 'wal-mart'],
        'XOM': ['exxon', 'exxonmobil'],
        'CVX': ['chevron'],
        'JNJ': ['johnson & johnson', 'j&j'],
        'PFE': ['pfizer'],
        'UNH': ['unitedhealth', 'united health'],
        'PG': ['procter & gamble', 'procter and gamble', 'p&g'],
        'KO': ['coca-cola', 'coca cola', 'coke'],
        'PEP': ['pepsi', 'pepsico'],
        'DIS': ['disney', 'walt disney'],
        'NFLX': ['netflix'],
        'INTC': ['intel'],
        'AMD': ['amd', 'advanced micro devices', 'lisa su'],
        'CRM': ['salesforce'],
        'ORCL': ['oracle'],
        'ADBE': ['adobe'],
        'BA': ['boeing'],
        'CAT': ['caterpillar'],
        'GE': ['general electric'],
        'HON': ['honeywell'],
        'LMT': ['lockheed martin', 'lockheed'],
        'RTX': ['raytheon'],
        'V': ['visa'],
        'MA': ['mastercard'],
        'AVGO': ['broadcom'],
    }
    
    # Topic keywords for classification
    TOPIC_KEYWORDS = {
        'rates': ['interest rate', 'fed', 'federal reserve', 'powell', 'rate hike', 'rate cut', 'fomc', 'monetary policy', 'central bank'],
        'inflation': ['inflation', 'cpi', 'consumer price', 'pce', 'deflation', 'price increase'],
        'earnings': ['earnings', 'eps', 'revenue', 'quarterly results', 'beat estimates', 'missed estimates', 'profit', 'guidance'],
        'oil': ['oil', 'crude', 'opec', 'petroleum', 'brent', 'wti', 'energy prices'],
        'war': ['war', 'conflict', 'military', 'invasion', 'sanctions', 'geopolitical', 'ukraine', 'russia', 'israel'],
        'regulation': ['regulation', 'sec', 'antitrust', 'ftc', 'lawsuit', 'doj', 'compliance', 'fine'],
        'crypto': ['bitcoin', 'crypto', 'ethereum', 'blockchain', 'cryptocurrency', 'digital currency'],
        'tech': ['tech', 'ai', 'artificial intelligence', 'machine learning', 'chip', 'semiconductor', 'software'],
        'china': ['china', 'chinese', 'beijing', 'tariff', 'trade war', 'xi jinping'],
        'recession': ['recession', 'slowdown', 'gdp', 'economic growth', 'unemployment', 'layoffs', 'downturn'],
        'market': ['stock market', 'wall street', 'nasdaq', 's&p', 'dow jones', 'rally', 'selloff', 'bull', 'bear'],
    }
    
    def __init__(
        self, 
        data_path: Optional[str] = None, 
        api_key: Optional[str] = None,
        cache_dir: str = "outputs/news_cache",
        cache_ttl_hours: int = 6,  # News stays fresh for 6 hours
        max_cache_age_days: int = 30,  # Delete news older than 30 days
    ):
        """
        Initialize news data loader with caching.
        
        Args:
            data_path: Path to news data directory (for local files)
            api_key: World News API key (defaults to embedded key)
            cache_dir: Directory to store cached news
            cache_ttl_hours: Hours before cached news needs refresh
            max_cache_age_days: Days before old news is deleted
        """
        self.data_path = Path(data_path) if data_path else None
        self.api_key = api_key or os.getenv("WORLD_NEWS_API_KEY", WORLD_NEWS_API_KEY)
        self._articles_cache: List[NewsArticle] = []
        
        # Cache configuration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.max_cache_age = timedelta(days=max_cache_age_days)
        
        # Track query history to avoid duplicate API calls
        self._query_cache_file = self.cache_dir / "query_cache.json"
        self._articles_cache_file = self.cache_dir / "articles_cache.json"
        self._query_history: Dict[str, datetime] = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load cached articles and query history from disk."""
        # Load query history
        if self._query_cache_file.exists():
            try:
                with open(self._query_cache_file, 'r') as f:
                    data = json.load(f)
                self._query_history = {
                    k: datetime.fromisoformat(v) 
                    for k, v in data.items()
                }
                logger.info(f"Loaded {len(self._query_history)} cached queries")
            except Exception as e:
                logger.warning(f"Could not load query cache: {e}")
        
        # Load cached articles
        if self._articles_cache_file.exists():
            try:
                with open(self._articles_cache_file, 'r') as f:
                    data = json.load(f)
                
                for item in data.get('articles', []):
                    try:
                        article = NewsArticle(
                            id=item['id'],
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            headline=item['headline'],
                            body=item.get('body'),
                            source=item['source'],
                            url=item.get('url'),
                            tickers=item.get('tickers', []),
                            topics=item.get('topics', []),
                            raw_sentiment=item.get('raw_sentiment'),
                        )
                        self._articles_cache.append(article)
                    except Exception:
                        continue
                
                logger.info(f"Loaded {len(self._articles_cache)} cached articles")
                
                # Clean up old articles
                self._cleanup_old_articles()
                
            except Exception as e:
                logger.warning(f"Could not load articles cache: {e}")
    
    def _save_cache(self):
        """Persist cache to disk."""
        try:
            # Save query history
            with open(self._query_cache_file, 'w') as f:
                json.dump({
                    k: v.isoformat() 
                    for k, v in self._query_history.items()
                }, f, indent=2)
            
            # Save articles
            articles_data = []
            for article in self._articles_cache:
                articles_data.append({
                    'id': article.id,
                    'timestamp': article.timestamp.isoformat(),
                    'headline': article.headline,
                    'body': article.body,
                    'source': article.source,
                    'url': article.url,
                    'tickers': article.tickers,
                    'topics': article.topics,
                    'raw_sentiment': article.raw_sentiment,
                })
            
            with open(self._articles_cache_file, 'w') as f:
                json.dump({
                    'articles': articles_data,
                    'last_updated': datetime.now().isoformat(),
                    'total_articles': len(articles_data),
                }, f, indent=2)
            
            logger.debug(f"Saved {len(self._articles_cache)} articles to cache")
            
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _cleanup_old_articles(self):
        """Remove articles older than max_cache_age."""
        cutoff = datetime.now() - self.max_cache_age
        original_count = len(self._articles_cache)
        
        self._articles_cache = [
            a for a in self._articles_cache
            if a.timestamp > cutoff
        ]
        
        removed = original_count - len(self._articles_cache)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old articles from cache")
            self._save_cache()
    
    def _make_query_key(self, query_type: str, params: Dict) -> str:
        """Generate a unique cache key for a query."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        hash_val = hashlib.md5(params_str.encode()).hexdigest()[:12]
        return f"{query_type}_{hash_val}"
    
    def _is_query_fresh(self, query_key: str) -> bool:
        """Check if a query result is still fresh in cache."""
        if query_key not in self._query_history:
            return False
        
        last_query_time = self._query_history[query_key]
        return datetime.now() - last_query_time < self.cache_ttl
    
    def _get_cached_articles(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """Get articles from cache matching criteria."""
        results = []
        
        for article in self._articles_cache:
            # Date filter
            if start_date and article.timestamp < start_date:
                continue
            if end_date and article.timestamp > end_date:
                continue
            
            # Ticker filter
            if tickers:
                if not any(t in article.tickers for t in tickers):
                    continue
            
            # Topic filter
            if topics:
                if not any(t in article.topics for t in topics):
                    continue
            
            results.append(article)
        
        return results
    
    def _add_to_cache(self, articles: List[NewsArticle], query_key: str):
        """Add articles to cache and mark query as complete."""
        # Add new articles (deduplicate by ID)
        existing_ids = {a.id for a in self._articles_cache}
        new_articles = 0
        
        for article in articles:
            if article.id not in existing_ids:
                self._articles_cache.append(article)
                existing_ids.add(article.id)
                new_articles += 1
        
        # Mark query as completed
        self._query_history[query_key] = datetime.now()
        
        # Save to disk
        self._save_cache()
        
        logger.debug(f"Added {new_articles} new articles to cache")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'total_articles': len(self._articles_cache),
            'total_queries_cached': len(self._query_history),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'api_calls_saved': self.api_calls_saved,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'oldest_article': min((a.timestamp for a in self._articles_cache), default=None),
            'newest_article': max((a.timestamp for a in self._articles_cache), default=None),
        }
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear the news cache.
        
        Args:
            older_than_days: If provided, only clear articles older than this
        """
        if older_than_days:
            cutoff = datetime.now() - timedelta(days=older_than_days)
            original = len(self._articles_cache)
            self._articles_cache = [
                a for a in self._articles_cache
                if a.timestamp > cutoff
            ]
            removed = original - len(self._articles_cache)
            logger.info(f"Cleared {removed} articles older than {older_than_days} days")
        else:
            self._articles_cache = []
            self._query_history = {}
            logger.info("Cleared entire news cache")
        
        self._save_cache()
        
    def load_news(
        self,
        start_date: datetime,
        end_date: datetime,
        source: str = "worldnews"
    ) -> List[NewsArticle]:
        """
        Load news articles for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            source: Data source ('worldnews', 'local', 'sample')
            
        Returns:
            List of NewsArticle objects
        """
        if source == "worldnews":
            return self._load_world_news_api(start_date, end_date)
        elif source == "local":
            return self._load_local(start_date, end_date)
        else:
            return self.generate_sample_news([], start_date, end_date)
    
    def _load_world_news_api(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[NewsArticle]:
        """
        Load news from World News API with intelligent caching.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of NewsArticle objects
        """
        articles = []
        api_calls_made = 0
        cache_hits = 0
        
        # Search for financial/market news - MORE SPECIFIC TERMS
        search_terms = [
            "stock market today",
            "S&P 500 earnings",
            "NASDAQ stocks",
            "Federal Reserve interest rates",
            "quarterly earnings report",
            "Wall Street trading",
            "stock price target",
            "analyst upgrade downgrade",
            "company revenue profit",
            "CEO investor",
        ]
        
        logger.info(f"Fetching news (checking cache first)...")
        
        for term in search_terms:
            # Create cache key for this query
            query_key = self._make_query_key("search", {
                "term": term,
                "start": start_date.date().isoformat(),
                "end": end_date.date().isoformat(),
            })
            
            # Check if we have fresh cached results
            if self._is_query_fresh(query_key):
                # Get from cache
                cached = self._get_cached_articles(start_date=start_date, end_date=end_date)
                # Filter by term relevance (simple keyword match)
                term_lower = term.lower()
                relevant = [a for a in cached if term_lower in a.headline.lower() 
                           or (a.body and term_lower in a.body.lower())]
                if relevant:
                    articles.extend(relevant)
                    cache_hits += 1
                    self.cache_hits += 1
                    logger.debug(f"  '{term}': {len(relevant)} articles (CACHED)")
                    continue
            
            # Need to fetch from API
            try:
                batch = self._search_world_news(
                    keywords=term,
                    start_date=start_date,
                    end_date=end_date,
                    limit=20
                )
                articles.extend(batch)
                api_calls_made += 1
                self.cache_misses += 1
                
                # Add to cache
                self._add_to_cache(batch, query_key)
                
                logger.debug(f"  '{term}': {len(batch)} articles (API)")
            except Exception as e:
                logger.warning(f"Error fetching '{term}': {e}")
                continue
        
        # Also fetch top business news (cache for longer)
        top_news_key = self._make_query_key("top_news", {"category": "business"})
        
        if self._is_query_fresh(top_news_key):
            # Use cached top news
            cached_top = self._get_cached_articles(start_date=start_date, end_date=end_date)
            cache_hits += 1
            self.cache_hits += 1
            logger.debug(f"  Top business news: {len(cached_top)} articles (CACHED)")
        else:
            try:
                top_news = self._get_top_news(category="business", limit=50)
                articles.extend(top_news)
                api_calls_made += 1
                self.cache_misses += 1
                self._add_to_cache(top_news, top_news_key)
                logger.debug(f"  Top business news: {len(top_news)} articles (API)")
            except Exception as e:
                logger.warning(f"Error fetching top news: {e}")
        
        # Deduplicate by ID
        seen_ids = set()
        unique_articles = []
        for article in articles:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique_articles.append(article)
        
        # Update saved API calls stat
        self.api_calls_saved += cache_hits
        
        logger.info(
            f"Loaded {len(unique_articles)} articles "
            f"(API calls: {api_calls_made}, Cache hits: {cache_hits}, "
            f"Total saved: {self.api_calls_saved})"
        )
        
        return sorted(unique_articles, key=lambda x: x.timestamp, reverse=True)
    
    def _search_world_news(
        self,
        keywords: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Search World News API for articles.
        """
        articles = []
        
        params = {
            "api-key": self.api_key,
            "text": keywords,
            "language": "en",
            "number": min(limit, 100),
            "sort": "publish-time",
            "sort-direction": "DESC",
        }
        
        # Add date filters
        if start_date:
            params["earliest-publish-date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["latest-publish-date"] = end_date.strftime("%Y-%m-%d")
        
        try:
            response = requests.get(
                f"{WORLD_NEWS_API_BASE}/search-news",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get("news", []):
                    article = self._parse_world_news_article(item)
                    if article:
                        articles.append(article)
                        
            elif response.status_code == 401:
                logger.error("World News API: Invalid API key")
            elif response.status_code == 429:
                logger.warning("World News API: Rate limit exceeded")
            else:
                logger.warning(f"World News API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"World News API request failed: {e}")
        
        return articles
    
    def _get_top_news(
        self,
        category: str = "business",
        country: str = "us",
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Get top news from World News API.
        """
        articles = []
        
        params = {
            "api-key": self.api_key,
            "source-country": country,
            "language": "en",
        }
        
        try:
            response = requests.get(
                f"{WORLD_NEWS_API_BASE}/top-news",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Top news returns nested structure
                top_news = data.get("top_news", [])
                for category_news in top_news:
                    for item in category_news.get("news", [])[:limit]:
                        article = self._parse_world_news_article(item)
                        # Double-check financial relevance for top news
                        if article and self._is_financial_news(article.headline, article.body or ""):
                            articles.append(article)
                            
        except requests.exceptions.RequestException as e:
            logger.warning(f"World News API top news failed: {e}")
        
        return articles
    
    # Words that indicate NON-financial news (filter these out)
    NON_FINANCIAL_KEYWORDS = {
        'football', 'basketball', 'soccer', 'baseball', 'hockey', 'nfl', 'nba', 
        'mlb', 'nhl', 'championship', 'touchdown', 'quarterback', 'coach fired',
        'celebrity', 'kardashian', 'movie', 'netflix series', 'tv show', 'actor',
        'actress', 'grammy', 'oscar', 'emmy', 'concert', 'album', 'music video',
        'recipe', 'cooking', 'weather forecast', 'horoscope', 'zodiac',
        'dating', 'wedding', 'divorce', 'scandal', 'affair',
    }
    
    # Words that indicate FINANCIAL news (prefer these)
    FINANCIAL_KEYWORDS = {
        'stock', 'shares', 'earnings', 'revenue', 'profit', 'loss', 'quarter',
        'ceo', 'cfo', 'investor', 'dividend', 'buyback', 'ipo', 'merger',
        'acquisition', 'sec', 'fed', 'interest rate', 'inflation', 'gdp',
        'market', 'trading', 'analyst', 'upgrade', 'downgrade', 'price target',
        'billion', 'million', 'guidance', 'forecast', 'estimate', 'beat', 'miss',
        'wall street', 'nasdaq', 's&p', 'dow jones', 'bond', 'treasury',
    }
    
    def _is_financial_news(self, headline: str, body: str = "") -> bool:
        """Check if an article is financial/market related."""
        text = f"{headline} {body}".lower()
        
        # Reject if contains non-financial keywords
        for keyword in self.NON_FINANCIAL_KEYWORDS:
            if keyword in text:
                return False
        
        # Accept if contains financial keywords
        for keyword in self.FINANCIAL_KEYWORDS:
            if keyword in text:
                return True
        
        # Default: reject (be strict about financial relevance)
        return False
    
    def _parse_world_news_article(self, item: dict) -> Optional[NewsArticle]:
        """
        Parse a World News API article into NewsArticle object.
        """
        try:
            # Parse timestamp
            publish_date = item.get("publish_date", "")
            if publish_date:
                try:
                    timestamp = pd.to_datetime(publish_date)
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            headline = item.get("title", "")
            body = item.get("text", item.get("summary", ""))
            
            if not headline:
                return None
            
            # FILTER: Only accept financial/market news
            if not self._is_financial_news(headline, body):
                logger.debug(f"Skipping non-financial article: {headline[:50]}...")
                return None
            
            # Entity linking
            text = f"{headline} {body}".lower()
            tickers = self._extract_tickers(text)
            topics = self._extract_topics(text)
            
            # Get sentiment if provided by API
            raw_sentiment = item.get("sentiment")
            if raw_sentiment is not None:
                try:
                    raw_sentiment = float(raw_sentiment)
                except:
                    raw_sentiment = None
            
            return NewsArticle(
                id=str(item.get("id", hash(headline))),
                timestamp=timestamp,
                headline=headline,
                body=body[:2000] if body else None,  # Limit body length
                source=item.get("source", {}).get("name", "Unknown") if isinstance(item.get("source"), dict) else str(item.get("source", "Unknown")),
                url=item.get("url"),
                tickers=tickers,
                topics=topics,
                raw_sentiment=raw_sentiment,
            )
            
        except Exception as e:
            logger.debug(f"Error parsing article: {e}")
            return None
    
    def fetch_news_for_symbols(
        self,
        symbols: List[str],
        days_back: int = 7
    ) -> List[NewsArticle]:
        """
        Fetch news specifically for given stock symbols with caching.
        
        Args:
            symbols: List of ticker symbols
            days_back: How many days back to search
            
        Returns:
            List of NewsArticle objects
        """
        articles = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        api_calls = 0
        cache_hits = 0
        
        # Search by company names
        for symbol in symbols[:20]:  # Limit to avoid too many API calls
            if symbol in self.TICKER_ENTITIES:
                company_names = self.TICKER_ENTITIES[symbol]
                
                # Create cache key for this symbol
                query_key = self._make_query_key("symbol", {
                    "symbol": symbol,
                    "days_back": days_back,
                })
                
                # Check cache first
                if self._is_query_fresh(query_key):
                    # Get cached articles for this symbol
                    cached = self._get_cached_articles(
                        tickers=[symbol],
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if cached:
                        articles.extend(cached)
                        cache_hits += 1
                        self.cache_hits += 1
                        logger.debug(f"  {symbol}: {len(cached)} articles (CACHED)")
                        continue
                
                # Need to fetch from API
                for name in company_names[:1]:  # Use first/main name
                    try:
                        batch = self._search_world_news(
                            keywords=name,
                            start_date=start_date,
                            end_date=end_date,
                            limit=10
                        )
                        # Tag articles with this ticker
                        for article in batch:
                            if symbol not in article.tickers:
                                article.tickers.append(symbol)
                        articles.extend(batch)
                        api_calls += 1
                        self.cache_misses += 1
                        
                        # Cache the results
                        self._add_to_cache(batch, query_key)
                        logger.debug(f"  {symbol}: {len(batch)} articles (API)")
                        
                    except Exception as e:
                        logger.debug(f"Error fetching news for {symbol}: {e}")
                        continue
        
        # Deduplicate
        seen_ids = set()
        unique = []
        for article in articles:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique.append(article)
        
        self.api_calls_saved += cache_hits
        
        logger.info(
            f"Fetched {len(unique)} articles for {len(symbols)} symbols "
            f"(API: {api_calls}, Cached: {cache_hits}, Total saved: {self.api_calls_saved})"
        )
        return unique
    
    def _load_local(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[NewsArticle]:
        """Load from local JSON files."""
        articles = []
        
        if self.data_path is None:
            return articles
            
        # Try loading from JSON files
        patterns = [
            self.data_path / "news.json",
            self.data_path / "news" / "*.json",
        ]
        
        for pattern in patterns:
            for path in Path(self.data_path).glob(str(pattern).replace(str(self.data_path) + "/", "")):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        
                    for item in data:
                        timestamp = pd.to_datetime(item.get('timestamp', item.get('date')))
                        
                        if start_date <= timestamp <= end_date:
                            headline = item.get('headline', item.get('title', ''))
                            body = item.get('body', item.get('content', ''))
                            
                            # Entity linking
                            text = f"{headline} {body}".lower()
                            tickers = self._extract_tickers(text)
                            topics = self._extract_topics(text)
                            
                            article = NewsArticle(
                                id=item.get('id', str(hash(headline))),
                                timestamp=timestamp,
                                headline=headline,
                                body=body,
                                source=item.get('source', 'unknown'),
                                url=item.get('url'),
                                tickers=tickers,
                                topics=topics,
                            )
                            articles.append(article)
                            
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
                    
        return sorted(articles, key=lambda x: x.timestamp)
    
    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract tickers from text using entity linking.
        
        Args:
            text: Lowercase text to search
            
        Returns:
            List of matched ticker symbols
        """
        tickers = []
        
        for ticker, entities in self.TICKER_ENTITIES.items():
            for entity in entities:
                if entity in text:
                    tickers.append(ticker)
                    break
                    
        # Also look for direct ticker mentions (e.g., $AAPL, AAPL)
        ticker_pattern = r'\$?([A-Z]{2,5})\b'
        matches = re.findall(ticker_pattern, text.upper())
        for match in matches:
            if match in self.TICKER_ENTITIES and match not in tickers:
                tickers.append(match)
                
        return list(set(tickers))
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from text using keyword matching.
        
        Args:
            text: Lowercase text to search
            
        Returns:
            List of matched topics
        """
        topics = []
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    topics.append(topic)
                    break
                    
        return topics
    
    def generate_sample_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        articles_per_day: int = 10,
        seed: int = 42
    ) -> List[NewsArticle]:
        """
        Generate sample news articles for testing.
        """
        np.random.seed(seed)
        
        if not symbols:
            symbols = list(self.TICKER_ENTITIES.keys())[:20]
        
        dates = pd.date_range(start_date, end_date, freq='D')
        articles = []
        
        headlines = [
            "{symbol} Reports Strong Quarterly Earnings, Beats Estimates",
            "{symbol} Shares Fall on Weak Guidance",
            "{symbol} Announces Major Acquisition",
            "{symbol} CEO Discusses Growth Strategy",
            "{symbol} Faces Regulatory Scrutiny",
            "Analysts Upgrade {symbol} to Buy",
            "{symbol} Expands Into New Markets",
            "{symbol} Cuts Jobs Amid Restructuring",
            "{symbol} Launches New Product Line",
            "{symbol} Stock Hits 52-Week High",
        ]
        
        for date in dates:
            for _ in range(articles_per_day):
                symbol = np.random.choice(symbols)
                headline = np.random.choice(headlines).format(symbol=symbol)
                
                # Random sentiment
                sentiment = np.random.uniform(-1, 1)
                
                # Random topics
                topics = list(np.random.choice(
                    list(self.TOPIC_KEYWORDS.keys()),
                    size=np.random.randint(0, 3),
                    replace=False
                ))
                
                article = NewsArticle(
                    id=f"{date.strftime('%Y%m%d')}_{len(articles)}",
                    timestamp=date + pd.Timedelta(hours=np.random.randint(6, 20)),
                    headline=headline,
                    body=f"Full article about {symbol}...",
                    source="sample",
                    url=None,
                    tickers=[symbol],
                    topics=topics,
                    raw_sentiment=sentiment,
                )
                articles.append(article)
                
        return articles
