"""
Alpha Vantage News Sentiment API adapter.
Replaces World News API for market news and sentiment data.
"""
import os
import requests
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# Alpha Vantage API Configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "IOCS4REOCIVL21MW")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


@dataclass
class TickerSentimentData:
    """Per-ticker sentiment data from Alpha Vantage - THIS IS THE KEY DATA WE WERE IGNORING."""
    ticker: str
    relevance_score: float  # 0-1, how relevant the article is to this ticker
    sentiment_score: float  # -1 to +1, sentiment toward this ticker
    sentiment_label: str    # Bearish, Somewhat-Bearish, Neutral, Somewhat-Bullish, Bullish


@dataclass
class AlphaVantageArticle:
    """Represents a news article from Alpha Vantage."""
    id: str
    timestamp: datetime
    headline: str
    summary: str
    source: str
    url: str
    tickers: List[str]
    topics: List[str]
    overall_sentiment: float  # -1 to 1
    overall_sentiment_label: str  # Bearish, Somewhat-Bearish, Neutral, Somewhat-Bullish, Bullish
    
    # THE CRITICAL DATA WE WERE IGNORING:
    ticker_sentiment_details: List[TickerSentimentData]  # Full per-ticker sentiment with relevance
    
    # Legacy field for backwards compatibility
    ticker_sentiments: Dict[str, float]  # Per-ticker sentiment scores (simple version)


class AlphaVantageNewsLoader:
    """
    Load news and sentiment data from Alpha Vantage API.
    
    Features:
    - Ticker-specific news
    - Topic-based filtering (earnings, ipo, mergers_and_acquisitions, etc.)
    - Built-in sentiment scores
    - Intelligent caching
    """
    
    # Available topics for filtering
    TOPICS = [
        'blockchain', 'earnings', 'ipo', 'mergers_and_acquisitions',
        'financial_markets', 'economy_fiscal', 'economy_monetary',
        'economy_macro', 'energy_transportation', 'finance',
        'life_sciences', 'manufacturing', 'real_estate',
        'retail_wholesale', 'technology'
    ]
    
    # Ticker to company name mapping (for logging)
    TICKER_NAMES = {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'META': 'Meta', 'NVDA': 'NVIDIA',
        'TSLA': 'Tesla', 'JPM': 'JPMorgan', 'BAC': 'Bank of America',
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "outputs/alpha_vantage_cache",
        cache_ttl_hours: int = 6,  # Increased: Alpha Vantage free tier only allows 25 req/day
        rate_limit_delay: float = 15.0,  # Conservative: 4 calls/min to stay under limit
    ):
        """
        Initialize Alpha Vantage news loader.
        
        Args:
            api_key: Alpha Vantage API key
            cache_dir: Directory to cache responses
            cache_ttl_hours: Hours before cache expires
            rate_limit_delay: Seconds between API calls (rate limiting)
        """
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.rate_limit_delay = rate_limit_delay
        
        # Track last API call time for rate limiting
        self._last_api_call = 0.0
        
        # Statistics
        self.api_calls_made = 0
        self.cache_hits = 0
        
        # Rate limit tracking
        self._rate_limited = False
        self._rate_limit_message = ""
        self._last_successful_fetch = None
        
        # Article cache
        self._articles_cache: List[AlphaVantageArticle] = []
        self._cache_file = self.cache_dir / "articles_cache.json"
        self._load_cache()
        
        # Load last fetch timestamp
        self._load_fetch_status()
    
    def _load_fetch_status(self):
        """Load last successful fetch timestamp and reset rate limit on new day."""
        status_file = self.cache_dir / "fetch_status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
                    last_fetch_str = data.get('last_successful_fetch', '')
                    if last_fetch_str:
                        self._last_successful_fetch = datetime.fromisoformat(last_fetch_str)
                    self._rate_limited = data.get('rate_limited', False)
                    self._rate_limit_message = data.get('rate_limit_message', '')
                    
                    # CRITICAL: Reset rate limit on a new day (midnight UTC)
                    # Alpha Vantage resets daily quota at midnight UTC
                    if self._rate_limited and self._last_successful_fetch:
                        last_date = self._last_successful_fetch.date()
                        today = datetime.now().date()
                        if today > last_date:
                            logger.info(f"🔄 New day detected - resetting Alpha Vantage rate limit (last fetch: {last_date}, today: {today})")
                            self._rate_limited = False
                            self._rate_limit_message = ""
                            self._save_fetch_status()
            except Exception as e:
                logger.debug(f"Could not load fetch status: {e}")
    
    def _save_fetch_status(self):
        """Save fetch status to disk."""
        status_file = self.cache_dir / "fetch_status.json"
        try:
            with open(status_file, 'w') as f:
                json.dump({
                    'last_successful_fetch': self._last_successful_fetch.isoformat() if self._last_successful_fetch else None,
                    'rate_limited': self._rate_limited,
                    'rate_limit_message': self._rate_limit_message,
                }, f)
        except:
            pass
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status for UI display."""
        cache_stats = self.get_cache_stats()
        newest = cache_stats.get('newest_article')
        
        hours_since_fresh = 0
        if newest:
            hours_since_fresh = (datetime.now() - newest).total_seconds() / 3600
        
        return {
            'rate_limited': self._rate_limited,
            'rate_limit_message': self._rate_limit_message,
            'last_successful_fetch': self._last_successful_fetch.isoformat() if self._last_successful_fetch else None,
            'hours_since_fresh_news': round(hours_since_fresh, 1),
            'cached_articles': len(self._articles_cache),
            'newest_article': newest.isoformat() if newest else None,
            'daily_limit': 25,
            'recommendation': 'Wait for daily reset (midnight UTC) or upgrade API key' if self._rate_limited else 'OK',
        }
    
    def _load_cache(self):
        """Load cached articles from disk."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    data = json.load(f)
                
                for item in data.get('articles', []):
                    try:
                        # Load detailed ticker sentiment
                        ticker_sentiment_details = []
                        for ts in item.get('ticker_sentiment_details', []):
                            ticker_sentiment_details.append(TickerSentimentData(
                                ticker=ts['ticker'],
                                relevance_score=ts.get('relevance_score', 0.0),
                                sentiment_score=ts.get('sentiment_score', 0.0),
                                sentiment_label=ts.get('sentiment_label', 'Neutral'),
                            ))
                        
                        article = AlphaVantageArticle(
                            id=item['id'],
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            headline=item['headline'],
                            summary=item.get('summary', ''),
                            source=item['source'],
                            url=item.get('url', ''),
                            tickers=item.get('tickers', []),
                            topics=item.get('topics', []),
                            overall_sentiment=item.get('overall_sentiment', 0.0),
                            overall_sentiment_label=item.get('overall_sentiment_label', 'Neutral'),
                            ticker_sentiment_details=ticker_sentiment_details,
                            ticker_sentiments=item.get('ticker_sentiments', {}),
                        )
                        self._articles_cache.append(article)
                    except Exception:
                        continue
                
                logger.info(f"Loaded {len(self._articles_cache)} articles from Alpha Vantage cache")
                
            except Exception as e:
                logger.warning(f"Could not load Alpha Vantage cache: {e}")
    
    def _save_cache(self):
        """Save articles to cache - INCLUDING detailed ticker sentiment."""
        try:
            articles_data = []
            for article in self._articles_cache:
                # Save detailed ticker sentiment
                ticker_sentiment_details = []
                if hasattr(article, 'ticker_sentiment_details'):
                    for ts in article.ticker_sentiment_details:
                        ticker_sentiment_details.append({
                            'ticker': ts.ticker,
                            'relevance_score': ts.relevance_score,
                            'sentiment_score': ts.sentiment_score,
                            'sentiment_label': ts.sentiment_label,
                        })
                
                articles_data.append({
                    'id': article.id,
                    'timestamp': article.timestamp.isoformat(),
                    'headline': article.headline,
                    'summary': article.summary,
                    'source': article.source,
                    'url': article.url,
                    'tickers': article.tickers,
                    'topics': article.topics,
                    'overall_sentiment': article.overall_sentiment,
                    'overall_sentiment_label': article.overall_sentiment_label,
                    'ticker_sentiment_details': ticker_sentiment_details,  # NEW
                    'ticker_sentiments': article.ticker_sentiments,
                })
            
            with open(self._cache_file, 'w') as f:
                json.dump({
                    'articles': articles_data,
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_api_call = time.time()
    
    def _make_cache_key(self, params: Dict) -> str:
        """Create a cache key for request parameters."""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    def _is_cache_fresh(self, cache_key: str) -> bool:
        """Check if a cached response is still fresh."""
        cache_file = self.cache_dir / f"response_{cache_key}.json"
        if not cache_file.exists():
            return False
        
        modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - modified_time < self.cache_ttl
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get a cached API response."""
        cache_file = self.cache_dir / f"response_{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_response_cache(self, cache_key: str, data: Dict):
        """Cache an API response."""
        cache_file = self.cache_dir / f"response_{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except:
            pass
    
    def fetch_news(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
        limit: int = 50,
        sort: str = "LATEST",
    ) -> List[AlphaVantageArticle]:
        """
        Fetch news articles from Alpha Vantage.
        
        Args:
            tickers: List of ticker symbols to filter by
            topics: List of topics to filter by
            time_from: Start time for news
            time_to: End time for news
            limit: Maximum number of articles
            sort: Sort order (LATEST, EARLIEST, RELEVANCE)
            
        Returns:
            List of AlphaVantageArticle objects
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": min(limit, 1000),
            "sort": sort,
        }
        
        # Add tickers
        if tickers:
            params["tickers"] = ",".join(tickers[:5])  # API limit
        
        # Add topics
        if topics:
            valid_topics = [t for t in topics if t in self.TOPICS]
            if valid_topics:
                params["topics"] = ",".join(valid_topics[:5])
        
        # Add time range
        if time_from:
            params["time_from"] = time_from.strftime("%Y%m%dT%H%M")
        if time_to:
            params["time_to"] = time_to.strftime("%Y%m%dT%H%M")
        
        # Check cache
        cache_key = self._make_cache_key(params)
        if self._is_cache_fresh(cache_key):
            cached = self._get_cached_response(cache_key)
            if cached:
                self.cache_hits += 1
                logger.debug(f"Alpha Vantage cache hit for {cache_key}")
                return self._parse_response(cached)
        
        # Rate limit and make API call
        self._rate_limit()
        
        try:
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            self.api_calls_made += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for API limit message (Note field)
                if "Note" in data:
                    self._rate_limited = True
                    self._rate_limit_message = data['Note']
                    self._save_fetch_status()
                    logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                    return []
                
                # Check for Information field (daily limit exceeded)
                if "Information" in data:
                    self._rate_limited = True
                    self._rate_limit_message = data['Information']
                    self._save_fetch_status()
                    logger.warning(f"Alpha Vantage daily limit: {data['Information']}")
                    return []
                
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage error: {data['Error Message']}")
                    return []
                
                # Success! Clear rate limit flag
                self._rate_limited = False
                self._rate_limit_message = ""
                self._last_successful_fetch = datetime.now()
                self._save_fetch_status()
                
                # Cache the response
                self._save_response_cache(cache_key, data)
                
                # Parse and return articles
                articles = self._parse_response(data)
                
                # Add to article cache
                self._add_to_cache(articles)
                
                return articles
                
            else:
                logger.error(f"Alpha Vantage API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return []
    
    def _parse_response(self, data: Dict) -> List[AlphaVantageArticle]:
        """Parse Alpha Vantage API response into articles - NOW EXTRACTING ALL DATA."""
        articles = []
        
        feed = data.get("feed", [])
        
        for item in feed:
            try:
                # Parse timestamp
                time_str = item.get("time_published", "")
                try:
                    timestamp = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
                except:
                    timestamp = datetime.now()
                
                # Extract FULL ticker sentiment data (THE KEY DATA WE WERE IGNORING)
                raw_ticker_sentiment = item.get("ticker_sentiment", [])
                
                # Build detailed ticker sentiment list
                ticker_sentiment_details = []
                ticker_sentiments = {}  # Legacy simple dict
                tickers = []
                
                for ts in raw_ticker_sentiment:
                    ticker = ts.get("ticker", "")
                    if not ticker:
                        continue
                    
                    tickers.append(ticker)
                    
                    # Extract the FULL data including relevance score
                    try:
                        relevance = float(ts.get("relevance_score", 0))
                    except:
                        relevance = 0.0
                    
                    try:
                        sent_score = float(ts.get("ticker_sentiment_score", 0))
                    except:
                        sent_score = 0.0
                    
                    sent_label = ts.get("ticker_sentiment_label", "Neutral")
                    
                    # Add to detailed list
                    ticker_sentiment_details.append(TickerSentimentData(
                        ticker=ticker,
                        relevance_score=relevance,
                        sentiment_score=sent_score,
                        sentiment_label=sent_label,
                    ))
                    
                    # Also add to simple dict for backwards compatibility
                    ticker_sentiments[ticker] = sent_score
                
                # Extract topics with relevance
                topics = []
                for t in item.get("topics", []):
                    topic = t.get("topic", "")
                    if topic:
                        topics.append(topic)
                
                # Overall sentiment
                try:
                    sentiment_score = float(item.get("overall_sentiment_score", 0))
                except:
                    sentiment_score = 0.0
                sentiment_label = item.get("overall_sentiment_label", "Neutral")
                
                # Create article with ALL data
                article = AlphaVantageArticle(
                    id=str(hash(item.get("title", "") + time_str)),
                    timestamp=timestamp,
                    headline=item.get("title", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", "Unknown"),
                    url=item.get("url", ""),
                    tickers=tickers,
                    topics=topics,
                    overall_sentiment=sentiment_score,
                    overall_sentiment_label=sentiment_label,
                    ticker_sentiment_details=ticker_sentiment_details,  # NEW: Full data
                    ticker_sentiments=ticker_sentiments,  # Legacy
                )
                
                articles.append(article)
                
            except Exception as e:
                logger.debug(f"Error parsing article: {e}")
                continue
        
        return articles
    
    def _add_to_cache(self, articles: List[AlphaVantageArticle]):
        """Add articles to the cache, deduplicating by ID and headline."""
        existing_ids = {a.id for a in self._articles_cache}
        existing_headlines = {a.headline.lower().strip() for a in self._articles_cache}
        
        new_count = 0
        for article in articles:
            headline_key = article.headline.lower().strip()
            # Deduplicate by both ID and headline
            if article.id not in existing_ids and headline_key not in existing_headlines:
                self._articles_cache.append(article)
                existing_ids.add(article.id)
                existing_headlines.add(headline_key)
                new_count += 1
        
        if new_count > 0:
            # Cleanup old articles to limit storage
            self._cleanup_old_articles()
            self._save_cache()
            logger.debug(f"Added {new_count} new articles to cache")
    
    def _cleanup_old_articles(self, max_articles: int = 500, max_age_days: int = 14):
        """
        Remove old and duplicate articles to save storage.
        Keeps most recent articles up to max_articles.
        """
        if len(self._articles_cache) <= max_articles:
            return
        
        # Remove articles older than max_age_days
        cutoff = datetime.now() - timedelta(days=max_age_days)
        self._articles_cache = [
            a for a in self._articles_cache if a.timestamp > cutoff
        ]
        
        # Deduplicate by headline (keep newest)
        seen_headlines = {}
        for article in sorted(self._articles_cache, key=lambda a: a.timestamp, reverse=True):
            headline_key = article.headline.lower().strip()
            if headline_key not in seen_headlines:
                seen_headlines[headline_key] = article
        
        self._articles_cache = list(seen_headlines.values())
        
        # If still too many, keep only the most recent
        if len(self._articles_cache) > max_articles:
            self._articles_cache = sorted(
                self._articles_cache, 
                key=lambda a: a.timestamp, 
                reverse=True
            )[:max_articles]
        
        logger.info(f"Cache cleanup: {len(self._articles_cache)} articles retained")
    
    def deduplicate_cache(self):
        """
        Manually run deduplication on existing cache.
        Call this to clean up any existing duplicates.
        """
        original_count = len(self._articles_cache)
        
        # Deduplicate by ID
        seen_ids = set()
        unique_by_id = []
        for article in self._articles_cache:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique_by_id.append(article)
        
        # Deduplicate by headline (keep newest)
        seen_headlines = {}
        for article in sorted(unique_by_id, key=lambda a: a.timestamp, reverse=True):
            headline_key = article.headline.lower().strip()
            if headline_key not in seen_headlines:
                seen_headlines[headline_key] = article
        
        self._articles_cache = list(seen_headlines.values())
        
        removed = original_count - len(self._articles_cache)
        if removed > 0:
            self._save_cache()
            logger.info(f"Removed {removed} duplicate articles, {len(self._articles_cache)} remain")
        
        return removed
    
    def fetch_market_news(
        self,
        days_back: int = 7,
    ) -> List[AlphaVantageArticle]:
        """
        Fetch general market/financial news.
        
        Args:
            days_back: Days of history to fetch
            
        Returns:
            List of articles
        """
        # RATE LIMIT CHECK: If rate limited, return cached data immediately
        if self._rate_limited and self._articles_cache:
            logger.info(f"Alpha Vantage rate limited - returning {len(self._articles_cache)} cached articles")
            return self._articles_cache
        
        all_articles = []
        
        # Fetch by relevant topics
        market_topics = [
            'financial_markets',
            'economy_macro',
            'economy_monetary',
            'economy_fiscal',
            'earnings',
        ]
        
        time_from = datetime.now() - timedelta(days=days_back)
        time_to = datetime.now()
        
        # Fetch each topic (with rate limiting)
        for topic in market_topics:
            logger.info(f"Fetching Alpha Vantage news: {topic}")
            articles = self.fetch_news(
                topics=[topic],
                time_from=time_from,
                time_to=time_to,
                limit=50,
            )
            all_articles.extend(articles)
            
            # Only one API call for free tier
            if self.api_calls_made >= 5:
                logger.warning("Alpha Vantage daily limit approaching, stopping")
                break
        
        # Deduplicate
        seen_ids = set()
        unique = []
        for article in all_articles:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique.append(article)
        
        logger.info(f"Fetched {len(unique)} unique articles from Alpha Vantage")
        return unique
    
    def fetch_ticker_news(
        self,
        symbols: List[str],
        days_back: int = 7,
    ) -> List[AlphaVantageArticle]:
        """
        Fetch news for specific ticker symbols.
        
        Args:
            symbols: List of ticker symbols
            days_back: Days of history
            
        Returns:
            List of articles
        """
        # RATE LIMIT CHECK: If rate limited, return cached data immediately
        if self._rate_limited and self._articles_cache:
            # Filter cached articles by requested symbols
            relevant = [a for a in self._articles_cache 
                       if any(s in [ts.ticker for ts in a.ticker_sentiments] for s in symbols)]
            if relevant:
                logger.info(f"Alpha Vantage rate limited - returning {len(relevant)} cached articles for tickers")
                return relevant
            # Fall back to all cached if no ticker matches
            logger.info(f"Alpha Vantage rate limited - returning {len(self._articles_cache)} cached articles")
            return self._articles_cache
        
        time_from = datetime.now() - timedelta(days=days_back)
        time_to = datetime.now()
        
        # Batch tickers (5 at a time due to API limits)
        all_articles = []
        
        for i in range(0, min(len(symbols), 15), 5):  # Limit total calls
            batch = symbols[i:i+5]
            
            logger.info(f"Fetching Alpha Vantage news for: {', '.join(batch)}")
            articles = self.fetch_news(
                tickers=batch,
                time_from=time_from,
                time_to=time_to,
                limit=50,
            )
            all_articles.extend(articles)
            
            # Rate limit awareness
            if self.api_calls_made >= 5:
                logger.warning("Alpha Vantage daily limit approaching, stopping")
                break
        
        return all_articles
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'total_articles': len(self._articles_cache),
            'api_calls_made': self.api_calls_made,
            'cache_hits': self.cache_hits,
            'oldest_article': min((a.timestamp for a in self._articles_cache), default=None),
            'newest_article': max((a.timestamp for a in self._articles_cache), default=None),
        }
    
    def clear_cache(self):
        """Clear the article cache."""
        self._articles_cache = []
        self._save_cache()
        
        # Clear response cache files
        for f in self.cache_dir.glob("response_*.json"):
            f.unlink()
        
        logger.info("Cleared Alpha Vantage cache")
    
    def reset_rate_limit(self):
        """
        Force reset the rate limit flag.
        Use this when you know the daily quota has reset (midnight UTC).
        """
        self._rate_limited = False
        self._rate_limit_message = ""
        self._save_fetch_status()
        logger.info("🔄 Force reset Alpha Vantage rate limit - ready to fetch fresh news")
        return True
    
    def force_refresh(self, days_back: int = 3) -> List[AlphaVantageArticle]:
        """
        Force refresh news, ignoring rate limit flag.
        Will reset rate limit and attempt to fetch fresh articles.
        
        Returns:
            List of fresh articles, or empty if still rate limited by API
        """
        # Reset rate limit
        self.reset_rate_limit()
        
        # Try to fetch fresh news
        articles = self.fetch_market_news(days_back=days_back)
        
        if articles:
            logger.info(f"✅ Force refresh successful: {len(articles)} fresh articles")
        else:
            logger.warning("⚠️ Force refresh failed - API may still be rate limiting")
        
        return articles
    
    def get_cached_articles(self) -> List[AlphaVantageArticle]:
        """Get all cached articles."""
        return self._articles_cache.copy()
