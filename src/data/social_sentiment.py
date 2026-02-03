"""
Social Sentiment Data Loader.
Fetches sentiment from Reddit (WSB), StockTwits, and other social sources.
Uses public APIs and web scraping with rate limiting.
"""
import re
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class SocialMention:
    """A single mention of a stock on social media."""
    symbol: str
    source: str  # 'reddit', 'stocktwits', 'twitter'
    text: str
    sentiment: float  # -1 to 1
    timestamp: datetime
    engagement: int  # Upvotes, likes, retweets
    url: Optional[str] = None


@dataclass
class SocialSentimentData:
    """Aggregated social sentiment for a symbol."""
    symbol: str
    mention_count: int
    avg_sentiment: float
    sentiment_std: float
    total_engagement: int
    mentions: List[SocialMention]
    is_trending: bool
    momentum: float  # Change in mentions vs previous period
    last_updated: datetime


class SocialSentimentLoader:
    """
    Loads social sentiment data from multiple sources.
    
    Sources:
    1. Reddit (WSB, stocks, investing, options)
    2. StockTwits (if API available)
    3. Twitter/X (limited without API)
    
    Uses sentiment analysis on text to score mentions.
    """
    
    # Subreddits to monitor
    REDDIT_SUBS = [
        'wallstreetbets',
        'stocks',
        'investing',
        'options',
        'pennystocks',
        'stockmarket',
    ]
    
    # Common ticker patterns to extract
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})|\b([A-Z]{2,5})\b')
    
    # Words that aren't tickers
    NOT_TICKERS = {
        'A', 'I', 'DD', 'OP', 'CEO', 'CFO', 'IPO', 'EPS', 'PE', 'ATH', 'ATL',
        'FOMO', 'YOLO', 'FD', 'ITM', 'OTM', 'ATM', 'DTE', 'WSB', 'SEC', 'FED',
        'GDP', 'CPI', 'THE', 'AND', 'FOR', 'YOU', 'ARE', 'BUT', 'NOT', 'ALL',
        'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'HAD', 'HOT', 'NEW',
        'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'GET', 'HAS', 'HIM',
        'HIS', 'HOW', 'MAN', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'USA', 'BUY',
        'ETF', 'LOL', 'IMO', 'IME', 'EOD', 'EOW', 'FYI', 'AI', 'EV', 'EPS',
    }
    
    # Valid tickers (major ones)
    VALID_TICKERS = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
        'AMD', 'INTC', 'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA', 'PYPL',
        'XOM', 'CVX', 'COP', 'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY',
        'KO', 'PEP', 'PG', 'WMT', 'COST', 'HD', 'MCD', 'NKE', 'DIS',
        'NFLX', 'BA', 'CAT', 'HON', 'UPS', 'T', 'VZ', 'TMUS',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
        'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'SOFI', 'RIVN', 'LCID',
        'COIN', 'HOOD', 'SQ', 'ROKU', 'SNOW', 'CRWD', 'ZS',
    }
    
    # Sentiment keywords
    BULLISH_WORDS = [
        'buy', 'calls', 'moon', 'rocket', '🚀', 'bullish', 'long', 'yolo',
        'diamond hands', '💎', 'squeeze', 'undervalued', 'breakout', 'buy the dip',
        'btfd', 'tendies', 'printer', 'ath', 'pump', 'gains', 'green', 'upside',
    ]
    
    BEARISH_WORDS = [
        'sell', 'puts', 'crash', 'dump', 'bearish', 'short', 'overvalued',
        'bubble', 'bag holder', '📉', 'red', 'loss', 'losses', 'down', 'drop',
        'falling', 'tank', 'drill', 'cave', 'bear', 'recession', 'correction',
    ]
    
    def __init__(
        self,
        cache_dir: str = "outputs/social_cache",
        cache_minutes: int = 30,
        rate_limit_seconds: float = 2.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_minutes = cache_minutes
        self.rate_limit_seconds = rate_limit_seconds
        
        self._cache: Dict[str, SocialSentimentData] = {}
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        
        # Load cached data
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached sentiment data."""
        cache_file = self.cache_dir / "social_sentiment.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Only load recent data
                    cutoff = datetime.now() - timedelta(hours=4)
                    for symbol, values in data.items():
                        last_updated = datetime.fromisoformat(values.get('last_updated', '2000-01-01'))
                        if last_updated > cutoff:
                            self._cache[symbol] = self._dict_to_sentiment_data(symbol, values)
                logger.info(f"Loaded {len(self._cache)} cached social sentiments")
            except Exception as e:
                logger.warning(f"Failed to load social cache: {e}")
    
    def _save_cache(self) -> None:
        """Save sentiment cache to disk."""
        cache_file = self.cache_dir / "social_sentiment.json"
        try:
            data = {}
            for symbol, sd in self._cache.items():
                data[symbol] = {
                    'mention_count': sd.mention_count,
                    'avg_sentiment': sd.avg_sentiment,
                    'sentiment_std': sd.sentiment_std,
                    'total_engagement': sd.total_engagement,
                    'is_trending': sd.is_trending,
                    'momentum': sd.momentum,
                    'last_updated': sd.last_updated.isoformat(),
                }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save social cache: {e}")
    
    def _dict_to_sentiment_data(self, symbol: str, values: dict) -> SocialSentimentData:
        """Convert dict to SocialSentimentData."""
        return SocialSentimentData(
            symbol=symbol,
            mention_count=values.get('mention_count', 0),
            avg_sentiment=values.get('avg_sentiment', 0.0),
            sentiment_std=values.get('sentiment_std', 0.5),
            total_engagement=values.get('total_engagement', 0),
            mentions=[],
            is_trending=values.get('is_trending', False),
            momentum=values.get('momentum', 0.0),
            last_updated=datetime.fromisoformat(values.get('last_updated', datetime.now().isoformat())),
        )
    
    def get_sentiment(
        self,
        symbols: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, SocialSentimentData]:
        """
        Get social sentiment for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            force_refresh: Force fetch even if cached
            
        Returns:
            Dict mapping symbol to SocialSentimentData
        """
        results = {}
        to_fetch = []
        
        now = datetime.now()
        with self._lock:
            for symbol in symbols:
                if symbol in self._cache and not force_refresh:
                    cached = self._cache[symbol]
                    age = now - cached.last_updated
                    if age < timedelta(minutes=self.cache_minutes):
                        results[symbol] = cached
                        continue
                to_fetch.append(symbol)
        
        # Fetch missing data
        if to_fetch:
            fetched = self._fetch_reddit_sentiment(to_fetch)
            results.update(fetched)
            
            # Update cache
            with self._lock:
                self._cache.update(fetched)
                self._save_cache()
        
        return results
    
    def _fetch_reddit_sentiment(self, symbols: List[str]) -> Dict[str, SocialSentimentData]:
        """Fetch sentiment from Reddit using public JSON endpoints."""
        results = {}
        
        # Aggregate mentions for each symbol
        mentions_by_symbol: Dict[str, List[SocialMention]] = {s: [] for s in symbols}
        
        for subreddit in self.REDDIT_SUBS[:3]:  # Limit to avoid rate limiting
            try:
                self._rate_limit()
                posts = self._fetch_subreddit_posts(subreddit)
                
                for post in posts:
                    extracted = self._extract_tickers(post.get('title', '') + ' ' + post.get('selftext', ''))
                    
                    for ticker in extracted:
                        if ticker in symbols:
                            sentiment = self._analyze_sentiment(
                                post.get('title', '') + ' ' + post.get('selftext', '')
                            )
                            
                            mention = SocialMention(
                                symbol=ticker,
                                source='reddit',
                                text=post.get('title', '')[:200],
                                sentiment=sentiment,
                                timestamp=datetime.fromtimestamp(post.get('created_utc', 0)),
                                engagement=post.get('ups', 0) + post.get('num_comments', 0),
                                url=f"https://reddit.com{post.get('permalink', '')}",
                            )
                            mentions_by_symbol[ticker].append(mention)
                
            except Exception as e:
                logger.warning(f"Failed to fetch from r/{subreddit}: {e}")
                continue
        
        # Aggregate results
        for symbol in symbols:
            mentions = mentions_by_symbol.get(symbol, [])
            
            if not mentions:
                # No mentions found - create neutral sentiment
                results[symbol] = SocialSentimentData(
                    symbol=symbol,
                    mention_count=0,
                    avg_sentiment=0.0,
                    sentiment_std=0.5,
                    total_engagement=0,
                    mentions=[],
                    is_trending=False,
                    momentum=0.0,
                    last_updated=datetime.now(),
                )
            else:
                sentiments = [m.sentiment for m in mentions]
                total_engagement = sum(m.engagement for m in mentions)
                
                results[symbol] = SocialSentimentData(
                    symbol=symbol,
                    mention_count=len(mentions),
                    avg_sentiment=float(sum(sentiments) / len(sentiments)),
                    sentiment_std=float(max(0.1, (sum((s - sum(sentiments)/len(sentiments))**2 for s in sentiments) / len(sentiments))**0.5)),
                    total_engagement=total_engagement,
                    mentions=mentions[:20],  # Keep top 20
                    is_trending=len(mentions) > 10 or total_engagement > 1000,
                    momentum=len(mentions) / 10.0,  # Simplified momentum
                    last_updated=datetime.now(),
                )
        
        return results
    
    def _fetch_subreddit_posts(self, subreddit: str) -> List[dict]:
        """Fetch recent posts from a subreddit using public JSON API."""
        try:
            import urllib.request
            import ssl
            
            # Create SSL context that doesn't verify (for compatibility)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
            
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'TradingBot/1.0'}
            )
            
            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                data = json.loads(response.read().decode('utf-8'))
                return [post['data'] for post in data.get('data', {}).get('children', [])]
            
        except Exception as e:
            logger.debug(f"Failed to fetch r/{subreddit}: {e}")
            return []
    
    def _rate_limit(self) -> None:
        """Rate limit requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_time = time.time()
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract valid ticker symbols from text."""
        matches = self.TICKER_PATTERN.findall(text)
        
        tickers = set()
        for match in matches:
            # Each match is a tuple (group1, group2)
            ticker = match[0] or match[1]
            if ticker and ticker.upper() not in self.NOT_TICKERS:
                ticker_upper = ticker.upper()
                # Only include known valid tickers or ones that look like tickers
                if ticker_upper in self.VALID_TICKERS:
                    tickers.add(ticker_upper)
        
        return list(tickers)
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using keyword matching."""
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in self.BULLISH_WORDS if word in text_lower)
        bearish_count = sum(1 for word in self.BEARISH_WORDS if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        # Sentiment from -1 to 1
        sentiment = (bullish_count - bearish_count) / total
        return sentiment
    
    def get_trending_tickers(self, min_mentions: int = 5) -> List[str]:
        """Get list of trending tickers based on social activity."""
        with self._lock:
            trending = [
                (sym, data.mention_count, data.total_engagement)
                for sym, data in self._cache.items()
                if data.mention_count >= min_mentions
            ]
        
        # Sort by engagement
        trending.sort(key=lambda x: x[2], reverse=True)
        return [t[0] for t in trending[:20]]


# Singleton instance
_social_loader: Optional[SocialSentimentLoader] = None


def get_social_sentiment_loader() -> SocialSentimentLoader:
    """Get singleton SocialSentimentLoader instance."""
    global _social_loader
    if _social_loader is None:
        _social_loader = SocialSentimentLoader()
    return _social_loader
