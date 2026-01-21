"""
Ticker-Level Sentiment Aggregator.

This module aggregates article-level sentiment into per-stock daily features.
Uses Alpha Vantage's ticker_sentiment data which includes:
- Per-ticker relevance scores
- Per-ticker sentiment scores
- Sentiment labels

This is the KEY DATA we were previously ignoring!
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StockSentimentFeatures:
    """
    Aggregated sentiment features for a single stock.
    These are the features that feed into the trading strategies.
    """
    ticker: str
    
    # Core sentiment metrics
    sentiment_score: float  # Weighted average sentiment (-1 to +1)
    sentiment_confidence: float  # How confident we are (based on relevance and volume)
    
    # Volume metrics
    news_volume: int  # Number of articles mentioning this stock
    high_relevance_count: int  # Articles with relevance > 0.5
    
    # Momentum metrics
    sentiment_momentum: float  # Today vs 7-day average
    sentiment_acceleration: float  # Change in momentum
    
    # Dispersion metrics
    sentiment_dispersion: float  # Std dev of individual scores (disagreement)
    bullish_ratio: float  # % of articles that are bullish
    
    # Time decay weighted
    recent_sentiment: float  # Last 24h sentiment (higher weight on recent)
    
    # Metadata
    last_article_time: Optional[datetime] = None
    freshness_hours: float = 0.0  # Hours since last article


@dataclass
class TickerSentimentAggregator:
    """
    Aggregates article-level sentiment into stock-level features.
    
    Key Algorithm:
    - Weight each article's sentiment by its relevance_score
    - Apply exponential time decay (24h half-life)
    - Compute momentum by comparing current vs historical
    - Track disagreement through dispersion
    """
    
    # Settings
    decay_half_life_hours: float = 24.0  # Half-life for time decay
    min_relevance_threshold: float = 0.3  # Minimum relevance to include
    high_relevance_threshold: float = 0.5  # What counts as "high relevance"
    lookback_days: int = 7  # Days of history to consider
    
    # Storage
    _stock_sentiments: Dict[str, StockSentimentFeatures] = field(default_factory=dict)
    _cache_path: Optional[Path] = None
    
    def __post_init__(self):
        self._stock_sentiments = {}
        if self._cache_path:
            self._load_cache()
    
    def aggregate_from_articles(
        self,
        articles: List,  # List of AlphaVantageArticle
        as_of: datetime,
        universe: Optional[List[str]] = None,
    ) -> Dict[str, StockSentimentFeatures]:
        """
        Aggregate sentiment from articles into per-stock features.
        
        Args:
            articles: List of AlphaVantageArticle objects
            as_of: Current timestamp (for time decay calculation)
            universe: Optional list of tickers to focus on
        
        Returns:
            Dict mapping ticker to StockSentimentFeatures
        """
        # Collect all sentiment data points per ticker
        ticker_data: Dict[str, List[Dict]] = {}
        
        for article in articles:
            # Skip articles without detailed sentiment
            if not hasattr(article, 'ticker_sentiment_details'):
                continue
            
            article_time = article.timestamp
            if article_time.tzinfo is None:
                # Make timezone-naive for comparison
                pass
            
            # Calculate time weight (exponential decay)
            try:
                hours_ago = (as_of.replace(tzinfo=None) - article_time.replace(tzinfo=None)).total_seconds() / 3600
            except:
                hours_ago = 24  # Default to 24 hours if error
            
            if hours_ago < 0:
                hours_ago = 0
            
            time_weight = math.exp(-hours_ago / self.decay_half_life_hours * math.log(2))
            
            # Process each ticker mentioned in the article
            for ts in article.ticker_sentiment_details:
                ticker = ts.ticker
                
                # Filter by universe if provided
                if universe and ticker not in universe:
                    continue
                
                # Filter by minimum relevance
                if ts.relevance_score < self.min_relevance_threshold:
                    continue
                
                # Initialize ticker data if needed
                if ticker not in ticker_data:
                    ticker_data[ticker] = []
                
                # Add data point
                ticker_data[ticker].append({
                    'timestamp': article_time,
                    'relevance': ts.relevance_score,
                    'sentiment': ts.sentiment_score,
                    'label': ts.sentiment_label,
                    'time_weight': time_weight,
                    'combined_weight': ts.relevance_score * time_weight,
                    'hours_ago': hours_ago,
                })
        
        # Aggregate per ticker
        results = {}
        
        for ticker, data_points in ticker_data.items():
            if not data_points:
                continue
            
            results[ticker] = self._aggregate_ticker(ticker, data_points, as_of)
        
        # Store results
        self._stock_sentiments = results
        
        logger.info(f"Aggregated sentiment for {len(results)} stocks from {len(articles)} articles")
        
        return results
    
    def _aggregate_ticker(
        self,
        ticker: str,
        data_points: List[Dict],
        as_of: datetime,
    ) -> StockSentimentFeatures:
        """Aggregate data points for a single ticker."""
        
        # Calculate weighted sentiment
        total_weight = sum(dp['combined_weight'] for dp in data_points)
        if total_weight > 0:
            weighted_sentiment = sum(
                dp['sentiment'] * dp['combined_weight']
                for dp in data_points
            ) / total_weight
        else:
            weighted_sentiment = 0.0
        
        # Calculate confidence (based on total weight and count)
        news_volume = len(data_points)
        high_relevance_count = sum(1 for dp in data_points if dp['relevance'] >= self.high_relevance_threshold)
        
        # Confidence increases with volume and relevance, max at 1.0
        confidence = min(1.0, (total_weight / 3.0) * min(1.0, news_volume / 5.0))
        
        # Calculate sentiment dispersion (disagreement)
        if len(data_points) > 1:
            mean_sentiment = sum(dp['sentiment'] for dp in data_points) / len(data_points)
            variance = sum((dp['sentiment'] - mean_sentiment) ** 2 for dp in data_points) / len(data_points)
            dispersion = math.sqrt(variance)
        else:
            dispersion = 0.0
        
        # Calculate bullish ratio
        bullish_count = sum(1 for dp in data_points if 'bullish' in dp['label'].lower())
        bullish_ratio = bullish_count / len(data_points) if data_points else 0.5
        
        # Calculate recent sentiment (last 24h)
        recent_points = [dp for dp in data_points if dp['hours_ago'] <= 24]
        if recent_points:
            recent_weight = sum(dp['combined_weight'] for dp in recent_points)
            if recent_weight > 0:
                recent_sentiment = sum(
                    dp['sentiment'] * dp['combined_weight']
                    for dp in recent_points
                ) / recent_weight
            else:
                recent_sentiment = weighted_sentiment
        else:
            recent_sentiment = weighted_sentiment
        
        # Calculate momentum (recent vs older)
        older_points = [dp for dp in data_points if dp['hours_ago'] > 24]
        if older_points:
            older_weight = sum(dp['combined_weight'] for dp in older_points)
            if older_weight > 0:
                older_sentiment = sum(
                    dp['sentiment'] * dp['combined_weight']
                    for dp in older_points
                ) / older_weight
                momentum = recent_sentiment - older_sentiment
            else:
                momentum = 0.0
        else:
            momentum = 0.0
        
        # Find last article time
        last_article_time = max(dp['timestamp'] for dp in data_points) if data_points else None
        freshness_hours = min(dp['hours_ago'] for dp in data_points) if data_points else 999
        
        return StockSentimentFeatures(
            ticker=ticker,
            sentiment_score=weighted_sentiment,
            sentiment_confidence=confidence,
            news_volume=news_volume,
            high_relevance_count=high_relevance_count,
            sentiment_momentum=momentum,
            sentiment_acceleration=0.0,  # Would need historical to calculate
            sentiment_dispersion=dispersion,
            bullish_ratio=bullish_ratio,
            recent_sentiment=recent_sentiment,
            last_article_time=last_article_time,
            freshness_hours=freshness_hours,
        )
    
    def get_sentiment_features(self, ticker: str) -> Optional[StockSentimentFeatures]:
        """Get aggregated sentiment features for a ticker."""
        return self._stock_sentiments.get(ticker)
    
    def get_all_features(self) -> Dict[str, StockSentimentFeatures]:
        """Get all aggregated features."""
        return self._stock_sentiments.copy()
    
    def get_bullish_stocks(self, threshold: float = 0.2) -> List[str]:
        """Get list of stocks with bullish sentiment above threshold."""
        return [
            ticker for ticker, feat in self._stock_sentiments.items()
            if feat.sentiment_score >= threshold and feat.sentiment_confidence >= 0.3
        ]
    
    def get_bearish_stocks(self, threshold: float = -0.2) -> List[str]:
        """Get list of stocks with bearish sentiment below threshold."""
        return [
            ticker for ticker, feat in self._stock_sentiments.items()
            if feat.sentiment_score <= threshold and feat.sentiment_confidence >= 0.3
        ]
    
    def get_momentum_stocks(self, threshold: float = 0.1) -> List[str]:
        """Get stocks with improving sentiment momentum."""
        return [
            ticker for ticker, feat in self._stock_sentiments.items()
            if feat.sentiment_momentum >= threshold
        ]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API/UI."""
        result = {}
        for ticker, feat in self._stock_sentiments.items():
            result[ticker] = {
                'sentiment_score': feat.sentiment_score,
                'sentiment_confidence': feat.sentiment_confidence,
                'news_volume': feat.news_volume,
                'high_relevance_count': feat.high_relevance_count,
                'sentiment_momentum': feat.sentiment_momentum,
                'bullish_ratio': feat.bullish_ratio,
                'recent_sentiment': feat.recent_sentiment,
                'freshness_hours': feat.freshness_hours,
            }
        return result
    
    def print_summary(self, top_n: int = 10):
        """Print summary of top sentiment stocks."""
        if not self._stock_sentiments:
            print("No sentiment data available")
            return
        
        # Sort by sentiment score
        sorted_stocks = sorted(
            self._stock_sentiments.items(),
            key=lambda x: x[1].sentiment_score,
            reverse=True
        )
        
        print("\n" + "=" * 60)
        print("TICKER SENTIMENT SUMMARY")
        print("=" * 60)
        
        print("\n📈 TOP BULLISH:")
        for ticker, feat in sorted_stocks[:top_n]:
            conf_bar = "█" * int(feat.sentiment_confidence * 10)
            print(f"  {ticker:6} Sent: {feat.sentiment_score:+.2f} "
                  f"Conf: [{conf_bar:10}] "
                  f"Vol: {feat.news_volume}")
        
        print("\n📉 TOP BEARISH:")
        for ticker, feat in sorted_stocks[-top_n:]:
            conf_bar = "█" * int(feat.sentiment_confidence * 10)
            print(f"  {ticker:6} Sent: {feat.sentiment_score:+.2f} "
                  f"Conf: [{conf_bar:10}] "
                  f"Vol: {feat.news_volume}")
        
        print("\n🚀 MOMENTUM LEADERS:")
        momentum_sorted = sorted(
            self._stock_sentiments.items(),
            key=lambda x: x[1].sentiment_momentum,
            reverse=True
        )
        for ticker, feat in momentum_sorted[:5]:
            print(f"  {ticker:6} Mom: {feat.sentiment_momentum:+.2f} "
                  f"Recent: {feat.recent_sentiment:+.2f}")
        
        print("=" * 60)
