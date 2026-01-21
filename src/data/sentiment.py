"""
Sentiment analysis for news and text data.
Provides entity-level sentiment with topic tags.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import re
import logging

from .news_data import NewsArticle

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment score for an asset at a point in time."""
    symbol: str
    timestamp: datetime
    sentiment_score: float  # [-1, 1]
    sentiment_volatility: float  # Std of recent scores
    sentiment_delta: float  # Change from previous period
    article_count: int
    topics: List[str]
    confidence: float  # Based on article count and agreement


class SentimentAnalyzer:
    """
    Analyze sentiment from news articles.
    Uses lexicon-based approach with optional transformer upgrade.
    """
    
    # Positive and negative word lists (simplified)
    POSITIVE_WORDS = {
        'beat', 'beats', 'exceeded', 'exceeds', 'strong', 'surge', 'surges',
        'gain', 'gains', 'rise', 'rises', 'up', 'higher', 'growth', 'grows',
        'profit', 'profits', 'bullish', 'upgrade', 'upgrades', 'buy',
        'outperform', 'optimistic', 'positive', 'record', 'high', 'best',
        'breakthrough', 'innovation', 'success', 'successful', 'win', 'wins',
        'expand', 'expansion', 'boost', 'boosts', 'rally', 'rallies',
    }
    
    NEGATIVE_WORDS = {
        'miss', 'misses', 'missed', 'weak', 'decline', 'declines', 'fall',
        'falls', 'drop', 'drops', 'down', 'lower', 'loss', 'losses',
        'bearish', 'downgrade', 'downgrades', 'sell', 'underperform',
        'pessimistic', 'negative', 'low', 'worst', 'concern', 'concerns',
        'risk', 'risks', 'warning', 'warns', 'cut', 'cuts', 'layoff',
        'layoffs', 'restructuring', 'lawsuit', 'investigation', 'fraud',
        'crash', 'crashes', 'plunge', 'plunges', 'recession', 'bankruptcy',
    }
    
    # Intensifiers and negators
    INTENSIFIERS = {'very', 'extremely', 'significantly', 'substantially', 'major'}
    NEGATORS = {'not', 'no', 'never', 'without', "n't", 'barely', 'hardly'}
    
    def __init__(
        self,
        decay_halflife_days: float = 3.0,
        use_transformer: bool = False
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            decay_halflife_days: Half-life for exponential decay of old articles
            use_transformer: Whether to use transformer model (if available)
        """
        self.decay_halflife = decay_halflife_days
        self.use_transformer = use_transformer
        self._transformer_model = None
        
        if use_transformer:
            self._load_transformer()
    
    def _load_transformer(self):
        """Load transformer model for sentiment (optional)."""
        try:
            from transformers import pipeline
            self._transformer_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True
            )
            logger.info("Loaded FinBERT transformer model")
        except Exception as e:
            logger.warning(f"Could not load transformer: {e}")
            self._transformer_model = None
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score in [-1, 1]
        """
        if self._transformer_model is not None:
            return self._analyze_transformer(text)
        else:
            return self._analyze_lexicon(text)
    
    def _analyze_lexicon(self, text: str) -> float:
        """Lexicon-based sentiment analysis."""
        words = text.lower().split()
        
        positive_count = 0
        negative_count = 0
        
        for i, word in enumerate(words):
            # Clean word
            word = re.sub(r'[^\w]', '', word)
            
            # Check for negation in previous 3 words
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in self.NEGATORS:
                    negated = True
                    break
            
            # Check for intensifier
            intensified = False
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                intensified = True
            
            multiplier = 1.5 if intensified else 1.0
            
            if word in self.POSITIVE_WORDS:
                if negated:
                    negative_count += multiplier
                else:
                    positive_count += multiplier
            elif word in self.NEGATIVE_WORDS:
                if negated:
                    positive_count += multiplier
                else:
                    negative_count += multiplier
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
            
        score = (positive_count - negative_count) / total
        return np.clip(score, -1, 1)
    
    def _analyze_transformer(self, text: str) -> float:
        """Transformer-based sentiment analysis."""
        try:
            result = self._transformer_model(text[:512])  # Truncate for model
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:
                return 0.0
        except Exception as e:
            logger.debug(f"Transformer failed: {e}")
            return self._analyze_lexicon(text)
    
    def aggregate_sentiment(
        self,
        articles: List[NewsArticle],
        symbols: List[str],
        as_of_date: datetime,
        lookback_days: int = 7
    ) -> Dict[str, SentimentScore]:
        """
        Aggregate sentiment scores per symbol with exponential decay.
        
        Args:
            articles: List of news articles
            symbols: Symbols to compute sentiment for
            as_of_date: Reference date (no look-ahead)
            lookback_days: Days to look back
            
        Returns:
            Dict of symbol -> SentimentScore
        """
        results = {}
        
        cutoff_date = as_of_date - pd.Timedelta(days=lookback_days)
        
        for symbol in symbols:
            # Filter articles for this symbol
            symbol_articles = [
                a for a in articles
                if symbol in a.tickers
                and cutoff_date <= a.timestamp <= as_of_date
            ]
            
            if not symbol_articles:
                results[symbol] = SentimentScore(
                    symbol=symbol,
                    timestamp=as_of_date,
                    sentiment_score=0.0,
                    sentiment_volatility=0.0,
                    sentiment_delta=0.0,
                    article_count=0,
                    topics=[],
                    confidence=0.0,
                )
                continue
            
            # Compute sentiment for each article
            scores = []
            weights = []
            all_topics = []
            
            for article in symbol_articles:
                # Use cached sentiment or compute
                if article.raw_sentiment is not None:
                    score = article.raw_sentiment
                else:
                    text = f"{article.headline} {article.body or ''}"
                    score = self.analyze_text(text)
                    article.raw_sentiment = score
                
                # Exponential decay weight
                days_ago = (as_of_date - article.timestamp).total_seconds() / 86400
                weight = np.exp(-np.log(2) * days_ago / self.decay_halflife)
                
                scores.append(score)
                weights.append(weight)
                all_topics.extend(article.topics)
            
            scores = np.array(scores)
            weights = np.array(weights)
            weights /= weights.sum()  # Normalize
            
            # Weighted average
            sentiment_score = np.average(scores, weights=weights)
            
            # Weighted std
            sentiment_volatility = np.sqrt(np.average((scores - sentiment_score)**2, weights=weights))
            
            # Delta: compare to previous period
            prev_cutoff = cutoff_date - pd.Timedelta(days=lookback_days)
            prev_articles = [
                a for a in articles
                if symbol in a.tickers
                and prev_cutoff <= a.timestamp < cutoff_date
            ]
            
            if prev_articles:
                prev_scores = []
                for article in prev_articles:
                    if article.raw_sentiment is not None:
                        prev_scores.append(article.raw_sentiment)
                    else:
                        text = f"{article.headline} {article.body or ''}"
                        prev_scores.append(self.analyze_text(text))
                prev_sentiment = np.mean(prev_scores)
                sentiment_delta = sentiment_score - prev_sentiment
            else:
                sentiment_delta = 0.0
            
            # Confidence based on article count and agreement
            agreement = 1.0 - sentiment_volatility
            count_factor = min(1.0, len(symbol_articles) / 10.0)
            confidence = agreement * count_factor
            
            # Topic aggregation
            unique_topics = list(set(all_topics))
            
            results[symbol] = SentimentScore(
                symbol=symbol,
                timestamp=as_of_date,
                sentiment_score=sentiment_score,
                sentiment_volatility=sentiment_volatility,
                sentiment_delta=sentiment_delta,
                article_count=len(symbol_articles),
                topics=unique_topics,
                confidence=confidence,
            )
        
        return results
