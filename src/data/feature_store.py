"""
Feature store for strategy inputs.
Produces timestamped features with strict alignment to prevent look-ahead bias.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging

from .market_data import MarketDataLoader
from .news_data import NewsDataLoader, NewsArticle
from .sentiment import SentimentAnalyzer, SentimentScore
from .regime import RegimeClassifier, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class Features:
    """
    Complete feature set for a given timestamp.
    All features are point-in-time (no look-ahead).
    """
    timestamp: datetime
    
    # Price features
    prices: Dict[str, float] = field(default_factory=dict)
    returns_1d: Dict[str, float] = field(default_factory=dict)
    returns_5d: Dict[str, float] = field(default_factory=dict)
    returns_21d: Dict[str, float] = field(default_factory=dict)
    returns_63d: Dict[str, float] = field(default_factory=dict)
    returns_126d: Dict[str, float] = field(default_factory=dict)
    returns_252d: Dict[str, float] = field(default_factory=dict)
    
    # INTRADAY FEATURES (Critical for HFT-lite strategies)
    intraday_returns: Dict[str, float] = field(default_factory=dict)  # Last 15-30 min return
    volume_ratio: Dict[str, float] = field(default_factory=dict)  # Current vol vs average
    vwap: Dict[str, float] = field(default_factory=dict)  # Current VWAP
    vwap_deviation: Dict[str, float] = field(default_factory=dict)  # Price deviation from VWAP
    opening_high: Dict[str, float] = field(default_factory=dict)  # First 30-min high
    opening_low: Dict[str, float] = field(default_factory=dict)  # First 30-min low
    
    # Volatility features
    volatility_21d: Dict[str, float] = field(default_factory=dict)
    volatility_63d: Dict[str, float] = field(default_factory=dict)
    
    # Moving averages
    ma_20: Dict[str, float] = field(default_factory=dict)
    ma_50: Dict[str, float] = field(default_factory=dict)
    ma_200: Dict[str, float] = field(default_factory=dict)
    
    # Technical indicators for intraday
    rsi_14: Dict[str, float] = field(default_factory=dict)  # 14-period RSI
    
    # Volume profile (for intraday strategies)
    volume_profile: Dict[str, Dict[str, float]] = field(default_factory=dict)  # symbol -> {avg, ratio, trend}
    
    # Regime
    regime: Optional[MarketRegime] = None
    
    # Sentiment
    sentiment: Dict[str, SentimentScore] = field(default_factory=dict)
    
    # Universe
    symbols: List[str] = field(default_factory=list)
    
    # Correlation matrix
    correlation_matrix: Optional[pd.DataFrame] = None
    
    # Covariance matrix
    covariance_matrix: Optional[pd.DataFrame] = None
    
    # Macro features from News Intelligence Pipeline
    macro_features: Optional[Any] = None  # DailyMacroFeatures
    risk_sentiment: Optional[Any] = None  # RiskSentiment
    
    # Flag indicating if intraday data is available
    has_intraday_data: bool = False


class FeatureStore:
    """
    Central feature computation and storage.
    Ensures all features are timestamp-aligned with no look-ahead.
    """
    
    def __init__(
        self,
        market_loader: MarketDataLoader,
        news_loader: Optional[NewsDataLoader] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        regime_classifier: Optional[RegimeClassifier] = None,
    ):
        """
        Initialize feature store.
        
        Args:
            market_loader: Market data loader
            news_loader: Optional news data loader
            sentiment_analyzer: Optional sentiment analyzer
            regime_classifier: Optional regime classifier
        """
        self.market_loader = market_loader
        self.news_loader = news_loader or NewsDataLoader()
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.regime_classifier = regime_classifier or RegimeClassifier()
        
        # Cache
        self._price_history: Dict[str, pd.DataFrame] = {}
        self._news_cache: List[NewsArticle] = []
        self._feature_cache: Dict[str, Features] = {}
        
    def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        source: str = "alpaca"
    ) -> None:
        """
        Load all required data into cache.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            source: Data source
        """
        logger.info(f"Loading data for {len(symbols)} symbols...")
        
        # Load price data with extra history for features
        history_start = start_date - pd.Timedelta(days=400)
        
        self._price_history = self.market_loader.load_ohlcv(
            symbols=symbols,
            start_date=history_start,
            end_date=end_date,
            source=source
        )
        
        logger.info(f"Loaded price data for {len(self._price_history)} symbols")
        
        # Load news data
        try:
            self._news_cache = self.news_loader.load_news(start_date, end_date)
            logger.info(f"Loaded {len(self._news_cache)} news articles")
        except Exception as e:
            logger.warning(f"Could not load news: {e}")
            self._news_cache = []
    
    def get_features(
        self,
        as_of: datetime,
        symbols: Optional[List[str]] = None
    ) -> Features:
        """
        Get features as of a specific timestamp.
        
        Args:
            as_of: Point-in-time date
            symbols: Optional subset of symbols
            
        Returns:
            Features object with all computed features
        """
        cache_key = str(as_of)
        
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        if symbols is None:
            symbols = list(self._price_history.keys())
        
        features = Features(timestamp=as_of, symbols=symbols)
        
        # Build price DataFrame up to as_of
        price_df = self._build_price_df(symbols, as_of)
        
        if len(price_df) == 0:
            return features
        
        # Compute features
        self._compute_price_features(features, price_df, symbols)
        self._compute_volatility_features(features, price_df, symbols)
        self._compute_ma_features(features, price_df, symbols)
        self._compute_rsi_features(features, price_df, symbols)  # RSI for intraday
        self._compute_correlation_features(features, price_df)
        
        # Regime classification
        try:
            features.regime = self.regime_classifier.classify(price_df, as_of=pd.Timestamp(as_of))
        except Exception as e:
            logger.warning(f"Regime classification failed: {e}")
        
        # Sentiment
        try:
            features.sentiment = self.sentiment_analyzer.aggregate_sentiment(
                self._news_cache, symbols, as_of
            )
        except Exception as e:
            logger.warning(f"Sentiment computation failed: {e}")
        
        self._feature_cache[cache_key] = features
        return features
    
    def _build_price_df(
        self,
        symbols: List[str],
        as_of: datetime
    ) -> pd.DataFrame:
        """Build price DataFrame up to as_of date."""
        close_prices = {}
        
        # Convert as_of to timezone-aware if needed
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tz is None:
            as_of_ts = as_of_ts.tz_localize('UTC')
        
        for symbol in symbols:
            if symbol in self._price_history:
                df = self._price_history[symbol]
                
                # Handle timezone compatibility
                try:
                    if df.index.tz is None:
                        # Index is naive, use naive timestamp
                        df = df.loc[:as_of]
                    else:
                        # Index is tz-aware, use tz-aware timestamp
                        df = df.loc[:as_of_ts]
                except Exception:
                    # Fallback: just use all data
                    pass
                
                if 'close' in df.columns and len(df) > 0:
                    close_prices[symbol] = df['close']
        
        if not close_prices:
            return pd.DataFrame()
            
        return pd.DataFrame(close_prices)
    
    def _compute_price_features(
        self,
        features: Features,
        price_df: pd.DataFrame,
        symbols: List[str]
    ) -> None:
        """Compute price and return features."""
        for symbol in symbols:
            if symbol not in price_df.columns:
                continue
                
            prices = price_df[symbol].dropna()
            if len(prices) == 0:
                continue
            
            features.prices[symbol] = prices.iloc[-1]
            
            # Returns at various horizons
            for period, attr in [
                (1, 'returns_1d'),
                (5, 'returns_5d'),
                (21, 'returns_21d'),
                (63, 'returns_63d'),
                (126, 'returns_126d'),
                (252, 'returns_252d'),
            ]:
                if len(prices) > period:
                    ret = (prices.iloc[-1] / prices.iloc[-period-1]) - 1
                    getattr(features, attr)[symbol] = ret
    
    def _compute_volatility_features(
        self,
        features: Features,
        price_df: pd.DataFrame,
        symbols: List[str]
    ) -> None:
        """Compute volatility features."""
        returns = price_df.pct_change().dropna()
        
        for symbol in symbols:
            if symbol not in returns.columns:
                continue
                
            rets = returns[symbol].dropna()
            
            if len(rets) >= 21:
                features.volatility_21d[symbol] = rets.tail(21).std() * np.sqrt(252)
            if len(rets) >= 63:
                features.volatility_63d[symbol] = rets.tail(63).std() * np.sqrt(252)
    
    def _compute_ma_features(
        self,
        features: Features,
        price_df: pd.DataFrame,
        symbols: List[str]
    ) -> None:
        """Compute moving average features."""
        for symbol in symbols:
            if symbol not in price_df.columns:
                continue
                
            prices = price_df[symbol].dropna()
            
            if len(prices) >= 20:
                features.ma_20[symbol] = prices.tail(20).mean()
            if len(prices) >= 50:
                features.ma_50[symbol] = prices.tail(50).mean()
            if len(prices) >= 200:
                features.ma_200[symbol] = prices.tail(200).mean()
    
    def _compute_rsi_features(
        self,
        features: Features,
        price_df: pd.DataFrame,
        symbols: List[str],
        period: int = 14
    ) -> None:
        """
        Compute RSI (Relative Strength Index) for all symbols.
        
        RSI = 100 - (100 / (1 + RS))
        Where RS = Average Gain / Average Loss over the period
        
        RSI < 30 = Oversold (buy signal)
        RSI > 70 = Overbought (sell signal)
        """
        for symbol in symbols:
            if symbol not in price_df.columns:
                continue
            
            prices = price_df[symbol].dropna()
            if len(prices) < period + 1:
                continue
            
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate average gain/loss (using EMA for smoothing)
            avg_gain = gains.ewm(span=period, adjust=False).mean()
            avg_loss = losses.ewm(span=period, adjust=False).mean()
            
            # Avoid division by zero
            avg_loss = avg_loss.replace(0, 0.001)
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Store latest RSI value
            features.rsi_14[symbol] = rsi.iloc[-1]
    
    def _compute_correlation_features(
        self,
        features: Features,
        price_df: pd.DataFrame,
        window: int = 63
    ) -> None:
        """Compute correlation and covariance matrices."""
        returns = price_df.pct_change().dropna()
        
        if len(returns) < window:
            return
        
        recent_returns = returns.tail(window)
        
        features.correlation_matrix = recent_returns.corr()
        features.covariance_matrix = recent_returns.cov() * 252  # Annualized
    
    def add_intraday_features(
        self,
        features: Features,
        symbols: List[str],
        timeframe: str = "15Min",
    ) -> Features:
        """
        Add intraday features to an existing Features object.
        
        This is CRITICAL for HFT-lite strategies - without this,
        intraday strategies fall back to daily data and are essentially random.
        
        Args:
            features: Existing Features object
            symbols: Symbols to get intraday data for
            timeframe: Intraday timeframe
            
        Returns:
            Features object with intraday data populated
        """
        try:
            intraday = self.market_loader.get_intraday_features(symbols, timeframe)
            
            if intraday and intraday.get('intraday_returns'):
                features.intraday_returns = intraday.get('intraday_returns', {})
                features.volume_ratio = intraday.get('volume_ratio', {})
                features.vwap = intraday.get('vwap', {})
                features.vwap_deviation = intraday.get('vwap_deviation', {})
                features.opening_high = intraday.get('opening_high', {})
                features.opening_low = intraday.get('opening_low', {})
                features.has_intraday_data = True
                
                logger.info(f"Added intraday features for {len(features.intraday_returns)} symbols")
            else:
                logger.warning("No intraday data available, strategies will use fallback")
                features.has_intraday_data = False
                
        except Exception as e:
            logger.warning(f"Could not add intraday features: {e}")
            features.has_intraday_data = False
        
        return features
    
    def get_rebalance_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "weekly"
    ) -> List[datetime]:
        """
        Get rebalance dates for backtesting.
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: 'daily', 'weekly', 'monthly'
            
        Returns:
            List of rebalance dates
        """
        if frequency == "daily":
            freq = "B"
        elif frequency == "weekly":
            freq = "W-FRI"
        elif frequency == "monthly":
            freq = "BM"
        else:
            freq = "B"
            
        dates = pd.date_range(start_date, end_date, freq=freq)
        return [d.to_pydatetime() for d in dates]
