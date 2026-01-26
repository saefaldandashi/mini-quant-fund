"""
Market data loading and preprocessing.
Supports OHLCV data from CSV/parquet with adapters for external APIs.

Now includes intraday bar fetching for HFT-lite strategies.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """
    Load and preprocess market data with strict timestamp alignment.
    Avoids look-ahead bias by only exposing data up to the current timestamp.
    """
    
    def __init__(self, data_path: Optional[str] = None, cache: bool = True):
        """
        Initialize the market data loader.
        
        Args:
            data_path: Path to data directory or file
            cache: Whether to cache loaded data
        """
        self.data_path = Path(data_path) if data_path else None
        self.cache = cache
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._returns_cache: Dict[str, pd.Series] = {}
        
    def load_ohlcv(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        source: str = "csv"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            source: Data source ('csv', 'parquet', 'alpaca', 'yahoo')
            
        Returns:
            Dict mapping symbol to DataFrame with OHLCV columns
        """
        data = {}
        
        for symbol in symbols:
            try:
                if source == "csv":
                    df = self._load_csv(symbol, start_date, end_date)
                elif source == "parquet":
                    df = self._load_parquet(symbol, start_date, end_date)
                elif source == "alpaca":
                    df = self._load_alpaca(symbol, start_date, end_date)
                else:
                    raise ValueError(f"Unknown source: {source}")
                
                if df is not None and len(df) > 0:
                    data[symbol] = df
                    
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
                continue
                
        return data
    
    def _load_csv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load from CSV file."""
        if self.data_path is None:
            return None
            
        # Try multiple file patterns
        patterns = [
            self.data_path / f"{symbol}.csv",
            self.data_path / f"{symbol.lower()}.csv",
            self.data_path / "ohlcv" / f"{symbol}.csv",
        ]
        
        for path in patterns:
            if path.exists():
                df = pd.read_csv(path, parse_dates=['date'], index_col='date')
                df = df.loc[start_date:end_date]
                df.columns = df.columns.str.lower()
                return df
                
        return None
    
    def _load_parquet(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load from Parquet file."""
        if self.data_path is None:
            return None
            
        path = self.data_path / f"{symbol}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if 'date' in df.columns:
                df = df.set_index('date')
            df = df.loc[start_date:end_date]
            df.columns = df.columns.str.lower()
            return df
            
        return None
    
    def _load_alpaca(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load from Alpaca API (if configured)."""
        try:
            import os
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import DataFeed
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                return None
                
            client = StockHistoricalDataClient(api_key, secret_key)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX
            )
            
            bars = client.get_stock_bars(request)
            
            if bars and hasattr(bars, 'data') and symbol in bars.data:
                symbol_bars = bars.data[symbol]
                if symbol_bars:
                    data = {
                        'open': [b.open for b in symbol_bars],
                        'high': [b.high for b in symbol_bars],
                        'low': [b.low for b in symbol_bars],
                        'close': [b.close for b in symbol_bars],
                        'volume': [b.volume for b in symbol_bars],
                    }
                    timestamps = [b.timestamp for b in symbol_bars]
                    df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
                    df = df.sort_index()
                    return df
                    
        except Exception as e:
            logger.debug(f"Alpaca load failed for {symbol}: {e}")
            
        return None
    
    def get_returns(
        self,
        prices: Dict[str, pd.DataFrame],
        period: int = 1
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: Dict of symbol -> OHLCV DataFrame
            period: Return period in days
            
        Returns:
            DataFrame with returns for each symbol
        """
        returns_dict = {}
        
        for symbol, df in prices.items():
            if 'close' in df.columns:
                returns_dict[symbol] = df['close'].pct_change(period)
            elif 'adj_close' in df.columns:
                returns_dict[symbol] = df['adj_close'].pct_change(period)
                
        returns = pd.DataFrame(returns_dict)
        return returns.dropna(how='all')
    
    def get_risk_free_rate(self, date: datetime) -> float:
        """
        Get risk-free rate proxy for a given date.
        Default: use 0.0 or load from data if available.
        """
        # TODO: Load from FRED or data file
        return 0.0001  # ~2.5% annual as daily rate
    
    def load_intraday_bars(
        self,
        symbols: List[str],
        timeframe: str = "15Min",
        days_back: int = 1,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday bars from Alpaca.
        
        CRITICAL for HFT-lite strategies - provides real 15/30-min bars
        instead of falling back to daily data.
        
        Args:
            symbols: List of symbols
            timeframe: "1Min", "5Min", "15Min", "30Min", "1Hour"
            days_back: How many days of intraday data
        
        Returns:
            Dict of symbol -> DataFrame with OHLCV + VWAP columns
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            import pytz
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                logger.warning("No Alpaca API keys for intraday data")
                return {}
            
            # Map timeframe strings to Alpaca TimeFrame objects
            timeframe_map = {
                "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            }
            
            if timeframe not in timeframe_map:
                logger.warning(f"Unknown timeframe {timeframe}, using 15Min")
                timeframe = "15Min"
            
            client = StockHistoricalDataClient(api_key, secret_key)
            
            end = datetime.now(pytz.UTC)
            start = end - timedelta(days=days_back)
            
            from alpaca.data.enums import DataFeed
            
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe_map[timeframe],
                start=start,
                end=end,
                feed=DataFeed.IEX,  # Use IEX feed (free tier compatible)
            )
            
            bars = client.get_stock_bars(request)
            
            result = {}
            for symbol in symbols:
                if bars and hasattr(bars, 'data') and symbol in bars.data:
                    symbol_bars = bars.data[symbol]
                    if symbol_bars:
                        df = pd.DataFrame([
                            {
                                'timestamp': b.timestamp,
                                'open': b.open,
                                'high': b.high,
                                'low': b.low,
                                'close': b.close,
                                'volume': b.volume,
                                'vwap': getattr(b, 'vwap', b.close),  # VWAP if available
                            }
                            for b in symbol_bars
                        ])
                        df.set_index('timestamp', inplace=True)
                        df = df.sort_index()
                        result[symbol] = df
            
            logger.info(f"Loaded intraday {timeframe} bars for {len(result)} symbols")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to load intraday bars: {e}")
            return {}
    
    def get_intraday_features(
        self,
        symbols: List[str],
        timeframe: str = "15Min",
    ) -> Dict[str, Dict]:
        """
        Compute intraday features from intraday bars.
        
        Returns features needed by HFT-lite strategies:
        - intraday_returns: Return over the last timeframe period
        - volume_ratio: Current volume vs average
        - vwap: Current VWAP
        - vwap_deviation: Price deviation from VWAP
        - opening_high: High of first 30 minutes
        - opening_low: Low of first 30 minutes
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe for bars
        
        Returns:
            Dict with intraday features for each symbol
        """
        bars = self.load_intraday_bars(symbols, timeframe, days_back=1)
        
        features = {
            'intraday_returns': {},
            'volume_ratio': {},
            'vwap': {},
            'vwap_deviation': {},
            'opening_high': {},
            'opening_low': {},
            'current_prices': {},
        }
        
        for symbol, df in bars.items():
            if len(df) < 2:
                continue
            
            try:
                # Current price
                current_price = df['close'].iloc[-1]
                features['current_prices'][symbol] = current_price
                
                # Intraday return (last bar vs previous bar)
                if len(df) >= 2:
                    features['intraday_returns'][symbol] = (
                        df['close'].iloc[-1] / df['close'].iloc[-2] - 1
                    )
                
                # Volume ratio (current vs average)
                avg_volume = df['volume'].mean()
                current_volume = df['volume'].iloc[-1]
                if avg_volume > 0:
                    features['volume_ratio'][symbol] = current_volume / avg_volume
                
                # VWAP
                features['vwap'][symbol] = df['vwap'].iloc[-1]
                
                # VWAP deviation
                if features['vwap'][symbol] > 0:
                    features['vwap_deviation'][symbol] = (
                        current_price - features['vwap'][symbol]
                    ) / features['vwap'][symbol]
                
                # Opening range (first ~2 bars assuming 15min = 30min opening)
                opening_bars = df.head(2)
                if len(opening_bars) >= 1:
                    features['opening_high'][symbol] = opening_bars['high'].max()
                    features['opening_low'][symbol] = opening_bars['low'].min()
                    
            except Exception as e:
                logger.debug(f"Could not compute intraday features for {symbol}: {e}")
                continue
        
        return features
    
    def generate_sample_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        seed: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate sample OHLCV data for testing.
        Uses GBM with realistic parameters.
        """
        np.random.seed(seed)
        
        dates = pd.date_range(start_date, end_date, freq='B')  # Business days
        n_days = len(dates)
        
        data = {}
        
        for symbol in symbols:
            # Random parameters per symbol
            mu = np.random.uniform(0.0001, 0.001)  # Daily drift
            sigma = np.random.uniform(0.01, 0.03)  # Daily vol
            initial_price = np.random.uniform(50, 500)
            
            # Generate log returns
            log_returns = np.random.normal(mu, sigma, n_days)
            log_prices = np.cumsum(log_returns)
            prices = initial_price * np.exp(log_prices)
            
            # Generate OHLCV
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            df['open'] = prices * (1 + np.random.uniform(-0.01, 0.01, n_days))
            df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
            df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
            df['volume'] = np.random.uniform(1e6, 1e8, n_days).astype(int)
            
            data[symbol] = df
            
        return data


# ============================================================
# HELPER FUNCTION FOR ALPACA DATA LOADING
# ============================================================

_alpaca_loader_instance = None

def get_market_data_loader(api_key: str = None, secret_key: str = None) -> 'AlpacaMarketDataLoader':
    """
    Get a singleton Alpaca market data loader instance.
    
    Args:
        api_key: Alpaca API key (optional, uses env var if not provided)
        secret_key: Alpaca secret key (optional, uses env var if not provided)
    
    Returns:
        AlpacaMarketDataLoader instance
    """
    global _alpaca_loader_instance
    
    if _alpaca_loader_instance is None:
        _alpaca_loader_instance = AlpacaMarketDataLoader(
            api_key=api_key or os.environ.get('ALPACA_API_KEY'),
            secret_key=secret_key or os.environ.get('ALPACA_SECRET_KEY'),
        )
    
    return _alpaca_loader_instance


class AlpacaMarketDataLoader:
    """
    Market data loader using Alpaca API.
    Used for fetching live price data for outcome tracking.
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY')
        self._client = None
        
        if self.api_key and self.secret_key:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
            except Exception as e:
                logger.warning(f"Could not initialize Alpaca client: {e}")
    
    def fetch_prices(self, symbols: List[str], lookback_days: int = 10) -> Optional[pd.DataFrame]:
        """
        Fetch historical prices for symbols.
        
        Returns DataFrame with columns = symbols, index = dates
        """
        if not self._client:
            logger.warning("Alpaca client not initialized")
            return None
        
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import DataFeed
            
            end = datetime.now()
            start = end - timedelta(days=lookback_days + 5)  # Buffer for weekends
            
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX,  # Use IEX feed (free tier)
            )
            
            bars = self._client.get_stock_bars(request)
            
            # Convert to DataFrame
            if hasattr(bars, 'df') and not bars.df.empty:
                df = bars.df
                
                # Pivot to get symbols as columns
                if 'symbol' in df.index.names:
                    df = df.reset_index()
                
                if 'symbol' in df.columns:
                    pivot = df.pivot(index='timestamp', columns='symbol', values='close')
                    return pivot
                else:
                    # Single symbol case
                    return pd.DataFrame({'close': df['close']})
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching prices from Alpaca: {e}")
            return None
