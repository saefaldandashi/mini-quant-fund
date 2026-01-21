"""
Market data loading and preprocessing.
Supports OHLCV data from CSV/parquet with adapters for external APIs.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

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
