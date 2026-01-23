"""
Cross-Asset Data Loader

Fetches historical price data for cross-asset correlation analysis:
1. Commodities (Oil, Gold, Copper, Natural Gas)
2. Currencies (DXY, EUR/USD, USD/JPY)
3. International Markets (Europe, Japan, China, Emerging)
4. Bonds/Credit (TLT, HYG, LQD)
5. Volatility (VIX)

Data sources:
- Yahoo Finance (free, no API key needed)
- Alpaca (for ETFs if available)
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class CrossAssetDataLoader:
    """
    Loads historical price data for cross-asset correlation analysis.
    
    Asset Classes:
    - Commodities: CL=F (Oil), GC=F (Gold), HG=F (Copper), NG=F (NatGas)
    - Currencies: DX-Y.NYB (DXY), EURUSD=X, USDJPY=X
    - International: EWG (Germany), EWJ (Japan), FXI (China), EEM (Emerging)
    - Bonds: TLT (Long Treasury), HYG (High Yield), LQD (Investment Grade)
    - Volatility: ^VIX
    """
    
    # Asset symbols to fetch
    COMMODITIES = {
        "CL=F": "Crude Oil",
        "GC=F": "Gold",
        "HG=F": "Copper",
        "NG=F": "Natural Gas",
        "SI=F": "Silver",
    }
    
    CURRENCIES = {
        "DX-Y.NYB": "Dollar Index (DXY)",
        "EURUSD=X": "EUR/USD",
        "USDJPY=X": "USD/JPY",
        "GBPUSD=X": "GBP/USD",
    }
    
    INTERNATIONAL = {
        "EWG": "Germany (DAX proxy)",
        "EWJ": "Japan (Nikkei proxy)",
        "FXI": "China (A-shares proxy)",
        "EWZ": "Brazil",
        "EWY": "South Korea",
        "EWT": "Taiwan",
        "EEM": "Emerging Markets",
        "VGK": "Europe",
    }
    
    BONDS = {
        "TLT": "20+ Year Treasury",
        "IEF": "7-10 Year Treasury",
        "HYG": "High Yield Corporate",
        "LQD": "Investment Grade Corporate",
        "TIP": "TIPS (Inflation Protected)",
    }
    
    VOLATILITY = {
        "^VIX": "VIX",
        "^VIX3M": "VIX 3-Month",
    }
    
    SECTORS = {
        "XLE": "Energy",
        "XLF": "Financials",
        "XLK": "Technology",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
    }
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.cache: Dict[str, Tuple[pd.Series, datetime]] = {}
        self.cache_duration = timedelta(minutes=15)
    
    def get_all_symbols(self) -> Dict[str, str]:
        """Get all tracked symbols with descriptions."""
        all_symbols = {}
        all_symbols.update(self.COMMODITIES)
        all_symbols.update(self.CURRENCIES)
        all_symbols.update(self.INTERNATIONAL)
        all_symbols.update(self.BONDS)
        all_symbols.update(self.VOLATILITY)
        all_symbols.update(self.SECTORS)
        return all_symbols
    
    def _fetch_yahoo_history(
        self,
        symbol: str,
        days: int = 252,
    ) -> Optional[pd.Series]:
        """
        Fetch historical prices from Yahoo Finance.
        Returns Series of closing prices.
        """
        # Check cache
        cache_key = f"{symbol}_{days}"
        if cache_key in self.cache:
            data, cached_at = self.cache[cache_key]
            if datetime.now() - cached_at < self.cache_duration:
                return data
        
        try:
            # Yahoo Finance v8 API
            end_ts = int(datetime.now().timestamp())
            start_ts = int((datetime.now() - timedelta(days=days + 30)).timestamp())
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
                "events": "history",
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Yahoo API error for {symbol}: {response.status_code}")
                return None
            
            data = response.json()
            
            if "chart" not in data or "result" not in data["chart"]:
                return None
            
            result = data["chart"]["result"][0]
            
            if "timestamp" not in result:
                return None
            
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]
            
            if "close" not in quotes:
                return None
            
            closes = quotes["close"]
            
            # Create Series
            dates = pd.to_datetime(timestamps, unit='s')
            series = pd.Series(closes, index=dates, name=symbol)
            
            # Remove NaN values
            series = series.dropna()
            
            # Take last N days
            series = series.tail(days)
            
            # Cache
            self.cache[cache_key] = (series, datetime.now())
            
            logger.debug(f"Fetched {len(series)} days for {symbol}")
            return series
            
        except Exception as e:
            logger.warning(f"Error fetching {symbol} from Yahoo: {e}")
            return None
    
    def fetch_commodity_prices(self, days: int = 252) -> Dict[str, pd.Series]:
        """Fetch all commodity prices."""
        return self._fetch_multiple(list(self.COMMODITIES.keys()), days)
    
    def fetch_currency_prices(self, days: int = 252) -> Dict[str, pd.Series]:
        """Fetch all currency prices."""
        return self._fetch_multiple(list(self.CURRENCIES.keys()), days)
    
    def fetch_international_prices(self, days: int = 252) -> Dict[str, pd.Series]:
        """Fetch all international market prices."""
        return self._fetch_multiple(list(self.INTERNATIONAL.keys()), days)
    
    def fetch_bond_prices(self, days: int = 252) -> Dict[str, pd.Series]:
        """Fetch all bond/credit prices."""
        return self._fetch_multiple(list(self.BONDS.keys()), days)
    
    def fetch_sector_prices(self, days: int = 252) -> Dict[str, pd.Series]:
        """Fetch all sector ETF prices."""
        return self._fetch_multiple(list(self.SECTORS.keys()), days)
    
    def fetch_volatility(self, days: int = 252) -> Dict[str, pd.Series]:
        """Fetch VIX data."""
        return self._fetch_multiple(list(self.VOLATILITY.keys()), days)
    
    def _fetch_multiple(
        self,
        symbols: List[str],
        days: int = 252,
    ) -> Dict[str, pd.Series]:
        """Fetch multiple symbols in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._fetch_yahoo_history, symbol, days): symbol
                for symbol in symbols
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data is not None and len(data) > 0:
                        results[symbol] = data
                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {e}")
        
        return results
    
    def fetch_all_cross_assets(
        self,
        days: int = 252,
    ) -> Dict[str, pd.Series]:
        """
        Fetch ALL cross-asset data for correlation analysis.
        Returns dict of symbol -> price series.
        """
        logger.info(f"Fetching cross-asset data for {days} days...")
        start_time = time.time()
        
        all_data = {}
        
        # Fetch each category
        categories = [
            ("Commodities", self.COMMODITIES),
            ("Currencies", self.CURRENCIES),
            ("International", self.INTERNATIONAL),
            ("Bonds", self.BONDS),
            ("Volatility", self.VOLATILITY),
            ("Sectors", self.SECTORS),
        ]
        
        for cat_name, symbols_dict in categories:
            symbols = list(symbols_dict.keys())
            data = self._fetch_multiple(symbols, days)
            all_data.update(data)
            logger.info(f"  {cat_name}: {len(data)}/{len(symbols)} symbols")
        
        elapsed = time.time() - start_time
        logger.info(f"Fetched {len(all_data)} cross-asset symbols in {elapsed:.1f}s")
        
        return all_data
    
    def get_summary(self) -> Dict:
        """Get summary of available data."""
        return {
            "commodities": list(self.COMMODITIES.keys()),
            "currencies": list(self.CURRENCIES.keys()),
            "international": list(self.INTERNATIONAL.keys()),
            "bonds": list(self.BONDS.keys()),
            "volatility": list(self.VOLATILITY.keys()),
            "sectors": list(self.SECTORS.keys()),
            "total_symbols": len(self.get_all_symbols()),
            "cached_symbols": len(self.cache),
        }


# Singleton instance
_cross_asset_loader: Optional[CrossAssetDataLoader] = None


def get_cross_asset_loader() -> CrossAssetDataLoader:
    """Get singleton instance of cross-asset data loader."""
    global _cross_asset_loader
    if _cross_asset_loader is None:
        _cross_asset_loader = CrossAssetDataLoader()
    return _cross_asset_loader
