"""
Fundamentals Data Loader.
Fetches real P/E, ROE, dividend yield, and other fundamental data.
Uses yfinance as primary source with caching for efficiency.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """Fundamental data for a single symbol."""
    symbol: str
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    dividend_yield: Optional[float] = None
    dividend_rate: Optional[float] = None
    payout_ratio: Optional[float] = None
    beta: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    fetched_at: Optional[datetime] = None
    
    @property
    def quality_score(self) -> float:
        """Calculate composite quality score (0-1)."""
        scores = []
        
        # ROE component (higher is better, capped at 50%)
        if self.roe is not None:
            scores.append(min(self.roe / 0.50, 1.0))
        
        # Profit margin (higher is better, capped at 30%)
        if self.profit_margin is not None:
            scores.append(min(self.profit_margin / 0.30, 1.0))
        
        # Debt/Equity (lower is better, penalize > 2.0)
        if self.debt_to_equity is not None:
            scores.append(max(0, 1 - self.debt_to_equity / 3.0))
        
        # Current ratio (1.5-2.5 is ideal)
        if self.current_ratio is not None:
            if 1.5 <= self.current_ratio <= 2.5:
                scores.append(1.0)
            elif self.current_ratio < 1.0:
                scores.append(0.3)
            else:
                scores.append(0.7)
        
        return np.mean(scores) if scores else 0.5
    
    @property
    def value_score(self) -> float:
        """Calculate composite value score (0-1). Lower P/E = higher value."""
        scores = []
        
        # P/E (lower is more value, but not negative)
        if self.pe_ratio is not None and self.pe_ratio > 0:
            # 10 is cheap, 30 is expensive
            pe_score = max(0, 1 - (self.pe_ratio - 10) / 30)
            scores.append(pe_score)
        
        # Price/Book (lower is more value)
        if self.price_to_book is not None and self.price_to_book > 0:
            pb_score = max(0, 1 - (self.price_to_book - 1) / 5)
            scores.append(pb_score)
        
        # Price/Sales (lower is more value)
        if self.price_to_sales is not None and self.price_to_sales > 0:
            ps_score = max(0, 1 - (self.price_to_sales - 1) / 8)
            scores.append(ps_score)
        
        return np.mean(scores) if scores else 0.5
    
    @property
    def growth_score(self) -> float:
        """Calculate growth score (0-1)."""
        scores = []
        
        if self.earnings_growth is not None:
            scores.append(min(max(self.earnings_growth / 0.30, 0), 1.0))
        
        if self.revenue_growth is not None:
            scores.append(min(max(self.revenue_growth / 0.25, 0), 1.0))
        
        return np.mean(scores) if scores else 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'pe_ratio': self.pe_ratio,
            'forward_pe': self.forward_pe,
            'peg_ratio': self.peg_ratio,
            'price_to_book': self.price_to_book,
            'price_to_sales': self.price_to_sales,
            'roe': self.roe,
            'roa': self.roa,
            'profit_margin': self.profit_margin,
            'operating_margin': self.operating_margin,
            'dividend_yield': self.dividend_yield,
            'dividend_rate': self.dividend_rate,
            'beta': self.beta,
            'market_cap': self.market_cap,
            'sector': self.sector,
            'industry': self.industry,
            'debt_to_equity': self.debt_to_equity,
            'quality_score': self.quality_score,
            'value_score': self.value_score,
            'growth_score': self.growth_score,
            'fetched_at': self.fetched_at.isoformat() if self.fetched_at else None,
        }


class FundamentalsLoader:
    """
    Loads and caches fundamental data from yfinance.
    Handles rate limiting and provides fallbacks.
    """
    
    def __init__(
        self,
        cache_dir: str = "outputs/fundamentals_cache",
        cache_hours: int = 4,  # Fundamentals don't change often
        max_concurrent: int = 5,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hours = cache_hours
        self.max_concurrent = max_concurrent
        
        self._cache: Dict[str, FundamentalData] = {}
        self._lock = threading.Lock()
        self._last_fetch: Dict[str, datetime] = {}
        
        # Load cached data
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached fundamentals from disk."""
        cache_file = self.cache_dir / "fundamentals.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for symbol, values in data.items():
                        if 'fetched_at' in values and values['fetched_at']:
                            values['fetched_at'] = datetime.fromisoformat(values['fetched_at'])
                        self._cache[symbol] = FundamentalData(**{
                            k: v for k, v in values.items() 
                            if k in FundamentalData.__dataclass_fields__
                        })
                logger.info(f"Loaded {len(self._cache)} cached fundamentals")
            except Exception as e:
                logger.warning(f"Failed to load fundamentals cache: {e}")
    
    def _save_cache(self) -> None:
        """Save fundamentals cache to disk."""
        cache_file = self.cache_dir / "fundamentals.json"
        try:
            data = {
                symbol: fd.to_dict() 
                for symbol, fd in self._cache.items()
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save fundamentals cache: {e}")
    
    def get_fundamentals(
        self,
        symbols: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, FundamentalData]:
        """
        Get fundamental data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            force_refresh: Force fetch even if cached
            
        Returns:
            Dict mapping symbol to FundamentalData
        """
        results = {}
        to_fetch = []
        
        with self._lock:
            now = datetime.now()
            for symbol in symbols:
                # Check cache
                if symbol in self._cache and not force_refresh:
                    cached = self._cache[symbol]
                    if cached.fetched_at:
                        age = now - cached.fetched_at
                        if age < timedelta(hours=self.cache_hours):
                            results[symbol] = cached
                            continue
                to_fetch.append(symbol)
        
        # Fetch missing data
        if to_fetch:
            fetched = self._fetch_batch(to_fetch)
            results.update(fetched)
            
            # Update cache
            with self._lock:
                self._cache.update(fetched)
                self._save_cache()
        
        return results
    
    def get_single(self, symbol: str, force_refresh: bool = False) -> Optional[FundamentalData]:
        """Get fundamentals for a single symbol."""
        result = self.get_fundamentals([symbol], force_refresh)
        return result.get(symbol)
    
    def _fetch_batch(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """Fetch fundamental data for a batch of symbols."""
        results = {}
        
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return results
        
        # Process in batches to avoid rate limits
        batch_size = min(self.max_concurrent, 10)
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            for symbol in batch:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if not info or 'symbol' not in info:
                        logger.warning(f"No data for {symbol}")
                        continue
                    
                    # Handle dividend yield - yfinance returns 'dividendYield' as percentage
                    # Use 'trailingAnnualDividendYield' which is in decimal form
                    div_yield = info.get('trailingAnnualDividendYield')
                    if div_yield is None:
                        # Fallback: convert percentage to decimal
                        pct_yield = info.get('dividendYield')
                        if pct_yield is not None:
                            div_yield = pct_yield / 100.0 if pct_yield > 1 else pct_yield
                    
                    fd = FundamentalData(
                        symbol=symbol,
                        pe_ratio=info.get('trailingPE'),
                        forward_pe=info.get('forwardPE'),
                        peg_ratio=info.get('pegRatio'),
                        price_to_book=info.get('priceToBook'),
                        price_to_sales=info.get('priceToSalesTrailing12Months'),
                        roe=info.get('returnOnEquity'),
                        roa=info.get('returnOnAssets'),
                        profit_margin=info.get('profitMargins'),
                        operating_margin=info.get('operatingMargins'),
                        gross_margin=info.get('grossMargins'),
                        dividend_yield=div_yield,
                        dividend_rate=info.get('dividendRate'),
                        payout_ratio=info.get('payoutRatio'),
                        beta=info.get('beta'),
                        market_cap=info.get('marketCap'),
                        enterprise_value=info.get('enterpriseValue'),
                        revenue_growth=info.get('revenueGrowth'),
                        earnings_growth=info.get('earningsGrowth'),
                        debt_to_equity=info.get('debtToEquity'),
                        current_ratio=info.get('currentRatio'),
                        quick_ratio=info.get('quickRatio'),
                        sector=info.get('sector'),
                        industry=info.get('industry'),
                        fetched_at=datetime.now(),
                    )
                    
                    results[symbol] = fd
                    logger.debug(f"Fetched fundamentals for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
                    continue
        
        return results
    
    def get_dividend_yields(self, symbols: List[str]) -> Dict[str, float]:
        """Get dividend yields for symbols."""
        fundamentals = self.get_fundamentals(symbols)
        return {
            symbol: fd.dividend_yield or 0.0
            for symbol, fd in fundamentals.items()
        }
    
    def get_pe_ratios(self, symbols: List[str]) -> Dict[str, float]:
        """Get P/E ratios for symbols."""
        fundamentals = self.get_fundamentals(symbols)
        return {
            symbol: fd.pe_ratio or float('inf')
            for symbol, fd in fundamentals.items()
        }
    
    def get_quality_scores(self, symbols: List[str]) -> Dict[str, float]:
        """Get quality scores for symbols."""
        fundamentals = self.get_fundamentals(symbols)
        return {
            symbol: fd.quality_score
            for symbol, fd in fundamentals.items()
        }
    
    def get_value_scores(self, symbols: List[str]) -> Dict[str, float]:
        """Get value scores for symbols."""
        fundamentals = self.get_fundamentals(symbols)
        return {
            symbol: fd.value_score
            for symbol, fd in fundamentals.items()
        }
    
    def rank_by_factor(
        self,
        symbols: List[str],
        factor: str = "value_score",
        ascending: bool = False,
    ) -> List[tuple]:
        """
        Rank symbols by a fundamental factor.
        
        Args:
            symbols: List of symbols to rank
            factor: Factor to rank by (value_score, quality_score, pe_ratio, etc.)
            ascending: Whether lower is better
            
        Returns:
            List of (symbol, score) tuples, sorted by factor
        """
        fundamentals = self.get_fundamentals(symbols)
        
        ranked = []
        for symbol, fd in fundamentals.items():
            score = getattr(fd, factor, None)
            if score is not None:
                ranked.append((symbol, score))
        
        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked
    
    def get_sector_breakdown(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Get symbols grouped by sector."""
        fundamentals = self.get_fundamentals(symbols)
        
        sectors: Dict[str, List[str]] = {}
        for symbol, fd in fundamentals.items():
            sector = fd.sector or "Unknown"
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)
        
        return sectors


# Fallback fundamental data for major stocks (when yfinance fails)
# Updated periodically - provides baseline data even if API is down
FALLBACK_FUNDAMENTALS = {
    # Tech Giants
    'AAPL': {'pe_ratio': 28.5, 'roe': 0.147, 'profit_margin': 0.24, 'debt_to_equity': 1.51, 'sector': 'Technology'},
    'MSFT': {'pe_ratio': 32.1, 'roe': 0.38, 'profit_margin': 0.34, 'debt_to_equity': 0.42, 'sector': 'Technology'},
    'GOOGL': {'pe_ratio': 22.5, 'roe': 0.25, 'profit_margin': 0.22, 'debt_to_equity': 0.06, 'sector': 'Technology'},
    'AMZN': {'pe_ratio': 58.2, 'roe': 0.17, 'profit_margin': 0.06, 'debt_to_equity': 0.59, 'sector': 'Consumer Cyclical'},
    'NVDA': {'pe_ratio': 65.0, 'roe': 0.69, 'profit_margin': 0.55, 'debt_to_equity': 0.17, 'sector': 'Technology'},
    'META': {'pe_ratio': 25.8, 'roe': 0.28, 'profit_margin': 0.29, 'debt_to_equity': 0.15, 'sector': 'Technology'},
    # Financials
    'JPM': {'pe_ratio': 11.2, 'roe': 0.14, 'profit_margin': 0.32, 'debt_to_equity': 1.21, 'sector': 'Financial Services'},
    'BAC': {'pe_ratio': 10.5, 'roe': 0.10, 'profit_margin': 0.28, 'debt_to_equity': 1.08, 'sector': 'Financial Services'},
    'GS': {'pe_ratio': 13.8, 'roe': 0.11, 'profit_margin': 0.21, 'debt_to_equity': 2.41, 'sector': 'Financial Services'},
    'WFC': {'pe_ratio': 12.1, 'roe': 0.10, 'profit_margin': 0.25, 'debt_to_equity': 0.92, 'sector': 'Financial Services'},
    # Healthcare
    'JNJ': {'pe_ratio': 14.8, 'roe': 0.23, 'profit_margin': 0.22, 'debt_to_equity': 0.44, 'sector': 'Healthcare'},
    'UNH': {'pe_ratio': 21.5, 'roe': 0.24, 'profit_margin': 0.05, 'debt_to_equity': 0.71, 'sector': 'Healthcare'},
    'PFE': {'pe_ratio': 10.2, 'roe': 0.08, 'profit_margin': 0.12, 'debt_to_equity': 0.81, 'sector': 'Healthcare'},
    # Consumer
    'KO': {'pe_ratio': 24.5, 'roe': 0.42, 'profit_margin': 0.24, 'debt_to_equity': 1.78, 'sector': 'Consumer Defensive'},
    'PEP': {'pe_ratio': 25.2, 'roe': 0.49, 'profit_margin': 0.10, 'debt_to_equity': 2.12, 'sector': 'Consumer Defensive'},
    'WMT': {'pe_ratio': 27.8, 'roe': 0.19, 'profit_margin': 0.02, 'debt_to_equity': 0.62, 'sector': 'Consumer Defensive'},
    # Energy
    'XOM': {'pe_ratio': 11.5, 'roe': 0.18, 'profit_margin': 0.11, 'debt_to_equity': 0.21, 'sector': 'Energy'},
    'CVX': {'pe_ratio': 12.8, 'roe': 0.15, 'profit_margin': 0.10, 'debt_to_equity': 0.17, 'sector': 'Energy'},
    # Industrials
    'CAT': {'pe_ratio': 15.2, 'roe': 0.55, 'profit_margin': 0.16, 'debt_to_equity': 1.99, 'sector': 'Industrials'},
    'HON': {'pe_ratio': 22.1, 'roe': 0.31, 'profit_margin': 0.15, 'debt_to_equity': 1.09, 'sector': 'Industrials'},
    'RTX': {'pe_ratio': 40.5, 'roe': 0.06, 'profit_margin': 0.06, 'debt_to_equity': 0.52, 'sector': 'Industrials'},
    'LOW': {'pe_ratio': 18.5, 'roe': 0.90, 'profit_margin': 0.08, 'debt_to_equity': 8.5, 'sector': 'Consumer Cyclical'},
    # More stocks
    'AVGO': {'pe_ratio': 32.5, 'roe': 0.30, 'profit_margin': 0.28, 'debt_to_equity': 1.74, 'sector': 'Technology'},
    'CME': {'pe_ratio': 21.0, 'roe': 0.11, 'profit_margin': 0.55, 'debt_to_equity': 0.12, 'sector': 'Financial Services'},
    'MTB': {'pe_ratio': 9.8, 'roe': 0.11, 'profit_margin': 0.32, 'debt_to_equity': 0.25, 'sector': 'Financial Services'},
    'GLW': {'pe_ratio': 38.5, 'roe': 0.06, 'profit_margin': 0.06, 'debt_to_equity': 0.64, 'sector': 'Technology'},
    'CDW': {'pe_ratio': 24.5, 'roe': 0.67, 'profit_margin': 0.05, 'debt_to_equity': 2.45, 'sector': 'Technology'},
    'JNPR': {'pe_ratio': 15.2, 'roe': 0.08, 'profit_margin': 0.10, 'debt_to_equity': 0.51, 'sector': 'Technology'},
}


class FundamentalsLoaderWithFallback(FundamentalsLoader):
    """Extended loader with fallback data when yfinance fails."""
    
    def get_fundamentals(
        self,
        symbols: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, FundamentalData]:
        """Get fundamentals with fallback to known data."""
        # First try the parent method
        results = super().get_fundamentals(symbols, force_refresh)
        
        # If we got nothing from yfinance, use fallback data
        if not results:
            logger.warning("yfinance returned no data, using fallback fundamentals")
            for symbol in symbols:
                if symbol in FALLBACK_FUNDAMENTALS:
                    fb = FALLBACK_FUNDAMENTALS[symbol]
                    results[symbol] = FundamentalData(
                        symbol=symbol,
                        pe_ratio=fb.get('pe_ratio'),
                        roe=fb.get('roe'),
                        profit_margin=fb.get('profit_margin'),
                        debt_to_equity=fb.get('debt_to_equity'),
                        sector=fb.get('sector'),
                        fetched_at=datetime.now(),
                    )
        else:
            # Supplement any missing symbols with fallback
            for symbol in symbols:
                if symbol not in results and symbol in FALLBACK_FUNDAMENTALS:
                    fb = FALLBACK_FUNDAMENTALS[symbol]
                    results[symbol] = FundamentalData(
                        symbol=symbol,
                        pe_ratio=fb.get('pe_ratio'),
                        roe=fb.get('roe'),
                        profit_margin=fb.get('profit_margin'),
                        debt_to_equity=fb.get('debt_to_equity'),
                        sector=fb.get('sector'),
                        fetched_at=datetime.now(),
                    )
        
        return results


# Singleton instance
_fundamentals_loader: Optional[FundamentalsLoader] = None


def get_fundamentals_loader() -> FundamentalsLoader:
    """Get the singleton FundamentalsLoader instance with fallback support."""
    global _fundamentals_loader
    if _fundamentals_loader is None:
        _fundamentals_loader = FundamentalsLoaderWithFallback()
    return _fundamentals_loader
