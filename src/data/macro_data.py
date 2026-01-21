"""
Macro Economic Data Loader

Fetches real-time macro indicators from:
1. Yahoo Finance - VIX, Treasury yields, market indices
2. FRED API - Economic indicators (CPI, GDP, unemployment, etc.)
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import requests

logger = logging.getLogger(__name__)


@dataclass
class MacroIndicators:
    """Real-time macro economic indicators."""
    timestamp: datetime
    
    # Market Volatility
    vix: float = 0.0
    vix_change: float = 0.0
    
    # Treasury Yields
    treasury_10y: float = 0.0
    treasury_2y: float = 0.0
    treasury_spread: float = 0.0  # 10y - 2y (yield curve)
    
    # Market Indices
    spy_price: float = 0.0
    spy_change_pct: float = 0.0
    spy_vs_200ma: float = 0.0  # % above/below 200-day MA
    
    # Dollar Index
    dxy: float = 0.0
    dxy_change: float = 0.0
    
    # Commodities
    gold_price: float = 0.0
    oil_price: float = 0.0
    
    # FRED Economic Data (if available)
    cpi_yoy: float = 0.0  # CPI year-over-year
    unemployment_rate: float = 0.0
    fed_funds_rate: float = 0.0
    financial_stress_index: float = 0.0
    
    # Computed Scores
    risk_score: float = 0.5  # 0 = risk-off, 1 = risk-on
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "vix": self.vix,
            "vix_change": self.vix_change,
            "treasury_10y": self.treasury_10y,
            "treasury_2y": self.treasury_2y,
            "treasury_spread": self.treasury_spread,
            "spy_price": self.spy_price,
            "spy_change_pct": self.spy_change_pct,
            "spy_vs_200ma": self.spy_vs_200ma,
            "dxy": self.dxy,
            "gold_price": self.gold_price,
            "oil_price": self.oil_price,
            "cpi_yoy": self.cpi_yoy,
            "unemployment_rate": self.unemployment_rate,
            "fed_funds_rate": self.fed_funds_rate,
            "financial_stress_index": self.financial_stress_index,
            "risk_score": self.risk_score,
        }


class YahooFinanceLoader:
    """Fetch market data from Yahoo Finance (no API key needed)."""
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    SYMBOLS = {
        "vix": "^VIX",
        "treasury_10y": "^TNX",
        "treasury_2y": "^IRX",  # 13-week T-bill as proxy
        "spy": "SPY",
        "dxy": "DX-Y.NYB",
        "gold": "GC=F",
        "oil": "CL=F",
    }
    
    def __init__(self):
        self.cache: Dict[str, tuple] = {}  # symbol -> (data, timestamp)
        self.cache_duration = timedelta(minutes=5)
    
    def _fetch_quote(self, symbol: str) -> Optional[Dict]:
        """Fetch quote data for a symbol."""
        cache_key = symbol
        now = datetime.now()
        
        # Check cache
        if cache_key in self.cache:
            data, cached_at = self.cache[cache_key]
            if now - cached_at < self.cache_duration:
                return data
        
        try:
            url = f"{self.BASE_URL}/{symbol}"
            params = {
                "interval": "1d",
                "range": "5d",
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    quote = result[0]
                    meta = quote.get("meta", {})
                    indicators = quote.get("indicators", {}).get("quote", [{}])[0]
                    
                    closes = indicators.get("close", [])
                    current_price = meta.get("regularMarketPrice", closes[-1] if closes else 0)
                    prev_close = meta.get("previousClose", closes[-2] if len(closes) > 1 else current_price)
                    
                    result_data = {
                        "price": current_price,
                        "prev_close": prev_close,
                        "change": current_price - prev_close if prev_close else 0,
                        "change_pct": ((current_price - prev_close) / prev_close * 100) if prev_close else 0,
                        "high": meta.get("regularMarketDayHigh", 0),
                        "low": meta.get("regularMarketDayLow", 0),
                        "closes": closes,
                    }
                    
                    self.cache[cache_key] = (result_data, now)
                    return result_data
                    
        except Exception as e:
            logger.warning(f"Error fetching {symbol} from Yahoo: {e}")
        
        return None
    
    def get_vix(self) -> tuple:
        """Get VIX level and change."""
        data = self._fetch_quote(self.SYMBOLS["vix"])
        if data:
            return data["price"], data["change"]
        return 20.0, 0.0  # Default
    
    def get_treasury_10y(self) -> float:
        """Get 10-year treasury yield."""
        data = self._fetch_quote(self.SYMBOLS["treasury_10y"])
        if data:
            return data["price"]
        return 4.0  # Default
    
    def get_spy_data(self) -> tuple:
        """Get SPY price and change %."""
        data = self._fetch_quote(self.SYMBOLS["spy"])
        if data:
            # Calculate 200-day MA approximation from available data
            closes = data.get("closes", [])
            ma = sum(closes) / len(closes) if closes else data["price"]
            vs_ma = ((data["price"] - ma) / ma * 100) if ma else 0
            
            return data["price"], data["change_pct"], vs_ma
        return 500.0, 0.0, 0.0
    
    def get_dxy(self) -> tuple:
        """Get Dollar Index."""
        data = self._fetch_quote(self.SYMBOLS["dxy"])
        if data:
            return data["price"], data["change"]
        return 100.0, 0.0
    
    def get_gold(self) -> float:
        """Get gold price."""
        data = self._fetch_quote(self.SYMBOLS["gold"])
        if data:
            return data["price"]
        return 2000.0
    
    def get_oil(self) -> float:
        """Get crude oil price."""
        data = self._fetch_quote(self.SYMBOLS["oil"])
        if data:
            return data["price"]
        return 75.0


class FREDLoader:
    """Fetch economic data from FRED API."""
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    # Key FRED series
    # Use percent change series where available for cleaner data
    SERIES = {
        "cpi_yoy": "CPALTT01USM657N",     # CPI All Items Total for US (YoY % change)
        "unemployment": "UNRATE",          # Unemployment rate (already %)
        "fed_funds": "FEDFUNDS",           # Fed Funds Rate (already %)
        "financial_stress": "STLFSI4",     # St. Louis Fed Financial Stress Index
        "gdp_growth": "A191RL1Q225SBEA",   # Real GDP growth (%)
        "pce": "PCEPI",                    # PCE inflation
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.cache: Dict[str, tuple] = {}
        self.cache_duration = timedelta(hours=6)  # FRED data updates less frequently
    
    def is_available(self) -> bool:
        """Check if FRED API is available."""
        return bool(self.api_key)
    
    def _fetch_series(self, series_id: str) -> Optional[float]:
        """Fetch latest value for a FRED series."""
        if not self.api_key:
            return None
        
        cache_key = series_id
        now = datetime.now()
        
        # Check cache
        if cache_key in self.cache:
            data, cached_at = self.cache[cache_key]
            if now - cached_at < self.cache_duration:
                return data
        
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                if observations:
                    value = observations[0].get("value", ".")
                    if value != ".":
                        result = float(value)
                        self.cache[cache_key] = (result, now)
                        return result
                        
        except Exception as e:
            logger.warning(f"Error fetching {series_id} from FRED: {e}")
        
        return None
    
    def get_cpi_yoy(self) -> float:
        """Get CPI year-over-year change."""
        value = self._fetch_series(self.SERIES["cpi_yoy"])
        return value if value else 3.0  # Default
    
    def get_unemployment(self) -> float:
        """Get unemployment rate."""
        value = self._fetch_series(self.SERIES["unemployment"])
        return value if value else 4.0  # Default
    
    def get_fed_funds_rate(self) -> float:
        """Get Fed Funds Rate."""
        value = self._fetch_series(self.SERIES["fed_funds"])
        return value if value else 5.0  # Default
    
    def get_financial_stress_index(self) -> float:
        """Get St. Louis Fed Financial Stress Index."""
        value = self._fetch_series(self.SERIES["financial_stress"])
        return value if value else 0.0  # 0 is normal


class MacroDataLoader:
    """
    Combined loader for all macro economic data.
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.yahoo = YahooFinanceLoader()
        self.fred = FREDLoader(fred_api_key)
        self.last_indicators: Optional[MacroIndicators] = None
        self.last_fetch: Optional[datetime] = None
        self.refresh_interval = timedelta(minutes=5)
    
    def fetch_all(self, force: bool = False) -> MacroIndicators:
        """Fetch all macro indicators."""
        now = datetime.now()
        
        # Use cached if recent
        if not force and self.last_fetch and self.last_indicators:
            if now - self.last_fetch < self.refresh_interval:
                return self.last_indicators
        
        logger.info("Fetching macro indicators...")
        
        # Yahoo Finance data
        vix, vix_change = self.yahoo.get_vix()
        treasury_10y = self.yahoo.get_treasury_10y()
        spy_price, spy_change, spy_vs_ma = self.yahoo.get_spy_data()
        dxy, dxy_change = self.yahoo.get_dxy()
        gold = self.yahoo.get_gold()
        oil = self.yahoo.get_oil()
        
        # FRED data (if available)
        cpi = self.fred.get_cpi_yoy() if self.fred.is_available() else 0.0
        unemployment = self.fred.get_unemployment() if self.fred.is_available() else 0.0
        fed_funds = self.fred.get_fed_funds_rate() if self.fred.is_available() else 0.0
        financial_stress = self.fred.get_financial_stress_index() if self.fred.is_available() else 0.0
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            vix=vix,
            spy_change=spy_change,
            treasury_spread=treasury_10y - 2.0,  # Approximate 2y
            financial_stress=financial_stress,
        )
        
        indicators = MacroIndicators(
            timestamp=now,
            vix=vix,
            vix_change=vix_change,
            treasury_10y=treasury_10y,
            treasury_2y=treasury_10y - 0.5,  # Approximate
            treasury_spread=0.5,  # Approximate
            spy_price=spy_price,
            spy_change_pct=spy_change,
            spy_vs_200ma=spy_vs_ma,
            dxy=dxy,
            dxy_change=dxy_change,
            gold_price=gold,
            oil_price=oil,
            cpi_yoy=cpi,
            unemployment_rate=unemployment,
            fed_funds_rate=fed_funds,
            financial_stress_index=financial_stress,
            risk_score=risk_score,
        )
        
        self.last_indicators = indicators
        self.last_fetch = now
        
        logger.info(f"Macro indicators updated: VIX={vix:.1f}, SPY={spy_price:.2f}, Risk={risk_score:.2f}")
        
        return indicators
    
    def _calculate_risk_score(
        self,
        vix: float,
        spy_change: float,
        treasury_spread: float,
        financial_stress: float,
    ) -> float:
        """
        Calculate overall risk score (0 = risk-off, 1 = risk-on).
        """
        score = 0.5  # Start neutral
        
        # VIX component (lower = more risk-on)
        if vix < 15:
            score += 0.15
        elif vix < 20:
            score += 0.05
        elif vix > 25:
            score -= 0.10
        elif vix > 30:
            score -= 0.20
        
        # SPY momentum
        if spy_change > 1:
            score += 0.10
        elif spy_change > 0:
            score += 0.05
        elif spy_change < -1:
            score -= 0.10
        elif spy_change < 0:
            score -= 0.05
        
        # Treasury spread (inverted = recession risk)
        if treasury_spread < 0:
            score -= 0.15  # Inverted yield curve
        elif treasury_spread > 1:
            score += 0.05
        
        # Financial stress
        if financial_stress > 1:
            score -= 0.15
        elif financial_stress > 0.5:
            score -= 0.05
        elif financial_stress < -0.5:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def get_status(self) -> Dict:
        """Get loader status."""
        return {
            "yahoo_finance": True,
            "fred_api": self.fred.is_available(),
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "cached_indicators": self.last_indicators is not None,
        }


# Global instance
_macro_loader: Optional[MacroDataLoader] = None

def get_macro_loader(fred_api_key: Optional[str] = None) -> MacroDataLoader:
    """Get or create the global macro loader."""
    global _macro_loader
    if _macro_loader is None:
        _macro_loader = MacroDataLoader(fred_api_key)
    return _macro_loader
