"""
Spread Analyzer - Analyzes bid-ask spreads to determine optimal execution strategy.

Spread Categories:
- TIGHT: < 0.05% - Very liquid, market order is fine
- NORMAL: 0.05-0.15% - Liquid, use aggressive limit
- MODERATE: 0.15-0.30% - Moderate liquidity, patient limit
- WIDE: 0.30-0.50% - Less liquid, very patient limit  
- ILLIQUID: > 0.50% - Poor liquidity, flag for review
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime


class SpreadCategory(Enum):
    """Categories of bid-ask spread width."""
    TIGHT = "tight"           # < 0.05%
    NORMAL = "normal"         # 0.05-0.15%
    MODERATE = "moderate"     # 0.15-0.30%
    WIDE = "wide"             # 0.30-0.50%
    ILLIQUID = "illiquid"     # > 0.50%


@dataclass
class SpreadAnalysis:
    """Result of spread analysis for a symbol."""
    symbol: str
    bid: float
    ask: float
    mid: float
    spread: float              # Absolute spread in $
    spread_pct: float          # Spread as percentage
    category: SpreadCategory
    bid_size: int              # Number of shares at bid
    ask_size: int              # Number of shares at ask
    recommended_limit_pct: float  # How far into spread to place limit (0-1)
    recommended_timeout: int   # Seconds to wait for limit fill
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'bid': self.bid,
            'ask': self.ask,
            'mid': self.mid,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'category': self.category.value,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'recommended_limit_pct': self.recommended_limit_pct,
            'recommended_timeout': self.recommended_timeout,
            'timestamp': self.timestamp.isoformat(),
        }


class SpreadAnalyzer:
    """
    Analyzes bid-ask spreads to determine execution strategy.
    
    Uses real-time quote data to classify spreads and recommend
    limit order parameters.
    """
    
    # Spread thresholds (as percentages)
    THRESHOLDS = {
        'tight': 0.05,      # < 0.05%
        'normal': 0.15,     # 0.05-0.15%
        'moderate': 0.30,   # 0.15-0.30%
        'wide': 0.50,       # 0.30-0.50%
        # > 0.50% = illiquid
    }
    
    # Limit order parameters by category
    LIMIT_PARAMS = {
        SpreadCategory.TIGHT: {
            'limit_pct': 0.0,      # Don't bother with limit
            'timeout': 0,          # Go straight to market
            'use_limit': False,
        },
        SpreadCategory.NORMAL: {
            'limit_pct': 0.25,     # 25% into spread
            'timeout': 30,         # 30 second timeout
            'use_limit': True,
        },
        SpreadCategory.MODERATE: {
            'limit_pct': 0.35,     # 35% into spread
            'timeout': 60,         # 60 second timeout
            'use_limit': True,
        },
        SpreadCategory.WIDE: {
            'limit_pct': 0.45,     # 45% into spread
            'timeout': 90,         # 90 second timeout
            'use_limit': True,
        },
        SpreadCategory.ILLIQUID: {
            'limit_pct': 0.50,     # 50% into spread (mid price)
            'timeout': 120,        # 2 minute timeout
            'use_limit': True,
        },
    }
    
    def __init__(self, data_client=None):
        """
        Initialize the spread analyzer.
        
        Args:
            data_client: Alpaca StockHistoricalDataClient for quotes
        """
        self.data_client = data_client
        self.analysis_cache: Dict[str, SpreadAnalysis] = {}
        self.cache_ttl = 5  # Cache valid for 5 seconds
    
    def analyze(
        self, 
        symbol: str,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        bid_size: Optional[int] = None,
        ask_size: Optional[int] = None,
    ) -> SpreadAnalysis:
        """
        Analyze the spread for a symbol.
        
        Args:
            symbol: Stock symbol
            bid: Bid price (optional, will fetch if not provided)
            ask: Ask price (optional, will fetch if not provided)
            bid_size: Size at bid
            ask_size: Size at ask
        
        Returns:
            SpreadAnalysis with recommendations
        """
        # If prices not provided, try to fetch
        if bid is None or ask is None:
            quote = self._get_quote(symbol)
            if quote:
                bid = quote.get('bid', 0)
                ask = quote.get('ask', 0)
                bid_size = quote.get('bid_size', 0)
                ask_size = quote.get('ask_size', 0)
            else:
                # Fallback to reasonable defaults
                logging.warning(f"Could not get quote for {symbol}, using defaults")
                return self._default_analysis(symbol)
        
        # Handle edge cases
        if bid <= 0 or ask <= 0 or ask <= bid:
            logging.warning(f"Invalid quote for {symbol}: bid={bid}, ask={ask}")
            return self._default_analysis(symbol)
        
        # Calculate spread metrics
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = (spread / mid) * 100
        
        # Categorize spread
        category = self._categorize_spread(spread_pct)
        
        # Get limit parameters
        params = self.LIMIT_PARAMS[category]
        
        analysis = SpreadAnalysis(
            symbol=symbol,
            bid=bid,
            ask=ask,
            mid=mid,
            spread=spread,
            spread_pct=spread_pct,
            category=category,
            bid_size=bid_size or 0,
            ask_size=ask_size or 0,
            recommended_limit_pct=params['limit_pct'],
            recommended_timeout=params['timeout'],
            timestamp=datetime.now(),
        )
        
        # Cache the analysis
        self.analysis_cache[symbol] = analysis
        
        return analysis
    
    def _categorize_spread(self, spread_pct: float) -> SpreadCategory:
        """Categorize spread based on percentage."""
        if spread_pct < self.THRESHOLDS['tight']:
            return SpreadCategory.TIGHT
        elif spread_pct < self.THRESHOLDS['normal']:
            return SpreadCategory.NORMAL
        elif spread_pct < self.THRESHOLDS['moderate']:
            return SpreadCategory.MODERATE
        elif spread_pct < self.THRESHOLDS['wide']:
            return SpreadCategory.WIDE
        else:
            return SpreadCategory.ILLIQUID
    
    def _get_quote(self, symbol: str) -> Optional[Dict]:
        """Fetch latest quote from data client."""
        if not self.data_client:
            return None
        
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'bid_size': int(quote.bid_size),
                    'ask_size': int(quote.ask_size),
                }
        except Exception as e:
            logging.warning(f"Error fetching quote for {symbol}: {e}")
        
        return None
    
    def _default_analysis(self, symbol: str) -> SpreadAnalysis:
        """Return conservative default analysis when quote unavailable."""
        return SpreadAnalysis(
            symbol=symbol,
            bid=0,
            ask=0,
            mid=0,
            spread=0,
            spread_pct=0.10,  # Assume 0.10% spread
            category=SpreadCategory.NORMAL,
            bid_size=0,
            ask_size=0,
            recommended_limit_pct=0.25,
            recommended_timeout=30,
            timestamp=datetime.now(),
        )
    
    def calculate_limit_price(
        self, 
        analysis: SpreadAnalysis, 
        side: str,
        aggression: float = 1.0,
    ) -> float:
        """
        Calculate the limit price based on spread analysis.
        
        Args:
            analysis: SpreadAnalysis for the symbol
            side: 'buy' or 'sell'
            aggression: Multiplier for limit aggressiveness (1.0 = normal)
        
        Returns:
            Limit price
        """
        if not self.LIMIT_PARAMS[analysis.category]['use_limit']:
            # For tight spreads, just use market (return mid as reference)
            return analysis.mid
        
        limit_pct = analysis.recommended_limit_pct * aggression
        limit_pct = min(limit_pct, 0.50)  # Cap at mid price
        
        if side.lower() == 'buy':
            # For buys, start from bid and move toward ask
            limit_price = analysis.bid + (analysis.spread * limit_pct)
        else:
            # For sells, start from ask and move toward bid
            limit_price = analysis.ask - (analysis.spread * limit_pct)
        
        # Round to 2 decimal places
        return round(limit_price, 2)
    
    def should_use_limit_order(self, analysis: SpreadAnalysis) -> bool:
        """Determine if a limit order should be used based on spread."""
        return self.LIMIT_PARAMS[analysis.category]['use_limit']
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all recent analyses."""
        if not self.analysis_cache:
            return {'message': 'No analyses cached'}
        
        categories = {}
        for analysis in self.analysis_cache.values():
            cat = analysis.category.value
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        return {
            'total_symbols': len(self.analysis_cache),
            'by_category': categories,
            'avg_spread_pct': sum(a.spread_pct for a in self.analysis_cache.values()) / len(self.analysis_cache),
        }
