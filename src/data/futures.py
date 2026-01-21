"""
Futures Data Module - Continuous price series and roll management.

Provides:
- Continuous futures price series (back-adjusted)
- Roll calendar and roll logic
- Contract specifications
- Data fetching for futures (Yahoo Finance / external sources)

NOTE: This module is for BACKTEST ONLY. Live futures trading 
requires a broker that supports futures (not Alpaca).
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging
import pandas as pd
import numpy as np

from src.core.instruments import FutureInstrument, AssetClass


class RollMethod(Enum):
    """Method for constructing continuous series."""
    BACK_ADJUST = "back_adjust"  # Adjust historical prices
    RATIO_ADJUST = "ratio_adjust"  # Use ratio adjustment
    CALENDAR_WEIGHTED = "calendar_weighted"  # Blend around roll date


@dataclass
class ContractSpec:
    """Specification for a futures contract type."""
    symbol_root: str  # e.g., "ES", "NQ", "CL"
    name: str
    underlying: str  # e.g., "SPY", "QQQ"
    exchange: str
    multiplier: float
    tick_size: float
    point_value: float
    margin_initial: float
    margin_maintenance: float
    trading_hours: str
    settlement_type: str = "cash"
    
    # Contract months (standard quarterly for equity futures)
    contract_months: List[str] = field(default_factory=lambda: ['H', 'M', 'U', 'Z'])
    
    # Roll parameters
    roll_days_before_expiry: int = 5
    
    def get_expiry_month_code(self, month: int) -> str:
        """Convert month number to futures month code."""
        codes = {
            1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
            7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
        }
        return codes.get(month, 'Z')
    
    def get_contract_symbol(self, year: int, month: int) -> str:
        """Generate contract symbol like ESH26."""
        month_code = self.get_expiry_month_code(month)
        return f"{self.symbol_root}{month_code}{str(year)[-2:]}"


# Standard contract specifications
FUTURES_SPECS = {
    'ES': ContractSpec(
        symbol_root='ES',
        name='S&P 500 E-mini',
        underlying='SPY',
        exchange='CME',
        multiplier=50.0,
        tick_size=0.25,
        point_value=12.50,
        margin_initial=15000.0,
        margin_maintenance=12000.0,
        trading_hours='Nearly 24h (Sun-Fri)',
    ),
    'NQ': ContractSpec(
        symbol_root='NQ',
        name='Nasdaq 100 E-mini',
        underlying='QQQ',
        exchange='CME',
        multiplier=20.0,
        tick_size=0.25,
        point_value=5.00,
        margin_initial=18000.0,
        margin_maintenance=14000.0,
        trading_hours='Nearly 24h (Sun-Fri)',
    ),
    'RTY': ContractSpec(
        symbol_root='RTY',
        name='Russell 2000 E-mini',
        underlying='IWM',
        exchange='CME',
        multiplier=50.0,
        tick_size=0.10,
        point_value=5.00,
        margin_initial=8000.0,
        margin_maintenance=6500.0,
        trading_hours='Nearly 24h (Sun-Fri)',
    ),
    'CL': ContractSpec(
        symbol_root='CL',
        name='Crude Oil',
        underlying='USO',
        exchange='NYMEX',
        multiplier=1000.0,
        tick_size=0.01,
        point_value=10.00,
        margin_initial=7000.0,
        margin_maintenance=6000.0,
        trading_hours='Nearly 24h (Sun-Fri)',
        contract_months=['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'],
    ),
    'GC': ContractSpec(
        symbol_root='GC',
        name='Gold',
        underlying='GLD',
        exchange='COMEX',
        multiplier=100.0,
        tick_size=0.10,
        point_value=10.00,
        margin_initial=9000.0,
        margin_maintenance=8000.0,
        trading_hours='Nearly 24h (Sun-Fri)',
        contract_months=['G', 'J', 'M', 'Q', 'V', 'Z'],
    ),
    'ZN': ContractSpec(
        symbol_root='ZN',
        name='10-Year T-Note',
        underlying='TLT',
        exchange='CBOT',
        multiplier=1000.0,
        tick_size=0.015625,  # 1/64 of a point
        point_value=15.625,
        margin_initial=2000.0,
        margin_maintenance=1800.0,
        trading_hours='Nearly 24h (Sun-Fri)',
    ),
}


@dataclass
class RollEvent:
    """Represents a futures roll event."""
    date: date
    old_contract: str
    new_contract: str
    old_price: float
    new_price: float
    roll_yield: float  # (new - old) / old
    adjustment: float  # For back-adjusted series


class FuturesRollCalendar:
    """
    Manages the roll calendar for futures contracts.
    
    Determines when to roll from one contract to the next.
    """
    
    def __init__(
        self,
        spec: ContractSpec,
        roll_days_before_expiry: int = 5,
    ):
        self.spec = spec
        self.roll_days = roll_days_before_expiry
        self._expiry_cache: Dict[str, date] = {}
    
    def get_expiry_date(self, year: int, month: int) -> date:
        """
        Get expiry date for a contract.
        
        For equity index futures, expiry is typically the 3rd Friday
        of the contract month.
        """
        cache_key = f"{year}-{month}"
        if cache_key in self._expiry_cache:
            return self._expiry_cache[cache_key]
        
        # Find 3rd Friday of the month
        first_day = date(year, month, 1)
        
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # 3rd Friday
        third_friday = first_friday + timedelta(weeks=2)
        
        self._expiry_cache[cache_key] = third_friday
        return third_friday
    
    def get_roll_date(self, year: int, month: int) -> date:
        """Get the roll date (N days before expiry)."""
        expiry = self.get_expiry_date(year, month)
        return expiry - timedelta(days=self.roll_days)
    
    def get_front_month(self, as_of: date) -> Tuple[int, int]:
        """
        Get the current front month contract (year, month).
        
        Returns the contract that should be traded as of the given date.
        """
        year = as_of.year
        
        # Check each contract month
        for month in sorted([3, 6, 9, 12]):  # Quarterly
            expiry = self.get_expiry_date(year, month)
            roll_date = self.get_roll_date(year, month)
            
            if as_of < roll_date:
                return (year, month)
        
        # Must be next year's first quarter
        return (year + 1, 3)
    
    def get_next_contract(self, year: int, month: int) -> Tuple[int, int]:
        """Get the next contract after the given one."""
        # Standard quarterly schedule
        quarters = [3, 6, 9, 12]
        
        try:
            idx = quarters.index(month)
            if idx < len(quarters) - 1:
                return (year, quarters[idx + 1])
            else:
                return (year + 1, quarters[0])
        except ValueError:
            # Not a quarterly month, just go to next quarter
            for q in quarters:
                if q > month:
                    return (year, q)
            return (year + 1, quarters[0])
    
    def should_roll(self, as_of: date, current_year: int, current_month: int) -> bool:
        """Check if we should roll from the current contract."""
        roll_date = self.get_roll_date(current_year, current_month)
        return as_of >= roll_date


class ContinuousFuturesSeries:
    """
    Builds continuous futures price series for backtesting.
    
    Uses back-adjustment to create a seamless series for signal generation.
    But tracks actual contracts for trading.
    """
    
    def __init__(
        self,
        spec: ContractSpec,
        roll_method: RollMethod = RollMethod.BACK_ADJUST,
    ):
        self.spec = spec
        self.roll_method = roll_method
        self.calendar = FuturesRollCalendar(spec)
        
        # Data storage
        self.raw_prices: Dict[str, pd.Series] = {}  # Per-contract prices
        self.continuous_prices: Optional[pd.Series] = None
        self.roll_events: List[RollEvent] = []
    
    def load_from_underlying(self, underlying_prices: pd.Series) -> pd.Series:
        """
        Create synthetic futures prices from underlying (e.g., SPY for ES).
        
        This is a simplified approach for backtesting when actual 
        futures data is not available.
        
        The synthetic price approximates futures by adding a small carry premium.
        """
        if underlying_prices.empty:
            return pd.Series(dtype=float)
        
        # Futures typically trade at a small premium to spot
        # Premium decreases as expiry approaches (basis convergence)
        
        # Simplified: add a constant multiplier and small carry
        carry_rate = 0.02  # ~2% annual carry (interest rate proxy)
        
        # Days to expiry varies, but we'll use average of 45 days
        avg_days_to_expiry = 45
        carry_premium = underlying_prices * carry_rate * avg_days_to_expiry / 365
        
        # Synthetic futures = spot * multiplier (we don't apply multiplier here,
        # that's handled in contract specs)
        synthetic = underlying_prices + carry_premium
        
        self.continuous_prices = synthetic
        return synthetic
    
    def build_continuous_series(
        self,
        contract_prices: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """
        Build continuous price series from individual contract data.
        
        Args:
            contract_prices: Dict of contract symbol -> DataFrame with 'close' column
            start_date: Start date for series
            end_date: End date for series
        
        Returns:
            Back-adjusted continuous price series
        """
        if not contract_prices:
            return pd.Series(dtype=float)
        
        self.raw_prices = {k: v['close'] for k, v in contract_prices.items() 
                          if 'close' in v.columns}
        
        # Build the series day by day
        all_dates = pd.date_range(start_date, end_date, freq='B')
        continuous = pd.Series(index=all_dates, dtype=float)
        
        # Track current contract and adjustment factor
        current_year, current_month = self.calendar.get_front_month(start_date)
        adjustment = 0.0
        
        for dt in all_dates:
            current_date = dt.date()
            
            # Check if we need to roll
            if self.calendar.should_roll(current_date, current_year, current_month):
                # Get prices for old and new contracts
                old_symbol = self.spec.get_contract_symbol(current_year, current_month)
                new_year, new_month = self.calendar.get_next_contract(current_year, current_month)
                new_symbol = self.spec.get_contract_symbol(new_year, new_month)
                
                if old_symbol in self.raw_prices and new_symbol in self.raw_prices:
                    old_series = self.raw_prices[old_symbol]
                    new_series = self.raw_prices[new_symbol]
                    
                    if dt in old_series.index and dt in new_series.index:
                        old_price = old_series[dt]
                        new_price = new_series[dt]
                        
                        # Record roll event
                        roll_yield = (new_price - old_price) / old_price
                        adjustment += old_price - new_price
                        
                        self.roll_events.append(RollEvent(
                            date=current_date,
                            old_contract=old_symbol,
                            new_contract=new_symbol,
                            old_price=old_price,
                            new_price=new_price,
                            roll_yield=roll_yield,
                            adjustment=adjustment,
                        ))
                
                # Move to new contract
                current_year, current_month = new_year, new_month
            
            # Get current contract price
            current_symbol = self.spec.get_contract_symbol(current_year, current_month)
            if current_symbol in self.raw_prices:
                price_series = self.raw_prices[current_symbol]
                if dt in price_series.index:
                    raw_price = price_series[dt]
                    continuous[dt] = raw_price + adjustment
        
        self.continuous_prices = continuous.dropna()
        return self.continuous_prices
    
    def get_current_contract(self, as_of: date) -> str:
        """Get the symbol of the current front month contract."""
        year, month = self.calendar.get_front_month(as_of)
        return self.spec.get_contract_symbol(year, month)
    
    def get_roll_yield_series(self) -> pd.Series:
        """Get series of roll yields for attribution."""
        if not self.roll_events:
            return pd.Series(dtype=float)
        
        dates = [e.date for e in self.roll_events]
        yields = [e.roll_yield for e in self.roll_events]
        return pd.Series(yields, index=dates)
    
    def get_total_roll_impact(self) -> float:
        """Get total roll impact (cost or yield)."""
        if not self.roll_events:
            return 0.0
        return sum(e.roll_yield for e in self.roll_events)


class FuturesDataLoader:
    """
    Loads and manages futures data.
    
    Supports:
    - Yahoo Finance for some futures
    - Synthetic data from underlying
    - External data files
    """
    
    def __init__(self):
        self.continuous_series: Dict[str, ContinuousFuturesSeries] = {}
    
    def load_from_yahoo(
        self,
        symbol_root: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.Series]:
        """
        Load futures data from Yahoo Finance.
        
        Yahoo has continuous futures for some contracts:
        - ES=F (S&P 500 E-mini)
        - NQ=F (Nasdaq 100 E-mini)
        - CL=F (Crude Oil)
        - GC=F (Gold)
        """
        import yfinance as yf
        
        # Map our symbols to Yahoo symbols
        yahoo_symbols = {
            'ES': 'ES=F',
            'NQ': 'NQ=F',
            'RTY': 'RTY=F',
            'CL': 'CL=F',
            'GC': 'GC=F',
            'ZN': 'ZN=F',
        }
        
        yahoo_symbol = yahoo_symbols.get(symbol_root)
        if not yahoo_symbol:
            logging.warning(f"No Yahoo symbol mapping for {symbol_root}")
            return None
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logging.warning(f"No data for {yahoo_symbol}")
                return None
            
            return hist['Close']
            
        except Exception as e:
            logging.error(f"Error loading {yahoo_symbol}: {e}")
            return None
    
    def get_continuous_series(
        self,
        symbol_root: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.Series]:
        """
        Get continuous futures series.
        
        First tries Yahoo Finance, then falls back to synthetic from underlying.
        """
        if symbol_root not in FUTURES_SPECS:
            logging.warning(f"Unknown futures symbol: {symbol_root}")
            return None
        
        spec = FUTURES_SPECS[symbol_root]
        
        # Try Yahoo first
        series = self.load_from_yahoo(symbol_root, start_date, end_date)
        
        if series is not None and not series.empty:
            # Wrap in ContinuousFuturesSeries for tracking
            cont = ContinuousFuturesSeries(spec)
            cont.continuous_prices = series
            self.continuous_series[symbol_root] = cont
            return series
        
        # Fall back to synthetic from underlying
        logging.info(f"Using synthetic futures for {symbol_root} from {spec.underlying}")
        
        try:
            import yfinance as yf
            underlying = yf.Ticker(spec.underlying)
            hist = underlying.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            cont = ContinuousFuturesSeries(spec)
            synthetic = cont.load_from_underlying(hist['Close'])
            self.continuous_series[symbol_root] = cont
            return synthetic
            
        except Exception as e:
            logging.error(f"Error creating synthetic futures: {e}")
            return None
    
    def get_all_futures_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, pd.Series]:
        """Load data for multiple futures symbols."""
        data = {}
        for symbol in symbols:
            series = self.get_continuous_series(symbol, start_date, end_date)
            if series is not None:
                data[symbol] = series
        return data


# Convenience function
def create_futures_instrument(
    symbol_root: str,
    as_of: date,
) -> Optional[FutureInstrument]:
    """
    Create a FutureInstrument for the front month contract.
    
    Args:
        symbol_root: e.g., "ES", "NQ"
        as_of: Date to determine front month
    
    Returns:
        FutureInstrument or None if symbol unknown
    """
    if symbol_root not in FUTURES_SPECS:
        return None
    
    spec = FUTURES_SPECS[symbol_root]
    calendar = FuturesRollCalendar(spec)
    year, month = calendar.get_front_month(as_of)
    expiry = calendar.get_expiry_date(year, month)
    
    contract_symbol = spec.get_contract_symbol(year, month)
    
    return FutureInstrument(
        symbol=contract_symbol,
        asset_class=AssetClass.FUTURE,
        underlying=spec.underlying,
        contract_code=contract_symbol,
        name=f"{spec.name} {spec.get_expiry_month_code(month)}{year}",
        multiplier=spec.multiplier,
        tick_size=spec.tick_size,
        point_value=spec.point_value,
        expiry_date=expiry,
        margin_initial=spec.margin_initial,
        margin_maintenance=spec.margin_maintenance,
        roll_days_before_expiry=spec.roll_days_before_expiry,
    )
