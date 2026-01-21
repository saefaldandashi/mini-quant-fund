"""
Instrument Types - Core data model for tradeable instruments.

Supports:
- Spot instruments (stocks, ETFs)
- Futures contracts with multipliers, margin, and rolling
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional, Dict, Any
from abc import ABC


class AssetClass(Enum):
    """Asset class classification."""
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    FUTURE = "future"
    INDEX = "index"
    COMMODITY = "commodity"
    FX = "fx"


class SettlementType(Enum):
    """Futures settlement type."""
    CASH = "cash"
    PHYSICAL = "physical"


@dataclass
class Instrument(ABC):
    """
    Base class for all tradeable instruments.
    
    This is the foundation of our instrument-aware system.
    All positions, orders, and PnL calculations reference instruments.
    """
    symbol: str
    asset_class: AssetClass
    currency: str = "USD"
    name: Optional[str] = None
    exchange: Optional[str] = None
    
    # Trading constraints
    tradeable: bool = True
    shortable: bool = True
    min_order_size: float = 1.0
    
    # Metadata
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    def __hash__(self):
        return hash(self.symbol)
    
    def __eq__(self, other):
        if isinstance(other, Instrument):
            return self.symbol == other.symbol
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "currency": self.currency,
            "name": self.name,
            "exchange": self.exchange,
            "tradeable": self.tradeable,
            "shortable": self.shortable,
            "sector": self.sector,
            "industry": self.industry,
        }


@dataclass
class SpotInstrument(Instrument):
    """
    Spot instrument (stocks, ETFs).
    
    Characteristics:
    - Direct ownership of shares
    - No expiry
    - Lot size typically 1 share
    - May have borrow cost for shorting
    """
    lot_size: float = 1.0
    
    # Borrow info for shorting
    borrow_rate: float = 0.02  # Annual borrow rate (2% default)
    hard_to_borrow: bool = False
    borrow_available: bool = True
    
    # Financing for longs
    financing_rate: float = 0.05  # Annual financing rate (5% default)
    
    # Beta for hedging calculations
    beta: float = 1.0
    
    def __post_init__(self):
        if self.asset_class not in [AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO]:
            self.asset_class = AssetClass.EQUITY
    
    @classmethod
    def from_symbol(cls, symbol: str, **kwargs) -> 'SpotInstrument':
        """Create a spot instrument from just a symbol."""
        return cls(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            **kwargs
        )
    
    def calculate_borrow_cost(self, notional: float, days: int = 1) -> float:
        """Calculate borrow cost for a short position."""
        if notional >= 0:
            return 0.0  # No borrow cost for longs
        return abs(notional) * self.borrow_rate * days / 365
    
    def calculate_financing_cost(self, notional: float, days: int = 1) -> float:
        """Calculate financing cost for a long position."""
        if notional <= 0:
            return 0.0  # No financing for shorts
        return notional * self.financing_rate * days / 365


@dataclass
class FutureInstrument(Instrument):
    """
    Futures contract instrument.
    
    Characteristics:
    - Contract-based with multiplier
    - Has expiry date
    - Margin requirements
    - Daily mark-to-market settlement
    - Requires rolling before expiry
    """
    underlying: str = ""  # e.g., "SPY" for ES futures
    contract_code: str = ""  # e.g., "ESH26" (ES March 2026)
    
    # Contract specifications
    multiplier: float = 1.0  # Contract multiplier (e.g., 50 for ES)
    tick_size: float = 0.01  # Minimum price movement
    point_value: float = 1.0  # Dollar value per point move
    
    # Expiry and settlement
    expiry_date: Optional[date] = None
    last_trading_date: Optional[date] = None
    settlement_type: SettlementType = SettlementType.CASH
    
    # Margin requirements
    margin_initial: float = 0.0  # Initial margin per contract
    margin_maintenance: float = 0.0  # Maintenance margin per contract
    
    # Rolling
    roll_days_before_expiry: int = 5  # Days before expiry to roll
    next_contract: Optional[str] = None  # Symbol of next contract to roll into
    
    def __post_init__(self):
        self.asset_class = AssetClass.FUTURE
        if not self.point_value:
            self.point_value = self.multiplier * self.tick_size
    
    def calculate_notional(self, contracts: int, price: float) -> float:
        """Calculate notional value of a futures position."""
        return contracts * self.multiplier * price
    
    def calculate_margin(self, contracts: int) -> float:
        """Calculate initial margin requirement."""
        return abs(contracts) * self.margin_initial
    
    def calculate_pnl(self, contracts: int, entry_price: float, current_price: float) -> float:
        """Calculate P/L for a futures position."""
        return contracts * self.multiplier * (current_price - entry_price)
    
    def days_to_expiry(self, as_of: date = None) -> int:
        """Calculate days to expiry."""
        if not self.expiry_date:
            return 999
        as_of = as_of or date.today()
        return (self.expiry_date - as_of).days
    
    def should_roll(self, as_of: date = None) -> bool:
        """Check if contract should be rolled."""
        return self.days_to_expiry(as_of) <= self.roll_days_before_expiry
    
    @classmethod
    def create_es_contract(cls, month_code: str, year: int, expiry: date) -> 'FutureInstrument':
        """Create an ES (S&P 500 E-mini) futures contract."""
        contract_code = f"ES{month_code}{str(year)[-2:]}"
        return cls(
            symbol=contract_code,
            underlying="SPY",
            contract_code=contract_code,
            asset_class=AssetClass.FUTURE,
            name=f"S&P 500 E-mini {month_code}{year}",
            multiplier=50.0,
            tick_size=0.25,
            point_value=12.50,  # 50 * 0.25
            expiry_date=expiry,
            margin_initial=15000.0,
            margin_maintenance=12000.0,
            roll_days_before_expiry=5,
        )
    
    @classmethod
    def create_nq_contract(cls, month_code: str, year: int, expiry: date) -> 'FutureInstrument':
        """Create an NQ (Nasdaq 100 E-mini) futures contract."""
        contract_code = f"NQ{month_code}{str(year)[-2:]}"
        return cls(
            symbol=contract_code,
            underlying="QQQ",
            contract_code=contract_code,
            asset_class=AssetClass.FUTURE,
            name=f"Nasdaq 100 E-mini {month_code}{year}",
            multiplier=20.0,
            tick_size=0.25,
            point_value=5.00,  # 20 * 0.25
            expiry_date=expiry,
            margin_initial=18000.0,
            margin_maintenance=14000.0,
            roll_days_before_expiry=5,
        )


# === INSTRUMENT REGISTRY ===

class InstrumentRegistry:
    """
    Central registry for all instruments.
    
    Provides:
    - Symbol lookup
    - Instrument creation from symbols
    - Caching
    """
    
    def __init__(self):
        self._instruments: Dict[str, Instrument] = {}
        self._spot_defaults = {
            'borrow_rate': 0.02,
            'financing_rate': 0.05,
        }
    
    def register(self, instrument: Instrument):
        """Register an instrument."""
        self._instruments[instrument.symbol] = instrument
    
    def get(self, symbol: str) -> Optional[Instrument]:
        """Get an instrument by symbol."""
        return self._instruments.get(symbol)
    
    def get_or_create_spot(self, symbol: str, **kwargs) -> SpotInstrument:
        """Get or create a spot instrument."""
        if symbol in self._instruments:
            inst = self._instruments[symbol]
            if isinstance(inst, SpotInstrument):
                return inst
        
        # Create new spot instrument
        inst = SpotInstrument.from_symbol(
            symbol,
            borrow_rate=kwargs.get('borrow_rate', self._spot_defaults['borrow_rate']),
            financing_rate=kwargs.get('financing_rate', self._spot_defaults['financing_rate']),
            **{k: v for k, v in kwargs.items() if k not in ['borrow_rate', 'financing_rate']}
        )
        self._instruments[symbol] = inst
        return inst
    
    def all_instruments(self) -> Dict[str, Instrument]:
        """Get all registered instruments."""
        return self._instruments.copy()
    
    def spot_instruments(self) -> Dict[str, SpotInstrument]:
        """Get all spot instruments."""
        return {k: v for k, v in self._instruments.items() if isinstance(v, SpotInstrument)}
    
    def future_instruments(self) -> Dict[str, FutureInstrument]:
        """Get all futures instruments."""
        return {k: v for k, v in self._instruments.items() if isinstance(v, FutureInstrument)}


# Global registry instance
_registry: Optional[InstrumentRegistry] = None

def get_instrument_registry() -> InstrumentRegistry:
    """Get the global instrument registry."""
    global _registry
    if _registry is None:
        _registry = InstrumentRegistry()
    return _registry
