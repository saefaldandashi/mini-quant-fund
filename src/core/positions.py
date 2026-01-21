"""
Position Management - Tracks long and short positions with proper accounting.

Handles:
- Long positions (positive quantity)
- Short positions (negative quantity)
- Spot and futures positions
- P/L calculation (realized and unrealized)
- Financing and borrow costs
- Margin tracking for futures
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, List, Any
import logging

from .instruments import Instrument, SpotInstrument, FutureInstrument, AssetClass


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """
    Represents a position in an instrument.
    
    Key properties:
    - quantity: Positive for long, negative for short
    - notional: Market value (can be negative for shorts)
    - pnl_unrealized: Unrealized P/L
    - pnl_realized: Realized P/L from closed trades
    """
    instrument: Instrument
    quantity: float = 0.0  # Shares or contracts (negative = short)
    avg_entry_price: float = 0.0
    market_price: float = 0.0
    
    # P/L tracking
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    
    # Cost tracking
    total_financing_cost: float = 0.0  # For longs
    total_borrow_cost: float = 0.0  # For shorts
    total_commission: float = 0.0
    
    # Futures-specific
    margin_used: float = 0.0
    
    # Timestamps
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.now()
        self.last_updated = datetime.now()
        self._update_derived()
    
    @property
    def side(self) -> PositionSide:
        """Get position side."""
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    @property
    def notional(self) -> float:
        """
        Calculate notional value.
        
        For spot: quantity * price
        For futures: quantity * multiplier * price
        """
        if isinstance(self.instrument, FutureInstrument):
            return self.quantity * self.instrument.multiplier * self.market_price
        return self.quantity * self.market_price
    
    @property
    def abs_notional(self) -> float:
        """Absolute notional value."""
        return abs(self.notional)
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        if isinstance(self.instrument, FutureInstrument):
            return self.quantity * self.instrument.multiplier * self.avg_entry_price
        return self.quantity * self.avg_entry_price
    
    @property
    def total_pnl(self) -> float:
        """Total P/L including all costs."""
        return self.pnl_unrealized + self.pnl_realized - self.total_costs
    
    @property
    def total_costs(self) -> float:
        """Total costs (financing + borrow + commission)."""
        return self.total_financing_cost + self.total_borrow_cost + self.total_commission
    
    def _update_derived(self):
        """Update derived fields based on current state."""
        self.last_updated = datetime.now()
        
        # Calculate unrealized P/L
        if isinstance(self.instrument, FutureInstrument):
            self.pnl_unrealized = self.instrument.calculate_pnl(
                int(self.quantity), self.avg_entry_price, self.market_price
            )
            self.margin_used = self.instrument.calculate_margin(int(self.quantity))
        else:
            self.pnl_unrealized = self.quantity * (self.market_price - self.avg_entry_price)
    
    def update_market_price(self, price: float):
        """Update market price and recalculate P/L."""
        self.market_price = price
        self._update_derived()
    
    def add_quantity(self, qty: float, price: float, commission: float = 0.0) -> float:
        """
        Add to position (positive = buy, negative = sell).
        
        Returns realized P/L if reducing/closing position.
        """
        realized_pnl = 0.0
        
        if self.is_flat:
            # Opening new position
            self.quantity = qty
            self.avg_entry_price = price
            self.market_price = price
        
        elif (self.quantity > 0 and qty > 0) or (self.quantity < 0 and qty < 0):
            # Adding to existing position (same direction)
            total_cost = self.quantity * self.avg_entry_price + qty * price
            self.quantity += qty
            if self.quantity != 0:
                self.avg_entry_price = total_cost / self.quantity
            self.market_price = price
        
        else:
            # Reducing or reversing position
            if abs(qty) <= abs(self.quantity):
                # Partial close
                realized_pnl = abs(qty) * (price - self.avg_entry_price)
                if self.is_short:
                    realized_pnl = -realized_pnl  # Reverse for shorts
                self.quantity += qty
                self.market_price = price
            else:
                # Close and reverse
                # First, close existing position
                realized_pnl = abs(self.quantity) * (price - self.avg_entry_price)
                if self.is_short:
                    realized_pnl = -realized_pnl
                
                # Then open new position in opposite direction
                remaining = qty + self.quantity  # What's left after closing
                self.quantity = remaining
                self.avg_entry_price = price
                self.market_price = price
        
        self.pnl_realized += realized_pnl
        self.total_commission += commission
        self._update_derived()
        
        return realized_pnl
    
    def close(self, price: float, commission: float = 0.0) -> float:
        """Close the entire position. Returns realized P/L."""
        return self.add_quantity(-self.quantity, price, commission)
    
    def apply_daily_costs(self, days: int = 1):
        """
        Apply daily financing/borrow costs.
        
        Call this at end of each trading day.
        """
        if isinstance(self.instrument, SpotInstrument):
            if self.is_long:
                cost = self.instrument.calculate_financing_cost(self.notional, days)
                self.total_financing_cost += cost
            elif self.is_short:
                cost = self.instrument.calculate_borrow_cost(self.notional, days)
                self.total_borrow_cost += cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.instrument.symbol,
            "asset_class": self.instrument.asset_class.value,
            "quantity": self.quantity,
            "side": self.side.value,
            "avg_entry_price": self.avg_entry_price,
            "market_price": self.market_price,
            "notional": self.notional,
            "cost_basis": self.cost_basis,
            "pnl_unrealized": self.pnl_unrealized,
            "pnl_realized": self.pnl_realized,
            "total_pnl": self.total_pnl,
            "financing_cost": self.total_financing_cost,
            "borrow_cost": self.total_borrow_cost,
            "commission": self.total_commission,
            "margin_used": self.margin_used,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


@dataclass
class Trade:
    """
    Represents a single trade execution.
    """
    instrument: Instrument
    quantity: float  # Positive = buy, negative = sell
    price: float
    timestamp: datetime
    
    # Costs
    commission: float = 0.0
    slippage: float = 0.0
    
    # Metadata
    order_id: Optional[str] = None
    strategy: Optional[str] = None
    is_roll: bool = False  # For futures rolling
    
    @property
    def side(self) -> str:
        return "BUY" if self.quantity > 0 else "SELL"
    
    @property
    def notional(self) -> float:
        if isinstance(self.instrument, FutureInstrument):
            return abs(self.quantity) * self.instrument.multiplier * self.price
        return abs(self.quantity) * self.price
    
    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.instrument.symbol,
            "side": self.side,
            "quantity": abs(self.quantity),
            "price": self.price,
            "notional": self.notional,
            "commission": self.commission,
            "slippage": self.slippage,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "is_roll": self.is_roll,
        }


class PositionBook:
    """
    Manages a collection of positions.
    
    Provides:
    - Position lookup by symbol
    - Aggregate metrics (gross/net exposure)
    - Long/short book separation
    """
    
    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self._positions.get(symbol)
    
    def get_or_create_position(self, instrument: Instrument) -> Position:
        """Get existing position or create new one."""
        if instrument.symbol not in self._positions:
            self._positions[instrument.symbol] = Position(instrument=instrument)
        return self._positions[instrument.symbol]
    
    def update_position(
        self,
        instrument: Instrument,
        quantity_delta: float,
        price: float,
        commission: float = 0.0,
        strategy: Optional[str] = None,
        is_roll: bool = False,
    ) -> float:
        """
        Update a position (buy/sell).
        
        Returns realized P/L.
        """
        position = self.get_or_create_position(instrument)
        realized_pnl = position.add_quantity(quantity_delta, price, commission)
        
        # Record trade
        trade = Trade(
            instrument=instrument,
            quantity=quantity_delta,
            price=price,
            timestamp=datetime.now(),
            commission=commission,
            strategy=strategy,
            is_roll=is_roll,
        )
        self._trades.append(trade)
        
        # Remove flat positions
        if position.is_flat:
            del self._positions[instrument.symbol]
        
        return realized_pnl
    
    def update_prices(self, prices: Dict[str, float]):
        """Update market prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].update_market_price(price)
    
    def apply_daily_costs(self, days: int = 1):
        """Apply daily financing/borrow costs to all positions."""
        for position in self._positions.values():
            position.apply_daily_costs(days)
    
    # === AGGREGATE METRICS ===
    
    @property
    def positions(self) -> Dict[str, Position]:
        """All positions."""
        return self._positions.copy()
    
    @property
    def long_positions(self) -> Dict[str, Position]:
        """All long positions."""
        return {k: v for k, v in self._positions.items() if v.is_long}
    
    @property
    def short_positions(self) -> Dict[str, Position]:
        """All short positions."""
        return {k: v for k, v in self._positions.items() if v.is_short}
    
    @property
    def gross_exposure(self) -> float:
        """Sum of absolute notional values."""
        return sum(p.abs_notional for p in self._positions.values())
    
    @property
    def net_exposure(self) -> float:
        """Sum of notional values (long - short)."""
        return sum(p.notional for p in self._positions.values())
    
    @property
    def long_exposure(self) -> float:
        """Total long notional."""
        return sum(p.notional for p in self._positions.values() if p.is_long)
    
    @property
    def short_exposure(self) -> float:
        """Total short notional (as positive number)."""
        return abs(sum(p.notional for p in self._positions.values() if p.is_short))
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P/L across all positions."""
        return sum(p.pnl_unrealized for p in self._positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """Total realized P/L from all trades."""
        return sum(p.pnl_realized for p in self._positions.values())
    
    @property
    def total_margin_used(self) -> float:
        """Total margin used for futures positions."""
        return sum(p.margin_used for p in self._positions.values())
    
    @property
    def total_costs(self) -> float:
        """Total financing + borrow + commission costs."""
        return sum(p.total_costs for p in self._positions.values())
    
    def get_trades(self, since: datetime = None) -> List[Trade]:
        """Get trade history."""
        if since:
            return [t for t in self._trades if t.timestamp >= since]
        return self._trades.copy()
    
    def summary(self) -> Dict[str, Any]:
        """Get position book summary."""
        return {
            "position_count": len(self._positions),
            "long_count": len(self.long_positions),
            "short_count": len(self.short_positions),
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "long_exposure": self.long_exposure,
            "short_exposure": self.short_exposure,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "total_margin_used": self.total_margin_used,
            "total_costs": self.total_costs,
            "trade_count": len(self._trades),
        }
