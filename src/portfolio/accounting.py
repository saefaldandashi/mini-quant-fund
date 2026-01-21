"""
Portfolio Accounting - Complete portfolio state management.

Handles:
- Cash and equity tracking
- Long/Short position accounting
- Futures margin requirements
- Gross/Net exposure calculation
- Leverage tracking
- Daily P/L and costs
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
import logging

from src.core.instruments import (
    Instrument, SpotInstrument, FutureInstrument, 
    AssetClass, get_instrument_registry
)
from src.core.positions import Position, PositionBook, Trade, PositionSide


@dataclass
class PortfolioState:
    """
    Complete portfolio state at a point in time.
    
    Tracks:
    - Cash and equity
    - All positions (long/short, spot/futures)
    - Exposure metrics
    - Margin requirements
    """
    
    # Cash and equity
    cash: float = 0.0
    initial_capital: float = 100000.0
    
    # Position book
    position_book: PositionBook = field(default_factory=PositionBook)
    
    # Timestamp
    as_of: datetime = field(default_factory=datetime.now)
    
    # Accumulated costs
    total_commission: float = 0.0
    total_financing: float = 0.0
    total_borrow: float = 0.0
    
    def __post_init__(self):
        if self.cash == 0:
            self.cash = self.initial_capital
    
    # === CORE METRICS ===
    
    @property
    def equity_value(self) -> float:
        """
        Total portfolio equity.
        
        = Cash + Unrealized P/L + Realized P/L
        """
        return (
            self.cash + 
            self.position_book.total_unrealized_pnl + 
            self.position_book.total_realized_pnl -
            self.position_book.total_costs
        )
    
    @property
    def portfolio_value(self) -> float:
        """Alias for equity_value."""
        return self.equity_value
    
    @property
    def gross_exposure(self) -> float:
        """Sum of absolute position values."""
        return self.position_book.gross_exposure
    
    @property
    def net_exposure(self) -> float:
        """Sum of position values (long - short)."""
        return self.position_book.net_exposure
    
    @property
    def long_exposure(self) -> float:
        """Total long exposure."""
        return self.position_book.long_exposure
    
    @property
    def short_exposure(self) -> float:
        """Total short exposure (positive number)."""
        return self.position_book.short_exposure
    
    @property
    def leverage(self) -> float:
        """Gross exposure / equity."""
        if self.equity_value <= 0:
            return 0.0
        return self.gross_exposure / self.equity_value
    
    @property
    def net_leverage(self) -> float:
        """Net exposure / equity."""
        if self.equity_value <= 0:
            return 0.0
        return self.net_exposure / self.equity_value
    
    # === MARGIN METRICS (Futures) ===
    
    @property
    def margin_used(self) -> float:
        """Total margin used for futures."""
        return self.position_book.total_margin_used
    
    @property
    def free_cash(self) -> float:
        """Cash available after margin requirements."""
        return max(0, self.cash - self.margin_used)
    
    @property
    def margin_excess(self) -> float:
        """Excess margin available."""
        return self.cash - self.margin_used
    
    # === P/L METRICS ===
    
    @property
    def total_pnl(self) -> float:
        """Total P/L (realized + unrealized)."""
        return (
            self.position_book.total_unrealized_pnl + 
            self.position_book.total_realized_pnl
        )
    
    @property
    def total_pnl_pct(self) -> float:
        """Total P/L as percentage of initial capital."""
        if self.initial_capital <= 0:
            return 0.0
        return self.total_pnl / self.initial_capital * 100
    
    @property
    def long_pnl(self) -> float:
        """P/L from long positions."""
        return sum(p.pnl_unrealized for p in self.position_book.long_positions.values())
    
    @property
    def short_pnl(self) -> float:
        """P/L from short positions."""
        return sum(p.pnl_unrealized for p in self.position_book.short_positions.values())
    
    # === POSITION MANAGEMENT ===
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self.position_book.get_position(symbol)
    
    def execute_trade(
        self,
        instrument: Instrument,
        quantity: float,
        price: float,
        commission: float = 0.0,
        strategy: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """
        Execute a trade and update portfolio state.
        
        Args:
            instrument: The instrument to trade
            quantity: Positive for buy, negative for sell
            price: Execution price
            commission: Commission cost
            strategy: Strategy that generated the trade
        
        Returns:
            Tuple of (realized_pnl, success)
        """
        # Calculate trade value
        if isinstance(instrument, FutureInstrument):
            trade_value = abs(quantity) * instrument.multiplier * price
        else:
            trade_value = abs(quantity) * price
        
        # Check if we have enough cash for buying
        if quantity > 0:  # Buying
            if trade_value + commission > self.free_cash:
                logging.warning(
                    f"Insufficient cash for {instrument.symbol}: "
                    f"need ${trade_value:.2f}, have ${self.free_cash:.2f}"
                )
                return 0.0, False
        
        # Execute the trade
        realized_pnl = self.position_book.update_position(
            instrument=instrument,
            quantity_delta=quantity,
            price=price,
            commission=commission,
            strategy=strategy,
        )
        
        # Update cash
        if quantity > 0:  # Bought
            self.cash -= trade_value + commission
        else:  # Sold
            self.cash += trade_value - commission
        
        self.total_commission += commission
        self.as_of = datetime.now()
        
        logging.info(
            f"Executed: {'BUY' if quantity > 0 else 'SELL'} {abs(quantity)} "
            f"{instrument.symbol} @ ${price:.2f}, realized P/L: ${realized_pnl:.2f}"
        )
        
        return realized_pnl, True
    
    def update_prices(self, prices: Dict[str, float]):
        """Update market prices for all positions."""
        self.position_book.update_prices(prices)
        self.as_of = datetime.now()
    
    def apply_daily_costs(self, days: int = 1):
        """Apply daily financing and borrow costs."""
        self.position_book.apply_daily_costs(days)
        
        # Track accumulated costs
        for position in self.position_book.positions.values():
            self.total_financing += position.total_financing_cost
            self.total_borrow += position.total_borrow_cost
    
    # === TARGET WEIGHT EXECUTION ===
    
    def calculate_target_positions(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        gross_exposure_target: float = 1.0,
    ) -> Dict[str, Tuple[float, str]]:
        """
        Calculate target positions from weights.
        
        Args:
            target_weights: Dict of symbol -> weight (can be negative for shorts)
            prices: Dict of symbol -> current price
            gross_exposure_target: Target gross exposure as fraction of equity
        
        Returns:
            Dict of symbol -> (quantity_delta, side)
        """
        target_notional = self.equity_value * gross_exposure_target
        trades = {}
        
        for symbol, weight in target_weights.items():
            if symbol not in prices:
                logging.warning(f"No price for {symbol}, skipping")
                continue
            
            price = prices[symbol]
            if price <= 0:
                continue
            
            # Calculate target notional for this position
            target_pos_notional = target_notional * weight
            
            # Get current position
            current_pos = self.get_position(symbol)
            current_notional = current_pos.notional if current_pos else 0.0
            
            # Calculate notional delta
            notional_delta = target_pos_notional - current_notional
            
            # Convert to shares
            quantity_delta = int(notional_delta / price)
            
            if quantity_delta != 0:
                side = "BUY" if quantity_delta > 0 else "SELL"
                trades[symbol] = (quantity_delta, side)
        
        return trades
    
    # === SERIALIZATION ===
    
    def summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            "as_of": self.as_of.isoformat(),
            "cash": self.cash,
            "equity_value": self.equity_value,
            "initial_capital": self.initial_capital,
            
            # Positions
            "position_count": len(self.position_book.positions),
            "long_count": len(self.position_book.long_positions),
            "short_count": len(self.position_book.short_positions),
            
            # Exposure
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "long_exposure": self.long_exposure,
            "short_exposure": self.short_exposure,
            "leverage": self.leverage,
            "net_leverage": self.net_leverage,
            
            # Margin
            "margin_used": self.margin_used,
            "free_cash": self.free_cash,
            
            # P/L
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "long_pnl": self.long_pnl,
            "short_pnl": self.short_pnl,
            "unrealized_pnl": self.position_book.total_unrealized_pnl,
            "realized_pnl": self.position_book.total_realized_pnl,
            
            # Costs
            "total_commission": self.total_commission,
            "total_financing": self.total_financing,
            "total_borrow": self.total_borrow,
            
            # Position book summary
            "positions": self.position_book.summary(),
        }
    
    def positions_df(self):
        """Get positions as a list of dicts (for DataFrame conversion)."""
        return [pos.to_dict() for pos in self.position_book.positions.values()]


class PortfolioManager:
    """
    High-level portfolio management interface.
    
    Provides:
    - Portfolio state tracking
    - Order generation from target weights
    - Risk constraint enforcement
    - Daily settlement
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_gross_exposure: float = 2.0,
        max_single_position: float = 0.20,
        min_free_cash_pct: float = 0.05,
    ):
        self.state = PortfolioState(
            initial_capital=initial_capital,
            cash=initial_capital,
        )
        
        # Risk limits
        self.max_gross_exposure = max_gross_exposure
        self.max_single_position = max_single_position
        self.min_free_cash_pct = min_free_cash_pct
        
        # Instrument registry
        self.registry = get_instrument_registry()
        
        # History
        self.state_history: List[Dict] = []
    
    def get_instrument(self, symbol: str) -> Instrument:
        """Get or create instrument."""
        return self.registry.get_or_create_spot(symbol)
    
    def rebalance_to_weights(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        dry_run: bool = False,
    ) -> List[Dict]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dict of symbol -> weight (can be negative)
            prices: Current prices
            dry_run: If True, don't execute trades
        
        Returns:
            List of trade dicts
        """
        # Normalize weights to respect gross exposure limit
        total_abs_weight = sum(abs(w) for w in target_weights.values())
        if total_abs_weight > self.max_gross_exposure:
            scale = self.max_gross_exposure / total_abs_weight
            target_weights = {k: v * scale for k, v in target_weights.items()}
            logging.info(f"Scaled weights by {scale:.2f} to meet gross exposure limit")
        
        # Clip individual weights
        clipped_weights = {}
        for symbol, weight in target_weights.items():
            clipped = max(-self.max_single_position, min(self.max_single_position, weight))
            if clipped != weight:
                logging.info(f"Clipped {symbol} weight from {weight:.2%} to {clipped:.2%}")
            clipped_weights[symbol] = clipped
        
        # Calculate trades
        trades_needed = self.state.calculate_target_positions(
            clipped_weights, prices, gross_exposure_target=1.0
        )
        
        executed_trades = []
        
        for symbol, (quantity, side) in trades_needed.items():
            if dry_run:
                executed_trades.append({
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs(quantity),
                    "price": prices[symbol],
                    "dry_run": True,
                })
            else:
                instrument = self.get_instrument(symbol)
                realized_pnl, success = self.state.execute_trade(
                    instrument=instrument,
                    quantity=quantity,
                    price=prices[symbol],
                    commission=0.0,  # Add commission model later
                )
                
                if success:
                    executed_trades.append({
                        "symbol": symbol,
                        "side": side,
                        "quantity": abs(quantity),
                        "price": prices[symbol],
                        "realized_pnl": realized_pnl,
                        "dry_run": False,
                    })
        
        return executed_trades
    
    def end_of_day(self, prices: Dict[str, float]):
        """
        End of day processing.
        
        - Update prices
        - Apply daily costs
        - Record state history
        """
        self.state.update_prices(prices)
        self.state.apply_daily_costs(1)
        
        # Record history
        self.state_history.append(self.state.summary())
    
    def get_exposure_summary(self) -> Dict[str, float]:
        """Get current exposure summary."""
        return {
            "gross": self.state.gross_exposure,
            "net": self.state.net_exposure,
            "long": self.state.long_exposure,
            "short": self.state.short_exposure,
            "leverage": self.state.leverage,
            "net_leverage": self.state.net_leverage,
        }
