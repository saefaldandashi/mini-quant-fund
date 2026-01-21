"""
Execution Report - Tracks and reports execution quality metrics.

Measures:
- Price improvement vs market orders
- Fill rates for limit orders
- Execution times
- Total savings
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_FALLBACK = "limit_fallback"  # Limit that converted to market


@dataclass
class ExecutionRecord:
    """Record of a single order execution."""
    symbol: str
    side: str                    # 'buy' or 'sell'
    quantity: int
    order_type: OrderType
    
    # Prices
    decision_price: float        # Price when decision was made
    limit_price: Optional[float] # Limit price if used
    fill_price: float            # Actual fill price
    
    # Timing
    decision_time: datetime
    submit_time: datetime
    fill_time: Optional[datetime]
    
    # Spread info
    spread_pct: float
    spread_category: str
    
    # Outcomes
    filled_at_limit: bool        # True if limit order filled
    price_improvement: float     # $ saved vs market
    price_improvement_pct: float # % saved vs market
    slippage: float              # Difference from expected
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'decision_price': self.decision_price,
            'limit_price': self.limit_price,
            'fill_price': self.fill_price,
            'decision_time': self.decision_time.isoformat(),
            'submit_time': self.submit_time.isoformat(),
            'fill_time': self.fill_time.isoformat() if self.fill_time else None,
            'spread_pct': self.spread_pct,
            'spread_category': self.spread_category,
            'filled_at_limit': self.filled_at_limit,
            'price_improvement': self.price_improvement,
            'price_improvement_pct': self.price_improvement_pct,
            'slippage': self.slippage,
        }


@dataclass
class ExecutionMetrics:
    """Aggregate execution metrics."""
    total_orders: int = 0
    limit_orders: int = 0
    market_orders: int = 0
    limit_fills: int = 0            # Limits that filled without fallback
    limit_fallbacks: int = 0        # Limits that fell back to market
    
    total_value: float = 0.0        # Total $ executed
    total_improvement: float = 0.0  # Total $ saved
    avg_improvement_pct: float = 0.0
    
    avg_execution_time_ms: float = 0.0
    avg_spread_pct: float = 0.0
    
    fill_rate: float = 0.0          # % of limits that filled
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ExecutionReport:
    """
    Tracks and reports on execution quality.
    
    Maintains a history of all executions and provides
    aggregate metrics to measure the effectiveness of
    the smart execution engine.
    """
    
    def __init__(self, storage_path: str = "outputs/execution_report.json"):
        self.storage_path = Path(storage_path)
        self.records: List[ExecutionRecord] = []
        self.session_records: List[ExecutionRecord] = []  # Current session only
        self._load()
    
    def _load(self):
        """Load historical records from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Keep last 1000 records
                for record_dict in data.get('records', [])[-1000:]:
                    try:
                        record = ExecutionRecord(
                            symbol=record_dict['symbol'],
                            side=record_dict['side'],
                            quantity=record_dict['quantity'],
                            order_type=OrderType(record_dict['order_type']),
                            decision_price=record_dict['decision_price'],
                            limit_price=record_dict.get('limit_price'),
                            fill_price=record_dict['fill_price'],
                            decision_time=datetime.fromisoformat(record_dict['decision_time']),
                            submit_time=datetime.fromisoformat(record_dict['submit_time']),
                            fill_time=datetime.fromisoformat(record_dict['fill_time']) if record_dict.get('fill_time') else None,
                            spread_pct=record_dict['spread_pct'],
                            spread_category=record_dict['spread_category'],
                            filled_at_limit=record_dict['filled_at_limit'],
                            price_improvement=record_dict['price_improvement'],
                            price_improvement_pct=record_dict['price_improvement_pct'],
                            slippage=record_dict['slippage'],
                        )
                        self.records.append(record)
                    except Exception as e:
                        logging.warning(f"Could not parse execution record: {e}")
                
                logging.info(f"Loaded {len(self.records)} execution records")
            except Exception as e:
                logging.warning(f"Could not load execution report: {e}")
    
    def _save(self):
        """Persist records to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'records': [r.to_dict() for r in self.records[-1000:]],
                    'last_updated': datetime.now().isoformat(),
                    'total_historical': len(self.records),
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save execution report: {e}")
    
    def record_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        decision_price: float,
        fill_price: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        spread_pct: float = 0.0,
        spread_category: str = "unknown",
        decision_time: Optional[datetime] = None,
        submit_time: Optional[datetime] = None,
        fill_time: Optional[datetime] = None,
        filled_at_limit: bool = False,
    ):
        """
        Record an execution for tracking.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            decision_price: Price when trade decision was made (market price)
            fill_price: Actual fill price
            order_type: Type of order used
            limit_price: Limit price if applicable
            spread_pct: Spread percentage at time of order
            spread_category: Category of spread
            decision_time: When decision was made
            submit_time: When order was submitted
            fill_time: When order was filled
            filled_at_limit: Whether limit order filled at limit
        """
        now = datetime.now()
        
        # Calculate price improvement
        if side.lower() == 'buy':
            # For buys, improvement = decision_price - fill_price (positive = good)
            price_improvement = decision_price - fill_price
        else:
            # For sells, improvement = fill_price - decision_price (positive = good)
            price_improvement = fill_price - decision_price
        
        price_improvement_total = price_improvement * quantity
        
        if decision_price > 0:
            price_improvement_pct = (price_improvement / decision_price) * 100
        else:
            price_improvement_pct = 0
        
        # Calculate slippage (expected vs actual)
        if limit_price and filled_at_limit:
            slippage = abs(fill_price - limit_price)
        else:
            slippage = abs(fill_price - decision_price)
        
        record = ExecutionRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            decision_price=decision_price,
            limit_price=limit_price,
            fill_price=fill_price,
            decision_time=decision_time or now,
            submit_time=submit_time or now,
            fill_time=fill_time or now,
            spread_pct=spread_pct,
            spread_category=spread_category,
            filled_at_limit=filled_at_limit,
            price_improvement=price_improvement_total,
            price_improvement_pct=price_improvement_pct,
            slippage=slippage,
        )
        
        self.records.append(record)
        self.session_records.append(record)
        self._save()
        
        logging.info(
            f"Execution recorded: {side} {quantity} {symbol} @ ${fill_price:.2f} "
            f"(improvement: ${price_improvement_total:.2f})"
        )
    
    def get_metrics(self, session_only: bool = False) -> ExecutionMetrics:
        """
        Calculate aggregate metrics.
        
        Args:
            session_only: If True, only use current session records
        
        Returns:
            ExecutionMetrics with aggregate stats
        """
        records = self.session_records if session_only else self.records
        
        if not records:
            return ExecutionMetrics()
        
        metrics = ExecutionMetrics()
        
        metrics.total_orders = len(records)
        metrics.limit_orders = sum(1 for r in records if r.order_type in [OrderType.LIMIT, OrderType.LIMIT_FALLBACK])
        metrics.market_orders = sum(1 for r in records if r.order_type == OrderType.MARKET)
        metrics.limit_fills = sum(1 for r in records if r.order_type == OrderType.LIMIT and r.filled_at_limit)
        metrics.limit_fallbacks = sum(1 for r in records if r.order_type == OrderType.LIMIT_FALLBACK)
        
        metrics.total_value = sum(r.fill_price * r.quantity for r in records)
        metrics.total_improvement = sum(r.price_improvement for r in records)
        
        if metrics.total_value > 0:
            metrics.avg_improvement_pct = (metrics.total_improvement / metrics.total_value) * 100
        
        # Calculate average execution time
        times = []
        for r in records:
            if r.fill_time and r.submit_time:
                delta = (r.fill_time - r.submit_time).total_seconds() * 1000
                times.append(delta)
        
        if times:
            metrics.avg_execution_time_ms = sum(times) / len(times)
        
        metrics.avg_spread_pct = sum(r.spread_pct for r in records) / len(records)
        
        if metrics.limit_orders > 0:
            metrics.fill_rate = (metrics.limit_fills / metrics.limit_orders) * 100
        
        return metrics
    
    def get_report_string(self, session_only: bool = True) -> str:
        """
        Generate a human-readable execution report.
        
        Args:
            session_only: If True, report on current session only
        
        Returns:
            Formatted report string
        """
        metrics = self.get_metrics(session_only=session_only)
        
        if metrics.total_orders == 0:
            return "No executions recorded yet."
        
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║                   EXECUTION REPORT                       ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Total Orders:        {metrics.total_orders:>6}                            ║",
            f"║  Limit Orders:        {metrics.limit_orders:>6}                            ║",
            f"║  Market Orders:       {metrics.market_orders:>6}                            ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Limit Fill Rate:     {metrics.fill_rate:>6.1f}%                           ║",
            f"║  Limit Fills:         {metrics.limit_fills:>6}                            ║",
            f"║  Limit Fallbacks:     {metrics.limit_fallbacks:>6}                            ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Total Value:         ${metrics.total_value:>12,.2f}                  ║",
            f"║  Price Improvement:   ${metrics.total_improvement:>12,.2f}                  ║",
            f"║  Avg Improvement:     {metrics.avg_improvement_pct:>6.3f}%                           ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Avg Execution Time:  {metrics.avg_execution_time_ms:>6.0f}ms                           ║",
            f"║  Avg Spread:          {metrics.avg_spread_pct:>6.3f}%                           ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
        ]
        
        return "\n".join(lines)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session executions."""
        metrics = self.get_metrics(session_only=True)
        return {
            'orders_executed': metrics.total_orders,
            'limit_orders': metrics.limit_orders,
            'market_orders': metrics.market_orders,
            'fill_rate': metrics.fill_rate,
            'total_value': metrics.total_value,
            'price_improvement': metrics.total_improvement,
            'avg_improvement_pct': metrics.avg_improvement_pct,
        }
    
    def clear_session(self):
        """Clear session records (keep historical)."""
        self.session_records = []
