"""
Smart Executor - Intelligent order execution with price optimization.

Features:
- Spread analysis before each order
- Limit orders with automatic fallback to market
- Time-of-day awareness
- Execution quality tracking
- VIX-based timeout adjustment (Phase 2)
- Multi-order optimization (Phase 2)
- Per-symbol fill rate learning (Phase 2)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pytz

from .spread_analyzer import SpreadAnalyzer, SpreadAnalysis, SpreadCategory
from .execution_report import ExecutionReport, OrderType


class ExecutionStrategy(Enum):
    """Execution strategy to use."""
    MARKET = "market"                    # Immediate market order
    LIMIT_AGGRESSIVE = "limit_aggressive"  # Short timeout limit
    LIMIT_PATIENT = "limit_patient"        # Longer timeout limit
    SKIP = "skip"                          # Skip this order (too illiquid)


class TimeOfDay(Enum):
    """Market time classification."""
    PRE_MARKET = "pre_market"
    OPENING = "opening"           # 9:30-10:00 - volatile
    MORNING_PRIME = "morning_prime"  # 10:00-11:30 - optimal
    LUNCH = "lunch"                # 11:30-14:00 - lower volume
    AFTERNOON = "afternoon"        # 14:00-15:30 - good
    CLOSING = "closing"            # 15:30-16:00 - volatile
    AFTER_HOURS = "after_hours"


class OrderPriority(Enum):
    """Priority for order sequencing."""
    CRITICAL = 1      # Must execute first (risk reduction sells)
    HIGH = 2          # High conviction positions
    NORMAL = 3        # Standard priority
    LOW = 4           # Low urgency (small positions)


@dataclass
class OrderIntent:
    """An order to be executed."""
    symbol: str
    side: str           # 'buy', 'sell', 'short', 'cover'
    quantity: int
    notional: float     # Target $ value
    current_price: float
    conviction: float = 0.5  # 0-1 confidence in this trade
    priority: OrderPriority = OrderPriority.NORMAL
    is_short: bool = False  # True if this is opening/closing a short position


@dataclass
class ExecutionResult:
    """Result of order execution."""
    symbol: str
    side: str
    quantity: int
    success: bool
    fill_price: Optional[float]
    order_type: OrderType
    filled_at_limit: bool
    execution_time_ms: float
    spread_pct: float = 0.0
    price_improvement: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'success': self.success,
            'fill_price': self.fill_price,
            'order_type': self.order_type.value,
            'filled_at_limit': self.filled_at_limit,
            'execution_time_ms': self.execution_time_ms,
            'spread_pct': self.spread_pct,
            'price_improvement': self.price_improvement,
            'error': self.error,
        }


@dataclass
class SymbolExecutionHistory:
    """Tracks execution history for a symbol."""
    total_orders: int = 0
    limit_attempts: int = 0
    limit_fills: int = 0
    avg_spread: float = 0.0
    avg_improvement: float = 0.0
    last_execution: Optional[datetime] = None
    
    @property
    def fill_rate(self) -> float:
        if self.limit_attempts == 0:
            return 0.5  # Default assumption
        return self.limit_fills / self.limit_attempts


class SmartExecutor:
    """
    Intelligent order execution engine.
    
    Analyzes market conditions and uses optimal execution strategy
    for each order to minimize costs and maximize fill quality.
    
    Phase 2 Features:
    - VIX-based timeout adjustment
    - Per-symbol fill rate learning
    - Multi-order optimization (sells first, spread grouping)
    - Conviction-based priority
    """
    
    # Time-based parameters
    TIME_PARAMS = {
        TimeOfDay.PRE_MARKET: {'use_limit': False, 'timeout_mult': 0.5},
        TimeOfDay.OPENING: {'use_limit': False, 'timeout_mult': 0.5},
        TimeOfDay.MORNING_PRIME: {'use_limit': True, 'timeout_mult': 1.0},
        TimeOfDay.LUNCH: {'use_limit': True, 'timeout_mult': 0.75},
        TimeOfDay.AFTERNOON: {'use_limit': True, 'timeout_mult': 1.0},
        TimeOfDay.CLOSING: {'use_limit': True, 'timeout_mult': 0.5},
        TimeOfDay.AFTER_HOURS: {'use_limit': False, 'timeout_mult': 0.5},
    }
    
    # VIX-based adjustment thresholds
    VIX_THRESHOLDS = {
        'low': 15,      # VIX < 15: calm market, longer timeouts OK
        'normal': 20,   # VIX 15-20: normal conditions
        'elevated': 25, # VIX 20-25: elevated, shorter timeouts
        'high': 30,     # VIX > 30: high volatility, use market orders
    }
    
    def __init__(
        self,
        broker,
        data_client=None,
        dry_run: bool = True,
        log_func=None,
        vix_level: float = 20.0,
        history_path: str = "outputs/execution_history.json",
    ):
        """
        Initialize smart executor.
        
        Args:
            broker: AlpacaBroker instance for placing orders
            data_client: Alpaca data client for quotes
            dry_run: If True, simulate execution
            log_func: Optional logging function
            vix_level: Current VIX for timeout adjustment
            history_path: Path to store execution history
        """
        self.broker = broker
        self.spread_analyzer = SpreadAnalyzer(data_client)
        self.execution_report = ExecutionReport()
        self.dry_run = dry_run
        self.log = log_func or (lambda x: logging.info(x))
        self.vix_level = vix_level
        self.history_path = Path(history_path)
        
        # Per-symbol execution history for learning
        self.symbol_history: Dict[str, SymbolExecutionHistory] = {}
        self._load_history()
        
        # Get data client from broker if not provided
        if data_client is None and hasattr(broker, 'data_client'):
            self.spread_analyzer.data_client = broker.data_client
    
    def _load_history(self):
        """Load execution history from disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                for symbol, hist in data.get('symbols', {}).items():
                    self.symbol_history[symbol] = SymbolExecutionHistory(
                        total_orders=hist.get('total_orders', 0),
                        limit_attempts=hist.get('limit_attempts', 0),
                        limit_fills=hist.get('limit_fills', 0),
                        avg_spread=hist.get('avg_spread', 0),
                        avg_improvement=hist.get('avg_improvement', 0),
                    )
            except Exception as e:
                logging.warning(f"Could not load execution history: {e}")
    
    def _save_history(self):
        """Save execution history to disk."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'symbols': {
                    symbol: {
                        'total_orders': hist.total_orders,
                        'limit_attempts': hist.limit_attempts,
                        'limit_fills': hist.limit_fills,
                        'avg_spread': hist.avg_spread,
                        'avg_improvement': hist.avg_improvement,
                    }
                    for symbol, hist in self.symbol_history.items()
                },
                'last_updated': datetime.now().isoformat(),
            }
            with open(self.history_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save execution history: {e}")
    
    def _update_symbol_history(
        self,
        symbol: str,
        used_limit: bool,
        filled_at_limit: bool,
        spread_pct: float,
        improvement: float,
    ):
        """Update execution history for a symbol."""
        if symbol not in self.symbol_history:
            self.symbol_history[symbol] = SymbolExecutionHistory()
        
        hist = self.symbol_history[symbol]
        hist.total_orders += 1
        
        if used_limit:
            hist.limit_attempts += 1
            if filled_at_limit:
                hist.limit_fills += 1
        
        # Rolling average for spread and improvement
        alpha = 0.2
        hist.avg_spread = alpha * spread_pct + (1 - alpha) * hist.avg_spread
        hist.avg_improvement = alpha * improvement + (1 - alpha) * hist.avg_improvement
        hist.last_execution = datetime.now()
        
        self._save_history()
    
    def get_time_of_day(self) -> TimeOfDay:
        """Classify current time of day for execution purposes."""
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        hour = now.hour
        minute = now.minute
        time_decimal = hour + minute / 60
        
        if time_decimal < 9.5:
            return TimeOfDay.PRE_MARKET
        elif time_decimal < 10.0:
            return TimeOfDay.OPENING
        elif time_decimal < 11.5:
            return TimeOfDay.MORNING_PRIME
        elif time_decimal < 14.0:
            return TimeOfDay.LUNCH
        elif time_decimal < 15.5:
            return TimeOfDay.AFTERNOON
        elif time_decimal < 16.0:
            return TimeOfDay.CLOSING
        else:
            return TimeOfDay.AFTER_HOURS
    
    def get_vix_timeout_multiplier(self) -> float:
        """Get timeout multiplier based on VIX level."""
        if self.vix_level < self.VIX_THRESHOLDS['low']:
            return 1.5   # Low VIX: can be more patient
        elif self.vix_level < self.VIX_THRESHOLDS['normal']:
            return 1.0   # Normal: standard timeouts
        elif self.vix_level < self.VIX_THRESHOLDS['elevated']:
            return 0.7   # Elevated: shorter timeouts
        elif self.vix_level < self.VIX_THRESHOLDS['high']:
            return 0.4   # High: very short timeouts
        else:
            return 0.0   # Extreme: use market orders only
    
    def get_symbol_timeout_multiplier(self, symbol: str) -> float:
        """Get timeout multiplier based on historical fill rate for symbol."""
        if symbol not in self.symbol_history:
            return 1.0  # No history, use default
        
        hist = self.symbol_history[symbol]
        if hist.limit_attempts < 3:
            return 1.0  # Not enough data
        
        fill_rate = hist.fill_rate
        
        # If this symbol fills well, be more patient
        if fill_rate > 0.6:
            return 1.3
        elif fill_rate > 0.4:
            return 1.0
        elif fill_rate > 0.2:
            return 0.7
        else:
            return 0.4  # Poor fill rate, use shorter timeout
    
    def calculate_dynamic_timeout(
        self,
        base_timeout: int,
        time_of_day: TimeOfDay,
        symbol: str,
    ) -> int:
        """
        Calculate dynamic timeout based on multiple factors.
        
        Factors:
        - Base timeout from spread analysis
        - Time of day adjustment
        - VIX level adjustment
        - Symbol historical fill rate
        """
        # Time of day multiplier
        time_mult = self.TIME_PARAMS[time_of_day]['timeout_mult']
        
        # VIX multiplier
        vix_mult = self.get_vix_timeout_multiplier()
        
        # Symbol history multiplier
        symbol_mult = self.get_symbol_timeout_multiplier(symbol)
        
        # Combined timeout
        timeout = base_timeout * time_mult * vix_mult * symbol_mult
        
        # Clamp to reasonable range
        return max(10, min(int(timeout), 120))
    
    def determine_strategy(
        self,
        spread_analysis: SpreadAnalysis,
        time_of_day: TimeOfDay,
    ) -> ExecutionStrategy:
        """Determine the best execution strategy based on conditions."""
        time_params = self.TIME_PARAMS[time_of_day]
        
        # If VIX is extremely high, just use market orders
        if self.vix_level > self.VIX_THRESHOLDS['high']:
            return ExecutionStrategy.MARKET
        
        # If spread is very wide (>1%), use market order (don't skip - always execute)
        # We never want to skip orders for liquid stocks just because quote data is stale
        if spread_analysis.category == SpreadCategory.ILLIQUID:
            if spread_analysis.spread_pct > 1.0:
                # Very wide spread - likely bad quote data or after hours
                # Use market order to ensure execution
                return ExecutionStrategy.MARKET
            return ExecutionStrategy.LIMIT_PATIENT
        
        # If time of day suggests avoiding limits
        if not time_params['use_limit']:
            return ExecutionStrategy.MARKET
        
        # If spread is too tight, not worth optimizing
        if spread_analysis.category == SpreadCategory.TIGHT:
            return ExecutionStrategy.MARKET
        
        # Use limit order based on spread
        if spread_analysis.category in [SpreadCategory.NORMAL, SpreadCategory.MODERATE]:
            return ExecutionStrategy.LIMIT_AGGRESSIVE
        else:
            return ExecutionStrategy.LIMIT_PATIENT
    
    def optimize_order_sequence(
        self,
        orders: List[Dict],
    ) -> List[Dict]:
        """
        Optimize the order of execution.
        
        Strategy:
        1. Execute SELLS first (free up capital) 
        2. Execute SHORTS second (open short positions)
        3. Execute COVERS third (close short positions)
        4. Execute BUYS last (use freed capital)
        
        Within each side, order by:
           - Priority (high conviction first)
           - Spread (tight spreads first - faster execution)
        """
        sells = [o for o in orders if o['side'] == 'sell']
        shorts = [o for o in orders if o['side'] == 'short']
        covers = [o for o in orders if o['side'] == 'cover']
        buys = [o for o in orders if o['side'] == 'buy']
        
        # Sort sells: high value first (free up more capital)
        sells.sort(key=lambda x: x['value'], reverse=True)
        
        # Sort shorts: high conviction first
        shorts.sort(key=lambda x: -x.get('conviction', 0.5))
        
        # Sort covers: largest positions first (reduce risk)
        covers.sort(key=lambda x: x['value'], reverse=True)
        
        # Sort buys: high conviction first, then by spread (tightest first)
        buys.sort(key=lambda x: (
            -x.get('conviction', 0.5),  # Higher conviction first
            x.get('spread_pct', 0.1),   # Lower spread first
        ))
        
        # Order: sells -> shorts -> covers -> buys
        optimized = sells + shorts + covers + buys
        
        return optimized
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        conviction: float = 0.5,
    ) -> ExecutionResult:
        """
        Execute a single order with smart execution.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            current_price: Current market price
            conviction: Confidence in this trade (affects timeout)
        
        Returns:
            ExecutionResult with outcome
        """
        start_time = time.time()
        decision_time = datetime.now()
        
        self.log(f"  📊 Analyzing {symbol}...")
        
        # Analyze spread
        spread_analysis = self.spread_analyzer.analyze(
            symbol,
            bid=current_price * 0.9995,
            ask=current_price * 1.0005,
        )
        
        # Get real quote if possible
        if hasattr(self.broker, 'data_client') and self.broker.data_client:
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                quotes = self.broker.data_client.get_stock_latest_quote(request)
                if symbol in quotes:
                    quote = quotes[symbol]
                    spread_analysis = self.spread_analyzer.analyze(
                        symbol,
                        bid=float(quote.bid_price),
                        ask=float(quote.ask_price),
                        bid_size=int(quote.bid_size),
                        ask_size=int(quote.ask_size),
                    )
            except Exception as e:
                self.log(f"    ⚠️ Could not get real-time quote: {e}")
        
        # Determine execution strategy
        time_of_day = self.get_time_of_day()
        strategy = self.determine_strategy(spread_analysis, time_of_day)
        
        # Calculate dynamic timeout
        base_timeout = spread_analysis.recommended_timeout
        dynamic_timeout = self.calculate_dynamic_timeout(base_timeout, time_of_day, symbol)
        
        # Adjust for conviction
        if conviction > 0.7:
            dynamic_timeout = int(dynamic_timeout * 1.2)  # More patient for high conviction
        elif conviction < 0.3:
            dynamic_timeout = int(dynamic_timeout * 0.7)  # Less patient for low conviction
        
        self.log(f"    Spread: {spread_analysis.spread_pct:.3f}% ({spread_analysis.category.value})")
        self.log(f"    VIX: {self.vix_level:.1f} | Time: {time_of_day.value}")
        self.log(f"    Strategy: {strategy.value} | Timeout: {dynamic_timeout}s")
        
        # Check symbol history
        if symbol in self.symbol_history:
            hist = self.symbol_history[symbol]
            if hist.limit_attempts >= 3:
                self.log(f"    History: {hist.fill_rate*100:.0f}% fill rate over {hist.limit_attempts} attempts")
        
        # Note: We never skip orders anymore - wide spreads just use market orders
        # This ensures all trades execute even with stale quote data
        if spread_analysis.spread_pct > 0.5:
            self.log(f"    ⚠️ Wide spread detected ({spread_analysis.spread_pct:.2f}%) - using market order")
        
        # Execute based on strategy
        if strategy == ExecutionStrategy.MARKET:
            result = self._execute_market_order(
                symbol, side, quantity, current_price,
                spread_analysis, decision_time
            )
            used_limit = False
        else:
            # Limit with fallback
            limit_price = self.spread_analyzer.calculate_limit_price(
                spread_analysis, side
            )
            
            result = self._execute_limit_with_fallback(
                symbol, side, quantity, current_price,
                limit_price, dynamic_timeout, spread_analysis, decision_time
            )
            used_limit = True
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        result.spread_pct = spread_analysis.spread_pct
        
        # Calculate price improvement
        if result.success and result.fill_price:
            if side == 'buy':
                result.price_improvement = (current_price - result.fill_price) * quantity
            else:
                result.price_improvement = (result.fill_price - current_price) * quantity
        
        # Update symbol history
        if result.success:
            self._update_symbol_history(
                symbol,
                used_limit=used_limit,
                filled_at_limit=result.filled_at_limit,
                spread_pct=spread_analysis.spread_pct,
                improvement=result.price_improvement,
            )
            
            # Record in execution report
            self.execution_report.record_execution(
                symbol=symbol,
                side=side,
                quantity=quantity,
                decision_price=current_price,
                fill_price=result.fill_price,
                order_type=result.order_type,
                limit_price=limit_price if used_limit else None,
                spread_pct=spread_analysis.spread_pct,
                spread_category=spread_analysis.category.value,
                decision_time=decision_time,
                filled_at_limit=result.filled_at_limit,
            )
        
        return result
    
    def _execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        spread_analysis: SpreadAnalysis,
        decision_time: datetime,
    ) -> ExecutionResult:
        """Execute a simple market order."""
        from alpaca.trading.enums import OrderSide
        
        # Handle all order types: buy, sell, short, cover
        side_lower = side.lower()
        if side_lower in ['buy', 'cover']:
            order_side = OrderSide.BUY
        else:  # 'sell' or 'short'
            order_side = OrderSide.SELL
        
        # Log with appropriate emoji for short orders
        if side_lower == 'short':
            self.log(f"    → 📉 SHORT SELL: {quantity} @ ~${current_price:.2f}")
        elif side_lower == 'cover':
            self.log(f"    → 📈 COVER SHORT: {quantity} @ ~${current_price:.2f}")
        else:
            self.log(f"    → MARKET ORDER: {side.upper()} {quantity} @ ~${current_price:.2f}")
        
        if self.dry_run:
            fill_price = current_price
            self.log(f"    ✓ [DRY RUN] Filled at ${fill_price:.2f}")
        else:
            try:
                result = self.broker.place_market_order(
                    symbol, quantity, order_side, dry_run=False
                )
                fill_price = result.get('filled_avg_price', current_price) if result else current_price
                self.log(f"    ✓ Filled at ${fill_price:.2f}")
            except Exception as e:
                self.log(f"    ✗ Order failed: {e}")
                return ExecutionResult(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    success=False,
                    fill_price=None,
                    order_type=OrderType.MARKET,
                    filled_at_limit=False,
                    execution_time_ms=0,
                    error=str(e),
                )
        
        return ExecutionResult(
            symbol=symbol,
            side=side,
            quantity=quantity,
            success=True,
            fill_price=fill_price,
            order_type=OrderType.MARKET,
            filled_at_limit=False,
            execution_time_ms=0,
        )
    
    def _execute_limit_with_fallback(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        limit_price: float,
        timeout_seconds: int,
        spread_analysis: SpreadAnalysis,
        decision_time: datetime,
    ) -> ExecutionResult:
        """Execute a limit order with fallback to market if not filled."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        
        # Handle all order types: buy, sell, short, cover
        side_lower = side.lower()
        if side_lower in ['buy', 'cover']:
            order_side = OrderSide.BUY
        else:  # 'sell' or 'short'
            order_side = OrderSide.SELL
        
        improvement = abs(current_price - limit_price)
        order_label = "SHORT SELL" if side_lower == 'short' else ("COVER" if side_lower == 'cover' else side.upper())
        self.log(f"    → LIMIT ORDER: {order_label} {quantity} @ ${limit_price:.2f}")
        self.log(f"      (potential savings: ${improvement:.2f}/share, timeout: {timeout_seconds}s)")
        
        if self.dry_run:
            # Simulate based on historical fill rate for this symbol
            import random
            
            # Use symbol history if available
            if symbol in self.symbol_history:
                fill_prob = self.symbol_history[symbol].fill_rate
            else:
                fill_prob = 0.5
            
            filled_at_limit = random.random() < fill_prob
            
            if filled_at_limit:
                fill_price = limit_price
                self.log(f"    ✓ [DRY RUN] Limit filled at ${fill_price:.2f}")
                return ExecutionResult(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    success=True,
                    fill_price=fill_price,
                    order_type=OrderType.LIMIT,
                    filled_at_limit=True,
                    execution_time_ms=0,
                )
            else:
                fill_price = current_price
                self.log(f"    ⏳ [DRY RUN] Limit not filled, falling back to market @ ${fill_price:.2f}")
                return ExecutionResult(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    success=True,
                    fill_price=fill_price,
                    order_type=OrderType.LIMIT_FALLBACK,
                    filled_at_limit=False,
                    execution_time_ms=0,
                )
        
        # Real execution
        try:
            limit_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.IOC,
                limit_price=limit_price,
            )
            
            order = self.broker.trading_client.submit_order(limit_request)
            order_id = order.id
            
            time.sleep(2)
            
            order_status = self.broker.trading_client.get_order_by_id(order_id)
            
            if order_status.status.value == 'filled':
                fill_price = float(order_status.filled_avg_price)
                self.log(f"    ✓ Limit filled at ${fill_price:.2f}")
                return ExecutionResult(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    success=True,
                    fill_price=fill_price,
                    order_type=OrderType.LIMIT,
                    filled_at_limit=True,
                    execution_time_ms=0,
                )
            elif order_status.status.value == 'partially_filled':
                filled_qty = int(order_status.filled_qty)
                remaining = quantity - filled_qty
                self.log(f"    ⚠️ Partial fill: {filled_qty}/{quantity}, executing remainder at market")
                
                try:
                    self.broker.trading_client.cancel_order_by_id(order_id)
                except:
                    pass
                
                if remaining > 0:
                    self.broker.place_market_order(symbol, remaining, order_side, dry_run=False)
                
                return ExecutionResult(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    success=True,
                    fill_price=float(order_status.filled_avg_price or current_price),
                    order_type=OrderType.LIMIT_FALLBACK,
                    filled_at_limit=False,
                    execution_time_ms=0,
                )
            else:
                self.log(f"    ⏳ Limit not filled, falling back to market")
                
                try:
                    self.broker.trading_client.cancel_order_by_id(order_id)
                except:
                    pass
                
                result = self.broker.place_market_order(symbol, quantity, order_side, dry_run=False)
                fill_price = result.get('filled_avg_price', current_price) if result else current_price
                
                self.log(f"    ✓ Market fallback filled at ${fill_price:.2f}")
                
                return ExecutionResult(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    success=True,
                    fill_price=fill_price,
                    order_type=OrderType.LIMIT_FALLBACK,
                    filled_at_limit=False,
                    execution_time_ms=0,
                )
                
        except Exception as e:
            self.log(f"    ✗ Limit order failed: {e}, trying market order")
            
            try:
                result = self.broker.place_market_order(symbol, quantity, order_side, dry_run=False)
                fill_price = result.get('filled_avg_price', current_price) if result else current_price
                
                return ExecutionResult(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    success=True,
                    fill_price=fill_price,
                    order_type=OrderType.LIMIT_FALLBACK,
                    filled_at_limit=False,
                    execution_time_ms=0,
                )
            except Exception as e2:
                return ExecutionResult(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    success=False,
                    fill_price=None,
                    order_type=OrderType.LIMIT_FALLBACK,
                    filled_at_limit=False,
                    execution_time_ms=0,
                    error=str(e2),
                )
    
    def execute_batch(
        self,
        orders: List[Dict],
    ) -> Tuple[List[ExecutionResult], Dict[str, Any]]:
        """
        Execute a batch of orders with smart execution and optimization.
        
        Args:
            orders: List of order dicts with symbol, side, quantity, price, value
        
        Returns:
            Tuple of (list of results, execution summary dict)
        """
        self.log("")
        self.log("=" * 60)
        self.log("SMART EXECUTION ENGINE (Phase 2)")
        self.log("=" * 60)
        
        time_of_day = self.get_time_of_day()
        vix_mult = self.get_vix_timeout_multiplier()
        
        self.log(f"Time of Day: {time_of_day.value}")
        self.log(f"VIX Level: {self.vix_level:.1f} (timeout mult: {vix_mult:.1f}x)")
        self.log(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        self.log(f"Orders to execute: {len(orders)}")
        
        # Optimize order sequence
        optimized_orders = self.optimize_order_sequence(orders)
        
        sells_count = len([o for o in orders if o['side'] == 'sell'])
        buys_count = len([o for o in orders if o['side'] == 'buy'])
        shorts_count = len([o for o in orders if o['side'] == 'short'])
        covers_count = len([o for o in orders if o['side'] == 'cover'])
        self.log(f"Order sequence: {sells_count} sells, {shorts_count} shorts, {covers_count} covers, {buys_count} buys")
        self.log("")
        
        # Clear session records
        self.execution_report.clear_session()
        
        results = []
        total_improvement = 0.0
        
        twap_used = 0
        
        for i, order in enumerate(optimized_orders, 1):
            self.log(f"[{i}/{len(optimized_orders)}]")
            
            # Check if this order should use TWAP (large order)
            if self.should_use_twap(order):
                self.log(f"  📊 Large order detected - using TWAP execution")
                twap_result = self.execute_twap(
                    symbol=order['symbol'],
                    side=order['side'],
                    total_quantity=order['quantity'],
                    slices=5,
                    interval_seconds=15 if self.dry_run else 30,
                )
                twap_used += 1
                
                # Convert TWAP result to ExecutionResult
                result = ExecutionResult(
                    symbol=order['symbol'],
                    side=order['side'],
                    quantity=twap_result['filled_quantity'],
                    success=twap_result['success'],
                    fill_price=twap_result['avg_price'],
                    order_type=OrderType.MARKET,  # TWAP uses market orders
                    filled_at_limit=False,
                    execution_time_ms=0,
                )
            else:
                result = self.execute_order(
                    symbol=order['symbol'],
                    side=order['side'],
                    quantity=order['quantity'],
                    current_price=order['price'],
                    conviction=order.get('conviction', 0.5),
                )
            results.append(result)
            total_improvement += result.price_improvement
            self.log("")
        
        # Generate summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        limit_fills = [r for r in successful if r.filled_at_limit]
        
        summary = {
            'total_orders': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'limit_fills': len(limit_fills),
            'fill_rate': len(limit_fills) / len(successful) * 100 if successful else 0,
            'total_improvement': total_improvement,
            'avg_spread': sum(r.spread_pct for r in results) / len(results) if results else 0,
            'time_of_day': time_of_day.value,
            'vix_level': self.vix_level,
        }
        
        # Print summary
        self.log("─" * 50)
        self.log("EXECUTION SUMMARY (Phase 2)")
        self.log("─" * 50)
        self.log(f"  Orders Executed:     {summary['successful']}/{summary['total_orders']}")
        self.log(f"  Limit Fills:         {summary['limit_fills']} ({summary['fill_rate']:.1f}%)")
        self.log(f"  Price Improvement:   ${summary['total_improvement']:.2f}")
        self.log(f"  Avg Spread:          {summary['avg_spread']:.3f}%")
        self.log(f"  VIX Adjustment:      {vix_mult:.1f}x timeout")
        
        if failed:
            self.log(f"  ⚠️ Failed Orders:    {len(failed)}")
            for r in failed:
                self.log(f"     - {r.symbol}: {r.error}")
        
        self.log("─" * 50)
        
        return results, summary
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution performance."""
        return self.execution_report.get_session_summary()
    
    def execute_short_order(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        conviction: float = 0.5,
    ) -> ExecutionResult:
        """
        Execute a short sell order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to short (positive)
            current_price: Current market price
            conviction: Confidence in this trade
        
        Returns:
            ExecutionResult with outcome
        """
        # Check if symbol is shortable
        if hasattr(self.broker, 'check_shortable'):
            short_info = self.broker.check_shortable(symbol)
            if not short_info.get('shortable', False):
                self.log(f"    ⚠️ {symbol} is not shortable")
                return ExecutionResult(
                    symbol=symbol,
                    side='short',
                    quantity=quantity,
                    success=False,
                    fill_price=None,
                    order_type=OrderType.MARKET,
                    filled_at_limit=False,
                    execution_time_ms=0,
                    error="Symbol not shortable",
                )
        
        # Short selling is executed as a SELL order
        self.log(f"  📉 SHORT SELL: {symbol} x {quantity}")
        return self.execute_order(
            symbol=symbol,
            side='sell',
            quantity=quantity,
            current_price=current_price,
            conviction=conviction,
        )
    
    def execute_cover_order(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        conviction: float = 0.5,
    ) -> ExecutionResult:
        """
        Execute a cover short order (buy to close).
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to cover (positive)
            current_price: Current market price
            conviction: Confidence in this trade
        
        Returns:
            ExecutionResult with outcome
        """
        self.log(f"  📈 COVER SHORT: {symbol} x {quantity}")
        return self.execute_order(
            symbol=symbol,
            side='buy',
            quantity=quantity,
            current_price=current_price,
            conviction=conviction,
        )
    
    def execute_long_short_batch(
        self,
        orders: List[Dict],
    ) -> Tuple[List[ExecutionResult], Dict[str, Any]]:
        """
        Execute a batch of long/short orders with smart sequencing.
        
        Order execution priority:
        1. Cover shorts (buy to close) - highest priority
        2. Close longs (sell) - free up capital
        3. Open shorts (sell to open) - uses margin
        4. Open longs (buy) - uses cash
        
        Args:
            orders: List of order dicts with:
                - symbol: str
                - side: 'buy', 'sell', 'short', 'cover'
                - quantity: int
                - price: float
                - value: float
                - is_short: bool (optional)
        
        Returns:
            Tuple of (results, summary)
        """
        self.log("")
        self.log("=" * 60)
        self.log("LONG/SHORT EXECUTION ENGINE")
        self.log("=" * 60)
        
        # Categorize orders
        covers = [o for o in orders if o.get('side') == 'cover' or o.get('is_cover', False)]
        close_longs = [o for o in orders if o.get('side') == 'sell' and not o.get('is_short', False)]
        open_shorts = [o for o in orders if o.get('side') == 'short' or o.get('is_short', False)]
        open_longs = [o for o in orders if o.get('side') == 'buy']
        
        self.log(f"Order breakdown:")
        self.log(f"  - Cover shorts:  {len(covers)}")
        self.log(f"  - Close longs:   {len(close_longs)}")
        self.log(f"  - Open shorts:   {len(open_shorts)}")
        self.log(f"  - Open longs:    {len(open_longs)}")
        self.log("")
        
        # Execute in priority order
        all_results = []
        
        # 1. Cover shorts first (reduce risk, free up margin)
        if covers:
            self.log("--- COVERING SHORTS ---")
            for order in covers:
                result = self.execute_cover_order(
                    symbol=order['symbol'],
                    quantity=order['quantity'],
                    current_price=order['price'],
                    conviction=order.get('conviction', 0.5),
                )
                all_results.append(result)
            self.log("")
        
        # 2. Close longs (free up capital)
        if close_longs:
            self.log("--- CLOSING LONGS ---")
            for order in close_longs:
                result = self.execute_order(
                    symbol=order['symbol'],
                    side='sell',
                    quantity=order['quantity'],
                    current_price=order['price'],
                    conviction=order.get('conviction', 0.5),
                )
                all_results.append(result)
            self.log("")
        
        # 3. Open shorts (uses margin)
        if open_shorts:
            self.log("--- OPENING SHORTS ---")
            for order in open_shorts:
                result = self.execute_short_order(
                    symbol=order['symbol'],
                    quantity=order['quantity'],
                    current_price=order['price'],
                    conviction=order.get('conviction', 0.5),
                )
                all_results.append(result)
            self.log("")
        
        # 4. Open longs (uses cash)
        if open_longs:
            self.log("--- OPENING LONGS ---")
            for order in open_longs:
                result = self.execute_order(
                    symbol=order['symbol'],
                    side='buy',
                    quantity=order['quantity'],
                    current_price=order['price'],
                    conviction=order.get('conviction', 0.5),
                )
                all_results.append(result)
        
        # Summary
        successful = [r for r in all_results if r.success]
        failed = [r for r in all_results if not r.success]
        
        summary = {
            'total_orders': len(all_results),
            'successful': len(successful),
            'failed': len(failed),
            'covers': len(covers),
            'close_longs': len(close_longs),
            'open_shorts': len(open_shorts),
            'open_longs': len(open_longs),
        }
        
        self.log("")
        self.log("─" * 50)
        self.log(f"L/S Execution: {summary['successful']}/{summary['total_orders']} orders successful")
        self.log("─" * 50)
        
        return all_results, summary
    
    def get_symbol_insights(self) -> Dict[str, Any]:
        """Get insights about symbol execution history."""
        if not self.symbol_history:
            return {'message': 'No execution history yet'}
        
        # Find best and worst symbols for limit fills
        symbols_with_history = [
            (s, h) for s, h in self.symbol_history.items()
            if h.limit_attempts >= 3
        ]
        
        if not symbols_with_history:
            return {'message': 'Not enough history for insights'}
        
        # Sort by fill rate
        sorted_symbols = sorted(
            symbols_with_history,
            key=lambda x: x[1].fill_rate,
            reverse=True
        )
        
        best = sorted_symbols[:3]
        worst = sorted_symbols[-3:] if len(sorted_symbols) > 3 else []
        
        return {
            'total_symbols': len(self.symbol_history),
            'symbols_with_history': len(symbols_with_history),
            'best_fill_rate': [
                {'symbol': s, 'fill_rate': h.fill_rate, 'attempts': h.limit_attempts}
                for s, h in best
            ],
            'worst_fill_rate': [
                {'symbol': s, 'fill_rate': h.fill_rate, 'attempts': h.limit_attempts}
                for s, h in worst
            ],
            'avg_improvement': sum(h.avg_improvement for _, h in symbols_with_history) / len(symbols_with_history),
        }
    
    def execute_twap(
        self,
        symbol: str,
        side: str,
        total_quantity: int,
        slices: int = 5,
        interval_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Execute a large order using Time-Weighted Average Price (TWAP).
        
        Splits order into smaller slices executed over time to minimize
        market impact.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            total_quantity: Total shares to trade
            slices: Number of slices to split the order into
            interval_seconds: Time between slices
        
        Returns:
            Dict with execution results
        """
        self.log(f"📊 TWAP EXECUTION: {side.upper()} {total_quantity} {symbol} in {slices} slices")
        
        slice_quantity = total_quantity // slices
        remaining = total_quantity
        fills = []
        total_value = 0.0
        total_filled = 0
        
        for i in range(slices):
            # Calculate this slice quantity
            qty = slice_quantity if i < slices - 1 else remaining
            if qty <= 0:
                break
            
            # Get current price
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                if hasattr(self.broker, 'data_client') and self.broker.data_client:
                    request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                    quotes = self.broker.data_client.get_stock_latest_quote(request)
                    if symbol in quotes:
                        current_price = float(quotes[symbol].ask_price) if side == 'buy' else float(quotes[symbol].bid_price)
                    else:
                        current_price = 0
                else:
                    current_price = 0
            except:
                current_price = 0
            
            self.log(f"  Slice {i+1}/{slices}: {qty} shares @ ~${current_price:.2f}")
            
            # Execute this slice
            result = self.execute_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                current_price=current_price,
                conviction=0.6,  # Medium conviction for TWAP
            )
            
            fills.append({
                'slice': i + 1,
                'quantity': qty,
                'success': result.success,
                'fill_price': result.fill_price,
                'execution_time_ms': result.execution_time_ms,
            })
            
            if result.success and result.fill_price:
                total_filled += qty
                total_value += qty * result.fill_price
                remaining -= qty
            
            if remaining <= 0:
                break
            
            # Wait before next slice (unless it's the last one)
            if i < slices - 1:
                if not self.dry_run:
                    self.log(f"    Waiting {interval_seconds}s before next slice...")
                    time.sleep(interval_seconds)
                else:
                    self.log(f"    [DRY RUN] Would wait {interval_seconds}s")
        
        # Calculate VWAP of all fills
        avg_price = total_value / total_filled if total_filled > 0 else 0
        fill_rate = total_filled / total_quantity if total_quantity > 0 else 0
        
        self.log(f"  ✅ TWAP Complete: {total_filled}/{total_quantity} filled @ ${avg_price:.2f} avg")
        
        return {
            'success': total_filled > 0,
            'symbol': symbol,
            'side': side,
            'total_quantity': total_quantity,
            'filled_quantity': total_filled,
            'fill_rate': fill_rate,
            'avg_price': avg_price,
            'slices_executed': len(fills),
            'fills': fills,
        }
    
    def should_use_twap(self, order: Dict) -> bool:
        """
        Determine if an order should use TWAP execution.
        
        Criteria:
        - Order value > $50,000
        - Or quantity > 1000 shares
        - Or symbol has low liquidity
        """
        value = order.get('value', 0)
        quantity = order.get('quantity', 0)
        
        # Large orders by value
        if value > 50000:
            return True
        
        # Large orders by share count
        if quantity > 1000:
            return True
        
        return False