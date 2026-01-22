"""
Transaction Cost Model

Comprehensive framework for estimating, tracking, and optimizing transaction costs.

Components:
1. Spread Cost - Half the bid-ask spread (pay to cross)
2. Slippage - Expected deviation from mid price
3. Market Impact - Price movement caused by our order
4. Commission - Broker fees (Alpaca is commission-free)
5. Borrow Cost - For short positions (annualized)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class LiquidityCategory(Enum):
    """Liquidity classification for stocks."""
    MEGA_CAP = "mega_cap"       # >$200B, spread < 0.02%
    LARGE_CAP = "large_cap"     # $10-200B, spread 0.02-0.05%
    MID_CAP = "mid_cap"         # $2-10B, spread 0.05-0.15%
    SMALL_CAP = "small_cap"     # <$2B, spread 0.15-0.50%
    ILLIQUID = "illiquid"       # Very wide spreads > 0.50%


@dataclass
class CostEstimate:
    """Estimated transaction costs for a single trade."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    notional_value: float
    
    # Cost components (in dollars)
    spread_cost: float = 0.0        # Half-spread crossing cost
    slippage_cost: float = 0.0      # Expected slippage
    market_impact_cost: float = 0.0 # Price impact of our order
    commission_cost: float = 0.0    # Broker commission (0 for Alpaca)
    borrow_cost: float = 0.0        # For shorts (annualized, prorated)
    
    # Total cost
    @property
    def total_cost(self) -> float:
        return (self.spread_cost + self.slippage_cost + 
                self.market_impact_cost + self.commission_cost + self.borrow_cost)
    
    # Cost as basis points of notional
    @property
    def total_cost_bps(self) -> float:
        if self.notional_value <= 0:
            return 0.0
        return (self.total_cost / self.notional_value) * 10000
    
    # Round-trip cost (for position that will be closed)
    @property
    def round_trip_cost(self) -> float:
        # Spread and slippage apply on entry and exit
        return (self.spread_cost + self.slippage_cost) * 2 + self.market_impact_cost
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'notional': self.notional_value,
            'spread_cost': self.spread_cost,
            'slippage_cost': self.slippage_cost,
            'impact_cost': self.market_impact_cost,
            'commission': self.commission_cost,
            'borrow_cost': self.borrow_cost,
            'total_cost': self.total_cost,
            'cost_bps': self.total_cost_bps,
        }


@dataclass
class TradeCostResult:
    """Result of cost-aware trade analysis."""
    should_trade: bool
    cost_estimate: CostEstimate
    expected_benefit: float  # Expected alpha in dollars
    net_expected_value: float  # Benefit - Cost
    reason: str = ""
    
    @property
    def benefit_cost_ratio(self) -> float:
        if self.cost_estimate.total_cost <= 0:
            return float('inf')
        return self.expected_benefit / self.cost_estimate.total_cost


@dataclass
class PortfolioCostSummary:
    """Summary of costs for a full rebalance."""
    timestamp: datetime
    total_trades: int = 0
    trades_executed: int = 0
    trades_skipped: int = 0
    
    # Cost totals
    total_spread_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_impact_cost: float = 0.0
    total_borrow_cost: float = 0.0
    total_cost: float = 0.0
    
    # Notional traded
    total_notional: float = 0.0
    
    # Cost in basis points
    @property
    def total_cost_bps(self) -> float:
        if self.total_notional <= 0:
            return 0.0
        return (self.total_cost / self.total_notional) * 10000
    
    # Savings from skipped trades
    cost_avoided: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'total_trades': self.total_trades,
            'executed': self.trades_executed,
            'skipped': self.trades_skipped,
            'spread_cost': self.total_spread_cost,
            'slippage_cost': self.total_slippage_cost,
            'impact_cost': self.total_impact_cost,
            'borrow_cost': self.total_borrow_cost,
            'total_cost': self.total_cost,
            'total_notional': self.total_notional,
            'cost_bps': self.total_cost_bps,
            'cost_avoided': self.cost_avoided,
        }


class TransactionCostModel:
    """
    Comprehensive transaction cost model.
    
    Estimates costs based on:
    - Current spread
    - Order size relative to ADV (average daily volume)
    - Time of day
    - Market volatility (VIX)
    - Historical execution data
    """
    
    # Default cost parameters (basis points)
    DEFAULT_PARAMS = {
        'base_slippage_bps': 3.0,        # Base slippage for liquid stocks
        'commission_bps': 0.0,           # Alpaca is commission-free
        'market_impact_coeff': 0.1,      # Impact coefficient
        'borrow_rate_annual': 0.02,      # 2% annual borrow rate for shorts
        'min_trade_threshold_bps': 10.0, # Min cost threshold to skip trade
        'min_benefit_ratio': 1.5,        # Expected benefit must be 1.5x cost
    }
    
    # Liquidity adjustments (multipliers)
    LIQUIDITY_MULTIPLIERS = {
        LiquidityCategory.MEGA_CAP: 0.5,
        LiquidityCategory.LARGE_CAP: 1.0,
        LiquidityCategory.MID_CAP: 2.0,
        LiquidityCategory.SMALL_CAP: 4.0,
        LiquidityCategory.ILLIQUID: 10.0,
    }
    
    # VIX adjustments
    VIX_THRESHOLDS = {
        'low': 15.0,      # < 15: 0.8x costs
        'normal': 20.0,   # 15-20: 1.0x costs
        'elevated': 25.0, # 20-25: 1.3x costs
        'high': 30.0,     # 25-30: 1.6x costs
        # > 30: 2.0x costs
    }
    
    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        vix_level: float = 18.0,
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.vix_level = vix_level
        
        # Historical data for improved estimates
        self.symbol_data: Dict[str, Dict] = {}
        
        # Track actual vs estimated for learning
        self.estimation_history: List[Dict] = []
    
    def set_vix(self, vix: float):
        """Update VIX level for cost adjustments."""
        self.vix_level = vix
    
    def get_vix_multiplier(self) -> float:
        """Get cost multiplier based on VIX."""
        if self.vix_level < self.VIX_THRESHOLDS['low']:
            return 0.8
        elif self.vix_level < self.VIX_THRESHOLDS['normal']:
            return 1.0
        elif self.vix_level < self.VIX_THRESHOLDS['elevated']:
            return 1.3
        elif self.vix_level < self.VIX_THRESHOLDS['high']:
            return 1.6
        else:
            return 2.0
    
    def classify_liquidity(
        self,
        symbol: str,
        spread_pct: float,
        market_cap: Optional[float] = None,
        adv: Optional[float] = None,
    ) -> LiquidityCategory:
        """Classify a symbol's liquidity."""
        # Primarily based on spread
        if spread_pct < 0.02:
            return LiquidityCategory.MEGA_CAP
        elif spread_pct < 0.05:
            return LiquidityCategory.LARGE_CAP
        elif spread_pct < 0.15:
            return LiquidityCategory.MID_CAP
        elif spread_pct < 0.50:
            return LiquidityCategory.SMALL_CAP
        else:
            return LiquidityCategory.ILLIQUID
    
    def estimate_spread_cost(
        self,
        spread_pct: float,
        notional: float,
    ) -> float:
        """Estimate cost of crossing the spread."""
        # Pay half the spread when crossing
        return notional * (spread_pct / 100) / 2
    
    def estimate_slippage(
        self,
        symbol: str,
        notional: float,
        liquidity: LiquidityCategory,
    ) -> float:
        """
        Estimate slippage based on liquidity, volatility, and LEARNED data.
        
        Uses historical actual slippage for this symbol if available,
        otherwise falls back to model-based estimates.
        """
        # === LEARNING: Use historical data if available ===
        if symbol in self.symbol_data and self.symbol_data[symbol]['trades'] >= 3:
            # We have enough historical data for this symbol
            learned_slippage_bps = self.symbol_data[symbol]['avg_slippage_bps']
            
            # Adjust for VIX (learned data might be from different conditions)
            vix_mult = self.get_vix_multiplier()
            
            # Weight learning vs model (more trades = more trust in learned)
            trade_count = self.symbol_data[symbol]['trades']
            learning_weight = min(0.8, trade_count / 10)  # Max 80% weight to learned
            
            # Model-based estimate
            base_slippage_bps = self.params['base_slippage_bps']
            liq_mult = self.LIQUIDITY_MULTIPLIERS[liquidity]
            model_slippage_bps = base_slippage_bps * liq_mult * vix_mult
            
            # Blend learned and model
            blended_slippage_bps = (
                learned_slippage_bps * learning_weight + 
                model_slippage_bps * (1 - learning_weight)
            )
            
            logger.debug(f"{symbol}: Using learned slippage ({trade_count} trades): "
                        f"{learned_slippage_bps:.1f}bps → {blended_slippage_bps:.1f}bps blended")
            
            return notional * blended_slippage_bps / 10000
        
        # === MODEL-BASED: Fall back to model if no history ===
        base_slippage_bps = self.params['base_slippage_bps']
        
        # Adjust for liquidity
        liq_mult = self.LIQUIDITY_MULTIPLIERS[liquidity]
        
        # Adjust for VIX
        vix_mult = self.get_vix_multiplier()
        
        slippage_bps = base_slippage_bps * liq_mult * vix_mult
        
        return notional * slippage_bps / 10000
    
    def estimate_market_impact(
        self,
        quantity: int,
        price: float,
        adv: Optional[float] = None,
        liquidity: LiquidityCategory = LiquidityCategory.LARGE_CAP,
    ) -> float:
        """
        Estimate market impact using simplified Almgren-Chriss model.
        
        Impact ~ σ * sqrt(Q / ADV) * coefficient
        
        For simplicity without ADV data, use liquidity category as proxy.
        """
        notional = quantity * price
        
        if adv and adv > 0:
            # Participation rate
            participation = notional / adv
            
            # Square root impact model
            impact_pct = self.params['market_impact_coeff'] * np.sqrt(participation)
            return notional * impact_pct
        else:
            # Use liquidity category as proxy
            # Impact scales with liquidity tier
            base_impact = {
                LiquidityCategory.MEGA_CAP: 0.0001,    # 1 bp
                LiquidityCategory.LARGE_CAP: 0.0003,   # 3 bps
                LiquidityCategory.MID_CAP: 0.0010,     # 10 bps
                LiquidityCategory.SMALL_CAP: 0.0025,   # 25 bps
                LiquidityCategory.ILLIQUID: 0.0050,    # 50 bps
            }
            return notional * base_impact.get(liquidity, 0.001)
    
    def estimate_borrow_cost(
        self,
        notional: float,
        holding_days: int = 1,
        is_short: bool = False,
    ) -> float:
        """Estimate borrow cost for short positions."""
        if not is_short:
            return 0.0
        
        annual_rate = self.params['borrow_rate_annual']
        daily_rate = annual_rate / 252
        
        return abs(notional) * daily_rate * holding_days
    
    def estimate_trade_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        spread_pct: float = 0.05,
        adv: Optional[float] = None,
        holding_days: int = 1,
    ) -> CostEstimate:
        """
        Estimate total transaction cost for a trade.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            price: Current price
            spread_pct: Current bid-ask spread as percentage
            adv: Average daily volume in dollars
            holding_days: Expected holding period (for borrow cost)
        """
        notional = quantity * price
        
        # Classify liquidity
        liquidity = self.classify_liquidity(symbol, spread_pct, adv=adv)
        
        # Calculate components
        spread_cost = self.estimate_spread_cost(spread_pct, notional)
        slippage_cost = self.estimate_slippage(symbol, notional, liquidity)
        impact_cost = self.estimate_market_impact(quantity, price, adv, liquidity)
        commission = notional * self.params['commission_bps'] / 10000
        
        # Borrow cost for shorts
        is_short = side.lower() in ['sell_short', 'short']
        borrow_cost = self.estimate_borrow_cost(notional, holding_days, is_short)
        
        return CostEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            notional_value=notional,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=impact_cost,
            commission_cost=commission,
            borrow_cost=borrow_cost,
        )
    
    def should_execute_trade(
        self,
        cost_estimate: CostEstimate,
        expected_return: float,
        confidence: float = 0.5,
    ) -> TradeCostResult:
        """
        Determine if a trade should be executed based on cost-benefit analysis.
        
        Args:
            cost_estimate: Estimated transaction costs
            expected_return: Expected return as decimal (e.g., 0.02 = 2%)
            confidence: Strategy confidence in this trade
        """
        # Expected benefit in dollars
        expected_benefit = cost_estimate.notional_value * expected_return * confidence
        
        total_cost = cost_estimate.total_cost
        net_value = expected_benefit - total_cost
        
        min_ratio = self.params['min_benefit_ratio']
        min_threshold_bps = self.params['min_trade_threshold_bps']
        
        # Skip if cost exceeds threshold and benefit doesn't justify
        if cost_estimate.total_cost_bps > min_threshold_bps:
            if expected_benefit < total_cost * min_ratio:
                return TradeCostResult(
                    should_trade=False,
                    cost_estimate=cost_estimate,
                    expected_benefit=expected_benefit,
                    net_expected_value=net_value,
                    reason=f"Cost ({cost_estimate.total_cost_bps:.1f} bps) exceeds benefit "
                           f"(ratio: {expected_benefit/total_cost:.2f}x < {min_ratio}x required)"
                )
        
        # Very small trades with high relative costs
        if cost_estimate.notional_value < 500 and cost_estimate.total_cost_bps > 50:
            return TradeCostResult(
                should_trade=False,
                cost_estimate=cost_estimate,
                expected_benefit=expected_benefit,
                net_expected_value=net_value,
                reason=f"Small trade (${cost_estimate.notional_value:.0f}) with high cost "
                       f"({cost_estimate.total_cost_bps:.0f} bps)"
            )
        
        return TradeCostResult(
            should_trade=True,
            cost_estimate=cost_estimate,
            expected_benefit=expected_benefit,
            net_expected_value=net_value,
            reason="Cost-benefit analysis passed"
        )
    
    def analyze_portfolio_costs(
        self,
        trades: List[Dict],
        expected_returns: Dict[str, float],
        confidences: Dict[str, float],
    ) -> Tuple[List[Dict], PortfolioCostSummary]:
        """
        Analyze and filter trades based on transaction costs.
        
        Args:
            trades: List of proposed trades [{'symbol', 'side', 'quantity', 'price', 'spread_pct'}]
            expected_returns: Expected returns by symbol
            confidences: Strategy confidence by symbol
        
        Returns:
            Tuple of (approved_trades, cost_summary)
        """
        approved_trades = []
        summary = PortfolioCostSummary(timestamp=datetime.now())
        
        for trade in trades:
            symbol = trade['symbol']
            
            # Estimate cost
            cost_estimate = self.estimate_trade_cost(
                symbol=symbol,
                side=trade['side'],
                quantity=trade['quantity'],
                price=trade['price'],
                spread_pct=trade.get('spread_pct', 0.05),
                adv=trade.get('adv'),
                holding_days=trade.get('holding_days', 20),  # ~1 month default
            )
            
            # Check if trade is worthwhile
            exp_return = expected_returns.get(symbol, 0.02)  # Default 2% expected
            confidence = confidences.get(symbol, 0.5)
            
            result = self.should_execute_trade(cost_estimate, exp_return, confidence)
            
            summary.total_trades += 1
            
            if result.should_trade:
                approved_trades.append({
                    **trade,
                    'cost_estimate': cost_estimate,
                    'expected_benefit': result.expected_benefit,
                    'net_value': result.net_expected_value,
                })
                
                summary.trades_executed += 1
                summary.total_spread_cost += cost_estimate.spread_cost
                summary.total_slippage_cost += cost_estimate.slippage_cost
                summary.total_impact_cost += cost_estimate.market_impact_cost
                summary.total_borrow_cost += cost_estimate.borrow_cost
                summary.total_cost += cost_estimate.total_cost
                summary.total_notional += cost_estimate.notional_value
            else:
                summary.trades_skipped += 1
                summary.cost_avoided += cost_estimate.total_cost
                
                logger.info(f"Skipping trade {symbol}: {result.reason}")
        
        return approved_trades, summary
    
    def record_actual_cost(
        self,
        symbol: str,
        estimated_cost: CostEstimate,
        actual_fill_price: float,
        mid_price_at_decision: float,
    ):
        """
        Record actual vs estimated costs for learning.
        
        Args:
            symbol: Stock symbol
            estimated_cost: The pre-trade cost estimate
            actual_fill_price: The price we actually got
            mid_price_at_decision: The mid price when we decided to trade
        """
        actual_slippage = abs(actual_fill_price - mid_price_at_decision)
        actual_slippage_bps = (actual_slippage / mid_price_at_decision) * 10000
        
        estimated_slippage_bps = estimated_cost.slippage_cost / estimated_cost.notional_value * 10000
        
        # Record for learning
        self.estimation_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'estimated_slippage_bps': estimated_slippage_bps,
            'actual_slippage_bps': actual_slippage_bps,
            'estimation_error_bps': actual_slippage_bps - estimated_slippage_bps,
        })
        
        # Update symbol-specific data
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = {
                'trades': 0,
                'avg_slippage_bps': actual_slippage_bps,
            }
        else:
            data = self.symbol_data[symbol]
            data['trades'] += 1
            # Exponential moving average
            alpha = 0.2
            data['avg_slippage_bps'] = alpha * actual_slippage_bps + (1 - alpha) * data['avg_slippage_bps']
    
    def get_cost_summary_for_ui(self) -> Dict[str, Any]:
        """Get cost summary for display in UI."""
        if not self.estimation_history:
            return {
                'total_trades_analyzed': 0,
                'avg_slippage_bps': 0,
                'avg_estimation_error_bps': 0,
                'symbols_with_learned_data': 0,
            }
        
        recent = self.estimation_history[-100:]  # Last 100 trades
        
        # Count symbols with enough data to use learning
        symbols_with_learning = sum(1 for s, d in self.symbol_data.items() if d.get('trades', 0) >= 3)
        
        return {
            'total_trades_analyzed': len(self.estimation_history),
            'avg_slippage_bps': np.mean([h['actual_slippage_bps'] for h in recent]),
            'avg_estimation_error_bps': np.mean([h['estimation_error_bps'] for h in recent]),
            'vix_level': self.vix_level,
            'vix_multiplier': self.get_vix_multiplier(),
            'symbols_with_learned_data': symbols_with_learning,
            'learning_status': f"{symbols_with_learning} symbols using learned estimates",
        }
    
    def save_learned_data(self, filepath: str = "outputs/cost_model_learned.json"):
        """Save learned cost data to file for persistence."""
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'symbol_data': self.symbol_data,
            'last_updated': datetime.now().isoformat(),
            'total_trades': len(self.estimation_history),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved cost model learned data: {len(self.symbol_data)} symbols")
    
    def load_learned_data(self, filepath: str = "outputs/cost_model_learned.json"):
        """Load learned cost data from file."""
        import json
        from pathlib import Path
        
        if not Path(filepath).exists():
            logger.info("No learned cost data found, starting fresh")
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.symbol_data = data.get('symbol_data', {})
            logger.info(f"Loaded cost model learned data: {len(self.symbol_data)} symbols")
        except Exception as e:
            logger.warning(f"Could not load learned cost data: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning process."""
        if not self.symbol_data:
            return {
                'symbols_tracked': 0,
                'symbols_with_learning': 0,
                'avg_trades_per_symbol': 0,
                'estimation_improvement': 'N/A',
            }
        
        symbols_with_learning = sum(1 for s, d in self.symbol_data.items() if d.get('trades', 0) >= 3)
        total_trades = sum(d.get('trades', 0) for d in self.symbol_data.values())
        
        # Calculate estimation improvement from history
        if len(self.estimation_history) >= 10:
            early = self.estimation_history[:10]
            recent = self.estimation_history[-10:]
            early_error = np.mean([abs(h['estimation_error_bps']) for h in early])
            recent_error = np.mean([abs(h['estimation_error_bps']) for h in recent])
            improvement = ((early_error - recent_error) / early_error * 100) if early_error > 0 else 0
            improvement_str = f"{improvement:.1f}% improvement" if improvement > 0 else f"{-improvement:.1f}% worse"
        else:
            improvement_str = "Collecting data..."
        
        return {
            'symbols_tracked': len(self.symbol_data),
            'symbols_with_learning': symbols_with_learning,
            'total_trades_recorded': total_trades,
            'avg_trades_per_symbol': total_trades / len(self.symbol_data) if self.symbol_data else 0,
            'estimation_improvement': improvement_str,
        }


# Global instance for use across modules
transaction_cost_model = TransactionCostModel()

# Try to load any existing learned data on startup
try:
    transaction_cost_model.load_learned_data()
except Exception as e:
    logger.debug(f"Could not load learned data on startup: {e}")
