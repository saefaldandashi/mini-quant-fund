"""
Leverage Manager - Controls and monitors leverage usage across the system.

Provides:
- Dynamic leverage limits based on VIX, drawdown, regime
- Margin tracking and validation
- Pre-trade leverage checks
- Real-time leverage monitoring
- Emergency kill switches
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
import json
from pathlib import Path


class LeverageState(Enum):
    """Current leverage health state."""
    HEALTHY = "healthy"           # All within limits
    CAUTION = "caution"           # Approaching limits
    WARNING = "warning"           # At limits
    CRITICAL = "critical"         # Exceeding limits
    EMERGENCY = "emergency"       # Kill switch triggered


@dataclass
class MarginSnapshot:
    """Point-in-time margin data from broker."""
    timestamp: datetime
    equity: float                  # Total account equity
    buying_power: float            # Available buying power
    regt_buying_power: float       # RegT buying power
    daytrading_buying_power: float # Day trading buying power (if PDT)
    initial_margin: float          # Margin required for current positions
    maintenance_margin: float      # Minimum margin to avoid liquidation
    cash: float                    # Cash balance
    
    @property
    def margin_used_pct(self) -> float:
        """Percentage of buying power used."""
        if self.buying_power <= 0:
            return 100.0
        return (1 - (self.buying_power / (self.equity * 2))) * 100
    
    @property
    def margin_buffer(self) -> float:
        """Buffer before margin call (as %)."""
        if self.equity <= 0:
            return 0.0
        return ((self.equity - self.maintenance_margin) / self.equity) * 100
    
    @property
    def current_leverage(self) -> float:
        """Current leverage ratio."""
        if self.equity <= 0:
            return 0.0
        # Leverage = (Equity + Borrowed) / Equity = Positions / Equity
        positions_value = self.equity * 2 - self.buying_power
        return positions_value / self.equity if self.equity > 0 else 0.0


@dataclass
class LeverageLimits:
    """Current leverage limits based on market conditions."""
    max_leverage: float = 2.0          # Absolute maximum
    vix_adjusted_max: float = 2.0      # VIX-based limit
    drawdown_adjusted_max: float = 2.0 # Drawdown-based limit
    effective_max: float = 2.0         # min of all limits
    
    overnight_max: float = 1.5         # Max for overnight positions
    single_position_max_pct: float = 10.0  # Max % gross per position
    sector_max_pct: float = 25.0       # Max % gross per sector
    
    # Margin buffer requirements
    min_margin_buffer_pct: float = 20.0  # Never use last 20%
    
    def calculate_effective(self):
        """Calculate effective max leverage from all constraints."""
        self.effective_max = min(
            self.max_leverage,
            self.vix_adjusted_max,
            self.drawdown_adjusted_max
        )


@dataclass
class LeverageConfig:
    """Configuration for leverage management."""
    # Absolute limits
    max_leverage_absolute: float = 2.0
    max_overnight_leverage: float = 1.5
    
    # VIX-based limits
    vix_thresholds: Dict[float, float] = field(default_factory=lambda: {
        15: 2.0,   # VIX < 15: full leverage
        20: 1.75,  # VIX 15-20: slight reduction
        25: 1.5,   # VIX 20-25: moderate reduction
        35: 1.0,   # VIX 25-35: no leverage
        100: 0.5,  # VIX > 35: reduce exposure
    })
    
    # Drawdown-based limits
    drawdown_thresholds: Dict[float, float] = field(default_factory=lambda: {
        5: 1.0,    # Drawdown < 5%: full leverage multiplier
        10: 0.75,  # Drawdown 5-10%: reduce by 25%
        15: 0.50,  # Drawdown 10-15%: reduce by 50%
        20: 0.25,  # Drawdown 15-20%: reduce by 75%
        100: 0.0,  # Drawdown > 20%: no leverage
    })
    
    # Margin buffer
    min_margin_buffer_pct: float = 20.0
    margin_alert_threshold: float = 75.0
    margin_auto_reduce_threshold: float = 85.0
    margin_emergency_threshold: float = 95.0
    
    # Position limits
    max_single_position_pct: float = 10.0
    max_sector_pct: float = 25.0
    
    # Daily loss limits
    daily_loss_halt_pct: float = 3.0    # Halt new trades
    daily_loss_close_pct: float = 5.0   # Close leveraged positions
    
    # Strategy-specific leverage limits
    strategy_leverage_limits: Dict[str, float] = field(default_factory=lambda: {
        'TimeSeriesMomentum': 1.25,
        'CrossSectionMomentum': 1.25,
        'MeanReversion': 1.5,
        'VolatilityRegimeVolTarget': 1.5,
        'Carry': 2.0,
        'ValueQualityTilt': 2.0,
        'RiskParityMinVar': 2.0,
        'TailRiskOverlay': 1.0,
        'NewsSentimentEvent': 1.25,
        'TS_Momentum_LS': 1.5,
        'MeanReversion_LS': 1.5,
        'QualityValue_LS': 2.0,
        'IntradayMomentum': 1.0,
        'VWAPDeviation': 1.0,
        'VolumeSpike': 1.0,
        'OpeningRangeBreakout': 1.0,
        'RelativeStrengthIntraday': 1.25,
        'QuickMeanReversion': 1.25,
    })
    
    # Margin interest rate (annual)
    margin_interest_rate: float = 0.07  # 7% annual
    
    # Ramp-up period (days to reach full leverage from start)
    leverage_ramp_days: int = 30


class LeverageManager:
    """
    Central manager for all leverage-related decisions.
    
    Integrates with:
    - Broker for margin data
    - Risk monitor for VIX and drawdown
    - Strategies for leverage limits
    - Execution for pre-trade checks
    """
    
    def __init__(
        self,
        config: Optional[LeverageConfig] = None,
        storage_path: str = "outputs/leverage_state.json"
    ):
        self.config = config or LeverageConfig()
        self.storage_path = Path(storage_path)
        
        # Current state
        self.current_limits = LeverageLimits()
        self.last_margin_snapshot: Optional[MarginSnapshot] = None
        self.state = LeverageState.HEALTHY
        
        # Tracking
        self.peak_equity: float = 0.0
        self.current_drawdown: float = 0.0
        self.current_vix: float = 20.0
        self.daily_pnl_pct: float = 0.0
        self.start_of_day_equity: float = 0.0
        
        # Kill switch state
        self.kill_switch_active: bool = False
        self.kill_switch_reason: Optional[str] = None
        self.kill_switch_until: Optional[datetime] = None
        
        # Leverage history for learning
        self.leverage_history: List[Dict] = []
        
        # Ramp-up tracking
        self.first_trade_date: Optional[datetime] = None
        
        self._load()
    
    def _load(self):
        """Load saved state."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.peak_equity = data.get('peak_equity', 0)
                self.first_trade_date = (
                    datetime.fromisoformat(data['first_trade_date'])
                    if data.get('first_trade_date') else None
                )
                self.leverage_history = data.get('leverage_history', [])[-1000:]
                logging.info(f"Loaded leverage state: peak_equity=${self.peak_equity:,.0f}")
            except Exception as e:
                logging.warning(f"Could not load leverage state: {e}")
    
    def _save(self):
        """Persist state."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'peak_equity': self.peak_equity,
                    'first_trade_date': (
                        self.first_trade_date.isoformat()
                        if self.first_trade_date else None
                    ),
                    'leverage_history': self.leverage_history[-1000:],
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save leverage state: {e}")
    
    # =========================================================================
    # MARGIN DATA UPDATES
    # =========================================================================
    
    def update_margin_data(
        self,
        equity: float,
        buying_power: float,
        regt_buying_power: float = 0,
        daytrading_buying_power: float = 0,
        initial_margin: float = 0,
        maintenance_margin: float = 0,
        cash: float = 0,
    ):
        """Update with fresh margin data from broker."""
        self.last_margin_snapshot = MarginSnapshot(
            timestamp=datetime.now(),
            equity=equity,
            buying_power=buying_power,
            regt_buying_power=regt_buying_power or buying_power,
            daytrading_buying_power=daytrading_buying_power or buying_power * 2,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            cash=cash,
        )
        
        # Update peak equity for drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        
        # Update daily P/L
        if self.start_of_day_equity > 0:
            self.daily_pnl_pct = (equity - self.start_of_day_equity) / self.start_of_day_equity * 100
        
        # Recalculate limits
        self._update_limits()
        
        # Check state
        self._check_state()
        
        # Record for history
        self.leverage_history.append({
            'timestamp': datetime.now().isoformat(),
            'equity': equity,
            'leverage': self.last_margin_snapshot.current_leverage,
            'vix': self.current_vix,
            'drawdown': self.current_drawdown,
            'state': self.state.value,
        })
        
        self._save()
    
    def update_vix(self, vix: float):
        """Update current VIX level."""
        self.current_vix = vix
        self._update_limits()
        self._check_state()
    
    def set_start_of_day_equity(self, equity: float):
        """Set start of day equity for daily P/L tracking."""
        self.start_of_day_equity = equity
    
    # =========================================================================
    # LIMIT CALCULATIONS
    # =========================================================================
    
    def _update_limits(self):
        """Recalculate all leverage limits."""
        # VIX-based limit
        vix_limit = self.config.max_leverage_absolute
        for threshold, limit in sorted(self.config.vix_thresholds.items()):
            if self.current_vix < threshold:
                vix_limit = limit
                break
        self.current_limits.vix_adjusted_max = vix_limit
        
        # Drawdown-based multiplier
        dd_mult = 1.0
        for threshold, mult in sorted(self.config.drawdown_thresholds.items()):
            if self.current_drawdown < threshold:
                dd_mult = mult
                break
        self.current_limits.drawdown_adjusted_max = (
            self.config.max_leverage_absolute * dd_mult
        )
        
        # Ramp-up adjustment (if enabled)
        ramp_mult = 1.0
        if self.first_trade_date and self.config.leverage_ramp_days > 0:
            days_active = (datetime.now() - self.first_trade_date).days
            if days_active < self.config.leverage_ramp_days:
                ramp_mult = days_active / self.config.leverage_ramp_days
        
        # Calculate effective max
        self.current_limits.max_leverage = self.config.max_leverage_absolute
        self.current_limits.calculate_effective()
        
        # Apply ramp-up
        self.current_limits.effective_max *= ramp_mult
        
        # Apply kill switch
        if self.kill_switch_active:
            self.current_limits.effective_max = min(
                self.current_limits.effective_max, 0.5
            )
    
    def _check_state(self):
        """Check and update leverage state."""
        if self.kill_switch_active:
            self.state = LeverageState.EMERGENCY
            return
        
        if not self.last_margin_snapshot:
            self.state = LeverageState.HEALTHY
            return
        
        margin = self.last_margin_snapshot
        
        # Check margin usage
        if margin.margin_used_pct >= self.config.margin_emergency_threshold:
            self.state = LeverageState.CRITICAL
        elif margin.margin_used_pct >= self.config.margin_auto_reduce_threshold:
            self.state = LeverageState.WARNING
        elif margin.margin_used_pct >= self.config.margin_alert_threshold:
            self.state = LeverageState.CAUTION
        else:
            self.state = LeverageState.HEALTHY
        
        # Check daily loss limits
        if self.daily_pnl_pct <= -self.config.daily_loss_close_pct:
            self._activate_kill_switch("Daily loss exceeded close threshold")
        elif self.daily_pnl_pct <= -self.config.daily_loss_halt_pct:
            if self.state.value < LeverageState.WARNING.value:
                self.state = LeverageState.WARNING
    
    # =========================================================================
    # KILL SWITCH
    # =========================================================================
    
    def _activate_kill_switch(self, reason: str, duration_hours: int = 24):
        """Activate emergency kill switch."""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_until = datetime.now() + timedelta(hours=duration_hours)
        self.state = LeverageState.EMERGENCY
        logging.critical(f"🚨 LEVERAGE KILL SWITCH ACTIVATED: {reason}")
        self._save()
    
    def check_kill_switch_expired(self):
        """Check if kill switch has expired."""
        if self.kill_switch_active and self.kill_switch_until:
            if datetime.now() >= self.kill_switch_until:
                self.deactivate_kill_switch()
    
    def deactivate_kill_switch(self):
        """Manually deactivate kill switch."""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.kill_switch_until = None
        self._update_limits()
        self._check_state()
        logging.info("✅ Leverage kill switch deactivated")
        self._save()
    
    # =========================================================================
    # PRE-TRADE CHECKS
    # =========================================================================
    
    def get_effective_leverage_limit(self) -> float:
        """Get current effective leverage limit."""
        self.check_kill_switch_expired()
        return self.current_limits.effective_max
    
    def get_strategy_leverage_limit(self, strategy_name: str) -> float:
        """Get leverage limit for a specific strategy."""
        strategy_limit = self.config.strategy_leverage_limits.get(
            strategy_name, 1.5  # Default to 1.5x
        )
        return min(strategy_limit, self.get_effective_leverage_limit())
    
    def calculate_blended_leverage_limit(
        self,
        strategy_weights: Dict[str, float],
        cross_asset_signals: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate blended leverage limit based on strategy mix AND cross-asset signals.
        
        Cross-asset stress reduces leverage:
        - Credit stress (HYG falling) → Reduce leverage
        - Oil volatility spike → Reduce leverage for energy
        - Gold surge (risk-off) → Reduce overall leverage
        """
        if not strategy_weights:
            return self.get_effective_leverage_limit()
        
        total_weight = sum(abs(w) for w in strategy_weights.values())
        if total_weight == 0:
            return self.get_effective_leverage_limit()
        
        blended = sum(
            abs(weight) * self.get_strategy_leverage_limit(name)
            for name, weight in strategy_weights.items()
        ) / total_weight
        
        base_limit = min(blended, self.get_effective_leverage_limit())
        
        # Apply cross-asset adjustments
        if cross_asset_signals:
            cross_asset_multiplier = self._calculate_cross_asset_leverage_adjustment(cross_asset_signals)
            base_limit *= cross_asset_multiplier
            if cross_asset_multiplier < 1.0:
                logging.info(f"Cross-asset reduced leverage: {cross_asset_multiplier:.2f}x")
        
        return max(1.0, base_limit)  # Never go below 1x
    
    def _calculate_cross_asset_leverage_adjustment(
        self,
        cross_asset_signals: Dict[str, float],
    ) -> float:
        """
        Calculate leverage reduction based on cross-asset stress.
        
        Returns multiplier (e.g., 0.8 = reduce to 80% of normal leverage)
        """
        multiplier = 1.0
        
        # Credit stress (negative credit_signal = stress)
        credit_signal = cross_asset_signals.get('credit_signal', 0)
        if credit_signal < -0.2:
            # Credit stress detected - reduce leverage
            reduction = min(0.3, abs(credit_signal) * 0.5)  # Up to 30% reduction
            multiplier *= (1 - reduction)
            logging.debug(f"Credit stress leverage reduction: {reduction:.1%}")
        
        # Gold surge (risk-off flight)
        gold_return = cross_asset_signals.get('gold_return', 0)
        if gold_return > 0.03:  # Gold up 3%+ = risk-off
            reduction = min(0.2, gold_return * 3)  # Up to 20% reduction
            multiplier *= (1 - reduction)
            logging.debug(f"Gold surge leverage reduction: {reduction:.1%}")
        
        # Oil volatility (rapid oil moves = market stress)
        oil_return = cross_asset_signals.get('oil_return', 0)
        if abs(oil_return) > 0.05:  # Oil moved 5%+ = elevated risk
            reduction = min(0.15, abs(oil_return) * 2)
            multiplier *= (1 - reduction)
            logging.debug(f"Oil volatility leverage reduction: {reduction:.1%}")
        
        # International market stress
        europe_signal = cross_asset_signals.get('europe_lead', 0)
        china_signal = cross_asset_signals.get('china_signal', 0)
        if europe_signal < -0.03 or china_signal < -0.03:  # International down 3%+
            reduction = min(0.15, max(abs(europe_signal), abs(china_signal)) * 2)
            multiplier *= (1 - reduction)
            logging.debug(f"International stress leverage reduction: {reduction:.1%}")
        
        return max(0.5, multiplier)  # Never reduce more than 50%
    
    def can_use_leverage(self) -> Tuple[bool, str]:
        """Check if leverage can be used right now."""
        self.check_kill_switch_expired()
        
        if self.kill_switch_active:
            return False, f"Kill switch active: {self.kill_switch_reason}"
        
        if self.state == LeverageState.EMERGENCY:
            return False, "Emergency state - leverage disabled"
        
        if self.state == LeverageState.CRITICAL:
            return False, "Critical margin state - reduce exposure first"
        
        if self.get_effective_leverage_limit() <= 1.0:
            return False, f"Leverage disabled (VIX={self.current_vix:.1f}, DD={self.current_drawdown:.1f}%)"
        
        return True, "Leverage available"
    
    def validate_trade_leverage(
        self,
        trade_value: float,
        current_positions_value: float,
        equity: float,
    ) -> Tuple[bool, str, float]:
        """
        Validate if a trade is within leverage limits.
        
        Returns:
            Tuple of (is_valid, reason, max_allowed_value)
        """
        can_leverage, reason = self.can_use_leverage()
        
        # Calculate post-trade leverage
        new_positions_value = current_positions_value + trade_value
        new_leverage = new_positions_value / equity if equity > 0 else 0
        
        max_leverage = self.get_effective_leverage_limit()
        max_value = equity * max_leverage - current_positions_value
        
        if new_leverage > max_leverage:
            return (
                False,
                f"Would exceed leverage limit ({new_leverage:.2f}x > {max_leverage:.2f}x)",
                max(0, max_value)
            )
        
        # Check margin buffer
        if self.last_margin_snapshot:
            margin = self.last_margin_snapshot
            buffer_pct = 100 - self.config.min_margin_buffer_pct
            available_for_trade = margin.buying_power * (buffer_pct / 100)
            
            if trade_value > available_for_trade:
                return (
                    False,
                    f"Would exceed margin buffer (need ${trade_value:,.0f}, have ${available_for_trade:,.0f})",
                    available_for_trade
                )
        
        return True, "Trade within leverage limits", trade_value
    
    # =========================================================================
    # MARGIN COST CALCULATIONS
    # =========================================================================
    
    def calculate_daily_margin_cost(self, borrowed_amount: float) -> float:
        """Calculate daily margin interest cost."""
        return borrowed_amount * self.config.margin_interest_rate / 365
    
    def calculate_holding_cost(
        self,
        position_value: float,
        leverage: float,
        holding_days: int
    ) -> float:
        """Calculate total margin cost for holding a leveraged position."""
        if leverage <= 1.0:
            return 0.0
        
        borrowed = position_value * (1 - 1/leverage)
        daily_cost = self.calculate_daily_margin_cost(borrowed)
        return daily_cost * holding_days
    
    def get_break_even_return(self, leverage: float, holding_days: int = 30) -> float:
        """Calculate minimum return needed to break even on margin costs."""
        if leverage <= 1.0:
            return 0.0
        
        # Annual margin cost as % of borrowed
        annual_cost_pct = self.config.margin_interest_rate * (leverage - 1)
        
        # Adjust for holding period
        holding_cost_pct = annual_cost_pct * (holding_days / 365)
        
        return holding_cost_pct
    
    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive leverage status."""
        self.check_kill_switch_expired()
        
        margin = self.last_margin_snapshot
        
        return {
            'state': self.state.value,
            'effective_max_leverage': self.get_effective_leverage_limit(),
            'current_leverage': margin.current_leverage if margin else 0,
            'vix': self.current_vix,
            'drawdown_pct': self.current_drawdown,
            'daily_pnl_pct': self.daily_pnl_pct,
            'margin_used_pct': margin.margin_used_pct if margin else 0,
            'margin_buffer_pct': margin.margin_buffer if margin else 100,
            'buying_power': margin.buying_power if margin else 0,
            'equity': margin.equity if margin else 0,
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_reason': self.kill_switch_reason,
            'limits': {
                'absolute_max': self.current_limits.max_leverage,
                'vix_adjusted': self.current_limits.vix_adjusted_max,
                'drawdown_adjusted': self.current_limits.drawdown_adjusted_max,
                'effective': self.current_limits.effective_max,
            },
            'thresholds': {
                'margin_alert': self.config.margin_alert_threshold,
                'margin_auto_reduce': self.config.margin_auto_reduce_threshold,
                'margin_emergency': self.config.margin_emergency_threshold,
                'daily_loss_halt': self.config.daily_loss_halt_pct,
                'daily_loss_close': self.config.daily_loss_close_pct,
            }
        }
    
    def get_leverage_recommendation(
        self,
        strategy_weights: Dict[str, float],
        expected_return: float,
        holding_days: int = 20
    ) -> Dict[str, Any]:
        """Get leverage recommendation for a proposed trade."""
        can_leverage, reason = self.can_use_leverage()
        
        if not can_leverage:
            return {
                'recommended_leverage': 1.0,
                'reason': reason,
                'expected_cost': 0,
                'break_even_return': 0,
            }
        
        # Get blended limit
        max_leverage = self.calculate_blended_leverage_limit(strategy_weights)
        
        # Calculate costs at different leverage levels
        recommendations = []
        for leverage in [1.0, 1.25, 1.5, 1.75, 2.0]:
            if leverage > max_leverage:
                continue
            
            break_even = self.get_break_even_return(leverage, holding_days)
            leveraged_return = expected_return * leverage
            net_return = leveraged_return - break_even
            
            recommendations.append({
                'leverage': leverage,
                'break_even': break_even,
                'expected_leveraged_return': leveraged_return,
                'net_return': net_return,
                'is_profitable': net_return > expected_return,
            })
        
        # Find optimal leverage
        profitable = [r for r in recommendations if r['is_profitable']]
        
        if profitable:
            optimal = max(profitable, key=lambda x: x['net_return'])
        else:
            optimal = recommendations[0] if recommendations else {
                'leverage': 1.0,
                'break_even': 0,
                'expected_leveraged_return': expected_return,
                'net_return': expected_return,
            }
        
        return {
            'recommended_leverage': optimal['leverage'],
            'reason': "Optimal risk-adjusted return" if optimal['leverage'] > 1 else "Leverage not beneficial",
            'max_allowed': max_leverage,
            'expected_cost': self.get_break_even_return(optimal['leverage'], holding_days),
            'break_even_return': optimal['break_even'],
            'all_options': recommendations,
        }
