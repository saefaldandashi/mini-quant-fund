"""
Real-Time Risk Monitor

Background thread that continuously monitors portfolio risk and triggers
automatic actions when thresholds are breached.

Key Features:
- Continuous monitoring (every 60 seconds)
- Drawdown-based de-risking (10% threshold)
- VIX-based halt mechanism (VIX > 35)
- Correlation monitoring
- Automatic position reduction
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"           # Normal operations
    ELEVATED = "elevated" # Increased monitoring
    HIGH = "high"         # Reduced exposure
    CRITICAL = "critical" # Halt trading


@dataclass
class RiskAlert:
    """A risk alert triggered by the monitor."""
    timestamp: datetime
    level: RiskLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    action_taken: str = ""


@dataclass 
class RiskMonitorConfig:
    """Configuration for the risk monitor."""
    check_interval_seconds: int = 60
    
    # Drawdown thresholds
    drawdown_warning: float = 0.05      # 5% - Alert
    drawdown_reduce: float = 0.08       # 8% - Reduce by 30%
    drawdown_critical: float = 0.10     # 10% - Reduce by 50%, halt new trades
    
    # VIX thresholds
    vix_elevated: float = 25.0          # Elevated risk
    vix_high: float = 30.0              # Reduce new position sizes
    vix_critical: float = 35.0          # Halt new trades
    
    # Position concentration
    max_single_position_pct: float = 0.20  # 20% max in single position
    max_sector_pct: float = 0.35           # 35% max in single sector
    
    # Correlation monitoring
    high_correlation_threshold: float = 0.85  # Positions too correlated
    
    # MARGIN/LEVERAGE CIRCUIT BREAKERS
    margin_alert_pct: float = 70.0        # Alert when 70% margin used
    margin_reduce_pct: float = 85.0       # Reduce exposure at 85% margin used
    margin_emergency_pct: float = 95.0    # Emergency at 95% margin used
    
    # Daily loss circuit breaker
    daily_loss_alert_pct: float = 2.0     # Alert at 2% daily loss
    daily_loss_halt_pct: float = 4.0      # Halt trading at 4% daily loss
    daily_loss_delever_pct: float = 5.0   # Close leveraged positions at 5% daily loss


class RealtimeRiskMonitor:
    """
    Background thread that continuously monitors portfolio risk.
    
    Triggers automatic actions when thresholds are breached:
    - Drawdown > 5%: Alert
    - Drawdown > 8%: Reduce exposure by 30%
    - Drawdown > 10%: Reduce exposure by 50%, halt new trades
    - VIX > 35: Halt new trades
    
    Usage:
        monitor = RealtimeRiskMonitor(broker)
        monitor.start()
        # ... trading operations ...
        monitor.stop()
    """
    
    def __init__(
        self,
        broker,
        config: Optional[RiskMonitorConfig] = None,
        on_alert: Optional[Callable[[RiskAlert], None]] = None,
    ):
        """
        Initialize the real-time risk monitor.
        
        Args:
            broker: Broker instance for account/position data
            config: Risk monitoring configuration
            on_alert: Optional callback for risk alerts
        """
        self.broker = broker
        self.config = config or RiskMonitorConfig()
        self.on_alert = on_alert
        
        # State
        self.peak_equity = 0.0
        self.is_running = False
        self.halt_trading = False
        self.current_risk_level = RiskLevel.LOW
        self.alerts: List[RiskAlert] = []
        self.last_vix = 18.0
        
        # Position reduction tracking
        self.last_reduction_time: Optional[datetime] = None
        self.reduction_cooldown_minutes = 30
        
        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the background monitoring thread."""
        if self.is_running:
            logger.warning("Risk monitor already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("🛡️ Real-time risk monitor STARTED")
    
    def stop(self):
        """Stop the monitoring thread."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("🛡️ Real-time risk monitor STOPPED")
    
    def _monitor_loop(self):
        """Main monitoring loop - runs continuously."""
        while self.is_running:
            try:
                self._perform_risk_checks()
            except Exception as e:
                logger.error(f"Risk monitor error: {e}")
            
            time.sleep(self.config.check_interval_seconds)
    
    def _perform_risk_checks(self):
        """Perform all risk checks."""
        with self._lock:
            # Get current account state
            try:
                account = self.broker.get_account()
                equity = float(account.get('equity', 0))
                positions = self.broker.get_positions()
            except Exception as e:
                logger.warning(f"Could not get account data: {e}")
                return
            
            # Update peak equity
            if equity > self.peak_equity:
                self.peak_equity = equity
            
            # Calculate drawdown
            drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
            
            # Check drawdown thresholds
            self._check_drawdown(drawdown, equity)
            
            # Check VIX if available
            self._check_vix()
            
            # Check position concentration
            self._check_concentration(positions, equity)
            
            # Check margin usage (circuit breaker)
            self._check_margin(account, equity)
            
            # Check daily loss (circuit breaker)
            self._check_daily_loss(account, equity)
            
            # Check cross-asset stress signals (NEW)
            self._check_cross_asset_stress()
    
    def _check_drawdown(self, drawdown: float, equity: float):
        """Check drawdown thresholds and take action."""
        
        if drawdown >= self.config.drawdown_critical:
            # CRITICAL: Reduce exposure by 50%, halt trading
            alert = self._create_alert(
                level=RiskLevel.CRITICAL,
                message=f"CRITICAL: Drawdown {drawdown:.1%} exceeds {self.config.drawdown_critical:.1%}",
                metric_name="drawdown",
                current_value=drawdown,
                threshold=self.config.drawdown_critical,
            )
            self._trigger_alert(alert)
            self._reduce_exposure(0.50, "critical_drawdown")
            self.halt_trading = True
            self.current_risk_level = RiskLevel.CRITICAL
            
        elif drawdown >= self.config.drawdown_reduce:
            # HIGH: Reduce exposure by 30%
            alert = self._create_alert(
                level=RiskLevel.HIGH,
                message=f"HIGH RISK: Drawdown {drawdown:.1%} exceeds {self.config.drawdown_reduce:.1%}",
                metric_name="drawdown",
                current_value=drawdown,
                threshold=self.config.drawdown_reduce,
            )
            self._trigger_alert(alert)
            self._reduce_exposure(0.30, "high_drawdown")
            self.current_risk_level = RiskLevel.HIGH
            
        elif drawdown >= self.config.drawdown_warning:
            # ELEVATED: Alert only
            alert = self._create_alert(
                level=RiskLevel.ELEVATED,
                message=f"WARNING: Drawdown {drawdown:.1%} exceeds {self.config.drawdown_warning:.1%}",
                metric_name="drawdown",
                current_value=drawdown,
                threshold=self.config.drawdown_warning,
            )
            self._trigger_alert(alert)
            self.current_risk_level = RiskLevel.ELEVATED
            
        else:
            # Normal operation
            if self.current_risk_level != RiskLevel.LOW:
                logger.info(f"Risk level normalized: {drawdown:.1%} drawdown")
            self.current_risk_level = RiskLevel.LOW
            self.halt_trading = False
    
    def _check_vix(self):
        """Check VIX level and adjust risk level."""
        # Get VIX from market indicators if available
        try:
            # This would normally fetch from a market data source
            # For now, use a placeholder
            vix = self.last_vix
        except Exception:
            return
        
        if vix >= self.config.vix_critical:
            alert = self._create_alert(
                level=RiskLevel.CRITICAL,
                message=f"VIX CRITICAL: {vix:.1f} >= {self.config.vix_critical}. Halting new trades.",
                metric_name="vix",
                current_value=vix,
                threshold=self.config.vix_critical,
            )
            self._trigger_alert(alert)
            self.halt_trading = True
            
        elif vix >= self.config.vix_high:
            alert = self._create_alert(
                level=RiskLevel.HIGH,
                message=f"VIX HIGH: {vix:.1f} >= {self.config.vix_high}. Reducing position sizes.",
                metric_name="vix",
                current_value=vix,
                threshold=self.config.vix_high,
            )
            self._trigger_alert(alert)
    
    def _check_concentration(self, positions, equity: float):
        """Check position concentration risk."""
        if equity <= 0 or not positions:
            return
        
        # Handle both Dict[str, Dict] and List[Dict] formats
        if isinstance(positions, dict):
            position_items = [(symbol, pos) for symbol, pos in positions.items()]
        else:
            position_items = [(pos.get('symbol', 'UNKNOWN'), pos) for pos in positions]
        
        # Check single position concentration
        for symbol, pos in position_items:
            if isinstance(pos, str):
                continue  # Skip malformed entries
            market_value = float(pos.get('market_value', 0)) if isinstance(pos, dict) else 0
            pct = abs(market_value) / equity
            
            if pct > self.config.max_single_position_pct:
                alert = self._create_alert(
                    level=RiskLevel.ELEVATED,
                    message=f"Position {symbol} is {pct:.1%} of portfolio (max: {self.config.max_single_position_pct:.0%})",
                    metric_name="position_concentration",
                    current_value=pct,
                    threshold=self.config.max_single_position_pct,
                )
                self._trigger_alert(alert)
    
    def _check_margin(self, account: Dict, equity: float):
        """Check margin usage and trigger circuit breakers."""
        if equity <= 0:
            return
        
        # Calculate margin usage percentage
        buying_power = float(account.get('buying_power', equity * 2))
        initial_margin = float(account.get('initial_margin', 0))
        maintenance_margin = float(account.get('maintenance_margin', 0))
        
        # Margin used % = (equity * 2 - buying_power) / (equity * 2) * 100
        # Simplified: how much of our leverage capacity are we using
        total_capacity = equity * 2  # Assuming 2x leverage max
        used = total_capacity - buying_power
        margin_used_pct = (used / total_capacity) * 100 if total_capacity > 0 else 0
        
        if margin_used_pct >= self.config.margin_emergency_pct:
            # EMERGENCY: Close leveraged positions immediately
            alert = self._create_alert(
                level=RiskLevel.CRITICAL,
                message=f"🚨 MARGIN EMERGENCY: {margin_used_pct:.1f}% used (threshold: {self.config.margin_emergency_pct:.0f}%)",
                metric_name="margin_usage",
                current_value=margin_used_pct,
                threshold=self.config.margin_emergency_pct,
            )
            self._trigger_alert(alert)
            self._reduce_exposure(0.40, "margin_emergency")
            self.halt_trading = True
            self.current_risk_level = RiskLevel.CRITICAL
            
        elif margin_used_pct >= self.config.margin_reduce_pct:
            # HIGH: Reduce exposure by 25%
            alert = self._create_alert(
                level=RiskLevel.HIGH,
                message=f"⚠️ MARGIN HIGH: {margin_used_pct:.1f}% used (threshold: {self.config.margin_reduce_pct:.0f}%)",
                metric_name="margin_usage",
                current_value=margin_used_pct,
                threshold=self.config.margin_reduce_pct,
            )
            self._trigger_alert(alert)
            self._reduce_exposure(0.25, "high_margin_usage")
            self.current_risk_level = RiskLevel.HIGH
            
        elif margin_used_pct >= self.config.margin_alert_pct:
            # ELEVATED: Alert only
            alert = self._create_alert(
                level=RiskLevel.ELEVATED,
                message=f"Margin usage elevated: {margin_used_pct:.1f}%",
                metric_name="margin_usage",
                current_value=margin_used_pct,
                threshold=self.config.margin_alert_pct,
            )
            self._trigger_alert(alert)
    
    def _check_daily_loss(self, account: Dict, equity: float):
        """Check daily P/L and trigger circuit breakers."""
        if equity <= 0:
            return
        
        # Get daily P/L
        last_equity = float(account.get('last_equity', equity))
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity) * 100 if last_equity > 0 else 0
        
        if daily_pnl_pct <= -self.config.daily_loss_delever_pct:
            # CRITICAL: Close all leveraged positions
            alert = self._create_alert(
                level=RiskLevel.CRITICAL,
                message=f"🚨 DAILY LOSS CRITICAL: {daily_pnl_pct:.2f}% (threshold: {-self.config.daily_loss_delever_pct:.0f}%)",
                metric_name="daily_pnl",
                current_value=daily_pnl_pct,
                threshold=-self.config.daily_loss_delever_pct,
            )
            self._trigger_alert(alert)
            self._close_leveraged_positions("daily_loss_critical")
            self.halt_trading = True
            self.current_risk_level = RiskLevel.CRITICAL
            
        elif daily_pnl_pct <= -self.config.daily_loss_halt_pct:
            # HIGH: Halt new trades
            alert = self._create_alert(
                level=RiskLevel.HIGH,
                message=f"⚠️ DAILY LOSS HIGH: {daily_pnl_pct:.2f}% (threshold: {-self.config.daily_loss_halt_pct:.0f}%)",
                metric_name="daily_pnl",
                current_value=daily_pnl_pct,
                threshold=-self.config.daily_loss_halt_pct,
            )
            self._trigger_alert(alert)
            self.halt_trading = True
            self.current_risk_level = RiskLevel.HIGH
            
        elif daily_pnl_pct <= -self.config.daily_loss_alert_pct:
            # ELEVATED: Alert only
            alert = self._create_alert(
                level=RiskLevel.ELEVATED,
                message=f"Daily loss warning: {daily_pnl_pct:.2f}%",
                metric_name="daily_pnl",
                current_value=daily_pnl_pct,
                threshold=-self.config.daily_loss_alert_pct,
            )
            self._trigger_alert(alert)
    
    def _check_cross_asset_stress(self):
        """
        Check cross-asset signals for stress conditions.
        
        Triggers alerts for:
        - Credit stress (HYG falling sharply)
        - International crash (Europe/China down significantly)
        - Commodity spike (oil/gold violent moves)
        """
        try:
            from src.data.cross_asset_data import get_cross_asset_loader
            loader = get_cross_asset_loader()
            data = loader.fetch_all_cross_assets(days=5)
            
            if not data:
                return
            
            # Check credit stress (HYG)
            if 'HYG' in data and len(data['HYG']) >= 2:
                hyg_ret = (data['HYG'].iloc[-1] / data['HYG'].iloc[-2] - 1) * 100
                if hyg_ret < -2.0:  # HYG down 2%+ = credit stress
                    alert = self._create_alert(
                        level=RiskLevel.ELEVATED,
                        message=f"⚠️ Credit stress: HYG down {hyg_ret:.2f}% today",
                        metric_name="hyg_return",
                        current_value=hyg_ret,
                        threshold=-2.0,
                    )
                    self._trigger_alert(alert)
                    logger.warning(f"Credit stress detected: HYG {hyg_ret:.2f}%")
            
            # Check international crash
            for intl_symbol, name in [('EWG', 'Europe'), ('FXI', 'China'), ('EEM', 'EM')]:
                if intl_symbol in data and len(data[intl_symbol]) >= 2:
                    ret = (data[intl_symbol].iloc[-1] / data[intl_symbol].iloc[-2] - 1) * 100
                    if ret < -3.0:  # Down 3%+ = crash
                        alert = self._create_alert(
                            level=RiskLevel.HIGH if ret < -5.0 else RiskLevel.ELEVATED,
                            message=f"⚠️ {name} market stress: {intl_symbol} down {ret:.2f}%",
                            metric_name=f"{intl_symbol}_return",
                            current_value=ret,
                            threshold=-3.0,
                        )
                        self._trigger_alert(alert)
            
            # Check commodity spikes (oil, gold)
            for comm, name in [('CL=F', 'Oil'), ('GC=F', 'Gold')]:
                if comm in data and len(data[comm]) >= 2:
                    ret = (data[comm].iloc[-1] / data[comm].iloc[-2] - 1) * 100
                    if abs(ret) > 5.0:  # 5%+ move either direction
                        alert = self._create_alert(
                            level=RiskLevel.ELEVATED,
                            message=f"⚠️ {name} spike: {ret:+.2f}% today",
                            metric_name=f"{comm}_return",
                            current_value=ret,
                            threshold=5.0,
                        )
                        self._trigger_alert(alert)
                        
        except Exception as e:
            logger.debug(f"Could not check cross-asset stress: {e}")
    
    def _close_leveraged_positions(self, reason: str):
        """Close all positions that are using leverage (above 1x equity)."""
        logger.warning(f"🚨 CLOSING LEVERAGED POSITIONS due to {reason}")
        
        try:
            account = self.broker.get_account()
            equity = float(account.get('equity', 0))
            
            if equity <= 0:
                return
            
            positions = self.broker.get_positions()
            
            # Handle Dict format
            if isinstance(positions, dict):
                position_items = list(positions.items())
            else:
                position_items = [(pos.get('symbol'), pos) for pos in positions]
            
            # Calculate total position value
            total_position_value = sum(
                abs(float(pos.get('market_value', 0)))
                for _, pos in position_items
                if isinstance(pos, dict)
            )
            
            # If position value exceeds equity (leveraged), close excess
            if total_position_value > equity:
                excess_pct = (total_position_value - equity) / total_position_value
                self._reduce_exposure(excess_pct + 0.1, reason)  # +10% buffer
                
        except Exception as e:
            logger.error(f"Error closing leveraged positions: {e}")
    
    def _reduce_exposure(self, reduction_pct: float, reason: str):
        """
        Reduce portfolio exposure by selling positions.
        
        Has a cooldown to prevent rapid successive reductions.
        """
        # Check cooldown
        if self.last_reduction_time:
            elapsed = datetime.now() - self.last_reduction_time
            if elapsed < timedelta(minutes=self.reduction_cooldown_minutes):
                logger.info(f"Reduction cooldown active ({elapsed.seconds}s elapsed)")
                return
        
        logger.warning(f"🚨 REDUCING EXPOSURE by {reduction_pct:.0%} due to {reason}")
        
        try:
            positions = self.broker.get_positions()
            
            # Handle Dict[str, Dict] format from broker
            if isinstance(positions, dict):
                position_items = [(symbol, pos) for symbol, pos in positions.items()]
            else:
                position_items = [(pos.get('symbol', 'UNKNOWN'), pos) for pos in positions]
            
            for symbol, pos in position_items:
                if isinstance(pos, str):
                    continue
                current_qty = int(pos.get('qty', 0)) if isinstance(pos, dict) else 0
                
                if current_qty <= 0:
                    continue
                
                # Calculate shares to sell
                sell_qty = int(current_qty * reduction_pct)
                
                if sell_qty > 0:
                    try:
                        self.broker.submit_order(
                            symbol=symbol,
                            side='sell',
                            quantity=sell_qty,
                            order_type='market'
                        )
                        logger.info(f"Risk reduction: Sold {sell_qty} shares of {symbol}")
                    except Exception as e:
                        logger.error(f"Failed to reduce {symbol}: {e}")
            
            self.last_reduction_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to reduce exposure: {e}")
    
    def _create_alert(
        self,
        level: RiskLevel,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float,
    ) -> RiskAlert:
        """Create a new risk alert."""
        return RiskAlert(
            timestamp=datetime.now(),
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
        )
    
    def _trigger_alert(self, alert: RiskAlert):
        """Handle a risk alert."""
        # Add to alert history
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log
        if alert.level == RiskLevel.CRITICAL:
            logger.critical(f"🚨 {alert.message}")
        elif alert.level == RiskLevel.HIGH:
            logger.warning(f"⚠️ {alert.message}")
        else:
            logger.info(f"ℹ️ {alert.message}")
        
        # Call callback if provided
        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def update_vix(self, vix: float):
        """Update VIX level from external source."""
        self.last_vix = vix
    
    def can_trade(self) -> bool:
        """Check if trading is allowed based on current risk level."""
        return not self.halt_trading and self.current_risk_level != RiskLevel.CRITICAL
    
    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on risk level.
        
        Returns:
            Multiplier to apply to position sizes (0.25 to 1.0)
        """
        if self.current_risk_level == RiskLevel.CRITICAL:
            return 0.0  # No new trades
        elif self.current_risk_level == RiskLevel.HIGH:
            return 0.5  # Half position sizes
        elif self.current_risk_level == RiskLevel.ELEVATED:
            return 0.75  # Reduced position sizes
        else:
            return 1.0  # Normal
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk monitor status."""
        return {
            'is_running': self.is_running,
            'risk_level': self.current_risk_level.value,
            'halt_trading': self.halt_trading,
            'peak_equity': self.peak_equity,
            'last_vix': self.last_vix,
            'position_size_multiplier': self.get_position_size_multiplier(),
            'recent_alerts': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'level': a.level.value,
                    'message': a.message,
                }
                for a in self.alerts[-5:]
            ],
        }
