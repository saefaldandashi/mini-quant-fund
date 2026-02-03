"""
Earnings Season Auto-Monitor

Runs in the background and automatically triggers rebalance when:
1. Major earnings news is detected (drops > 5%, beats with surge > 5%)
2. High-confidence earnings signals are generated
3. Multiple sources confirm the same signal

This ensures we don't miss opportunities like MSFT dropping 10% on bad earnings.
"""
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
import pytz

from src.strategies.earnings_reactor import get_earnings_reactor, EarningsSignal

logger = logging.getLogger(__name__)


@dataclass
class EarningsMonitorConfig:
    """Configuration for the earnings monitor."""
    
    # Check interval (seconds) - more frequent during market hours
    check_interval_market_hours: int = 300  # 5 minutes
    check_interval_after_hours: int = 900   # 15 minutes
    
    # Thresholds for auto-trigger
    min_confidence_auto_trigger: float = 0.55  # 55% confidence
    min_price_move_auto_trigger: float = 5.0   # 5% price move mentioned
    
    # How many signals needed to trigger
    min_signals_for_trigger: int = 1  # Just 1 high-confidence signal is enough
    
    # Cooldown after triggering (prevent spam)
    trigger_cooldown_minutes: int = 30
    
    # Only trade these symbols (None = all)
    whitelist_symbols: Optional[Set[str]] = None
    
    # Never trade these symbols
    blacklist_symbols: Set[str] = field(default_factory=lambda: {'SPY', 'QQQ', 'IWM', 'DIA'})
    
    # Enable during market hours only
    market_hours_only: bool = False
    
    # Maximum auto-triggers per day
    max_triggers_per_day: int = 5


@dataclass
class TriggerEvent:
    """Record of an auto-trigger event."""
    timestamp: datetime
    signals: List[EarningsSignal]
    reason: str
    success: bool = False
    result: Optional[str] = None


class EarningsMonitor:
    """
    Background monitor that watches for earnings events and auto-triggers rebalance.
    """
    
    def __init__(self, config: Optional[EarningsMonitorConfig] = None):
        self.config = config or EarningsMonitorConfig()
        self.reactor = get_earnings_reactor()
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Tracking
        self._processed_signals: Set[str] = set()  # Signals we've already acted on
        self._trigger_history: List[TriggerEvent] = []
        self._last_trigger: Optional[datetime] = None
        self._triggers_today: int = 0
        self._last_trigger_date: Optional[datetime] = None
        
        # Callback for triggering rebalance
        self._rebalance_callback: Optional[Callable] = None
        
        # Stats
        self._scans_performed: int = 0
        self._signals_detected: int = 0
        self._auto_triggers: int = 0
    
    def set_rebalance_callback(self, callback: Callable):
        """Set the callback function to trigger rebalance."""
        self._rebalance_callback = callback
    
    def _get_signal_key(self, signal: EarningsSignal) -> str:
        """Generate unique key for a signal to prevent duplicates."""
        return f"{signal.symbol}:{signal.direction}:{signal.headline[:50]}"
    
    def _should_trigger(self, signals: List[EarningsSignal]) -> tuple[bool, str]:
        """
        Determine if we should auto-trigger rebalance.
        
        Returns:
            Tuple of (should_trigger, reason)
        """
        # Check cooldown
        if self._last_trigger:
            cooldown = timedelta(minutes=self.config.trigger_cooldown_minutes)
            if datetime.now(pytz.UTC) - self._last_trigger < cooldown:
                mins_left = (cooldown - (datetime.now(pytz.UTC) - self._last_trigger)).seconds // 60
                return False, f"Cooldown active ({mins_left}m remaining)"
        
        # Check daily limit
        today = datetime.now(pytz.UTC).date()
        if self._last_trigger_date != today:
            self._triggers_today = 0
            self._last_trigger_date = today
        
        if self._triggers_today >= self.config.max_triggers_per_day:
            return False, f"Daily limit reached ({self.config.max_triggers_per_day})"
        
        # Filter to new, high-confidence signals
        actionable_signals = []
        
        for signal in signals:
            key = self._get_signal_key(signal)
            
            # Skip already processed
            if key in self._processed_signals:
                continue
            
            # Check symbol whitelist/blacklist
            if self.config.whitelist_symbols and signal.symbol not in self.config.whitelist_symbols:
                continue
            if signal.symbol in self.config.blacklist_symbols:
                continue
            
            # Check confidence threshold
            if signal.confidence >= self.config.min_confidence_auto_trigger:
                actionable_signals.append(signal)
            
            # Check price move threshold
            elif signal.price_move_pct and abs(signal.price_move_pct) >= self.config.min_price_move_auto_trigger:
                actionable_signals.append(signal)
        
        if len(actionable_signals) >= self.config.min_signals_for_trigger:
            symbols = [s.symbol for s in actionable_signals]
            directions = [s.direction for s in actionable_signals]
            return True, f"High-confidence signals: {', '.join(f'{d.upper()} {s}' for s, d in zip(symbols, directions))}"
        
        return False, "No actionable signals"
    
    def _perform_scan(self):
        """Perform a single scan of news sources."""
        try:
            from app import geopolitical_intel, alpha_vantage_news
            
            # Scan sources
            geo_signals = self.reactor.scan_geopolitical_intel(geopolitical_intel)
            av_signals = self.reactor.scan_alpha_vantage(alpha_vantage_news)
            
            self._scans_performed += 1
            
            # Get all recent signals
            all_signals = self.reactor.get_recent_signals(max_age_hours=4)
            self._signals_detected = len(all_signals)
            
            # Check if we should trigger
            should_trigger, reason = self._should_trigger(all_signals)
            
            if should_trigger:
                logger.info(f"🚨 AUTO-TRIGGER: {reason}")
                self._trigger_rebalance(all_signals, reason)
            else:
                logger.debug(f"Scan complete: {len(all_signals)} signals, no trigger ({reason})")
                
        except Exception as e:
            logger.error(f"Earnings monitor scan error: {e}")
    
    def _trigger_rebalance(self, signals: List[EarningsSignal], reason: str):
        """Trigger an automatic rebalance."""
        # Mark signals as processed
        for signal in signals:
            key = self._get_signal_key(signal)
            self._processed_signals.add(key)
        
        # Record the trigger
        event = TriggerEvent(
            timestamp=datetime.now(pytz.UTC),
            signals=signals,
            reason=reason,
        )
        
        self._last_trigger = datetime.now(pytz.UTC)
        self._triggers_today += 1
        self._auto_triggers += 1
        
        # Execute callback
        if self._rebalance_callback:
            try:
                logger.info(f"🎯 Executing auto-rebalance: {reason}")
                
                # Call the rebalance function
                result = self._rebalance_callback(dry_run=False, allow_after_hours=True, force_rebalance=True)
                
                event.success = result[0] if isinstance(result, tuple) else bool(result)
                event.result = str(result[1])[:200] if isinstance(result, tuple) and len(result) > 1 else str(result)[:200]
                
                logger.info(f"✅ Auto-rebalance completed: {'SUCCESS' if event.success else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"Auto-rebalance error: {e}")
                event.success = False
                event.result = str(e)
        else:
            logger.warning("No rebalance callback set - cannot auto-trigger")
            event.result = "No callback set"
        
        self._trigger_history.append(event)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        logger.info("📡 Earnings monitor started")
        
        while self._running:
            try:
                # Determine check interval based on market hours
                # Simple check: 9:30 AM - 4:00 PM ET on weekdays
                now = datetime.now(pytz.timezone('US/Eastern'))
                is_market_hours = (
                    now.weekday() < 5 and  # Monday-Friday
                    now.hour >= 9 and now.hour < 16 and
                    (now.hour > 9 or now.minute >= 30)
                )
                
                if self.config.market_hours_only and not is_market_hours:
                    logger.debug("Outside market hours, skipping scan")
                    time.sleep(self.config.check_interval_after_hours)
                    continue
                
                # Perform scan
                self._perform_scan()
                
                # Sleep until next scan
                interval = (
                    self.config.check_interval_market_hours 
                    if is_market_hours 
                    else self.config.check_interval_after_hours
                )
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(60)  # Wait a minute on error
        
        logger.info("📡 Earnings monitor stopped")
    
    def start(self):
        """Start the background monitor."""
        with self._lock:
            if self._running:
                logger.warning("Earnings monitor already running")
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("🚀 Earnings monitor STARTED")
    
    def stop(self):
        """Stop the background monitor."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._thread:
                self._thread.join(timeout=5)
                self._thread = None
            logger.info("🛑 Earnings monitor STOPPED")
    
    def force_scan(self) -> Dict:
        """Force an immediate scan (for testing/debugging)."""
        self._perform_scan()
        
        return {
            "scans_performed": self._scans_performed,
            "signals_detected": self._signals_detected,
            "auto_triggers": self._auto_triggers,
            "last_trigger": self._last_trigger.isoformat() if self._last_trigger else None,
            "triggers_today": self._triggers_today,
        }
    
    def get_status(self) -> Dict:
        """Get current monitor status."""
        return {
            "running": self._running,
            "scans_performed": self._scans_performed,
            "signals_detected": self._signals_detected,
            "auto_triggers": self._auto_triggers,
            "triggers_today": self._triggers_today,
            "max_triggers_per_day": self.config.max_triggers_per_day,
            "last_trigger": self._last_trigger.isoformat() if self._last_trigger else None,
            "check_interval_market": f"{self.config.check_interval_market_hours}s",
            "check_interval_after_hours": f"{self.config.check_interval_after_hours}s",
            "min_confidence": f"{self.config.min_confidence_auto_trigger:.0%}",
            "recent_triggers": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "reason": t.reason,
                    "success": t.success,
                    "symbols": [s.symbol for s in t.signals],
                }
                for t in self._trigger_history[-10:]
            ],
        }
    
    def clear_processed(self):
        """Clear processed signals (allows re-triggering on same news)."""
        self._processed_signals.clear()
        logger.info("Cleared processed signals")


# Singleton instance
_earnings_monitor: Optional[EarningsMonitor] = None


def get_earnings_monitor() -> EarningsMonitor:
    """Get the singleton earnings monitor instance."""
    global _earnings_monitor
    if _earnings_monitor is None:
        _earnings_monitor = EarningsMonitor()
    return _earnings_monitor


def start_earnings_monitor(rebalance_callback: Callable) -> EarningsMonitor:
    """Start the earnings monitor with a rebalance callback."""
    monitor = get_earnings_monitor()
    monitor.set_rebalance_callback(rebalance_callback)
    monitor.start()
    return monitor


def stop_earnings_monitor():
    """Stop the earnings monitor."""
    global _earnings_monitor
    if _earnings_monitor:
        _earnings_monitor.stop()
