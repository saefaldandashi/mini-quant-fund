"""
WIN ACCELERATION ENGINE

When the system is winning, PRESS THE ADVANTAGE.

This is the mirror of the loss awareness system but for the upside.
Instead of only reducing exposure on bad days, this module INCREASES
exposure on good days — creating the asymmetric return profile needed
for 100-300% annual returns.

Key principles:
1. When strategies are working, deploy MORE capital (not less)
2. Pyramid into winners — add to positions that are already profitable
3. Track win streaks and accelerate proportionally
4. Never violate core risk limits — acceleration works WITHIN the safety net
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AccelerationMode(Enum):
    """Current acceleration mode."""
    SURGE = "surge"           # Maximum press — daily +2% AND weekly +5%
    MOMENTUM = "momentum"     # Strong press — daily +1% AND weekly +3%
    PRESSING = "pressing"     # Moderate press — daily +0.5%
    NORMAL = "normal"         # No acceleration
    CAUTIOUS = "cautious"     # Slight pullback
    DEFENSIVE = "defensive"   # Loss awareness took over


@dataclass 
class AccelerationState:
    """Current state of the acceleration engine."""
    mode: AccelerationMode = AccelerationMode.NORMAL
    multiplier: float = 1.0
    
    # P&L tracking
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    
    # Streak tracking
    consecutive_wins: int = 0
    win_streak_bonus: float = 0.0
    
    # What's working
    top_performing_sectors: List[str] = field(default_factory=list)
    top_performing_strategies: List[str] = field(default_factory=list)
    
    # Pyramiding candidates
    pyramid_candidates: List[Dict] = field(default_factory=list)
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.now)


class WinAccelerator:
    """
    Tracks performance and accelerates exposure when winning.
    
    Acceleration tiers:
    - Daily P&L > +0.5%: 1.10x (10% more exposure)
    - Daily P&L > +1.0% AND Weekly > +2%: 1.25x
    - Daily P&L > +1.5% AND Weekly > +3%: 1.35x  
    - Daily P&L > +2.0% AND Weekly > +5%: 1.50x (maximum)
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.state = AccelerationState()
        
        # Configuration
        self.acceleration_tiers = [
            # (daily_min, weekly_min, multiplier, mode)
            (0.020, 0.050, 1.50, AccelerationMode.SURGE),
            (0.015, 0.030, 1.35, AccelerationMode.MOMENTUM),
            (0.010, 0.020, 1.25, AccelerationMode.MOMENTUM),
            (0.005, 0.000, 1.10, AccelerationMode.PRESSING),
        ]
        
        # Win streak configuration
        self.streak_bonus_per_win = 0.02  # +2% per consecutive win
        self.max_streak_bonus = 0.15  # Max +15% from streak
        
        # Pyramiding configuration
        self.pyramid_threshold = 0.03  # Add to positions up > 3%
        self.pyramid_max_addition = 0.50  # Max 50% more on top of original
        
        # History
        self.daily_pnl_history: List[float] = []
        self.acceleration_history: List[Dict] = []
    
    def update(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float = 0.0,
        positions: Optional[List[Dict]] = None,
        consecutive_wins: int = 0,
    ) -> AccelerationState:
        """
        Update the acceleration engine with current performance data.
        
        Args:
            daily_pnl_pct: Today's P&L as percentage
            weekly_pnl_pct: This week's P&L as percentage
            positions: Current positions with unrealized P&L
            consecutive_wins: Number of consecutive profitable periods
        
        Returns:
            Updated AccelerationState
        """
        if not self.enabled:
            self.state = AccelerationState()
            return self.state
        
        self.state.daily_pnl_pct = daily_pnl_pct
        self.state.weekly_pnl_pct = weekly_pnl_pct
        self.state.consecutive_wins = consecutive_wins
        self.state.last_updated = datetime.now()
        
        # Track daily P&L history
        self.daily_pnl_history.append(daily_pnl_pct)
        if len(self.daily_pnl_history) > 30:
            self.daily_pnl_history = self.daily_pnl_history[-30:]
        
        # Determine acceleration tier
        self.state.mode = AccelerationMode.NORMAL
        self.state.multiplier = 1.0
        
        for daily_min, weekly_min, multiplier, mode in self.acceleration_tiers:
            if daily_pnl_pct >= daily_min and weekly_pnl_pct >= weekly_min:
                self.state.mode = mode
                self.state.multiplier = multiplier
                break
        
        # Add win streak bonus
        self.state.win_streak_bonus = 0.0
        if consecutive_wins >= 2:
            self.state.win_streak_bonus = min(
                self.max_streak_bonus,
                consecutive_wins * self.streak_bonus_per_win
            )
            self.state.multiplier += self.state.win_streak_bonus
        
        # Cap total multiplier
        self.state.multiplier = min(1.50, self.state.multiplier)
        
        # If daily P&L is negative, don't accelerate
        if daily_pnl_pct < 0:
            if daily_pnl_pct < -0.005:
                self.state.mode = AccelerationMode.DEFENSIVE
                self.state.multiplier = 1.0  # Don't reduce — loss awareness handles that
            else:
                self.state.mode = AccelerationMode.CAUTIOUS
                self.state.multiplier = 1.0
        
        # Identify pyramiding candidates
        self.state.pyramid_candidates = []
        if positions and self.state.mode in [AccelerationMode.SURGE, AccelerationMode.MOMENTUM]:
            for pos in positions:
                unrealized_plpc = pos.get('unrealized_plpc', 0)
                if unrealized_plpc and unrealized_plpc > self.pyramid_threshold:
                    self.state.pyramid_candidates.append({
                        'symbol': pos.get('symbol', '?'),
                        'pnl_pct': unrealized_plpc,
                        'suggested_addition': min(
                            self.pyramid_max_addition,
                            unrealized_plpc * 2  # Scale addition to profitability
                        ),
                    })
        
        # Store in history
        self.acceleration_history.append({
            'timestamp': datetime.now().isoformat(),
            'mode': self.state.mode.value,
            'multiplier': self.state.multiplier,
            'daily_pnl': daily_pnl_pct,
            'weekly_pnl': weekly_pnl_pct,
            'streak': consecutive_wins,
        })
        if len(self.acceleration_history) > 100:
            self.acceleration_history = self.acceleration_history[-100:]
        
        return self.state
    
    def get_multiplier(self) -> float:
        """Get the current acceleration multiplier."""
        if not self.enabled:
            return 1.0
        return self.state.multiplier
    
    def get_pyramid_weights(
        self,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Get adjusted weights with pyramiding applied to winners.
        
        Only applies in SURGE or MOMENTUM mode.
        
        Args:
            current_weights: Current position weights
        
        Returns:
            Adjusted weights with pyramiding
        """
        if not self.enabled or not self.state.pyramid_candidates:
            return current_weights
        
        adjusted = dict(current_weights)
        
        for candidate in self.state.pyramid_candidates:
            symbol = candidate['symbol']
            if symbol in adjusted and adjusted[symbol] > 0:
                addition = adjusted[symbol] * candidate['suggested_addition']
                adjusted[symbol] += addition
                logger.info(
                    f"🔺 PYRAMID: Adding {addition:.1%} to {symbol} "
                    f"(up {candidate['pnl_pct']:.1%})"
                )
        
        return adjusted
    
    def get_status(self) -> Dict:
        """Get current acceleration status for API/frontend."""
        return {
            'enabled': self.enabled,
            'mode': self.state.mode.value,
            'multiplier': self.state.multiplier,
            'daily_pnl_pct': self.state.daily_pnl_pct,
            'weekly_pnl_pct': self.state.weekly_pnl_pct,
            'consecutive_wins': self.state.consecutive_wins,
            'win_streak_bonus': self.state.win_streak_bonus,
            'pyramid_candidates': len(self.state.pyramid_candidates),
            'pyramid_details': self.state.pyramid_candidates[:5],  # Top 5
            'last_updated': self.state.last_updated.isoformat(),
            'history_length': len(self.acceleration_history),
        }
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recent acceleration history."""
        return self.acceleration_history[-limit:]


# Singleton
_win_accelerator: Optional[WinAccelerator] = None


def get_win_accelerator() -> WinAccelerator:
    """Get the singleton win accelerator."""
    global _win_accelerator
    if _win_accelerator is None:
        _win_accelerator = WinAccelerator()
    return _win_accelerator
