"""
PERFORMANCE AWARENESS & ADAPTATION SYSTEM

This module makes the bot self-aware of its performance and forces it to:
1. Recognize when it's losing money → reduce exposure, exit losers
2. Recognize when it's WINNING → INCREASE exposure, press advantage
3. Analyze WHY performance is what it is
4. Adapt strategy dynamically

KEY CHANGE: System is now ASYMMETRIC:
- When winning: PRESS the advantage (increase exposure up to 1.5x)
- When losing: REDUCE exposure (existing behavior, slightly relaxed)
- This is the single most important change for 100-300% annual returns
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PerformanceState(Enum):
    """Current performance state of the portfolio — now includes UPSIDE states."""
    SURGE = "surge"              # > 2% up today AND > 5% up this week (NEW)
    MOMENTUM = "momentum"        # > 1% up today AND > 3% up this week (NEW)
    EXCELLENT = "excellent"      # > 1.5% up today
    GOOD = "good"                # 0.3% to 1.5% up
    NEUTRAL = "neutral"          # -0.3% to 0.3%
    CONCERNING = "concerning"    # -0.3% to -1.5%
    BAD = "bad"                  # -1.5% to -3%
    CRITICAL = "critical"        # > -3% (circuit breaker territory)


@dataclass
class LossAnalysis:
    """Analysis of portfolio performance (wins AND losses)."""
    total_pnl: float
    total_pnl_pct: float
    state: PerformanceState
    losing_positions: List[Dict]
    winning_positions: List[Dict]
    biggest_loser: Optional[Dict]
    biggest_winner: Optional[Dict]
    
    # Reasons for performance
    reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Exposure adjustments
    recommended_exposure: float = 0.95  # Default 95% (was 80%)
    should_exit_losers: bool = False
    should_go_to_cash: bool = False
    positions_to_exit: List[str] = field(default_factory=list)
    
    # Win acceleration metadata
    acceleration_active: bool = False
    acceleration_multiplier: float = 1.0


class LossAwarenessSystem:
    """
    Makes the trading bot self-aware of its performance.
    
    ASYMMETRIC BY DESIGN:
    - When winning: INCREASE exposure (press advantage)
    - When losing: REDUCE exposure (protect capital)
    """
    
    def __init__(self):
        # Performance thresholds — RETUNED
        self.thresholds = {
            "excellent": 0.015,     # > 1.5% (was 2%)
            "good": 0.003,         # > 0.3% (was 0.5%)
            "neutral": -0.003,     # > -0.3% (was -0.5%)
            "concerning": -0.015,  # > -1.5% (unchanged)
            "bad": -0.03,          # > -3% (unchanged)
        }
        
        # Weekly P&L tracking for MOMENTUM/SURGE detection
        self.weekly_pnl_pct: float = 0.0
        
        # Exposure adjustments by state — NOW ASYMMETRIC (key change)
        # CRITICAL FIX: Less aggressive reduction to maintain better capital deployment
        self.exposure_adjustments = {
            PerformanceState.SURGE: 1.50,       # 150% exposure — PRESS HARD (NEW)
            PerformanceState.MOMENTUM: 1.35,    # 135% exposure — press advantage (NEW)
            PerformanceState.EXCELLENT: 1.20,   # 120% exposure — INCREASE (was 1.0!)
            PerformanceState.GOOD: 1.05,        # 105% — slight boost (was 0.9 — REDUCING on good days!)
            PerformanceState.NEUTRAL: 0.95,     # 95% — near-full (was 0.8)
            PerformanceState.CONCERNING: 0.75,  # CRITICAL FIX: 75% (was 65%) — less aggressive
            PerformanceState.BAD: 0.55,         # CRITICAL FIX: 55% (was 40%) — less aggressive
            PerformanceState.CRITICAL: 0.30,     # CRITICAL FIX: 30% (was 15%) — maintain some activity
        }
        
        # Loss tolerance for individual positions
        self.position_loss_threshold = -0.06  # Exit at -6% loss (was -5%)
        self.position_loss_fast_exit = -0.04  # Consider exit at -4% if overall losing (was -3%)
        
        # Tracking
        self.analysis_history: List[LossAnalysis] = []
        self.last_analysis: Optional[LossAnalysis] = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0  # NEW: track winning streaks
        
    def update_weekly_pnl(self, weekly_pnl_pct: float):
        """Update weekly P&L for MOMENTUM/SURGE detection."""
        self.weekly_pnl_pct = weekly_pnl_pct
        
    def get_performance_state(self, pnl_pct: float) -> PerformanceState:
        """Determine current performance state — includes new UPSIDE states."""
        # Check for SURGE first (requires both daily AND weekly conditions)
        if pnl_pct > 0.02 and self.weekly_pnl_pct > 0.05:
            return PerformanceState.SURGE
        
        # Check for MOMENTUM
        if pnl_pct > 0.01 and self.weekly_pnl_pct > 0.03:
            return PerformanceState.MOMENTUM
        
        # Standard states
        if pnl_pct > self.thresholds["excellent"]:
            return PerformanceState.EXCELLENT
        elif pnl_pct > self.thresholds["good"]:
            return PerformanceState.GOOD
        elif pnl_pct > self.thresholds["neutral"]:
            return PerformanceState.NEUTRAL
        elif pnl_pct > self.thresholds["concerning"]:
            return PerformanceState.CONCERNING
        elif pnl_pct > self.thresholds["bad"]:
            return PerformanceState.BAD
        else:
            return PerformanceState.CRITICAL
    
    def analyze_losses(
        self,
        positions: List[Dict],
        total_pnl: float,
        total_pnl_pct: float,
        market_data: Optional[Dict] = None,
    ) -> LossAnalysis:
        """
        Analyze portfolio performance and determine optimal response.
        
        Args:
            positions: List of position dicts with symbol, pnl, pnl_pct, etc.
            total_pnl: Total P&L in dollars
            total_pnl_pct: Total P&L as percentage
            market_data: Optional dict with market context (SPY change, VIX, etc.)
        
        Returns:
            LossAnalysis with reasons and recommendations
        """
        state = self.get_performance_state(total_pnl_pct)
        
        # Separate winners and losers
        losers = [p for p in positions if p.get('unrealized_pl', 0) < 0]
        winners = [p for p in positions if p.get('unrealized_pl', 0) >= 0]
        
        # Sort by P&L
        losers.sort(key=lambda x: x.get('unrealized_pl', 0))
        winners.sort(key=lambda x: x.get('unrealized_pl', 0), reverse=True)
        
        analysis = LossAnalysis(
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            state=state,
            losing_positions=losers,
            winning_positions=winners,
            biggest_loser=losers[0] if losers else None,
            biggest_winner=winners[0] if winners else None,
        )
        
        # Analyze reasons
        self._analyze_reasons(analysis, market_data)
        
        # Generate recommendations
        self._generate_recommendations(analysis)
        
        # Store analysis
        self.last_analysis = analysis
        self.analysis_history.append(analysis)
        
        # Track consecutive wins/losses
        if state in [PerformanceState.CONCERNING, PerformanceState.BAD, PerformanceState.CRITICAL]:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        elif state in [PerformanceState.GOOD, PerformanceState.EXCELLENT, PerformanceState.MOMENTUM, PerformanceState.SURGE]:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            # Neutral — don't reset streaks immediately
            pass
        
        return analysis
    
    def _analyze_reasons(self, analysis: LossAnalysis, market_data: Optional[Dict]):
        """Identify reasons for performance."""
        reasons = []
        
        # Check if most positions are losing
        total_positions = len(analysis.losing_positions) + len(analysis.winning_positions)
        if total_positions > 0:
            loser_ratio = len(analysis.losing_positions) / total_positions
            winner_ratio = len(analysis.winning_positions) / total_positions
            
            if winner_ratio > 0.7:
                reasons.append(f"Most positions ({winner_ratio:.0%}) are profitable — strategies working well")
            elif loser_ratio > 0.7:
                reasons.append(f"Most positions ({loser_ratio:.0%}) are in the red — possible market-wide downturn")
            elif loser_ratio > 0.5:
                reasons.append(f"More than half ({loser_ratio:.0%}) of positions are losing")
        
        # Check for concentrated losses
        if analysis.biggest_loser:
            loser_pnl = analysis.biggest_loser.get('unrealized_pl', 0)
            if analysis.total_pnl != 0 and loser_pnl != 0:
                if analysis.total_pnl < 0 and abs(loser_pnl / analysis.total_pnl) > 0.5:
                    sym = analysis.biggest_loser.get('symbol', '?')
                    reasons.append(f"Concentrated loss: {sym} accounts for >50% of total losses")
        
        # Check for big winners
        if analysis.biggest_winner:
            winner_pnl = analysis.biggest_winner.get('unrealized_pl', 0)
            if winner_pnl > 0:
                sym = analysis.biggest_winner.get('symbol', '?')
                reasons.append(f"Top winner: {sym} at +${winner_pnl:,.2f}")
        
        # Check market context
        if market_data:
            spy_change = market_data.get('spy_change', 0)
            vix = market_data.get('vix', 0)
            
            if spy_change < -0.01:
                reasons.append(f"Market is down (SPY {spy_change*100:+.1f}%) — broad selloff")
            elif spy_change > 0.01:
                reasons.append(f"Market is up (SPY {spy_change*100:+.1f}%) — broad rally")
            if vix > 25:
                reasons.append(f"High volatility (VIX {vix:.1f}) — unstable conditions")
            elif vix < 15:
                reasons.append(f"Low volatility (VIX {vix:.1f}) — calm market, good for leverage")
        
        # Check for individual position issues
        for pos in analysis.losing_positions:
            pnl_pct = pos.get('unrealized_plpc', 0)
            sym = pos.get('symbol', '?')
            if pnl_pct < -0.06:
                reasons.append(f"{sym} down {pnl_pct*100:.1f}% — significant individual loss")
        
        if not reasons:
            reasons.append("General market fluctuation — no specific catalyst identified")
        
        analysis.reasons = reasons
    
    def _generate_recommendations(self, analysis: LossAnalysis):
        """Generate actionable recommendations — NOW INCLUDES WIN ACCELERATION."""
        recommendations = []
        positions_to_exit = []
        
        state = analysis.state
        
        # === WINNING STATES (NEW) ===
        if state == PerformanceState.SURGE:
            recommendations.append("🚀🚀 SURGE MODE: Strategy is crushing it — MAXIMUM PRESS")
            recommendations.append("Increase exposure to 150%, pyramid into winners")
            recommendations.append("Add to best-performing positions")
            analysis.recommended_exposure = 1.50
            analysis.acceleration_active = True
            analysis.acceleration_multiplier = 1.50
            
        elif state == PerformanceState.MOMENTUM:
            recommendations.append("🔥 MOMENTUM: Strong winning streak — press advantage")
            recommendations.append("Increase exposure to 135%")
            analysis.recommended_exposure = 1.35
            analysis.acceleration_active = True
            analysis.acceleration_multiplier = 1.35
            
        elif state == PerformanceState.EXCELLENT:
            recommendations.append("🚀 EXCELLENT: Strategy is working well — INCREASE exposure")
            recommendations.append("Scale up to 120% — let the system work")
            analysis.recommended_exposure = 1.20
            analysis.acceleration_active = True
            analysis.acceleration_multiplier = 1.20
            
        elif state == PerformanceState.GOOD:
            recommendations.append("📈 GOOD: Performance positive — slight boost")
            analysis.recommended_exposure = 1.05
            
        elif state == PerformanceState.NEUTRAL:
            recommendations.append("📊 NEUTRAL: Maintain near-full exposure (95%)")
            analysis.recommended_exposure = 0.95
            
        # === LOSING STATES ===
        elif state == PerformanceState.CONCERNING:
            recommendations.append("⚠️ CONCERNING: Reduce exposure to 75%")  # CRITICAL FIX: Updated
            recommendations.append("Consider exiting worst performers")
            analysis.recommended_exposure = 0.75  # CRITICAL FIX: 75% (was 65%)
            if analysis.biggest_loser:
                if analysis.biggest_loser.get('unrealized_plpc', 0) < self.position_loss_threshold:
                    positions_to_exit.append(analysis.biggest_loser.get('symbol'))
                    
        elif state == PerformanceState.BAD:
            recommendations.append("📉 BAD: Significant losses — reduce exposure to 55%")  # CRITICAL FIX: Updated
            recommendations.append("Exit positions with losses > 4%")
            analysis.recommended_exposure = 0.55  # CRITICAL FIX: 55% (was 40%)
            analysis.should_exit_losers = True
            for pos in analysis.losing_positions:
                if pos.get('unrealized_plpc', 0) < self.position_loss_fast_exit:
                    positions_to_exit.append(pos.get('symbol'))
                    
        elif state == PerformanceState.CRITICAL:
            recommendations.append("🚨 CRITICAL: Major losses — reduce to 30% exposure")  # CRITICAL FIX: Updated
            recommendations.append("Keep only best shorts and hedges active")
            analysis.recommended_exposure = 0.30  # CRITICAL FIX: 30% (was 15%)
            analysis.should_exit_losers = True
            # Exit all losing longs
            for pos in analysis.losing_positions:
                positions_to_exit.append(pos.get('symbol'))
        
        # Win streak bonus
        if self.consecutive_wins >= 3:
            recommendations.append(f"🔥 {self.consecutive_wins} consecutive wins — confidence high")
            # Boost exposure further on win streaks (up to 10% bonus)
            streak_bonus = min(0.10, self.consecutive_wins * 0.02)
            analysis.recommended_exposure = min(1.50, analysis.recommended_exposure + streak_bonus)
        
        # Consecutive losses recommendation — CRITICAL FIX: More tolerant
        if self.consecutive_losses >= 5:
            recommendations.append(f"⚠️ {self.consecutive_losses} consecutive losing periods — STRATEGY MAY NOT BE WORKING")
            recommendations.append("Consider going to 35% exposure until conditions improve")  # CRITICAL FIX: 35% (was 25%)
            analysis.should_go_to_cash = True
            analysis.recommended_exposure = min(analysis.recommended_exposure, 0.35)  # CRITICAL FIX: 35% (was 25%)
        elif self.consecutive_losses >= 3:
            recommendations.append(f"⚠️ {self.consecutive_losses} consecutive losses — reducing confidence")
            analysis.recommended_exposure = min(analysis.recommended_exposure, 0.60)  # CRITICAL FIX: 60% (was 50%)
        
        analysis.recommendations = recommendations
        analysis.positions_to_exit = positions_to_exit
    
    def get_adjusted_exposure(self, base_exposure: float = 0.95) -> float:
        """
        Get exposure adjusted for current performance.
        
        ASYMMETRIC: Increases on wins, decreases on losses.
        """
        if not self.last_analysis:
            return base_exposure
        
        adjustment = self.exposure_adjustments.get(
            self.last_analysis.state,
            0.95
        )
        
        # Win streak bonus
        if self.consecutive_wins >= 3:
            streak_bonus = min(0.10, self.consecutive_wins * 0.02)
            adjustment = min(1.50, adjustment + streak_bonus)
        
        # Consecutive loss penalty (more tolerant — 5 losses instead of 2)
        if self.consecutive_losses >= 5:
            adjustment *= 0.5
        elif self.consecutive_losses >= 3:
            adjustment *= 0.75
        
        # For winning states, allow going ABOVE base exposure
        if adjustment > 1.0:
            return base_exposure * adjustment  # Can exceed base
        else:
            return min(base_exposure, adjustment)
    
    def get_acceleration_info(self) -> Dict:
        """Get current acceleration state for the frontend."""
        if not self.last_analysis:
            return {
                "state": "neutral",
                "multiplier": 1.0,
                "active": False,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
            }
        
        return {
            "state": self.last_analysis.state.value,
            "multiplier": self.last_analysis.acceleration_multiplier,
            "active": self.last_analysis.acceleration_active,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "recommended_exposure": self.last_analysis.recommended_exposure,
        }
    
    def get_analysis_summary(self) -> str:
        """Get a human-readable summary of current analysis."""
        if not self.last_analysis:
            return "No analysis available yet"
        
        a = self.last_analysis
        lines = [
            "=" * 60,
            "📊 PERFORMANCE AWARENESS ANALYSIS",
            "=" * 60,
            f"State: {a.state.value.upper()}",
            f"Total P&L: ${a.total_pnl:+,.2f} ({a.total_pnl_pct*100:+.2f}%)",
            f"Winning positions: {len(a.winning_positions)}",
            f"Losing positions: {len(a.losing_positions)}",
            f"Win streak: {self.consecutive_wins} | Loss streak: {self.consecutive_losses}",
            "",
        ]
        
        if a.acceleration_active:
            lines.append(f"🔥 ACCELERATION ACTIVE: {a.acceleration_multiplier:.0%} exposure")
            lines.append("")
        
        lines.append("REASONS:")
        for reason in a.reasons:
            lines.append(f"  • {reason}")
        
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        for rec in a.recommendations:
            lines.append(f"  → {rec}")
        
        lines.append("")
        lines.append(f"Recommended exposure: {a.recommended_exposure*100:.0f}%")
        
        if a.positions_to_exit:
            lines.append(f"Positions to EXIT: {', '.join(a.positions_to_exit)}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def should_trade(self) -> Tuple[bool, str]:
        """
        Determine if the bot should continue trading.
        
        Returns:
            Tuple of (should_trade, reason)
        """
        if not self.last_analysis:
            return True, "No analysis yet — proceed with caution"
        
        a = self.last_analysis
        
        if a.should_go_to_cash:
            return False, f"Performance is {a.state.value} — GO TO CASH"
        
        if a.state == PerformanceState.CRITICAL:
            return False, "Critical losses — trading halted"
        
        if self.consecutive_losses >= 7:
            return False, f"{self.consecutive_losses} consecutive losses — strategy not working"
        
        return True, f"OK to trade at {a.recommended_exposure*100:.0f}% exposure"


# Singleton instance
_loss_awareness: Optional[LossAwarenessSystem] = None


def get_loss_awareness() -> LossAwarenessSystem:
    """Get the singleton loss awareness system."""
    global _loss_awareness
    if _loss_awareness is None:
        _loss_awareness = LossAwarenessSystem()
    return _loss_awareness
