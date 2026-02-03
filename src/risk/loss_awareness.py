"""
LOSS AWARENESS & ADAPTATION SYSTEM

This module makes the bot self-aware of its performance and forces it to:
1. Recognize when it's losing money
2. Analyze WHY it's losing
3. Adapt strategy (reduce exposure, exit losers, go to cash)
4. Learn from mistakes

When the bot is losing, it should:
- REDUCE position sizes
- EXIT losing positions faster
- GO TO CASH if strategy isn't working
- QUESTION its assumptions about the market
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PerformanceState(Enum):
    """Current performance state of the portfolio."""
    EXCELLENT = "excellent"      # > 2% up today
    GOOD = "good"                # 0.5% to 2% up
    NEUTRAL = "neutral"          # -0.5% to 0.5%
    CONCERNING = "concerning"    # -0.5% to -1.5%
    BAD = "bad"                  # -1.5% to -3%
    CRITICAL = "critical"        # > -3% (circuit breaker territory)


@dataclass
class LossAnalysis:
    """Analysis of why the portfolio is losing."""
    total_pnl: float
    total_pnl_pct: float
    state: PerformanceState
    losing_positions: List[Dict]
    winning_positions: List[Dict]
    biggest_loser: Optional[Dict]
    biggest_winner: Optional[Dict]
    
    # Reasons for losses
    reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Exposure adjustments
    recommended_exposure: float = 0.8  # Default 80%
    should_exit_losers: bool = False
    should_go_to_cash: bool = False
    positions_to_exit: List[str] = field(default_factory=list)


class LossAwarenessSystem:
    """
    Makes the trading bot self-aware of its performance.
    
    When losing:
    - Reduces position sizes
    - Exits losers faster
    - Goes to cash if necessary
    - Analyzes what went wrong
    """
    
    def __init__(self):
        # Performance thresholds
        self.thresholds = {
            "excellent": 0.02,      # > 2%
            "good": 0.005,          # > 0.5%
            "neutral": -0.005,      # > -0.5%
            "concerning": -0.015,   # > -1.5%
            "bad": -0.03,           # > -3%
        }
        
        # Exposure adjustments by state
        self.exposure_adjustments = {
            PerformanceState.EXCELLENT: 1.0,    # Full exposure
            PerformanceState.GOOD: 0.9,         # 90% exposure
            PerformanceState.NEUTRAL: 0.8,      # 80% exposure
            PerformanceState.CONCERNING: 0.5,   # 50% exposure - REDUCE
            PerformanceState.BAD: 0.25,         # 25% exposure - MINIMIZE
            PerformanceState.CRITICAL: 0.0,     # 0% - GO TO CASH
        }
        
        # Loss tolerance for individual positions
        self.position_loss_threshold = -0.05  # Exit at -5% loss
        self.position_loss_fast_exit = -0.03  # Consider exit at -3% if overall losing
        
        # Tracking
        self.analysis_history: List[LossAnalysis] = []
        self.last_analysis: Optional[LossAnalysis] = None
        self.consecutive_losses = 0
        
    def get_performance_state(self, pnl_pct: float) -> PerformanceState:
        """Determine current performance state."""
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
        Analyze why the portfolio is losing and what to do about it.
        
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
        
        # Analyze reasons for losses
        self._analyze_reasons(analysis, market_data)
        
        # Generate recommendations
        self._generate_recommendations(analysis)
        
        # Store analysis
        self.last_analysis = analysis
        self.analysis_history.append(analysis)
        
        # Track consecutive losses
        if state in [PerformanceState.CONCERNING, PerformanceState.BAD, PerformanceState.CRITICAL]:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        return analysis
    
    def _analyze_reasons(self, analysis: LossAnalysis, market_data: Optional[Dict]):
        """Identify reasons for the losses."""
        reasons = []
        
        # Check if most positions are losing
        total_positions = len(analysis.losing_positions) + len(analysis.winning_positions)
        if total_positions > 0:
            loser_ratio = len(analysis.losing_positions) / total_positions
            if loser_ratio > 0.7:
                reasons.append(f"Most positions ({loser_ratio:.0%}) are in the red - possible market-wide downturn")
            elif loser_ratio > 0.5:
                reasons.append(f"More than half ({loser_ratio:.0%}) of positions are losing")
        
        # Check for concentrated losses
        if analysis.biggest_loser:
            loser_pnl = analysis.biggest_loser.get('unrealized_pl', 0)
            if analysis.total_pnl != 0 and abs(loser_pnl / analysis.total_pnl) > 0.5:
                sym = analysis.biggest_loser.get('symbol', '?')
                reasons.append(f"Concentrated loss: {sym} accounts for >50% of total losses")
        
        # Check market context
        if market_data:
            spy_change = market_data.get('spy_change', 0)
            vix = market_data.get('vix', 0)
            
            if spy_change < -0.01:
                reasons.append(f"Market is down (SPY {spy_change*100:+.1f}%) - broad selloff")
            if vix > 25:
                reasons.append(f"High volatility (VIX {vix:.1f}) - unstable market conditions")
        
        # Check for individual position issues
        for pos in analysis.losing_positions:
            pnl_pct = pos.get('unrealized_plpc', 0)
            sym = pos.get('symbol', '?')
            if pnl_pct < -0.05:
                reasons.append(f"{sym} down {pnl_pct*100:.1f}% - significant individual loss")
        
        # If no specific reasons found
        if not reasons:
            reasons.append("General market fluctuation - no specific catalyst identified")
        
        analysis.reasons = reasons
    
    def _generate_recommendations(self, analysis: LossAnalysis):
        """Generate actionable recommendations based on losses."""
        recommendations = []
        positions_to_exit = []
        
        state = analysis.state
        
        # State-based recommendations
        if state == PerformanceState.CRITICAL:
            recommendations.append("🚨 CRITICAL: Daily loss limit approaching - HALT NEW TRADES")
            recommendations.append("⚠️ Close all positions and go to CASH")
            analysis.should_go_to_cash = True
            analysis.recommended_exposure = 0.0
            # Exit all losing positions
            for pos in analysis.losing_positions:
                positions_to_exit.append(pos.get('symbol'))
                
        elif state == PerformanceState.BAD:
            recommendations.append("📉 BAD: Significant losses - reduce exposure to 25%")
            recommendations.append("Exit positions with losses > 3%")
            analysis.recommended_exposure = 0.25
            analysis.should_exit_losers = True
            # Exit big losers
            for pos in analysis.losing_positions:
                if pos.get('unrealized_plpc', 0) < self.position_loss_fast_exit:
                    positions_to_exit.append(pos.get('symbol'))
                    
        elif state == PerformanceState.CONCERNING:
            recommendations.append("⚠️ CONCERNING: Reduce exposure to 50%")
            recommendations.append("Consider exiting worst performers")
            analysis.recommended_exposure = 0.50
            # Exit only the worst loser if it's bad
            if analysis.biggest_loser:
                if analysis.biggest_loser.get('unrealized_plpc', 0) < self.position_loss_threshold:
                    positions_to_exit.append(analysis.biggest_loser.get('symbol'))
                    
        elif state == PerformanceState.NEUTRAL:
            recommendations.append("📊 NEUTRAL: Maintain current exposure (80%)")
            analysis.recommended_exposure = 0.80
            
        elif state == PerformanceState.GOOD:
            recommendations.append("📈 GOOD: Performance is positive - maintain strategy")
            analysis.recommended_exposure = 0.90
            
        elif state == PerformanceState.EXCELLENT:
            recommendations.append("🚀 EXCELLENT: Strategy is working well")
            recommendations.append("Consider taking profits on biggest winners")
            analysis.recommended_exposure = 1.0
        
        # Consecutive losses recommendation
        if self.consecutive_losses >= 3:
            recommendations.append(f"⚠️ {self.consecutive_losses} consecutive losing periods - STRATEGY MAY NOT BE WORKING")
            recommendations.append("Consider going to 100% cash until market conditions improve")
            analysis.should_go_to_cash = True
            analysis.recommended_exposure = min(analysis.recommended_exposure, 0.25)
        
        analysis.recommendations = recommendations
        analysis.positions_to_exit = positions_to_exit
    
    def get_adjusted_exposure(self, base_exposure: float = 0.8) -> float:
        """
        Get exposure adjusted for current performance.
        
        If losing, reduce exposure. If winning, maintain or increase.
        """
        if not self.last_analysis:
            return base_exposure
        
        adjustment = self.exposure_adjustments.get(
            self.last_analysis.state,
            0.8
        )
        
        # Further reduce if consecutive losses
        if self.consecutive_losses >= 2:
            adjustment *= 0.5
        
        return min(base_exposure, adjustment)
    
    def get_analysis_summary(self) -> str:
        """Get a human-readable summary of current analysis."""
        if not self.last_analysis:
            return "No analysis available yet"
        
        a = self.last_analysis
        lines = [
            "=" * 60,
            "📊 LOSS AWARENESS ANALYSIS",
            "=" * 60,
            f"State: {a.state.value.upper()}",
            f"Total P&L: ${a.total_pnl:+,.2f} ({a.total_pnl_pct*100:+.2f}%)",
            f"Winning positions: {len(a.winning_positions)}",
            f"Losing positions: {len(a.losing_positions)}",
            "",
            "REASONS FOR PERFORMANCE:",
        ]
        
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
            return True, "No analysis yet - proceed with caution"
        
        a = self.last_analysis
        
        if a.should_go_to_cash:
            return False, f"Performance is {a.state.value} - GO TO CASH"
        
        if a.state == PerformanceState.CRITICAL:
            return False, "Critical losses - trading halted"
        
        if self.consecutive_losses >= 5:
            return False, f"{self.consecutive_losses} consecutive losses - strategy not working"
        
        return True, f"OK to trade at {a.recommended_exposure*100:.0f}% exposure"


# Singleton instance
_loss_awareness: Optional[LossAwarenessSystem] = None


def get_loss_awareness() -> LossAwarenessSystem:
    """Get the singleton loss awareness system."""
    global _loss_awareness
    if _loss_awareness is None:
        _loss_awareness = LossAwarenessSystem()
    return _loss_awareness
