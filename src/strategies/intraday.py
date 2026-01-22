"""
Intraday/Short-Term Trading Strategies

Designed for 15-30 minute trading intervals (HFT-lite).
These strategies focus on:
- Intraday momentum and mean reversion
- Volume-based signals
- Technical levels and breakouts
- Quick in-and-out trades

NOT designed for:
- Dividend capture
- Long-term value investing
- Multi-day holds
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import logging

from src.strategies.base import Strategy, SignalOutput

logger = logging.getLogger(__name__)


@dataclass
class IntradaySignalOutput(SignalOutput):
    """Extended signal output for intraday strategies."""
    holding_period_minutes: int = 30  # Expected hold time
    urgency: str = "normal"  # "immediate", "normal", "patient"
    entry_type: str = "market"  # "market", "limit", "aggressive_limit"


class IntradayMomentumStrategy(Strategy):
    """
    Intraday Momentum Strategy
    
    Logic: Stocks that moved significantly in the last 15-30 minutes
    tend to continue in that direction for the next 15-30 minutes.
    
    - Uses short-term price momentum (15/30/60 minute returns)
    - Filters by volume confirmation
    - Quick entry/exit signals
    """
    
    def __init__(self, name: str = "IntradayMomentum", config: Optional[Dict] = None):
        super().__init__(name, config)
        self.lookback_minutes = self.config.get('lookback_minutes', 30)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.005)  # 0.5%
        self.volume_multiplier = self.config.get('volume_multiplier', 1.5)  # 1.5x avg volume
        self.max_positions = self.config.get('max_positions', 10)
        self.position_weight = self.config.get('position_weight', 0.05)  # 5% per position
        self._required_features = ['prices', 'intraday_returns', 'volume_ratio']
    
    def generate_signals(self, features, t: datetime) -> IntradaySignalOutput:
        """Generate intraday momentum signals."""
        if not hasattr(features, 'intraday_returns') or not features.intraday_returns:
            # Fallback to daily momentum if intraday not available
            return self._fallback_to_daily(features, t)
        
        desired_weights = {}
        long_candidates = []
        short_candidates = []
        
        for symbol in features.prices.keys():
            # Get intraday momentum
            intraday_ret = features.intraday_returns.get(symbol, 0)
            volume_ratio = features.volume_ratio.get(symbol, 1.0) if hasattr(features, 'volume_ratio') else 1.0
            
            # Get sentiment if available (boost/filter based on news)
            sentiment = self.ticker_sentiments.get(symbol, 0) if self.ticker_sentiments else 0
            
            # Strong upward momentum with volume confirmation
            # Boost if positive sentiment, skip if very negative
            if intraday_ret > self.momentum_threshold and volume_ratio > self.volume_multiplier:
                if sentiment < -0.5:  # Skip if sentiment is very negative
                    continue
                # Boost score by sentiment
                score = intraday_ret * (1 + sentiment * 0.3)  # Up to 30% boost
                long_candidates.append((symbol, score, volume_ratio))
            
            # Strong downward momentum with volume confirmation (for shorting or avoiding)
            # More likely to short if negative sentiment
            elif intraday_ret < -self.momentum_threshold and volume_ratio > self.volume_multiplier:
                if sentiment > 0.5:  # Skip shorting if sentiment very positive
                    continue
                # Boost short score by negative sentiment
                score = intraday_ret * (1 - sentiment * 0.3)
                short_candidates.append((symbol, score, volume_ratio))
        
        # Sort by momentum strength
        long_candidates.sort(key=lambda x: x[1], reverse=True)
        short_candidates.sort(key=lambda x: x[1])  # Most negative first
        
        # Take top N
        for symbol, ret, vol in long_candidates[:self.max_positions]:
            # Weight by momentum strength
            weight = self.position_weight * min(2.0, abs(ret) / self.momentum_threshold)
            desired_weights[symbol] = weight
        
        explanation = {
            "strategy": self.name,
            "logic": f"Long stocks with >{self.momentum_threshold*100:.1f}% move and {self.volume_multiplier}x volume",
            "long_candidates": len(long_candidates),
            "short_candidates": len(short_candidates),
            "top_longs": [(s, f"{r*100:.2f}%") for s, r, _ in long_candidates[:5]],
            "holding_period": f"{self.lookback_minutes} minutes",
        }
        
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.003,  # 0.3% expected per trade
            risk_estimate=0.01,
            confidence=0.65 if long_candidates else 0.3,
            explanation=explanation,
            holding_period_minutes=self.lookback_minutes,
            urgency="immediate" if long_candidates else "normal",
        )
    
    def _fallback_to_daily(self, features, t: datetime) -> IntradaySignalOutput:
        """Fallback when intraday data isn't available."""
        # Use today's move from open (if available) or recent daily momentum
        desired_weights = {}
        candidates = []
        
        # Use 1-day return as proxy for intraday momentum
        if hasattr(features, 'returns_1d') and features.returns_1d:
            for symbol, ret in features.returns_1d.items():
                if abs(ret) > 0.01:  # 1% threshold for daily
                    candidates.append((symbol, ret))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for symbol, ret in candidates[:self.max_positions]:
                if ret > 0:
                    desired_weights[symbol] = self.position_weight
        
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.002,
            risk_estimate=0.015,
            confidence=0.5,
            explanation={"fallback": "Using daily returns as proxy"},
            holding_period_minutes=60,
        )


class VWAPReversionStrategy(Strategy):
    """
    VWAP Mean Reversion Strategy
    
    Logic: When price deviates significantly from VWAP, it tends to revert.
    
    - Long when price is significantly BELOW VWAP (oversold)
    - Short/avoid when price is significantly ABOVE VWAP (overbought)
    - Best used in range-bound conditions
    """
    
    def __init__(self, name: str = "VWAPReversion", config: Optional[Dict] = None):
        super().__init__(name, config)
        self.deviation_threshold = self.config.get('deviation_threshold', 0.01)  # 1%
        self.max_deviation = self.config.get('max_deviation', 0.03)  # 3% max
        self.max_positions = self.config.get('max_positions', 8)
        self.position_weight = self.config.get('position_weight', 0.04)
        self._required_features = ['prices', 'vwap']
    
    def generate_signals(self, features, t: datetime) -> IntradaySignalOutput:
        """Generate VWAP reversion signals."""
        desired_weights = {}
        oversold = []
        overbought = []
        
        if not hasattr(features, 'vwap') or not features.vwap:
            # Fallback: use 20-day MA as proxy for "fair value"
            return self._fallback_to_ma(features, t)
        
        for symbol in features.prices.keys():
            price = features.prices.get(symbol, 0)
            vwap = features.vwap.get(symbol, price)
            
            if price <= 0 or vwap <= 0:
                continue
            
            deviation = (price - vwap) / vwap
            
            # Oversold - price below VWAP
            if deviation < -self.deviation_threshold and deviation > -self.max_deviation:
                oversold.append((symbol, deviation, price, vwap))
            
            # Overbought - price above VWAP
            elif deviation > self.deviation_threshold and deviation < self.max_deviation:
                overbought.append((symbol, deviation, price, vwap))
        
        # Sort by deviation magnitude
        oversold.sort(key=lambda x: x[1])  # Most oversold first
        
        # Long oversold stocks (expecting reversion up)
        for symbol, dev, price, vwap in oversold[:self.max_positions]:
            # Weight inversely proportional to deviation (more oversold = higher weight)
            weight = self.position_weight * min(2.0, abs(dev) / self.deviation_threshold)
            desired_weights[symbol] = weight
        
        explanation = {
            "strategy": self.name,
            "logic": f"Long stocks >{self.deviation_threshold*100:.1f}% below VWAP",
            "oversold": [(s, f"{d*100:.2f}%") for s, d, _, _ in oversold[:5]],
            "overbought": [(s, f"{d*100:.2f}%") for s, d, _, _ in overbought[:5]],
            "holding_period": "15-60 minutes (until reversion)",
        }
        
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.005,  # 0.5% expected from reversion
            risk_estimate=0.008,
            confidence=0.6 if oversold else 0.3,
            explanation=explanation,
            holding_period_minutes=30,
            urgency="normal",
        )
    
    def _fallback_to_ma(self, features, t: datetime) -> IntradaySignalOutput:
        """Fallback using moving average as fair value proxy."""
        desired_weights = {}
        
        if hasattr(features, 'ma_20') and features.ma_20:
            oversold = []
            for symbol in features.prices.keys():
                price = features.prices.get(symbol, 0)
                ma = features.ma_20.get(symbol, price)
                if price > 0 and ma > 0:
                    deviation = (price - ma) / ma
                    if deviation < -0.02:  # 2% below MA
                        oversold.append((symbol, deviation))
            
            oversold.sort(key=lambda x: x[1])
            for symbol, dev in oversold[:self.max_positions]:
                desired_weights[symbol] = self.position_weight
        
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.003,
            risk_estimate=0.01,
            confidence=0.4,
            explanation={"fallback": "Using 20-day MA as fair value proxy"},
            holding_period_minutes=60,
        )


class VolumeSpikeStrategy(Strategy):
    """
    Volume Spike Detection Strategy
    
    Logic: Unusual volume often precedes price moves.
    
    - Detect stocks with abnormally high volume
    - Combine with price direction for signal
    - Often indicates institutional activity
    """
    
    def __init__(self, name: str = "VolumeSpike", config: Optional[Dict] = None):
        super().__init__(name, config)
        self.volume_threshold = self.config.get('volume_threshold', 2.0)  # 2x normal
        self.price_confirm_threshold = self.config.get('price_confirm_threshold', 0.003)  # 0.3%
        self.max_positions = self.config.get('max_positions', 8)
        self.position_weight = self.config.get('position_weight', 0.04)
        self._required_features = ['prices', 'volume_ratio']
    
    def generate_signals(self, features, t: datetime) -> IntradaySignalOutput:
        """Generate volume spike signals."""
        desired_weights = {}
        bullish_spikes = []
        bearish_spikes = []
        
        for symbol in features.prices.keys():
            volume_ratio = 1.0
            price_change = 0.0
            
            if hasattr(features, 'volume_ratio') and features.volume_ratio:
                volume_ratio = features.volume_ratio.get(symbol, 1.0)
            
            if hasattr(features, 'returns_1d') and features.returns_1d:
                price_change = features.returns_1d.get(symbol, 0)
            
            # High volume with price confirmation
            if volume_ratio >= self.volume_threshold:
                if price_change > self.price_confirm_threshold:
                    bullish_spikes.append((symbol, volume_ratio, price_change))
                elif price_change < -self.price_confirm_threshold:
                    bearish_spikes.append((symbol, volume_ratio, price_change))
        
        # Sort by volume spike magnitude
        bullish_spikes.sort(key=lambda x: x[1], reverse=True)
        
        # Long bullish volume spikes
        for symbol, vol_ratio, ret in bullish_spikes[:self.max_positions]:
            # Higher weight for stronger volume
            weight = self.position_weight * min(1.5, vol_ratio / self.volume_threshold)
            desired_weights[symbol] = weight
        
        explanation = {
            "strategy": self.name,
            "logic": f"Long stocks with {self.volume_threshold}x volume + positive price",
            "bullish_spikes": [(s, f"{v:.1f}x vol, {r*100:.2f}%") for s, v, r in bullish_spikes[:5]],
            "bearish_spikes": [(s, f"{v:.1f}x vol, {r*100:.2f}%") for s, v, r in bearish_spikes[:5]],
            "holding_period": "15-30 minutes (ride the momentum)",
        }
        
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.004,
            risk_estimate=0.012,
            confidence=0.7 if bullish_spikes else 0.3,
            explanation=explanation,
            holding_period_minutes=20,
            urgency="immediate" if bullish_spikes else "patient",
        )


class RelativeStrengthIntradayStrategy(Strategy):
    """
    Intraday Relative Strength Strategy
    
    Logic: Stocks outperforming today are likely to continue outperforming
    for the next 15-30 minutes.
    
    - Compare each stock's intraday return to SPY
    - Long the strongest relative performers
    - Avoid/short the weakest
    """
    
    def __init__(self, name: str = "RelativeStrengthIntraday", config: Optional[Dict] = None):
        super().__init__(name, config)
        self.outperformance_threshold = self.config.get('outperformance_threshold', 0.005)  # 0.5%
        self.max_positions = self.config.get('max_positions', 10)
        self.position_weight = self.config.get('position_weight', 0.05)
        self._required_features = ['prices', 'returns_1d']
    
    def generate_signals(self, features, t: datetime) -> IntradaySignalOutput:
        """Generate relative strength signals."""
        desired_weights = {}
        
        # Get benchmark (SPY) return
        spy_return = 0.0
        if hasattr(features, 'returns_1d') and features.returns_1d:
            spy_return = features.returns_1d.get('SPY', 0)
        
        outperformers = []
        underperformers = []
        
        for symbol in features.prices.keys():
            if symbol == 'SPY':
                continue
            
            stock_return = 0.0
            if hasattr(features, 'returns_1d') and features.returns_1d:
                stock_return = features.returns_1d.get(symbol, 0)
            
            relative_return = stock_return - spy_return
            
            if relative_return > self.outperformance_threshold:
                outperformers.append((symbol, relative_return, stock_return))
            elif relative_return < -self.outperformance_threshold:
                underperformers.append((symbol, relative_return, stock_return))
        
        # Sort by relative strength
        outperformers.sort(key=lambda x: x[1], reverse=True)
        
        # Long strongest relative performers
        for symbol, rel_ret, abs_ret in outperformers[:self.max_positions]:
            weight = self.position_weight * min(1.5, rel_ret / self.outperformance_threshold)
            desired_weights[symbol] = weight
        
        explanation = {
            "strategy": self.name,
            "logic": f"Long stocks outperforming SPY by >{self.outperformance_threshold*100:.1f}%",
            "spy_return": f"{spy_return*100:.2f}%",
            "outperformers": [(s, f"+{r*100:.2f}% vs SPY") for s, r, _ in outperformers[:5]],
            "underperformers": [(s, f"{r*100:.2f}% vs SPY") for s, r, _ in underperformers[:5]],
            "holding_period": "15-30 minutes",
        }
        
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.003,
            risk_estimate=0.008,
            confidence=0.6 if outperformers else 0.3,
            explanation=explanation,
            holding_period_minutes=25,
        )


class OpeningRangeBreakoutStrategy(Strategy):
    """
    Opening Range Breakout (ORB) Strategy
    
    Logic: The first 15-30 minutes establishes a range.
    Breakouts above/below this range signal direction for the day.
    
    - Track opening high/low for each stock
    - Long on breakout above opening high
    - Short/avoid on breakdown below opening low
    """
    
    def __init__(self, name: str = "OpeningRangeBreakout", config: Optional[Dict] = None):
        super().__init__(name, config)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.001)  # 0.1% above range
        self.range_minutes = self.config.get('range_minutes', 30)
        self.max_positions = self.config.get('max_positions', 8)
        self.position_weight = self.config.get('position_weight', 0.05)
        self._required_features = ['prices', 'opening_high', 'opening_low']
    
    def generate_signals(self, features, t: datetime) -> IntradaySignalOutput:
        """Generate ORB signals."""
        desired_weights = {}
        bullish_breakouts = []
        bearish_breakdowns = []
        
        # Check if we have opening range data
        if not hasattr(features, 'opening_high') or not features.opening_high:
            return self._fallback_to_daily_range(features, t)
        
        for symbol in features.prices.keys():
            price = features.prices.get(symbol, 0)
            opening_high = features.opening_high.get(symbol, price)
            opening_low = features.opening_low.get(symbol, price)
            
            if price <= 0 or opening_high <= 0:
                continue
            
            range_size = (opening_high - opening_low) / opening_low if opening_low > 0 else 0
            
            # Breakout above opening high
            if price > opening_high * (1 + self.breakout_threshold):
                breakout_pct = (price - opening_high) / opening_high
                bullish_breakouts.append((symbol, breakout_pct, range_size))
            
            # Breakdown below opening low
            elif price < opening_low * (1 - self.breakout_threshold):
                breakdown_pct = (opening_low - price) / opening_low
                bearish_breakdowns.append((symbol, breakdown_pct, range_size))
        
        # Sort by breakout magnitude
        bullish_breakouts.sort(key=lambda x: x[1], reverse=True)
        
        # Long bullish breakouts
        for symbol, breakout, range_size in bullish_breakouts[:self.max_positions]:
            weight = self.position_weight * min(1.5, breakout / 0.01 + 0.5)  # Scale by breakout size
            desired_weights[symbol] = weight
        
        explanation = {
            "strategy": self.name,
            "logic": f"Long stocks breaking above {self.range_minutes}-min opening range",
            "bullish_breakouts": [(s, f"+{b*100:.2f}%") for s, b, _ in bullish_breakouts[:5]],
            "bearish_breakdowns": [(s, f"-{b*100:.2f}%") for s, b, _ in bearish_breakdowns[:5]],
            "holding_period": "Until close or reversal",
        }
        
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.005,
            risk_estimate=0.01,
            confidence=0.65 if bullish_breakouts else 0.3,
            explanation=explanation,
            holding_period_minutes=60,
            urgency="immediate" if bullish_breakouts else "patient",
        )
    
    def _fallback_to_daily_range(self, features, t: datetime) -> IntradaySignalOutput:
        """Fallback when opening range data isn't available."""
        # Use yesterday's high/low as range
        desired_weights = {}
        
        # This would need historical high/low data
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.002,
            risk_estimate=0.012,
            confidence=0.3,
            explanation={"fallback": "Opening range data not available"},
            holding_period_minutes=60,
        )


class QuickMeanReversionStrategy(Strategy):
    """
    Quick Mean Reversion Strategy
    
    Logic: Stocks that moved too fast in one direction
    often snap back quickly (within minutes).
    
    - Detect overextended moves in last 5-15 minutes
    - Fade the move (bet on reversal)
    - Tight stops, quick profits
    """
    
    def __init__(self, name: str = "QuickMeanReversion", config: Optional[Dict] = None):
        super().__init__(name, config)
        self.overextension_threshold = self.config.get('overextension_threshold', 0.015)  # 1.5%
        self.max_positions = self.config.get('max_positions', 6)
        self.position_weight = self.config.get('position_weight', 0.03)  # Smaller size for reversal
        self._required_features = ['prices', 'returns_1d']
    
    def generate_signals(self, features, t: datetime) -> IntradaySignalOutput:
        """Generate quick mean reversion signals."""
        desired_weights = {}
        overextended_down = []
        overextended_up = []
        
        for symbol in features.prices.keys():
            recent_return = 0.0
            if hasattr(features, 'returns_1d') and features.returns_1d:
                recent_return = features.returns_1d.get(symbol, 0)
            
            # Overextended to the downside - fade with long
            if recent_return < -self.overextension_threshold:
                overextended_down.append((symbol, recent_return))
            
            # Overextended to the upside - fade with short (or avoid)
            elif recent_return > self.overextension_threshold:
                overextended_up.append((symbol, recent_return))
        
        # Sort by magnitude
        overextended_down.sort(key=lambda x: x[1])  # Most oversold first
        
        # Long the most overextended down (betting on bounce)
        for symbol, ret in overextended_down[:self.max_positions]:
            # Smaller weight for riskier reversal plays
            weight = self.position_weight * min(1.5, abs(ret) / self.overextension_threshold)
            desired_weights[symbol] = weight
        
        explanation = {
            "strategy": self.name,
            "logic": f"Fade moves >{self.overextension_threshold*100:.1f}% (mean reversion)",
            "oversold_bounce": [(s, f"{r*100:.2f}%") for s, r in overextended_down[:5]],
            "overbought_avoid": [(s, f"+{r*100:.2f}%") for s, r in overextended_up[:5]],
            "holding_period": "5-15 minutes (quick reversal)",
            "risk_note": "Tight stops required - reversal plays are risky",
        }
        
        return IntradaySignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=desired_weights,
            expected_return=0.003,  # Smaller expected return
            risk_estimate=0.015,  # Higher risk
            confidence=0.5 if overextended_down else 0.2,
            explanation=explanation,
            holding_period_minutes=10,
            urgency="immediate",
            entry_type="limit",  # Use limit orders for reversals
        )


# Factory function to create intraday strategies
def create_intraday_strategies(config: Optional[Dict] = None) -> List[Strategy]:
    """Create all intraday strategies."""
    config = config or {}
    
    return [
        IntradayMomentumStrategy(config=config.get('intraday_momentum', {})),
        VWAPReversionStrategy(config=config.get('vwap_reversion', {})),
        VolumeSpikeStrategy(config=config.get('volume_spike', {})),
        RelativeStrengthIntradayStrategy(config=config.get('relative_strength', {})),
        OpeningRangeBreakoutStrategy(config=config.get('orb', {})),
        QuickMeanReversionStrategy(config=config.get('quick_reversion', {})),
    ]
