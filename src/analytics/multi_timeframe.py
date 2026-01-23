"""
Multi-Timeframe Signal Fusion

Combines signals across different timeframes to generate higher-conviction trades:
- Daily trend direction
- 4-hour momentum
- 1-hour structure
- 15-min entry timing
- 5-min micro-confirmation

The principle: Higher timeframe provides direction, lower timeframe provides entry.
Agreement across timeframes = highest conviction.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class TimeframeTrend(Enum):
    """Trend classification for a timeframe."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


class EntrySignal(Enum):
    """Entry signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TimeframeAnalysis:
    """Analysis for a single timeframe."""
    timeframe: str  # "1D", "4H", "1H", "15Min", "5Min"
    symbol: str
    
    # Trend
    trend: TimeframeTrend = TimeframeTrend.NEUTRAL
    trend_strength: float = 0.0
    
    # Momentum
    momentum: float = 0.0  # -1 to 1
    momentum_divergence: bool = False  # Price vs momentum diverging
    
    # Structure
    above_ema_20: bool = False
    above_ema_50: bool = False
    ema_crossover: Optional[str] = None  # "bullish", "bearish", None
    
    # Key levels
    near_support: bool = False
    near_resistance: bool = False
    breakout: Optional[str] = None  # "upward", "downward", None
    
    # Volume
    volume_confirmation: bool = False
    volume_ratio: float = 1.0  # vs average
    
    # Volatility
    volatility_percentile: float = 50.0  # 0-100
    
    timestamp: Optional[datetime] = None


@dataclass
class FusedSignal:
    """
    Fused signal from multiple timeframes.
    Higher agreement = higher conviction.
    """
    symbol: str
    
    # Direction
    direction: EntrySignal = EntrySignal.HOLD
    conviction: float = 0.5  # 0-1
    
    # Timeframe alignment score (how many timeframes agree)
    alignment_score: float = 0.0  # 0-1
    aligned_timeframes: List[str] = field(default_factory=list)
    conflicting_timeframes: List[str] = field(default_factory=list)
    
    # Entry quality
    entry_quality: float = 0.5  # 0-1 (based on lower timeframe setup)
    
    # Risk parameters
    suggested_stop_pct: float = 0.02  # 2% default
    suggested_target_pct: float = 0.04  # 4% default
    risk_reward: float = 2.0
    
    # Components
    daily_trend: TimeframeTrend = TimeframeTrend.NEUTRAL
    hourly_momentum: float = 0.0
    intraday_entry: bool = False
    
    # Metadata
    rationale: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "conviction": self.conviction,
            "alignment_score": self.alignment_score,
            "aligned_timeframes": self.aligned_timeframes,
            "entry_quality": self.entry_quality,
            "suggested_stop_pct": self.suggested_stop_pct,
            "suggested_target_pct": self.suggested_target_pct,
            "risk_reward": self.risk_reward,
            "rationale": self.rationale,
        }


class MultiTimeframeFusion:
    """
    Combines signals across multiple timeframes.
    
    Strategy:
    1. Daily/Weekly: Determine overall trend (trade WITH this trend)
    2. 4H/1H: Identify momentum and pullbacks
    3. 15Min/5Min: Find precise entry points
    
    Only trade when:
    - Higher timeframe trend is clear
    - Lower timeframe shows entry setup in direction of trend
    - Volume confirms the move
    """
    
    TIMEFRAMES = ["1D", "4H", "1H", "15Min", "5Min"]
    
    def __init__(self):
        # Cache for timeframe data
        self.timeframe_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        # Cache for analysis results
        self.analysis_cache: Dict[str, Dict[str, TimeframeAnalysis]] = {}
        # Generated signals
        self.fused_signals: Dict[str, FusedSignal] = {}
    
    def update_data(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ):
        """
        Update price data for a symbol/timeframe.
        DataFrame should have: open, high, low, close, volume.
        """
        if symbol not in self.timeframe_data:
            self.timeframe_data[symbol] = {}
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            logger.warning(f"Missing columns for {symbol}/{timeframe}")
            return
        
        self.timeframe_data[symbol][timeframe] = df.copy()
    
    def analyze_timeframe(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[TimeframeAnalysis]:
        """Analyze a single timeframe for a symbol."""
        if symbol not in self.timeframe_data:
            return None
        if timeframe not in self.timeframe_data[symbol]:
            return None
        
        df = self.timeframe_data[symbol][timeframe]
        
        if len(df) < 50:
            return None
        
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Calculate indicators
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            
            # Current values
            current_close = close.iloc[-1]
            current_ema20 = ema_20.iloc[-1]
            current_ema50 = ema_50.iloc[-1]
            current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
            
            # Trend determination
            returns_20 = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
            ema_slope = (ema_20.iloc[-1] / ema_20.iloc[-5] - 1) if len(ema_20) >= 5 else 0
            
            if returns_20 > 0.05 and ema_slope > 0.01:
                trend = TimeframeTrend.STRONG_UP
            elif returns_20 > 0.02 or ema_slope > 0.005:
                trend = TimeframeTrend.UP
            elif returns_20 < -0.05 and ema_slope < -0.01:
                trend = TimeframeTrend.STRONG_DOWN
            elif returns_20 < -0.02 or ema_slope < -0.005:
                trend = TimeframeTrend.DOWN
            else:
                trend = TimeframeTrend.NEUTRAL
            
            trend_strength = min(1.0, abs(returns_20) / 0.10)
            
            # Momentum (RSI-based)
            momentum = (current_rsi - 50) / 50  # -1 to 1
            
            # Check for divergence (price up, RSI down or vice versa)
            price_direction = 1 if returns_20 > 0 else -1
            rsi_direction = 1 if rsi.iloc[-1] > rsi.iloc[-20] else -1 if len(rsi) >= 20 else 0
            momentum_divergence = price_direction != rsi_direction
            
            # EMA crossover
            ema_crossover = None
            if len(ema_20) >= 5:
                if ema_20.iloc[-1] > ema_50.iloc[-1] and ema_20.iloc[-5] <= ema_50.iloc[-5]:
                    ema_crossover = "bullish"
                elif ema_20.iloc[-1] < ema_50.iloc[-1] and ema_20.iloc[-5] >= ema_50.iloc[-5]:
                    ema_crossover = "bearish"
            
            # Support/Resistance (simple pivot-based)
            recent_low = low.tail(20).min()
            recent_high = high.tail(20).max()
            near_support = (current_close - recent_low) / (recent_high - recent_low + 0.001) < 0.2
            near_resistance = (current_close - recent_low) / (recent_high - recent_low + 0.001) > 0.8
            
            # Breakout detection
            breakout = None
            if current_close > recent_high * 0.99:
                breakout = "upward"
            elif current_close < recent_low * 1.01:
                breakout = "downward"
            
            # Volume
            avg_volume = volume.tail(20).mean()
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volume_confirmation = volume_ratio > 1.5  # Above average volume
            
            # Volatility (ATR percentile)
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)
            atr = tr.rolling(14).mean()
            current_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0
            atr_pct = current_atr / current_close * 100 if current_close > 0 else 0
            
            # Percentile of ATR in history
            if len(atr.dropna()) > 20:
                vol_percentile = (atr.dropna() < current_atr).mean() * 100
            else:
                vol_percentile = 50.0
            
            analysis = TimeframeAnalysis(
                timeframe=timeframe,
                symbol=symbol,
                trend=trend,
                trend_strength=trend_strength,
                momentum=momentum,
                momentum_divergence=momentum_divergence,
                above_ema_20=current_close > current_ema20,
                above_ema_50=current_close > current_ema50,
                ema_crossover=ema_crossover,
                near_support=near_support,
                near_resistance=near_resistance,
                breakout=breakout,
                volume_confirmation=volume_confirmation,
                volume_ratio=volume_ratio,
                volatility_percentile=vol_percentile,
                timestamp=datetime.now(),
            )
            
            # Cache result
            if symbol not in self.analysis_cache:
                self.analysis_cache[symbol] = {}
            self.analysis_cache[symbol][timeframe] = analysis
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing {symbol}/{timeframe}: {e}")
            return None
    
    def generate_fused_signal(self, symbol: str) -> Optional[FusedSignal]:
        """
        Generate a fused signal for a symbol by combining all timeframes.
        """
        # Analyze all available timeframes
        analyses = {}
        for tf in self.TIMEFRAMES:
            analysis = self.analyze_timeframe(symbol, tf)
            if analysis:
                analyses[tf] = analysis
        
        if not analyses:
            return None
        
        # Get the highest timeframe analysis (trend anchor)
        anchor_tf = None
        for tf in ["1D", "4H", "1H", "15Min", "5Min"]:
            if tf in analyses:
                anchor_tf = tf
                break
        
        if not anchor_tf:
            return None
        
        anchor = analyses[anchor_tf]
        
        # Determine overall direction from highest timeframe
        if anchor.trend in [TimeframeTrend.STRONG_UP, TimeframeTrend.UP]:
            direction = EntrySignal.BUY
        elif anchor.trend in [TimeframeTrend.STRONG_DOWN, TimeframeTrend.DOWN]:
            direction = EntrySignal.SELL
        else:
            direction = EntrySignal.HOLD
        
        # Check alignment across timeframes
        aligned = []
        conflicting = []
        
        for tf, analysis in analyses.items():
            if direction == EntrySignal.BUY:
                if analysis.trend in [TimeframeTrend.STRONG_UP, TimeframeTrend.UP]:
                    aligned.append(tf)
                elif analysis.trend in [TimeframeTrend.STRONG_DOWN, TimeframeTrend.DOWN]:
                    conflicting.append(tf)
            elif direction == EntrySignal.SELL:
                if analysis.trend in [TimeframeTrend.STRONG_DOWN, TimeframeTrend.DOWN]:
                    aligned.append(tf)
                elif analysis.trend in [TimeframeTrend.STRONG_UP, TimeframeTrend.UP]:
                    conflicting.append(tf)
        
        # Alignment score
        alignment_score = len(aligned) / len(analyses) if analyses else 0
        
        # Entry quality (from lowest timeframe)
        entry_tf = None
        for tf in reversed(self.TIMEFRAMES):
            if tf in analyses:
                entry_tf = tf
                break
        
        entry_quality = 0.5
        if entry_tf and entry_tf in analyses:
            entry_analysis = analyses[entry_tf]
            
            # Good entry conditions
            if direction == EntrySignal.BUY:
                if entry_analysis.near_support:
                    entry_quality += 0.2
                if entry_analysis.volume_confirmation:
                    entry_quality += 0.15
                if entry_analysis.ema_crossover == "bullish":
                    entry_quality += 0.15
                if entry_analysis.breakout == "upward":
                    entry_quality += 0.1
            elif direction == EntrySignal.SELL:
                if entry_analysis.near_resistance:
                    entry_quality += 0.2
                if entry_analysis.volume_confirmation:
                    entry_quality += 0.15
                if entry_analysis.ema_crossover == "bearish":
                    entry_quality += 0.15
                if entry_analysis.breakout == "downward":
                    entry_quality += 0.1
            
            entry_quality = min(1.0, entry_quality)
        
        # Calculate conviction
        conviction = (
            alignment_score * 0.4 +           # Timeframe alignment
            anchor.trend_strength * 0.3 +     # Trend strength
            entry_quality * 0.2 +             # Entry setup quality
            (0.1 if len(conflicting) == 0 else 0)  # No conflicts bonus
        )
        
        # Upgrade to STRONG if high conviction
        if conviction > 0.75:
            if direction == EntrySignal.BUY:
                direction = EntrySignal.STRONG_BUY
            elif direction == EntrySignal.SELL:
                direction = EntrySignal.STRONG_SELL
        
        # Downgrade if conflicts
        if len(conflicting) >= 2:
            if direction in [EntrySignal.STRONG_BUY, EntrySignal.STRONG_SELL]:
                direction = EntrySignal.BUY if direction == EntrySignal.STRONG_BUY else EntrySignal.SELL
            conviction *= 0.7
        
        # Calculate stop and target based on volatility
        volatility_pct = anchor.volatility_percentile / 100
        base_stop = 0.02  # 2% default
        
        # Higher volatility = wider stop
        stop_pct = base_stop * (1 + volatility_pct * 0.5)
        
        # Target at 2:1 risk-reward minimum
        target_pct = stop_pct * 2.5
        
        # Build rationale
        rationale_parts = [
            f"{anchor_tf} trend: {anchor.trend.value}",
            f"Aligned: {aligned}",
        ]
        if conflicting:
            rationale_parts.append(f"Conflicts: {conflicting}")
        if entry_tf:
            entry_a = analyses[entry_tf]
            if entry_a.near_support:
                rationale_parts.append("Near support")
            if entry_a.near_resistance:
                rationale_parts.append("Near resistance")
            if entry_a.volume_confirmation:
                rationale_parts.append("Volume confirmed")
            if entry_a.breakout:
                rationale_parts.append(f"Breakout: {entry_a.breakout}")
        
        signal = FusedSignal(
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            alignment_score=alignment_score,
            aligned_timeframes=aligned,
            conflicting_timeframes=conflicting,
            entry_quality=entry_quality,
            suggested_stop_pct=stop_pct,
            suggested_target_pct=target_pct,
            risk_reward=target_pct / stop_pct if stop_pct > 0 else 2.0,
            daily_trend=anchor.trend,
            hourly_momentum=analyses.get("1H", analyses.get("15Min", anchor)).momentum,
            intraday_entry=entry_quality > 0.6,
            rationale="; ".join(rationale_parts),
            timestamp=datetime.now(),
        )
        
        self.fused_signals[symbol] = signal
        
        return signal
    
    def get_all_signals(self) -> Dict[str, FusedSignal]:
        """Get all generated fused signals."""
        return self.fused_signals
    
    def get_high_conviction_signals(
        self,
        min_conviction: float = 0.7,
        direction_filter: Optional[EntrySignal] = None,
    ) -> List[FusedSignal]:
        """Get signals above conviction threshold."""
        signals = [
            sig for sig in self.fused_signals.values()
            if sig.conviction >= min_conviction
        ]
        
        if direction_filter:
            signals = [sig for sig in signals if sig.direction == direction_filter]
        
        return sorted(signals, key=lambda x: -x.conviction)
    
    def get_summary(self) -> Dict:
        """Get summary of multi-timeframe analysis."""
        signals = list(self.fused_signals.values())
        
        if not signals:
            return {"total_signals": 0}
        
        return {
            "total_signals": len(signals),
            "high_conviction": len([s for s in signals if s.conviction > 0.7]),
            "buy_signals": len([s for s in signals if s.direction in [EntrySignal.BUY, EntrySignal.STRONG_BUY]]),
            "sell_signals": len([s for s in signals if s.direction in [EntrySignal.SELL, EntrySignal.STRONG_SELL]]),
            "avg_conviction": np.mean([s.conviction for s in signals]),
            "avg_alignment": np.mean([s.alignment_score for s in signals]),
            "strongest_buys": [
                {"symbol": s.symbol, "conviction": s.conviction}
                for s in sorted(
                    [sig for sig in signals if sig.direction in [EntrySignal.BUY, EntrySignal.STRONG_BUY]],
                    key=lambda x: -x.conviction
                )[:5]
            ],
            "strongest_sells": [
                {"symbol": s.symbol, "conviction": s.conviction}
                for s in sorted(
                    [sig for sig in signals if sig.direction in [EntrySignal.SELL, EntrySignal.STRONG_SELL]],
                    key=lambda x: -x.conviction
                )[:5]
            ],
        }


# Singleton instance
_multi_tf_fusion: Optional[MultiTimeframeFusion] = None


def get_multi_timeframe_fusion() -> MultiTimeframeFusion:
    """Get singleton instance of multi-timeframe fusion engine."""
    global _multi_tf_fusion
    if _multi_tf_fusion is None:
        _multi_tf_fusion = MultiTimeframeFusion()
    return _multi_tf_fusion
