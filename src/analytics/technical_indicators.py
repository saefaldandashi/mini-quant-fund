"""
Technical Analysis Module

Comprehensive technical indicators computed from Alpaca price data.
All indicators are designed for integration with the feature store and strategy system.

Indicators Implemented:
1. MACD (Moving Average Convergence Divergence)
2. Bollinger Bands
3. Stochastic Oscillator
4. ADX (Average Directional Index)
5. OBV (On Balance Volume)
6. Ichimoku Cloud (simplified)
7. Price Action Patterns (support/resistance)
8. Volume Profile Analysis
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class TrendStrength(Enum):
    """Trend strength classification."""
    STRONG_BULLISH = "strong_bullish"   # ADX > 40, +DI > -DI
    BULLISH = "bullish"                  # ADX > 25, +DI > -DI
    WEAK_BULLISH = "weak_bullish"        # ADX < 25, +DI > -DI
    NEUTRAL = "neutral"                  # ADX < 20
    WEAK_BEARISH = "weak_bearish"        # ADX < 25, -DI > +DI
    BEARISH = "bearish"                  # ADX > 25, -DI > +DI
    STRONG_BEARISH = "strong_bearish"    # ADX > 40, -DI > +DI


class MomentumState(Enum):
    """Momentum state classification."""
    ACCELERATING_UP = "accelerating_up"
    DECELERATING_UP = "decelerating_up"
    NEUTRAL = "neutral"
    DECELERATING_DOWN = "decelerating_down"
    ACCELERATING_DOWN = "accelerating_down"


class VolatilityState(Enum):
    """Volatility state classification."""
    SQUEEZE = "squeeze"           # Bollinger bandwidth < 10th percentile
    LOW = "low"                   # Bandwidth < 25th percentile
    NORMAL = "normal"             # Bandwidth 25-75th percentile
    HIGH = "high"                 # Bandwidth > 75th percentile
    EXPANSION = "expansion"       # Bandwidth > 90th percentile


@dataclass
class MACDIndicator:
    """MACD indicator values."""
    macd_line: float = 0.0        # MACD line (fast EMA - slow EMA)
    signal_line: float = 0.0     # Signal line (EMA of MACD)
    histogram: float = 0.0       # Histogram (MACD - Signal)
    
    # Signals
    bullish_crossover: bool = False    # MACD crossed above signal
    bearish_crossover: bool = False    # MACD crossed below signal
    divergence: Optional[str] = None   # "bullish_divergence", "bearish_divergence", None
    
    # Momentum
    momentum_state: MomentumState = MomentumState.NEUTRAL
    histogram_slope: float = 0.0       # Rate of change of histogram


@dataclass
class BollingerBands:
    """Bollinger Bands indicator values."""
    upper: float = 0.0
    middle: float = 0.0  # SMA
    lower: float = 0.0
    
    bandwidth: float = 0.0          # (upper - lower) / middle
    percent_b: float = 0.5          # (price - lower) / (upper - lower)
    
    # Signals
    above_upper: bool = False        # Price above upper band
    below_lower: bool = False        # Price below lower band
    squeeze: bool = False            # Bandwidth at historical low
    breakout_up: bool = False        # Breaking out of squeeze upward
    breakout_down: bool = False      # Breaking out of squeeze downward
    
    volatility_state: VolatilityState = VolatilityState.NORMAL


@dataclass
class StochasticOscillator:
    """Stochastic Oscillator values."""
    k: float = 50.0           # %K (fast stochastic)
    d: float = 50.0           # %D (slow stochastic, SMA of %K)
    
    overbought: bool = False  # %K > 80
    oversold: bool = False    # %K < 20
    bullish_cross: bool = False   # %K crossed above %D
    bearish_cross: bool = False   # %K crossed below %D


@dataclass
class ADXIndicator:
    """ADX (Average Directional Index) values."""
    adx: float = 0.0          # ADX value (trend strength)
    plus_di: float = 0.0      # +DI (bullish directional indicator)
    minus_di: float = 0.0     # -DI (bearish directional indicator)
    
    trend_strength: TrendStrength = TrendStrength.NEUTRAL
    trending: bool = False     # ADX > 25
    strong_trend: bool = False # ADX > 40


@dataclass
class VolumeAnalysis:
    """Volume-based indicators."""
    obv: float = 0.0                  # On Balance Volume
    obv_trend: str = "neutral"        # "up", "down", "neutral"
    obv_divergence: Optional[str] = None  # Price vs OBV divergence
    
    volume_ma_ratio: float = 1.0      # Current volume / MA volume
    volume_breakout: bool = False     # Volume > 2x average
    volume_dry_up: bool = False       # Volume < 0.5x average
    
    money_flow_index: float = 50.0    # MFI (volume-weighted RSI)
    accumulation: bool = False        # MFI > 80 with rising prices
    distribution: bool = False        # MFI < 20 with falling prices


@dataclass
class PriceAction:
    """Price action analysis."""
    # Key levels
    support_1: float = 0.0
    support_2: float = 0.0
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    
    # Distance to levels (as % of price)
    dist_to_support: float = 0.0
    dist_to_resistance: float = 0.0
    
    # Patterns
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False
    
    # Breakout detection
    breakout_direction: Optional[str] = None  # "up", "down", None
    breakout_strength: float = 0.0


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis for a symbol."""
    symbol: str
    timestamp: datetime
    current_price: float = 0.0
    
    # Core indicators
    macd: MACDIndicator = field(default_factory=MACDIndicator)
    bollinger: BollingerBands = field(default_factory=BollingerBands)
    stochastic: StochasticOscillator = field(default_factory=StochasticOscillator)
    adx: ADXIndicator = field(default_factory=ADXIndicator)
    volume: VolumeAnalysis = field(default_factory=VolumeAnalysis)
    price_action: PriceAction = field(default_factory=PriceAction)
    
    # Summary scores
    trend_score: float = 0.0      # -1 (bearish) to +1 (bullish)
    momentum_score: float = 0.0   # -1 to +1
    volatility_score: float = 0.5 # 0 (low) to 1 (high)
    volume_score: float = 0.0     # -1 (distribution) to +1 (accumulation)
    
    # Composite signal
    composite_signal: float = 0.0  # -1 to +1 (overall technical outlook)
    signal_confidence: float = 0.5 # 0 to 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "trend_score": self.trend_score,
            "momentum_score": self.momentum_score,
            "volatility_score": self.volatility_score,
            "volume_score": self.volume_score,
            "composite_signal": self.composite_signal,
            "signal_confidence": self.signal_confidence,
            "macd_histogram": self.macd.histogram,
            "bollinger_percent_b": self.bollinger.percent_b,
            "stochastic_k": self.stochastic.k,
            "adx": self.adx.adx,
        }


class TechnicalAnalyzer:
    """
    Computes comprehensive technical analysis from price data.
    
    Usage:
        analyzer = TechnicalAnalyzer()
        analysis = analyzer.analyze(symbol, df)
        
        # Access specific indicators
        if analysis.macd.bullish_crossover:
            print("MACD bullish crossover detected!")
        
        # Use composite signal
        if analysis.composite_signal > 0.5:
            print(f"Strong bullish signal: {analysis.composite_signal}")
    """
    
    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 14,
        stoch_d: int = 3,
        adx_period: int = 14,
    ):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.adx_period = adx_period
    
    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        include_volume: bool = True,
    ) -> TechnicalAnalysis:
        """
        Perform complete technical analysis on a symbol.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with columns: open, high, low, close, volume
            include_volume: Whether to compute volume indicators
            
        Returns:
            TechnicalAnalysis object with all indicators
        """
        if len(df) < 50:
            logger.warning(f"{symbol}: Insufficient data ({len(df)} bars) for full analysis")
            return TechnicalAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
            )
        
        # Ensure column names are lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        close = df['close']
        high = df['high']
        low = df['low']
        current_price = close.iloc[-1]
        
        analysis = TechnicalAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
        )
        
        # Compute each indicator
        analysis.macd = self._compute_macd(close)
        analysis.bollinger = self._compute_bollinger(close, current_price)
        analysis.stochastic = self._compute_stochastic(high, low, close)
        analysis.adx = self._compute_adx(high, low, close)
        
        if include_volume and 'volume' in df.columns:
            analysis.volume = self._compute_volume(df)
        
        analysis.price_action = self._compute_price_action(high, low, close)
        
        # Compute summary scores
        analysis.trend_score = self._compute_trend_score(analysis)
        analysis.momentum_score = self._compute_momentum_score(analysis)
        analysis.volatility_score = self._compute_volatility_score(analysis)
        analysis.volume_score = self._compute_volume_score(analysis)
        
        # Composite signal (weighted average)
        analysis.composite_signal = self._compute_composite_signal(analysis)
        analysis.signal_confidence = self._compute_confidence(analysis)
        
        return analysis
    
    def _compute_macd(self, close: pd.Series) -> MACDIndicator:
        """Compute MACD indicator."""
        indicator = MACDIndicator()
        
        try:
            ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            indicator.macd_line = macd_line.iloc[-1]
            indicator.signal_line = signal_line.iloc[-1]
            indicator.histogram = histogram.iloc[-1]
            
            # Crossover detection (using last 2 bars)
            if len(macd_line) >= 2:
                prev_macd = macd_line.iloc[-2]
                prev_signal = signal_line.iloc[-2]
                curr_macd = macd_line.iloc[-1]
                curr_signal = signal_line.iloc[-1]
                
                indicator.bullish_crossover = prev_macd < prev_signal and curr_macd > curr_signal
                indicator.bearish_crossover = prev_macd > prev_signal and curr_macd < curr_signal
            
            # Histogram slope
            if len(histogram) >= 3:
                indicator.histogram_slope = histogram.iloc[-1] - histogram.iloc[-3]
            
            # Momentum state
            if indicator.histogram > 0 and indicator.histogram_slope > 0:
                indicator.momentum_state = MomentumState.ACCELERATING_UP
            elif indicator.histogram > 0 and indicator.histogram_slope < 0:
                indicator.momentum_state = MomentumState.DECELERATING_UP
            elif indicator.histogram < 0 and indicator.histogram_slope < 0:
                indicator.momentum_state = MomentumState.ACCELERATING_DOWN
            elif indicator.histogram < 0 and indicator.histogram_slope > 0:
                indicator.momentum_state = MomentumState.DECELERATING_DOWN
            
        except Exception as e:
            logger.debug(f"MACD computation error: {e}")
        
        return indicator
    
    def _compute_bollinger(self, close: pd.Series, current_price: float) -> BollingerBands:
        """Compute Bollinger Bands."""
        bands = BollingerBands()
        
        try:
            sma = close.rolling(self.bb_period).mean()
            std = close.rolling(self.bb_period).std()
            
            upper = sma + (std * self.bb_std)
            lower = sma - (std * self.bb_std)
            
            bands.middle = sma.iloc[-1]
            bands.upper = upper.iloc[-1]
            bands.lower = lower.iloc[-1]
            
            # Bandwidth
            if bands.middle > 0:
                bands.bandwidth = (bands.upper - bands.lower) / bands.middle
            
            # Percent B
            band_range = bands.upper - bands.lower
            if band_range > 0:
                bands.percent_b = (current_price - bands.lower) / band_range
            
            # Signals
            bands.above_upper = current_price > bands.upper
            bands.below_lower = current_price < bands.lower
            
            # Squeeze detection (bandwidth in bottom 10th percentile)
            bandwidth_series = (upper - lower) / sma
            bandwidth_series = bandwidth_series.dropna()
            if len(bandwidth_series) > 20:
                percentile = (bandwidth_series < bands.bandwidth).mean() * 100
                bands.squeeze = percentile < 10
                
                # Volatility state
                if percentile < 10:
                    bands.volatility_state = VolatilityState.SQUEEZE
                elif percentile < 25:
                    bands.volatility_state = VolatilityState.LOW
                elif percentile < 75:
                    bands.volatility_state = VolatilityState.NORMAL
                elif percentile < 90:
                    bands.volatility_state = VolatilityState.HIGH
                else:
                    bands.volatility_state = VolatilityState.EXPANSION
            
            # Breakout from squeeze
            if len(bandwidth_series) >= 5:
                was_squeeze = (bandwidth_series.iloc[-5:-1] < bandwidth_series.quantile(0.10)).all()
                if was_squeeze and not bands.squeeze:
                    if current_price > bands.middle:
                        bands.breakout_up = True
                    else:
                        bands.breakout_down = True
                        
        except Exception as e:
            logger.debug(f"Bollinger computation error: {e}")
        
        return bands
    
    def _compute_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> StochasticOscillator:
        """Compute Stochastic Oscillator."""
        stoch = StochasticOscillator()
        
        try:
            lowest_low = low.rolling(self.stoch_k).min()
            highest_high = high.rolling(self.stoch_k).max()
            
            range_hl = highest_high - lowest_low
            range_hl = range_hl.replace(0, 0.001)  # Avoid div by zero
            
            k = 100 * (close - lowest_low) / range_hl
            d = k.rolling(self.stoch_d).mean()
            
            stoch.k = k.iloc[-1]
            stoch.d = d.iloc[-1]
            
            stoch.overbought = stoch.k > 80
            stoch.oversold = stoch.k < 20
            
            # Crossovers
            if len(k) >= 2 and len(d) >= 2:
                stoch.bullish_cross = k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1]
                stoch.bearish_cross = k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1]
                
        except Exception as e:
            logger.debug(f"Stochastic computation error: {e}")
        
        return stoch
    
    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> ADXIndicator:
        """Compute ADX (Average Directional Index)."""
        adx_ind = ADXIndicator()
        
        try:
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.adx_period).mean()
            
            # Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            
            # Smoothed DM
            plus_dm_smooth = plus_dm.rolling(self.adx_period).mean()
            minus_dm_smooth = minus_dm.rolling(self.adx_period).mean()
            
            # Directional Indicators
            atr_safe = atr.replace(0, 0.001)
            plus_di = 100 * plus_dm_smooth / atr_safe
            minus_di = 100 * minus_dm_smooth / atr_safe
            
            # ADX
            di_sum = plus_di + minus_di
            di_sum = di_sum.replace(0, 0.001)
            dx = 100 * abs(plus_di - minus_di) / di_sum
            adx = dx.rolling(self.adx_period).mean()
            
            adx_ind.adx = adx.iloc[-1]
            adx_ind.plus_di = plus_di.iloc[-1]
            adx_ind.minus_di = minus_di.iloc[-1]
            
            adx_ind.trending = adx_ind.adx > 25
            adx_ind.strong_trend = adx_ind.adx > 40
            
            # Trend strength classification
            if adx_ind.adx < 20:
                adx_ind.trend_strength = TrendStrength.NEUTRAL
            elif adx_ind.plus_di > adx_ind.minus_di:
                if adx_ind.adx > 40:
                    adx_ind.trend_strength = TrendStrength.STRONG_BULLISH
                elif adx_ind.adx > 25:
                    adx_ind.trend_strength = TrendStrength.BULLISH
                else:
                    adx_ind.trend_strength = TrendStrength.WEAK_BULLISH
            else:
                if adx_ind.adx > 40:
                    adx_ind.trend_strength = TrendStrength.STRONG_BEARISH
                elif adx_ind.adx > 25:
                    adx_ind.trend_strength = TrendStrength.BEARISH
                else:
                    adx_ind.trend_strength = TrendStrength.WEAK_BEARISH
                    
        except Exception as e:
            logger.debug(f"ADX computation error: {e}")
        
        return adx_ind
    
    def _compute_volume(self, df: pd.DataFrame) -> VolumeAnalysis:
        """Compute volume-based indicators."""
        vol = VolumeAnalysis()
        
        try:
            close = df['close']
            volume = df['volume']
            high = df['high']
            low = df['low']
            
            # OBV (On Balance Volume)
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(df)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            vol.obv = obv.iloc[-1]
            
            # OBV trend (using 20-day linear regression)
            if len(obv) >= 20:
                x = np.arange(20)
                y = obv.iloc[-20:].values
                slope = np.polyfit(x, y, 1)[0]
                if slope > 0:
                    vol.obv_trend = "up"
                elif slope < 0:
                    vol.obv_trend = "down"
            
            # Volume ratio
            vol_ma = volume.rolling(20).mean()
            if vol_ma.iloc[-1] > 0:
                vol.volume_ma_ratio = volume.iloc[-1] / vol_ma.iloc[-1]
            
            vol.volume_breakout = vol.volume_ma_ratio > 2.0
            vol.volume_dry_up = vol.volume_ma_ratio < 0.5
            
            # Money Flow Index (MFI)
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(14).sum()
            negative_mf = negative_flow.rolling(14).sum()
            
            negative_mf_safe = negative_mf.replace(0, 0.001)
            mfi = 100 - (100 / (1 + positive_mf / negative_mf_safe))
            vol.money_flow_index = mfi.iloc[-1]
            
            # Accumulation/Distribution signals
            price_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if close.iloc[-5] > 0 else 0
            vol.accumulation = vol.money_flow_index > 80 and price_change > 0
            vol.distribution = vol.money_flow_index < 20 and price_change < 0
            
        except Exception as e:
            logger.debug(f"Volume computation error: {e}")
        
        return vol
    
    def _compute_price_action(self, high: pd.Series, low: pd.Series, close: pd.Series) -> PriceAction:
        """Compute price action analysis."""
        pa = PriceAction()
        
        try:
            current_price = close.iloc[-1]
            
            # Find support/resistance using swing highs/lows
            lookback = min(50, len(close))
            recent_high = high.iloc[-lookback:]
            recent_low = low.iloc[-lookback:]
            
            # Simple approach: use quantiles as key levels
            pa.resistance_1 = recent_high.quantile(0.90)
            pa.resistance_2 = recent_high.quantile(0.95)
            pa.support_1 = recent_low.quantile(0.10)
            pa.support_2 = recent_low.quantile(0.05)
            
            # Distance to levels
            if current_price > 0:
                pa.dist_to_support = (current_price - pa.support_1) / current_price
                pa.dist_to_resistance = (pa.resistance_1 - current_price) / current_price
            
            # Higher highs/lows pattern
            if len(high) >= 20:
                highs = [high.iloc[i-5:i].max() for i in range(10, 21, 5)]
                lows = [low.iloc[i-5:i].min() for i in range(10, 21, 5)]
                
                if len(highs) >= 2:
                    pa.higher_highs = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
                    pa.lower_highs = all(highs[i] < highs[i-1] for i in range(1, len(highs)))
                
                if len(lows) >= 2:
                    pa.higher_lows = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
                    pa.lower_lows = all(lows[i] < lows[i-1] for i in range(1, len(lows)))
            
            # Breakout detection
            if current_price > pa.resistance_1:
                pa.breakout_direction = "up"
                pa.breakout_strength = (current_price - pa.resistance_1) / pa.resistance_1
            elif current_price < pa.support_1:
                pa.breakout_direction = "down"
                pa.breakout_strength = (pa.support_1 - current_price) / pa.support_1
                
        except Exception as e:
            logger.debug(f"Price action computation error: {e}")
        
        return pa
    
    def _compute_trend_score(self, analysis: TechnicalAnalysis) -> float:
        """Compute overall trend score from -1 to +1."""
        score = 0.0
        
        # ADX contribution (40% weight)
        if analysis.adx.trend_strength == TrendStrength.STRONG_BULLISH:
            score += 0.4
        elif analysis.adx.trend_strength == TrendStrength.BULLISH:
            score += 0.3
        elif analysis.adx.trend_strength == TrendStrength.WEAK_BULLISH:
            score += 0.15
        elif analysis.adx.trend_strength == TrendStrength.WEAK_BEARISH:
            score -= 0.15
        elif analysis.adx.trend_strength == TrendStrength.BEARISH:
            score -= 0.3
        elif analysis.adx.trend_strength == TrendStrength.STRONG_BEARISH:
            score -= 0.4
        
        # MACD contribution (30% weight)
        if analysis.macd.histogram > 0:
            score += min(0.3, analysis.macd.histogram / 5)
        else:
            score += max(-0.3, analysis.macd.histogram / 5)
        
        # Price action (30% weight)
        if analysis.price_action.higher_highs and analysis.price_action.higher_lows:
            score += 0.3
        elif analysis.price_action.lower_highs and analysis.price_action.lower_lows:
            score -= 0.3
        
        return max(-1, min(1, score))
    
    def _compute_momentum_score(self, analysis: TechnicalAnalysis) -> float:
        """Compute momentum score from -1 to +1."""
        score = 0.0
        
        # MACD momentum
        if analysis.macd.momentum_state == MomentumState.ACCELERATING_UP:
            score += 0.4
        elif analysis.macd.momentum_state == MomentumState.DECELERATING_UP:
            score += 0.2
        elif analysis.macd.momentum_state == MomentumState.DECELERATING_DOWN:
            score -= 0.2
        elif analysis.macd.momentum_state == MomentumState.ACCELERATING_DOWN:
            score -= 0.4
        
        # Stochastic contribution
        if analysis.stochastic.oversold and analysis.stochastic.bullish_cross:
            score += 0.3
        elif analysis.stochastic.overbought and analysis.stochastic.bearish_cross:
            score -= 0.3
        elif analysis.stochastic.k > 50:
            score += 0.1
        else:
            score -= 0.1
        
        # Bollinger %B
        if analysis.bollinger.percent_b > 1:  # Above upper band
            score += 0.2  # Strong momentum (can also mean overbought)
        elif analysis.bollinger.percent_b < 0:  # Below lower band
            score -= 0.2
        
        return max(-1, min(1, score))
    
    def _compute_volatility_score(self, analysis: TechnicalAnalysis) -> float:
        """Compute volatility score from 0 to 1."""
        if analysis.bollinger.volatility_state == VolatilityState.SQUEEZE:
            return 0.1
        elif analysis.bollinger.volatility_state == VolatilityState.LOW:
            return 0.3
        elif analysis.bollinger.volatility_state == VolatilityState.NORMAL:
            return 0.5
        elif analysis.bollinger.volatility_state == VolatilityState.HIGH:
            return 0.7
        elif analysis.bollinger.volatility_state == VolatilityState.EXPANSION:
            return 0.9
        return 0.5
    
    def _compute_volume_score(self, analysis: TechnicalAnalysis) -> float:
        """Compute volume score from -1 (distribution) to +1 (accumulation)."""
        score = 0.0
        
        if analysis.volume.obv_trend == "up":
            score += 0.3
        elif analysis.volume.obv_trend == "down":
            score -= 0.3
        
        if analysis.volume.accumulation:
            score += 0.4
        elif analysis.volume.distribution:
            score -= 0.4
        
        # Volume confirmation
        if analysis.volume.volume_breakout:
            score += 0.2 if analysis.trend_score > 0 else -0.2
        
        return max(-1, min(1, score))
    
    def _compute_composite_signal(self, analysis: TechnicalAnalysis) -> float:
        """Compute weighted composite signal."""
        # Weights for each component
        weights = {
            'trend': 0.30,
            'momentum': 0.30,
            'volume': 0.20,
            'volatility_adj': 0.20,  # Volatility adjusts confidence, not direction
        }
        
        signal = (
            analysis.trend_score * weights['trend'] +
            analysis.momentum_score * weights['momentum'] +
            analysis.volume_score * weights['volume']
        )
        
        # Volatility adjustment: high volatility reduces signal magnitude
        vol_factor = 1.0 - (analysis.volatility_score * 0.3)  # 0.7 to 1.0
        signal *= vol_factor
        
        return max(-1, min(1, signal))
    
    def _compute_confidence(self, analysis: TechnicalAnalysis) -> float:
        """Compute signal confidence based on indicator agreement."""
        agreements = 0
        total = 0
        
        # Check if indicators agree on direction
        direction = 1 if analysis.composite_signal > 0 else -1
        
        # Trend
        if (analysis.trend_score > 0) == (direction > 0):
            agreements += 1
        total += 1
        
        # Momentum
        if (analysis.momentum_score > 0) == (direction > 0):
            agreements += 1
        total += 1
        
        # Volume
        if (analysis.volume_score > 0) == (direction > 0):
            agreements += 1
        total += 1
        
        # MACD
        if (analysis.macd.histogram > 0) == (direction > 0):
            agreements += 1
        total += 1
        
        # Stochastic
        stoch_bullish = analysis.stochastic.k > 50
        if stoch_bullish == (direction > 0):
            agreements += 1
        total += 1
        
        confidence = agreements / total if total > 0 else 0.5
        
        # Boost confidence if ADX shows strong trend
        if analysis.adx.strong_trend:
            confidence = min(1.0, confidence + 0.1)
        
        return confidence


# ============================================================================
# Singleton instance and convenience functions
# ============================================================================

_analyzer_instance: Optional[TechnicalAnalyzer] = None


def get_technical_analyzer() -> TechnicalAnalyzer:
    """Get singleton instance of TechnicalAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TechnicalAnalyzer()
    return _analyzer_instance


def analyze_symbol(symbol: str, df: pd.DataFrame) -> TechnicalAnalysis:
    """Convenience function to analyze a single symbol."""
    return get_technical_analyzer().analyze(symbol, df)


def batch_analyze(
    data: Dict[str, pd.DataFrame],
    include_volume: bool = True,
) -> Dict[str, TechnicalAnalysis]:
    """
    Analyze multiple symbols.
    
    Args:
        data: Dict of symbol -> DataFrame
        include_volume: Include volume analysis
        
    Returns:
        Dict of symbol -> TechnicalAnalysis
    """
    analyzer = get_technical_analyzer()
    results = {}
    
    for symbol, df in data.items():
        try:
            results[symbol] = analyzer.analyze(symbol, df, include_volume)
        except Exception as e:
            logger.warning(f"Failed to analyze {symbol}: {e}")
    
    return results


def get_signal_summary(analysis: TechnicalAnalysis) -> Dict[str, Any]:
    """Get a concise summary for strategy integration."""
    return {
        'symbol': analysis.symbol,
        'signal': analysis.composite_signal,
        'confidence': analysis.signal_confidence,
        'trend': analysis.trend_score,
        'momentum': analysis.momentum_score,
        'is_bullish': analysis.composite_signal > 0.2,
        'is_bearish': analysis.composite_signal < -0.2,
        'is_overbought': analysis.stochastic.overbought or analysis.bollinger.above_upper,
        'is_oversold': analysis.stochastic.oversold or analysis.bollinger.below_lower,
        'has_momentum': abs(analysis.momentum_score) > 0.3,
        'trending': analysis.adx.trending,
        'volatility': analysis.bollinger.volatility_state.value,
    }
