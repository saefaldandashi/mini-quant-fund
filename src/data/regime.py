"""
Market regime classification.
Identifies trend, volatility, and correlation regimes.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrendRegime(Enum):
    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    NEUTRAL = "neutral"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_down"


class VolatilityRegime(Enum):
    LOW = "low_vol"
    NORMAL = "normal_vol"
    HIGH = "high_vol"
    EXTREME = "extreme_vol"


class RiskRegime(Enum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


@dataclass
class MarketRegime:
    """Complete market regime classification."""
    timestamp: pd.Timestamp
    trend: TrendRegime
    trend_strength: float  # 0 to 1
    volatility: VolatilityRegime
    volatility_percentile: float  # 0 to 1
    correlation_regime: float  # Average pairwise correlation
    risk_regime: RiskRegime
    description: str
    spy_tlt_correlation: float = 0.0  # SPY/TLT correlation (negative = normal, positive = crisis)


class RegimeClassifier:
    """
    Classify market regimes using technical indicators.
    No look-ahead bias - uses only data available at each timestamp.
    """
    
    def __init__(
        self,
        fast_ma: int = 20,
        slow_ma: int = 100,
        vol_window: int = 21,
        vol_lookback: int = 252,
        corr_window: int = 63
    ):
        """
        Initialize regime classifier.
        
        Args:
            fast_ma: Fast moving average period
            slow_ma: Slow moving average period
            vol_window: Window for realized volatility
            vol_lookback: Lookback for volatility percentile
            corr_window: Window for correlation calculation
        """
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.corr_window = corr_window
        
    def classify(
        self,
        prices: pd.DataFrame,
        benchmark: str = "SPY",
        as_of: Optional[pd.Timestamp] = None
    ) -> MarketRegime:
        """
        Classify current market regime.
        
        Args:
            prices: DataFrame with symbol columns and datetime index
            benchmark: Benchmark symbol for regime
            as_of: Date to classify (default: latest)
            
        Returns:
            MarketRegime object
        """
        if as_of is not None:
            prices = prices.loc[:as_of]
            
        if len(prices) < self.slow_ma:
            return self._default_regime(prices.index[-1])
        
        # Get benchmark prices
        if benchmark in prices.columns:
            bench = prices[benchmark]
        else:
            bench = prices.iloc[:, 0]  # Use first column
            
        timestamp = prices.index[-1]
        
        # Trend classification
        trend, trend_strength = self._classify_trend(bench)
        
        # Volatility classification
        vol_regime, vol_percentile = self._classify_volatility(bench)
        
        # Correlation regime
        corr_regime = self._calculate_correlation_regime(prices)
        
        # SPY/TLT correlation (flight to safety indicator)
        spy_tlt_corr = self._calculate_spy_tlt_correlation(prices)
        
        # Risk regime (composite) - now includes SPY/TLT correlation
        risk_regime = self._classify_risk(trend, vol_regime, corr_regime, spy_tlt_corr)
        
        # Generate description
        description = self._generate_description(
            trend, trend_strength, vol_regime, vol_percentile, corr_regime, risk_regime
        )
        
        return MarketRegime(
            timestamp=timestamp,
            trend=trend,
            trend_strength=trend_strength,
            volatility=vol_regime,
            volatility_percentile=vol_percentile,
            correlation_regime=corr_regime,
            risk_regime=risk_regime,
            description=description,
            spy_tlt_correlation=spy_tlt_corr,
        )
    
    def _classify_trend(self, prices: pd.Series) -> Tuple[TrendRegime, float]:
        """Classify trend based on MA crossover."""
        fast = prices.rolling(self.fast_ma).mean()
        slow = prices.rolling(self.slow_ma).mean()
        
        current_fast = fast.iloc[-1]
        current_slow = slow.iloc[-1]
        current_price = prices.iloc[-1]
        
        # Trend strength: normalized distance between MAs
        trend_strength = abs(current_fast - current_slow) / current_price
        trend_strength = min(1.0, trend_strength * 10)  # Scale to 0-1
        
        # Direction
        if current_fast > current_slow:
            if trend_strength > 0.5:
                trend = TrendRegime.STRONG_UP
            elif trend_strength > 0.2:
                trend = TrendRegime.WEAK_UP
            else:
                trend = TrendRegime.NEUTRAL
        else:
            if trend_strength > 0.5:
                trend = TrendRegime.STRONG_DOWN
            elif trend_strength > 0.2:
                trend = TrendRegime.WEAK_DOWN
            else:
                trend = TrendRegime.NEUTRAL
                
        return trend, trend_strength
    
    def _classify_volatility(self, prices: pd.Series) -> Tuple[VolatilityRegime, float]:
        """Classify volatility regime."""
        returns = prices.pct_change().dropna()
        
        if len(returns) < self.vol_lookback:
            return VolatilityRegime.NORMAL, 0.5
        
        # Current realized vol
        current_vol = returns.tail(self.vol_window).std() * np.sqrt(252)
        
        # Historical vol for percentile
        rolling_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        # Percentile
        percentile = (rolling_vol < current_vol).mean()
        
        # Classification
        if percentile < 0.25:
            regime = VolatilityRegime.LOW
        elif percentile < 0.75:
            regime = VolatilityRegime.NORMAL
        elif percentile < 0.95:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME
            
        return regime, percentile
    
    def _calculate_correlation_regime(self, prices: pd.DataFrame) -> float:
        """Calculate average pairwise correlation."""
        if len(prices.columns) < 2:
            return 0.0
            
        returns = prices.pct_change().dropna()
        
        if len(returns) < self.corr_window:
            return 0.5
            
        recent_returns = returns.tail(self.corr_window)
        corr_matrix = recent_returns.corr()
        
        # Average off-diagonal correlation
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_corr = corr_matrix.values[mask].mean()
        
        return avg_corr
    
    def _calculate_spy_tlt_correlation(self, prices: pd.DataFrame) -> float:
        """
        Calculate SPY/TLT correlation.
        
        This is a key flight-to-safety indicator:
        - Negative correlation (-0.3 to -0.5): Normal markets, stocks/bonds inversely correlated
        - Near zero (0 to 0.2): Transition period, uncertainty
        - Positive correlation (> 0.3): Crisis regime, both selling off (rare but dangerous)
        
        Returns:
            Correlation between SPY and TLT over the correlation window
        """
        if 'SPY' not in prices.columns or 'TLT' not in prices.columns:
            return 0.0  # No data
        
        returns = prices[['SPY', 'TLT']].pct_change().dropna()
        
        if len(returns) < self.corr_window:
            return 0.0
        
        recent = returns.tail(self.corr_window)
        corr = recent['SPY'].corr(recent['TLT'])
        
        return corr if not np.isnan(corr) else 0.0
    
    def _classify_risk(
        self,
        trend: TrendRegime,
        vol: VolatilityRegime,
        corr: float,
        spy_tlt_corr: float = 0.0
    ) -> RiskRegime:
        """Composite risk regime classification."""
        score = 0
        
        # Trend contribution
        if trend in [TrendRegime.STRONG_UP, TrendRegime.WEAK_UP]:
            score += 1
        elif trend in [TrendRegime.STRONG_DOWN, TrendRegime.WEAK_DOWN]:
            score -= 1
            
        # Vol contribution (high vol = risk off)
        if vol in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            score -= 1
        elif vol == VolatilityRegime.LOW:
            score += 0.5
            
        # Correlation contribution (high corr = risk off)
        if corr > 0.7:
            score -= 0.5
        
        # SPY/TLT correlation (crisis indicator)
        # Positive correlation means both stocks and bonds selling = severe risk off
        if spy_tlt_corr > 0.3:
            score -= 1.0  # Strong risk-off signal
            logger.warning(f"Crisis indicator: SPY/TLT correlation = {spy_tlt_corr:.2f}")
        elif spy_tlt_corr < -0.3:
            score += 0.3  # Normal inverse relationship = healthy market
            
        if score > 0.5:
            return RiskRegime.RISK_ON
        elif score < -0.5:
            return RiskRegime.RISK_OFF
        else:
            return RiskRegime.NEUTRAL
    
    def _generate_description(
        self,
        trend: TrendRegime,
        trend_strength: float,
        vol: VolatilityRegime,
        vol_percentile: float,
        corr: float,
        risk: RiskRegime
    ) -> str:
        """Generate human-readable regime description."""
        parts = []
        
        # Trend
        trend_map = {
            TrendRegime.STRONG_UP: "Strong uptrend",
            TrendRegime.WEAK_UP: "Weak uptrend",
            TrendRegime.NEUTRAL: "Range-bound",
            TrendRegime.WEAK_DOWN: "Weak downtrend",
            TrendRegime.STRONG_DOWN: "Strong downtrend",
        }
        parts.append(f"{trend_map[trend]} (strength: {trend_strength:.1%})")
        
        # Volatility
        vol_map = {
            VolatilityRegime.LOW: "low",
            VolatilityRegime.NORMAL: "normal",
            VolatilityRegime.HIGH: "elevated",
            VolatilityRegime.EXTREME: "extreme",
        }
        parts.append(f"{vol_map[vol]} volatility ({vol_percentile:.0%} percentile)")
        
        # Correlation
        parts.append(f"correlation at {corr:.2f}")
        
        # Risk
        risk_map = {
            RiskRegime.RISK_ON: "Risk-on environment",
            RiskRegime.NEUTRAL: "Neutral risk",
            RiskRegime.RISK_OFF: "Risk-off environment",
        }
        parts.append(risk_map[risk])
        
        return "; ".join(parts)
    
    def _default_regime(self, timestamp: pd.Timestamp) -> MarketRegime:
        """Return default regime when insufficient data."""
        return MarketRegime(
            timestamp=timestamp,
            trend=TrendRegime.NEUTRAL,
            trend_strength=0.0,
            volatility=VolatilityRegime.NORMAL,
            volatility_percentile=0.5,
            correlation_regime=0.5,
            risk_regime=RiskRegime.NEUTRAL,
            description="Insufficient data for regime classification",
        )
