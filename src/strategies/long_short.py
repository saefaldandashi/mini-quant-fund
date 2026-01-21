"""
Long/Short Strategies - Market neutral and directional L/S strategies.

These strategies output NEGATIVE weights for short positions.

Strategies:
1. CrossSectionalMomentumLS: Long top momentum, short bottom momentum
2. TimeSeriesMomentumLS: Long/short based on trend direction
3. MeanReversionLS: Pairs trading / z-score based
4. QualityValueLS: Long quality/value, short expensive/junk
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import logging

from .base import Strategy, SignalOutput
from src.data.feature_store import Features


class CrossSectionalMomentumLS(Strategy):
    """
    Cross-sectional momentum with long/short.
    
    Long the top quantile by momentum, short the bottom quantile.
    Can be sector-neutral if enabled.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CS_Momentum_LS", config)
        
        # Configuration
        self.lookback_days = config.get('lookback_days', 126) if config else 126
        self.long_quantile = config.get('long_quantile', 0.2) if config else 0.2  # Top 20%
        self.short_quantile = config.get('short_quantile', 0.2) if config else 0.2  # Bottom 20%
        self.target_gross = config.get('target_gross', 1.0) if config else 1.0  # 100% gross
        self.target_net = config.get('target_net', 0.0) if config else 0.0  # Market neutral
        
        self._required_features = ['returns_126d', 'prices', 'volatility_21d']
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate market-neutral momentum signals."""
        
        # Get momentum scores
        if self.lookback_days == 126:
            momentum = features.returns_126d
        elif self.lookback_days == 63:
            momentum = features.returns_63d
        elif self.lookback_days == 21:
            momentum = features.returns_21d
        else:
            momentum = features.returns_126d
        
        if not momentum:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": "No momentum data available"}
            )
        
        # Rank stocks by momentum
        sorted_stocks = sorted(momentum.items(), key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
        valid_stocks = [(s, m) for s, m in sorted_stocks if m is not None]
        
        if len(valid_stocks) < 10:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.1,
                explanation={"error": "Not enough stocks for L/S momentum"}
            )
        
        n_stocks = len(valid_stocks)
        n_long = max(1, int(n_stocks * self.long_quantile))
        n_short = max(1, int(n_stocks * self.short_quantile))
        
        # Select longs (top momentum)
        longs = valid_stocks[:n_long]
        
        # Select shorts (bottom momentum)
        shorts = valid_stocks[-n_short:]
        
        # Calculate weights (equal-weighted within each leg)
        long_weight = (self.target_gross / 2) / n_long
        short_weight = -(self.target_gross / 2) / n_short
        
        weights = {}
        
        for symbol, mom in longs:
            weights[symbol] = long_weight
        
        for symbol, mom in shorts:
            weights[symbol] = short_weight
        
        # Adjust for target net exposure
        current_net = sum(weights.values())
        if abs(current_net - self.target_net) > 0.01:
            # Adjust to hit target net
            adjustment = (self.target_net - current_net) / len(weights)
            weights = {k: v + adjustment for k, v in weights.items()}
        
        # Calculate expected returns
        long_mom = np.mean([m for _, m in longs])
        short_mom = np.mean([m for _, m in shorts])
        spread = long_mom - short_mom
        
        # Confidence based on spread magnitude
        confidence = min(1.0, max(0.0, spread / 0.5))  # Full confidence at 50% spread
        
        # Risk estimate (lower for market-neutral)
        risk = 0.10  # Typical for L/S equity strategy
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=spread * 0.5,  # Discount the spread
            risk_estimate=risk,
            confidence=confidence,
            regime_fit=0.6,
            diversification_score=0.7,  # L/S adds diversification
            explanation={
                "type": "Cross-Sectional Momentum L/S",
                "n_longs": n_long,
                "n_shorts": n_short,
                "long_avg_momentum": f"{long_mom:.1%}",
                "short_avg_momentum": f"{short_mom:.1%}",
                "spread": f"{spread:.1%}",
                "gross_exposure": f"{self.target_gross:.0%}",
                "net_exposure": f"{sum(weights.values()):.1%}",
            }
        )


class TimeSeriesMomentumLS(Strategy):
    """
    Time-series momentum with long/short capability.
    
    Goes long assets with positive trend, short assets with negative trend.
    Position size scales with trend strength.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TS_Momentum_LS", config)
        
        self.lookback_days = config.get('lookback_days', 126) if config else 126
        self.vol_lookback = config.get('vol_lookback', 21) if config else 21
        self.vol_target = config.get('vol_target', 0.10) if config else 0.10
        self.max_position = config.get('max_position', 0.15) if config else 0.15
        
        self._required_features = ['returns_126d', 'volatility_21d', 'prices']
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate time-series momentum signals."""
        
        momentum = features.returns_126d
        volatility = features.volatility_21d
        
        if not momentum or not volatility:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": "Missing momentum or volatility data"}
            )
        
        weights = {}
        expected_returns = {}
        
        for symbol in momentum:
            mom = momentum.get(symbol)
            vol = volatility.get(symbol, 0.20)
            
            if mom is None or vol is None or vol == 0:
                continue
            
            # Direction based on trend sign
            direction = 1 if mom > 0 else -1
            
            # Size based on vol targeting
            vol_scalar = self.vol_target / vol if vol > 0 else 0.5
            
            # Strength based on momentum magnitude
            strength = min(1.0, abs(mom) / 0.3)  # Full strength at 30% momentum
            
            # Final weight
            raw_weight = direction * vol_scalar * strength * self.max_position
            
            # Clip to max
            weight = max(-self.max_position, min(self.max_position, raw_weight))
            
            if abs(weight) > 0.01:
                weights[symbol] = weight
                expected_returns[symbol] = mom * 0.3  # Discount
        
        if not weights:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": "No valid positions"}
            )
        
        # Stats
        n_longs = sum(1 for w in weights.values() if w > 0)
        n_shorts = sum(1 for w in weights.values() if w < 0)
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=self._calculate_expected_return(weights, expected_returns),
            expected_returns_by_asset=expected_returns,
            risk_estimate=self.vol_target,
            confidence=0.6,
            regime_fit=0.7,  # Works well in trending markets
            diversification_score=0.6,
            explanation={
                "type": "Time-Series Momentum L/S",
                "n_longs": n_longs,
                "n_shorts": n_shorts,
                "gross_exposure": f"{gross:.1%}",
                "net_exposure": f"{net:.1%}",
                "vol_target": f"{self.vol_target:.1%}",
            }
        )


class MeanReversionLS(Strategy):
    """
    Mean reversion strategy using z-scores.
    
    Shorts overbought stocks (high z-score), longs oversold stocks (low z-score).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("MeanReversion_LS", config)
        
        self.z_threshold = config.get('z_threshold', 1.5) if config else 1.5
        self.position_size = config.get('position_size', 0.05) if config else 0.05
        self.max_positions = config.get('max_positions', 10) if config else 10
        
        self._required_features = ['prices', 'volatility_21d', 'returns_21d']
    
    def _calculate_zscore(self, returns_21d: float, vol_21d: float) -> float:
        """Calculate z-score from recent return and volatility."""
        if vol_21d is None or vol_21d == 0:
            return 0.0
        # Annualize the return and compare to vol
        annualized_return = returns_21d * (252 / 21)
        return annualized_return / vol_21d if vol_21d > 0 else 0.0
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate mean reversion signals based on z-scores."""
        
        returns_21d = features.returns_21d
        volatility = features.volatility_21d
        
        if not returns_21d or not volatility:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": "Missing data for z-score calculation"}
            )
        
        # Calculate z-scores
        zscores = {}
        for symbol in returns_21d:
            ret = returns_21d.get(symbol)
            vol = volatility.get(symbol)
            if ret is not None and vol is not None:
                zscores[symbol] = self._calculate_zscore(ret, vol)
        
        if not zscores:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": "No z-scores calculated"}
            )
        
        # Sort by z-score
        sorted_zscores = sorted(zscores.items(), key=lambda x: x[1])
        
        weights = {}
        
        # Long oversold (low z-score, negative = beaten down)
        longs = [(s, z) for s, z in sorted_zscores if z < -self.z_threshold][:self.max_positions // 2]
        
        # Short overbought (high z-score, positive = extended)
        shorts = [(s, z) for s, z in sorted_zscores if z > self.z_threshold][-self.max_positions // 2:]
        
        for symbol, z in longs:
            # Size inversely proportional to z-score (more oversold = larger position)
            size = min(self.position_size * (abs(z) / self.z_threshold), self.position_size * 2)
            weights[symbol] = size
        
        for symbol, z in shorts:
            # Size proportional to z-score (more overbought = larger short)
            size = min(self.position_size * (abs(z) / self.z_threshold), self.position_size * 2)
            weights[symbol] = -size
        
        if not weights:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"message": "No extreme z-scores found"}
            )
        
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=0.05,  # Mean reversion targets smaller, consistent returns
            risk_estimate=0.08,
            confidence=0.5,
            regime_fit=0.8 if features.regime and features.regime.trend_strength < 0.5 else 0.3,
            diversification_score=0.8,
            explanation={
                "type": "Mean Reversion L/S",
                "z_threshold": self.z_threshold,
                "n_longs": len(longs),
                "n_shorts": len(shorts),
                "gross_exposure": f"{gross:.1%}",
                "net_exposure": f"{net:.1%}",
                "avg_long_zscore": f"{np.mean([z for _, z in longs]):.2f}" if longs else "N/A",
                "avg_short_zscore": f"{np.mean([z for _, z in shorts]):.2f}" if shorts else "N/A",
            }
        )


class QualityValueLS(Strategy):
    """
    Quality/Value long-short strategy.
    
    Longs high quality/value stocks, shorts low quality/expensive stocks.
    Uses momentum as quality proxy when fundamentals unavailable.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("QualityValue_LS", config)
        
        self.target_gross = config.get('target_gross', 0.6) if config else 0.6
        self.n_positions = config.get('n_positions', 10) if config else 10
        
        self._required_features = ['returns_126d', 'volatility_21d', 'returns_21d']
    
    def _calculate_quality_score(
        self, 
        symbol: str,
        momentum_126d: float,
        momentum_21d: float, 
        volatility: float
    ) -> float:
        """
        Calculate quality score.
        
        Higher score = more quality (long candidate)
        Lower score = less quality (short candidate)
        
        When fundamentals unavailable, uses:
        - Positive momentum = quality (growing)
        - Low volatility = quality (stable)
        - Momentum consistency = quality
        """
        if momentum_126d is None or volatility is None:
            return 0.0
        
        # Momentum component (normalized)
        mom_score = np.clip(momentum_126d / 0.3, -1, 1)  # -1 to 1
        
        # Volatility component (low vol = high quality)
        vol_score = np.clip(1.0 - volatility / 0.4, 0, 1)  # 0 to 1, lower vol = higher
        
        # Momentum consistency (short-term aligned with long-term)
        if momentum_21d is not None:
            consistency = 1.0 if (momentum_126d > 0) == (momentum_21d > 0) else 0.5
        else:
            consistency = 0.75
        
        # Combine
        quality = mom_score * 0.5 + vol_score * 0.3 + consistency * 0.2
        
        return quality
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate quality/value L/S signals."""
        
        momentum_126d = features.returns_126d
        momentum_21d = features.returns_21d
        volatility = features.volatility_21d
        
        if not momentum_126d:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": "No momentum data for quality scoring"}
            )
        
        # Calculate quality scores
        quality_scores = {}
        for symbol in momentum_126d:
            mom_126 = momentum_126d.get(symbol)
            mom_21 = momentum_21d.get(symbol) if momentum_21d else None
            vol = volatility.get(symbol) if volatility else 0.20
            
            if mom_126 is not None:
                quality_scores[symbol] = self._calculate_quality_score(
                    symbol, mom_126, mom_21, vol
                )
        
        if len(quality_scores) < 4:
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": "Not enough stocks for quality ranking"}
            )
        
        # Rank by quality
        sorted_scores = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        n_each = self.n_positions // 2
        
        # Long high quality
        longs = sorted_scores[:n_each]
        
        # Short low quality
        shorts = sorted_scores[-n_each:]
        
        # Calculate weights
        long_weight = (self.target_gross / 2) / len(longs) if longs else 0
        short_weight = -(self.target_gross / 2) / len(shorts) if shorts else 0
        
        weights = {}
        for symbol, score in longs:
            weights[symbol] = long_weight
        for symbol, score in shorts:
            weights[symbol] = short_weight
        
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=0.08,
            risk_estimate=0.10,
            confidence=0.55,
            regime_fit=0.6,
            diversification_score=0.7,
            explanation={
                "type": "Quality/Value L/S",
                "n_longs": len(longs),
                "n_shorts": len(shorts),
                "avg_long_quality": f"{np.mean([s for _, s in longs]):.2f}",
                "avg_short_quality": f"{np.mean([s for _, s in shorts]):.2f}",
                "gross_exposure": f"{gross:.1%}",
                "net_exposure": f"{net:.1%}",
            }
        )


# === FACTORY FUNCTION ===

def create_long_short_strategies(config: Optional[Dict[str, Any]] = None) -> List[Strategy]:
    """
    Create all L/S strategies.
    
    Args:
        config: Optional configuration dict with strategy-specific configs
    
    Returns:
        List of L/S strategy instances
    """
    config = config or {}
    
    return [
        CrossSectionalMomentumLS(config.get('cs_momentum', {})),
        TimeSeriesMomentumLS(config.get('ts_momentum', {})),
        MeanReversionLS(config.get('mean_reversion', {})),
        QualityValueLS(config.get('quality_value', {})),
    ]
