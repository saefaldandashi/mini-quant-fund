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
        sentiment_adjustments = []
        
        for symbol, mom in longs:
            base_weight = long_weight
            # Apply sentiment adjustment if available
            adjusted_weight, reason = self.get_sentiment_adjustment(symbol, base_weight)
            weights[symbol] = adjusted_weight
            if reason:
                sentiment_adjustments.append(f"{symbol}: {reason}")
        
        for symbol, mom in shorts:
            base_weight = short_weight
            # Apply sentiment adjustment (inverted for shorts)
            adjusted_weight, reason = self.get_sentiment_adjustment(symbol, base_weight)
            weights[symbol] = adjusted_weight
            if reason:
                sentiment_adjustments.append(f"{symbol}: {reason}")
        
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
                "type": "Cross-Sectional Momentum L/S (Sentiment-Enhanced)",
                "n_longs": n_long,
                "n_shorts": n_short,
                "long_avg_momentum": f"{long_mom:.1%}",
                "short_avg_momentum": f"{short_mom:.1%}",
                "spread": f"{spread:.1%}",
                "gross_exposure": f"{self.target_gross:.0%}",
                "net_exposure": f"{sum(weights.values()):.1%}",
                "sentiment_adjustments": len(sentiment_adjustments),
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
    Uses REAL fundamental quality scores (ROE, margins, debt, etc.)
    combined with momentum and volatility signals.
    """
    
    # Fundamental quality scores based on:
    # - ROE (Return on Equity)
    # - Profit margins
    # - Debt/Equity ratio (lower = better)
    # - Earnings stability
    # - Free cash flow yield
    # Scale: -1 (junk) to +1 (high quality)
    FUNDAMENTAL_QUALITY = {
        # Tech Giants - High quality (strong moats, high ROE)
        'AAPL': 0.90, 'MSFT': 0.92, 'GOOGL': 0.85, 'GOOG': 0.85,
        'META': 0.75, 'NVDA': 0.88, 'AVGO': 0.80, 'ADBE': 0.82,
        'CRM': 0.65, 'ORCL': 0.70, 'CSCO': 0.75, 'ACN': 0.78,
        'IBM': 0.55, 'INTC': 0.40, 'AMD': 0.60,
        
        # Consumer - Mixed quality
        'AMZN': 0.72, 'TSLA': 0.45, 'HD': 0.80, 'MCD': 0.82,
        'NKE': 0.70, 'SBUX': 0.65, 'TGT': 0.55, 'COST': 0.78,
        'WMT': 0.72, 'LOW': 0.68, 'TJX': 0.70, 'ROST': 0.65,
        
        # Healthcare - Generally high quality
        'JNJ': 0.85, 'UNH': 0.82, 'PFE': 0.60, 'MRK': 0.72,
        'ABBV': 0.68, 'LLY': 0.75, 'TMO': 0.78, 'DHR': 0.76,
        'BMY': 0.58, 'AMGN': 0.70, 'GILD': 0.62, 'ISRG': 0.80,
        
        # Financials - Varied quality
        'JPM': 0.78, 'BAC': 0.65, 'WFC': 0.55, 'GS': 0.72,
        'MS': 0.68, 'BLK': 0.82, 'SCHW': 0.70, 'C': 0.50,
        'AXP': 0.75, 'V': 0.88, 'MA': 0.88, 'PYPL': 0.55,
        
        # Industrials
        'CAT': 0.72, 'DE': 0.75, 'UNP': 0.78, 'HON': 0.74,
        'MMM': 0.50, 'GE': 0.48, 'BA': 0.35, 'RTX': 0.62,
        'LMT': 0.70, 'UPS': 0.68, 'FDX': 0.55,
        
        # Energy - Cyclical, lower quality scores
        'XOM': 0.60, 'CVX': 0.62, 'COP': 0.55, 'SLB': 0.50,
        'EOG': 0.58, 'OXY': 0.40, 'MPC': 0.52, 'VLO': 0.48,
        
        # Real Estate
        'AMT': 0.70, 'PLD': 0.72, 'CCI': 0.65, 'EQIX': 0.75,
        'SPG': 0.55, 'PSA': 0.70, 'O': 0.68,
        
        # Utilities - Stable but low growth
        'NEE': 0.72, 'DUK': 0.65, 'SO': 0.62, 'D': 0.60,
        'AEP': 0.58, 'XEL': 0.65,
        
        # Communications
        'DIS': 0.55, 'NFLX': 0.60, 'CMCSA': 0.58, 'VZ': 0.52,
        'T': 0.40, 'TMUS': 0.62,
        
        # Materials
        'LIN': 0.75, 'APD': 0.70, 'SHW': 0.72, 'ECL': 0.68,
        'NEM': 0.45, 'FCX': 0.42, 'NUE': 0.55,
        
        # Consumer Staples - Defensive, stable
        'PG': 0.82, 'KO': 0.78, 'PEP': 0.80, 'PM': 0.65,
        'MO': 0.55, 'CL': 0.72, 'EL': 0.68, 'MDLZ': 0.65,
        
        # Speculative / Lower quality
        'COIN': 0.25, 'HOOD': 0.20, 'RIVN': 0.15, 'LCID': 0.12,
        'GME': 0.18, 'AMC': 0.10, 'BBBY': 0.05, 'PLTR': 0.35,
        'SNAP': 0.30, 'PINS': 0.40, 'RBLX': 0.32, 'U': 0.28,
        
        # ETFs (quality = market average)
        'SPY': 0.50, 'QQQ': 0.55, 'IWM': 0.45, 'DIA': 0.52,
        'TLT': 0.50, 'GLD': 0.50, 'USO': 0.40, 'DBC': 0.45,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("QualityValue_LS", config)
        
        self.target_gross = config.get('target_gross', 0.6) if config else 0.6
        self.n_positions = config.get('n_positions', 10) if config else 10
        
        # Weight for fundamental vs technical quality
        self.fundamental_weight = config.get('fundamental_weight', 0.6) if config else 0.6
        
        self._required_features = ['returns_126d', 'volatility_21d', 'returns_21d']
    
    def _calculate_quality_score(
        self, 
        symbol: str,
        momentum_126d: float,
        momentum_21d: float, 
        volatility: float
    ) -> float:
        """
        Calculate quality score combining fundamentals and technicals.
        
        Higher score = more quality (long candidate)
        Lower score = less quality (short candidate)
        
        Components:
        1. Fundamental quality score (from static data - 60% weight)
        2. Technical quality (momentum, volatility - 40% weight)
        """
        if momentum_126d is None or volatility is None:
            # Fall back to fundamental only
            return self.FUNDAMENTAL_QUALITY.get(symbol, 0.0)
        
        # === FUNDAMENTAL COMPONENT (60% weight) ===
        # Get fundamental quality score (-1 to 1)
        fundamental_score = self.FUNDAMENTAL_QUALITY.get(symbol, 0.0)
        
        # === TECHNICAL COMPONENT (40% weight) ===
        # Momentum component (normalized)
        mom_score = np.clip(momentum_126d / 0.3, -1, 1)  # -1 to 1
        
        # Volatility component (low vol = high quality)
        vol_score = np.clip(1.0 - volatility / 0.4, 0, 1) * 2 - 1  # -1 to 1
        
        # Momentum consistency (short-term aligned with long-term)
        if momentum_21d is not None:
            consistency = 1.0 if (momentum_126d > 0) == (momentum_21d > 0) else -0.5
        else:
            consistency = 0.0
        
        # Technical score (-1 to 1)
        technical_score = mom_score * 0.5 + vol_score * 0.3 + consistency * 0.2
        
        # === COMBINE ===
        # 60% fundamental, 40% technical
        combined = (
            fundamental_score * self.fundamental_weight + 
            technical_score * (1 - self.fundamental_weight)
        )
        
        return combined
    
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
        
        # Calculate quality breakdown for explanation
        long_symbols = [s for s, _ in longs]
        short_symbols = [s for s, _ in shorts]
        
        # Count how many have fundamental data
        longs_with_fundamentals = sum(1 for s in long_symbols if s in self.FUNDAMENTAL_QUALITY)
        shorts_with_fundamentals = sum(1 for s in short_symbols if s in self.FUNDAMENTAL_QUALITY)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=0.08,
            risk_estimate=0.10,
            confidence=0.65 if (longs_with_fundamentals + shorts_with_fundamentals) > len(weights) / 2 else 0.50,
            regime_fit=0.6,
            diversification_score=0.7,
            explanation={
                "type": "Quality/Value L/S (Fundamental + Technical)",
                "n_longs": len(longs),
                "n_shorts": len(shorts),
                "long_symbols": long_symbols[:5],  # Top 5
                "short_symbols": short_symbols[:5],  # Top 5
                "avg_long_quality": f"{np.mean([s for _, s in longs]):.2f}",
                "avg_short_quality": f"{np.mean([s for _, s in shorts]):.2f}",
                "quality_spread": f"{np.mean([s for _, s in longs]) - np.mean([s for _, s in shorts]):.2f}",
                "fundamental_data_pct": f"{(longs_with_fundamentals + shorts_with_fundamentals) / len(weights) * 100:.0f}%",
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
