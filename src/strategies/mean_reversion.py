"""
Mean Reversion Strategy.
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from .base import Strategy, SignalOutput
from src.data.feature_store import Features
from src.data.regime import TrendRegime, VolatilityRegime


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy.
    Buys oversold assets, sells overbought assets based on deviation from moving averages.
    
    NOW WITH MACRO AWARENESS:
    - More aggressive in calm macro environments
    - Conservative when financial stress is high
    - Uses peer consensus to confirm reversions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("MeanReversion", config)
        self._required_features = ['prices', 'ma_20', 'ma_50', 'volatility_21d']
        
        # Config
        self.z_threshold = config.get('z_threshold', 2.0) if config else 2.0
        self.ma_type = config.get('ma_type', 'ma_20') if config else 'ma_20'
        self.max_position = config.get('max_position', 0.1) if config else 0.1
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate mean reversion signals with macro awareness."""
        weights = {}
        expected_returns = {}
        explanations = {}
        
        # MACRO ADJUSTMENT - mean reversion is dangerous in crisis
        macro_multiplier = 1.0
        macro_reason = None
        
        if self.macro_features is not None:
            fin_stress = getattr(self.macro_features, 'financial_stress_index', 0)
            geo_risk = getattr(self.macro_features, 'geopolitical_risk_index', 0)
            
            # Mean reversion can fail catastrophically in stress
            if fin_stress > 0.5:
                macro_multiplier = 0.3  # Heavy reduction
                macro_reason = f"High financial stress ({fin_stress:.2f}) - catching falling knives risk"
            elif geo_risk > 0.5:
                macro_multiplier = 0.5
                macro_reason = f"High geopolitical risk ({geo_risk:.2f})"
            elif fin_stress < -0.3 and geo_risk < -0.3:
                macro_multiplier = 1.3  # Safe to be aggressive
                macro_reason = "Low stress environment favors mean reversion"
        
        if self.risk_sentiment is not None:
            if hasattr(self.risk_sentiment, 'equity_bias'):
                # In strong risk-off, don't buy dips
                if self.risk_sentiment.equity_bias < -0.5:
                    macro_multiplier *= 0.5
                    macro_reason = "Strong risk-off - avoid buying dips"
        
        # Get moving average
        ma = getattr(features, self.ma_type, features.ma_20)
        
        for symbol in features.symbols:
            price = features.prices.get(symbol)
            ma_val = ma.get(symbol)
            vol = features.volatility_21d.get(symbol, 0.20)
            
            if price is None or ma_val is None or vol <= 0:
                continue
            
            # Calculate z-score (deviation from MA in vol units)
            deviation = (price - ma_val) / ma_val
            z_score = deviation / (vol / np.sqrt(252) * 20)  # Rough daily vol * lookback
            
            # Signal: negative z-score = oversold = buy
            if abs(z_score) > self.z_threshold:
                # Fade the move
                raw_weight = -np.sign(z_score) * min(abs(z_score) / self.z_threshold, 2.0)
                weight = raw_weight * self.max_position
                
                # APPLY MACRO ADJUSTMENT
                weight *= macro_multiplier
                
                # CHECK PEER CONSENSUS
                peer_avg, peer_agreement = self.get_peer_consensus(symbol)
                consensus_note = None
                if peer_agreement > 0.5:
                    # If peers agree with our contrarian view, boost
                    if (weight > 0 and peer_avg > 0) or (weight < 0 and peer_avg < 0):
                        weight *= 1.15
                        consensus_note = "Peer consensus confirms reversion"
                    # If peers disagree, be more cautious
                    elif abs(peer_avg) > 0.05:
                        weight *= 0.85
                        consensus_note = "Peers disagree - reduced position"
                
                weights[symbol] = weight
                expected_returns[symbol] = -deviation * 0.5  # Expect mean reversion
                explanations[symbol] = {
                    'price': price,
                    'ma': ma_val,
                    'z_score': z_score,
                    'signal': 'oversold' if z_score < 0 else 'overbought',
                    'macro_multiplier': macro_multiplier,
                    'macro_reason': macro_reason,
                    'consensus_note': consensus_note,
                }
        
        # Regime fit - mean reversion works better in range-bound markets
        regime_fit = 0.5
        if features.regime:
            if features.regime.trend == TrendRegime.NEUTRAL:
                regime_fit = 0.8
            elif features.regime.trend in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
                regime_fit = 0.2  # Poor fit in trending markets
            
            # Higher vol = more reversion opportunities but also more risk
            if features.regime.volatility == VolatilityRegime.HIGH:
                regime_fit *= 0.8
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=0.5,
            explanation={
                'active_signals': len(weights),
                'oversold': sum(1 for e in explanations.values() if e.get('signal') == 'oversold'),
                'overbought': sum(1 for e in explanations.values() if e.get('signal') == 'overbought'),
                'details': explanations,
            },
            regime_fit=regime_fit,
            diversification_score=0.5,
        )
