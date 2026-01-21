"""
Tail Risk Overlay Strategy.
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from .base import Strategy, SignalOutput
from src.data.feature_store import Features
from src.data.regime import VolatilityRegime, RiskRegime


class TailRiskOverlayStrategy(Strategy):
    """
    Tail Risk Overlay Strategy.
    Reduces exposure when tail risk indicators are elevated.
    Acts as a defensive overlay to other strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TailRiskOverlay", config)
        self._required_features = ['volatility_21d', 'volatility_63d', 'regime', 'correlation_matrix']
        
        # Config
        self.vol_trigger = config.get('vol_trigger', 0.25) if config else 0.25
        self.corr_trigger = config.get('corr_trigger', 0.7) if config else 0.7
        self.min_exposure = config.get('min_exposure', 0.3) if config else 0.3
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate tail risk overlay signals."""
        weights = {}
        
        # Calculate tail risk indicators
        tail_risk_score = self._calculate_tail_risk(features)
        
        # Exposure scaling: 1.0 = full, min_exposure when max risk
        exposure = 1.0 - (1.0 - self.min_exposure) * tail_risk_score
        
        # This strategy provides a scaling factor, not specific asset weights
        # In practice, it would be applied as an overlay to other strategies
        n_assets = len(features.symbols)
        base_weight = exposure / n_assets
        
        for symbol in features.symbols:
            weights[symbol] = base_weight
        
        # Determine risk status
        if tail_risk_score > 0.7:
            risk_status = "ELEVATED - Significant position reduction"
        elif tail_risk_score > 0.4:
            risk_status = "MODERATE - Slight position reduction"
        else:
            risk_status = "NORMAL - Full exposure"
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=0.0,  # Overlay doesn't predict returns
            risk_estimate=features.volatility_21d.get('SPY', 0.15) if features.volatility_21d else 0.15,
            confidence=0.75,
            explanation={
                'tail_risk_score': tail_risk_score,
                'exposure_multiplier': exposure,
                'risk_status': risk_status,
                'vol_regime': features.regime.volatility.value if features.regime else 'unknown',
                'risk_regime': features.regime.risk_regime.value if features.regime else 'unknown',
            },
            regime_fit=0.8,  # Always relevant
            diversification_score=0.5,
        )
    
    def _calculate_tail_risk(self, features: Features) -> float:
        """
        Calculate composite tail risk score (0 to 1).
        Now includes macro news intelligence signals.
        """
        score = 0.0
        weight_sum = 0.0
        
        # 1. Volatility regime (30% weight)
        if features.regime:
            vol_score = 0.0
            if features.regime.volatility == VolatilityRegime.EXTREME:
                vol_score = 1.0
            elif features.regime.volatility == VolatilityRegime.HIGH:
                vol_score = 0.6
            elif features.regime.volatility == VolatilityRegime.NORMAL:
                vol_score = 0.2
            else:
                vol_score = 0.0
            
            score += vol_score * 0.3
            weight_sum += 0.3
        
        # 2. Correlation regime (20% weight)
        if features.regime:
            corr = features.regime.correlation_regime
            if corr > self.corr_trigger:
                corr_score = min(1.0, (corr - self.corr_trigger) / (1.0 - self.corr_trigger))
            else:
                corr_score = 0.0
            
            score += corr_score * 0.2
            weight_sum += 0.2
        
        # 3. Vol of vol (20% weight)
        if features.volatility_21d and features.volatility_63d:
            vols_21 = list(features.volatility_21d.values())
            vols_63 = list(features.volatility_63d.values())
            
            if vols_21 and vols_63:
                avg_21 = np.mean(vols_21)
                avg_63 = np.mean(vols_63)
                
                if avg_63 > 0:
                    vol_ratio = avg_21 / avg_63
                    if vol_ratio > 1.5:
                        vov_score = min(1.0, (vol_ratio - 1.0) / 1.0)
                    else:
                        vov_score = 0.0
                    
                    score += vov_score * 0.2
                    weight_sum += 0.2
        
        # 4. NEWS INTELLIGENCE: Geopolitical Risk (15% weight)
        if hasattr(features, 'macro_features') and features.macro_features:
            geo_risk = features.macro_features.geopolitical_risk_index
            if geo_risk > 0:
                # Scale 0-1 range to tail risk contribution
                geo_score = min(1.0, geo_risk * 1.5)  # Amplify signal
                score += geo_score * 0.15
                weight_sum += 0.15
        
        # 5. NEWS INTELLIGENCE: Financial Stress (15% weight)
        if hasattr(features, 'macro_features') and features.macro_features:
            fin_stress = features.macro_features.financial_stress_index
            if fin_stress > 0:
                stress_score = min(1.0, fin_stress * 1.5)
                score += stress_score * 0.15
                weight_sum += 0.15
        
        # Normalize
        if weight_sum > 0:
            return score / weight_sum
        return 0.0
