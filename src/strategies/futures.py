"""
Futures Strategies - Macro and carry strategies using futures.

NOTE: These strategies are for BACKTEST ONLY.
Live futures trading requires a broker that supports futures (not Alpaca).

Strategies:
1. FuturesCarryStrategy: Trade carry (backwardation/contango)
2. FuturesMacroOverlay: Allocate across equity/bond/commodity futures based on macro
3. FuturesTrendFollowing: CTA-style trend following across futures
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging

from .base import Strategy, SignalOutput
from src.data.feature_store import Features


class FuturesCarryStrategy(Strategy):
    """
    Futures Carry Strategy (Backtest Only).
    
    Trades based on the shape of the futures term structure:
    - Long contracts in backwardation (spot > futures)
    - Short contracts in contango (futures > spot)
    
    For simplicity, we use ETF proxies to simulate this:
    - Long commodity ETFs when carry is positive
    - Short/avoid when carry is negative
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Futures_Carry", config)
        
        self.position_size = config.get('position_size', 0.10) if config else 0.10
        
        # ETF proxies for different asset classes
        self.asset_proxies = {
            'equity': 'SPY',
            'commodities': 'DBC',  # Commodity index
            'gold': 'GLD',
            'oil': 'USO',
            'bonds': 'TLT',
        }
        
        self._required_features = ['prices', 'returns_21d']
    
    def _estimate_carry(self, features: Features) -> Dict[str, float]:
        """
        Estimate carry for each asset class.
        
        In a production system, this would use actual futures curves.
        Here we use proxies:
        - For commodities: Use 21d momentum as carry proxy
        - For bonds: Use yield curve slope
        - For equities: Use VIX/realized vol spread
        """
        carry_estimates = {}
        
        # Commodity carry proxy: momentum (backwardation often = positive momentum)
        for asset, proxy in self.asset_proxies.items():
            mom = features.returns_21d.get(proxy) if features.returns_21d else None
            vol = features.volatility_21d.get(proxy, 0.15) if features.volatility_21d else 0.15
            
            if mom is not None:
                # Normalize by volatility to get Sharpe-like carry
                carry = mom / vol if vol > 0 else 0
                carry_estimates[asset] = carry
        
        return carry_estimates
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate carry-based signals with graceful error handling."""
        try:
            carry = self._estimate_carry(features)
            
            if not carry:
                return SignalOutput(
                    strategy_name=self.name,
                    timestamp=t,
                    confidence=0.0,
                    explanation={"error": "Could not estimate carry - missing data"}
                )
            
            weights = {}
            
            for asset, carry_signal in carry.items():
                proxy = self.asset_proxies[asset]
                
                # Check if proxy is in available data
                if not features.prices or proxy not in features.prices:
                    logging.debug(f"{self.name}: {proxy} not in available prices, skipping")
                    continue
                
                # Long positive carry, short negative carry
                if carry_signal > 0.5:  # Threshold for positive carry
                    weights[proxy] = self.position_size
                elif carry_signal < -0.5:  # Strong negative carry
                    weights[proxy] = -self.position_size * 0.5  # Smaller short
                # Else flat
            
            if not weights:
                return SignalOutput(
                    strategy_name=self.name,
                    timestamp=t,
                    confidence=0.1,
                    explanation={"message": "No strong carry signals or missing ETF data"}
                )
            
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                desired_weights=weights,
                expected_return=0.05,
                risk_estimate=0.08,
                confidence=0.5,
                regime_fit=0.6,
                diversification_score=0.8,
                explanation={
                    "type": "Futures Carry (Backtest Only)",
                    "carry_signals": {k: f"{v:.2f}" for k, v in carry.items()},
                    "positions": len(weights),
                }
            )
        except Exception as e:
            logging.warning(f"{self.name} error: {e}")
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": f"Strategy disabled: {str(e)[:50]}"}
            )


class FuturesMacroOverlay(Strategy):
    """
    Macro Overlay Strategy using futures proxies (Backtest Only).
    
    Allocates across equity/bond/commodity based on macro regime:
    - Risk-on: Long equities, short bonds
    - Risk-off: Short equities, long bonds, long gold
    - Inflationary: Long commodities, short bonds
    - Deflationary: Long bonds, short commodities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Futures_Macro", config)
        
        self.max_position = config.get('max_position', 0.20) if config else 0.20
        
        # Proxy ETFs
        self.proxies = {
            'equity': 'SPY',
            'bonds': 'TLT',
            'gold': 'GLD',
            'commodities': 'DBC',
        }
        
        self._required_features = ['regime', 'prices']
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate macro overlay signals based on regime with graceful error handling."""
        try:
            weights = {}
            regime = features.regime if hasattr(features, 'regime') else None
            
            # Check if proxies are available in the data
            available_proxies = {k: v for k, v in self.proxies.items() 
                                 if features.prices and v in features.prices}
            
            if not available_proxies:
                return SignalOutput(
                    strategy_name=self.name,
                    timestamp=t,
                    confidence=0.0,
                    explanation={"error": "No ETF proxies available in data"}
                )
            
            # Determine regime allocation
            if regime:
                trend_strength = regime.trend_strength if hasattr(regime, 'trend_strength') else 0.5
                vol_percentile = regime.volatility_percentile if hasattr(regime, 'volatility_percentile') else 50
                
                # Risk-on/off based on trend and vol
                if trend_strength > 0.5 and vol_percentile < 60:
                    # Risk-on: Long equity, short bonds
                    if 'equity' in available_proxies:
                        weights[self.proxies['equity']] = self.max_position
                    if 'bonds' in available_proxies:
                        weights[self.proxies['bonds']] = -self.max_position * 0.5
                    regime_type = "risk_on"
                    
                elif vol_percentile > 70:
                    # Risk-off: Short equity, long bonds, long gold
                    if 'equity' in available_proxies:
                        weights[self.proxies['equity']] = -self.max_position * 0.5
                    if 'bonds' in available_proxies:
                        weights[self.proxies['bonds']] = self.max_position
                    if 'gold' in available_proxies:
                        weights[self.proxies['gold']] = self.max_position * 0.5
                    regime_type = "risk_off"
                    
                else:
                    # Neutral
                    if 'equity' in available_proxies:
                        weights[self.proxies['equity']] = self.max_position * 0.3
                    if 'bonds' in available_proxies:
                        weights[self.proxies['bonds']] = self.max_position * 0.3
                    regime_type = "neutral"
            else:
                # Default balanced allocation
                if 'equity' in available_proxies:
                    weights[self.proxies['equity']] = self.max_position * 0.4
                if 'bonds' in available_proxies:
                    weights[self.proxies['bonds']] = self.max_position * 0.4
                regime_type = "default"
            
            # Use macro features if available
            if self.macro_features:
                # Adjust based on inflation expectation
                inflation_pressure = self.macro_features.get('macro_inflation_pressure_index', 0)
                
                if inflation_pressure > 0.5 and 'commodities' in available_proxies:
                    # Inflationary: Add commodities, reduce bonds
                    weights[self.proxies['commodities']] = weights.get(self.proxies['commodities'], 0) + self.max_position * 0.3
                    if 'bonds' in weights:
                        weights[self.proxies['bonds']] = weights.get(self.proxies['bonds'], 0) * 0.5
                elif inflation_pressure < -0.5 and 'bonds' in available_proxies:
                    # Deflationary: Add bonds, reduce commodities
                    weights[self.proxies['bonds']] = weights.get(self.proxies['bonds'], 0) + self.max_position * 0.3
                    if 'commodities' in weights:
                        weights[self.proxies['commodities']] = weights.get(self.proxies['commodities'], 0) * 0.5
            
            # Clean up small weights
            weights = {k: v for k, v in weights.items() if abs(v) > 0.01}
            
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                desired_weights=weights,
                expected_return=0.06,
                risk_estimate=0.10,
                confidence=0.55 if weights else 0.0,
                regime_fit=0.8,
                diversification_score=0.9,
                explanation={
                    "type": "Futures Macro Overlay (Backtest Only)",
                    "regime": regime_type,
                    "allocations": {k: f"{v:.1%}" for k, v in weights.items()},
                }
            )
        except Exception as e:
            logging.warning(f"{self.name} error: {e}")
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": f"Strategy disabled: {str(e)[:50]}"}
            )


class FuturesTrendFollowing(Strategy):
    """
    Trend Following Strategy (CTA-style) using ETF proxies (Backtest Only).
    
    Goes long/short based on price trends across multiple asset classes.
    Position sizing based on volatility targeting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Futures_Trend", config)
        
        self.vol_target = config.get('vol_target', 0.10) if config else 0.10
        self.max_position = config.get('max_position', 0.25) if config else 0.25
        self.lookback = config.get('lookback', 126) if config else 126
        
        # Assets to trade
        self.assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'USO']
        
        self._required_features = ['returns_126d', 'volatility_21d', 'prices']
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate trend following signals with graceful error handling."""
        try:
            momentum = features.returns_126d if hasattr(features, 'returns_126d') else {}
            volatility = features.volatility_21d if hasattr(features, 'volatility_21d') else {}
            
            if not momentum:
                return SignalOutput(
                    strategy_name=self.name,
                    timestamp=t,
                    confidence=0.0,
                    explanation={"error": "No momentum data available"}
                )
            
            # Only use assets that are in the data
            available_assets = [a for a in self.assets 
                               if features.prices and a in features.prices]
            
            if not available_assets:
                return SignalOutput(
                    strategy_name=self.name,
                    timestamp=t,
                    confidence=0.0,
                    explanation={"error": "No ETF assets available in data"}
                )
            
            weights = {}
            
            for asset in available_assets:
                mom = momentum.get(asset)
                vol = volatility.get(asset, 0.15)
                
                if mom is None or vol is None:
                    continue
                
                # Direction from trend
                direction = 1 if mom > 0 else -1
                
                # Size based on vol targeting
                vol_scalar = self.vol_target / vol if vol > 0 else 0.5
                
                # Trend strength
                strength = min(1.0, abs(mom) / 0.2)  # Full strength at 20% momentum
                
                # Final weight
                raw_weight = direction * vol_scalar * strength * self.max_position
                weight = max(-self.max_position, min(self.max_position, raw_weight))
                
                if abs(weight) > 0.01:
                    weights[asset] = weight
            
            if not weights:
                return SignalOutput(
                    strategy_name=self.name,
                    timestamp=t,
                    confidence=0.0,
                    explanation={"error": "No valid trend signals"}
                )
            
            n_long = sum(1 for w in weights.values() if w > 0)
            n_short = sum(1 for w in weights.values() if w < 0)
            gross = sum(abs(w) for w in weights.values())
            net = sum(weights.values())
            
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                desired_weights=weights,
                expected_return=0.08,
                risk_estimate=self.vol_target,
                confidence=0.6,
                regime_fit=0.7,  # Trend following works in trending markets
                diversification_score=0.8,
                explanation={
                    "type": "Futures Trend Following (Backtest Only)",
                    "n_long": n_long,
                    "n_short": n_short,
                    "gross_exposure": f"{gross:.1%}",
                    "net_exposure": f"{net:.1%}",
                }
            )
        except Exception as e:
            logging.warning(f"{self.name} error: {e}")
            return SignalOutput(
                strategy_name=self.name,
                timestamp=t,
                confidence=0.0,
                explanation={"error": f"Strategy disabled: {str(e)[:50]}"}
            )


# === FACTORY FUNCTION ===

def create_futures_strategies(config: Optional[Dict[str, Any]] = None) -> List[Strategy]:
    """
    Create all futures strategies.
    
    NOTE: These are BACKTEST ONLY strategies that use ETF proxies.
    Live futures trading requires a broker that supports futures.
    
    Args:
        config: Optional configuration dict
    
    Returns:
        List of futures strategy instances
    """
    config = config or {}
    
    return [
        FuturesCarryStrategy(config.get('carry', {})),
        FuturesMacroOverlay(config.get('macro', {})),
        FuturesTrendFollowing(config.get('trend', {})),
    ]
