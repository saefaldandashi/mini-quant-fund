"""
Order Flow / Market Microstructure Strategy.
Analyzes order imbalances, volume patterns, and market depth for signals.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .base import Strategy, SignalOutput
from src.data.feature_store import Features

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowData:
    """Order flow data for a symbol."""
    symbol: str
    volume: float
    avg_volume: float
    volume_ratio: float
    bid_volume: float
    ask_volume: float
    order_imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    vwap_deviation: float   # Current price vs VWAP
    spread_bps: float       # Bid-ask spread in basis points
    
    @property
    def signal_strength(self) -> float:
        """Calculate overall signal strength."""
        # Combine imbalance and volume ratio
        strength = abs(self.order_imbalance) * min(self.volume_ratio, 2.0)
        return np.tanh(strength)  # Bounded to -1 to 1


class OrderFlowStrategy(Strategy):
    """
    Order Flow Strategy.
    
    Uses market microstructure signals:
    1. Order Imbalance - net buying vs selling pressure
    2. Volume Anomalies - unusual volume relative to average
    3. VWAP Deviation - price vs volume-weighted average
    4. Spread Analysis - liquidity conditions
    5. Time-of-Day Patterns - opening/closing auction effects
    
    Note: This strategy works best with intraday data.
    Without real-time order book data, it uses proxies from price/volume.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("OrderFlow", config)
        self._required_features = ['prices', 'volumes', 'volatility_21d']
        
        config = config or {}
        self.volume_threshold = config.get('volume_threshold', 1.5)  # 1.5x avg volume
        self.imbalance_threshold = config.get('imbalance_threshold', 0.3)  # 30% imbalance
        self.vwap_deviation_threshold = config.get('vwap_deviation_threshold', 0.01)  # 1%
        self.max_positions = config.get('max_positions', 8)
        self.position_size = config.get('position_size', 0.08)  # 8% per position
        
        # Historical data for calculations
        self._volume_history: Dict[str, List[float]] = {}
        self._price_history: Dict[str, List[float]] = {}
        self._volume_x_price_history: Dict[str, List[float]] = {}  # For VWAP
        
        # Track signals
        self._last_signals: Dict[str, Tuple[str, datetime]] = {}  # symbol -> (direction, time)
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate order flow based signals."""
        
        # Update history
        self._update_history(features)
        
        # Calculate order flow metrics for each symbol
        flow_data = self._calculate_order_flow(features)
        
        if not flow_data:
            return self._empty_signal(t)
        
        # Generate signals
        signals = []
        for symbol, data in flow_data.items():
            signal = self._analyze_flow(data, features, t)
            if signal:
                signals.append((symbol, data, signal))
        
        # Prioritize by signal strength
        signals.sort(key=lambda x: abs(x[1].signal_strength), reverse=True)
        
        # Take top N
        selected = signals[:self.max_positions]
        
        # Construct weights
        weights = {}
        expected_returns = {}
        
        for symbol, data, signal_type in selected:
            if signal_type == 'buy':
                weights[symbol] = self.position_size
                expected_returns[symbol] = data.signal_strength * 0.02
            elif signal_type == 'sell':
                weights[symbol] = -self.position_size
                expected_returns[symbol] = data.signal_strength * 0.02
            
            # Track signal
            self._last_signals[symbol] = (signal_type, t)
        
        if not weights:
            return self._empty_signal(t)
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        # Confidence based on signal quality
        avg_strength = np.mean([s[1].signal_strength for s in selected])
        confidence = 0.4 + avg_strength * 0.3
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=confidence,
            explanation={
                'active_signals': len(selected),
                'signals': [
                    {
                        'symbol': s[0],
                        'direction': s[2],
                        'volume_ratio': round(s[1].volume_ratio, 2),
                        'imbalance': round(s[1].order_imbalance, 3),
                        'vwap_dev': round(s[1].vwap_deviation, 4),
                        'strength': round(s[1].signal_strength, 3),
                    }
                    for s in selected
                ],
            },
            regime_fit=0.6,
            diversification_score=0.7,
        )
    
    def _update_history(self, features: Features) -> None:
        """Update rolling price and volume history."""
        for symbol in features.symbols:
            if symbol not in self._volume_history:
                self._volume_history[symbol] = []
                self._price_history[symbol] = []
                self._volume_x_price_history[symbol] = []
            
            price = features.prices.get(symbol)
            volume = features.volumes.get(symbol) if hasattr(features, 'volumes') else None
            
            if price:
                self._price_history[symbol].append(price)
                # Keep 30 periods
                if len(self._price_history[symbol]) > 30:
                    self._price_history[symbol] = self._price_history[symbol][-30:]
            
            if volume:
                self._volume_history[symbol].append(volume)
                if len(self._volume_history[symbol]) > 30:
                    self._volume_history[symbol] = self._volume_history[symbol][-30:]
                
                if price:
                    self._volume_x_price_history[symbol].append(volume * price)
                    if len(self._volume_x_price_history[symbol]) > 30:
                        self._volume_x_price_history[symbol] = self._volume_x_price_history[symbol][-30:]
    
    def _calculate_order_flow(self, features: Features) -> Dict[str, OrderFlowData]:
        """Calculate order flow metrics for all symbols."""
        results = {}
        
        for symbol in features.symbols:
            price = features.prices.get(symbol)
            if not price:
                continue
            
            # Get volume data
            volume_history = self._volume_history.get(symbol, [])
            price_history = self._price_history.get(symbol, [])
            vwap_history = self._volume_x_price_history.get(symbol, [])
            
            # Need at least some history
            if len(volume_history) < 5 or len(price_history) < 5:
                continue
            
            current_volume = volume_history[-1] if volume_history else 1
            avg_volume = np.mean(volume_history[:-1]) if len(volume_history) > 1 else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Estimate order imbalance from price movement + volume
            # Price up + high volume = buying pressure, Price down + high volume = selling
            if len(price_history) >= 2:
                price_change = (price_history[-1] - price_history[-2]) / price_history[-2]
                
                # Volume-weighted price change as imbalance proxy
                imbalance = np.tanh(price_change * volume_ratio * 10)
            else:
                imbalance = 0.0
            
            # VWAP calculation
            if len(vwap_history) >= 5 and sum(volume_history[-5:]) > 0:
                vwap = sum(vwap_history[-5:]) / sum(volume_history[-5:])
                vwap_deviation = (price - vwap) / vwap
            else:
                vwap_deviation = 0.0
            
            # Estimate bid/ask volumes (proxy)
            # If price went up, assume more bid volume
            if len(price_history) >= 2:
                if price_history[-1] > price_history[-2]:
                    bid_vol = current_volume * 0.6
                    ask_vol = current_volume * 0.4
                else:
                    bid_vol = current_volume * 0.4
                    ask_vol = current_volume * 0.6
            else:
                bid_vol = current_volume * 0.5
                ask_vol = current_volume * 0.5
            
            # Spread estimation (use volatility as proxy)
            volatility = features.volatility_21d.get(symbol, 0.02)
            spread_bps = volatility * 20  # Rough approximation
            
            results[symbol] = OrderFlowData(
                symbol=symbol,
                volume=current_volume,
                avg_volume=avg_volume,
                volume_ratio=volume_ratio,
                bid_volume=bid_vol,
                ask_volume=ask_vol,
                order_imbalance=imbalance,
                vwap_deviation=vwap_deviation,
                spread_bps=spread_bps,
            )
        
        return results
    
    def _analyze_flow(
        self,
        data: OrderFlowData,
        features: Features,
        t: datetime
    ) -> Optional[str]:
        """Analyze order flow data and generate signal."""
        
        # Check volume threshold
        if data.volume_ratio < self.volume_threshold:
            return None  # Not enough volume
        
        # Strong imbalance signals
        if data.order_imbalance > self.imbalance_threshold:
            # Strong buying pressure
            return 'buy'
        elif data.order_imbalance < -self.imbalance_threshold:
            # Strong selling pressure
            return 'sell'
        
        # VWAP reversion signals
        if data.vwap_deviation < -self.vwap_deviation_threshold and data.volume_ratio > 1.3:
            # Price below VWAP with high volume - potential buy
            return 'buy'
        elif data.vwap_deviation > self.vwap_deviation_threshold and data.volume_ratio > 1.3:
            # Price above VWAP with high volume - could go either way
            # Momentum continuation or exhaustion
            if data.order_imbalance > 0:
                return 'buy'  # Continuation
            else:
                return 'sell'  # Exhaustion
        
        return None
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'note': 'No order flow signals'},
        )
    
    def _calculate_expected_return(
        self,
        weights: Dict[str, float],
        expected_returns: Dict[str, float]
    ) -> float:
        return sum(
            weights.get(s, 0) * expected_returns.get(s, 0)
            for s in weights
        )
    
    def _calculate_risk(
        self,
        weights: Dict[str, float],
        cov_matrix
    ) -> float:
        if cov_matrix is None:
            return 0.15
        
        try:
            syms = list(weights.keys())
            valid_syms = [s for s in syms if s in cov_matrix.columns]
            if not valid_syms:
                return 0.15
            
            w = np.array([weights.get(s, 0) for s in valid_syms])
            cov = cov_matrix.loc[valid_syms, valid_syms].values
            var = w.T @ cov @ w
            
            return float(np.sqrt(var))
        except Exception:
            return 0.15
