"""
Statistical Arbitrage / Pairs Trading Strategy.
Identifies cointegrated pairs and trades mean reversion of the spread.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from .base import Strategy, SignalOutput
from src.data.feature_store import Features

logger = logging.getLogger(__name__)


@dataclass
class PairStats:
    """Statistics for a trading pair."""
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    zscore: float
    correlation: float
    half_life: float
    is_cointegrated: bool


class PairsTradingStrategy(Strategy):
    """
    Statistical Arbitrage / Pairs Trading Strategy.
    
    Identifies pairs with:
    1. High correlation
    2. Cointegration (statistically stable spread)
    3. Mean-reverting spreads
    
    Trades when spread deviates from mean (z-score > threshold).
    """
    
    # Pre-defined pairs with historically stable relationships
    DEFAULT_PAIRS = [
        # Tech
        ('MSFT', 'AAPL'),
        ('GOOGL', 'META'),
        ('AMD', 'NVDA'),
        ('CRM', 'ADBE'),
        # Finance
        ('JPM', 'BAC'),
        ('GS', 'MS'),
        ('V', 'MA'),
        # Consumer
        ('KO', 'PEP'),
        ('HD', 'LOW'),
        ('MCD', 'YUM'),
        ('WMT', 'TGT'),
        # Energy
        ('XOM', 'CVX'),
        # Healthcare
        ('UNH', 'CVS'),
        ('JNJ', 'PFE'),
        ('ABBV', 'MRK'),
        # Industrial
        ('CAT', 'DE'),
        ('BA', 'LMT'),
        # Telecom
        ('T', 'VZ'),
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("PairsTrading", config)
        self._required_features = ['prices', 'returns_21d', 'covariance_matrix']
        
        # Config
        config = config or {}
        self.entry_zscore = config.get('entry_zscore', 2.0)  # Enter when |z| > 2
        self.exit_zscore = config.get('exit_zscore', 0.5)    # Exit when |z| < 0.5
        self.stop_zscore = config.get('stop_zscore', 3.5)    # Stop loss at |z| > 3.5
        self.lookback = config.get('lookback', 60)           # Days for stats
        self.min_correlation = config.get('min_correlation', 0.7)
        self.max_pairs = config.get('max_pairs', 5)          # Max simultaneous pairs
        self.weight_per_pair = config.get('weight_per_pair', 0.15)  # 15% per pair (7.5% each leg)
        
        # Custom pairs or use defaults
        self.pairs = config.get('pairs', self.DEFAULT_PAIRS)
        
        # State: Track active pair positions
        self.active_positions: Dict[Tuple[str, str], Dict] = {}
        
        # Cache for pair statistics
        self._pair_stats: Dict[Tuple[str, str], PairStats] = {}
        self._price_history: Dict[str, List[float]] = {}
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate pairs trading signals."""
        weights = {}
        expected_returns = {}
        
        # Update price history
        self._update_price_history(features)
        
        # Calculate pair statistics
        valid_pairs = self._calculate_pair_stats(features)
        
        if not valid_pairs:
            return self._empty_signal(t)
        
        # Generate signals for each pair
        signals = []
        for pair_key, stats in valid_pairs.items():
            signal = self._analyze_pair(stats)
            if signal:
                signals.append((pair_key, stats, signal))
        
        # Prioritize by z-score magnitude (further from mean = stronger signal)
        signals.sort(key=lambda x: abs(x[1].zscore), reverse=True)
        
        # Take top N pairs
        selected_signals = signals[:self.max_pairs]
        
        # Construct portfolio weights
        for pair_key, stats, signal_type in selected_signals:
            symbol_a, symbol_b = pair_key
            
            # Weight per leg (half of pair weight)
            leg_weight = self.weight_per_pair / 2
            
            if signal_type == 'long_spread':
                # Long A, Short B (spread is below mean)
                weights[symbol_a] = leg_weight
                weights[symbol_b] = -leg_weight * stats.hedge_ratio
                
                # Expected return based on z-score
                exp_ret = abs(stats.zscore) * stats.spread_std / stats.spread_mean * 0.5
                expected_returns[symbol_a] = exp_ret / 2
                expected_returns[symbol_b] = exp_ret / 2
                
            elif signal_type == 'short_spread':
                # Short A, Long B (spread is above mean)
                weights[symbol_a] = -leg_weight
                weights[symbol_b] = leg_weight * stats.hedge_ratio
                
                exp_ret = abs(stats.zscore) * stats.spread_std / stats.spread_mean * 0.5
                expected_returns[symbol_a] = exp_ret / 2
                expected_returns[symbol_b] = exp_ret / 2
            
            # Track position
            self.active_positions[pair_key] = {
                'signal': signal_type,
                'entry_zscore': stats.zscore,
                'entry_time': t,
            }
        
        if not weights:
            return self._empty_signal(t)
        
        # Calculate portfolio metrics
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        # Confidence based on number of pairs and their z-scores
        avg_zscore = np.mean([abs(s[1].zscore) for s in selected_signals])
        confidence = min(0.8, 0.4 + avg_zscore * 0.1)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=confidence,
            explanation={
                'active_pairs': len(selected_signals),
                'pairs': [
                    {
                        'pair': f"{s[0][0]}/{s[0][1]}",
                        'zscore': round(s[1].zscore, 2),
                        'signal': s[2],
                        'correlation': round(s[1].correlation, 2),
                    }
                    for s in selected_signals
                ],
                'avg_zscore': round(avg_zscore, 2),
            },
            regime_fit=0.7,  # Pairs trading works in most regimes
            diversification_score=0.9,  # Hedged positions
        )
    
    def _update_price_history(self, features: Features) -> None:
        """Update rolling price history."""
        for symbol in features.symbols:
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            
            price = features.prices.get(symbol)
            if price:
                self._price_history[symbol].append(price)
                # Keep only lookback + buffer
                if len(self._price_history[symbol]) > self.lookback + 10:
                    self._price_history[symbol] = self._price_history[symbol][-self.lookback - 10:]
    
    def _calculate_pair_stats(self, features: Features) -> Dict[Tuple[str, str], PairStats]:
        """Calculate statistics for all pairs."""
        valid_pairs = {}
        
        for symbol_a, symbol_b in self.pairs:
            # Check if both symbols available
            if symbol_a not in features.symbols or symbol_b not in features.symbols:
                continue
            
            # Need sufficient history
            history_a = self._price_history.get(symbol_a, [])
            history_b = self._price_history.get(symbol_b, [])
            
            min_len = min(len(history_a), len(history_b))
            if min_len < self.lookback:
                # Use returns data instead if available
                ret_a = features.returns_21d.get(symbol_a, {})
                ret_b = features.returns_21d.get(symbol_b, {})
                
                if not ret_a or not ret_b:
                    continue
                
                # Simple stats without full history
                price_a = features.prices.get(symbol_a, 0)
                price_b = features.prices.get(symbol_b, 0)
                
                if price_a == 0 or price_b == 0:
                    continue
                
                # Estimate hedge ratio from prices
                hedge_ratio = price_a / price_b
                spread = price_a - hedge_ratio * price_b
                
                # Use correlation from returns if available
                if features.covariance_matrix is not None:
                    cov_matrix = features.covariance_matrix
                    if symbol_a in cov_matrix.columns and symbol_b in cov_matrix.columns:
                        cov_ab = cov_matrix.loc[symbol_a, symbol_b]
                        var_a = cov_matrix.loc[symbol_a, symbol_a]
                        var_b = cov_matrix.loc[symbol_b, symbol_b]
                        correlation = cov_ab / np.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0
                    else:
                        correlation = 0.5
                else:
                    correlation = 0.5
                
                # Default stats for new pairs
                stats = PairStats(
                    symbol_a=symbol_a,
                    symbol_b=symbol_b,
                    hedge_ratio=hedge_ratio,
                    spread_mean=0.0,
                    spread_std=1.0,
                    zscore=0.0,
                    correlation=correlation,
                    half_life=10.0,
                    is_cointegrated=correlation > self.min_correlation,
                )
                
                if correlation >= self.min_correlation:
                    valid_pairs[(symbol_a, symbol_b)] = stats
                    
                continue
            
            # Calculate with full history
            prices_a = np.array(history_a[-min_len:])
            prices_b = np.array(history_b[-min_len:])
            
            # Calculate hedge ratio via OLS
            hedge_ratio = np.cov(prices_a, prices_b)[0, 1] / np.var(prices_b)
            
            # Calculate spread
            spread = prices_a - hedge_ratio * prices_b
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            
            # Current z-score
            current_spread = prices_a[-1] - hedge_ratio * prices_b[-1]
            zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Correlation
            correlation = np.corrcoef(prices_a, prices_b)[0, 1]
            
            # Estimate half-life of mean reversion (using Ornstein-Uhlenbeck)
            half_life = self._estimate_half_life(spread)
            
            # Simple cointegration test (ADF would be better but keeping it simple)
            is_cointegrated = (
                correlation >= self.min_correlation and
                half_life > 1 and half_life < 50
            )
            
            if is_cointegrated:
                stats = PairStats(
                    symbol_a=symbol_a,
                    symbol_b=symbol_b,
                    hedge_ratio=hedge_ratio,
                    spread_mean=spread_mean,
                    spread_std=spread_std,
                    zscore=zscore,
                    correlation=correlation,
                    half_life=half_life,
                    is_cointegrated=is_cointegrated,
                )
                valid_pairs[(symbol_a, symbol_b)] = stats
                self._pair_stats[(symbol_a, symbol_b)] = stats
        
        return valid_pairs
    
    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """Estimate half-life of mean reversion using Ornstein-Uhlenbeck."""
        try:
            # Delta spread
            delta = np.diff(spread)
            spread_lag = spread[:-1]
            
            # OLS: delta = a + b * spread_lag
            X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
            beta = np.linalg.lstsq(X, delta, rcond=None)[0]
            
            # Half-life = -log(2) / beta[1]
            if beta[1] < 0:
                half_life = -np.log(2) / beta[1]
            else:
                half_life = 100  # Not mean-reverting
            
            return half_life
        except Exception:
            return 100  # Default to non-mean-reverting
    
    def _analyze_pair(self, stats: PairStats) -> Optional[str]:
        """Analyze pair and generate signal."""
        pair_key = (stats.symbol_a, stats.symbol_b)
        
        # Check existing position
        if pair_key in self.active_positions:
            pos = self.active_positions[pair_key]
            
            # Check for exit
            if abs(stats.zscore) < self.exit_zscore:
                # Close position
                del self.active_positions[pair_key]
                return None
            
            # Check for stop loss
            if abs(stats.zscore) > self.stop_zscore:
                # Stop out
                del self.active_positions[pair_key]
                return None
            
            # Maintain position
            return pos['signal']
        
        # No existing position - check for entry
        if stats.zscore > self.entry_zscore:
            # Spread is above mean - short the spread (short A, long B)
            return 'short_spread'
        elif stats.zscore < -self.entry_zscore:
            # Spread is below mean - long the spread (long A, short B)
            return 'long_spread'
        
        return None
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'note': 'No valid pairs trading opportunities'},
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
        cov_matrix: Optional[pd.DataFrame]
    ) -> float:
        if cov_matrix is None:
            return 0.1
        
        try:
            syms = list(weights.keys())
            valid_syms = [s for s in syms if s in cov_matrix.columns]
            if not valid_syms:
                return 0.1
            
            w = np.array([weights.get(s, 0) for s in valid_syms])
            cov = cov_matrix.loc[valid_syms, valid_syms].values
            var = w.T @ cov @ w
            
            return float(np.sqrt(var))
        except Exception:
            return 0.1
