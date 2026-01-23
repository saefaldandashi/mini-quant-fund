"""
Dynamic Strategy Weight Optimizer

Optimizes strategy weights in real-time based on:
1. Rolling performance (momentum in strategy returns)
2. Regime fit (which strategies work in current market)
3. Correlation between strategies (diversification)
4. Drawdown control (reduce failing strategies faster)
5. Cross-asset signals (incorporate market-wide signals)

The goal is to allocate more to what's working, less to what isn't,
while maintaining diversification.
"""

import logging
import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformanceSnapshot:
    """Performance snapshot for a strategy."""
    strategy_name: str
    
    # Returns at different horizons
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_20d: float = 0.0
    
    # Risk metrics
    volatility_20d: float = 0.1
    sharpe_20d: float = 0.0
    max_drawdown_20d: float = 0.0
    
    # Win rate
    win_rate_20d: float = 0.5
    
    # Regime performance
    current_regime_return: float = 0.0
    regime_fit_score: float = 0.5
    
    # Correlation with other strategies
    avg_correlation: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WeightOptimizationResult:
    """Result of weight optimization."""
    original_weights: Dict[str, float]
    optimized_weights: Dict[str, float]
    
    # Adjustments made
    adjustments: Dict[str, float] = field(default_factory=dict)  # strategy -> adjustment factor
    
    # Metrics
    expected_sharpe_improvement: float = 0.0
    diversification_score: float = 0.0
    
    # Reasoning
    rationale: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "original": self.original_weights,
            "optimized": self.optimized_weights,
            "adjustments": self.adjustments,
            "expected_sharpe_improvement": self.expected_sharpe_improvement,
            "diversification_score": self.diversification_score,
            "rationale": self.rationale,
        }


class DynamicWeightOptimizer:
    """
    Dynamically optimizes strategy weights based on recent performance
    and market conditions.
    
    Key Features:
    1. Momentum-based weight adjustment (what's working)
    2. Mean-reversion adjustment (avoid chasing too much)
    3. Correlation penalty (diversification)
    4. Drawdown acceleration (cut losers faster)
    5. Regime conditioning (boost regime-fit strategies)
    """
    
    def __init__(
        self,
        strategy_names: List[str],
        storage_path: str = "outputs/dynamic_weights.json",
        # Optimization parameters
        momentum_lookback_days: int = 20,
        momentum_weight: float = 0.3,
        mean_reversion_weight: float = 0.1,
        correlation_penalty: float = 0.2,
        drawdown_acceleration: float = 2.0,
        min_weight: float = 0.02,
        max_weight: float = 0.40,
    ):
        self.strategy_names = strategy_names
        self.storage_path = Path(storage_path)
        
        # Parameters
        self.momentum_lookback = momentum_lookback_days
        self.momentum_weight = momentum_weight
        self.mean_reversion_weight = mean_reversion_weight
        self.correlation_penalty = correlation_penalty
        self.drawdown_acceleration = drawdown_acceleration
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # State
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self.daily_returns: Dict[str, List[float]] = defaultdict(list)
        self.current_weights: Dict[str, float] = {}
        self.optimization_history: List[Dict] = []
        
        self._load()
        self._initialize_weights()
    
    def _load(self):
        """Load saved state."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.current_weights = data.get('current_weights', {})
                self.optimization_history = data.get('optimization_history', [])[-100:]
                logger.info("Loaded dynamic weight optimizer state")
            except Exception as e:
                logger.warning(f"Could not load dynamic weights: {e}")
    
    def _save(self):
        """Save state."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'current_weights': self.current_weights,
                    'optimization_history': self.optimization_history[-100:],
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save dynamic weights: {e}")
    
    def _initialize_weights(self):
        """Initialize equal weights if not set."""
        if not self.current_weights:
            equal_weight = 1.0 / len(self.strategy_names) if self.strategy_names else 0.1
            self.current_weights = {name: equal_weight for name in self.strategy_names}
    
    def record_daily_return(
        self,
        strategy_name: str,
        daily_return: float,
        regime: Optional[str] = None,
    ):
        """Record a strategy's daily return."""
        self.daily_returns[strategy_name].append(daily_return)
        
        # Keep only last 60 days
        if len(self.daily_returns[strategy_name]) > 60:
            self.daily_returns[strategy_name] = self.daily_returns[strategy_name][-60:]
        
        # Record performance snapshot
        self.performance_history[strategy_name].append({
            'return': daily_return,
            'regime': regime,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Keep only last 100 entries
        if len(self.performance_history[strategy_name]) > 100:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-100:]
    
    def get_performance_snapshot(
        self,
        strategy_name: str,
    ) -> StrategyPerformanceSnapshot:
        """Get current performance snapshot for a strategy."""
        returns = self.daily_returns.get(strategy_name, [])
        
        if len(returns) < 5:
            return StrategyPerformanceSnapshot(strategy_name=strategy_name)
        
        returns_array = np.array(returns)
        
        # Calculate metrics
        return_1d = returns_array[-1] if len(returns_array) >= 1 else 0
        return_5d = np.sum(returns_array[-5:]) if len(returns_array) >= 5 else 0
        return_20d = np.sum(returns_array[-20:]) if len(returns_array) >= 20 else np.sum(returns_array)
        
        # Volatility
        vol = np.std(returns_array[-20:]) * np.sqrt(252) if len(returns_array) >= 20 else 0.1
        
        # Sharpe (annualized)
        mean_return = np.mean(returns_array[-20:]) * 252 if len(returns_array) >= 20 else 0
        sharpe = mean_return / vol if vol > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns_array[-20:]) if len(returns_array) >= 20 else np.array([1])
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate
        wins = np.sum(returns_array[-20:] > 0) if len(returns_array) >= 20 else 0
        total = min(20, len(returns_array))
        win_rate = wins / total if total > 0 else 0.5
        
        return StrategyPerformanceSnapshot(
            strategy_name=strategy_name,
            return_1d=return_1d,
            return_5d=return_5d,
            return_20d=return_20d,
            volatility_20d=vol,
            sharpe_20d=sharpe,
            max_drawdown_20d=max_dd,
            win_rate_20d=win_rate,
        )
    
    def calculate_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between strategies."""
        correlations = {}
        
        for name1 in self.strategy_names:
            correlations[name1] = {}
            returns1 = self.daily_returns.get(name1, [])
            
            for name2 in self.strategy_names:
                if name1 == name2:
                    correlations[name1][name2] = 1.0
                    continue
                
                returns2 = self.daily_returns.get(name2, [])
                
                # Need at least 10 overlapping observations
                min_len = min(len(returns1), len(returns2))
                if min_len < 10:
                    correlations[name1][name2] = 0.0
                    continue
                
                # Calculate correlation
                r1 = np.array(returns1[-min_len:])
                r2 = np.array(returns2[-min_len:])
                
                corr = np.corrcoef(r1, r2)[0, 1]
                correlations[name1][name2] = corr if not np.isnan(corr) else 0.0
        
        return correlations
    
    def optimize_weights(
        self,
        base_weights: Dict[str, float],
        regime: Optional[str] = None,
        cross_asset_signals: Optional[List[Any]] = None,
    ) -> WeightOptimizationResult:
        """
        Optimize strategy weights based on recent performance.
        
        Steps:
        1. Calculate performance snapshots
        2. Apply momentum adjustment (boost winners)
        3. Apply mean-reversion adjustment (don't chase too much)
        4. Apply correlation penalty (diversify)
        5. Apply drawdown acceleration (cut losers faster)
        6. Normalize and enforce limits
        """
        rationale = []
        adjustments = {}
        
        # Get performance snapshots
        snapshots = {
            name: self.get_performance_snapshot(name)
            for name in self.strategy_names
        }
        
        # Get correlation matrix
        correlations = self.calculate_correlation_matrix()
        
        # Start with base weights
        optimized = base_weights.copy()
        
        # === 1. MOMENTUM ADJUSTMENT ===
        # Boost strategies with positive momentum
        for name, snapshot in snapshots.items():
            if name not in optimized:
                continue
            
            # Use 5-day return as momentum signal
            momentum_signal = snapshot.return_5d
            
            # Scale to adjustment factor (e.g., 5% return → 1.05x multiplier)
            momentum_adj = 1.0 + (momentum_signal * self.momentum_weight * 10)
            momentum_adj = max(0.7, min(1.5, momentum_adj))  # Cap at 70%-150%
            
            if abs(momentum_adj - 1.0) > 0.05:
                old_weight = optimized[name]
                optimized[name] = old_weight * momentum_adj
                adjustments[name] = adjustments.get(name, 1.0) * momentum_adj
                rationale.append(f"{name}: {momentum_adj:.2f}x momentum (5d ret: {snapshot.return_5d*100:.1f}%)")
        
        # === 2. MEAN REVERSION ADJUSTMENT ===
        # Slightly reduce extremely hot strategies
        for name, snapshot in snapshots.items():
            if name not in optimized:
                continue
            
            # If 20d return is very high, apply mean reversion
            if snapshot.return_20d > 0.10:  # 10%+ in 20 days is extreme
                reversion_factor = 1.0 - (snapshot.return_20d - 0.10) * self.mean_reversion_weight
                reversion_factor = max(0.8, reversion_factor)
                
                optimized[name] = optimized[name] * reversion_factor
                adjustments[name] = adjustments.get(name, 1.0) * reversion_factor
                rationale.append(f"{name}: {reversion_factor:.2f}x mean-reversion")
        
        # === 3. CORRELATION PENALTY ===
        # Penalize strategies highly correlated with larger strategies
        sorted_strategies = sorted(optimized.items(), key=lambda x: -x[1])
        
        for i, (name, weight) in enumerate(sorted_strategies):
            if i == 0:  # Skip largest
                continue
            
            # Check correlation with larger strategies
            for larger_name, larger_weight in sorted_strategies[:i]:
                corr = correlations.get(name, {}).get(larger_name, 0)
                
                if corr > 0.7:  # Highly correlated
                    penalty = 1.0 - (corr - 0.7) * self.correlation_penalty
                    penalty = max(0.7, penalty)
                    
                    optimized[name] = optimized[name] * penalty
                    adjustments[name] = adjustments.get(name, 1.0) * penalty
                    rationale.append(f"{name}: {penalty:.2f}x corr penalty (r={corr:.2f} with {larger_name})")
                    break  # Only apply once per strategy
        
        # === 4. DRAWDOWN ACCELERATION ===
        # Reduce strategies in drawdown faster
        for name, snapshot in snapshots.items():
            if name not in optimized:
                continue
            
            if snapshot.max_drawdown_20d > 0.05:  # 5%+ drawdown
                # Accelerate reduction
                dd_factor = 1.0 - (snapshot.max_drawdown_20d * self.drawdown_acceleration)
                dd_factor = max(0.5, dd_factor)  # Floor at 50%
                
                optimized[name] = optimized[name] * dd_factor
                adjustments[name] = adjustments.get(name, 1.0) * dd_factor
                rationale.append(f"{name}: {dd_factor:.2f}x drawdown cut (DD: {snapshot.max_drawdown_20d*100:.1f}%)")
        
        # === 5. REGIME FIT BONUS ===
        # Boost strategies that historically perform well in current regime
        if regime:
            for name in optimized:
                regime_perf = self._get_regime_performance(name, regime)
                
                if regime_perf > 0.02:  # Good in this regime
                    bonus = 1.0 + (regime_perf * 2)
                    bonus = min(1.3, bonus)
                    
                    optimized[name] = optimized[name] * bonus
                    adjustments[name] = adjustments.get(name, 1.0) * bonus
                    rationale.append(f"{name}: {bonus:.2f}x regime fit ({regime})")
        
        # === 6. NORMALIZE AND ENFORCE LIMITS ===
        # Enforce min/max weights
        for name in optimized:
            if abs(optimized[name]) < self.min_weight:
                optimized[name] = 0  # Below minimum → zero out
            elif abs(optimized[name]) > self.max_weight:
                # Cap at max
                optimized[name] = self.max_weight * (1 if optimized[name] > 0 else -1)
        
        # Normalize to sum to same total as base weights
        base_total = sum(base_weights.values())
        optimized_total = sum(optimized.values())
        
        if optimized_total > 0:
            scale = base_total / optimized_total
            optimized = {k: v * scale for k, v in optimized.items()}
        
        # Calculate metrics
        avg_corr = self._calculate_avg_correlation(optimized, correlations)
        div_score = 1.0 - avg_corr  # Higher diversity = lower correlation
        
        # Estimate sharpe improvement
        base_sharpe = self._estimate_portfolio_sharpe(base_weights, snapshots)
        opt_sharpe = self._estimate_portfolio_sharpe(optimized, snapshots)
        sharpe_improvement = opt_sharpe - base_sharpe
        
        result = WeightOptimizationResult(
            original_weights=base_weights,
            optimized_weights=optimized,
            adjustments=adjustments,
            expected_sharpe_improvement=sharpe_improvement,
            diversification_score=div_score,
            rationale=rationale[:10],  # Top 10 reasons
        )
        
        # Update current weights and save
        self.current_weights = optimized
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'sharpe_improvement': sharpe_improvement,
            'adjustments_count': len(adjustments),
        })
        self._save()
        
        logger.info(f"Optimized weights: {len(adjustments)} adjustments, "
                   f"Sharpe improvement: {sharpe_improvement:.3f}")
        
        return result
    
    def _get_regime_performance(self, strategy_name: str, regime: str) -> float:
        """Get average performance of strategy in a specific regime."""
        history = self.performance_history.get(strategy_name, [])
        
        regime_returns = [
            h['return'] for h in history
            if h.get('regime') == regime
        ]
        
        if not regime_returns:
            return 0.0
        
        return np.mean(regime_returns)
    
    def _calculate_avg_correlation(
        self,
        weights: Dict[str, float],
        correlations: Dict[str, Dict[str, float]],
    ) -> float:
        """Calculate weighted average correlation of portfolio."""
        if not weights:
            return 0.0
        
        total_corr = 0.0
        count = 0
        
        names = list(weights.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                w1 = abs(weights.get(name1, 0))
                w2 = abs(weights.get(name2, 0))
                corr = correlations.get(name1, {}).get(name2, 0)
                
                total_corr += w1 * w2 * corr
                count += 1
        
        return total_corr / count if count > 0 else 0.0
    
    def _estimate_portfolio_sharpe(
        self,
        weights: Dict[str, float],
        snapshots: Dict[str, StrategyPerformanceSnapshot],
    ) -> float:
        """Estimate portfolio Sharpe ratio from strategy snapshots."""
        if not weights:
            return 0.0
        
        # Weighted average Sharpe
        total_sharpe = 0.0
        total_weight = 0.0
        
        for name, weight in weights.items():
            if name in snapshots:
                total_sharpe += abs(weight) * snapshots[name].sharpe_20d
                total_weight += abs(weight)
        
        return total_sharpe / total_weight if total_weight > 0 else 0.0
    
    def get_summary(self) -> Dict:
        """Get summary of dynamic weight optimization."""
        return {
            "strategies_tracked": len(self.strategy_names),
            "current_weights": self.current_weights,
            "optimization_history_count": len(self.optimization_history),
            "last_optimization": self.optimization_history[-1] if self.optimization_history else None,
            "parameters": {
                "momentum_weight": self.momentum_weight,
                "mean_reversion_weight": self.mean_reversion_weight,
                "correlation_penalty": self.correlation_penalty,
                "drawdown_acceleration": self.drawdown_acceleration,
            },
        }


# Singleton instance
_dynamic_optimizer: Optional[DynamicWeightOptimizer] = None


def get_dynamic_optimizer(strategy_names: Optional[List[str]] = None) -> DynamicWeightOptimizer:
    """Get singleton instance of dynamic weight optimizer."""
    global _dynamic_optimizer
    if _dynamic_optimizer is None:
        _dynamic_optimizer = DynamicWeightOptimizer(strategy_names or [])
    return _dynamic_optimizer
