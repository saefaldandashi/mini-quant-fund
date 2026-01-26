"""
Ensemble optimizer for combining strategy signals.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from scipy.optimize import minimize
import logging

from src.strategies.base import SignalOutput
from src.data.feature_store import Features
from .debate_engine import StrategyScore

logger = logging.getLogger(__name__)


class EnsembleMode(Enum):
    """Ensemble combination modes."""
    WEIGHTED_VOTE = "weighted_vote"
    CONVEX_OPTIMIZATION = "convex_optimization"
    STACKING = "stacking"


class EnsembleOptimizer:
    """
    Combines strategy signals into final portfolio weights.
    Supports multiple ensemble modes with risk constraints.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Constraints
        self.max_position = self.config.get('max_position', 0.15)
        self.max_sector_exposure = self.config.get('max_sector_exposure', 0.30)
        self.max_leverage = self.config.get('max_leverage', 1.0)
        self.max_turnover = self.config.get('max_turnover', 0.50)
        self.vol_target = self.config.get('vol_target', 0.12)
        
        # Risk parameters
        self.risk_aversion = self.config.get('risk_aversion', 2.0)
        
        # Sector mapping (simplified)
        self.sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'META': 'tech',
            'AMZN': 'tech', 'NVDA': 'tech', 'NFLX': 'tech', 'AMD': 'tech',
            'JPM': 'finance', 'BAC': 'finance', 'GS': 'finance', 'MS': 'finance',
            'XOM': 'energy', 'CVX': 'energy',
            'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare',
            'KO': 'consumer', 'PEP': 'consumer', 'PG': 'consumer', 'WMT': 'consumer',
            # ETF proxies used by futures strategies
            'SPY': 'index', 'QQQ': 'index', 'IWM': 'index',
            'TLT': 'bonds', 'IEF': 'bonds',
            'GLD': 'commodities', 'SLV': 'commodities',
            'USO': 'commodities', 'DBC': 'commodities',
        }
        
        # Strategy type classification for signal conflict resolution
        self.strategy_types = {
            # Long-only strategies
            'TimeSeriesMomentum': 'equity_long',
            'CrossSectionMomentum': 'equity_long',
            'MeanReversion': 'equity_long',
            'VolatilityRegimeVolTarget': 'equity_long',
            'Carry': 'equity_long',
            'ValueQualityTilt': 'equity_long',
            'RiskParityMinVar': 'equity_long',
            'TailRiskOverlay': 'equity_long',
            'NewsSentimentEvent': 'equity_long',
            # Long/Short strategies
            'CS_Momentum_LS': 'equity_ls',
            'TS_Momentum_LS': 'equity_ls',
            'MeanReversion_LS': 'equity_ls',
            'QualityValue_LS': 'equity_ls',
            # Futures strategies (use ETF proxies)
            'Futures_Carry': 'futures_proxy',
            'Futures_Macro': 'futures_proxy',
            'Futures_Trend': 'futures_proxy',
        }
    
    def combine(
        self,
        signals: Dict[str, SignalOutput],
        scores: Dict[str, StrategyScore],
        features: Features,
        current_weights: Dict[str, float],
        mode: EnsembleMode = EnsembleMode.WEIGHTED_VOTE,
        strategy_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Combine strategy signals into final weights.
        
        Args:
            signals: Strategy signals
            scores: Strategy scores from debate
            features: Current features
            current_weights: Current portfolio weights
            mode: Ensemble mode
            strategy_weights: Optional learned weights for each strategy (from learning engine)
            
        Returns:
            Tuple of (final_weights, metadata)
        """
        # Calculate strategy correlation penalties to encourage diversification
        correlation_adjustments = self._calculate_strategy_correlations(signals, scores)
        
        # Apply correlation penalties to scores
        adjusted_scores = {}
        for name, score in scores.items():
            penalty = correlation_adjustments.get(name, 0.0)
            # Create a modified score with correlation penalty (copy all fields, adjust total_score)
            adjusted_scores[name] = StrategyScore(
                strategy_name=score.strategy_name,
                alpha_score=score.alpha_score,
                regime_fit_score=score.regime_fit_score,
                diversification_score=score.diversification_score,
                drawdown_score=score.drawdown_score,
                sentiment_score=score.sentiment_score,
                total_score=max(0.1, score.total_score * (1.0 - penalty)),
                rationale=score.rationale,
                strengths=score.strengths,
                weaknesses=score.weaknesses,
            )
        
        if mode == EnsembleMode.WEIGHTED_VOTE:
            raw_weights = self._weighted_vote(signals, adjusted_scores, strategy_weights)
        elif mode == EnsembleMode.CONVEX_OPTIMIZATION:
            raw_weights = self._convex_optimization(signals, adjusted_scores, features, strategy_weights)
        elif mode == EnsembleMode.STACKING:
            raw_weights = self._stacking(signals, adjusted_scores, features, strategy_weights)
        else:
            raw_weights = self._weighted_vote(signals, adjusted_scores, strategy_weights)
        
        # Apply constraints
        final_weights, constraints_applied = self._apply_constraints(
            raw_weights, features, current_weights
        )
        
        # Calculate metadata
        metadata = {
            'mode': mode.value,
            'raw_weights': raw_weights,
            'constraints_applied': constraints_applied,
            'strategy_contributions': self._calculate_contributions(signals, scores),
            'portfolio_vol': self._estimate_portfolio_vol(final_weights, features),
            'turnover': self._calculate_turnover(final_weights, current_weights),
        }
        
        return final_weights, metadata
    
    def _weighted_vote(
        self,
        signals: Dict[str, SignalOutput],
        scores: Dict[str, StrategyScore],
        strategy_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Simple weighted voting based on strategy scores.
        
        If strategy_weights are provided (from learning engine), 
        they are blended with debate scores for the final weighting.
        """
        combined = {}
        
        # Calculate effective weights for each strategy
        effective_weights = {}
        
        for name in signals:
            score = scores.get(name)
            if score is None:
                continue
            
            # Base weight from debate score
            debate_weight = score.total_score
            
            # Blend with learned weight if available
            if strategy_weights and name in strategy_weights:
                learned = strategy_weights[name]
                # Blend: 50% debate, 50% learned
                effective_weights[name] = 0.5 * debate_weight + 0.5 * (learned * sum(s.total_score for s in scores.values()))
            else:
                effective_weights[name] = debate_weight
        
        total_weight = sum(effective_weights.values())
        
        if total_weight <= 0:
            return {}
        
        # Track contributions by strategy type for conflict detection
        symbol_contributions = {}  # symbol -> list of (strategy_type, weight)
        
        for name, signal in signals.items():
            if name not in effective_weights:
                continue
            
            weight_factor = effective_weights[name] / total_weight
            strategy_type = self.strategy_types.get(name, 'unknown')
            
            for symbol, asset_weight in signal.desired_weights.items():
                weighted = weight_factor * asset_weight
                combined[symbol] = combined.get(symbol, 0) + weighted
                
                # Track for conflict detection
                if symbol not in symbol_contributions:
                    symbol_contributions[symbol] = []
                symbol_contributions[symbol].append({
                    'strategy': name,
                    'type': strategy_type,
                    'weight': weighted,
                })
        
        # Resolve conflicts where strategies disagree on direction
        combined = self._resolve_signal_conflicts(combined, symbol_contributions)
        
        return combined
    
    def _resolve_signal_conflicts(
        self,
        combined: Dict[str, float],
        contributions: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        Resolve conflicts when strategies disagree on direction.
        
        Rules:
        1. If L/S strategies have a strong short signal, preserve some short
        2. If only long-only strategies conflict with L/S shorts, L/S gets more weight
        3. Log conflicts for debugging
        
        NOTE: L/S strategies are designed to short, so their shorts should be 
        respected even when long-only strategies disagree.
        """
        resolved = dict(combined)
        
        for symbol, contribs in contributions.items():
            if len(contribs) < 2:
                continue
            
            # Check for directional conflict
            longs = [c for c in contribs if c['weight'] > 0]
            shorts = [c for c in contribs if c['weight'] < 0]
            
            if longs and shorts:
                # Conflict detected
                long_total = sum(c['weight'] for c in longs)
                short_total = sum(abs(c['weight']) for c in shorts)
                
                # Check if shorts are from L/S strategies (they should be respected)
                ls_shorts = [c for c in shorts if c['type'] == 'equity_ls']
                ls_short_weight = sum(abs(c['weight']) for c in ls_shorts)
                
                if ls_short_weight > 0:
                    # L/S strategies are shorting this stock
                    # Give them 2x weight in the conflict resolution
                    boosted_short = short_total + ls_short_weight  # Double the L/S short signal
                    net = long_total - boosted_short
                    
                    # If the net is now negative, keep it as a short
                    if net < 0:
                        resolved[symbol] = net * 0.8  # Slight discount but preserve short
                    else:
                        # Long still wins, but reduce position due to conflict
                        resolved[symbol] = net * 0.5
                else:
                    # No L/S short - regular conflict resolution
                    net = long_total - short_total
                    resolved[symbol] = net * 0.7
                
                logger.debug(
                    f"Signal conflict for {symbol}: "
                    f"long={long_total:.2%}, short={-short_total:.2%}, ls_short={-ls_short_weight:.2%} -> {resolved[symbol]:.2%}"
                )
        
        return resolved
    
    def _convex_optimization(
        self,
        signals: Dict[str, SignalOutput],
        scores: Dict[str, StrategyScore],
        features: Features,
        strategy_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Convex optimization to maximize score-weighted returns minus risk."""
        # Get all symbols
        all_symbols = set()
        for signal in signals.values():
            all_symbols.update(signal.desired_weights.keys())
        
        symbols = sorted(all_symbols)
        n = len(symbols)
        
        if n == 0:
            return {}
        
        # Build expected returns vector (score-weighted average)
        expected_returns = np.zeros(n)
        total_score = sum(s.total_score for s in scores.values())
        
        if total_score > 0:
            for name, signal in signals.items():
                score = scores.get(name, StrategyScore(
                    strategy_name=name,
                    alpha_score=0.0,
                    regime_fit_score=0.0,
                    diversification_score=0.0,
                    drawdown_score=0.0,
                    sentiment_score=0.0,
                    total_score=0.0,
                    rationale="Default score"
                ))
                weight = score.total_score / total_score
                
                for i, symbol in enumerate(symbols):
                    if symbol in signal.expected_returns_by_asset:
                        expected_returns[i] += weight * signal.expected_returns_by_asset[symbol]
        
        # Get covariance matrix
        if features.covariance_matrix is not None:
            cov_symbols = [s for s in symbols if s in features.covariance_matrix.columns]
            if len(cov_symbols) > 0:
                cov = features.covariance_matrix.loc[cov_symbols, cov_symbols].values
                # Pad for missing symbols
                full_cov = np.eye(n) * 0.04  # Default 20% vol
                for i, s in enumerate(symbols):
                    if s in cov_symbols:
                        for j, t in enumerate(symbols):
                            if t in cov_symbols:
                                ci = cov_symbols.index(s)
                                cj = cov_symbols.index(t)
                                full_cov[i, j] = cov[ci, cj]
                cov = full_cov
            else:
                cov = np.eye(n) * 0.04
        else:
            cov = np.eye(n) * 0.04
        
        # Optimization: maximize return - risk_aversion * variance
        def objective(w):
            ret = w @ expected_returns
            var = w @ cov @ w
            return -(ret - self.risk_aversion * var)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Fully invested
        ]
        
        bounds = [(0, self.max_position) for _ in range(n)]  # Long only, position limits
        
        # Initial guess: equal weight
        x0 = np.ones(n) / n
        
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 200}
            )
            
            if result.success:
                weights = result.x
            else:
                weights = x0
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            weights = x0
        
        # Build weight dict
        return {symbol: float(weights[i]) for i, symbol in enumerate(symbols) if weights[i] > 0.001}
    
    def _stacking(
        self,
        signals: Dict[str, SignalOutput],
        scores: Dict[str, StrategyScore],
        features: Features,
        learned_strategy_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Meta-model stacking approach.
        Uses strategy outputs as features to predict optimal weights.
        Simplified version: weighted by confidence and score.
        Incorporates learned weights if provided.
        """
        combined = {}
        
        # Calculate strategy weights
        calc_strategy_weights = {}
        for name, score in scores.items():
            signal = signals.get(name)
            if signal:
                # Base: combine score, confidence, and regime fit
                base_weight = (
                    score.total_score * 
                    signal.confidence * 
                    (0.5 + 0.5 * signal.regime_fit)
                )
                
                # Incorporate learned weights if available
                if learned_strategy_weights and name in learned_strategy_weights:
                    learned = learned_strategy_weights[name]
                    # Blend base calculation with learned weight
                    calc_strategy_weights[name] = 0.6 * base_weight + 0.4 * learned
                else:
                    calc_strategy_weights[name] = base_weight
        
        total = sum(calc_strategy_weights.values())
        if total <= 0:
            return {}
        
        calc_strategy_weights = {k: v / total for k, v in calc_strategy_weights.items()}
        
        # Combine signals
        for name, signal in signals.items():
            sw = calc_strategy_weights.get(name, 0)
            for symbol, weight in signal.desired_weights.items():
                combined[symbol] = combined.get(symbol, 0) + sw * weight
        
        return combined
    
    def _apply_constraints(
        self,
        weights: Dict[str, float],
        features: Features,
        current_weights: Dict[str, float]
    ) -> Tuple[Dict[str, float], List[str]]:
        """Apply risk and position constraints."""
        constraints_applied = []
        final = dict(weights)
        
        # 1. Position limits
        clipped = False
        for symbol in final:
            if abs(final[symbol]) > self.max_position:
                final[symbol] = np.sign(final[symbol]) * self.max_position
                clipped = True
        if clipped:
            constraints_applied.append("Position size limit applied")
        
        # 2. Sector exposure
        sector_exposure = {}
        for symbol, weight in final.items():
            sector = self.sector_map.get(symbol, 'other')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + abs(weight)
        
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                # Scale down sector
                scale = self.max_sector_exposure / exposure
                for symbol in final:
                    if self.sector_map.get(symbol, 'other') == sector:
                        final[symbol] *= scale
                constraints_applied.append(f"Sector limit applied to {sector}")
        
        # 3. Leverage limit
        total_exposure = sum(abs(w) for w in final.values())
        if total_exposure > self.max_leverage:
            scale = self.max_leverage / total_exposure
            final = {k: v * scale for k, v in final.items()}
            constraints_applied.append("Leverage limit applied")
        
        # 4. Turnover limit
        turnover = self._calculate_turnover(final, current_weights)
        if turnover > self.max_turnover:
            # Blend towards current weights
            blend = self.max_turnover / turnover
            all_symbols = set(final.keys()) | set(current_weights.keys())
            
            blended = {}
            for symbol in all_symbols:
                new_w = final.get(symbol, 0)
                old_w = current_weights.get(symbol, 0)
                blended[symbol] = old_w + blend * (new_w - old_w)
            
            final = {k: v for k, v in blended.items() if abs(v) > 0.001}
            constraints_applied.append(f"Turnover limit applied ({turnover:.1%} -> {self.max_turnover:.1%})")
        
        # 5. Volatility targeting
        port_vol = self._estimate_portfolio_vol(final, features)
        if port_vol > self.vol_target * 1.2:  # Allow 20% buffer
            scale = self.vol_target / port_vol
            final = {k: v * scale for k, v in final.items()}
            constraints_applied.append(f"Vol target applied ({port_vol:.1%} -> {self.vol_target:.1%})")
        
        # Clean up tiny weights (use smaller threshold to preserve valid small positions)
        final = {k: v for k, v in final.items() if abs(v) > 0.002}
        
        # Normalize based on portfolio type
        # For long-only: normalize to sum to 1.0
        # For L/S: normalize GROSS exposure to target, preserve net exposure
        total_net = sum(final.values())
        total_gross = sum(abs(v) for v in final.values())
        
        # Check if this is a L/S portfolio (has both longs and shorts)
        has_longs = any(v > 0 for v in final.values())
        has_shorts = any(v < 0 for v in final.values())
        
        if has_longs and has_shorts:
            # L/S Portfolio: normalize gross exposure to 1.0 (or less)
            if total_gross > 1.2:  # Allow 20% buffer
                scale = 1.0 / total_gross
                final = {k: v * scale for k, v in final.items()}
                constraints_applied.append(f"L/S gross exposure normalized ({total_gross:.1%} -> 1.0)")
        elif total_net > 0.1 and abs(total_net - 1.0) > 0.01:
            # Long-only: normalize to sum to 1.0
            final = {k: v / total_net for k, v in final.items()}
        
        return final, constraints_applied
    
    def _calculate_turnover(
        self,
        new_weights: Dict[str, float],
        old_weights: Dict[str, float]
    ) -> float:
        """Calculate portfolio turnover."""
        all_symbols = set(new_weights.keys()) | set(old_weights.keys())
        
        turnover = sum(
            abs(new_weights.get(s, 0) - old_weights.get(s, 0))
            for s in all_symbols
        ) / 2
        
        return turnover
    
    def _calculate_strategy_correlations(
        self,
        signals: Dict[str, SignalOutput],
        scores: Dict[str, StrategyScore]
    ) -> Dict[str, float]:
        """
        Calculate correlation-based penalties for strategies.
        
        Strategies with highly correlated signal outputs get penalized
        to encourage portfolio diversification and reduce redundancy.
        
        Returns:
            Dict of strategy_name -> penalty (0.0 to 0.5)
        """
        if len(signals) < 2:
            return {}
        
        # Build signal vectors for each strategy
        # Each vector contains weights for all symbols
        all_symbols = set()
        for signal in signals.values():
            all_symbols.update(signal.desired_weights.keys())
        
        all_symbols = sorted(list(all_symbols))
        if not all_symbols:
            return {}
        
        # Create weight matrix: strategies x symbols
        strategy_names = list(signals.keys())
        n_strategies = len(strategy_names)
        n_symbols = len(all_symbols)
        
        weight_matrix = np.zeros((n_strategies, n_symbols))
        for i, name in enumerate(strategy_names):
            for j, symbol in enumerate(all_symbols):
                weight_matrix[i, j] = signals[name].desired_weights.get(symbol, 0.0)
        
        # Calculate correlation matrix between strategies
        # Using Pearson correlation of signal vectors
        penalties = {}
        
        for i, name in enumerate(strategy_names):
            my_vector = weight_matrix[i]
            my_norm = np.linalg.norm(my_vector)
            if my_norm < 1e-8:
                penalties[name] = 0.0
                continue
            
            # Calculate average correlation with other strategies
            correlations = []
            for j, other_name in enumerate(strategy_names):
                if i == j:
                    continue
                
                other_vector = weight_matrix[j]
                other_norm = np.linalg.norm(other_vector)
                
                if other_norm < 1e-8:
                    continue
                
                # Cosine similarity (correlation for centered data)
                correlation = np.dot(my_vector, other_vector) / (my_norm * other_norm)
                
                # Weight by the other strategy's score (high correlation with good strategy is less bad)
                other_score = scores.get(other_name, StrategyScore(
                    strategy_name=other_name,
                    alpha_score=0.0,
                    regime_fit_score=0.0,
                    diversification_score=0.0,
                    drawdown_score=0.0,
                    sentiment_score=0.0,
                    total_score=0.5,
                    rationale="Default score"
                ))
                score_factor = other_score.total_score
                
                # Only penalize positive correlations (same signals)
                if correlation > 0.3:  # Threshold for "correlated"
                    correlations.append(correlation * (1 - score_factor * 0.5))
            
            if correlations:
                # Average correlation with others, scaled to penalty
                avg_corr = np.mean(correlations)
                # Max penalty of 0.3 (30% reduction) for perfect correlation
                penalty = min(0.3, avg_corr * 0.5)
                penalties[name] = penalty
                
                if penalty > 0.1:
                    logger.debug(f"Strategy {name} correlation penalty: {penalty:.2%}")
            else:
                penalties[name] = 0.0
        
        return penalties
    
    def _estimate_portfolio_vol(
        self,
        weights: Dict[str, float],
        features: Features
    ) -> float:
        """Estimate portfolio volatility."""
        if not weights:
            return 0.0
        
        if features.covariance_matrix is None:
            # Use average individual vol
            vols = [features.volatility_21d.get(s, 0.20) for s in weights]
            return np.mean(vols) if vols else 0.15
        
        symbols = [s for s in weights if s in features.covariance_matrix.columns]
        if len(symbols) == 0:
            return 0.15
        
        w = np.array([weights.get(s, 0) for s in symbols])
        cov = features.covariance_matrix.loc[symbols, symbols].values
        
        try:
            var = w @ cov @ w
            return np.sqrt(max(0, var))
        except:
            return 0.15
    
    def _calculate_contributions(
        self,
        signals: Dict[str, SignalOutput],
        scores: Dict[str, StrategyScore]
    ) -> Dict[str, float]:
        """Calculate each strategy's contribution to final weights."""
        contributions = {}
        total_score = sum(s.total_score for s in scores.values())
        
        if total_score > 0:
            for name, score in scores.items():
                contributions[name] = score.total_score / total_score
        
        return contributions
