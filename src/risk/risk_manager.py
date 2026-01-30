"""
Risk management and constraint enforcement.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging

from src.data.feature_store import Features

logger = logging.getLogger(__name__)


@dataclass
class RiskConstraints:
    """Risk constraint configuration."""
    max_position_size: float = 0.05  # REDUCED from 15% to 5% - prevents huge losses
    max_sector_exposure: float = 0.25  # REDUCED from 30% to 25%
    max_leverage: float = 1.0
    max_turnover: float = 0.50
    max_drawdown_trigger: float = 0.10  # REDUCED from 15% to 10% - tighter control
    vol_target: float = 0.12
    vol_ceiling: float = 0.18  # REDUCED from 20% to 18%
    
    # Stop-loss / take-profit - NOW ENABLED BY DEFAULT
    enable_stop_loss: bool = True  # CRITICAL FIX: Enable stop-loss!
    stop_loss_pct: float = 0.02    # 2% stop-loss (was 5%)
    enable_take_profit: bool = True  # CRITICAL FIX: Enable take-profit!
    take_profit_pct: float = 0.04  # 4% take-profit (was 20%) - locks in gains
    
    # TRAILING STOP-LOSS (NEW) - Protects profits by moving stop up with price
    enable_trailing_stop: bool = True
    trailing_stop_activation: float = 0.02  # Activate after 2% profit
    trailing_stop_distance: float = 0.015   # Trail 1.5% behind peak
    
    # CORRELATION LIMIT (NEW) - Prevents holding too many correlated positions
    enable_correlation_limit: bool = True
    max_pairwise_correlation: float = 0.80  # Max allowed correlation between any two positions
    max_avg_correlation: float = 0.60       # Max average correlation of portfolio
    
    # === LONG/SHORT CONSTRAINTS ===
    enable_shorting: bool = True
    max_gross_exposure: float = 1.5  # REDUCED from 200% to 150% gross
    net_exposure_min: float = -0.2   # REDUCED from -30% to -20% net short
    net_exposure_max: float = 1.0    # Can be 100% net long
    max_short_position: float = 0.05  # REDUCED from 10% to 5% max per short
    max_long_position: float = 0.05   # REDUCED from 15% to 5% max per long
    max_total_short: float = 0.30     # REDUCED from 100% to 30% max total short
    
    # === MARGIN CONSTRAINTS (Futures) ===
    min_free_cash_pct: float = 0.10   # Keep 10% cash buffer
    max_margin_usage_pct: float = 0.80  # Use max 80% of available margin


@dataclass
class RiskCheckResult:
    """Result of risk check."""
    approved: bool
    original_weights: Dict[str, float]
    approved_weights: Dict[str, float]
    violations: List[str] = field(default_factory=list)
    adjustments: List[str] = field(default_factory=list)
    risk_metrics: Dict[str, float] = field(default_factory=dict)


class RiskManager:
    """
    Enforces risk constraints on proposed portfolio weights.
    """
    
    def __init__(
        self,
        constraints: Optional[RiskConstraints] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize risk manager.
        
        Args:
            constraints: Risk constraints
            config: Additional configuration
        """
        self.constraints = constraints or RiskConstraints()
        self.config = config or {}
        
        # Sector mapping
        self.sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'META': 'tech',
            'AMZN': 'tech', 'NVDA': 'tech', 'NFLX': 'tech', 'AMD': 'tech', 'INTC': 'tech',
            'JPM': 'finance', 'BAC': 'finance', 'GS': 'finance', 'MS': 'finance', 'C': 'finance',
            'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy',
            'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare', 'MRK': 'healthcare',
            'KO': 'consumer', 'PEP': 'consumer', 'PG': 'consumer', 'WMT': 'consumer', 'HD': 'consumer',
            'DIS': 'media', 'NFLX': 'media', 'CMCSA': 'media',
        }
        
        # State tracking
        self.entry_prices: Dict[str, float] = {}
        self.high_water_mark: float = 0.0
        self.current_drawdown: float = 0.0
        
        # Trailing stop tracking - peak price since entry for each position
        self.position_peaks: Dict[str, float] = {}  # symbol -> highest price since entry
    
    def check_and_approve(
        self,
        proposed_weights: Dict[str, float],
        features: Features,
        current_weights: Optional[Dict[str, float]] = None,
        current_nav: float = 1.0
    ) -> RiskCheckResult:
        """
        Check proposed weights against constraints and produce approved weights.
        
        Args:
            proposed_weights: Proposed portfolio weights
            features: Current market features
            current_weights: Current portfolio weights
            current_nav: Current portfolio NAV
            
        Returns:
            RiskCheckResult with approved weights
        """
        current_weights = current_weights or {}
        
        result = RiskCheckResult(
            approved=True,
            original_weights=dict(proposed_weights),
            approved_weights=dict(proposed_weights),
        )
        
        # Update drawdown
        self._update_drawdown(current_nav)
        result.risk_metrics['drawdown'] = self.current_drawdown
        
        # Run checks
        self._check_position_limits(result)
        self._check_sector_exposure(result)
        self._check_leverage(result)
        self._check_turnover(result, current_weights)
        self._check_volatility(result, features)
        self._check_drawdown(result)
        
        # Long/Short specific checks
        if self.constraints.enable_shorting:
            self._check_gross_exposure(result)
            self._check_net_exposure(result)
            self._check_short_limits(result)
        
        # Stop-loss / take-profit / trailing stop
        if self.constraints.enable_stop_loss or self.constraints.enable_take_profit or self.constraints.enable_trailing_stop:
            self._check_stop_take(result, features)
        
        # Correlation limit check
        if self.constraints.enable_correlation_limit:
            self._check_correlation_limit(result, features)
        
        # Calculate final metrics
        result.risk_metrics['leverage'] = sum(abs(w) for w in result.approved_weights.values())
        result.risk_metrics['turnover'] = self._calc_turnover(
            result.approved_weights, current_weights
        )
        result.risk_metrics['portfolio_vol'] = self._estimate_vol(
            result.approved_weights, features
        )
        result.risk_metrics['n_positions'] = len([
            w for w in result.approved_weights.values() if abs(w) > 0.001
        ])
        
        return result
    
    def _check_position_limits(self, result: RiskCheckResult) -> None:
        """Check individual position limits."""
        max_size = self.constraints.max_position_size
        
        for symbol, weight in list(result.approved_weights.items()):
            if abs(weight) > max_size:
                clipped = np.sign(weight) * max_size
                result.approved_weights[symbol] = clipped
                result.adjustments.append(
                    f"Clipped {symbol}: {weight:.1%} -> {clipped:.1%}"
                )
    
    def _check_sector_exposure(self, result: RiskCheckResult) -> None:
        """Check sector exposure limits."""
        max_sector = self.constraints.max_sector_exposure
        
        # Calculate sector exposures
        sector_exp = {}
        for symbol, weight in result.approved_weights.items():
            sector = self.sector_map.get(symbol, 'other')
            sector_exp[sector] = sector_exp.get(sector, 0) + abs(weight)
        
        result.risk_metrics['sector_exposure'] = sector_exp
        
        # Check and scale if needed
        for sector, exposure in sector_exp.items():
            if exposure > max_sector:
                scale = max_sector / exposure
                for symbol in result.approved_weights:
                    if self.sector_map.get(symbol, 'other') == sector:
                        result.approved_weights[symbol] *= scale
                
                result.adjustments.append(
                    f"Scaled {sector} sector: {exposure:.1%} -> {max_sector:.1%}"
                )
    
    def _check_leverage(self, result: RiskCheckResult) -> None:
        """Check total leverage."""
        max_lev = self.constraints.max_leverage
        current_lev = sum(abs(w) for w in result.approved_weights.values())
        
        if current_lev > max_lev:
            scale = max_lev / current_lev
            result.approved_weights = {
                k: v * scale for k, v in result.approved_weights.items()
            }
            result.adjustments.append(
                f"Reduced leverage: {current_lev:.1%} -> {max_lev:.1%}"
            )
    
    def _check_turnover(
        self,
        result: RiskCheckResult,
        current_weights: Dict[str, float]
    ) -> None:
        """Check turnover limits."""
        max_to = self.constraints.max_turnover
        turnover = self._calc_turnover(result.approved_weights, current_weights)
        
        if turnover > max_to:
            # Blend towards current
            blend = max_to / turnover
            all_symbols = set(result.approved_weights.keys()) | set(current_weights.keys())
            
            blended = {}
            for symbol in all_symbols:
                new_w = result.approved_weights.get(symbol, 0)
                old_w = current_weights.get(symbol, 0)
                blended[symbol] = old_w + blend * (new_w - old_w)
            
            result.approved_weights = {k: v for k, v in blended.items() if abs(v) > 0.001}
            result.adjustments.append(
                f"Reduced turnover: {turnover:.1%} -> {max_to:.1%}"
            )
    
    def _check_volatility(
        self,
        result: RiskCheckResult,
        features: Features
    ) -> None:
        """Check portfolio volatility."""
        port_vol = self._estimate_vol(result.approved_weights, features)
        vol_ceil = self.constraints.vol_ceiling
        
        if port_vol > vol_ceil:
            scale = self.constraints.vol_target / port_vol
            result.approved_weights = {
                k: v * scale for k, v in result.approved_weights.items()
            }
            result.adjustments.append(
                f"Reduced vol: {port_vol:.1%} -> target {self.constraints.vol_target:.1%}"
            )
    
    def _check_drawdown(self, result: RiskCheckResult) -> None:
        """Check drawdown constraints."""
        dd_trigger = self.constraints.max_drawdown_trigger
        
        if self.current_drawdown > dd_trigger:
            # Reduce exposure significantly
            reduction = 0.5 * (1 + self.current_drawdown / dd_trigger)
            reduction = min(0.8, reduction)
            
            result.approved_weights = {
                k: v * (1 - reduction) for k, v in result.approved_weights.items()
            }
            result.violations.append(
                f"DRAWDOWN TRIGGER: {self.current_drawdown:.1%} > {dd_trigger:.1%}"
            )
            result.adjustments.append(
                f"Reduced exposure by {reduction:.0%} due to drawdown"
            )
    
    def _check_stop_take(
        self,
        result: RiskCheckResult,
        features: Features
    ) -> None:
        """Check stop-loss, trailing stop, and take-profit levels."""
        for symbol, weight in list(result.approved_weights.items()):
            if symbol not in self.entry_prices:
                continue
            
            entry = self.entry_prices[symbol]
            current = features.prices.get(symbol)
            
            if current is None or entry <= 0:
                continue
            
            # For longs: pnl positive when price goes up
            # For shorts: pnl positive when price goes down
            is_long = weight > 0
            if is_long:
                pnl = (current - entry) / entry
            else:
                pnl = (entry - current) / entry  # Inverted for shorts
            
            # === TRAILING STOP (NEW) ===
            # Protects profits by moving stop up with price
            if self.constraints.enable_trailing_stop:
                activation = self.constraints.trailing_stop_activation
                trail_distance = self.constraints.trailing_stop_distance
                
                # Update peak price tracking
                if symbol not in self.position_peaks:
                    self.position_peaks[symbol] = current if is_long else entry  # Entry for shorts
                
                # Update peak (highest for longs, lowest for shorts)
                if is_long:
                    if current > self.position_peaks[symbol]:
                        self.position_peaks[symbol] = current
                    peak = self.position_peaks[symbol]
                    peak_pnl = (peak - entry) / entry
                else:
                    if current < self.position_peaks.get(symbol, entry):
                        self.position_peaks[symbol] = current
                    peak = self.position_peaks[symbol]
                    peak_pnl = (entry - peak) / entry
                
                # Check if trailing stop is activated (we've hit activation threshold)
                if peak_pnl >= activation:
                    # Calculate trailing stop level
                    trailing_pnl = peak_pnl - trail_distance
                    
                    # If current P&L falls below trailing stop, exit
                    if pnl < trailing_pnl:
                        result.approved_weights[symbol] = 0
                        result.adjustments.append(
                            f"TRAILING STOP triggered for {symbol} "
                            f"(Peak: {peak_pnl:.1%}, Now: {pnl:.1%}, Trail: {trailing_pnl:.1%})"
                        )
                        # Clean up tracking
                        self.position_peaks.pop(symbol, None)
                        continue
            
            # === FIXED STOP-LOSS ===
            if self.constraints.enable_stop_loss:
                if pnl < -self.constraints.stop_loss_pct:
                    result.approved_weights[symbol] = 0
                    result.adjustments.append(
                        f"STOP-LOSS triggered for {symbol} (P&L: {pnl:.1%})"
                    )
                    # Clean up trailing stop tracking
                    self.position_peaks.pop(symbol, None)
                    continue
            
            # === TAKE-PROFIT ===
            if self.constraints.enable_take_profit:
                if pnl > self.constraints.take_profit_pct:
                    # Reduce position by half
                    result.approved_weights[symbol] *= 0.5
                    result.adjustments.append(
                        f"TAKE-PROFIT triggered for {symbol} (P&L: {pnl:.1%})"
                    )
    
    # === LONG/SHORT CONSTRAINT CHECKS ===
    
    def _check_gross_exposure(self, result: RiskCheckResult) -> None:
        """
        Check gross exposure (sum of absolute weights).
        
        Gross exposure = |long| + |short|
        """
        max_gross = self.constraints.max_gross_exposure
        
        gross = sum(abs(w) for w in result.approved_weights.values())
        
        result.risk_metrics['gross_exposure'] = gross
        
        if gross > max_gross:
            scale = max_gross / gross
            result.approved_weights = {
                k: v * scale for k, v in result.approved_weights.items()
            }
            result.adjustments.append(
                f"Reduced gross exposure: {gross:.1%} -> {max_gross:.1%}"
            )
    
    def _check_net_exposure(self, result: RiskCheckResult) -> None:
        """
        Check net exposure (long - short).
        
        Enforces net exposure band [min, max].
        """
        net_min = self.constraints.net_exposure_min
        net_max = self.constraints.net_exposure_max
        
        # Calculate long and short exposure
        long_exp = sum(w for w in result.approved_weights.values() if w > 0)
        short_exp = sum(abs(w) for w in result.approved_weights.values() if w < 0)
        net = long_exp - short_exp
        
        result.risk_metrics['net_exposure'] = net
        result.risk_metrics['long_exposure'] = long_exp
        result.risk_metrics['short_exposure'] = short_exp
        
        if net < net_min:
            # Too short - reduce shorts or add longs
            # For now, scale shorts down
            adjustment_needed = net_min - net
            if short_exp > 0:
                scale = max(0, (short_exp - adjustment_needed) / short_exp)
                for symbol, weight in result.approved_weights.items():
                    if weight < 0:
                        result.approved_weights[symbol] = weight * scale
                
                result.adjustments.append(
                    f"Reduced short exposure: net {net:.1%} -> {net_min:.1%}"
                )
                result.violations.append(
                    f"NET EXPOSURE below minimum: {net:.1%} < {net_min:.1%}"
                )
        
        elif net > net_max:
            # Too long - reduce longs
            adjustment_needed = net - net_max
            if long_exp > 0:
                scale = max(0, (long_exp - adjustment_needed) / long_exp)
                for symbol, weight in result.approved_weights.items():
                    if weight > 0:
                        result.approved_weights[symbol] = weight * scale
                
                result.adjustments.append(
                    f"Reduced long exposure: net {net:.1%} -> {net_max:.1%}"
                )
    
    def _check_short_limits(self, result: RiskCheckResult) -> None:
        """
        Check short-specific limits:
        - Max per short position
        - Max total short exposure
        """
        if not self.constraints.enable_shorting:
            # Remove all shorts if shorting disabled
            for symbol, weight in list(result.approved_weights.items()):
                if weight < 0:
                    result.approved_weights[symbol] = 0
                    result.adjustments.append(
                        f"Removed short {symbol}: shorting disabled"
                    )
            return
        
        max_short_pos = self.constraints.max_short_position
        max_total_short = self.constraints.max_total_short
        
        # Check individual short limits
        for symbol, weight in list(result.approved_weights.items()):
            if weight < -max_short_pos:
                result.approved_weights[symbol] = -max_short_pos
                result.adjustments.append(
                    f"Clipped short {symbol}: {weight:.1%} -> {-max_short_pos:.1%}"
                )
        
        # Check total short exposure
        total_short = sum(abs(w) for w in result.approved_weights.values() if w < 0)
        
        if total_short > max_total_short:
            scale = max_total_short / total_short
            for symbol, weight in result.approved_weights.items():
                if weight < 0:
                    result.approved_weights[symbol] = weight * scale
            
            result.adjustments.append(
                f"Reduced total short: {total_short:.1%} -> {max_total_short:.1%}"
            )
    
    def _check_correlation_limit(self, result: RiskCheckResult, features: Features) -> None:
        """
        Check portfolio correlation limits.
        
        Prevents holding too many highly correlated positions.
        When correlations are too high, reduces the smaller position.
        """
        if features.correlation_matrix is None or features.correlation_matrix.empty:
            return
        
        max_pairwise = self.constraints.max_pairwise_correlation
        max_avg = self.constraints.max_avg_correlation
        
        # Get symbols in current portfolio
        portfolio_symbols = [s for s in result.approved_weights.keys() 
                           if abs(result.approved_weights[s]) > 0.005]
        
        if len(portfolio_symbols) < 2:
            return
        
        # Check pairwise correlations
        violations = []
        corr_matrix = features.correlation_matrix
        
        for i, sym1 in enumerate(portfolio_symbols):
            for sym2 in portfolio_symbols[i+1:]:
                if sym1 not in corr_matrix.columns or sym2 not in corr_matrix.columns:
                    continue
                
                try:
                    corr = abs(corr_matrix.loc[sym1, sym2])
                    
                    if corr > max_pairwise:
                        violations.append((sym1, sym2, corr))
                except:
                    continue
        
        # Handle violations by reducing the smaller position
        for sym1, sym2, corr in violations:
            w1 = abs(result.approved_weights.get(sym1, 0))
            w2 = abs(result.approved_weights.get(sym2, 0))
            
            # Reduce the smaller position by 50%
            if w1 > w2:
                # Reduce sym2
                old = result.approved_weights[sym2]
                result.approved_weights[sym2] = old * 0.5
                result.adjustments.append(
                    f"Reduced {sym2} (corr with {sym1}: {corr:.0%} > {max_pairwise:.0%})"
                )
            else:
                # Reduce sym1
                old = result.approved_weights[sym1]
                result.approved_weights[sym1] = old * 0.5
                result.adjustments.append(
                    f"Reduced {sym1} (corr with {sym2}: {corr:.0%} > {max_pairwise:.0%})"
                )
        
        # Check average correlation
        total_corr = 0
        count = 0
        for i, sym1 in enumerate(portfolio_symbols):
            for sym2 in portfolio_symbols[i+1:]:
                if sym1 in corr_matrix.columns and sym2 in corr_matrix.columns:
                    try:
                        total_corr += abs(corr_matrix.loc[sym1, sym2])
                        count += 1
                    except:
                        pass
        
        if count > 0:
            avg_corr = total_corr / count
            result.risk_metrics['avg_portfolio_correlation'] = avg_corr
            
            if avg_corr > max_avg:
                result.violations.append(
                    f"Portfolio avg correlation {avg_corr:.0%} > {max_avg:.0%}"
                )
    
    def _update_drawdown(self, current_nav: float) -> None:
        """Update high water mark and drawdown."""
        if current_nav > self.high_water_mark:
            self.high_water_mark = current_nav
        
        if self.high_water_mark > 0:
            self.current_drawdown = (self.high_water_mark - current_nav) / self.high_water_mark
        else:
            self.current_drawdown = 0.0
    
    def _calc_turnover(
        self,
        new_weights: Dict[str, float],
        old_weights: Dict[str, float]
    ) -> float:
        """Calculate turnover between portfolios."""
        all_symbols = set(new_weights.keys()) | set(old_weights.keys())
        return sum(
            abs(new_weights.get(s, 0) - old_weights.get(s, 0))
            for s in all_symbols
        ) / 2
    
    def _estimate_vol(
        self,
        weights: Dict[str, float],
        features: Features
    ) -> float:
        """Estimate portfolio volatility."""
        if not weights:
            return 0.0
        
        if features.covariance_matrix is None:
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
    
    def update_entry_prices(
        self,
        new_positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> None:
        """Update entry prices for new positions."""
        for symbol, weight in new_positions.items():
            if abs(weight) > 0.001:
                if symbol not in self.entry_prices:
                    self.entry_prices[symbol] = prices.get(symbol, 0)
            else:
                # Position closed
                self.entry_prices.pop(symbol, None)
    
    def reset(self) -> None:
        """Reset risk manager state."""
        self.entry_prices = {}
        self.high_water_mark = 0.0
        self.current_drawdown = 0.0
        self.position_peaks = {}  # Reset trailing stop tracking
    
    def get_exposure_summary(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Get exposure summary for a set of weights.
        
        Returns:
            Dict with 'gross', 'net', 'long', 'short' exposures
        """
        long_exp = sum(w for w in weights.values() if w > 0)
        short_exp = sum(abs(w) for w in weights.values() if w < 0)
        
        return {
            'gross': long_exp + short_exp,
            'net': long_exp - short_exp,
            'long': long_exp,
            'short': short_exp,
            'n_longs': sum(1 for w in weights.values() if w > 0),
            'n_shorts': sum(1 for w in weights.values() if w < 0),
        }
    
    def validate_for_shorting(self, symbol: str, shortable: bool = True) -> bool:
        """
        Validate if a symbol can be shorted.
        
        Args:
            symbol: Stock symbol
            shortable: Whether the stock is shortable (from broker)
        
        Returns:
            True if shorting is allowed
        """
        if not self.constraints.enable_shorting:
            return False
        return shortable
