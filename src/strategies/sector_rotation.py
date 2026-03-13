"""
Sector Rotation Strategy.
Rotates between sectors based on economic cycle, momentum, and relative strength.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .base import Strategy, SignalOutput
from src.data.feature_store import Features

logger = logging.getLogger(__name__)


class SectorRotationStrategy(Strategy):
    """
    Sector Rotation Strategy.
    
    Rotates between sectors based on:
    1. Relative strength (momentum)
    2. Economic cycle indicators
    3. Volatility regime
    
    Uses sector ETFs or representative stocks.
    """
    
    # Sector ETFs and representative stocks
    SECTORS = {
        'Technology': {
            'etf': 'XLK',
            'stocks': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'AMD'],
            'cycle_preference': ['expansion', 'late_cycle'],
        },
        'Healthcare': {
            'etf': 'XLV',
            'stocks': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO'],
            'cycle_preference': ['recession', 'early_cycle'],
        },
        'Financials': {
            'etf': 'XLF',
            'stocks': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK'],
            'cycle_preference': ['early_cycle', 'expansion'],
        },
        'Consumer Discretionary': {
            'etf': 'XLY',
            'stocks': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW'],
            'cycle_preference': ['expansion', 'late_cycle'],
        },
        'Consumer Staples': {
            'etf': 'XLP',
            'stocks': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'CL'],
            'cycle_preference': ['recession', 'late_cycle'],
        },
        'Industrials': {
            'etf': 'XLI',
            'stocks': ['HON', 'UPS', 'CAT', 'BA', 'GE', 'LMT', 'DE'],
            'cycle_preference': ['early_cycle', 'expansion'],
        },
        'Energy': {
            'etf': 'XLE',
            'stocks': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'OXY'],
            'cycle_preference': ['late_cycle', 'early_cycle'],
        },
        'Utilities': {
            'etf': 'XLU',
            'stocks': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE'],
            'cycle_preference': ['recession'],
        },
        'Real Estate': {
            'etf': 'XLRE',
            'stocks': ['PLD', 'AMT', 'EQIX', 'SPG', 'O', 'DLR', 'PSA'],
            'cycle_preference': ['expansion', 'early_cycle'],
        },
        'Materials': {
            'etf': 'XLB',
            'stocks': ['LIN', 'SHW', 'APD', 'FCX', 'NEM', 'NUE', 'ECL'],
            'cycle_preference': ['late_cycle', 'expansion'],
        },
        'Communication Services': {
            'etf': 'XLC',
            'stocks': ['GOOGL', 'META', 'NFLX', 'DIS', 'VZ', 'T', 'TMUS'],
            'cycle_preference': ['expansion'],
        },
    }
    
    # Economic cycle definitions
    CYCLE_STAGES = ['early_cycle', 'expansion', 'late_cycle', 'recession']
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("SectorRotation", config)
        self._required_features = ['prices', 'returns_21d', 'returns_63d', 'regime']
        
        config = config or {}
        self.top_n_sectors = config.get('top_n_sectors', 3)  # How many sectors to overweight
        self.momentum_weight = config.get('momentum_weight', 0.5)
        self.cycle_weight = config.get('cycle_weight', 0.3)
        self.volatility_weight = config.get('volatility_weight', 0.2)
        self.use_etfs = config.get('use_etfs', False)  # Use stocks by default
        
        # NEW: Fast rotation mode - use shorter lookbacks
        self.fast_rotation = config.get('fast_rotation', True)  # Use 5-day momentum
        self.intraday_adaptation = config.get('intraday_adaptation', True)  # React to intraday moves
        
        # Track sector performance
        self._sector_returns: Dict[str, List[float]] = {s: [] for s in self.SECTORS}
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate sector rotation signals."""
        
        # 1. Calculate sector momentum scores
        sector_momentum = self._calculate_sector_momentum(features)
        
        # 2. Determine economic cycle
        cycle = self._estimate_cycle(features)
        cycle_scores = self._get_cycle_scores(cycle)
        
        # 3. Volatility adjustment
        vol_scores = self._get_volatility_scores(features)
        
        # 4. Combine scores
        combined_scores = {}
        for sector in self.SECTORS:
            mom = sector_momentum.get(sector, 0)
            cyc = cycle_scores.get(sector, 0)
            vol = vol_scores.get(sector, 0)
            
            combined_scores[sector] = (
                self.momentum_weight * mom +
                self.cycle_weight * cyc +
                self.volatility_weight * vol
            )
        
        # 5. Rank sectors
        ranked_sectors = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 6. Construct portfolio
        weights = {}
        expected_returns = {}
        
        # Overweight top sectors
        top_sectors = [s for s, _ in ranked_sectors[:self.top_n_sectors]]
        bottom_sectors = [s for s, _ in ranked_sectors[-self.top_n_sectors:]]
        
        total_weight_per_sector = 0.8 / self.top_n_sectors
        
        for sector in top_sectors:
            sector_info = self.SECTORS[sector]
            
            if self.use_etfs:
                symbols = [sector_info['etf']]
            else:
                # Use top 3 stocks from sector
                symbols = [s for s in sector_info['stocks'][:3] if s in features.symbols]
            
            if not symbols:
                continue
            
            weight_per_stock = total_weight_per_sector / len(symbols)
            
            for symbol in symbols:
                weights[symbol] = weight_per_stock
                expected_returns[symbol] = combined_scores[sector] * 0.02 / 252
        
        # Optional: Short bottom sectors (market neutral)
        # Uncomment to enable long/short sector rotation
        # for sector in bottom_sectors:
        #     sector_info = self.SECTORS[sector]
        #     symbols = [s for s in sector_info['stocks'][:2] if s in features.symbols]
        #     weight_per_stock = -0.1 / len(symbols) if symbols else 0
        #     for symbol in symbols:
        #         weights[symbol] = weight_per_stock
        #         expected_returns[symbol] = abs(combined_scores[sector]) * 0.01
        
        if not weights:
            return self._empty_signal(t)
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=0.6,
            explanation={
                'top_sectors': top_sectors,
                'bottom_sectors': bottom_sectors,
                'cycle_estimate': cycle,
                'sector_scores': {
                    k: round(v, 3) for k, v in combined_scores.items()
                },
            },
            regime_fit=0.7,
            diversification_score=0.7,
        )
    
    def _calculate_sector_momentum(self, features: Features) -> Dict[str, float]:
        """Calculate momentum score for each sector - now with fast rotation support."""
        sector_scores = {}
        
        for sector, info in self.SECTORS.items():
            stocks = info['stocks']
            returns_5d = []
            returns_21d = []
            intraday_moves = []
            
            for stock in stocks:
                # Fast rotation: prioritize 5-day returns
                if self.fast_rotation:
                    ret_5 = features.returns_5d.get(stock) if hasattr(features, 'returns_5d') else None
                    if ret_5 is not None and not np.isnan(ret_5):
                        returns_5d.append(ret_5)
                
                # 21-day returns for medium-term signal
                ret_21 = features.returns_21d.get(stock)
                if ret_21 is not None and not np.isnan(ret_21):
                    returns_21d.append(ret_21)
                
                # Intraday adaptation: use 1-day returns as well
                if self.intraday_adaptation:
                    ret_1 = features.returns_1d.get(stock) if hasattr(features, 'returns_1d') else None
                    if ret_1 is not None and not np.isnan(ret_1):
                        intraday_moves.append(ret_1)
            
            # Combine fast and slow momentum
            fast_momentum = np.mean(returns_5d) if returns_5d else 0.0
            slow_momentum = np.mean(returns_21d) if returns_21d else 0.0
            intraday_momentum = np.mean(intraday_moves) if intraday_moves else 0.0
            
            # Weight: 40% fast (5d), 30% slow (21d), 30% intraday (1d)
            if self.fast_rotation:
                combined = 0.4 * fast_momentum + 0.3 * slow_momentum + 0.3 * intraday_momentum
            else:
                combined = slow_momentum
            
            # Scale to roughly -1 to 1
            sector_scores[sector] = np.tanh(combined * 10)
        
        return sector_scores
    
    def _estimate_cycle(self, features: Features) -> str:
        """Estimate current economic cycle stage."""
        
        # Use regime information if available
        if features.regime:
            from src.data.regime import TrendRegime, VolatilityRegime
            
            trend = features.regime.trend
            vol = features.regime.volatility
            
            if vol == VolatilityRegime.HIGH:
                return 'recession'
            elif trend == TrendRegime.STRONG_UP:
                return 'expansion'
            elif trend == TrendRegime.WEAK_UP:
                return 'late_cycle'
            elif trend == TrendRegime.WEAK_DOWN:
                return 'early_cycle'
            else:
                return 'recession'
        
        # Default to expansion
        return 'expansion'
    
    def _get_cycle_scores(self, cycle: str) -> Dict[str, float]:
        """Get sector scores based on economic cycle."""
        scores = {}
        
        for sector, info in self.SECTORS.items():
            preferred_cycles = info['cycle_preference']
            
            if cycle in preferred_cycles:
                # Position in preference list matters
                idx = preferred_cycles.index(cycle)
                scores[sector] = 1.0 - idx * 0.2  # First preference = 1.0, second = 0.8
            else:
                scores[sector] = -0.2  # Slight negative for non-preferred
        
        return scores
    
    def _get_volatility_scores(self, features: Features) -> Dict[str, float]:
        """Score sectors based on volatility (lower vol = higher score in high vol regimes)."""
        scores = {}
        
        # Defensive sectors in high volatility
        defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
        cyclical_sectors = ['Technology', 'Consumer Discretionary', 'Financials']
        
        is_high_vol = False
        if features.regime:
            from src.data.regime import VolatilityRegime
            is_high_vol = features.regime.volatility == VolatilityRegime.HIGH
        
        for sector in self.SECTORS:
            if is_high_vol:
                if sector in defensive_sectors:
                    scores[sector] = 0.8
                elif sector in cyclical_sectors:
                    scores[sector] = -0.3
                else:
                    scores[sector] = 0.2
            else:
                if sector in cyclical_sectors:
                    scores[sector] = 0.5
                elif sector in defensive_sectors:
                    scores[sector] = 0.1
                else:
                    scores[sector] = 0.3
        
        return scores
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'note': 'No sector rotation signals'},
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
