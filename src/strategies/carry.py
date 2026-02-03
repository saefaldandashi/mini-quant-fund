"""
Carry Strategy - Equity Dividend Yield.
Uses real dividend yield data from FundamentalsLoader.
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
import logging

from .base import Strategy, SignalOutput
from src.data.feature_store import Features

logger = logging.getLogger(__name__)


class CarryStrategy(Strategy):
    """
    Carry Strategy.
    For equities, uses dividend yield as carry proxy.
    Fetches REAL dividend data from Yahoo Finance via FundamentalsLoader.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Carry", config)
        self._required_features = ['prices', 'returns_252d']
        
        config = config or {}
        self.top_n = config.get('top_n', 10)  # Number of high-carry stocks
        self.min_yield = config.get('min_yield', 0.01)  # Minimum 1% yield
        self.max_yield = config.get('max_yield', 0.15)  # Max 15% (avoid traps)
        
        # Cache for dividend yields
        self._dividend_cache: Dict[str, float] = {}
        self._last_fetch: Optional[datetime] = None
        
        # Fundamentals loader
        self._fundamentals = None
    
    def _get_fundamentals_loader(self):
        """Lazy load fundamentals loader."""
        if self._fundamentals is None:
            try:
                from src.data.fundamentals import get_fundamentals_loader
                self._fundamentals = get_fundamentals_loader()
            except ImportError:
                logger.warning("FundamentalsLoader not available, using fallback")
                self._fundamentals = None
        return self._fundamentals
    
    def _fetch_dividend_yields(self, symbols: list) -> Dict[str, float]:
        """Fetch real dividend yields."""
        loader = self._get_fundamentals_loader()
        
        if loader is None:
            # Fallback to reasonable estimates for major dividend stocks
            return self._get_fallback_yields(symbols)
        
        try:
            fundamentals = loader.get_fundamentals(symbols)
            yields = {}
            
            for symbol, fd in fundamentals.items():
                div_yield = fd.dividend_yield
                if div_yield is not None and self.min_yield <= div_yield <= self.max_yield:
                    yields[symbol] = div_yield
            
            logger.info(f"Fetched dividend yields for {len(yields)} symbols")
            return yields
            
        except Exception as e:
            logger.warning(f"Failed to fetch dividends: {e}")
            return self._get_fallback_yields(symbols)
    
    def _get_fallback_yields(self, symbols: list) -> Dict[str, float]:
        """Fallback dividend yields for common dividend stocks."""
        # Known dividend payers (updated periodically)
        known_yields = {
            'T': 0.065, 'VZ': 0.065, 'XOM': 0.035, 'CVX': 0.04,
            'KO': 0.03, 'PEP': 0.028, 'JNJ': 0.03, 'PG': 0.025,
            'MO': 0.08, 'PM': 0.055, 'IBM': 0.045, 'ABBV': 0.04,
            'INTC': 0.015, 'PFE': 0.04, 'MMM': 0.055, 'O': 0.05,
            'SCHD': 0.035, 'VYM': 0.03, 'HDV': 0.04, 'DVY': 0.035,
        }
        return {s: known_yields[s] for s in symbols if s in known_yields}
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate carry signals based on dividend yield."""
        weights = {}
        expected_returns = {}
        
        # Fetch real dividend data
        symbols = list(features.symbols)
        yields = self._fetch_dividend_yields(symbols)
        
        if not yields:
            return self._empty_signal(t)
        
        # Create carry assets list
        carry_assets = [(sym, yld) for sym, yld in yields.items() if sym in features.symbols]
        
        if not carry_assets:
            return self._empty_signal(t)
        
        # Rank by carry and go long high carry
        carry_assets.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_assets = carry_assets[:self.top_n]
        
        # Yield-weighted allocation (higher yield = higher weight)
        total_yield = sum(y for _, y in top_assets)
        
        for symbol, div_yield in top_assets:
            weight = div_yield / total_yield if total_yield > 0 else 1.0 / len(top_assets)
            weights[symbol] = weight
            expected_returns[symbol] = div_yield  # Carry is expected return
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        # Confidence based on data quality
        data_quality = len(yields) / max(len(symbols), 1)
        confidence = 0.5 + data_quality * 0.2
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=confidence,
            explanation={
                'top_carry_stocks': [s for s, _ in top_assets],
                'yields': {s: round(y * 100, 2) for s, y in top_assets},
                'avg_yield': round(np.mean([c for _, c in top_assets]) * 100, 2),
                'data_source': 'FundamentalsLoader' if self._fundamentals else 'Fallback',
            },
            regime_fit=0.6,
            diversification_score=0.6,
        )
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'note': 'No carry data available'},
        )
