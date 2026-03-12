"""
Value and Quality Tilt Strategy.
Uses REAL fundamental data from FundamentalsLoader.
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
import logging

from .base import Strategy, SignalOutput
from src.data.feature_store import Features

logger = logging.getLogger(__name__)


class ValueQualityTiltStrategy(Strategy):
    """
    Value and Quality Tilt Strategy.
    Tilts towards value (low P/E) and quality (high ROE) stocks.
    Now uses REAL fundamental data from Yahoo Finance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ValueQualityTilt", config)
        self._required_features = ['prices']
        
        config = config or {}
        self.top_n = config.get('top_n', 10)
        self.value_weight = config.get('value_weight', 0.5)  # Weight for value score
        self.quality_weight = config.get('quality_weight', 0.5)  # Weight for quality score
        self.min_market_cap = config.get('min_market_cap', 1e9)  # $1B minimum
        
        # Cache
        self._score_cache: Dict[str, Dict[str, float]] = {}
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
                logger.warning("FundamentalsLoader not available")
        return self._fundamentals
    
    def _calculate_scores(self, symbols: list) -> Dict[str, Dict[str, float]]:
        """Calculate value and quality scores using real data."""
        loader = self._get_fundamentals_loader()
        
        if loader is None:
            return self._get_fallback_scores(symbols)
        
        try:
            fundamentals = loader.get_fundamentals(symbols)
            scores = {}
            
            for symbol, fd in fundamentals.items():
                # Skip if market cap too small
                if fd.market_cap and fd.market_cap < self.min_market_cap:
                    continue
                
                scores[symbol] = {
                    'value_score': fd.value_score,
                    'quality_score': fd.quality_score,
                    'combined_score': (
                        self.value_weight * fd.value_score +
                        self.quality_weight * fd.quality_score
                    ),
                    'pe_ratio': fd.pe_ratio,
                    'roe': fd.roe,
                    'profit_margin': fd.profit_margin,
                    'debt_to_equity': fd.debt_to_equity,
                }
            
            logger.info(f"Calculated VQ scores for {len(scores)} symbols")
            return scores
            
        except Exception as e:
            logger.warning(f"Failed to calculate scores: {e}")
            return self._get_fallback_scores(symbols)
    
    def _get_fallback_scores(self, symbols: list) -> Dict[str, Dict[str, float]]:
        """Fallback scores based on known characteristics."""
        known_scores = {
            # High value + quality
            'JPM': {'value_score': 0.7, 'quality_score': 0.8, 'combined_score': 0.75},
            'JNJ': {'value_score': 0.6, 'quality_score': 0.85, 'combined_score': 0.725},
            'PG': {'value_score': 0.55, 'quality_score': 0.8, 'combined_score': 0.675},
            'KO': {'value_score': 0.5, 'quality_score': 0.85, 'combined_score': 0.675},
            'XOM': {'value_score': 0.7, 'quality_score': 0.6, 'combined_score': 0.65},
            'WMT': {'value_score': 0.5, 'quality_score': 0.75, 'combined_score': 0.625},
            'BAC': {'value_score': 0.75, 'quality_score': 0.5, 'combined_score': 0.625},
            # Medium
            'AAPL': {'value_score': 0.3, 'quality_score': 0.9, 'combined_score': 0.6},
            'MSFT': {'value_score': 0.25, 'quality_score': 0.95, 'combined_score': 0.6},
            'GOOGL': {'value_score': 0.35, 'quality_score': 0.8, 'combined_score': 0.575},
            # Low value (growth)
            'TSLA': {'value_score': 0.1, 'quality_score': 0.4, 'combined_score': 0.25},
            'NVDA': {'value_score': 0.1, 'quality_score': 0.8, 'combined_score': 0.45},
            'AMD': {'value_score': 0.15, 'quality_score': 0.6, 'combined_score': 0.375},
        }
        return {s: known_scores[s] for s in symbols if s in known_scores}
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate value/quality signals using real fundamentals."""
        weights = {}
        expected_returns = {}
        
        # Get real scores
        symbols = list(features.symbols)
        scores = self._calculate_scores(symbols)
        
        if not scores:
            return self._empty_signal(t)
        
        # Create scored assets list
        scored_assets = [
            (sym, data['combined_score'])
            for sym, data in scores.items()
            if sym in features.symbols
        ]
        
        if not scored_assets:
            return self._empty_signal(t)
        
        # Sort by combined score descending
        scored_assets.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_assets = scored_assets[:self.top_n]
        
        # Score-weighted allocation
        total_score = sum(s for _, s in top_assets)
        
        for symbol, score in top_assets:
            weight = score / total_score if total_score > 0 else 1.0 / len(top_assets)
            weights[symbol] = weight
            expected_returns[symbol] = (0.02 + score * 0.04) / 252  # Daily scale
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        # Value tends to underperform in strong bull markets
        regime_fit = 0.6
        if features.regime:
            from src.data.regime import TrendRegime
            if features.regime.trend == TrendRegime.STRONG_UP:
                regime_fit = 0.4
            elif features.regime.trend in [TrendRegime.WEAK_DOWN, TrendRegime.STRONG_DOWN]:
                regime_fit = 0.7
        
        # Confidence based on data quality
        data_quality = len(scores) / max(len(symbols), 1)
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
                'top_picks': [s for s, _ in top_assets],
                'scores': {
                    s: {
                        'combined': round(scores[s]['combined_score'], 2),
                        'value': round(scores[s]['value_score'], 2),
                        'quality': round(scores[s]['quality_score'], 2),
                    }
                    for s, _ in top_assets if s in scores
                },
                'avg_combined_score': round(np.mean([s for _, s in top_assets]), 2),
                'data_source': 'FundamentalsLoader' if self._fundamentals else 'Fallback',
            },
            regime_fit=regime_fit,
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
            explanation={'note': 'No value/quality data available'},
        )
