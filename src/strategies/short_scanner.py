"""
Fundamental-Driven Short Scanner

Identifies short opportunities based on:
1. News-driven signals (earnings miss, downgrades, regulatory issues)
2. Valuation-based signals (high P/E, extended from MA)
3. Technical breakdown signals (below 200 DMA, death cross)

Integrates with:
- News sentiment system for negative catalysts
- Feature store for technical indicators
- Strategy ensemble for signal generation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShortCandidate:
    """A potential short candidate with scoring."""
    symbol: str
    
    # Signal scores (0-1)
    news_score: float = 0.0
    valuation_score: float = 0.0
    technical_score: float = 0.0
    cross_asset_score: float = 0.0  # NEW: Cross-asset driven score
    
    # Composite score
    total_score: float = 0.0
    
    # Signal details
    news_signals: List[str] = field(default_factory=list)
    valuation_signals: List[str] = field(default_factory=list)
    technical_signals: List[str] = field(default_factory=list)
    cross_asset_signals: List[str] = field(default_factory=list)  # NEW
    
    # Risk metrics
    short_interest: float = 0.0  # If available
    borrow_rate: float = 0.0     # Annual borrow cost %
    avg_volume: float = 0.0      # Liquidity
    
    # Recommendation
    conviction: str = "low"  # low, medium, high
    recommended_weight: float = 0.0
    
    def calculate_total(self, weights: Optional[Dict[str, float]] = None):
        """Calculate weighted total score."""
        weights = weights or {
            'news': 0.35,
            'valuation': 0.25,
            'technical': 0.25,
            'cross_asset': 0.15,  # NEW: Cross-asset weight
        }
        
        self.total_score = (
            self.news_score * weights.get('news', 0.35) +
            self.valuation_score * weights.get('valuation', 0.25) +
            self.technical_score * weights.get('technical', 0.25) +
            self.cross_asset_score * weights.get('cross_asset', 0.15)
        )
        
        # Determine conviction - more aggressive thresholds to generate shorts
        if self.total_score >= 0.55:
            self.conviction = "high"
            self.recommended_weight = -0.05  # 5% short
        elif self.total_score >= 0.40:
            self.conviction = "medium"
            self.recommended_weight = -0.03  # 3% short
        elif self.total_score >= 0.25:
            self.conviction = "low"
            self.recommended_weight = -0.02  # 2% short
        else:
            self.conviction = "none"
            self.recommended_weight = 0.0


@dataclass
class ShortScannerConfig:
    """Configuration for short scanner."""
    # Signal source weights
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        'news': 0.40,
        'valuation': 0.30,
        'technical': 0.30,
    })
    
    # Minimum scores to be considered - relaxed to generate more shorts
    min_total_score: float = 0.25
    min_news_score: float = 0.0     # Allow shorts without news
    min_sources_agreeing: int = 1    # Only need 1 source to agree (technical OR fundamental)
    
    # News signal thresholds
    news_sentiment_threshold: float = -0.3  # Bearish threshold
    earnings_miss_boost: float = 0.3        # Extra score for earnings miss
    downgrade_boost: float = 0.2            # Extra score for analyst downgrade
    regulatory_boost: float = 0.4           # Extra score for regulatory issues
    
    # Valuation thresholds - relaxed to find more shorts
    pe_vs_sector_threshold: float = 1.5     # P/E > 1.5x sector = high
    price_above_200ma_threshold: float = 0.15  # 15% above = extended (was 25%)
    
    # Technical thresholds - relaxed for more shorts
    below_200ma_days: int = 3               # Must be below for N days
    rsi_overbought: float = 65.0            # RSI above = overbought (was 70)
    
    # Risk limits
    max_borrow_rate: float = 10.0           # Skip if > 10% annual
    min_avg_volume: float = 500000          # Minimum daily volume
    
    # Output limits
    max_short_candidates: int = 10          # Max shorts to recommend


class ShortScanner:
    """
    Scans for short opportunities using fundamental and technical analysis.
    
    Usage:
        scanner = ShortScanner()
        candidates = scanner.scan(
            symbols=UNIVERSE,
            news_sentiments=news_data,
            features=feature_store_data,
            prices=price_data
        )
    """
    
    def __init__(self, config: Optional[ShortScannerConfig] = None):
        self.config = config or ShortScannerConfig()
        self.last_scan_time: Optional[datetime] = None
        self.last_candidates: List[ShortCandidate] = []
    
    # Cross-asset symbol mappings
    ENERGY_SYMBOLS = ['XOM', 'CVX', 'SLB', 'OXY', 'COP', 'HAL', 'MPC', 'VLO', 'PSX']
    MULTINATIONAL_SYMBOLS = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'META', 'NVDA', 'AMZN', 'NFLX']
    FINANCIAL_SYMBOLS = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW']
    CHINA_EXPOSED_SYMBOLS = ['AAPL', 'TSLA', 'NKE', 'SBUX', 'WYNN', 'LVS', 'QCOM']
    
    def scan(
        self,
        symbols: List[str],
        news_sentiments: Dict[str, Dict],
        features: Optional[Dict] = None,
        prices: Optional[Dict[str, float]] = None,
        ma_200: Optional[Dict[str, float]] = None,
        rsi_values: Optional[Dict[str, float]] = None,
        cross_asset_signals: Optional[Dict[str, float]] = None,  # NEW
    ) -> List[ShortCandidate]:
        """
        Scan for short opportunities.
        
        Args:
            symbols: List of symbols to scan
            news_sentiments: Dict of symbol -> sentiment data
            features: Feature store data
            prices: Current prices
            ma_200: 200-day moving averages
            rsi_values: RSI values
            cross_asset_signals: Cross-asset signals (oil_signal, dxy_signal, etc.)
        
        Returns:
            List of ShortCandidate sorted by total_score descending
        """
        logger.info(f"🔍 Short scanner: Scanning {len(symbols)} symbols")
        
        # Process cross-asset signals
        cross_asset_signals = cross_asset_signals or {}
        
        candidates = []
        
        for symbol in symbols:
            candidate = self._analyze_symbol(
                symbol=symbol,
                news_sentiment=news_sentiments.get(symbol),
                features=features,
                price=prices.get(symbol) if prices else None,
                ma_200_value=ma_200.get(symbol) if ma_200 else None,
                rsi=rsi_values.get(symbol) if rsi_values else None,
                cross_asset_signals=cross_asset_signals,
            )
            
            if candidate and candidate.total_score >= self.config.min_total_score:
                candidates.append(candidate)
        
        # Sort by score
        candidates.sort(key=lambda x: -x.total_score)
        
        # Limit
        candidates = candidates[:self.config.max_short_candidates]
        
        self.last_scan_time = datetime.now()
        self.last_candidates = candidates
        
        logger.info(f"🎯 Short scanner: Found {len(candidates)} candidates")
        for c in candidates[:5]:
            logger.info(f"  {c.symbol}: score={c.total_score:.2f}, conviction={c.conviction}")
        
        return candidates
    
    def _analyze_symbol(
        self,
        symbol: str,
        news_sentiment: Optional[Dict],
        features: Optional[Dict],
        price: Optional[float],
        ma_200_value: Optional[float],
        rsi: Optional[float],
        cross_asset_signals: Optional[Dict[str, float]] = None,
    ) -> Optional[ShortCandidate]:
        """Analyze a single symbol for short opportunity."""
        candidate = ShortCandidate(symbol=symbol)
        sources_signaling = 0
        
        # 1. NEWS ANALYSIS
        news_score = self._analyze_news(symbol, news_sentiment, candidate)
        candidate.news_score = news_score
        if news_score > 0.3:
            sources_signaling += 1
        
        # 2. VALUATION ANALYSIS
        val_score = self._analyze_valuation(symbol, features, price, ma_200_value, candidate)
        candidate.valuation_score = val_score
        if val_score > 0.3:
            sources_signaling += 1
        
        # 3. TECHNICAL ANALYSIS
        tech_score = self._analyze_technicals(symbol, price, ma_200_value, rsi, candidate)
        candidate.technical_score = tech_score
        if tech_score > 0.3:
            sources_signaling += 1
        
        # 4. CROSS-ASSET ANALYSIS (NEW)
        cross_score = self._analyze_cross_asset(symbol, cross_asset_signals, candidate)
        candidate.cross_asset_score = cross_score
        if cross_score > 0.3:
            sources_signaling += 1
        
        # Check minimum sources agreeing (now 2 of 4 possible)
        if sources_signaling < self.config.min_sources_agreeing:
            return None
        
        # Calculate total score
        candidate.calculate_total(self.config.signal_weights)
        
        return candidate
    
    def _analyze_cross_asset(
        self,
        symbol: str,
        cross_asset_signals: Optional[Dict[str, float]],
        candidate: ShortCandidate
    ) -> float:
        """
        Analyze cross-asset signals for short opportunities.
        
        Cross-asset short triggers:
        - Oil declining → Short energy stocks
        - Strong dollar (DXY rising) → Short multinationals
        - Credit stress (HYG falling) → Short financials
        - China weakness → Short China-exposed
        """
        if not cross_asset_signals:
            return 0.0
        
        score = 0.0
        
        # 1. Oil decline → Energy shorts
        oil_signal = cross_asset_signals.get('oil_signal', 0)
        if symbol in self.ENERGY_SYMBOLS and oil_signal < -0.15:
            score += min(0.5, abs(oil_signal))
            candidate.cross_asset_signals.append(f"Oil declining ({oil_signal:.2f})")
        
        # 2. Strong dollar → Multinational shorts
        dxy_signal = cross_asset_signals.get('dxy_signal', 0)
        if symbol in self.MULTINATIONAL_SYMBOLS and dxy_signal > 0.15:
            score += min(0.4, dxy_signal)
            candidate.cross_asset_signals.append(f"Strong dollar ({dxy_signal:.2f})")
        
        # 3. Credit stress → Financial shorts
        credit_signal = cross_asset_signals.get('credit_signal', 0)
        if symbol in self.FINANCIAL_SYMBOLS and credit_signal < -0.2:
            score += min(0.5, abs(credit_signal))
            candidate.cross_asset_signals.append(f"Credit stress ({credit_signal:.2f})")
        
        # 4. China weakness → China-exposed shorts
        china_signal = cross_asset_signals.get('china_signal', 0)
        if symbol in self.CHINA_EXPOSED_SYMBOLS and china_signal < -0.2:
            score += min(0.4, abs(china_signal))
            candidate.cross_asset_signals.append(f"China weakness ({china_signal:.2f})")
        
        # 5. Europe weakness → EU-exposed
        europe_signal = cross_asset_signals.get('europe_lead', 0)
        if europe_signal < -0.2:
            # General market weakness signal
            score += min(0.2, abs(europe_signal))
            candidate.cross_asset_signals.append(f"Europe weak ({europe_signal:.2f})")
        
        return min(1.0, score)
    
    def _analyze_news(
        self,
        symbol: str,
        news_sentiment: Optional[Dict],
        candidate: ShortCandidate
    ) -> float:
        """
        Analyze news for short signals.
        
        Looks for:
        - Negative sentiment overall
        - Earnings miss keywords
        - Analyst downgrade keywords
        - Regulatory/legal issue keywords
        """
        if not news_sentiment:
            return 0.0
        
        score = 0.0
        
        # Get sentiment score
        sent_score = news_sentiment.get('sentiment_score', 0)
        sent_conf = news_sentiment.get('sentiment_confidence', 0.5)
        
        # Negative sentiment = short signal
        if sent_score < self.config.news_sentiment_threshold:
            # Scale: -0.3 to -1.0 maps to 0.3 to 1.0
            score = min(1.0, abs(sent_score) * sent_conf)
            candidate.news_signals.append(f"Bearish sentiment: {sent_score:.2f}")
        
        # Check for specific catalysts in headlines (if available)
        headlines = news_sentiment.get('headlines', [])
        headline_text = ' '.join(headlines).lower() if headlines else ''
        
        # Earnings miss
        earnings_keywords = ['miss', 'disappoints', 'below expectations', 'guidance cut', 'lowers outlook']
        for kw in earnings_keywords:
            if kw in headline_text:
                score = min(1.0, score + self.config.earnings_miss_boost)
                candidate.news_signals.append(f"Earnings concern: '{kw}'")
                break
        
        # Analyst downgrade
        downgrade_keywords = ['downgrade', 'sell rating', 'underperform', 'price target cut']
        for kw in downgrade_keywords:
            if kw in headline_text:
                score = min(1.0, score + self.config.downgrade_boost)
                candidate.news_signals.append(f"Analyst action: '{kw}'")
                break
        
        # Regulatory/legal
        regulatory_keywords = ['sec investigation', 'lawsuit', 'fda reject', 'recall', 'fraud']
        for kw in regulatory_keywords:
            if kw in headline_text:
                score = min(1.0, score + self.config.regulatory_boost)
                candidate.news_signals.append(f"Regulatory concern: '{kw}'")
                break
        
        return min(1.0, score)
    
    def _analyze_valuation(
        self,
        symbol: str,
        features: Optional[Dict],
        price: Optional[float],
        ma_200_value: Optional[float],
        candidate: ShortCandidate
    ) -> float:
        """
        Analyze valuation for short signals.
        
        Looks for:
        - P/E significantly above sector average
        - Price significantly above 200 DMA (extended)
        """
        score = 0.0
        
        # Price extension above 200 DMA
        if price and ma_200_value and ma_200_value > 0:
            extension = (price - ma_200_value) / ma_200_value
            
            if extension > self.config.price_above_200ma_threshold:
                # Extended above MA = potential short
                # Map 15%-50% extension to 0.2-0.6 score
                ext_score = min(0.6, 0.2 + extension * 0.8)
                score += ext_score
                candidate.valuation_signals.append(f"Extended {extension:.0%} above 200 DMA")
        
        # Check P/E if available in features
        if features:
            pe_ratio = features.get('pe_ratio', {}).get(symbol)
            sector_pe = features.get('sector_pe', {}).get(symbol)
            
            if pe_ratio and sector_pe and sector_pe > 0:
                pe_vs_sector = pe_ratio / sector_pe
                
                if pe_vs_sector > self.config.pe_vs_sector_threshold:
                    # High P/E vs sector = overvalued
                    pe_score = min(0.5, 0.2 + (pe_vs_sector - 2.0) * 0.15)
                    score += pe_score
                    candidate.valuation_signals.append(f"P/E {pe_vs_sector:.1f}x sector average")
        
        return min(1.0, score)
    
    def _analyze_technicals(
        self,
        symbol: str,
        price: Optional[float],
        ma_200_value: Optional[float],
        rsi: Optional[float],
        candidate: ShortCandidate
    ) -> float:
        """
        Analyze technicals for short signals.
        
        Looks for:
        - Price ABOVE 200 DMA by >30% (overextended, mean reversion opportunity)
        - Price below 200 DMA (breakdown)
        - RSI overbought (reversal setup)
        """
        score = 0.0
        
        # RSI overbought - key short signal
        if rsi and rsi > self.config.rsi_overbought:
            # Map 70-100 to 0.3-0.6
            rsi_score = min(0.6, 0.3 + (rsi - 70) / 100)
            score += rsi_score
            candidate.technical_signals.append(f"RSI overbought: {rsi:.1f}")
        
        # Price analysis relative to 200 DMA
        if price and ma_200_value and ma_200_value > 0:
            pct_from_ma = (price - ma_200_value) / ma_200_value
            
            # Price ABOVE 200 DMA (overextended) - prime short candidate
            if pct_from_ma > 0.15:  # >15% above MA200 (lowered from 30%)
                # Scale score: 15%-50% above maps to 0.2-0.6
                extended_score = min(0.6, 0.2 + (pct_from_ma - 0.15) * 1.15)
                score += extended_score
                candidate.technical_signals.append(f"Extended {pct_from_ma:.0%} above 200 DMA")
            
            # Price below 200 DMA (breakdown) - also bearish
            elif pct_from_ma < 0:
                below_pct = abs(pct_from_ma)
                below_score = min(0.4, 0.2 + below_pct)
                score += below_score
                candidate.technical_signals.append(f"Below 200 DMA by {below_pct:.0%}")
        
        return min(1.0, score)
    
    def get_short_weights(self) -> Dict[str, float]:
        """
        Get recommended short weights from last scan.
        
        Returns:
            Dict of symbol -> negative weight for shorts
        """
        return {
            c.symbol: c.recommended_weight
            for c in self.last_candidates
            if c.recommended_weight < 0
        }
    
    def get_summary(self) -> Dict:
        """Get summary of last scan."""
        if not self.last_candidates:
            return {
                'scan_time': None,
                'candidates': 0,
                'high_conviction': 0,
                'medium_conviction': 0,
                'low_conviction': 0,
            }
        
        return {
            'scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'candidates': len(self.last_candidates),
            'high_conviction': len([c for c in self.last_candidates if c.conviction == 'high']),
            'medium_conviction': len([c for c in self.last_candidates if c.conviction == 'medium']),
            'low_conviction': len([c for c in self.last_candidates if c.conviction == 'low']),
            'top_candidates': [
                {
                    'symbol': c.symbol,
                    'score': c.total_score,
                    'conviction': c.conviction,
                    'news_signals': c.news_signals[:2],
                    'technical_signals': c.technical_signals[:2],
                }
                for c in self.last_candidates[:5]
            ]
        }


# Global instance
_scanner: Optional[ShortScanner] = None


def get_short_scanner(config: Optional[ShortScannerConfig] = None) -> ShortScanner:
    """Get or create the global short scanner instance."""
    global _scanner
    if _scanner is None or config is not None:
        _scanner = ShortScanner(config)
    return _scanner
