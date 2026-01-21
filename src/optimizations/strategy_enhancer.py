"""
Strategy Enhancement Module

Implements the 7-phase enhancement plan:
- Phase 1: Position Sizing Overhaul
- Phase 2: Sentiment-Driven Selection
- Phase 3: Focused Universe Filtering
- Phase 4: Regime Detection
- Phase 5: Debate Improvements
- Phase 6: Active Learning
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedConfig:
    """Configuration for strategy enhancements."""
    # Risk appetite: conservative, moderate, aggressive, maximum
    risk_appetite: str = "moderate"
    
    # Position sizing
    kelly_multiplier: float = 0.5
    min_position_pct: float = 0.02
    max_positions: int = 20
    min_investment_floor: float = 0.50  # Minimum 50% of target
    
    # Sentiment
    min_sentiment_magnitude: float = 0.10
    min_relevance: float = 0.25
    
    # Universe
    max_tradeable_stocks: int = 25
    require_sentiment: bool = True
    require_momentum: bool = True
    
    # Regime
    auto_regime_adjustment: bool = True


@dataclass 
class RegimeState:
    """Current market regime assessment."""
    regime: str  # "strong_bull", "mild_bull", "neutral", "mild_bear", "strong_bear"
    score: float  # 0-1, higher = more bullish
    exposure_multiplier: float  # 0-1, how much to invest
    indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FilteredStock:
    """Stock that passed the filtering pipeline."""
    symbol: str
    sentiment_score: float
    sentiment_momentum: float
    relevance: float
    news_count: int
    momentum_score: float
    conviction_tier: str  # very_high, high, medium
    target_weight: float
    reasons: List[str] = field(default_factory=list)


class StrategyEnhancer:
    """
    Enhances strategy signals with better position sizing,
    sentiment integration, and regime awareness.
    """
    
    def __init__(self, config: Optional[EnhancedConfig] = None):
        self.config = config or EnhancedConfig()
        self.regime_state: Optional[RegimeState] = None
        self.filtered_universe: List[FilteredStock] = []
        
        # Load risk appetite settings
        self._apply_risk_appetite(self.config.risk_appetite)
    
    def _apply_risk_appetite(self, appetite: str):
        """Apply risk appetite settings from config."""
        try:
            import config as app_config
            settings = app_config.RISK_APPETITE_SETTINGS.get(appetite, {})
            if settings:
                self.config.kelly_multiplier = settings.get("kelly_multiplier", 0.5)
                self.config.min_position_pct = settings.get("min_position_pct", 0.02)
                self.config.max_positions = settings.get("max_positions", 20)
                logger.info(f"Applied risk appetite '{appetite}': Kelly={self.config.kelly_multiplier}, "
                           f"MinPos={self.config.min_position_pct}, MaxPos={self.config.max_positions}")
        except Exception as e:
            logger.warning(f"Could not load risk appetite settings: {e}")
    
    # =========================================================
    # PHASE 1: POSITION SIZING
    # =========================================================
    
    def enhance_position_sizes(
        self,
        base_weights: Dict[str, float],
        confidences: Dict[str, float],
        target_exposure: float,
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Enhance position sizes with:
        - Kelly multiplier adjustment
        - Minimum position enforcement
        - Investment floor scaling
        - Maximum positions limit
        
        Returns:
            Enhanced weights and reasons for each adjustment
        """
        reasons = {}
        
        if not base_weights:
            return {}, {}
        
        # Step 1: Apply Kelly multiplier
        adjusted = {}
        for symbol, weight in base_weights.items():
            confidence = confidences.get(symbol, 0.5)
            # Recalibrate confidence (current scores are too conservative)
            calibrated_confidence = self._calibrate_confidence(confidence)
            
            # Apply Kelly multiplier
            new_weight = weight * self.config.kelly_multiplier * calibrated_confidence
            adjusted[symbol] = new_weight
            reasons[symbol] = f"Kelly {self.config.kelly_multiplier}x, conf {calibrated_confidence:.2f}"
        
        # Step 2: Take top N positions by weight first (before min filter)
        sorted_stocks = sorted(adjusted.items(), key=lambda x: -x[1])
        top_positions = dict(sorted_stocks[:self.config.max_positions])
        
        for s in set(adjusted.keys()) - set(top_positions.keys()):
            reasons[s] = f"Not in top {self.config.max_positions}"
        
        # Step 3: Apply investment floor BEFORE min position filter
        # This ensures we have enough allocation to meet minimums
        total_weight = sum(top_positions.values())
        if total_weight < self.config.min_investment_floor and total_weight > 0:
            scale_factor = self.config.min_investment_floor / total_weight
            top_positions = {s: w * scale_factor for s, w in top_positions.items()}
            for s in top_positions:
                reasons[s] = f"{reasons.get(s, '')} | Scaled {scale_factor:.2f}x for floor"
            logger.info(f"Applied investment floor: scaled {scale_factor:.2f}x to reach {self.config.min_investment_floor*100:.0f}%")
        
        # Step 4: Now filter by minimum position size
        min_weight = self.config.min_position_pct
        filtered = {s: w for s, w in top_positions.items() if w >= min_weight}
        
        # If min filter removes too many, keep at least some positions
        if len(filtered) < 5 and len(top_positions) >= 5:
            # Keep top 5-10 regardless of min size
            sorted_remaining = sorted(top_positions.items(), key=lambda x: -x[1])
            filtered = dict(sorted_remaining[:min(10, len(sorted_remaining))])
            logger.info(f"Relaxed min position filter to keep {len(filtered)} positions")
        
        for s in set(top_positions.keys()) - set(filtered.keys()):
            if s not in reasons or "Scaled" not in reasons.get(s, ''):
                reasons[s] = f"Below min position {min_weight*100:.1f}%"
        
        # Step 5: Normalize if exceeds 100%
        total_weight = sum(filtered.values())
        if total_weight > 1.0:
            filtered = {s: w / total_weight for s, w in filtered.items()}
        
        return filtered, reasons
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Recalibrate confidence scores (they're too conservative)."""
        # Map 0-0.3 -> 0-0.2, 0.3-0.6 -> 0.3-0.7, 0.6-1.0 -> 0.7-1.0
        if raw_confidence <= 0.3:
            return raw_confidence * 0.67
        elif raw_confidence <= 0.6:
            return 0.2 + (raw_confidence - 0.3) * 1.33
        else:
            return 0.7 + (raw_confidence - 0.6) * 0.75
    
    # =========================================================
    # PHASE 2: SENTIMENT-DRIVEN SELECTION
    # =========================================================
    
    def filter_by_sentiment(
        self,
        symbols: List[str],
        ticker_sentiments: Dict[str, any],
    ) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Filter stocks based on sentiment quality.
        
        Returns:
            Filtered symbols and sentiment details
        """
        filtered = []
        details = {}
        
        for symbol in symbols:
            sentiment = ticker_sentiments.get(symbol)
            
            if sentiment is None:
                if not self.config.require_sentiment:
                    filtered.append(symbol)
                    details[symbol] = {"status": "no_sentiment", "included": not self.config.require_sentiment}
                continue
            
            # Extract sentiment features
            score = getattr(sentiment, 'sentiment_score', 0) if hasattr(sentiment, 'sentiment_score') else sentiment.get('sentiment_score', 0)
            relevance = getattr(sentiment, 'confidence', 0.5) if hasattr(sentiment, 'confidence') else sentiment.get('confidence', 0.5)
            news_count = getattr(sentiment, 'news_volume', 0) if hasattr(sentiment, 'news_volume') else sentiment.get('news_volume', 0)
            
            # Check thresholds
            magnitude = abs(score)
            passes = True
            reason = []
            
            if magnitude < self.config.min_sentiment_magnitude:
                passes = False
                reason.append(f"weak signal ({score:.2f})")
            
            if relevance < self.config.min_relevance:
                passes = False
                reason.append(f"low relevance ({relevance:.2f})")
            
            if passes:
                filtered.append(symbol)
            
            details[symbol] = {
                "score": score,
                "relevance": relevance,
                "news_count": news_count,
                "magnitude": magnitude,
                "passes": passes,
                "reason": ", ".join(reason) if reason else "passed"
            }
        
        logger.info(f"Sentiment filter: {len(filtered)}/{len(symbols)} passed")
        return filtered, details
    
    def calculate_sentiment_momentum(
        self,
        symbol: str,
        historical_sentiment: List[Tuple[datetime, float]],
    ) -> float:
        """Calculate sentiment momentum (7-day change)."""
        if not historical_sentiment or len(historical_sentiment) < 2:
            return 0.0
        
        # Sort by date
        sorted_sent = sorted(historical_sentiment, key=lambda x: x[0])
        
        # Get recent vs older average
        mid = len(sorted_sent) // 2
        recent_avg = np.mean([s[1] for s in sorted_sent[mid:]])
        older_avg = np.mean([s[1] for s in sorted_sent[:mid]])
        
        return recent_avg - older_avg
    
    # =========================================================
    # PHASE 3: FOCUSED UNIVERSE
    # =========================================================
    
    def filter_universe(
        self,
        symbols: List[str],
        prices: Dict[str, float],
        volumes: Dict[str, float],
        momentum_scores: Dict[str, float],
        sentiment_scores: Dict[str, float],
        conviction_scores: Dict[str, float],
    ) -> List[FilteredStock]:
        """
        Apply the full filtering pipeline to reduce universe.
        
        Pipeline:
        1. Sentiment filter (has recent news + clear signal)
        2. Momentum filter (price > 20-day MA)
        3. Top conviction (highest combined score)
        """
        candidates = []
        
        for symbol in symbols:
            price = prices.get(symbol, 0)
            volume = volumes.get(symbol, 0)
            momentum = momentum_scores.get(symbol, 0)
            sentiment = sentiment_scores.get(symbol, 0)
            conviction = conviction_scores.get(symbol, 0)
            
            reasons = []
            
            # Check momentum filter
            if self.config.require_momentum and momentum <= 0:
                continue
            elif momentum > 0:
                reasons.append(f"Momentum: +{momentum:.1%}")
            
            # Check sentiment
            if self.config.require_sentiment and abs(sentiment) < self.config.min_sentiment_magnitude:
                continue
            elif abs(sentiment) >= self.config.min_sentiment_magnitude:
                direction = "Bullish" if sentiment > 0 else "Bearish"
                reasons.append(f"Sentiment: {direction} ({sentiment:.2f})")
            
            # Determine conviction tier
            tier, target_weight = self._get_conviction_tier(conviction)
            if tier == "low":
                continue
            reasons.append(f"Conviction: {tier} ({conviction:.2f})")
            
            candidates.append(FilteredStock(
                symbol=symbol,
                sentiment_score=sentiment,
                sentiment_momentum=0,  # Would need historical data
                relevance=0.5,
                news_count=0,
                momentum_score=momentum,
                conviction_tier=tier,
                target_weight=target_weight,
                reasons=reasons,
            ))
        
        # Sort by conviction and take top N
        candidates.sort(key=lambda x: -conviction_scores.get(x.symbol, 0))
        self.filtered_universe = candidates[:self.config.max_tradeable_stocks]
        
        logger.info(f"Universe filter: {len(self.filtered_universe)}/{len(symbols)} stocks selected")
        return self.filtered_universe
    
    def _get_conviction_tier(self, score: float) -> Tuple[str, float]:
        """Map conviction score to tier and target weight."""
        try:
            import config as app_config
            for tier, settings in app_config.CONVICTION_TIERS.items():
                if score >= settings["min_score"]:
                    return tier, settings["position_pct"]
        except:
            pass
        
        # Default tiers
        if score >= 0.8:
            return "very_high", 0.10
        elif score >= 0.6:
            return "high", 0.07
        elif score >= 0.4:
            return "medium", 0.04
        else:
            return "low", 0.0
    
    # =========================================================
    # PHASE 4: REGIME DETECTION
    # =========================================================
    
    def detect_regime(
        self,
        spy_price: float,
        spy_200ma: float,
        vix: float,
        macro_sentiment: float,
        geo_risk: float,
        financial_stress: float,
        breadth: float = 0.5,
    ) -> RegimeState:
        """
        Detect current market regime and recommended exposure.
        """
        indicators = {}
        
        # SPY vs 200-day MA (0 = below, 1 = above)
        spy_signal = 1.0 if spy_price > spy_200ma else 0.0
        indicators["spy_vs_200ma"] = spy_signal
        
        # VIX (higher = more bearish)
        if vix < 15:
            vix_signal = 1.0
        elif vix < 20:
            vix_signal = 0.7
        elif vix < 25:
            vix_signal = 0.4
        elif vix < 30:
            vix_signal = 0.2
        else:
            vix_signal = 0.0
        indicators["vix"] = vix_signal
        
        # Macro sentiment (-1 to 1 -> 0 to 1)
        macro_signal = (macro_sentiment + 1) / 2
        indicators["macro_sentiment"] = macro_signal
        
        # Geo risk (higher = worse, invert)
        geo_signal = 1.0 - min(1.0, geo_risk)
        indicators["geo_risk"] = geo_signal
        
        # Financial stress (higher = worse, invert)
        stress_signal = 1.0 - min(1.0, financial_stress)
        indicators["financial_stress"] = stress_signal
        
        # Breadth (already 0-1)
        indicators["breadth"] = breadth
        
        # Calculate weighted score
        try:
            import config as app_config
            weights = app_config.REGIME_SETTINGS["indicators"]
        except:
            weights = {
                "spy_vs_200ma": 0.25,
                "vix": 0.20,
                "macro_sentiment": 0.20,
                "geo_risk": 0.15,
                "financial_stress": 0.10,
                "breadth": 0.10,
            }
        
        score = sum(
            indicators.get(k.replace("_level", ""), 0.5) * v 
            for k, v in weights.items()
        )
        
        # Determine regime
        if score >= 0.7:
            regime = "strong_bull"
            exposure = 1.0
        elif score >= 0.55:
            regime = "mild_bull"
            exposure = 0.8
        elif score >= 0.45:
            regime = "neutral"
            exposure = 0.6
        elif score >= 0.3:
            regime = "mild_bear"
            exposure = 0.4
        else:
            regime = "strong_bear"
            exposure = 0.2
        
        self.regime_state = RegimeState(
            regime=regime,
            score=score,
            exposure_multiplier=exposure,
            indicators=indicators,
            timestamp=datetime.now(),
        )
        
        logger.info(f"Regime detected: {regime} (score={score:.2f}, exposure={exposure:.0%})")
        return self.regime_state
    
    def get_regime_adjusted_exposure(self, base_exposure: float) -> float:
        """Adjust exposure based on current regime."""
        if not self.config.auto_regime_adjustment or not self.regime_state:
            return base_exposure
        
        return base_exposure * self.regime_state.exposure_multiplier
    
    # =========================================================
    # PHASE 5 & 6: DEBATE & LEARNING (Helpers)
    # =========================================================
    
    def get_strategy_weight_adjustments(
        self,
        strategy_performance: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate weight adjustments based on recent performance.
        Top 3 strategies get boost, bottom 3 get penalty.
        """
        if not strategy_performance:
            return {}
        
        sorted_strats = sorted(strategy_performance.items(), key=lambda x: -x[1])
        n = len(sorted_strats)
        
        adjustments = {}
        for i, (strat, perf) in enumerate(sorted_strats):
            if i < n // 3:  # Top third
                adjustments[strat] = 1.5
            elif i < 2 * n // 3:  # Middle third
                adjustments[strat] = 1.0
            else:  # Bottom third
                adjustments[strat] = 0.5
        
        return adjustments
    
    def calculate_consensus_scaling(
        self,
        symbol: str,
        strategy_votes: Dict[str, bool],
    ) -> float:
        """
        Calculate position scaling based on strategy consensus.
        """
        votes_for = sum(1 for v in strategy_votes.values() if v)
        total = len(strategy_votes)
        
        if total == 0:
            return 0.0
        
        if votes_for >= 7:
            return 1.0
        elif votes_for >= 5:
            return 0.7
        elif votes_for >= 4:
            return 0.4
        else:
            return 0.0


# Global instance
_enhancer: Optional[StrategyEnhancer] = None

def get_enhancer(config: Optional[EnhancedConfig] = None) -> StrategyEnhancer:
    """Get or create the global enhancer instance."""
    global _enhancer
    if _enhancer is None or config is not None:
        _enhancer = StrategyEnhancer(config)
    return _enhancer
