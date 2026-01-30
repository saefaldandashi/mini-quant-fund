"""
Fast Debate Engine for HFT-Lite Trading

This module provides rule-based strategy scoring without LLM calls.
Designed for <1 second debate resolution for intraday trading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Strategy categories
INTRADAY_STRATEGIES = {
    'IntradayMomentum', 'VWAPReversion', 'VolumeSpike',
    'RelativeStrengthIntraday', 'OpeningRangeBreakout', 'QuickMeanReversion',
    'NewsSentimentEvent',
}

POSITION_STRATEGIES = {
    'TimeSeriesMomentum', 'CrossSectionMomentum', 'MeanReversion',
    'VolatilityRegimeVolTarget', 'CarryStrategy', 'ValueQualityTilt',
    'RiskParityMinVar', 'TailRiskOverlay',
}

LONG_SHORT_STRATEGIES = {
    'CS_Momentum_LS', 'TS_Momentum_LS', 'MeanReversion_LS', 'QualityValue_LS',
}


@dataclass
class FastDebateScore:
    """Fast scoring result."""
    strategy_name: str
    total_score: float
    confidence_score: float
    urgency_score: float
    regime_score: float
    history_score: float
    blend_weight: float  # Intraday vs position blend


@dataclass
class MarketContext:
    """Current market conditions for fast scoring."""
    vix: float = 20.0
    spy_change_pct: float = 0.0
    hour_of_day: int = 12
    minute_of_hour: int = 0
    news_velocity: int = 0  # Articles in last 30 min
    regime: str = 'neutral'  # 'high_vol', 'low_vol', 'trending_up', 'trending_down', 'neutral'
    
    @classmethod
    def from_features(cls, features, current_time: datetime):
        """Build context from features."""
        vix = 20.0
        spy_change = 0.0
        
        if hasattr(features, 'vix') and features.vix:
            vix = features.vix
        
        if hasattr(features, 'returns_1d') and features.returns_1d:
            spy_change = features.returns_1d.get('SPY', 0) * 100
        
        # Determine regime
        regime = 'neutral'
        if vix > 25:
            regime = 'high_vol'
        elif vix < 15:
            regime = 'low_vol'
        elif spy_change > 1.0:
            regime = 'trending_up'
        elif spy_change < -1.0:
            regime = 'trending_down'
        
        return cls(
            vix=vix,
            spy_change_pct=spy_change,
            hour_of_day=current_time.hour,
            minute_of_hour=current_time.minute,
            regime=regime,
        )


def get_strategy_blend_weights(context: MarketContext) -> Dict[str, float]:
    """
    Determine blend weights for intraday vs position strategies.
    
    Returns:
        Dict with 'intraday' and 'position' weights (sum to 1.0)
    """
    intraday_weight = 0.6  # Default: slightly favor intraday for HFT-lite
    position_weight = 0.4
    
    # VIX adjustment (high vol = more intraday, quick exits)
    if context.vix > 30:
        intraday_weight += 0.25
        position_weight -= 0.25
    elif context.vix > 25:
        intraday_weight += 0.15
        position_weight -= 0.15
    elif context.vix < 15:
        # Low vol, trends persist, favor position
        intraday_weight -= 0.15
        position_weight += 0.15
    
    # Time of day adjustment
    if 9 <= context.hour_of_day <= 10:
        # First hour: Opening Range Breakout is most effective
        intraday_weight += 0.1
    elif 15 <= context.hour_of_day <= 16:
        # Last hour: End-of-day momentum/reversion
        intraday_weight += 0.05
    elif 11 <= context.hour_of_day <= 14:
        # Midday lull: positions can work
        position_weight += 0.05
    
    # News velocity (many articles = breaking event = intraday)
    if context.news_velocity > 10:
        intraday_weight += 0.15
    elif context.news_velocity > 5:
        intraday_weight += 0.08
    
    # Normalize
    total = intraday_weight + position_weight
    return {
        'intraday': max(0.2, min(0.9, intraday_weight / total)),
        'position': max(0.1, min(0.8, position_weight / total)),
    }


def fast_score_strategy(
    strategy_name: str,
    signal_confidence: float,
    signal_urgency: str,
    context: MarketContext,
    historical_accuracy: float = 0.5,
) -> FastDebateScore:
    """
    Fast rule-based strategy scoring (no LLM).
    
    Args:
        strategy_name: Name of the strategy
        signal_confidence: Strategy's reported confidence (0-1)
        signal_urgency: 'immediate', 'normal', or 'patient'
        context: Current market context
        historical_accuracy: Historical win rate (0-1)
    
    Returns:
        FastDebateScore with breakdown
    """
    # Base confidence score (0-0.3)
    confidence_score = signal_confidence * 0.3
    
    # Urgency score (0-0.2)
    urgency_map = {'immediate': 0.2, 'normal': 0.1, 'patient': 0.05}
    urgency_score = urgency_map.get(signal_urgency, 0.1)
    
    # Regime alignment score (0-0.25)
    regime_score = 0.1  # Base
    
    if strategy_name in INTRADAY_STRATEGIES:
        if context.regime == 'high_vol':
            regime_score = 0.25  # Intraday shines in volatility
        elif context.regime in ('trending_up', 'trending_down'):
            regime_score = 0.15  # Momentum can work
        elif context.regime == 'low_vol':
            regime_score = 0.08  # Less opportunity
    elif strategy_name in POSITION_STRATEGIES:
        if context.regime == 'low_vol':
            regime_score = 0.20  # Trends persist
        elif context.regime in ('trending_up', 'trending_down'):
            regime_score = 0.18
        elif context.regime == 'high_vol':
            regime_score = 0.05  # Position strategies struggle
    elif strategy_name in LONG_SHORT_STRATEGIES:
        # L/S strategies can profit in ANY regime - boost significantly
        if context.regime == 'high_vol':
            regime_score = 0.25  # Can profit both ways - excellent in volatility
        elif context.regime in ('trending_up', 'trending_down'):
            regime_score = 0.22  # Market-neutral can capitalize on spreads
        else:
            regime_score = 0.18  # Still valuable for hedging and alpha
    
    # Historical performance score (0-0.25)
    history_score = historical_accuracy * 0.25
    
    # Get blend weight
    blend_weights = get_strategy_blend_weights(context)
    if strategy_name in INTRADAY_STRATEGIES:
        blend_weight = blend_weights['intraday']
    elif strategy_name in POSITION_STRATEGIES:
        blend_weight = blend_weights['position']
    elif strategy_name in LONG_SHORT_STRATEGIES:
        # L/S strategies are CRITICAL for shorts - boost them!
        # Average of intraday and position, plus a 20% boost to ensure shorts are generated
        blend_weight = (blend_weights['intraday'] + blend_weights['position']) / 2 + 0.20
    else:
        blend_weight = 0.5  # Unknown strategies get equal weight
    
    # Total score
    raw_score = confidence_score + urgency_score + regime_score + history_score
    total_score = raw_score * blend_weight * 2  # Scale by blend weight
    
    # Clamp to 0-1
    total_score = max(0.0, min(1.0, total_score))
    
    return FastDebateScore(
        strategy_name=strategy_name,
        total_score=total_score,
        confidence_score=confidence_score,
        urgency_score=urgency_score,
        regime_score=regime_score,
        history_score=history_score,
        blend_weight=blend_weight,
    )


def fast_debate(
    signals: Dict[str, Any],
    features: Any,
    current_time: datetime,
    historical_performance: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Fast rule-based debate (no LLM calls).
    
    Args:
        signals: Dict of strategy_name -> SignalOutput
        features: Current market features
        current_time: Current timestamp
        historical_performance: Dict of strategy_name -> accuracy
    
    Returns:
        Tuple of (strategy_scores, metadata)
    """
    import time
    start = time.time()
    
    # Build market context
    context = MarketContext.from_features(features, current_time)
    
    historical_performance = historical_performance or {}
    
    scores = {}
    score_details = {}
    
    for name, signal in signals.items():
        confidence = getattr(signal, 'confidence', 0.5)
        urgency = getattr(signal, 'urgency', 'normal')
        
        result = fast_score_strategy(
            strategy_name=name,
            signal_confidence=confidence,
            signal_urgency=urgency,
            context=context,
            historical_accuracy=historical_performance.get(name, 0.5),
        )
        
        scores[name] = result.total_score
        score_details[name] = {
            'confidence': result.confidence_score,
            'urgency': result.urgency_score,
            'regime': result.regime_score,
            'history': result.history_score,
            'blend_weight': result.blend_weight,
        }
    
    elapsed = (time.time() - start) * 1000
    
    metadata = {
        'debate_type': 'fast_rule_based',
        'execution_time_ms': elapsed,
        'market_context': {
            'vix': context.vix,
            'regime': context.regime,
            'hour': context.hour_of_day,
        },
        'blend_weights': get_strategy_blend_weights(context),
        'score_details': score_details,
    }
    
    logger.info(f"⚡ Fast debate completed in {elapsed:.1f}ms ({len(signals)} strategies)")
    
    return scores, metadata


def get_news_urgency_weight(article_time: datetime, current_time: datetime) -> float:
    """
    Calculate urgency weight based on news recency.
    
    More recent news gets higher weight for intraday trading.
    """
    age_seconds = (current_time - article_time).total_seconds()
    age_minutes = age_seconds / 60
    
    if age_minutes < 5:
        return 2.0   # Breaking news: 2x weight
    elif age_minutes < 15:
        return 1.5   # Very recent: 1.5x weight
    elif age_minutes < 30:
        return 1.2   # Recent: 1.2x weight
    elif age_minutes < 60:
        return 1.0   # Normal: 1x weight
    elif age_minutes < 240:
        return 0.5   # Old: 0.5x weight
    else:
        return 0.2   # Stale: 0.2x weight


def calculate_news_velocity(articles: List[Dict], current_time: datetime, window_minutes: int = 30) -> int:
    """
    Count articles published in the last N minutes.
    
    High velocity = breaking news event = favor intraday strategies.
    """
    count = 0
    cutoff = current_time.timestamp() - (window_minutes * 60)
    
    for article in articles:
        pub_time = article.get('time_published') or article.get('published_at')
        if pub_time:
            try:
                # Parse various time formats
                if isinstance(pub_time, str):
                    if 'T' in pub_time:
                        from datetime import datetime as dt
                        pub_ts = dt.fromisoformat(pub_time.replace('Z', '+00:00')).timestamp()
                    else:
                        # Alpha Vantage format: YYYYMMDDTHHMMSS
                        pub_ts = datetime.strptime(pub_time[:15], '%Y%m%dT%H%M%S').timestamp()
                else:
                    pub_ts = pub_time
                
                if pub_ts >= cutoff:
                    count += 1
            except:
                pass
    
    return count
