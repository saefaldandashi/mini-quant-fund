"""
News Sentiment and Event-Driven Strategy.
Enhanced with:
- Macro/geo/finance news intelligence
- TICKER-LEVEL SENTIMENT FROM ALPHA VANTAGE (the key data we were missing!)
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from .base import Strategy, SignalOutput
from src.data.feature_store import Features
from src.data.sentiment import SentimentScore


class NewsSentimentEventStrategy(Strategy):
    """
    News Sentiment and Event-Driven Strategy.
    
    NOW USES TICKER-LEVEL SENTIMENT FROM ALPHA VANTAGE:
    - Per-stock sentiment scores with relevance weighting
    - Sentiment momentum (improving vs deteriorating)
    - Bullish/bearish ratio from multiple articles
    
    Also integrates with News Intelligence Pipeline:
    - Uses macro indices (inflation, growth, central bank hawkishness)
    - Uses risk sentiment (risk-on/risk-off from news)
    - Adjusts positions based on geopolitical and financial stress
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("NewsSentimentEvent", config)
        self._required_features = ['sentiment', 'volatility_21d']
        
        # Config - lowered threshold since we now have better data
        self.sentiment_threshold = config.get('sentiment_threshold', 0.15) if config else 0.15
        self.min_confidence = config.get('min_confidence', 0.2) if config else 0.2
        self.max_position = config.get('max_position', 0.08) if config else 0.08
        
        # Ticker sentiment storage (set externally by app.py)
        self.ticker_sentiments: Dict = {}
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate sentiment-based signals using macro intelligence."""
        weights = {}
        expected_returns = {}
        explanations = {}
        
        # Get macro intelligence if available
        macro_features = getattr(features, 'macro_features', None)
        risk_sentiment = getattr(features, 'risk_sentiment', None)
        
        # Determine overall market stance from macro intelligence
        macro_stance = 0.0  # -1 = defensive, 0 = neutral, +1 = aggressive
        macro_info = {}
        
        if macro_features:
            # Use macro indices to adjust stance
            macro_stance = (
                macro_features.overall_risk_sentiment_index * 0.4 +  # Risk sentiment
                macro_features.growth_momentum_index * 0.3 +  # Growth direction
                -macro_features.financial_stress_index * 0.15 +  # Stress = defensive
                -macro_features.geopolitical_risk_index * 0.15  # Geo risk = defensive
            )
            macro_stance = np.clip(macro_stance, -1, 1)
            
            macro_info = {
                'risk_sentiment': macro_features.overall_risk_sentiment_index,
                'growth_momentum': macro_features.growth_momentum_index,
                'cb_hawkishness': macro_features.central_bank_hawkishness_index,
                'geo_risk': macro_features.geopolitical_risk_index,
                'fin_stress': macro_features.financial_stress_index,
                'top_events': macro_features.top_events[:3] if macro_features.top_events else [],
            }
        
        if risk_sentiment:
            macro_info['equity_bias'] = risk_sentiment.equity_bias
            macro_info['rates_bias'] = risk_sentiment.rates_bias
        
        # PRIORITY 1: Use TICKER-LEVEL SENTIMENT from Alpha Vantage (THE KEY DATA!)
        ticker_sentiment_used = False
        
        if self.ticker_sentiments:
            for symbol in features.symbols:
                if symbol not in self.ticker_sentiments:
                    continue
                
                ts = self.ticker_sentiments[symbol]
                
                # Get sentiment data - ts is a StockSentimentFeatures object or dict
                if hasattr(ts, 'sentiment_score'):
                    sent_score = ts.sentiment_score
                    confidence = ts.sentiment_confidence
                    momentum = ts.sentiment_momentum
                    news_vol = ts.news_volume
                    bullish_ratio = ts.bullish_ratio
                else:
                    # Dict format
                    sent_score = ts.get('sentiment_score', 0)
                    confidence = ts.get('sentiment_confidence', 0)
                    momentum = ts.get('sentiment_momentum', 0)
                    news_vol = ts.get('news_volume', 0)
                    bullish_ratio = ts.get('bullish_ratio', 0.5)
                
                # Skip low confidence
                if confidence < self.min_confidence:
                    continue
                
                # Only trade on significant sentiment
                if abs(sent_score) > self.sentiment_threshold:
                    # Base weight from sentiment * confidence
                    raw_weight = sent_score * confidence
                    
                    # BOOST for momentum (improving sentiment is more actionable)
                    if momentum > 0 and sent_score > 0:
                        raw_weight *= (1 + min(0.5, momentum))  # Up to 50% boost
                    elif momentum < 0 and sent_score < 0:
                        raw_weight *= (1 + min(0.5, abs(momentum)))  # Amplify shorts with momentum
                    
                    # ADJUST BY MACRO STANCE
                    if macro_stance < -0.2:
                        # Defensive: reduce longs, amplify shorts
                        if raw_weight > 0:
                            raw_weight *= (1 + macro_stance)
                        else:
                            raw_weight *= (1 - macro_stance)
                    elif macro_stance > 0.2:
                        # Aggressive: amplify longs, reduce shorts
                        if raw_weight > 0:
                            raw_weight *= (1 + macro_stance * 0.5)
                        else:
                            raw_weight *= (1 - macro_stance * 0.5)
                    
                    weight = np.clip(raw_weight, -self.max_position, self.max_position)
                    
                    weights[symbol] = weight
                    expected_returns[symbol] = sent_score * 0.25  # Higher expected return with better data
                    
                    explanations[symbol] = {
                        'sentiment': sent_score,
                        'confidence': confidence,
                        'momentum': momentum,
                        'news_volume': news_vol,
                        'bullish_ratio': bullish_ratio,
                        'source': 'alpha_vantage_ticker',
                        'macro_adjusted': macro_stance != 0,
                    }
                    ticker_sentiment_used = True
        
        # FALLBACK: Use old sentiment data if no ticker sentiment
        if not ticker_sentiment_used and features.sentiment:
            for symbol in features.symbols:
                sent: SentimentScore = features.sentiment.get(symbol)
                
                if sent is None or sent.confidence < self.min_confidence:
                    continue
                
                if abs(sent.sentiment_score) > self.sentiment_threshold:
                    raw_weight = sent.sentiment_score * sent.confidence
                    
                    if macro_stance < -0.2:
                        if raw_weight > 0:
                            raw_weight *= (1 + macro_stance)
                        else:
                            raw_weight *= (1 - macro_stance)
                    elif macro_stance > 0.2:
                        if raw_weight > 0:
                            raw_weight *= (1 + macro_stance * 0.5)
                        else:
                            raw_weight *= (1 - macro_stance * 0.5)
                    
                    weight = np.clip(raw_weight, -self.max_position, self.max_position)
                    
                    weights[symbol] = weight
                    expected_returns[symbol] = sent.sentiment_score * 0.2
                    
                    explanations[symbol] = {
                        'sentiment': sent.sentiment_score,
                        'confidence': sent.confidence,
                        'source': 'legacy_sentiment',
                        'macro_adjusted': macro_stance != 0,
                    }
        
        # If no stock-specific sentiment but have macro stance, use equity bias
        if not weights and risk_sentiment and abs(risk_sentiment.equity_bias) > 0.2:
            # Use macro equity bias to tilt portfolio
            bias = risk_sentiment.equity_bias
            n_stocks = min(10, len(features.symbols))
            weight_per_stock = bias * 0.05 / n_stocks  # Small positions
            
            for symbol in features.symbols[:n_stocks]:
                weights[symbol] = weight_per_stock
                expected_returns[symbol] = bias * 0.1
                explanations[symbol] = {'macro_driven': True, 'equity_bias': bias}
        
        # Normalize to prevent over-exposure
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 1.0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        # Categorize signals
        bullish = [s for s, w in weights.items() if w > 0]
        bearish = [s for s, w in weights.items() if w < 0]
        
        # Confidence based on data quality
        confidence = 0.5
        if macro_features and macro_features.data_quality_score > 0:
            confidence = 0.4 + macro_features.data_quality_score * 0.4
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=confidence,
            explanation={
                'bullish_signals': bullish,
                'bearish_signals': bearish,
                'total_signals': len(weights),
                'avg_sentiment': np.mean(list(weights.values())) if weights else 0,
                'macro_stance': macro_stance,
                'macro_info': macro_info,
                'details': explanations,
            },
            regime_fit=0.7 if macro_features else 0.5,
            diversification_score=0.4,
        )
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'note': 'No sentiment data available'},
        )
