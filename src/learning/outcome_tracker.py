"""
Outcome Tracker - Links signals to actual P&L outcomes.

This is CRITICAL for validating whether our signals are actually predictive.
Without this, we're flying blind.
"""
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Record of a signal we generated."""
    signal_id: str
    timestamp: datetime
    ticker: str
    direction: str  # 'long', 'short', 'neutral'
    weight: float  # Position weight
    confidence: float
    
    # Context at signal time
    sentiment_score: Optional[float] = None
    macro_stance: Optional[float] = None
    regime: Optional[str] = None
    strategy_source: Optional[str] = None
    
    # Outcome (filled later)
    outcome_1d: Optional[float] = None  # Return after 1 day
    outcome_5d: Optional[float] = None  # Return after 5 days
    outcome_timestamp: Optional[datetime] = None
    was_correct: Optional[bool] = None


@dataclass
class OutcomeTracker:
    """
    Tracks signals and their outcomes to measure predictive accuracy.
    
    Key Questions We Can Answer:
    - Are our signals predictive?
    - Which strategies produce winning signals?
    - Which market regimes are we good/bad at?
    - Is sentiment data actually helping?
    """
    
    storage_path: str = "outputs/signal_outcomes.json"
    
    # Signal storage
    signals: List[SignalRecord] = field(default_factory=list)
    
    # Metrics cache
    _metrics_cache: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.signals = []
        self._metrics_cache = {}
        self._load()
    
    def _load(self):
        """Load signals from storage."""
        path = Path(self.storage_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                for item in data.get('signals', []):
                    try:
                        # Parse timestamps
                        ts = datetime.fromisoformat(item['timestamp'])
                        outcome_ts = None
                        if item.get('outcome_timestamp'):
                            outcome_ts = datetime.fromisoformat(item['outcome_timestamp'])
                        
                        self.signals.append(SignalRecord(
                            signal_id=item['signal_id'],
                            timestamp=ts,
                            ticker=item['ticker'],
                            direction=item['direction'],
                            weight=item['weight'],
                            confidence=item['confidence'],
                            sentiment_score=item.get('sentiment_score'),
                            macro_stance=item.get('macro_stance'),
                            regime=item.get('regime'),
                            strategy_source=item.get('strategy_source'),
                            outcome_1d=item.get('outcome_1d'),
                            outcome_5d=item.get('outcome_5d'),
                            outcome_timestamp=outcome_ts,
                            was_correct=item.get('was_correct'),
                        ))
                    except Exception as e:
                        logger.debug(f"Error loading signal: {e}")
                
                logger.info(f"Loaded {len(self.signals)} signal records")
                
            except Exception as e:
                logger.warning(f"Could not load outcome tracker: {e}")
    
    def _save(self):
        """Save signals to storage."""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'signals': [],
                'last_updated': datetime.now().isoformat(),
            }
            
            for sig in self.signals:
                data['signals'].append({
                    'signal_id': sig.signal_id,
                    'timestamp': sig.timestamp.isoformat(),
                    'ticker': sig.ticker,
                    'direction': sig.direction,
                    'weight': float(sig.weight) if sig.weight is not None else None,
                    'confidence': float(sig.confidence) if sig.confidence is not None else None,
                    'sentiment_score': float(sig.sentiment_score) if sig.sentiment_score is not None else None,
                    'macro_stance': float(sig.macro_stance) if sig.macro_stance is not None else None,
                    'regime': sig.regime,
                    'strategy_source': sig.strategy_source,
                    'outcome_1d': float(sig.outcome_1d) if sig.outcome_1d is not None else None,
                    'outcome_5d': float(sig.outcome_5d) if sig.outcome_5d is not None else None,
                    'outcome_timestamp': sig.outcome_timestamp.isoformat() if sig.outcome_timestamp else None,
                    'was_correct': bool(sig.was_correct) if sig.was_correct is not None else None,
                })
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save outcome tracker: {e}")
    
    def record_signal(
        self,
        ticker: str,
        direction: str,
        weight: float,
        confidence: float,
        sentiment_score: Optional[float] = None,
        macro_stance: Optional[float] = None,
        regime: Optional[str] = None,
        strategy_source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Record a new signal. Returns signal_id.
        """
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        signal_id = f"{ticker}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        signal = SignalRecord(
            signal_id=signal_id,
            timestamp=timestamp,
            ticker=ticker,
            direction=direction,
            weight=weight,
            confidence=confidence,
            sentiment_score=sentiment_score,
            macro_stance=macro_stance,
            regime=regime,
            strategy_source=strategy_source,
        )
        
        self.signals.append(signal)
        self._save()
        
        return signal_id
    
    def record_batch_signals(
        self,
        weights: Dict[str, float],
        confidences: Dict[str, float],
        sentiment_scores: Optional[Dict[str, float]] = None,
        macro_stance: Optional[float] = None,
        regime: Optional[str] = None,
        strategy_source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> List[str]:
        """Record multiple signals at once."""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        
        signal_ids = []
        
        for ticker, weight in weights.items():
            if abs(weight) < 0.001:
                continue
            
            direction = 'long' if weight > 0 else 'short'
            confidence = confidences.get(ticker, 0.5)
            sent_score = sentiment_scores.get(ticker) if sentiment_scores else None
            
            signal_id = self.record_signal(
                ticker=ticker,
                direction=direction,
                weight=weight,
                confidence=confidence,
                sentiment_score=sent_score,
                macro_stance=macro_stance,
                regime=regime,
                strategy_source=strategy_source,
                timestamp=timestamp,
            )
            signal_ids.append(signal_id)
        
        return signal_ids
    
    def update_outcomes(
        self,
        price_returns: Dict[str, Dict[str, float]],  # ticker -> {'1d': return, '5d': return}
    ) -> int:
        """
        Update outcomes for signals that don't have them yet.
        Returns number of signals updated.
        """
        updated = 0
        now = datetime.now(pytz.UTC)
        
        for signal in self.signals:
            # Skip if already has outcome
            if signal.outcome_5d is not None:
                continue
            
            # Skip if signal too recent (need at least 1 day)
            signal_age = (now.replace(tzinfo=None) - signal.timestamp.replace(tzinfo=None)).total_seconds() / 3600
            if signal_age < 24:  # Less than 1 day old
                continue
            
            # Check if we have return data
            if signal.ticker in price_returns:
                returns = price_returns[signal.ticker]
                
                signal.outcome_1d = returns.get('1d')
                signal.outcome_5d = returns.get('5d', returns.get('1d'))
                signal.outcome_timestamp = now
                
                # Determine if signal was correct
                if signal.outcome_5d is not None:
                    if signal.direction == 'long':
                        signal.was_correct = signal.outcome_5d > 0
                    elif signal.direction == 'short':
                        signal.was_correct = signal.outcome_5d < 0
                    else:
                        signal.was_correct = None
                
                updated += 1
        
        if updated > 0:
            self._save()
            logger.info(f"Updated outcomes for {updated} signals")
        
        return updated
    
    def get_accuracy_metrics(self) -> Dict:
        """Calculate accuracy metrics for our signals."""
        completed = [s for s in self.signals if s.was_correct is not None]
        
        if not completed:
            return {
                'total_signals': len(self.signals),
                'completed_signals': 0,
                'accuracy': None,
                'message': 'No completed signals yet'
            }
        
        correct = sum(1 for s in completed if s.was_correct)
        accuracy = correct / len(completed)
        
        # By direction
        long_signals = [s for s in completed if s.direction == 'long']
        short_signals = [s for s in completed if s.direction == 'short']
        
        long_accuracy = sum(1 for s in long_signals if s.was_correct) / len(long_signals) if long_signals else None
        short_accuracy = sum(1 for s in short_signals if s.was_correct) / len(short_signals) if short_signals else None
        
        # By regime
        regime_accuracy = {}
        regimes = set(s.regime for s in completed if s.regime)
        for regime in regimes:
            regime_signals = [s for s in completed if s.regime == regime]
            if regime_signals:
                regime_accuracy[regime] = sum(1 for s in regime_signals if s.was_correct) / len(regime_signals)
        
        # By strategy
        strategy_accuracy = {}
        strategies = set(s.strategy_source for s in completed if s.strategy_source)
        for strategy in strategies:
            strat_signals = [s for s in completed if s.strategy_source == strategy]
            if strat_signals:
                strategy_accuracy[strategy] = sum(1 for s in strat_signals if s.was_correct) / len(strat_signals)
        
        # Average returns
        avg_return_correct = None
        avg_return_wrong = None
        
        correct_signals = [s for s in completed if s.was_correct and s.outcome_5d is not None]
        wrong_signals = [s for s in completed if not s.was_correct and s.outcome_5d is not None]
        
        if correct_signals:
            avg_return_correct = sum(abs(s.outcome_5d) for s in correct_signals) / len(correct_signals)
        if wrong_signals:
            avg_return_wrong = sum(abs(s.outcome_5d) for s in wrong_signals) / len(wrong_signals)
        
        return {
            'total_signals': len(self.signals),
            'completed_signals': len(completed),
            'accuracy': accuracy,
            'long_accuracy': long_accuracy,
            'short_accuracy': short_accuracy,
            'regime_accuracy': regime_accuracy,
            'strategy_accuracy': strategy_accuracy,
            'avg_return_when_correct': avg_return_correct,
            'avg_return_when_wrong': avg_return_wrong,
            'win_loss_ratio': avg_return_correct / avg_return_wrong if avg_return_correct and avg_return_wrong else None,
        }
    
    def get_sentiment_effectiveness(self) -> Dict:
        """
        Analyze whether sentiment signals are actually predictive.
        This answers: "Is using Alpha Vantage sentiment worth it?"
        """
        completed = [s for s in self.signals if s.was_correct is not None]
        
        # Signals with sentiment data
        with_sentiment = [s for s in completed if s.sentiment_score is not None]
        without_sentiment = [s for s in completed if s.sentiment_score is None]
        
        result = {
            'signals_with_sentiment': len(with_sentiment),
            'signals_without_sentiment': len(without_sentiment),
        }
        
        if with_sentiment:
            result['accuracy_with_sentiment'] = sum(1 for s in with_sentiment if s.was_correct) / len(with_sentiment)
            
            # High confidence sentiment
            high_conf = [s for s in with_sentiment if abs(s.sentiment_score) > 0.3]
            if high_conf:
                result['accuracy_high_sentiment'] = sum(1 for s in high_conf if s.was_correct) / len(high_conf)
        
        if without_sentiment:
            result['accuracy_without_sentiment'] = sum(1 for s in without_sentiment if s.was_correct) / len(without_sentiment)
        
        # Sentiment value-add
        if 'accuracy_with_sentiment' in result and 'accuracy_without_sentiment' in result:
            result['sentiment_value_add'] = result['accuracy_with_sentiment'] - result['accuracy_without_sentiment']
            result['is_sentiment_helpful'] = result['sentiment_value_add'] > 0.05  # At least 5% better
        
        return result
    
    def print_summary(self):
        """Print summary of signal performance."""
        metrics = self.get_accuracy_metrics()
        sentiment = self.get_sentiment_effectiveness()
        
        print("\n" + "=" * 60)
        print("SIGNAL OUTCOME TRACKER - SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Signals: {metrics['total_signals']}")
        print(f"Completed (with outcomes): {metrics['completed_signals']}")
        
        if metrics['accuracy'] is not None:
            print(f"\n📊 ACCURACY: {metrics['accuracy']:.1%}")
            if metrics['long_accuracy']:
                print(f"   Long signals: {metrics['long_accuracy']:.1%}")
            if metrics['short_accuracy']:
                print(f"   Short signals: {metrics['short_accuracy']:.1%}")
            
            if metrics['regime_accuracy']:
                print("\n📈 By Regime:")
                for regime, acc in metrics['regime_accuracy'].items():
                    print(f"   {regime}: {acc:.1%}")
            
            if metrics['strategy_accuracy']:
                print("\n🎯 By Strategy:")
                for strat, acc in sorted(metrics['strategy_accuracy'].items(), key=lambda x: x[1], reverse=True):
                    print(f"   {strat}: {acc:.1%}")
            
            if metrics['avg_return_when_correct']:
                print(f"\n💰 Avg Return (Correct): {metrics['avg_return_when_correct']:.2%}")
            if metrics['avg_return_when_wrong']:
                print(f"💸 Avg Return (Wrong): {metrics['avg_return_when_wrong']:.2%}")
        else:
            print("\n⏳ No completed signals yet - need more time for outcomes")
        
        print("\n🔍 SENTIMENT EFFECTIVENESS:")
        if sentiment.get('is_sentiment_helpful') is not None:
            if sentiment['is_sentiment_helpful']:
                print(f"   ✅ Sentiment IS helping (+{sentiment['sentiment_value_add']:.1%} accuracy)")
            else:
                print(f"   ⚠️ Sentiment NOT helping ({sentiment['sentiment_value_add']:+.1%})")
        else:
            print("   Not enough data to determine")
        
        print("=" * 60)
