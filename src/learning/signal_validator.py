"""
Signal Validator - Validates signals before trading.

Critical for preventing bad trades based on:
1. Consistency checks (do our signals align with raw data?)
2. Confidence thresholds
3. Data freshness checks
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pytz

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of signal validation."""
    is_valid: bool
    confidence_score: float  # 0-1, how confident we are in the signal
    warnings: List[str]
    blocking_issues: List[str]
    adjustments: Dict[str, Any]  # Suggested adjustments to the signal


class SignalValidator:
    """
    Validates trading signals before execution.
    
    Validation Checks:
    1. Sentiment-Direction Consistency
    2. Data Freshness
    3. Signal Confidence Thresholds
    4. Market Condition Checks
    5. Cross-Signal Consistency
    """
    
    def __init__(
        self,
        min_confidence: float = 0.3,
        max_staleness_hours: float = 24.0,
        consistency_threshold: float = 0.5,
    ):
        self.min_confidence = min_confidence
        self.max_staleness_hours = max_staleness_hours
        self.consistency_threshold = consistency_threshold
        
        # Track validation stats
        self.total_validated = 0
        self.total_passed = 0
        self.total_warnings = 0
        self.total_blocked = 0
    
    def validate_signal(
        self,
        ticker: str,
        signal_direction: str,  # 'long' or 'short'
        signal_weight: float,
        signal_confidence: float,
        ticker_sentiment: Optional[Dict] = None,
        macro_sentiment: Optional[float] = None,
        data_timestamp: Optional[datetime] = None,
        current_time: Optional[datetime] = None,
    ) -> ValidationResult:
        """
        Validate a single trading signal.
        
        Returns ValidationResult with pass/fail and confidence adjustments.
        """
        warnings = []
        blocking_issues = []
        adjustments = {}
        
        confidence_score = signal_confidence
        
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        
        # === CHECK 1: Minimum Confidence ===
        if signal_confidence < self.min_confidence:
            warnings.append(f"Low confidence ({signal_confidence:.2f} < {self.min_confidence})")
            confidence_score *= 0.5
        
        # === CHECK 2: Sentiment-Direction Consistency ===
        if ticker_sentiment:
            sent_score = ticker_sentiment.get('sentiment_score', 0)
            sent_conf = ticker_sentiment.get('sentiment_confidence', 0)
            
            # Check if signal direction matches sentiment
            if signal_direction == 'long' and sent_score < -0.2:
                # Going long but sentiment is bearish - this IS concerning
                warnings.append(f"Long signal but bearish sentiment ({sent_score:.2f})")
                confidence_score *= 0.7
                adjustments['sentiment_conflict'] = True
            
            elif signal_direction == 'short' and sent_score > 0.2:
                # Going short but sentiment is bullish
                # FOR VALUE SHORTS: This is actually a CONTRARIAN opportunity!
                # Market is overly optimistic about potentially overvalued stock
                # Don't penalize - just note it as a contrarian trade
                warnings.append(f"Contrarian short: bullish sentiment ({sent_score:.2f}) on short candidate")
                adjustments['contrarian_short'] = True
                # SLIGHT boost for contrarian shorts (value opportunity)
                if sent_score > 0.4:
                    confidence_score = min(1.0, confidence_score * 1.1)
                    adjustments['contrarian_boost'] = True
            
            elif signal_direction == 'long' and sent_score > 0.3 and sent_conf > 0.5:
                # Signal and sentiment agree strongly - boost confidence
                confidence_score = min(1.0, confidence_score * 1.2)
                adjustments['sentiment_confirmation'] = True
            
            elif signal_direction == 'short' and sent_score < -0.3 and sent_conf > 0.5:
                # Short signal with bearish sentiment - strong confirmation!
                confidence_score = min(1.0, confidence_score * 1.25)
                adjustments['sentiment_confirmation'] = True
                adjustments['short_confirmed'] = True
            
            # Check sentiment freshness
            freshness = ticker_sentiment.get('freshness_hours', 999)
            if freshness > 48:
                warnings.append(f"Stale sentiment data ({freshness:.0f}h old)")
                confidence_score *= 0.8
        
        # === CHECK 3: Macro Sentiment Alignment ===
        if macro_sentiment is not None:
            if signal_direction == 'long' and macro_sentiment < -0.3:
                warnings.append(f"Long signal in risk-off macro ({macro_sentiment:.2f})")
                confidence_score *= 0.8
            elif signal_direction == 'short' and macro_sentiment > 0.3:
                warnings.append(f"Short signal in risk-on macro ({macro_sentiment:.2f})")
                confidence_score *= 0.8
        
        # === CHECK 4: Data Freshness ===
        if data_timestamp:
            try:
                ts = data_timestamp
                if ts.tzinfo is None:
                    ts = pytz.UTC.localize(ts)
                ct = current_time
                if ct.tzinfo is None:
                    ct = pytz.UTC.localize(ct)
                
                hours_old = (ct - ts).total_seconds() / 3600
                
                if hours_old > self.max_staleness_hours:
                    blocking_issues.append(f"Data too old ({hours_old:.0f}h > {self.max_staleness_hours}h)")
                elif hours_old > self.max_staleness_hours * 0.5:
                    warnings.append(f"Data getting stale ({hours_old:.0f}h)")
                    confidence_score *= 0.9
            except Exception as e:
                warnings.append(f"Could not check data freshness: {e}")
        
        # === CHECK 5: Position Size Validation ===
        if abs(signal_weight) > 0.15:
            warnings.append(f"Large position size ({signal_weight:.1%})")
        
        if abs(signal_weight) > 0.25:
            blocking_issues.append(f"Position too large ({signal_weight:.1%} > 25%)")
        
        # === CHECK 6: Zero Weight Signals ===
        if abs(signal_weight) < 0.001:
            # Not really a signal, just neutral
            return ValidationResult(
                is_valid=True,
                confidence_score=0,
                warnings=[],
                blocking_issues=[],
                adjustments={'is_neutral': True}
            )
        
        # === FINAL DECISION ===
        is_valid = len(blocking_issues) == 0
        
        # Apply minimum confidence floor
        confidence_score = max(0.1, min(1.0, confidence_score))
        
        # Track stats
        self.total_validated += 1
        if is_valid:
            self.total_passed += 1
        else:
            self.total_blocked += 1
        self.total_warnings += len(warnings)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            warnings=warnings,
            blocking_issues=blocking_issues,
            adjustments=adjustments,
        )
    
    def validate_portfolio(
        self,
        signals: Dict[str, Dict],
        ticker_sentiments: Dict[str, Dict],
        macro_sentiment: Optional[float] = None,
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Validate an entire portfolio of signals.
        
        Returns:
            - Adjusted weights with validation applied
            - List of all warnings/issues
        """
        adjusted_weights = {}
        all_warnings = []
        
        for ticker, signal in signals.items():
            direction = 'long' if signal.get('weight', 0) > 0 else 'short'
            weight = signal.get('weight', 0)
            confidence = signal.get('confidence', 0.5)
            
            result = self.validate_signal(
                ticker=ticker,
                signal_direction=direction,
                signal_weight=weight,
                signal_confidence=confidence,
                ticker_sentiment=ticker_sentiments.get(ticker),
                macro_sentiment=macro_sentiment,
            )
            
            if not result.is_valid:
                all_warnings.append(f"❌ {ticker}: BLOCKED - {', '.join(result.blocking_issues)}")
                adjusted_weights[ticker] = 0  # Don't trade blocked signals
            else:
                # Apply confidence adjustment to weight
                adjustment_factor = result.confidence_score / max(0.1, confidence)
                adjusted_weight = weight * adjustment_factor
                adjusted_weights[ticker] = adjusted_weight
                
                if result.warnings:
                    for w in result.warnings:
                        all_warnings.append(f"⚠️ {ticker}: {w}")
                
                if result.adjustments.get('sentiment_confirmation'):
                    all_warnings.append(f"✅ {ticker}: Sentiment confirms signal")
        
        return adjusted_weights, all_warnings
    
    def get_stats(self) -> Dict:
        """Get validation statistics."""
        return {
            'total_validated': self.total_validated,
            'total_passed': self.total_passed,
            'total_blocked': self.total_blocked,
            'total_warnings': self.total_warnings,
            'pass_rate': self.total_passed / max(1, self.total_validated),
            'block_rate': self.total_blocked / max(1, self.total_validated),
        }


class MarketConfirmationValidator:
    """
    Validates signals AFTER trading by checking market confirmation.
    Used to track which signals actually worked.
    """
    
    def __init__(self, confirmation_threshold: float = 0.002):
        self.confirmation_threshold = confirmation_threshold  # 0.2% move
        self.confirmed_signals = []
        self.unconfirmed_signals = []
    
    def check_confirmation(
        self,
        ticker: str,
        signal_direction: str,
        signal_timestamp: datetime,
        expected_move: str,  # 'bullish' or 'bearish'
        actual_return: float,
        check_window_hours: int = 24,
    ) -> Dict:
        """
        Check if market moved in the expected direction.
        
        Returns confirmation status and details.
        """
        is_confirmed = False
        
        if expected_move == 'bullish' and actual_return > self.confirmation_threshold:
            is_confirmed = True
        elif expected_move == 'bearish' and actual_return < -self.confirmation_threshold:
            is_confirmed = True
        
        result = {
            'ticker': ticker,
            'signal_direction': signal_direction,
            'expected_move': expected_move,
            'actual_return': actual_return,
            'is_confirmed': is_confirmed,
            'timestamp': signal_timestamp.isoformat() if signal_timestamp else None,
            'check_window_hours': check_window_hours,
        }
        
        if is_confirmed:
            self.confirmed_signals.append(result)
        else:
            self.unconfirmed_signals.append(result)
        
        return result
    
    def get_confirmation_rate(self) -> float:
        """Get the overall confirmation rate."""
        total = len(self.confirmed_signals) + len(self.unconfirmed_signals)
        if total == 0:
            return 0.0
        return len(self.confirmed_signals) / total
    
    def get_stats(self) -> Dict:
        """Get confirmation statistics."""
        return {
            'confirmed': len(self.confirmed_signals),
            'unconfirmed': len(self.unconfirmed_signals),
            'confirmation_rate': self.get_confirmation_rate(),
        }
