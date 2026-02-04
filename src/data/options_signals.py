"""
Options Market Signals

Monitors options market data for predictive signals:
- Put/Call Ratio (sentiment indicator)
- Unusual Options Activity
- Implied Volatility Skew
- Options Volume Anomalies

Uses free data from Yahoo Finance options chain.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pytz

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not available - options signals disabled")


@dataclass
class OptionsSignal:
    """Options market signal for a symbol."""
    symbol: str
    timestamp: datetime
    put_call_ratio: float           # < 0.7 = bullish, > 1.0 = bearish
    iv_percentile: float            # IV vs historical (0-100)
    call_volume_ratio: float        # Today's call volume vs avg
    put_volume_ratio: float         # Today's put volume vs avg
    unusual_activity: bool          # True if volume > 3x average
    iv_skew: float                  # Call IV vs Put IV (negative = fear)
    signal_direction: str           # 'bullish', 'bearish', 'neutral'
    signal_strength: float          # 0-1
    rationale: str


class OptionsSignalLoader:
    """
    Fetches and analyzes options market data for trading signals.
    
    Key signals:
    1. Put/Call Ratio:
       - < 0.7: Extremely bullish (contrarian bearish)
       - 0.7-1.0: Normal/neutral
       - > 1.0: Bearish (contrarian bullish after extremes)
       
    2. Implied Volatility:
       - High IV = expensive options, expect volatility
       - Low IV = cheap options, potential breakout
       
    3. Unusual Activity:
       - Large volume = institutional positioning
       - Smart money often uses options before stock moves
    """
    
    def __init__(self, cache_minutes: int = 30):
        self.cache_minutes = cache_minutes
        self._cache: Dict[str, OptionsSignal] = {}
        self._cache_time: Optional[datetime] = None
        
        # Historical data for comparison
        self._historical_pcr: Dict[str, List[float]] = {}
        self._historical_iv: Dict[str, List[float]] = {}
    
    def get_options_signal(self, symbol: str) -> Optional[OptionsSignal]:
        """Get options signal for a single symbol."""
        if not HAS_YFINANCE:
            return None
        
        # Check cache
        if self._is_cache_valid() and symbol in self._cache:
            return self._cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get options chain for nearest expiry
            expirations = ticker.options
            if not expirations:
                return None
            
            # Use nearest monthly expiry (skip weekly)
            exp = expirations[0]  # Nearest expiry
            for e in expirations[:5]:
                exp_dt = datetime.strptime(e, '%Y-%m-%d')
                if (exp_dt - datetime.now()).days > 7:
                    exp = e
                    break
            
            chain = ticker.option_chain(exp)
            calls = chain.calls
            puts = chain.puts
            
            if calls.empty or puts.empty:
                return None
            
            # Calculate Put/Call Ratio
            total_call_vol = calls['volume'].sum()
            total_put_vol = puts['volume'].sum()
            put_call_ratio = total_put_vol / max(1, total_call_vol)
            
            # Calculate IV averages
            call_iv = calls['impliedVolatility'].mean() if 'impliedVolatility' in calls else 0.3
            put_iv = puts['impliedVolatility'].mean() if 'impliedVolatility' in puts else 0.3
            avg_iv = (call_iv + put_iv) / 2
            
            # IV Skew (put IV vs call IV)
            iv_skew = put_iv - call_iv  # Positive = more fear, negative = complacency
            
            # Estimate IV percentile (simplified - compare to 30% baseline)
            iv_percentile = min(100, (avg_iv / 0.30) * 50)
            
            # Volume ratios (simplified - use 1.0 as baseline)
            call_volume_ratio = min(5.0, total_call_vol / max(1, len(calls) * 100))
            put_volume_ratio = min(5.0, total_put_vol / max(1, len(puts) * 100))
            
            # Unusual activity detection
            unusual_activity = (call_volume_ratio > 3.0 or put_volume_ratio > 3.0)
            
            # Determine signal
            direction, strength, rationale = self._interpret_signal(
                put_call_ratio, iv_percentile, iv_skew, unusual_activity,
                call_volume_ratio, put_volume_ratio
            )
            
            signal = OptionsSignal(
                symbol=symbol,
                timestamp=datetime.now(pytz.UTC),
                put_call_ratio=put_call_ratio,
                iv_percentile=iv_percentile,
                call_volume_ratio=call_volume_ratio,
                put_volume_ratio=put_volume_ratio,
                unusual_activity=unusual_activity,
                iv_skew=iv_skew,
                signal_direction=direction,
                signal_strength=strength,
                rationale=rationale,
            )
            
            self._cache[symbol] = signal
            self._cache_time = datetime.now(pytz.UTC)
            
            return signal
            
        except Exception as e:
            logger.debug(f"Options signal error for {symbol}: {e}")
            return None
    
    def get_batch_signals(self, symbols: List[str]) -> Dict[str, OptionsSignal]:
        """Get options signals for multiple symbols."""
        signals = {}
        for symbol in symbols:
            signal = self.get_options_signal(symbol)
            if signal:
                signals[symbol] = signal
        return signals
    
    def _interpret_signal(
        self,
        pcr: float,
        iv_pct: float,
        iv_skew: float,
        unusual: bool,
        call_ratio: float,
        put_ratio: float
    ) -> Tuple[str, float, str]:
        """Interpret options data into trading signal."""
        
        reasons = []
        bullish_score = 0
        bearish_score = 0
        
        # Put/Call Ratio Analysis
        if pcr < 0.5:
            bullish_score += 0.5  # Very bullish options flow
            reasons.append(f"Low P/C ratio ({pcr:.2f})")
        elif pcr < 0.7:
            bullish_score += 0.3
            reasons.append(f"Bullish P/C ratio ({pcr:.2f})")
        elif pcr > 1.5:
            bearish_score += 0.5  # High put activity
            reasons.append(f"High P/C ratio ({pcr:.2f})")
        elif pcr > 1.2:
            bearish_score += 0.3
            reasons.append(f"Elevated P/C ratio ({pcr:.2f})")
        
        # IV Skew Analysis
        if iv_skew > 0.1:
            bearish_score += 0.2  # Put premiums higher = fear
            reasons.append(f"Put skew (fear)")
        elif iv_skew < -0.05:
            bullish_score += 0.2  # Call premiums higher = greed
            reasons.append(f"Call skew (greed)")
        
        # Unusual Activity
        if unusual:
            if call_ratio > put_ratio:
                bullish_score += 0.3
                reasons.append(f"Unusual call activity ({call_ratio:.1f}x)")
            else:
                bearish_score += 0.3
                reasons.append(f"Unusual put activity ({put_ratio:.1f}x)")
        
        # IV Level (contrarian)
        if iv_pct > 80:
            reasons.append(f"High IV ({iv_pct:.0f}%ile)")
        elif iv_pct < 20:
            reasons.append(f"Low IV ({iv_pct:.0f}%ile)")
        
        # Determine direction
        net_score = bullish_score - bearish_score
        
        if net_score > 0.3:
            direction = 'bullish'
            strength = min(1.0, net_score)
        elif net_score < -0.3:
            direction = 'bearish'
            strength = min(1.0, abs(net_score))
        else:
            direction = 'neutral'
            strength = 0.2
        
        rationale = "; ".join(reasons) if reasons else "No clear signal"
        
        return direction, strength, rationale
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_time:
            return False
        age = (datetime.now(pytz.UTC) - self._cache_time).total_seconds() / 60
        return age < self.cache_minutes
    
    def get_market_sentiment(self) -> Dict:
        """Get overall market options sentiment from SPY and QQQ."""
        signals = self.get_batch_signals(['SPY', 'QQQ', 'IWM'])
        
        if not signals:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        bullish_count = sum(1 for s in signals.values() if s.signal_direction == 'bullish')
        bearish_count = sum(1 for s in signals.values() if s.signal_direction == 'bearish')
        avg_strength = sum(s.signal_strength for s in signals.values()) / len(signals)
        
        if bullish_count > bearish_count:
            sentiment = 'bullish'
        elif bearish_count > bullish_count:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': avg_strength,
            'signals': {
                sym: {
                    'direction': s.signal_direction,
                    'strength': s.signal_strength,
                    'pcr': s.put_call_ratio,
                    'unusual': s.unusual_activity,
                }
                for sym, s in signals.items()
            }
        }
    
    def get_summary(self) -> Dict:
        """Get summary of cached signals."""
        return {
            'cached_symbols': len(self._cache),
            'cache_age_min': (
                (datetime.now(pytz.UTC) - self._cache_time).total_seconds() / 60
                if self._cache_time else None
            ),
            'bullish_signals': sum(
                1 for s in self._cache.values() if s.signal_direction == 'bullish'
            ),
            'bearish_signals': sum(
                1 for s in self._cache.values() if s.signal_direction == 'bearish'
            ),
            'unusual_activity': [
                s.symbol for s in self._cache.values() if s.unusual_activity
            ],
        }


# Singleton instance
_options_loader: Optional[OptionsSignalLoader] = None

def get_options_loader() -> OptionsSignalLoader:
    """Get singleton options signal loader."""
    global _options_loader
    if _options_loader is None:
        _options_loader = OptionsSignalLoader()
    return _options_loader
