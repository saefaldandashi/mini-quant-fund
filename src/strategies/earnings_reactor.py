"""
Earnings Reaction Module - Reacts to earnings news in real-time.

This module monitors geopolitical and news feeds for earnings-related headlines
and generates immediate trading signals based on:
- Earnings beats → Long signal
- Earnings misses → Short signal
- Guidance changes → Appropriate signal
- Large price drops → Short confirmation
- Large price spikes → Long confirmation
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)


@dataclass
class EarningsSignal:
    """Signal generated from earnings news."""
    symbol: str
    direction: str  # 'long' or 'short'
    confidence: float  # 0-1
    headline: str
    reason: str
    timestamp: datetime
    price_move_pct: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'headline': self.headline,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'price_move_pct': self.price_move_pct,
        }


# Mapping of company names to tickers
COMPANY_TICKER_MAP = {
    'microsoft': 'MSFT',
    'apple': 'AAPL',
    'amazon': 'AMZN',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'meta': 'META',
    'facebook': 'META',
    'nvidia': 'NVDA',
    'tesla': 'TSLA',
    'netflix': 'NFLX',
    'amd': 'AMD',
    'intel': 'INTC',
    'adobe': 'ADBE',
    'salesforce': 'CRM',
    'oracle': 'ORCL',
    'cisco': 'CSCO',
    'qualcomm': 'QCOM',
    'broadcom': 'AVGO',
    'paypal': 'PYPL',
    'uber': 'UBER',
    'airbnb': 'ABNB',
    'snap': 'SNAP',
    'twitter': 'TWTR',
    'spotify': 'SPOT',
    'zoom': 'ZM',
    'palantir': 'PLTR',
    'coinbase': 'COIN',
    'robinhood': 'HOOD',
    'rivian': 'RIVN',
    'lucid': 'LCID',
    'jpmorgan': 'JPM',
    'jp morgan': 'JPM',
    'goldman sachs': 'GS',
    'goldman': 'GS',
    'morgan stanley': 'MS',
    'bank of america': 'BAC',
    'wells fargo': 'WFC',
    'citigroup': 'C',
    'visa': 'V',
    'mastercard': 'MA',
    'american express': 'AXP',
    'disney': 'DIS',
    'walmart': 'WMT',
    'target': 'TGT',
    'costco': 'COST',
    'home depot': 'HD',
    'lowes': 'LOW',
    'starbucks': 'SBUX',
    'mcdonalds': 'MCD',
    'nike': 'NKE',
    'boeing': 'BA',
    'lockheed': 'LMT',
    'caterpillar': 'CAT',
    'exxon': 'XOM',
    'chevron': 'CVX',
    'conocophillips': 'COP',
    'pfizer': 'PFE',
    'johnson & johnson': 'JNJ',
    'johnson and johnson': 'JNJ',
    'merck': 'MRK',
    'abbvie': 'ABBV',
    'moderna': 'MRNA',
    'eli lilly': 'LLY',
    'united health': 'UNH',
    'unitedhealth': 'UNH',
    'cvs': 'CVS',
    'at&t': 'T',
    'verizon': 'VZ',
    't-mobile': 'TMUS',
    'comcast': 'CMCSA',
    '3m': 'MMM',
    'general electric': 'GE',
    'honeywell': 'HON',
    'raytheon': 'RTX',
    'ford': 'F',
    'general motors': 'GM',
}

# Patterns indicating NEGATIVE earnings/performance
NEGATIVE_PATTERNS = [
    r'miss(ed|es)?',
    r'disappoint(ed|ing|s)?',
    r'below\s+expect',
    r'worse\s+than',
    r'weak(er|ness)?',
    r'declin(e|ed|ing)',
    r'drop(ped|s)?',
    r'fall(s|en)?',
    r'plunge(d|s)?',
    r'tank(ed|s)?',
    r'crash(ed|es)?',
    r'slump(ed|s)?',
    r'tumbl(e|ed|es)',
    r'sink(s|ing)?',
    r'slide(s|d)?',
    r'shed(s)?\s+\$?\d+',
    r'lose(s)?\s+\$?\d+',
    r'wipe(d|s)?\s+(out|off)',
    r'erase(d|s)?',
    r'cut(s)?\s+(guidance|forecast|outlook)',
    r'lower(ed|s)?\s+(guidance|forecast|outlook)',
    r'warn(s|ed|ing)?',
    r'concern(s|ed)?',
    r'trouble(d|s)?',
    r'struggle(d|s)?',
    r'shortfall',
    r'miss(ed)?\s+estimate',
    r'below\s+consensus',
    r'revenue\s+miss',
    r'profit\s+miss',
    r'earnings\s+miss',
    r'guidance\s+cut',
    r'spook(ed|s)?',
    r'spooked\s+investors',
    r'rattl(e|ed|es)',
    r'worr(y|ies|ied)',
]

# Patterns indicating POSITIVE earnings/performance
POSITIVE_PATTERNS = [
    r'beat(s)?',
    r'exceed(ed|s)?',
    r'surpass(ed|es)?',
    r'top(ped|s)?',
    r'above\s+expect',
    r'better\s+than',
    r'strong(er)?',
    r'surge(d|s)?',
    r'soar(ed|s)?',
    r'jump(ed|s)?',
    r'rally',
    r'rise(s)?',
    r'gain(ed|s)?',
    r'climb(ed|s)?',
    r'spike(d|s)?',
    r'raise(d|s)?\s+(guidance|forecast|outlook)',
    r'boost(ed|s)?',
    r'upbeat',
    r'optimis(tic|m)',
    r'record\s+(revenue|profit|earnings)',
    r'all-time\s+high',
    r'blow(n)?\s+past',
    r'crush(ed|es)?',
    r'smash(ed|es)?',
]

# Patterns for price movement extraction
PRICE_MOVE_PATTERNS = [
    r'(\d+(?:\.\d+)?)\s*%\s*(drop|fall|decline|down|lower|slide|plunge)',
    r'(drop|fall|decline|down|lower|slide|plunge)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*%',
    r'shed(?:s)?\s+\$(\d+(?:\.\d+)?)\s*(billion|million|bn|m)',
    r'(\d+(?:\.\d+)?)\s*%\s*(rise|gain|jump|up|higher|surge|rally)',
    r'(rise|gain|jump|up|higher|surge|rally)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*%',
]


class EarningsReactor:
    """
    Reacts to earnings news and generates trading signals.
    
    Monitors:
    - Geopolitical intelligence feed
    - Alpha Vantage news
    - Any news headlines
    
    Generates:
    - Short signals on earnings misses
    - Long signals on earnings beats
    - Confidence based on headline strength and price move
    """
    
    def __init__(self):
        self.signals: List[EarningsSignal] = []
        self.processed_headlines: set = set()  # Avoid duplicates
        self.last_scan: Optional[datetime] = None
    
    def extract_ticker(self, headline: str) -> Optional[str]:
        """Extract ticker from headline using company name mapping."""
        headline_lower = headline.lower()
        
        # Check company name mapping FIRST (more reliable)
        for company, ticker in COMPANY_TICKER_MAP.items():
            if company in headline_lower:
                return ticker
        
        # Check for direct ticker mentions with $ prefix (e.g., $MSFT)
        ticker_match = re.search(r'\$([A-Z]{2,5})\b', headline)
        if ticker_match:
            potential_ticker = ticker_match.group(1)
            return potential_ticker
        
        # Avoid common words that look like tickers
        TICKER_BLACKLIST = {'AI', 'US', 'UK', 'EU', 'CEO', 'CFO', 'COO', 'CTO', 'IPO', 'API', 'GDP', 'CPI'}
        
        # Check for ticker mentions without $ (less reliable - need additional context)
        ticker_match = re.search(r'\b([A-Z]{2,4})\b(?:\s+(?:stock|shares|drops|falls|rises|gains|surges|tumbles|plunges))', headline)
        if ticker_match:
            potential_ticker = ticker_match.group(1)
            if potential_ticker not in TICKER_BLACKLIST:
                return potential_ticker
        
        return None
    
    def extract_price_move(self, headline: str) -> Optional[float]:
        """Extract price movement percentage from headline."""
        headline_lower = headline.lower()
        
        for pattern in PRICE_MOVE_PATTERNS:
            match = re.search(pattern, headline_lower)
            if match:
                groups = match.groups()
                for g in groups:
                    try:
                        pct = float(g)
                        if 0 < pct < 100:  # Reasonable percentage
                            # Determine sign based on context
                            if any(word in pattern for word in ['drop', 'fall', 'decline', 'down', 'lower', 'slide', 'plunge', 'shed']):
                                return -pct
                            else:
                                return pct
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def analyze_sentiment(self, headline: str) -> Tuple[str, float, str]:
        """
        Analyze headline sentiment.
        
        Returns:
            Tuple of (direction, confidence, reason)
        """
        headline_lower = headline.lower()
        
        # Count negative patterns
        negative_score = 0
        negative_reasons = []
        for pattern in NEGATIVE_PATTERNS:
            if re.search(pattern, headline_lower):
                negative_score += 1
                negative_reasons.append(pattern.replace(r'\s+', ' ').replace('\\', ''))
        
        # Count positive patterns
        positive_score = 0
        positive_reasons = []
        for pattern in POSITIVE_PATTERNS:
            if re.search(pattern, headline_lower):
                positive_score += 1
                positive_reasons.append(pattern.replace(r'\s+', ' ').replace('\\', ''))
        
        # Determine direction
        if negative_score > positive_score:
            direction = 'short'
            confidence = min(0.9, 0.3 + (negative_score * 0.15))
            reason = f"Negative signals: {', '.join(negative_reasons[:3])}"
        elif positive_score > negative_score:
            direction = 'long'
            confidence = min(0.9, 0.3 + (positive_score * 0.15))
            reason = f"Positive signals: {', '.join(positive_reasons[:3])}"
        else:
            direction = 'neutral'
            confidence = 0.0
            reason = "Mixed or neutral sentiment"
        
        return direction, confidence, reason
    
    def process_headline(self, headline: str, timestamp: datetime = None) -> Optional[EarningsSignal]:
        """
        Process a single headline and generate signal if applicable.
        
        Args:
            headline: News headline
            timestamp: When the headline was published
        
        Returns:
            EarningsSignal if actionable, None otherwise
        """
        # Skip if already processed
        headline_key = headline[:100].lower()
        if headline_key in self.processed_headlines:
            return None
        
        self.processed_headlines.add(headline_key)
        
        # Extract ticker
        ticker = self.extract_ticker(headline)
        if not ticker:
            return None
        
        # Analyze sentiment
        direction, confidence, reason = self.analyze_sentiment(headline)
        
        if direction == 'neutral' or confidence < 0.3:
            return None
        
        # Extract price move if mentioned
        price_move = self.extract_price_move(headline)
        
        # Boost confidence if large price move mentioned
        if price_move is not None:
            if abs(price_move) >= 10:
                confidence = min(0.95, confidence + 0.2)
            elif abs(price_move) >= 5:
                confidence = min(0.9, confidence + 0.1)
        
        signal = EarningsSignal(
            symbol=ticker,
            direction=direction,
            confidence=confidence,
            headline=headline,
            reason=reason,
            timestamp=timestamp or datetime.now(pytz.UTC),
            price_move_pct=price_move,
        )
        
        self.signals.append(signal)
        logger.info(f"🎯 EARNINGS SIGNAL: {direction.upper()} {ticker} (conf: {confidence:.0%}) - {reason}")
        
        return signal
    
    def scan_geopolitical_intel(self, geo_intel) -> List[EarningsSignal]:
        """
        Scan geopolitical intelligence for earnings signals.
        
        Args:
            geo_intel: GeopoliticalIntelligence instance
        
        Returns:
            List of new signals
        """
        signals = []
        
        try:
            events = geo_intel.get_filtered_events(auto_refresh_if_empty=False)
            
            for event in events:
                headline = event.headline if hasattr(event, 'headline') else str(event)
                timestamp = event.timestamp if hasattr(event, 'timestamp') else None
                
                signal = self.process_headline(headline, timestamp)
                if signal:
                    signals.append(signal)
        except Exception as e:
            logger.warning(f"Error scanning geo intel: {e}")
        
        return signals
    
    def scan_alpha_vantage(self, av_news) -> List[EarningsSignal]:
        """
        Scan Alpha Vantage news for earnings signals.
        
        Args:
            av_news: AlphaVantageNewsLoader instance
        
        Returns:
            List of new signals
        """
        signals = []
        
        try:
            articles = av_news.get_cached_articles()
            
            for article in articles:
                headline = article.headline if hasattr(article, 'headline') else ''
                timestamp = article.timestamp if hasattr(article, 'timestamp') else None
                
                signal = self.process_headline(headline, timestamp)
                if signal:
                    signals.append(signal)
        except Exception as e:
            logger.warning(f"Error scanning AV news: {e}")
        
        return signals
    
    def get_trading_signals(self, max_age_hours: int = 24) -> Dict[str, float]:
        """
        Get trading weights from recent signals.
        
        Args:
            max_age_hours: Only consider signals from last N hours
        
        Returns:
            Dict of symbol -> weight (-1 to +1)
        """
        weights = {}
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=max_age_hours)
        
        for signal in self.signals:
            # Check age
            if signal.timestamp:
                ts = signal.timestamp
                if ts.tzinfo is None:
                    ts = pytz.UTC.localize(ts)
                if ts < cutoff:
                    continue
            
            # Calculate weight
            base_weight = signal.confidence
            if signal.direction == 'short':
                base_weight = -base_weight
            
            # Boost for large price moves
            if signal.price_move_pct is not None:
                if abs(signal.price_move_pct) >= 10:
                    base_weight *= 1.5
                elif abs(signal.price_move_pct) >= 5:
                    base_weight *= 1.2
            
            # Aggregate by symbol (most recent takes precedence)
            symbol = signal.symbol
            if symbol not in weights:
                weights[symbol] = base_weight
            else:
                # Average with existing
                weights[symbol] = (weights[symbol] + base_weight) / 2
        
        # Clamp weights
        for symbol in weights:
            weights[symbol] = max(-1.0, min(1.0, weights[symbol]))
        
        return weights
    
    def get_recent_signals(self, max_age_hours: int = 24) -> List[EarningsSignal]:
        """Get signals from the last N hours."""
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=max_age_hours)
        
        recent = []
        for signal in self.signals:
            if signal.timestamp:
                ts = signal.timestamp
                if ts.tzinfo is None:
                    ts = pytz.UTC.localize(ts)
                if ts >= cutoff:
                    recent.append(signal)
        
        return sorted(recent, key=lambda x: x.timestamp, reverse=True)
    
    def clear_old_signals(self, max_age_hours: int = 48):
        """Remove signals older than max_age_hours."""
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=max_age_hours)
        
        self.signals = [
            s for s in self.signals
            if s.timestamp and (
                s.timestamp if s.timestamp.tzinfo else pytz.UTC.localize(s.timestamp)
            ) >= cutoff
        ]


# Singleton instance
_earnings_reactor: Optional[EarningsReactor] = None

def get_earnings_reactor() -> EarningsReactor:
    """Get singleton earnings reactor instance."""
    global _earnings_reactor
    if _earnings_reactor is None:
        _earnings_reactor = EarningsReactor()
    return _earnings_reactor
