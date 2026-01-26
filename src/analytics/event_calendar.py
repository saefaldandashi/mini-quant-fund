"""
Event Calendar Integration

Tracks market-moving events and adjusts risk/exposure accordingly:
1. Earnings Calendar - Reduce position size before earnings
2. FOMC Meetings - Reduce leverage on Fed days
3. Economic Releases - NFP, CPI, GDP impact
4. Options Expiration - Increased volatility risk
5. Dividend Ex-Dates - Factor into total return
6. Index Rebalances - Anticipate forced flows

The system pre-positions and adjusts exposure around known events.
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import os

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of market events."""
    EARNINGS = "earnings"
    FOMC = "fomc"
    ECONOMIC_RELEASE = "economic"
    OPTIONS_EXPIRY = "options_expiry"
    DIVIDEND = "dividend"
    INDEX_REBALANCE = "index_rebalance"
    IPO = "ipo"
    SPLIT = "split"
    CONFERENCE = "conference"
    # Cross-asset event types
    OIL_INVENTORY = "oil_inventory"  # EIA Weekly Report
    ECB_MEETING = "ecb"              # European Central Bank
    OPEC_MEETING = "opec"            # OPEC+ decisions
    CHINA_PMI = "china_pmi"          # China manufacturing data
    BOJ_MEETING = "boj"              # Bank of Japan
    COMMODITY_REPORT = "commodity"    # Other commodity reports
    OTHER = "other"


class EventImpact(Enum):
    """Expected impact level."""
    CRITICAL = "critical"    # Halt trading / major de-risk
    HIGH = "high"           # Significant position reduction
    MEDIUM = "medium"       # Moderate position reduction
    LOW = "low"             # Minor adjustment
    INFORMATIONAL = "info"  # No action needed


@dataclass
class MarketEvent:
    """A scheduled market event."""
    event_id: str
    event_type: EventType
    
    # Timing
    event_date: date
    event_time: Optional[str] = None  # e.g., "14:00 ET"
    
    # Affected assets
    affected_symbols: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    market_wide: bool = False
    
    # Impact assessment
    impact: EventImpact = EventImpact.MEDIUM
    expected_volatility_increase: float = 1.0  # Multiplier (1.5 = 50% higher vol expected)
    
    # Metadata
    description: str = ""
    source: str = ""
    
    # Pre/post windows
    pre_event_days: int = 1  # Start adjusting N days before
    post_event_days: int = 1  # Return to normal N days after
    
    def is_active(self, check_date: date) -> bool:
        """Check if event window is active."""
        start = self.event_date - timedelta(days=self.pre_event_days)
        end = self.event_date + timedelta(days=self.post_event_days)
        return start <= check_date <= end
    
    def days_until(self, check_date: date) -> int:
        """Days until event (negative if past)."""
        return (self.event_date - check_date).days
    
    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "event_date": self.event_date.isoformat(),
            "event_time": self.event_time,
            "affected_symbols": self.affected_symbols,
            "affected_sectors": self.affected_sectors,
            "market_wide": self.market_wide,
            "impact": self.impact.value,
            "expected_volatility_increase": self.expected_volatility_increase,
            "description": self.description,
        }


@dataclass
class RiskAdjustment:
    """Risk adjustment based on upcoming events."""
    symbol: str
    
    # Adjustments
    position_multiplier: float = 1.0  # 0.5 = reduce by 50%
    leverage_multiplier: float = 1.0  # 0.5 = halve leverage
    
    # Reasons
    triggering_events: List[str] = field(default_factory=list)
    rationale: str = ""
    
    # Timing
    valid_until: Optional[date] = None


class EventCalendar:
    """
    Manages market event calendar and generates risk adjustments.
    
    Data Sources:
    - Manual FOMC calendar (known in advance)
    - Economic calendar from FRED
    - Earnings from Alpha Vantage or manual
    - Options expiry (3rd Friday of month)
    
    Actions:
    - Pre-earnings: Reduce single-stock exposure
    - FOMC day: Reduce overall leverage
    - NFP/CPI: Reduce exposure morning of release
    - Opex: Be aware of gamma effects
    """
    
    # 2026 FOMC Meeting Dates (known schedule)
    FOMC_DATES_2026 = [
        date(2026, 1, 28), date(2026, 1, 29),  # Jan meeting
        date(2026, 3, 17), date(2026, 3, 18),  # Mar meeting
        date(2026, 5, 5), date(2026, 5, 6),    # May meeting
        date(2026, 6, 16), date(2026, 6, 17),  # Jun meeting
        date(2026, 7, 28), date(2026, 7, 29),  # Jul meeting
        date(2026, 9, 15), date(2026, 9, 16),  # Sep meeting
        date(2026, 11, 3), date(2026, 11, 4),  # Nov meeting
        date(2026, 12, 15), date(2026, 12, 16), # Dec meeting
    ]
    
    # Monthly economic release schedule (approximate)
    ECONOMIC_RELEASES = {
        "NFP": {"day": "first_friday", "time": "08:30 ET", "impact": EventImpact.HIGH},
        "CPI": {"day": "mid_month", "time": "08:30 ET", "impact": EventImpact.HIGH},
        "PPI": {"day": "mid_month", "time": "08:30 ET", "impact": EventImpact.MEDIUM},
        "Retail_Sales": {"day": "mid_month", "time": "08:30 ET", "impact": EventImpact.MEDIUM},
        "GDP": {"day": "end_month", "time": "08:30 ET", "impact": EventImpact.MEDIUM},
        "PCE": {"day": "end_month", "time": "08:30 ET", "impact": EventImpact.MEDIUM},
    }
    
    # Impact factors by event type
    IMPACT_FACTORS = {
        EventType.EARNINGS: {
            EventImpact.HIGH: {"position_mult": 0.3, "leverage_mult": 0.5},
            EventImpact.MEDIUM: {"position_mult": 0.5, "leverage_mult": 0.75},
            EventImpact.LOW: {"position_mult": 0.8, "leverage_mult": 0.9},
        },
        EventType.FOMC: {
            EventImpact.CRITICAL: {"position_mult": 0.7, "leverage_mult": 0.5},
            EventImpact.HIGH: {"position_mult": 0.8, "leverage_mult": 0.75},
        },
        EventType.ECONOMIC_RELEASE: {
            EventImpact.HIGH: {"position_mult": 0.85, "leverage_mult": 0.8},
            EventImpact.MEDIUM: {"position_mult": 0.9, "leverage_mult": 0.9},
        },
        EventType.OPTIONS_EXPIRY: {
            EventImpact.MEDIUM: {"position_mult": 0.95, "leverage_mult": 0.9},
        },
        # Cross-asset event impacts
        EventType.OIL_INVENTORY: {
            EventImpact.MEDIUM: {"position_mult": 0.7, "leverage_mult": 0.8},  # Reduce energy
        },
        EventType.ECB_MEETING: {
            EventImpact.HIGH: {"position_mult": 0.8, "leverage_mult": 0.7},  # Reduce EUR-exposed
        },
        EventType.OPEC_MEETING: {
            EventImpact.HIGH: {"position_mult": 0.5, "leverage_mult": 0.6},  # Major oil impact
        },
        EventType.CHINA_PMI: {
            EventImpact.MEDIUM: {"position_mult": 0.8, "leverage_mult": 0.85},  # China-exposed
        },
        EventType.BOJ_MEETING: {
            EventImpact.MEDIUM: {"position_mult": 0.85, "leverage_mult": 0.85},  # Yen-exposed
        },
    }
    
    # Cross-asset affected sectors/symbols
    OIL_AFFECTED_SYMBOLS = ['XOM', 'CVX', 'SLB', 'OXY', 'COP', 'XLE']
    EUR_AFFECTED_SYMBOLS = ['AAPL', 'MSFT', 'GOOG', 'XLI', 'EWG']
    CHINA_AFFECTED_SYMBOLS = ['AAPL', 'TSLA', 'NKE', 'SBUX', 'FXI']
    
    def __init__(
        self,
        storage_path: str = "outputs/event_calendar.json",
        earnings_api_key: Optional[str] = None,
    ):
        self.storage_path = Path(storage_path)
        self.earnings_api_key = earnings_api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        
        # Event storage
        self.events: Dict[str, MarketEvent] = {}
        self.earnings_cache: Dict[str, List[Dict]] = {}
        
        self._load()
        self._generate_known_events()
    
    def _load(self):
        """Load saved events."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.earnings_cache = data.get('earnings_cache', {})
                logger.info("Loaded event calendar data")
            except Exception as e:
                logger.warning(f"Could not load event calendar: {e}")
    
    def _save(self):
        """Save events to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'earnings_cache': self.earnings_cache,
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save event calendar: {e}")
    
    def _generate_known_events(self):
        """Generate events from known schedules."""
        today = date.today()
        
        # Add FOMC meetings
        for fomc_date in self.FOMC_DATES_2026:
            if fomc_date >= today - timedelta(days=7):  # Only future + recent
                event_id = f"fomc_{fomc_date.isoformat()}"
                self.events[event_id] = MarketEvent(
                    event_id=event_id,
                    event_type=EventType.FOMC,
                    event_date=fomc_date,
                    event_time="14:00 ET",
                    market_wide=True,
                    impact=EventImpact.HIGH,
                    expected_volatility_increase=1.5,
                    description="FOMC Meeting - Rate Decision",
                    source="Federal Reserve",
                    pre_event_days=1,
                    post_event_days=1,
                )
        
        # Add EIA Oil Inventory Report (every Wednesday at 10:30 AM ET)
        self._add_weekly_oil_inventory(today)
        
        # Add ECB meetings (known 2026 dates)
        self._add_ecb_meetings(today)
        
        # Add OPEC meetings (known 2026 dates)
        self._add_opec_meetings(today)
        
        # Add China PMI (first day of each month)
        self._add_china_pmi(today)
        
        # Add monthly options expiry (3rd Friday)
        for month in range(1, 13):
            year = today.year
            if month < today.month:
                year += 1
            
            # Find 3rd Friday
            first_day = date(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)
            
            if third_friday >= today:
                event_id = f"opex_{third_friday.isoformat()}"
                self.events[event_id] = MarketEvent(
                    event_id=event_id,
                    event_type=EventType.OPTIONS_EXPIRY,
                    event_date=third_friday,
                    event_time="16:00 ET",
                    market_wide=True,
                    impact=EventImpact.MEDIUM,
                    expected_volatility_increase=1.2,
                    description="Monthly Options Expiration",
                    source="CBOE",
                    pre_event_days=0,
                    post_event_days=0,
                )
        
        # Add approximate economic release dates for next 3 months
        for months_ahead in range(3):
            target_month = (today.month + months_ahead - 1) % 12 + 1
            target_year = today.year + ((today.month + months_ahead - 1) // 12)
            
            # NFP - First Friday
            first_day = date(target_year, target_month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            
            if first_friday >= today:
                event_id = f"nfp_{first_friday.isoformat()}"
                self.events[event_id] = MarketEvent(
                    event_id=event_id,
                    event_type=EventType.ECONOMIC_RELEASE,
                    event_date=first_friday,
                    event_time="08:30 ET",
                    market_wide=True,
                    impact=EventImpact.HIGH,
                    expected_volatility_increase=1.3,
                    description="Non-Farm Payrolls Release",
                    source="BLS",
                    pre_event_days=0,
                    post_event_days=0,
                )
            
            # CPI - Around 12th-14th of month
            cpi_date = date(target_year, target_month, 13)
            if cpi_date >= today:
                event_id = f"cpi_{cpi_date.isoformat()}"
                self.events[event_id] = MarketEvent(
                    event_id=event_id,
                    event_type=EventType.ECONOMIC_RELEASE,
                    event_date=cpi_date,
                    event_time="08:30 ET",
                    market_wide=True,
                    impact=EventImpact.HIGH,
                    expected_volatility_increase=1.4,
                    description="Consumer Price Index (CPI) Release",
                    source="BLS",
                    pre_event_days=0,
                    post_event_days=0,
                )
        
        logger.info(f"Generated {len(self.events)} known calendar events")
    
    def _add_weekly_oil_inventory(self, today: date):
        """Add EIA Oil Inventory reports (every Wednesday at 10:30 AM ET)."""
        # Add next 12 Wednesdays
        current = today
        for _ in range(12):
            # Find next Wednesday
            days_until_wednesday = (2 - current.weekday()) % 7
            if days_until_wednesday == 0 and current == today:
                days_until_wednesday = 7
            next_wednesday = current + timedelta(days=days_until_wednesday)
            
            event_id = f"eia_oil_{next_wednesday.isoformat()}"
            self.events[event_id] = MarketEvent(
                event_id=event_id,
                event_type=EventType.OIL_INVENTORY,
                event_date=next_wednesday,
                event_time="10:30 ET",
                affected_symbols=self.OIL_AFFECTED_SYMBOLS,
                affected_sectors=["XLE"],
                market_wide=False,
                impact=EventImpact.MEDIUM,
                expected_volatility_increase=1.3,
                description="EIA Weekly Oil Inventory Report",
                source="EIA",
                pre_event_days=0,
                post_event_days=0,
            )
            current = next_wednesday + timedelta(days=1)
    
    def _add_ecb_meetings(self, today: date):
        """Add ECB meeting dates for 2026."""
        # Known 2026 ECB meeting dates (approximate)
        ecb_dates = [
            date(2026, 1, 22),
            date(2026, 3, 5),
            date(2026, 4, 16),
            date(2026, 6, 4),
            date(2026, 7, 23),
            date(2026, 9, 10),
            date(2026, 10, 29),
            date(2026, 12, 17),
        ]
        
        for ecb_date in ecb_dates:
            if ecb_date >= today - timedelta(days=3):
                event_id = f"ecb_{ecb_date.isoformat()}"
                self.events[event_id] = MarketEvent(
                    event_id=event_id,
                    event_type=EventType.ECB_MEETING,
                    event_date=ecb_date,
                    event_time="07:45 ET",
                    affected_symbols=self.EUR_AFFECTED_SYMBOLS,
                    affected_sectors=["EWG", "VGK"],
                    market_wide=False,
                    impact=EventImpact.HIGH,
                    expected_volatility_increase=1.4,
                    description="ECB Interest Rate Decision",
                    source="ECB",
                    pre_event_days=1,
                    post_event_days=0,
                )
    
    def _add_opec_meetings(self, today: date):
        """Add OPEC+ meeting dates for 2026."""
        # Known 2026 OPEC+ meeting dates (approximate)
        opec_dates = [
            date(2026, 2, 1),
            date(2026, 4, 3),
            date(2026, 6, 1),
            date(2026, 8, 3),
            date(2026, 10, 4),
            date(2026, 12, 5),
        ]
        
        for opec_date in opec_dates:
            if opec_date >= today - timedelta(days=3):
                event_id = f"opec_{opec_date.isoformat()}"
                self.events[event_id] = MarketEvent(
                    event_id=event_id,
                    event_type=EventType.OPEC_MEETING,
                    event_date=opec_date,
                    event_time="06:00 ET",
                    affected_symbols=self.OIL_AFFECTED_SYMBOLS,
                    affected_sectors=["XLE"],
                    market_wide=False,
                    impact=EventImpact.HIGH,
                    expected_volatility_increase=1.6,
                    description="OPEC+ Meeting - Production Decision",
                    source="OPEC",
                    pre_event_days=1,
                    post_event_days=1,
                )
    
    def _add_china_pmi(self, today: date):
        """Add China PMI releases (1st of each month)."""
        for months_ahead in range(6):
            target_month = (today.month + months_ahead) % 12 or 12
            target_year = today.year + ((today.month + months_ahead - 1) // 12)
            pmi_date = date(target_year, target_month, 1)
            
            if pmi_date >= today:
                event_id = f"china_pmi_{pmi_date.isoformat()}"
                self.events[event_id] = MarketEvent(
                    event_id=event_id,
                    event_type=EventType.CHINA_PMI,
                    event_date=pmi_date,
                    event_time="21:00 ET (prev day)",
                    affected_symbols=self.CHINA_AFFECTED_SYMBOLS,
                    affected_sectors=["FXI", "EEM"],
                    market_wide=False,
                    impact=EventImpact.MEDIUM,
                    expected_volatility_increase=1.2,
                    description="China Manufacturing PMI Release",
                    source="NBS China",
                    pre_event_days=0,
                    post_event_days=1,
                )
    
    def add_earnings_event(
        self,
        symbol: str,
        earnings_date: date,
        is_before_market: bool = True,
    ):
        """Add an earnings event for a symbol."""
        event_id = f"earnings_{symbol}_{earnings_date.isoformat()}"
        
        self.events[event_id] = MarketEvent(
            event_id=event_id,
            event_type=EventType.EARNINGS,
            event_date=earnings_date,
            event_time="BMO" if is_before_market else "AMC",
            affected_symbols=[symbol],
            market_wide=False,
            impact=EventImpact.HIGH,
            expected_volatility_increase=2.0,  # Earnings can double vol
            description=f"{symbol} Earnings Report",
            pre_event_days=2,  # Start reducing 2 days before
            post_event_days=1,  # Return to normal 1 day after
        )
    
    def get_active_events(
        self,
        check_date: Optional[date] = None,
        symbol: Optional[str] = None,
    ) -> List[MarketEvent]:
        """Get all events active on a given date."""
        check_date = check_date or date.today()
        
        active = []
        for event in self.events.values():
            if event.is_active(check_date):
                # Filter by symbol if specified
                if symbol:
                    if event.market_wide or symbol in event.affected_symbols:
                        active.append(event)
                else:
                    active.append(event)
        
        return sorted(active, key=lambda e: e.event_date)
    
    def get_upcoming_events(
        self,
        days_ahead: int = 7,
        event_types: Optional[List[EventType]] = None,
    ) -> List[MarketEvent]:
        """Get events in the next N days."""
        today = date.today()
        end_date = today + timedelta(days=days_ahead)
        
        upcoming = []
        for event in self.events.values():
            if today <= event.event_date <= end_date:
                if event_types is None or event.event_type in event_types:
                    upcoming.append(event)
        
        return sorted(upcoming, key=lambda e: e.event_date)
    
    def get_risk_adjustment(
        self,
        symbol: str,
        check_date: Optional[date] = None,
    ) -> RiskAdjustment:
        """
        Calculate risk adjustment for a symbol based on upcoming events.
        """
        check_date = check_date or date.today()
        
        active_events = self.get_active_events(check_date, symbol)
        
        if not active_events:
            return RiskAdjustment(symbol=symbol)
        
        # Start with no adjustment
        position_mult = 1.0
        leverage_mult = 1.0
        event_ids = []
        rationale_parts = []
        
        for event in active_events:
            # Get impact factors
            type_impacts = self.IMPACT_FACTORS.get(event.event_type, {})
            impact_factors = type_impacts.get(event.impact, {})
            
            if impact_factors:
                # Use minimum of current and new (most restrictive)
                new_pos = impact_factors.get("position_mult", 1.0)
                new_lev = impact_factors.get("leverage_mult", 1.0)
                
                if new_pos < position_mult:
                    position_mult = new_pos
                if new_lev < leverage_mult:
                    leverage_mult = new_lev
                
                event_ids.append(event.event_id)
                
                days_to_event = event.days_until(check_date)
                if days_to_event > 0:
                    rationale_parts.append(f"{event.description} in {days_to_event} days")
                elif days_to_event == 0:
                    rationale_parts.append(f"{event.description} TODAY")
                else:
                    rationale_parts.append(f"{event.description} {-days_to_event} days ago")
        
        # Find when adjustment expires
        latest_end = check_date
        for event in active_events:
            event_end = event.event_date + timedelta(days=event.post_event_days)
            if event_end > latest_end:
                latest_end = event_end
        
        return RiskAdjustment(
            symbol=symbol,
            position_multiplier=position_mult,
            leverage_multiplier=leverage_mult,
            triggering_events=event_ids,
            rationale="; ".join(rationale_parts),
            valid_until=latest_end,
        )
    
    def get_portfolio_adjustment(
        self,
        symbols: List[str],
        check_date: Optional[date] = None,
    ) -> Dict[str, RiskAdjustment]:
        """Get risk adjustments for entire portfolio."""
        return {symbol: self.get_risk_adjustment(symbol, check_date) for symbol in symbols}
    
    def get_market_risk_factor(
        self,
        check_date: Optional[date] = None,
    ) -> Tuple[float, str]:
        """
        Get overall market risk factor based on macro events.
        Returns (multiplier, reason) where multiplier < 1 = reduce exposure.
        """
        check_date = check_date or date.today()
        
        market_events = [
            e for e in self.get_active_events(check_date)
            if e.market_wide
        ]
        
        if not market_events:
            return (1.0, "No significant market events")
        
        # Find most impactful event
        min_mult = 1.0
        most_impactful = None
        
        for event in market_events:
            type_impacts = self.IMPACT_FACTORS.get(event.event_type, {})
            impact_factors = type_impacts.get(event.impact, {})
            
            leverage_mult = impact_factors.get("leverage_mult", 1.0)
            if leverage_mult < min_mult:
                min_mult = leverage_mult
                most_impactful = event
        
        if most_impactful:
            days = most_impactful.days_until(check_date)
            if days == 0:
                reason = f"{most_impactful.description} TODAY - reduce exposure"
            elif days > 0:
                reason = f"{most_impactful.description} in {days} days"
            else:
                reason = f"{most_impactful.description} was {-days} days ago"
            
            return (min_mult, reason)
        
        return (1.0, "No adjustment needed")
    
    def get_summary(self) -> Dict:
        """Get calendar summary."""
        today = date.today()
        upcoming_7d = self.get_upcoming_events(7)
        
        return {
            "total_events": len(self.events),
            "active_today": len(self.get_active_events(today)),
            "upcoming_7_days": len(upcoming_7d),
            "next_fomc": next(
                (e for e in self.events.values() 
                 if e.event_type == EventType.FOMC and e.event_date >= today),
                None
            ),
            "next_opex": next(
                (e for e in self.events.values() 
                 if e.event_type == EventType.OPTIONS_EXPIRY and e.event_date >= today),
                None
            ),
            "upcoming_events": [
                {
                    "date": e.event_date.isoformat(),
                    "type": e.event_type.value,
                    "description": e.description,
                    "impact": e.impact.value,
                }
                for e in upcoming_7d[:10]
            ],
        }


# Singleton instance
_event_calendar: Optional[EventCalendar] = None


def get_event_calendar() -> EventCalendar:
    """Get singleton instance of event calendar."""
    global _event_calendar
    if _event_calendar is None:
        _event_calendar = EventCalendar()
    return _event_calendar
