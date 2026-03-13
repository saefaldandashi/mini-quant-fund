"""
Calendar Effects / Seasonality Strategy.
Exploits documented market anomalies based on time of day/week/month/year.
"""
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from .base import Strategy, SignalOutput
from src.data.feature_store import Features

logger = logging.getLogger(__name__)


class CalendarEffectsStrategy(Strategy):
    """
    Calendar Effects Strategy.
    
    Exploits documented market anomalies:
    1. Day-of-Week Effect (Monday weakness, Friday strength)
    2. Turn-of-Month Effect (last & first days of month)
    3. Pre-Holiday Effect (bullish before holidays)
    4. January Effect (small caps outperform in January)
    5. Sell in May / Halloween Effect
    6. End-of-Quarter Window Dressing
    7. Triple Witching (quarterly options expiration)
    8. Tax-Loss Selling (December)
    """
    
    # US Market Holidays (approximate - 2024-2026)
    HOLIDAYS = [
        (1, 1),   # New Year's Day
        (1, 15),  # MLK Day (approx)
        (2, 19),  # Presidents Day (approx)
        (5, 27),  # Memorial Day (approx)
        (6, 19),  # Juneteenth
        (7, 4),   # Independence Day
        (9, 2),   # Labor Day (approx)
        (11, 28), # Thanksgiving (approx)
        (12, 25), # Christmas
    ]
    
    # Sector ETFs for tactical adjustments
    SMALL_CAP_ETFS = ['IWM', 'IJR', 'VB']
    LARGE_CAP_ETFS = ['SPY', 'QQQ', 'DIA']
    DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV']  # Utilities, Staples, Healthcare
    CYCLICAL_SECTORS = ['XLY', 'XLK', 'XLI']   # Discretionary, Tech, Industrials
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CalendarEffects", config)
        self._required_features = ['prices', 'volatility_21d', 'regime']
        
        config = config or {}
        self.base_weight = config.get('base_weight', 0.5)  # Default exposure
        self.effect_strength = config.get('effect_strength', 0.15)  # Max adjustment
        
        # Effect toggles
        self.enable_dow = config.get('enable_dow', True)      # Day of week
        self.enable_tom = config.get('enable_tom', True)      # Turn of month
        self.enable_holiday = config.get('enable_holiday', True)
        self.enable_january = config.get('enable_january', True)
        self.enable_sell_may = config.get('enable_sell_may', True)
        self.enable_window_dressing = config.get('enable_window_dressing', True)
    
    def generate_signals(self, features: Features, t: datetime) -> SignalOutput:
        """Generate signals based on calendar effects."""
        
        # Calculate effect scores
        effects = self._calculate_effects(t)
        
        # Aggregate into net bias
        net_bias = sum(effects.values())  # -1 to 1 scale
        
        # Construct weights
        weights = {}
        expected_returns = {}
        
        # Get available symbols
        available_symbols = [s for s in features.symbols if s in features.prices]
        
        if not available_symbols:
            return self._empty_signal(t)
        
        # Determine what to buy/sell based on effects
        if effects.get('january_effect', 0) > 0:
            # January: Prefer small caps
            for symbol in available_symbols:
                if symbol in self.SMALL_CAP_ETFS or symbol in ['IWM', 'ARKK']:
                    weights[symbol] = self.base_weight * 0.3
                    expected_returns[symbol] = 0.03 / 21  # ~1 month effect
        
        if effects.get('sell_in_may', 0) < 0:
            # May-October: Reduce exposure, prefer defensive
            for symbol in available_symbols:
                if symbol in self.DEFENSIVE_SECTORS:
                    weights[symbol] = self.base_weight * 0.2
                    expected_returns[symbol] = 0.01 / 126  # ~6 month effect
        elif effects.get('sell_in_may', 0) > 0:
            # November-April: More aggressive
            for symbol in available_symbols:
                if symbol in self.CYCLICAL_SECTORS:
                    weights[symbol] = self.base_weight * 0.25
                    expected_returns[symbol] = 0.02 / 126  # ~6 month effect
        
        # Turn of month: Go long broad market
        if effects.get('turn_of_month', 0) > 0:
            for symbol in available_symbols:
                if symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']:
                    weights[symbol] = weights.get(symbol, 0) + self.base_weight * 0.15
                    expected_returns[symbol] = expected_returns.get(symbol, 0) + 0.005 / 5
        
        # Pre-holiday effect: Small long bias
        if effects.get('pre_holiday', 0) > 0:
            for symbol in available_symbols[:5]:  # Top 5 most liquid
                weights[symbol] = weights.get(symbol, 0) + self.effect_strength
                expected_returns[symbol] = expected_returns.get(symbol, 0) + 0.003 / 5
        
        # Day of week adjustment
        dow_effect = effects.get('day_of_week', 0)
        if dow_effect != 0:
            # Scale existing positions
            for symbol in weights:
                weights[symbol] *= (1 + dow_effect * 0.2)
        
        # Window dressing (end of quarter)
        if effects.get('window_dressing', 0) > 0:
            # Institutions buy winners - already captured by momentum strategies
            pass
        
        if not weights:
            # Default: Equal weight top 5 symbols with base weight
            for symbol in available_symbols[:5]:
                weights[symbol] = self.base_weight * 0.1
                expected_returns[symbol] = 0.005 / 21
        
        # Normalize
        total = sum(abs(w) for w in weights.values())
        if total > 1.0:
            weights = {k: v / total for k, v in weights.items()}
        
        exp_ret = self._calculate_expected_return(weights, expected_returns)
        risk = self._calculate_risk(weights, features.covariance_matrix)
        
        # Confidence based on strength of effects
        max_effect = max(abs(v) for v in effects.values()) if effects else 0
        confidence = 0.3 + max_effect * 0.3  # 0.3 to 0.6
        
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=exp_ret,
            expected_returns_by_asset=expected_returns,
            risk_estimate=risk,
            confidence=confidence,
            explanation={
                'effects': {k: round(v, 2) for k, v in effects.items()},
                'net_bias': round(net_bias, 2),
                'date_info': {
                    'day_of_week': t.strftime('%A'),
                    'day_of_month': t.day,
                    'month': t.strftime('%B'),
                },
            },
            regime_fit=0.5,  # Works across regimes but not strong
            diversification_score=0.7,
        )
    
    def _calculate_effects(self, t: datetime) -> Dict[str, float]:
        """Calculate all calendar effects for given date."""
        effects = {}
        
        # 1. Day of Week Effect
        if self.enable_dow:
            dow = t.weekday()  # 0=Monday, 4=Friday
            if dow == 0:
                effects['day_of_week'] = -0.3  # Monday weakness
            elif dow == 4:
                effects['day_of_week'] = 0.2   # Friday strength
            else:
                effects['day_of_week'] = 0.0
        
        # 2. Turn of Month Effect
        if self.enable_tom:
            day = t.day
            # Last 3 days and first 3 days of month
            if day <= 3:
                effects['turn_of_month'] = 0.4  # Strong positive
            elif day >= 28:  # Close to end of month
                effects['turn_of_month'] = 0.3
            else:
                effects['turn_of_month'] = 0.0
        
        # 3. Pre-Holiday Effect
        if self.enable_holiday:
            effects['pre_holiday'] = 0.0
            tomorrow = t + timedelta(days=1)
            for month, day in self.HOLIDAYS:
                if tomorrow.month == month and tomorrow.day == day:
                    effects['pre_holiday'] = 0.5
                    break
        
        # 4. January Effect
        if self.enable_january:
            if t.month == 1:
                effects['january_effect'] = 0.4
            elif t.month == 12 and t.day >= 15:
                # Pre-January buildup
                effects['january_effect'] = 0.2
            else:
                effects['january_effect'] = 0.0
        
        # 5. Sell in May / Halloween Effect
        if self.enable_sell_may:
            # November-April: Good months (+1)
            # May-October: Bad months (-1)
            if t.month in [11, 12, 1, 2, 3, 4]:
                effects['sell_in_may'] = 0.3
            else:
                effects['sell_in_may'] = -0.3
        
        # 6. Window Dressing (last week of quarter)
        if self.enable_window_dressing:
            is_quarter_end = t.month in [3, 6, 9, 12]
            if is_quarter_end and t.day >= 25:
                effects['window_dressing'] = 0.3
            else:
                effects['window_dressing'] = 0.0
        
        # 7. Tax-Loss Selling (December)
        if t.month == 12 and t.day >= 15:
            effects['tax_loss_selling'] = -0.2  # Pressure on losers
        else:
            effects['tax_loss_selling'] = 0.0
        
        return effects
    
    def _empty_signal(self, t: datetime) -> SignalOutput:
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={'note': 'No calendar effect signals'},
        )
    
    def _calculate_expected_return(
        self,
        weights: Dict[str, float],
        expected_returns: Dict[str, float]
    ) -> float:
        return sum(
            weights.get(s, 0) * expected_returns.get(s, 0)
            for s in weights
        )
    
    def _calculate_risk(
        self,
        weights: Dict[str, float],
        cov_matrix
    ) -> float:
        if cov_matrix is None:
            return 0.12
        
        try:
            syms = list(weights.keys())
            valid_syms = [s for s in syms if s in cov_matrix.columns]
            if not valid_syms:
                return 0.12
            
            w = np.array([weights.get(s, 0) for s in valid_syms])
            cov = cov_matrix.loc[valid_syms, valid_syms].values
            var = w.T @ cov @ w
            
            return float(np.sqrt(var))
        except Exception:
            return 0.12
