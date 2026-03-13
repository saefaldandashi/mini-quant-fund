"""Trading strategies module."""
from .base import Strategy, SignalOutput
from .momentum import TimeSeriesMomentumStrategy, CrossSectionMomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .volatility import VolatilityRegimeVolTargetStrategy
from .carry import CarryStrategy
from .value_quality import ValueQualityTiltStrategy
from .risk_parity import RiskParityMinVarStrategy
from .tail_risk import TailRiskOverlayStrategy
from .sentiment_event import NewsSentimentEventStrategy
from .ml_ensemble import MLMetaEnsembleStrategy

# Long/Short strategies
from .long_short import (
    CrossSectionalMomentumLS,
    TimeSeriesMomentumLS,
    MeanReversionLS,
    QualityValueLS,
    create_long_short_strategies,
)

# Futures strategies (Backtest only - uses ETF proxies)
from .futures import (
    FuturesCarryStrategy,
    FuturesMacroOverlay,
    FuturesTrendFollowing,
    create_futures_strategies,
)

# Intraday/Short-Term strategies (15-30 minute trading)
from .intraday import (
    IntradayMomentumStrategy,
    VWAPReversionStrategy,
    VolumeSpikeStrategy,
    RelativeStrengthIntradayStrategy,
    OpeningRangeBreakoutStrategy,
    QuickMeanReversionStrategy,
    IntradaySignalOutput,
    create_intraday_strategies,
)

# NEW: Alpha Enhancement Strategies
from .pairs import PairsTradingStrategy
from .sector_rotation import SectorRotationStrategy
from .calendar_effects import CalendarEffectsStrategy
from .order_flow import OrderFlowStrategy

__all__ = [
    # Base
    'Strategy',
    'SignalOutput',
    
    # Long-only strategies (POSITION TRADING - days/weeks)
    'TimeSeriesMomentumStrategy',
    'CrossSectionMomentumStrategy',
    'MeanReversionStrategy',
    'VolatilityRegimeVolTargetStrategy',
    'CarryStrategy',
    'ValueQualityTiltStrategy',
    'RiskParityMinVarStrategy',
    'TailRiskOverlayStrategy',
    'NewsSentimentEventStrategy',
    'MLMetaEnsembleStrategy',
    
    # Long/Short strategies
    'CrossSectionalMomentumLS',
    'TimeSeriesMomentumLS',
    'MeanReversionLS',
    'QualityValueLS',
    'create_long_short_strategies',
    
    # Futures strategies (Backtest only)
    'FuturesCarryStrategy',
    'FuturesMacroOverlay',
    'FuturesTrendFollowing',
    'create_futures_strategies',
    
    # INTRADAY/SHORT-TERM strategies (15-30 minute trading - HFT-lite)
    'IntradayMomentumStrategy',
    'VWAPReversionStrategy',
    'VolumeSpikeStrategy',
    'RelativeStrengthIntradayStrategy',
    'OpeningRangeBreakoutStrategy',
    'QuickMeanReversionStrategy',
    'IntradaySignalOutput',
    'create_intraday_strategies',
    
    # Alpha Enhancement strategies
    'PairsTradingStrategy',
    'SectorRotationStrategy',
    'CalendarEffectsStrategy',
    'OrderFlowStrategy',
    
    # Helper functions
    'get_all_strategies',
    'create_alpha_strategies',
]


def create_alpha_strategies(config: dict = None) -> list:
    """
    Create all alpha enhancement strategies.
    
    Returns:
        List of alpha strategy instances
    """
    return [
        PairsTradingStrategy(config),
        SectorRotationStrategy(config),
        CalendarEffectsStrategy(config),
        OrderFlowStrategy(config),
    ]


def get_all_strategies(
    enable_intraday: bool = True,
    enable_long_short: bool = True,
    enable_futures: bool = True,
    enable_alpha: bool = True,
):
    """
    Get all available trading strategies, excluding disabled ones.
    
    Args:
        enable_intraday: Include intraday/short-term strategies
        enable_long_short: Include long/short strategies
        enable_futures: Include futures strategies
        enable_alpha: Include alpha enhancement strategies (pairs, sector, calendar, orderflow)
        
    Returns:
        List of strategy instances (disabled strategies already filtered out)
    """
    from config import DISABLED_STRATEGIES
    
    strategies = []
    
    # Intraday strategies (primary for HFT-lite)
    if enable_intraday:
        strategies.extend(create_intraday_strategies())
    
    # Long/Short strategies
    if enable_long_short:
        strategies.extend(create_long_short_strategies())
    
    # Futures strategies (backtest only)
    if enable_futures:
        strategies.extend(create_futures_strategies())
    
    # Alpha enhancement strategies
    if enable_alpha:
        strategies.extend(create_alpha_strategies())
    
    return [s for s in strategies if s.name not in DISABLED_STRATEGIES]
