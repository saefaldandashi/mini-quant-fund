"""
Analytics Module

Advanced analytics for cross-asset correlation, multi-timeframe analysis,
event calendar, factor exposure, and predictive signal generation.
"""

from .cross_correlation import (
    CrossCorrelationEngine,
    CrossAssetSignal,
    CorrelationPair,
    AssetClass,
    CorrelationRegime,
    get_cross_correlation_engine,
)

from .multi_timeframe import (
    MultiTimeframeFusion,
    FusedSignal,
    TimeframeAnalysis,
    TimeframeTrend,
    EntrySignal,
    get_multi_timeframe_fusion,
)

from .event_calendar import (
    EventCalendar,
    MarketEvent,
    RiskAdjustment,
    EventType,
    EventImpact,
    get_event_calendar,
)

from .factor_exposure import (
    FactorExposureManager,
    PortfolioExposure,
    SymbolExposure,
    ExposureLimits,
    Sector,
    Factor,
    get_factor_manager,
)

from .dynamic_weights import (
    DynamicWeightOptimizer,
    WeightOptimizationResult,
    StrategyPerformanceSnapshot,
    get_dynamic_optimizer,
)

__all__ = [
    # Cross-Correlation
    "CrossCorrelationEngine",
    "CrossAssetSignal",
    "CorrelationPair",
    "AssetClass",
    "CorrelationRegime",
    "get_cross_correlation_engine",
    # Multi-Timeframe
    "MultiTimeframeFusion",
    "FusedSignal",
    "TimeframeAnalysis",
    "TimeframeTrend",
    "EntrySignal",
    "get_multi_timeframe_fusion",
    # Event Calendar
    "EventCalendar",
    "MarketEvent",
    "RiskAdjustment",
    "EventType",
    "EventImpact",
    "get_event_calendar",
    # Factor Exposure
    "FactorExposureManager",
    "PortfolioExposure",
    "SymbolExposure",
    "ExposureLimits",
    "Sector",
    "Factor",
    "get_factor_manager",
    # Dynamic Weights
    "DynamicWeightOptimizer",
    "WeightOptimizationResult",
    "StrategyPerformanceSnapshot",
    "get_dynamic_optimizer",
]
