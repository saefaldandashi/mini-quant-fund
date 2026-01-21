"""
Reporting Module - Generates professional reports from live data.

This module provides:
- Live data collection (no parquet dependency)
- Chart generation (equity curve, drawdown, holdings, strategies)
- HTML report rendering
- Learning feedback integration

Usage:
    from src.reporting import generate_report
    
    result = generate_report(
        broker=broker,
        learning_engine=learning_engine,
        app_state={'last_run_status': ...},
        report_type='daily',
        output_path='outputs/reports/daily_2026-01-21.html',
    )
    
    if result['success']:
        print(f"Report saved to: {result['path']}")
"""

# Core exports
from .report_renderer import generate_report, ReportRenderer
from .live_collectors import (
    LiveDataCollector,
    ReportData,
    PortfolioMetrics,
    Position,
    Trade,
    StrategyPerformance,
    MacroSnapshot,
)
from .report_charts import ReportChartGenerator
from .learning_feedback import ReportLearningFeedback, TrendInsight

# Keep ValidationError for backward compatibility
class ValidationError(Exception):
    """Report validation error."""
    pass

__all__ = [
    # Main function
    'generate_report',
    
    # Collectors
    'LiveDataCollector',
    'ReportData',
    'PortfolioMetrics',
    'Position',
    'Trade',
    'StrategyPerformance',
    'MacroSnapshot',
    
    # Rendering
    'ReportRenderer',
    'ReportChartGenerator',
    
    # Learning
    'ReportLearningFeedback',
    'TrendInsight',
    
    # Errors
    'ValidationError',
]
