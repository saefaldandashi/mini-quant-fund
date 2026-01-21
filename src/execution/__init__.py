"""
Smart Execution Engine - Optimizes order execution for better prices.

Components:
- SpreadAnalyzer: Analyzes bid-ask spreads to determine execution strategy
- SmartExecutor: Orchestrates intelligent order execution
- ExecutionReport: Tracks and reports execution quality
- TransactionCostModel: Estimates and tracks transaction costs
"""

from .spread_analyzer import SpreadAnalyzer, SpreadCategory
from .smart_executor import SmartExecutor, ExecutionStrategy
from .execution_report import ExecutionReport, ExecutionMetrics
from .transaction_costs import TransactionCostModel, CostEstimate, PortfolioCostSummary

__all__ = [
    'SpreadAnalyzer',
    'SpreadCategory',
    'SmartExecutor', 
    'ExecutionStrategy',
    'ExecutionReport',
    'ExecutionMetrics',
    'TransactionCostModel',
    'CostEstimate',
    'PortfolioCostSummary',
]
