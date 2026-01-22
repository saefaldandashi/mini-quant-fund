"""
Live Data Collectors - Fetch report data directly from APIs.

Uses VERIFIED methods from AlpacaBroker:
- get_account() -> portfolio value, cash, equity
- get_positions() -> current positions with P/L
- get_portfolio_history(days) -> equity curve for returns/vol/Sharpe
- get_historical_bars(['SPY'], days) -> benchmark prices
- get_orders(status='all') -> trade history
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    portfolio_value: float
    cash: float
    equity: float
    buying_power: float
    
    # Returns (computed from portfolio history)
    return_1d: float = 0.0
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_mtd: float = 0.0
    return_ytd: float = 0.0
    
    # Benchmark comparison
    benchmark_return_1d: float = 0.0
    benchmark_return_1m: float = 0.0
    alpha_1d: float = 0.0
    alpha_1m: float = 0.0
    
    # Risk metrics (computed from portfolio history)
    volatility_20d: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown_30d: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Data quality indicators
    has_portfolio_history: bool = False
    has_benchmark_data: bool = False
    data_points: int = 0
    has_sufficient_data_for_vol: bool = False  # Need 20+ points
    has_sufficient_data_for_sharpe: bool = False  # Need 20+ points
    
    # Long/Short exposure metrics
    gross_exposure: float = 0.0  # |long| + |short|
    net_exposure: float = 0.0  # long - short
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    leverage: float = 1.0  # gross / equity
    net_leverage: float = 1.0  # net / equity
    long_count: int = 0
    short_count: int = 0


@dataclass
class Position:
    """Single position with metrics."""
    symbol: str
    quantity: int
    current_price: float
    avg_cost: float
    market_value: float
    weight: float  # Portfolio weight
    unrealized_pnl: float
    unrealized_pnl_pct: float
    change_today: float = 0.0
    change_today_pct: float = 0.0
    side: str = "long"  # "long" or "short"


@dataclass
class Trade:
    """Trade/order record."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    filled_price: float
    value: float
    status: str


@dataclass
class StrategyPerformance:
    """Strategy performance data."""
    name: str
    weight: float
    win_rate: float
    contribution: float
    confidence: float
    debate_score: float


@dataclass
class MacroSnapshot:
    """Macro environment snapshot with real-time data."""
    regime_label: str
    regime_confidence: float
    vix: float
    spy_price: float
    treasury_10y: float
    risk_sentiment: str
    top_events: List[str]
    
    # Computed Indices
    inflation_pressure: float = 0.0
    growth_momentum: float = 0.0
    geopolitical_risk: float = 0.0
    financial_stress: float = 0.0
    
    # Additional market data
    spy_change_pct: float = 0.0
    gold_price: float = 0.0
    oil_price: float = 0.0
    cpi_yoy: float = 0.0
    unemployment_rate: float = 0.0
    fed_funds_rate: float = 0.0
    
    # Data quality
    has_live_data: bool = False


@dataclass
class ReportData:
    """Complete report data - assembled from live sources."""
    report_date: datetime
    report_type: str  # 'daily', 'weekly', 'monthly'
    
    # Core metrics
    portfolio: PortfolioMetrics
    positions: List[Position]
    trades: List[Trade]
    
    # Strategy data
    strategy_performance: List[StrategyPerformance]
    debate_summary: Dict[str, Any]
    
    # Macro context
    macro: MacroSnapshot
    
    # Learning insights
    patterns_found: int
    recommendations: List[str]
    
    # Time series for charts (pd.Series with datetime index)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    drawdown_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    
    # Data quality flags
    data_quality: Dict[str, bool] = field(default_factory=dict)


class LiveDataCollector:
    """
    Collects report data from live sources using VERIFIED broker methods.
    """
    
    def __init__(self, broker, learning_engine=None, app_state: Dict = None):
        """
        Initialize collector.
        
        Args:
            broker: AlpacaBroker instance
            learning_engine: LearningEngine instance (optional)
            app_state: Dict with app global state (optional)
        """
        self.broker = broker
        self.learning_engine = learning_engine
        self.app_state = app_state or {}
    
    def collect_report_data(
        self,
        report_date: datetime = None,
        report_type: str = 'daily',
        days_back: int = 30,
    ) -> ReportData:
        """
        Collect all data for report generation.
        
        Args:
            report_date: Report date (default: now)
            report_type: 'daily', 'weekly', or 'monthly'
            days_back: Days of history to fetch
        
        Returns:
            ReportData with all metrics
        """
        if report_date is None:
            report_date = datetime.now()
        
        data_quality = {}
        
        # 1. Get portfolio history (CRITICAL for metrics)
        logging.info("Fetching portfolio history from Alpaca...")
        equity_curve = self.broker.get_portfolio_history(days_back)
        has_portfolio_history = len(equity_curve) > 0
        data_quality['portfolio_history'] = has_portfolio_history
        logging.info(f"Portfolio history: {len(equity_curve)} data points")
        
        # 2. Get benchmark (SPY) history
        logging.info("Fetching SPY benchmark data...")
        benchmark_data = self.broker.get_historical_bars(['SPY'], days=days_back)
        benchmark_curve = benchmark_data.get('SPY', pd.Series(dtype=float))
        has_benchmark_data = len(benchmark_curve) > 0
        data_quality['benchmark'] = has_benchmark_data
        logging.info(f"Benchmark (SPY) data: {len(benchmark_curve)} data points")
        
        # 3. Collect portfolio metrics with REAL calculations
        portfolio = self._collect_portfolio_metrics(
            equity_curve, benchmark_curve, has_portfolio_history, has_benchmark_data
        )
        
        # 4. Get current positions and exposure
        positions, exposure = self._collect_positions()
        data_quality['positions'] = len(positions) > 0
        
        # Update portfolio with exposure metrics
        if exposure and portfolio.portfolio_value > 0:
            portfolio.gross_exposure = exposure.get('gross', 0)
            portfolio.net_exposure = exposure.get('net', 0)
            portfolio.long_exposure = exposure.get('long', 0)
            portfolio.short_exposure = exposure.get('short', 0)
            portfolio.leverage = exposure.get('gross', 0) / portfolio.portfolio_value
            portfolio.net_leverage = exposure.get('net', 0) / portfolio.portfolio_value
            portfolio.long_count = exposure.get('long_count', 0)
            portfolio.short_count = exposure.get('short_count', 0)
        
        # 5. Get trade history
        trades = self._collect_trades(days_back)
        data_quality['trades'] = len(trades) > 0
        
        # 6. Get strategy performance
        strategy_perf = self._collect_strategy_performance()
        data_quality['strategy_performance'] = len(strategy_perf) > 0
        
        # 7. Get debate summary
        debate_summary = self._collect_debate_summary()
        
        # 8. Get macro context
        macro = self._collect_macro_snapshot()
        
        # 9. Get learning insights
        patterns, recommendations = self._collect_learning_insights()
        
        # 10. Compute drawdown curve
        drawdown_curve = self._compute_drawdown_series(equity_curve)
        
        return ReportData(
            report_date=report_date,
            report_type=report_type,
            portfolio=portfolio,
            positions=positions,
            trades=trades,
            strategy_performance=strategy_perf,
            debate_summary=debate_summary,
            macro=macro,
            patterns_found=patterns,
            recommendations=recommendations,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            drawdown_curve=drawdown_curve,
            data_quality=data_quality,
        )
    
    def _collect_portfolio_metrics(
        self,
        equity_curve: pd.Series,
        benchmark_curve: pd.Series,
        has_portfolio_history: bool,
        has_benchmark_data: bool,
    ) -> PortfolioMetrics:
        """
        Collect portfolio metrics with REAL calculations from portfolio history.
        """
        try:
            # Get current account state
            account = self.broker.get_account()
            
            portfolio_value = float(account.get('equity', 0))
            cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            
            # Calculate returns from REAL portfolio history
            returns = self._compute_returns(equity_curve, portfolio_value) if has_portfolio_history else {}
            
            # Calculate benchmark returns
            benchmark_returns = self._compute_benchmark_returns(benchmark_curve) if has_benchmark_data else {}
            
            # Calculate volatility from REAL history
            volatility = self._compute_volatility(equity_curve) if has_portfolio_history else 0.0
            
            # Calculate drawdown from REAL history
            current_dd, max_dd = self._compute_drawdown(equity_curve) if has_portfolio_history else (0.0, 0.0)
            
            # Calculate Sharpe ratio from REAL history
            sharpe = self._compute_sharpe_ratio(equity_curve) if has_portfolio_history else 0.0
            
            # Calculate alpha
            alpha_1d = returns.get('1d', 0) - benchmark_returns.get('1d', 0)
            alpha_1m = returns.get('1m', 0) - benchmark_returns.get('1m', 0)
            
            # Determine if we have enough data for specific calculations
            data_points = len(equity_curve)
            has_sufficient_data_for_vol = data_points >= 20
            has_sufficient_data_for_sharpe = data_points >= 20
            
            return PortfolioMetrics(
                portfolio_value=portfolio_value,
                cash=cash,
                equity=portfolio_value,
                buying_power=buying_power,
                return_1d=returns.get('1d', 0),
                return_1w=returns.get('1w', 0),
                return_1m=returns.get('1m', 0),
                return_mtd=returns.get('mtd', 0),
                return_ytd=returns.get('ytd', 0),
                benchmark_return_1d=benchmark_returns.get('1d', 0),
                benchmark_return_1m=benchmark_returns.get('1m', 0),
                alpha_1d=alpha_1d,
                alpha_1m=alpha_1m,
                volatility_20d=volatility,
                current_drawdown=current_dd,
                max_drawdown_30d=max_dd,
                sharpe_ratio=sharpe,
                has_portfolio_history=has_portfolio_history,
                has_benchmark_data=has_benchmark_data,
                data_points=data_points,
                has_sufficient_data_for_vol=has_sufficient_data_for_vol,
                has_sufficient_data_for_sharpe=has_sufficient_data_for_sharpe,
            )
        except Exception as e:
            logging.error(f"Error collecting portfolio metrics: {e}")
            return PortfolioMetrics(
                portfolio_value=0, cash=0, equity=0, buying_power=0,
                has_portfolio_history=False, has_benchmark_data=False, data_points=0
            )
    
    def _compute_returns(self, history: pd.Series, current_value: float = None) -> Dict[str, float]:
        """Compute returns from REAL portfolio history."""
        returns = {'1d': 0, '1w': 0, '1m': 0, 'mtd': 0, 'ytd': 0}
        
        if len(history) < 2:
            logging.warning(f"Insufficient data for returns: {len(history)} points")
            return returns
        
        try:
            current = current_value if current_value else float(history.iloc[-1])
            
            # 1 day return
            if len(history) >= 2:
                prev = float(history.iloc[-2])
                if prev > 0:
                    returns['1d'] = (current - prev) / prev
                    logging.info(f"1D Return: {returns['1d']*100:.2f}%")
            
            # 1 week return (5 trading days)
            if len(history) >= 5:
                prev = float(history.iloc[-5])
                if prev > 0:
                    returns['1w'] = (current - prev) / prev
            
            # 1 month return (20 trading days)
            if len(history) >= 20:
                prev = float(history.iloc[-20])
                if prev > 0:
                    returns['1m'] = (current - prev) / prev
                    logging.info(f"1M Return: {returns['1m']*100:.2f}%")
            
            # MTD (month to date)
            now = datetime.now()
            month_start = datetime(now.year, now.month, 1)
            mtd_history = history[history.index >= pd.Timestamp(month_start)]
            if len(mtd_history) > 0:
                prev = float(mtd_history.iloc[0])
                if prev > 0:
                    returns['mtd'] = (current - prev) / prev
            
            # YTD (year to date)
            year_start = datetime(now.year, 1, 1)
            ytd_history = history[history.index >= pd.Timestamp(year_start)]
            if len(ytd_history) > 0:
                prev = float(ytd_history.iloc[0])
                if prev > 0:
                    returns['ytd'] = (current - prev) / prev
                    
        except Exception as e:
            logging.warning(f"Error computing returns: {e}")
        
        return returns
    
    def _compute_benchmark_returns(self, benchmark: pd.Series) -> Dict[str, float]:
        """Compute benchmark returns from SPY price history."""
        returns = {'1d': 0, '1w': 0, '1m': 0}
        
        if len(benchmark) < 2:
            return returns
        
        try:
            current = float(benchmark.iloc[-1])
            
            if len(benchmark) >= 2:
                prev = float(benchmark.iloc[-2])
                if prev > 0:
                    returns['1d'] = (current - prev) / prev
            
            if len(benchmark) >= 5:
                prev = float(benchmark.iloc[-5])
                if prev > 0:
                    returns['1w'] = (current - prev) / prev
            
            if len(benchmark) >= 20:
                prev = float(benchmark.iloc[-20])
                if prev > 0:
                    returns['1m'] = (current - prev) / prev
                    
        except Exception as e:
            logging.warning(f"Error computing benchmark returns: {e}")
        
        return returns
    
    def _compute_volatility(self, history: pd.Series, window: int = 20) -> float:
        """
        Compute annualized volatility from portfolio history.
        
        Formula: std(daily_returns) * sqrt(252)
        """
        if len(history) < max(2, window):
            logging.warning(f"Insufficient data for volatility: {len(history)} points, need {window}")
            return 0.0
        
        try:
            daily_returns = history.pct_change().dropna()
            
            if len(daily_returns) < 2:
                return 0.0
            
            # Use available data, but cap at window
            returns_to_use = daily_returns.tail(min(window, len(daily_returns)))
            
            vol = float(returns_to_use.std() * np.sqrt(252))
            logging.info(f"Volatility (20d): {vol*100:.1f}% annualized (from {len(returns_to_use)} returns)")
            return vol
            
        except Exception as e:
            logging.warning(f"Error computing volatility: {e}")
            return 0.0
    
    def _compute_drawdown(self, history: pd.Series) -> Tuple[float, float]:
        """Compute current and max drawdown from portfolio history."""
        if len(history) < 2:
            return 0.0, 0.0
        
        try:
            running_max = history.expanding().max()
            drawdown = (history - running_max) / running_max
            
            current_dd = float(drawdown.iloc[-1])
            max_dd = float(drawdown.min())
            
            logging.info(f"Drawdown: current={current_dd*100:.1f}%, max={max_dd*100:.1f}%")
            return current_dd, max_dd
            
        except Exception as e:
            logging.warning(f"Error computing drawdown: {e}")
            return 0.0, 0.0
    
    def _compute_drawdown_series(self, history: pd.Series) -> pd.Series:
        """Compute full drawdown series for charting."""
        if len(history) < 2:
            return pd.Series(dtype=float)
        
        try:
            running_max = history.expanding().max()
            drawdown = (history - running_max) / running_max
            return drawdown
        except Exception:
            return pd.Series(dtype=float)
    
    def _compute_sharpe_ratio(self, history: pd.Series, risk_free_rate: float = None) -> float:
        """
        Compute Sharpe ratio from portfolio history.
        
        Formula: (annualized_return - risk_free_rate) / annualized_volatility
        
        Uses actual 10Y Treasury rate if available, otherwise 5% default.
        """
        if len(history) < 20:
            logging.warning(f"Insufficient data for Sharpe: {len(history)} points, need 20+")
            return 0.0
        
        try:
            # Get risk-free rate from app state (actual Treasury rate) or default
            if risk_free_rate is None:
                if 'market_indicators' in self.app_state:
                    mi = self.app_state['market_indicators']
                    risk_free_rate = getattr(mi, 'treasury_10y', 5.0) / 100.0
                else:
                    risk_free_rate = 0.05  # Default 5%
            
            daily_returns = history.pct_change().dropna()
            
            if len(daily_returns) < 20:
                return 0.0
            
            # Annualized return
            mean_daily = daily_returns.mean()
            annualized_return = mean_daily * 252
            
            # Annualized volatility
            vol = daily_returns.std() * np.sqrt(252)
            
            if vol == 0:
                return 0.0
            
            # Sharpe ratio
            sharpe = (annualized_return - risk_free_rate) / vol
            
            logging.info(
                f"Sharpe: {sharpe:.2f} "
                f"(return={annualized_return*100:.1f}%, vol={vol*100:.1f}%, rf={risk_free_rate*100:.1f}%)"
            )
            return float(sharpe)
            
        except Exception as e:
            logging.warning(f"Error computing Sharpe: {e}")
            return 0.0
    
    def _collect_positions(self) -> Tuple[List[Position], Dict[str, float]]:
        """
        Collect current positions from Alpaca.
        
        Returns:
            Tuple of (positions list, exposure dict)
        """
        positions = []
        exposure = {
            "gross": 0.0,
            "net": 0.0,
            "long": 0.0,
            "short": 0.0,
            "long_count": 0,
            "short_count": 0,
        }
        
        try:
            raw_positions = self.broker.get_positions()
            
            # Calculate total absolute value for weights
            total_abs_value = sum(
                abs(float(p.get('market_value', 0))) for p in raw_positions.values()
            )
            
            for symbol, pos in raw_positions.items():
                qty = int(pos.get('qty', 0))
                market_value = float(pos.get('market_value', 0))
                
                # Determine side from quantity
                side = "short" if qty < 0 else "long"
                
                positions.append(Position(
                    symbol=symbol,
                    quantity=qty,
                    current_price=float(pos.get('current_price', 0)),
                    avg_cost=float(pos.get('avg_entry_price', 0)),
                    market_value=market_value,
                    weight=abs(market_value) / total_abs_value if total_abs_value > 0 else 0,
                    unrealized_pnl=float(pos.get('pnl', 0)),
                    unrealized_pnl_pct=float(pos.get('pnl_pct', 0)),
                    side=side,
                ))
                
                # Track exposure
                if side == "long":
                    exposure["long"] += market_value
                    exposure["long_count"] += 1
                else:
                    exposure["short"] += abs(market_value)
                    exposure["short_count"] += 1
            
            exposure["gross"] = exposure["long"] + exposure["short"]
            exposure["net"] = exposure["long"] - exposure["short"]
            
            # Sort by absolute weight (largest first)
            positions.sort(key=lambda p: abs(p.weight), reverse=True)
            
            long_count = exposure["long_count"]
            short_count = exposure["short_count"]
            logging.info(f"Positions: {len(positions)} holdings ({long_count} long, {short_count} short)")
            
        except Exception as e:
            logging.warning(f"Could not collect positions: {e}")
        
        return positions, exposure
    
    def _collect_trades(self, days_back: int = 7) -> List[Trade]:
        """Collect recent trades from Alpaca order history."""
        trades = []
        try:
            orders = self.broker.get_orders(status='all', limit=100)
            
            cutoff = datetime.now() - timedelta(days=days_back)
            
            for order in orders:
                # Only include filled orders
                if not hasattr(order, 'filled_at') or not order.filled_at:
                    continue
                if not hasattr(order, 'filled_qty') or not order.filled_qty:
                    continue
                    
                filled_at = order.filled_at
                if isinstance(filled_at, str):
                    try:
                        filled_at = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
                    except:
                        continue
                
                # Check if within time range
                if hasattr(filled_at, 'replace') and filled_at.replace(tzinfo=None) >= cutoff:
                    trades.append(Trade(
                        timestamp=filled_at,
                        symbol=order.symbol,
                        side=order.side.value if hasattr(order.side, 'value') else str(order.side),
                        quantity=int(order.filled_qty or 0),
                        filled_price=float(order.filled_avg_price or 0),
                        value=float(order.filled_qty or 0) * float(order.filled_avg_price or 0),
                        status='filled',
                    ))
            
            trades.sort(key=lambda t: t.timestamp, reverse=True)
            logging.info(f"Trades: {len(trades)} in last {days_back} days")
        except Exception as e:
            logging.warning(f"Could not collect trades: {e}")
        
        return trades
    
    def _collect_strategy_performance(self) -> List[StrategyPerformance]:
        """Collect strategy performance from learning engine and app state."""
        strategies = []
        try:
            # Get actual performance metrics from learning engine's performance tracker
            perf_metrics = {}
            if self.learning_engine:
                try:
                    # Access the actual performance tracker for real win rates
                    tracker = self.learning_engine.performance_tracker
                    for name, metrics in tracker.metrics.items():
                        perf_metrics[name] = {
                            'win_rate': metrics.win_rate if metrics.total_predictions > 0 else 0.5,
                            'accuracy': metrics.accuracy,
                            'confidence': metrics.avg_confidence_when_right if metrics.avg_confidence_when_right > 0 else 0.5,
                            'predictions': metrics.total_predictions,
                        }
                except Exception as e:
                    logging.debug(f"Could not get performance tracker metrics: {e}")
            
            # Get debate scores and weights from app state
            # The app stores these directly in last_run_status (NOT nested under debate_info)
            debate_scores = {}
            final_weights = {}
            last_run = self.app_state.get('last_run_status', {})
            
            # Primary source: debate_scores or strategy_scores (they contain the same data)
            raw_scores = last_run.get('debate_scores') or last_run.get('strategy_scores') or {}
            final_weights = last_run.get('final_weights', {})
            
            logging.debug(f"Raw debate scores from app_state: {raw_scores}")
            logging.debug(f"Final weights from app_state: {final_weights}")
            
            if isinstance(raw_scores, dict):
                for name, score in raw_scores.items():
                    # Skip non-strategy keys
                    if name in ['total_predictions', 'strategies_tracked', 'overall_accuracy', 'best_strategy']:
                        continue
                    debate_scores[name] = float(score) if isinstance(score, (int, float)) else 0.0
            
            # Combine all strategy names from different sources
            all_strategy_names = set()
            all_strategy_names.update(perf_metrics.keys())
            all_strategy_names.update(debate_scores.keys())
            all_strategy_names.update(final_weights.keys())
            
            # Get learned weights as additional source
            learned_weights = {}
            if self.learning_engine:
                try:
                    summary = self.learning_engine.get_learning_summary()
                    learned_weights = summary.get('learned_weights', {}).get('current_weights', {})
                    all_strategy_names.update(learned_weights.keys())
                except Exception:
                    pass
            
            # Build strategy list with actual data
            for name in all_strategy_names:
                perf = perf_metrics.get(name, {})
                
                # Get actual win rate (use accuracy as fallback)
                actual_win_rate = perf.get('win_rate', 0.0)
                if actual_win_rate == 0 and perf.get('predictions', 0) == 0:
                    # No data yet - use neutral but NOT 50% exactly to indicate no data
                    actual_win_rate = 0.0
                
                # Get actual confidence from calibration data
                actual_confidence = perf.get('confidence', 0.0)
                if actual_confidence == 0 and perf.get('predictions', 0) == 0:
                    actual_confidence = 0.0
                
                # Get debate score
                actual_debate_score = debate_scores.get(name, 0.0)
                
                # Get weight from best available source
                weight = final_weights.get(name) or learned_weights.get(name, 0.1)
                
                strategies.append(StrategyPerformance(
                    name=name,
                    weight=weight,
                    win_rate=actual_win_rate,
                    contribution=0,  # Would need trade-level data to calculate
                    confidence=actual_confidence,
                    debate_score=actual_debate_score,
                ))
            
            # Sort by debate score, then by win rate
            strategies.sort(key=lambda s: (s.debate_score, s.win_rate), reverse=True)
            
            # Log what we found for debugging
            logging.info(f"Collected {len(strategies)} strategies: " + 
                        ", ".join([f"{s.name}(wr={s.win_rate:.0%}, ds={s.debate_score:.2f})" 
                                   for s in strategies[:3]]))
            
        except Exception as e:
            logging.warning(f"Could not collect strategy performance: {e}")
            import traceback
            logging.debug(traceback.format_exc())
        
        return strategies
    
    def _collect_debate_summary(self) -> Dict[str, Any]:
        """Collect debate summary from app state."""
        try:
            if 'last_run_status' in self.app_state:
                debate_info = self.app_state['last_run_status'].get('debate_info', {})
                return {
                    'top_strategies': list(debate_info.get('scores', {}).keys())[:3],
                    'final_weights': debate_info.get('final_weights', {}),
                    'transcript_available': bool(debate_info.get('transcript')),
                }
        except Exception as e:
            logging.warning(f"Could not collect debate summary: {e}")
        
        return {'top_strategies': [], 'final_weights': {}, 'transcript_available': False}
    
    def _collect_macro_snapshot(self) -> MacroSnapshot:
        """
        Collect macro snapshot using LIVE data from Yahoo Finance and FRED.
        
        Falls back to app state if live fetch fails.
        """
        try:
            # === FETCH LIVE MACRO DATA ===
            from src.data.macro_data import get_macro_loader
            import os
            
            fred_key = os.getenv('FRED_API_KEY')
            macro_loader = get_macro_loader(fred_key)
            indicators = macro_loader.fetch_all()
            
            logging.info(f"Fetched live macro data: VIX={indicators.vix:.1f}, SPY=${indicators.spy_price:.2f}")
            
            # === DETERMINE REGIME FROM REAL DATA ===
            regime_label = self._compute_regime(indicators)
            regime_confidence = abs(indicators.risk_score - 0.5) * 2  # How far from neutral
            
            # === CONVERT RISK SCORE TO SENTIMENT ===
            if indicators.risk_score > 0.6:
                risk_sentiment = 'RISK_ON'
            elif indicators.risk_score < 0.4:
                risk_sentiment = 'RISK_OFF'
            else:
                risk_sentiment = 'NEUTRAL'
            
            # === COMPUTE MACRO INDICES FROM REAL DATA ===
            # Inflation pressure: CPI YoY - normalize to -1 to 1
            inflation_pressure = (indicators.cpi_yoy - 3.0) / 3.0 if indicators.cpi_yoy else 0.0
            
            # Growth momentum: SPY vs 200MA
            growth_momentum = indicators.spy_vs_200ma / 5.0 if indicators.spy_vs_200ma else 0.0
            
            # Financial stress: Direct from FRED
            fin_stress = indicators.financial_stress_index
            
            # Geo risk: VIX above normal suggests uncertainty
            geo_risk = (indicators.vix - 20.0) / 15.0 if indicators.vix else 0.0
            
            # === GET TOP EVENTS FROM APP STATE (if available) ===
            top_events = []
            macro_features = self.app_state.get('last_macro_features', {})
            features = macro_features.get('features')
            if features and hasattr(features, 'top_events'):
                top_events = features.top_events[:5]
            
            return MacroSnapshot(
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                vix=indicators.vix,
                spy_price=indicators.spy_price,
                treasury_10y=indicators.treasury_10y,
                risk_sentiment=risk_sentiment,
                top_events=top_events,
                inflation_pressure=inflation_pressure,
                growth_momentum=growth_momentum,
                geopolitical_risk=geo_risk,
                financial_stress=fin_stress,
                spy_change_pct=indicators.spy_change_pct,
                gold_price=indicators.gold_price,
                oil_price=indicators.oil_price,
                cpi_yoy=indicators.cpi_yoy,
                unemployment_rate=indicators.unemployment_rate,
                fed_funds_rate=indicators.fed_funds_rate,
                has_live_data=True,
            )
            
        except Exception as e:
            logging.warning(f"Could not fetch live macro data: {e}, falling back to app state")
            return self._collect_macro_snapshot_from_app_state()
    
    def _compute_regime(self, indicators) -> str:
        """Compute market regime from macro indicators."""
        score = indicators.risk_score
        vix = indicators.vix
        spy_change = indicators.spy_change_pct
        
        # High VIX = bear/crisis
        if vix > 30:
            return 'crisis'
        elif vix > 25:
            return 'high_vol_bear'
        
        # Risk score determines bull/bear
        if score > 0.6 and spy_change > 0:
            return 'bull_trending'
        elif score > 0.55:
            return 'bull_sideways'
        elif score < 0.35:
            return 'bear_trending'
        elif score < 0.45:
            return 'bear_sideways'
        else:
            return 'neutral'
    
    def _collect_macro_snapshot_from_app_state(self) -> MacroSnapshot:
        """Fallback: collect macro snapshot from app state."""
        try:
            macro_features = self.app_state.get('last_macro_features', {})
            features = macro_features.get('features')
            
            # Get regime from app state
            regime_label = 'unknown'
            regime_confidence = 0.5
            if 'current_regime' in self.app_state:
                regime = self.app_state['current_regime']
                if hasattr(regime, 'risk_regime'):
                    regime_label = regime.risk_regime.value
                    regime_confidence = getattr(regime, 'volatility_percentile', 0.5)
            
            # Get market data from app state
            vix = 20.0
            spy_price = 0.0
            treasury_10y = 4.0
            
            if 'market_indicators' in self.app_state:
                mi = self.app_state['market_indicators']
                vix = getattr(mi, 'vix', 20.0)
                spy_price = getattr(mi, 'spy_price', 0)
                treasury_10y = getattr(mi, 'treasury_10y', 4.0)
            
            # Get indices from features
            inflation = getattr(features, 'macro_inflation_pressure_index', 0) if features else 0
            growth = getattr(features, 'growth_momentum_index', 0) if features else 0
            geo_risk = getattr(features, 'geopolitical_risk_index', 0) if features else 0
            fin_stress = getattr(features, 'financial_stress_index', 0) if features else 0
            
            # Risk sentiment
            risk_sentiment = 'NEUTRAL'
            if macro_features.get('risk_sentiment'):
                rs = macro_features['risk_sentiment']
                risk_sentiment = rs.value if hasattr(rs, 'value') else str(rs)
            
            # Top events
            top_events = []
            if features and hasattr(features, 'top_events'):
                top_events = features.top_events[:5]
            
            return MacroSnapshot(
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                vix=vix,
                spy_price=spy_price,
                treasury_10y=treasury_10y,
                risk_sentiment=risk_sentiment,
                top_events=top_events,
                inflation_pressure=inflation,
                growth_momentum=growth,
                geopolitical_risk=geo_risk,
                financial_stress=fin_stress,
            )
        except Exception as e:
            logging.error(f"Macro snapshot fallback failed: {e}")
            return MacroSnapshot(
                regime_label='unknown', regime_confidence=0.5,
                vix=20.0, spy_price=0, treasury_10y=4.0,
                risk_sentiment='NEUTRAL', top_events=[],
            )
    
    def _collect_learning_insights(self) -> Tuple[int, List[str]]:
        """Collect learning insights."""
        patterns = 0
        recommendations = []
        
        try:
            if self.learning_engine:
                summary = self.learning_engine.get_learning_summary()
                patterns = summary.get('patterns_found', 0)
                recommendations = summary.get('recommendations', [])[:5]
        except Exception as e:
            logging.warning(f"Could not collect learning insights: {e}")
        
        return patterns, recommendations
