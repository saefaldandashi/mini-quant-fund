"""
Walk-forward backtester with performance attribution.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from src.data.feature_store import FeatureStore, Features
from src.data.market_data import MarketDataLoader
from src.strategies.base import Strategy, SignalOutput
from src.debate.debate_engine import DebateEngine, DebateTranscript
from src.debate.ensemble import EnsembleOptimizer, EnsembleMode
from src.risk.risk_manager import RiskManager, RiskConstraints

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: datetime
    end_date: datetime
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 10.0  # 10 bps = 0.1%
    slippage_bps: float = 5.0
    ensemble_mode: str = "weighted_vote"
    output_dir: str = "outputs"


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Performance metrics
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_turnover: float = 0.0
    
    # Time series
    nav_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Attribution
    strategy_weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Debate transcripts
    debate_transcripts: List[DebateTranscript] = field(default_factory=list)
    
    # Metadata
    config: Optional[BacktestConfig] = None
    runtime_seconds: float = 0.0
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "BACKTEST RESULTS",
            "=" * 50,
            f"Total Return: {self.total_return:.2%}",
            f"CAGR: {self.cagr:.2%}",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Sortino Ratio: {self.sortino_ratio:.2f}",
            f"Max Drawdown: {self.max_drawdown:.2%}",
            f"Avg Turnover: {self.avg_turnover:.2%}",
            "=" * 50,
        ]
        
        if self.strategy_contributions:
            lines.append("STRATEGY CONTRIBUTIONS:")
            for name, contrib in sorted(
                self.strategy_contributions.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                lines.append(f"  {name}: {contrib:.2%}")
        
        return "\n".join(lines)


class Backtester:
    """
    Walk-forward backtester with debate mechanism.
    """
    
    def __init__(
        self,
        strategies: List[Strategy],
        feature_store: FeatureStore,
        config: BacktestConfig,
        risk_constraints: Optional[RiskConstraints] = None
    ):
        """
        Initialize backtester.
        
        Args:
            strategies: List of strategies to test
            feature_store: Feature store with loaded data
            config: Backtest configuration
            risk_constraints: Risk constraints
        """
        self.strategies = strategies
        self.feature_store = feature_store
        self.config = config
        
        # Components
        self.debate_engine = DebateEngine()
        self.ensemble = EnsembleOptimizer()
        self.risk_manager = RiskManager(risk_constraints)
        
        # State
        self.current_weights: Dict[str, float] = {}
        self.nav = config.initial_capital
        self.nav_history: List[Tuple[datetime, float]] = []
        self.weights_history: List[Tuple[datetime, Dict[str, float]]] = []
        self.strategy_weights_history: List[Tuple[datetime, Dict[str, float]]] = []
        self.debate_transcripts: List[DebateTranscript] = []
        self.turnovers: List[float] = []
    
    def run(self) -> BacktestResult:
        """
        Run the backtest.
        
        Returns:
            BacktestResult with all metrics and history
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Get rebalance dates
        rebalance_dates = self.feature_store.get_rebalance_dates(
            self.config.start_date,
            self.config.end_date,
            self.config.rebalance_frequency
        )
        
        logger.info(f"Running {len(rebalance_dates)} rebalance periods")
        
        # Initialize
        self.nav = self.config.initial_capital
        self.nav_history = [(self.config.start_date, self.nav)]
        last_features = None
        
        for i, date in enumerate(rebalance_dates):
            try:
                # Get features
                features = self.feature_store.get_features(date)
                
                if not features.symbols:
                    logger.warning(f"No features available for {date}")
                    continue
                
                # Update NAV based on market moves since last rebalance
                if last_features is not None and self.current_weights:
                    self._update_nav(last_features, features)
                
                # Record NAV
                self.nav_history.append((date, self.nav))
                
                # Run rebalance
                new_weights, transcript = self._rebalance(features)
                
                # Apply transaction costs
                turnover = self._calculate_turnover(new_weights)
                self.turnovers.append(turnover)
                
                cost = turnover * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000
                self.nav *= (1 - cost)
                
                # Update state
                self.current_weights = new_weights
                self.weights_history.append((date, dict(new_weights)))
                self.debate_transcripts.append(transcript)
                
                last_features = features
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(rebalance_dates)} periods, NAV: ${self.nav:,.0f}")
                    
            except Exception as e:
                logger.error(f"Error on {date}: {e}")
                continue
        
        # Build results
        result = self._build_result()
        result.runtime_seconds = time.time() - start_time
        result.config = self.config
        
        logger.info(f"Backtest completed in {result.runtime_seconds:.1f}s")
        logger.info(result.summary())
        
        return result
    
    def _rebalance(
        self,
        features: Features
    ) -> Tuple[Dict[str, float], DebateTranscript]:
        """
        Run the rebalance process.
        """
        # Get signals from all strategies
        signals: Dict[str, SignalOutput] = {}
        
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signals(features, features.timestamp)
                signals[strategy.name] = signal
            except Exception as e:
                logger.debug(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not signals:
            return {}, DebateTranscript(
                timestamp=features.timestamp,
                regime=features.regime,
                summary="No signals generated"
            )
        
        # Run debate
        scores, transcript = self.debate_engine.run_debate(
            signals, features, self.risk_manager.current_drawdown
        )
        
        # Ensemble
        ensemble_mode = EnsembleMode(self.config.ensemble_mode)
        combined_weights, metadata = self.ensemble.combine(
            signals, scores, features, self.current_weights, ensemble_mode
        )
        
        # Record strategy weights
        strategy_weights = metadata.get('strategy_contributions', {})
        self.strategy_weights_history.append((features.timestamp, strategy_weights))
        
        # Risk check
        risk_result = self.risk_manager.check_and_approve(
            combined_weights, features, self.current_weights, self.nav
        )
        
        transcript.final_weights = risk_result.approved_weights
        transcript.constraints_applied = risk_result.adjustments
        
        return risk_result.approved_weights, transcript
    
    def _update_nav(
        self,
        old_features: Features,
        new_features: Features
    ) -> None:
        """Update NAV based on price changes."""
        pnl = 0.0
        
        for symbol, weight in self.current_weights.items():
            old_price = old_features.prices.get(symbol)
            new_price = new_features.prices.get(symbol)
            
            if old_price and new_price and old_price > 0:
                ret = (new_price - old_price) / old_price
                pnl += weight * ret
        
        self.nav *= (1 + pnl)
    
    def _calculate_turnover(self, new_weights: Dict[str, float]) -> float:
        """Calculate turnover from current to new weights."""
        all_symbols = set(new_weights.keys()) | set(self.current_weights.keys())
        return sum(
            abs(new_weights.get(s, 0) - self.current_weights.get(s, 0))
            for s in all_symbols
        ) / 2
    
    def _build_result(self) -> BacktestResult:
        """Build backtest result object."""
        result = BacktestResult()
        
        # Build NAV series
        if self.nav_history:
            dates, navs = zip(*self.nav_history)
            result.nav_series = pd.Series(navs, index=pd.DatetimeIndex(dates))
            
            # Returns
            result.returns_series = result.nav_series.pct_change().dropna()
            
            # Performance metrics
            result.total_return = (self.nav - self.config.initial_capital) / self.config.initial_capital
            
            n_years = (self.config.end_date - self.config.start_date).days / 365.25
            if n_years > 0:
                result.cagr = (self.nav / self.config.initial_capital) ** (1 / n_years) - 1
            
            if len(result.returns_series) > 0:
                mean_ret = result.returns_series.mean() * 252
                std_ret = result.returns_series.std() * np.sqrt(252)
                
                if std_ret > 0:
                    result.sharpe_ratio = mean_ret / std_ret
                
                downside_ret = result.returns_series[result.returns_series < 0]
                if len(downside_ret) > 0:
                    downside_std = downside_ret.std() * np.sqrt(252)
                    if downside_std > 0:
                        result.sortino_ratio = mean_ret / downside_std
            
            # Max drawdown
            rolling_max = result.nav_series.expanding().max()
            drawdown = (result.nav_series - rolling_max) / rolling_max
            result.max_drawdown = abs(drawdown.min())
        
        # Turnover
        if self.turnovers:
            result.avg_turnover = np.mean(self.turnovers)
        
        # Weights history
        if self.weights_history:
            dates, weights = zip(*self.weights_history)
            result.weights_history = pd.DataFrame(weights, index=pd.DatetimeIndex(dates))
        
        # Strategy weights history
        if self.strategy_weights_history:
            dates, strat_weights = zip(*self.strategy_weights_history)
            result.strategy_weights_history = pd.DataFrame(
                strat_weights, index=pd.DatetimeIndex(dates)
            )
            
            # Average contributions
            result.strategy_contributions = result.strategy_weights_history.mean().to_dict()
        
        # Transcripts
        result.debate_transcripts = self.debate_transcripts
        
        return result
    
    def save_results(self, result: BacktestResult, output_dir: Optional[str] = None) -> str:
        """
        Save backtest results to files.
        
        Args:
            result: Backtest result
            output_dir: Output directory
            
        Returns:
            Path to output directory
        """
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = output_path / f"summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(result.summary())
        
        # Save NAV series
        if not result.nav_series.empty:
            nav_path = output_path / f"nav_{timestamp}.csv"
            result.nav_series.to_csv(nav_path, header=['NAV'])
        
        # Save weights history
        if not result.weights_history.empty:
            weights_path = output_path / f"weights_{timestamp}.csv"
            result.weights_history.to_csv(weights_path)
        
        # Save strategy contributions
        strat_path = output_path / f"strategy_contributions_{timestamp}.json"
        with open(strat_path, 'w') as f:
            json.dump(result.strategy_contributions, f, indent=2)
        
        # Save config
        config_path = output_path / f"config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump({
                'start_date': str(result.config.start_date) if result.config else None,
                'end_date': str(result.config.end_date) if result.config else None,
                'rebalance_frequency': result.config.rebalance_frequency if result.config else None,
                'initial_capital': result.config.initial_capital if result.config else None,
                'transaction_cost_bps': result.config.transaction_cost_bps if result.config else None,
                'ensemble_mode': result.config.ensemble_mode if result.config else None,
            }, f, indent=2)
        
        # Save sample debate transcripts
        if result.debate_transcripts:
            debates_path = output_path / f"debates_{timestamp}.txt"
            with open(debates_path, 'w') as f:
                # Save first and last few transcripts
                for t in result.debate_transcripts[:3] + result.debate_transcripts[-3:]:
                    f.write(t.to_string())
                    f.write("\n\n")
        
        logger.info(f"Results saved to {output_path}")
        return str(output_path)
    
    def generate_report(self, result: BacktestResult) -> str:
        """
        Generate HTML report.
        
        Args:
            result: Backtest result
            
        Returns:
            HTML content
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d4ff; }}
        h2 {{ color: #0099cc; border-bottom: 1px solid #333; padding-bottom: 10px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 15px; background: #16213e; border-radius: 8px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
        .metric-label {{ font-size: 12px; color: #888; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #0f3460; color: #00d4ff; }}
        tr:hover {{ background: #1a1a3e; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
    </style>
</head>
<body>
    <h1>Multi-Strategy Quant Debate Bot - Backtest Report</h1>
    
    <h2>Performance Summary</h2>
    <div class="metric">
        <div class="metric-value {'positive' if result.total_return > 0 else 'negative'}">{result.total_return:.2%}</div>
        <div class="metric-label">Total Return</div>
    </div>
    <div class="metric">
        <div class="metric-value">{result.cagr:.2%}</div>
        <div class="metric-label">CAGR</div>
    </div>
    <div class="metric">
        <div class="metric-value">{result.sharpe_ratio:.2f}</div>
        <div class="metric-label">Sharpe Ratio</div>
    </div>
    <div class="metric">
        <div class="metric-value">{result.sortino_ratio:.2f}</div>
        <div class="metric-label">Sortino Ratio</div>
    </div>
    <div class="metric">
        <div class="metric-value negative">{result.max_drawdown:.2%}</div>
        <div class="metric-label">Max Drawdown</div>
    </div>
    <div class="metric">
        <div class="metric-value">{result.avg_turnover:.2%}</div>
        <div class="metric-label">Avg Turnover</div>
    </div>
    
    <h2>Strategy Contributions</h2>
    <table>
        <tr><th>Strategy</th><th>Avg Weight</th></tr>
        {''.join(f"<tr><td>{name}</td><td>{weight:.2%}</td></tr>" for name, weight in sorted(result.strategy_contributions.items(), key=lambda x: x[1], reverse=True))}
    </table>
    
    <h2>Configuration</h2>
    <table>
        <tr><td>Start Date</td><td>{result.config.start_date if result.config else 'N/A'}</td></tr>
        <tr><td>End Date</td><td>{result.config.end_date if result.config else 'N/A'}</td></tr>
        <tr><td>Rebalance Frequency</td><td>{result.config.rebalance_frequency if result.config else 'N/A'}</td></tr>
        <tr><td>Initial Capital</td><td>${result.config.initial_capital:,.0f}</td></tr>
        <tr><td>Ensemble Mode</td><td>{result.config.ensemble_mode if result.config else 'N/A'}</td></tr>
    </table>
    
    <footer style="margin-top: 50px; color: #666; font-size: 12px;">
        Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Runtime: {result.runtime_seconds:.1f}s
    </footer>
</body>
</html>
"""
        return html
