"""
CLI for the Multi-Strategy Quant Debate Bot.
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.market_data import MarketDataLoader
from src.data.news_data import NewsDataLoader
from src.data.sentiment import SentimentAnalyzer
from src.data.feature_store import FeatureStore
from src.data.regime import RegimeClassifier
from src.strategies import (
    TimeSeriesMomentumStrategy,
    CrossSectionMomentumStrategy,
    MeanReversionStrategy,
    VolatilityRegimeVolTargetStrategy,
    CarryStrategy,
    ValueQualityTiltStrategy,
    RiskParityMinVarStrategy,
    TailRiskOverlayStrategy,
    NewsSentimentEventStrategy,
    MLMetaEnsembleStrategy,
)
from src.backtest import Backtester, BacktestConfig
from src.risk import RiskConstraints


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_strategies(config: dict) -> list:
    """Create strategy instances from config."""
    strategies = []
    
    strat_config = config.get('strategies', {})
    
    # Create individual strategies
    if strat_config.get('time_series_momentum', {}).get('enabled', True):
        strategies.append(TimeSeriesMomentumStrategy(strat_config.get('time_series_momentum')))
    
    if strat_config.get('cross_section_momentum', {}).get('enabled', True):
        strategies.append(CrossSectionMomentumStrategy(strat_config.get('cross_section_momentum')))
    
    if strat_config.get('mean_reversion', {}).get('enabled', True):
        strategies.append(MeanReversionStrategy(strat_config.get('mean_reversion')))
    
    if strat_config.get('volatility', {}).get('enabled', True):
        strategies.append(VolatilityRegimeVolTargetStrategy(strat_config.get('volatility')))
    
    if strat_config.get('carry', {}).get('enabled', True):
        strategies.append(CarryStrategy(strat_config.get('carry')))
    
    if strat_config.get('value_quality', {}).get('enabled', True):
        strategies.append(ValueQualityTiltStrategy(strat_config.get('value_quality')))
    
    if strat_config.get('risk_parity', {}).get('enabled', True):
        strategies.append(RiskParityMinVarStrategy(strat_config.get('risk_parity')))
    
    if strat_config.get('tail_risk', {}).get('enabled', True):
        strategies.append(TailRiskOverlayStrategy(strat_config.get('tail_risk')))
    
    if strat_config.get('sentiment', {}).get('enabled', True):
        strategies.append(NewsSentimentEventStrategy(strat_config.get('sentiment')))
    
    # ML Meta-Ensemble uses the other strategies
    if strat_config.get('ml_ensemble', {}).get('enabled', False):
        strategies.append(MLMetaEnsembleStrategy(strategies.copy(), strat_config.get('ml_ensemble')))
    
    return strategies


def cmd_backtest(args, config: dict):
    """Run backtest command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting backtest...")
    
    # Parse dates
    backtest_config = config.get('backtest', {})
    start_date = datetime.strptime(
        args.start_date or backtest_config.get('start_date', '2024-01-01'),
        '%Y-%m-%d'
    )
    end_date = datetime.strptime(
        args.end_date or backtest_config.get('end_date', '2025-01-01'),
        '%Y-%m-%d'
    )
    
    # Create data loaders
    data_config = config.get('data', {})
    market_loader = MarketDataLoader(data_config.get('data_path'))
    news_loader = NewsDataLoader(data_config.get('news_path'))
    sentiment_analyzer = SentimentAnalyzer(
        decay_halflife_days=data_config.get('sentiment_decay', 3.0)
    )
    regime_classifier = RegimeClassifier()
    
    # Create feature store
    feature_store = FeatureStore(
        market_loader, news_loader, sentiment_analyzer, regime_classifier
    )
    
    # Load data
    universe = config.get('universe', [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH',
        'KO', 'PEP', 'PG', 'WMT', 'HD', 'DIS', 'NFLX'
    ])
    
    # Try to load real data, fall back to sample
    source = data_config.get('source', 'alpaca')
    try:
        feature_store.load_data(universe, start_date, end_date, source)
    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
        logger.info("Generating sample data for demo...")
        
        # Generate sample data
        sample_data = market_loader.generate_sample_data(
            universe, start_date - timedelta(days=300), end_date
        )
        feature_store._price_history = sample_data
        
        # Generate sample news
        sample_news = news_loader.generate_sample_news(
            universe, start_date, end_date
        )
        feature_store._news_cache = sample_news
    
    # Create strategies
    strategies = create_strategies(config)
    logger.info(f"Created {len(strategies)} strategies")
    
    # Create backtest config
    bt_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=backtest_config.get('frequency', 'weekly'),
        initial_capital=backtest_config.get('initial_capital', 100000),
        transaction_cost_bps=backtest_config.get('transaction_cost_bps', 10),
        slippage_bps=backtest_config.get('slippage_bps', 5),
        ensemble_mode=backtest_config.get('ensemble_mode', 'weighted_vote'),
        output_dir=args.output or 'outputs',
    )
    
    # Create risk constraints
    risk_config = config.get('risk', {})
    risk_constraints = RiskConstraints(
        max_position_size=risk_config.get('max_position', 0.15),
        max_sector_exposure=risk_config.get('max_sector', 0.30),
        max_leverage=risk_config.get('max_leverage', 1.0),
        max_turnover=risk_config.get('max_turnover', 0.50),
        vol_target=risk_config.get('vol_target', 0.12),
    )
    
    # Run backtest
    backtester = Backtester(strategies, feature_store, bt_config, risk_constraints)
    result = backtester.run()
    
    # Save results
    output_path = backtester.save_results(result)
    
    # Generate HTML report
    html = backtester.generate_report(result)
    report_path = Path(output_path) / 'report.html'
    with open(report_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Report saved to {report_path}")
    
    print(result.summary())


def cmd_papertrade(args, config: dict):
    """Run paper trading command."""
    logger = logging.getLogger(__name__)
    logger.info("Paper trading mode - generating signals only...")
    
    # Similar setup to backtest, but for current date
    data_config = config.get('data', {})
    market_loader = MarketDataLoader(data_config.get('data_path'))
    
    feature_store = FeatureStore(
        market_loader,
        NewsDataLoader(data_config.get('news_path')),
        SentimentAnalyzer(),
        RegimeClassifier()
    )
    
    universe = config.get('universe', [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'XOM', 'JNJ', 'KO', 'PG', 'WMT'
    ])
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)
    
    source = data_config.get('source', 'alpaca')
    try:
        feature_store.load_data(universe, start_date, end_date, source)
    except Exception as e:
        logger.error(f"Could not load market data: {e}")
        logger.info("Using sample data for demo")
        sample_data = market_loader.generate_sample_data(
            universe, start_date, end_date
        )
        feature_store._price_history = sample_data
    
    strategies = create_strategies(config)
    
    # Get current features
    features = feature_store.get_features(end_date)
    
    # Generate signals
    print("\n" + "=" * 60)
    print("PAPER TRADING SIGNALS")
    print("=" * 60)
    print(f"Date: {end_date.strftime('%Y-%m-%d')}")
    
    if features.regime:
        print(f"Regime: {features.regime.description}")
    
    print("\nStrategy Signals:")
    print("-" * 60)
    
    for strategy in strategies:
        try:
            signal = strategy.generate_signals(features, end_date)
            print(f"\n{strategy.name}:")
            print(f"  Confidence: {signal.confidence:.1%}")
            print(f"  Expected Return: {signal.expected_return:.2%}")
            print(f"  Risk: {signal.risk_estimate:.2%}")
            
            top_positions = sorted(
                signal.desired_weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            if top_positions:
                print("  Top Positions:")
                for symbol, weight in top_positions:
                    print(f"    {symbol}: {weight:.1%}")
                    
        except Exception as e:
            print(f"\n{strategy.name}: ERROR - {e}")
    
    # Save signals to file
    signals_path = Path(args.output or 'outputs') / 'signals.json'
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    signals_data = {
        'timestamp': end_date.isoformat(),
        'regime': features.regime.description if features.regime else None,
        'signals': {}
    }
    
    for strategy in strategies:
        try:
            signal = strategy.generate_signals(features, end_date)
            signals_data['signals'][strategy.name] = {
                'weights': signal.desired_weights,
                'confidence': signal.confidence,
                'expected_return': signal.expected_return,
            }
        except:
            pass
    
    with open(signals_path, 'w') as f:
        json.dump(signals_data, f, indent=2)
    
    print(f"\nSignals saved to {signals_path}")


def cmd_report(args, config: dict):
    """Generate report from saved results."""
    logger = logging.getLogger(__name__)
    
    output_dir = Path(args.output or 'outputs')
    
    # Find latest results
    nav_files = list(output_dir.glob('nav_*.csv'))
    if not nav_files:
        logger.error("No backtest results found")
        return
    
    latest = max(nav_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest.stem.replace('nav_', '')
    
    logger.info(f"Generating report from {timestamp}")
    
    # Load data and generate report
    import pandas as pd
    
    nav = pd.read_csv(latest, index_col=0, parse_dates=True)
    
    print(f"NAV series loaded: {len(nav)} data points")
    print(f"Final NAV: ${nav.iloc[-1].values[0]:,.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Strategy Quant Debate Bot"
    )
    parser.add_argument(
        '--config', '-c',
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level', '-l',
        default='INFO',
        help='Logging level'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run backtest')
    bt_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    bt_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    bt_parser.add_argument('--output', '-o', help='Output directory')
    
    # Paper trade command
    pt_parser = subparsers.add_parser('papertrade', help='Generate paper trading signals')
    pt_parser.add_argument('--output', '-o', help='Output directory')
    
    # Report command
    rp_parser = subparsers.add_parser('report', help='Generate report')
    rp_parser.add_argument('--output', '-o', help='Results directory')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logging.warning(f"Config not found: {config_path}, using defaults")
        config = {}
    
    # Run command
    if args.command == 'backtest':
        cmd_backtest(args, config)
    elif args.command == 'papertrade':
        cmd_papertrade(args, config)
    elif args.command == 'report':
        cmd_report(args, config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
