#!/usr/bin/env python3
"""
Quick script to run a backtest with sample data.
Demonstrates the multi-strategy debate bot in action.
"""
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

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
)
from src.backtest import Backtester, BacktestConfig
from src.risk import RiskConstraints


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    """Run sample backtest."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 60)
    print("MULTI-STRATEGY QUANT DEBATE BOT")
    print("Sample Backtest")
    print("=" * 60 + "\n")
    
    # Configuration
    universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE',
        'KO', 'PEP', 'PG', 'WMT', 'HD', 'DIS', 'NFLX'
    ]
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Create data loaders
    logger.info("Setting up data loaders...")
    market_loader = MarketDataLoader()
    news_loader = NewsDataLoader()
    sentiment_analyzer = SentimentAnalyzer()
    regime_classifier = RegimeClassifier()
    
    # Create feature store
    feature_store = FeatureStore(
        market_loader, news_loader, sentiment_analyzer, regime_classifier
    )
    
    # Try to load real data, fall back to sample
    logger.info("Loading market data...")
    try:
        feature_store.load_data(universe, start_date, end_date, source='alpaca')
        if len(feature_store._price_history) == 0:
            raise ValueError("No data loaded from Alpaca")
        logger.info(f"Loaded real market data for {len(feature_store._price_history)} symbols")
    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
        logger.info("Generating sample data for demo...")
        
        sample_data = market_loader.generate_sample_data(
            universe, start_date - timedelta(days=300), end_date
        )
        feature_store._price_history = sample_data
        
        sample_news = news_loader.generate_sample_news(
            universe, start_date, end_date
        )
        feature_store._news_cache = sample_news
        logger.info(f"Generated sample data for {len(sample_data)} symbols")
    
    # Create strategies
    logger.info("Creating strategies...")
    strategies = [
        TimeSeriesMomentumStrategy(),
        CrossSectionMomentumStrategy({'top_n': 5}),
        MeanReversionStrategy(),
        VolatilityRegimeVolTargetStrategy(),
        CarryStrategy(),
        ValueQualityTiltStrategy(),
        RiskParityMinVarStrategy(),
        TailRiskOverlayStrategy(),
        NewsSentimentEventStrategy(),
    ]
    
    logger.info(f"Created {len(strategies)} strategies:")
    for s in strategies:
        print(f"  - {s.name}")
    
    # Create backtest config
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency='weekly',
        initial_capital=100000,
        transaction_cost_bps=10,
        slippage_bps=5,
        ensemble_mode='weighted_vote',
        output_dir='outputs',
    )
    
    # Create risk constraints
    risk_constraints = RiskConstraints(
        max_position_size=0.15,
        max_sector_exposure=0.30,
        max_leverage=1.0,
        max_turnover=0.50,
        vol_target=0.12,
    )
    
    # Run backtest
    logger.info("\nRunning backtest...")
    backtester = Backtester(strategies, feature_store, config, risk_constraints)
    result = backtester.run()
    
    # Print results
    print("\n" + result.summary())
    
    # Save results
    output_path = backtester.save_results(result)
    
    # Generate HTML report
    html = backtester.generate_report(result)
    report_path = Path(output_path) / 'report.html'
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"\nReport saved to: {report_path}")
    
    # Print sample debate transcript
    if result.debate_transcripts:
        print("\n" + "=" * 60)
        print("SAMPLE DEBATE TRANSCRIPT")
        print(result.debate_transcripts[-1].to_string())
    
    return result


if __name__ == '__main__':
    main()
