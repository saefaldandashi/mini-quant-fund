#!/usr/bin/env python3
"""
CLI tool for generating reports.

Usage:
    python generate_report.py --type daily --date 2026-01-20
    python generate_report.py --type weekly --week 2026-W03
    python generate_report.py --type monthly --month 2026-01
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.reporting import ReportEngine, ReportType
from broker_alpaca import AlpacaBroker
from src.data.market_data import MarketDataLoader
from src.data.news_data import NewsDataLoader
from src.data.sentiment import SentimentAnalyzer
from src.data.regime import RegimeClassifier
from src.data.feature_store import FeatureStore
from src.learning.learning_engine import LearningEngine


def main():
    parser = argparse.ArgumentParser(description='Generate investment reports')
    parser.add_argument('--type', choices=['daily', 'weekly', 'monthly'], 
                       required=True, help='Report type')
    parser.add_argument('--date', help='Date for daily report (YYYY-MM-DD)')
    parser.add_argument('--week', help='Week for weekly report (YYYY-Wxx)')
    parser.add_argument('--month', help='Month for monthly report (YYYY-MM)')
    parser.add_argument('--output', help='Output PDF path (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.type == 'daily' and not args.date:
        parser.error("--date required for daily reports")
    elif args.type == 'weekly' and not args.week:
        parser.error("--week required for weekly reports")
    elif args.type == 'monthly' and not args.month:
        parser.error("--month required for monthly reports")
    
    # Get API keys
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        sys.exit(1)
    
    # Initialize components
    broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
    
    # Initialize data loaders
    market_loader = MarketDataLoader(broker)
    news_loader = NewsDataLoader()
    sentiment_analyzer = SentimentAnalyzer()
    regime_classifier = RegimeClassifier()
    feature_store = FeatureStore(market_loader, news_loader, sentiment_analyzer, regime_classifier)
    
    # Initialize learning engine
    strategy_names = [
        "TimeSeriesMomentum", "CrossSectionMomentum", "MeanReversion",
        "VolatilityRegimeVolTarget", "Carry", "ValueQualityTilt",
        "RiskParityMinVar", "TailRiskOverlay", "NewsSentimentEvent"
    ]
    learning_engine = LearningEngine(strategy_names=strategy_names, outputs_dir="outputs")
    
    # Initialize report engine
    report_engine = ReportEngine(outputs_dir="outputs")
    
    # Determine report date
    if args.type == 'daily':
        report_date = datetime.fromisoformat(args.date)
    elif args.type == 'weekly':
        # Parse week string (YYYY-Wxx)
        year, week = args.week.split('-W')
        # Approximate: first day of week
        report_date = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
    else:  # monthly
        report_date = datetime.strptime(args.month, "%Y-%m")
    
    print(f"Generating {args.type} report for {report_date.date()}...")
    
    # Collect data
    report_data = report_engine.collect_daily_data(
        broker=broker,
        learning_engine=learning_engine,
        feature_store=feature_store,
        end_date=report_date,
    )
    
    # Generate report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = None
    
    if args.type == 'daily':
        result = report_engine.generate_daily_report(report_data, output_path)
    elif args.type == 'weekly':
        result = report_engine.generate_weekly_report(report_data, output_path)
    else:
        result = report_engine.generate_monthly_report(report_data, output_path)
    
    print(f"✅ Report generated: {result['pdf_path']}")


if __name__ == '__main__':
    main()
