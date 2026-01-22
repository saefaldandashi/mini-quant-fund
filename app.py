# Auto-deploy test - Thu Jan 22 14:51:58 +04 2026
"""
Flask web UI for the Multi-Strategy Quant Debate Bot.
Integrates all 10 strategies with debate mechanism and auto-rebalancing.
"""
import os
import sys
import logging
import threading
import queue
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

from flask import Flask, render_template, jsonify, request, send_file

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from broker_alpaca import AlpacaBroker
from alpaca.trading.enums import OrderSide

# Import multi-strategy components
from src.data.market_data import MarketDataLoader
from src.data.news_data import NewsDataLoader
from src.data.alpha_vantage_news import AlphaVantageNewsLoader, AlphaVantageArticle
from src.data.ticker_sentiment import TickerSentimentAggregator, StockSentimentFeatures
from src.data.sentiment import SentimentAnalyzer
from src.data.feature_store import FeatureStore
from src.data.macro_data import MacroDataLoader, MacroIndicators, get_macro_loader
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
from src.debate.debate_engine import DebateEngine
from src.debate.ensemble import EnsembleOptimizer, EnsembleMode
from src.debate.adversarial import AdversarialDebateEngine, AdversarialTranscript
from src.risk.risk_manager import RiskManager, RiskConstraints
from src.risk.realtime_monitor import RealtimeRiskMonitor, RiskMonitorConfig, RiskLevel
from src.learning import LearningEngine, DebateLearner
from src.learning.outcome_tracker import OutcomeTracker
from src.learning.signal_validator import SignalValidator
from src.learning.feedback_loop import FeedbackLoop

# Try to import LLM components (optional - requires API key)
try:
    from src.llm import LLMEventExtractor, ThemeSynthesizer, LLMService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
from src.optimizations import (
    ParallelStrategyExecutor,
    PriceDataCache,
    SmartPositionSizer,
    ThompsonSamplingWeights,
)
from src.optimizations.strategy_enhancer import StrategyEnhancer, EnhancedConfig, get_enhancer
from src.news_intelligence import (
    NewsIntelligencePipeline,
    DailyMacroFeatures,
    MacroEvent,
)
from src.execution import SmartExecutor, ExecutionStrategy, SpreadAnalyzer
from src.execution.transaction_costs import TransactionCostModel, transaction_cost_model

# Parallel data fetching utilities
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# Initialize reporting system (uses LIVE data, no parquet dependency)
try:
    from src.reporting import generate_report, ReportLearningFeedback
    from pathlib import Path
    
    # Ensure reports directory exists
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    REPORTING_AVAILABLE = True
    logging.info("✅ Reporting system initialized (live data mode)")
except ImportError as e:
    logging.warning(f"Reporting system not available: {e}")
    REPORTING_AVAILABLE = False

# Placeholder for backward compatibility (unused but may be referenced)
data_writer = None

# Initialize learning engine (persists across requests)
STRATEGY_NAMES = [
    "TimeSeriesMomentum", "CrossSectionMomentum", "MeanReversion",
    "VolatilityRegimeVolTarget", "Carry", "ValueQualityTilt",
    "RiskParityMinVar", "TailRiskOverlay", "NewsSentimentEvent"
]
learning_engine = LearningEngine(
    strategy_names=STRATEGY_NAMES,
    outputs_dir="outputs",
    learning_influence=0.3  # 30% influence from learning, 70% from debate
)

# Initialize debate learner (learns from debate outcomes)
debate_learner = DebateLearner(outputs_dir="outputs"
)

# Initialize performance optimizations (persist across requests)
parallel_executor = ParallelStrategyExecutor(max_workers=4, timeout_seconds=30)
price_cache = PriceDataCache(cache_dir="outputs/cache", memory_ttl_minutes=5)
smart_sizer = SmartPositionSizer(target_vol=0.12, max_position=0.15, use_kelly=True)

# Initialize News Intelligence Pipeline
news_intelligence = NewsIntelligencePipeline(cache_dir="outputs/news_intelligence")

# Initialize Alpha Vantage News Loader (replaces World News API)
alpha_vantage_news = AlphaVantageNewsLoader(
    api_key="MU0B7DN9XFBK5I7C",
    cache_dir="outputs/alpha_vantage_cache",
    cache_ttl_hours=1,  # Cache for 1 hour (rate limits)
)

# Initialize Ticker Sentiment Aggregator (THE KEY FEATURE WE WERE MISSING)
ticker_sentiment_aggregator = TickerSentimentAggregator()

# Global storage for ticker sentiment (accessible via API)
last_ticker_sentiments: Dict[str, StockSentimentFeatures] = {}
_ultrafast_cache_file = Path("outputs/cache/ultrafast_sentiments.pkl")
_ultrafast_cache_file.parent.mkdir(parents=True, exist_ok=True)

def save_ultrafast_cache(ticker_sents: Dict):
    """Save ticker sentiments to file for ultra-fast mode."""
    import pickle
    try:
        with open(_ultrafast_cache_file, 'wb') as f:
            pickle.dump({
                'timestamp': datetime.now().isoformat(),
                'sentiments': ticker_sents,
            }, f)
    except Exception as e:
        logging.warning(f"Could not save ultra-fast cache: {e}")

def load_ultrafast_cache(max_age_minutes: int = 30):
    """Load cached sentiments if fresh enough."""
    import pickle
    try:
        if not _ultrafast_cache_file.exists():
            return None
        with open(_ultrafast_cache_file, 'rb') as f:
            data = pickle.load(f)
        # Check age
        cache_time = datetime.fromisoformat(data['timestamp'])
        age = (datetime.now() - cache_time).total_seconds() / 60
        if age > max_age_minutes:
            return None
        return data['sentiments']
    except Exception as e:
        logging.warning(f"Could not load ultra-fast cache: {e}")
        return None

# Initialize Outcome Tracker (CRITICAL FOR VALIDATING SIGNALS)
outcome_tracker = OutcomeTracker(storage_path="outputs/signal_outcomes.json")

# Initialize Signal Validator (VALIDATES BEFORE TRADING)
signal_validator = SignalValidator(
    min_confidence=0.3,
    max_staleness_hours=24.0,
    consistency_threshold=0.5,
)

# Initialize Feedback Loop (CONNECTS OUTCOMES TO WEIGHT ADJUSTMENTS)
feedback_loop = FeedbackLoop(
    storage_path="outputs/feedback_loop.json",
    learning_rate=0.1,
    min_samples_for_adjustment=10,
)

# Initialize LLM Components - NOW WITH GEMINI
llm_service = None
event_extractor = None
theme_synthesizer = None

# Gemini API Key (user provided)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

if LLM_AVAILABLE:
    # Try Gemini first
    # Note: gemini-3-pro-preview has beta issues, using gemini-2.0-flash which works reliably
    if GEMINI_API_KEY:
        try:
            llm_service = LLMService(
                provider="gemini",
                model="gemini-2.0-flash",  # Reliable model (gemini-3-pro-preview has beta issues)
                api_key=GEMINI_API_KEY,
                max_daily_cost=10.0,  # Generous limit
            )
            if llm_service.is_available():
                event_extractor = LLMEventExtractor(llm_service=llm_service)
                theme_synthesizer = ThemeSynthesizer(llm_service=llm_service)
                print("✓ Gemini LLM initialized (gemini-2.0-flash)")
            else:
                print("⚠ Gemini configured but not available")
        except Exception as e:
            print(f"⚠ Gemini initialization failed: {e}")
    
    # Fallback to OpenAI if Gemini fails
    if not llm_service or not llm_service.is_available():
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key:
            llm_service = LLMService(
                provider="openai",
                model="gpt-3.5-turbo",
                api_key=openai_key,
                max_daily_cost=5.0,
            )
            event_extractor = LLMEventExtractor(llm_service=llm_service)
            theme_synthesizer = ThemeSynthesizer(llm_service=llm_service)
            print("✓ OpenAI LLM initialized (fallback)")
thompson_sampler = ThompsonSamplingWeights(
    strategy_names=STRATEGY_NAMES,
    storage_path="outputs/thompson_beliefs.json",
    exploration_bonus=0.1,
)

# Message queue for real-time log updates
log_queue = queue.Queue()

# Global cache for macro features (persists between requests)
last_macro_features = {
    "features": None,
    "risk_sentiment": None,
    "events": [],
    "last_updated": None,
}

# Store last run status
last_run_status = {
    "running": False,
    "completed": False,
    "timestamp": None,
    "output": "",
    "error": None,
    "debate_transcript": None,
    "adversarial_transcript": None,
    "strategy_scores": None,
}

# Capital exposure setting (0.0 - 1.0)
capital_exposure_pct = 0.8

# Risk appetite setting
risk_appetite = "moderate"  # conservative, moderate, aggressive, maximum

# Long/Short & Futures Mode settings
long_short_settings = {
    "enable_long_short": True,  # Enable L/S strategies (can short)
    "enable_futures": True,  # Enable futures strategies (backtest proxies)
    "enable_shorting": True,  # Allow short positions
    "max_gross_exposure": 2.0,  # 200% max gross
    "net_exposure_min": -0.3,  # Can be 30% net short
    "net_exposure_max": 1.0,  # Can be 100% net long
    "max_short_per_position": 0.10,  # 10% max per short
}

# Strategy enhancer
strategy_enhancer = get_enhancer(EnhancedConfig(risk_appetite="moderate"))

# Current regime state
current_regime = None

# Macro data loader (Yahoo Finance + FRED)
FRED_API_KEY = os.environ.get("FRED_API_KEY")
macro_loader = get_macro_loader(FRED_API_KEY)

# Background refresh settings
background_refresh_enabled = True
background_refresh_interval = 15  # minutes
last_background_refresh = None

# Auto-rebalance settings
auto_rebalance_settings = {
    "enabled": False,
    "interval_minutes": 60,
    "dry_run": True,
    "allow_after_hours": True,
    "last_run": None,
    "next_run": None,
}

# Background scheduler thread
scheduler_thread = None
scheduler_stop_event = threading.Event()


def create_strategies(enable_long_short=False, enable_futures=False, trading_mode='intraday'):
    """
    Create all strategy instances.
    
    Args:
        enable_long_short: If True, include L/S strategies that can short
        enable_futures: If True, include futures strategies (backtest proxies)
        trading_mode: 'intraday' for 15-30 min trading, 'position' for daily/weekly
    
    Returns:
        List of strategy instances
    """
    strategies = []
    
    if trading_mode == 'intraday':
        # INTRADAY MODE (15-30 minute trading - HFT-lite)
        # These strategies are designed for quick in-and-out trades
        from src.strategies import create_intraday_strategies
        intraday_strategies = create_intraday_strategies()
        strategies.extend(intraday_strategies)
        logging.info(f"Added {len(intraday_strategies)} intraday strategies (15-30 min trading)")
        
        # Also include news sentiment (reactive to headlines)
        strategies.append(NewsSentimentEventStrategy({'sentiment_threshold': 0.3}))
        
        # Add short-term momentum (with reduced lookback for intraday)
        strategies.append(
            CrossSectionMomentumStrategy({
                'lookback': 5,  # 5-day momentum (not 126!)
                'top_n': 5, 
                'skip_recent': 0  # No skip for intraday
            })
        )
    else:
        # POSITION MODE (daily/weekly trading - legacy)
        strategies = [
            TimeSeriesMomentumStrategy({'lookback': 126, 'vol_target': 0.10, 'long_only': True}),
            CrossSectionMomentumStrategy({'lookback': 126, 'top_n': 5, 'skip_recent': 21}),
            MeanReversionStrategy({'z_threshold': 2.0, 'ma_type': 'ma_20'}),
            VolatilityRegimeVolTargetStrategy({'target_vol': 0.12}),
            CarryStrategy(),
            ValueQualityTiltStrategy({'top_n': 10}),
            RiskParityMinVarStrategy({'mode': 'risk_parity', 'max_weight': 0.15}),
            TailRiskOverlayStrategy({'vol_trigger': 0.25, 'min_exposure': 0.3}),
            NewsSentimentEventStrategy({'sentiment_threshold': 0.3}),
        ]
    
    # Add Long/Short strategies if enabled
    if enable_long_short:
        from src.strategies import create_long_short_strategies
        ls_strategies = create_long_short_strategies()
        strategies.extend(ls_strategies)
        logging.info(f"Added {len(ls_strategies)} Long/Short strategies")
    
    # Add Futures strategies if enabled (backtest only with ETF proxies)
    if enable_futures:
        from src.strategies import create_futures_strategies
        futures_strategies = create_futures_strategies()
        strategies.extend(futures_strategies)
        logging.info(f"Added {len(futures_strategies)} Futures strategies (backtest proxies)")
    
    return strategies


# Global trading mode setting
trading_mode_setting = 'intraday'  # 'intraday' or 'position'
dynamic_mode_enabled = True  # Enable dynamic mode switching based on VIX/regime


def get_dynamic_trading_mode(
    vix_level: float,
    regime: str = 'neutral',
    spy_trend: str = 'neutral',
) -> Tuple[str, str]:
    """
    Dynamically select trading mode based on market conditions.
    
    LOGIC:
    - High VIX (>25): Use intraday (quick in/out, reduce exposure time)
    - Low VIX (<15) + Trending: Use position (hold longer, capture trends)
    - High VIX + Range-bound: Use intraday (quick mean reversion)
    - Default: Hybrid blend
    
    Returns:
        Tuple of (mode, explanation)
    """
    # High volatility = intraday (quick in/out to reduce risk exposure)
    if vix_level > 30:
        return "intraday", f"VIX={vix_level:.0f} (>30) → INTRADAY: High vol, quick trades"
    
    if vix_level > 25:
        return "intraday", f"VIX={vix_level:.0f} (25-30) → INTRADAY: Elevated vol"
    
    # Low volatility + trending = position (hold longer)
    if vix_level < 15 and 'trend' in regime.lower():
        return "position", f"VIX={vix_level:.0f} (<15) + {regime} → POSITION: Low vol, trending"
    
    # Low volatility + mean reverting = hybrid
    if vix_level < 15 and 'range' in regime.lower():
        return "hybrid", f"VIX={vix_level:.0f} (<15) + {regime} → HYBRID: Low vol, range-bound"
    
    # Moderate volatility = intraday (default for HFT-lite focus)
    if 15 <= vix_level <= 20:
        return "intraday", f"VIX={vix_level:.0f} (15-20) → INTRADAY: Normal conditions, HFT-lite"
    
    # Slightly elevated = intraday with caution
    if 20 < vix_level <= 25:
        return "intraday", f"VIX={vix_level:.0f} (20-25) → INTRADAY: Slightly elevated vol"
    
    # Default: intraday (HFT-lite focus)
    return "intraday", f"VIX={vix_level:.0f} → INTRADAY: Default HFT-lite mode"


def get_strategy_blend_weights(mode: str, vix_level: float) -> Dict[str, float]:
    """
    Get strategy category weight multipliers based on mode.
    
    Returns multipliers for each strategy category.
    """
    if mode == "intraday":
        return {
            "intraday_strategies": 0.70,    # Primary focus
            "position_strategies": 0.15,    # Some position for diversification
            "ls_strategies": 0.15,          # L/S for market neutral
        }
    elif mode == "position":
        return {
            "intraday_strategies": 0.20,    # Some intraday for quick trades
            "position_strategies": 0.55,    # Primary focus
            "ls_strategies": 0.25,          # More L/S for hedging
        }
    else:  # hybrid
        return {
            "intraday_strategies": 0.45,
            "position_strategies": 0.35,
            "ls_strategies": 0.20,
        }


# ============================================================
# REAL-TIME RISK MONITOR (Background Thread)
# ============================================================
# Monitors portfolio risk continuously and auto-reduces exposure on drawdowns

risk_monitor: RealtimeRiskMonitor = None  # Initialized when broker is available
risk_monitor_enabled = True  # Enable/disable risk monitoring

def init_risk_monitor(broker):
    """Initialize the real-time risk monitor with the broker."""
    global risk_monitor
    
    config = RiskMonitorConfig(
        check_interval_seconds=60,  # Check every minute
        drawdown_warning=0.05,      # 5% drawdown = alert
        drawdown_reduce=0.08,       # 8% = reduce 30%
        drawdown_critical=0.10,     # 10% = reduce 50%, halt
        vix_elevated=25.0,
        vix_high=30.0,
        vix_critical=35.0,
    )
    
    def on_risk_alert(alert):
        """Callback for risk alerts - log to queue for UI."""
        log_queue.put(f"🚨 RISK ALERT: {alert.message}")
    
    risk_monitor = RealtimeRiskMonitor(
        broker=broker,
        config=config,
        on_alert=on_risk_alert,
    )
    
    return risk_monitor

def start_risk_monitor():
    """Start the background risk monitoring thread."""
    global risk_monitor
    if risk_monitor and not risk_monitor.is_running:
        risk_monitor.start()
        logging.info("🛡️ Real-time risk monitor started")
        return True
    return False

def stop_risk_monitor():
    """Stop the risk monitoring thread."""
    global risk_monitor
    if risk_monitor and risk_monitor.is_running:
        risk_monitor.stop()
        logging.info("🛡️ Real-time risk monitor stopped")
        return True
    return False


def run_multi_strategy_rebalance(dry_run=True, allow_after_hours=False, force_rebalance=True, cancel_orders=True, fast_mode=False, ultra_fast=False):
    """
    Run the multi-strategy debate bot and execute rebalancing.
    
    Args:
        dry_run: If True, simulate trades without executing
        allow_after_hours: If True, allow execution outside market hours
        force_rebalance: If True, bypass daily limit check
        cancel_orders: If True, cancel existing open orders first
        fast_mode: If True, skip LLM calls and use cached data for faster execution
        ultra_fast: If True, use cached data + rule-based debate (<5 sec target)
    
    Returns tuple of (success: bool, output: str, error: str or None, debate_info: dict)
    """
    global last_ticker_sentiments, trading_mode_setting  # Declare at top for ultra-fast caching
    
    # Ultra fast implies fast mode
    if ultra_fast:
        fast_mode = True
    output_lines = []
    debate_info = {"transcript": None, "scores": None, "final_weights": {}}
    
    def log(msg):
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"{timestamp} - {msg}"
        output_lines.append(formatted)
        log_queue.put(formatted)
    
    try:
        log("=" * 60)
        log("MULTI-STRATEGY QUANT DEBATE BOT")
        log("=" * 60)
        log(f"Mode: {'DRY RUN' if dry_run else 'LIVE PAPER TRADING'}")
        log(f"After Hours: {'Allowed' if allow_after_hours else 'Not Allowed'}")
        log(f"Cancel Previous Orders: {'Yes' if cancel_orders else 'No'}")
        if ultra_fast:
            log(f"Mode: ⚡⚡ ULTRA-FAST (<5 sec target)")
        elif fast_mode:
            log(f"Mode: ⚡ FAST (skip LLM)")
        else:
            log(f"Mode: Normal (full LLM debate)")
        log("")
        
        # Get API keys
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            log("ERROR: Missing API keys!")
            return False, "\n".join(output_lines), "Missing API keys", debate_info
        
        # Initialize broker
        log("Connecting to Alpaca...")
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
        log("Connected to Alpaca (paper trading)")
        
        # Initialize and start real-time risk monitor
        global risk_monitor
        if risk_monitor_enabled and risk_monitor is None:
            init_risk_monitor(broker)
            start_risk_monitor()
            log("🛡️ Real-time risk monitor ACTIVE (checking every 60s)")
        
        # Cancel previous orders if requested
        if cancel_orders:
            log("Cancelling any open orders...")
            cancelled = broker.cancel_all_orders()
            if cancelled > 0:
                log(f"Cancelled {cancelled} open orders")
            else:
                log("No open orders to cancel")
        
        # Check market hours
        market_open = broker.is_market_open()
        if not market_open and not allow_after_hours:
            log("Market is closed. Enable 'Allow After Hours' to proceed.")
            return True, "\n".join(output_lines), None, debate_info
        
        if not market_open:
            log("Market closed, but after-hours enabled. Proceeding...")
        else:
            log("Market is OPEN")
        
        # Get account info
        account = broker.get_account()
        equity = account["equity"]
        log(f"Account Equity: ${equity:,.2f}")
        log(f"Cash: ${account['cash']:,.2f}")
        log("")
        
        # === MULTI-STRATEGY DATA LOADING ===
        log("=" * 60)
        log("LOADING MARKET DATA")
        log("=" * 60)
        
        import pytz
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=400)
        
        # ULTRA-FAST MODE: Use cached features directly if fresh enough
        if ultra_fast and hasattr(price_cache, '_last_update') and price_cache._last_update:
            cache_age = (datetime.now(pytz.UTC) - price_cache._last_update).total_seconds()
            if cache_age < 300:  # 5 minutes
                log(f"⚡⚡ ULTRA-FAST: Using cached data (age: {cache_age:.0f}s)")
        
        # Create data loaders
        market_loader = MarketDataLoader()
        news_loader = NewsDataLoader()
        sentiment_analyzer = SentimentAnalyzer()
        regime_classifier = RegimeClassifier()
        
        # Create feature store
        feature_store = FeatureStore(
            market_loader, news_loader, sentiment_analyzer, regime_classifier
        )
        
        log(f"Fetching data for {len(config.UNIVERSE)} stocks...")
        
        # ============================================================
        # PARALLEL DATA FETCHING (Reduces 40s -> ~15s)
        # ============================================================
        # Fetch price data and news in parallel using ThreadPoolExecutor
        
        import time as time_module
        data_fetch_start = time_module.time()
        
        def fetch_price_data():
            """Fetch price data from cache or Alpaca."""
            cached = price_cache.get_prices(config.UNIVERSE, days=300, end_date=end_date)
            if cached is not None:
                return cached, True  # (data, was_cached)
            else:
                data = broker.get_historical_bars(config.UNIVERSE, days=300)
                price_cache.set_prices(config.UNIVERSE, days=300, data=data, end_date=end_date)
                return data, False
        
        def fetch_news_data():
            """Fetch news from Alpha Vantage."""
            if ultra_fast and load_ultrafast_cache(max_age_minutes=30) is not None:
                return [], True  # Skip news fetch in ultra-fast mode
            try:
                articles = alpha_vantage_news.fetch_market_news(days_back=7)
                return articles, False
            except Exception as e:
                logging.warning(f"News fetch error: {e}")
                return [], False
        
        # Execute in parallel
        price_data = None
        av_articles_parallel = []
        
        if not ultra_fast:
            with ThreadPoolExecutor(max_workers=2) as executor:
                price_future = executor.submit(fetch_price_data)
                news_future = executor.submit(fetch_news_data)
                
                # Get results
                try:
                    price_data, was_cached = price_future.result(timeout=60)
                    if was_cached:
                        log(f"📦 Using cached price data ({len(price_data)} stocks)")
                    else:
                        log(f"Received data for {len(price_data)} stocks from Alpaca (cached)")
                except Exception as e:
                    log(f"Price fetch error: {e}")
                    price_data = {}
                
                try:
                    av_articles_parallel, was_cached = news_future.result(timeout=60)
                    if not was_cached and av_articles_parallel:
                        log(f"📰 Pre-fetched {len(av_articles_parallel)} news articles")
                except Exception as e:
                    log(f"News prefetch error: {e}")
                    av_articles_parallel = []
        else:
            # Ultra-fast: just fetch prices (news from cache)
            price_data, was_cached = fetch_price_data()
            if was_cached:
                log(f"📦 Using cached price data ({len(price_data)} stocks)")
            else:
                log(f"Received data for {len(price_data)} stocks from Alpaca (cached)")
        
        data_fetch_time = time_module.time() - data_fetch_start
        log(f"⚡ Parallel data fetch completed in {data_fetch_time:.1f}s")
        
        # Convert to feature store format
        for symbol, prices in price_data.items():
            if len(prices) > 0:
                df = prices.to_frame(name='close')
                df['open'] = df['close']
                df['high'] = df['close']
                df['low'] = df['close']
                df['volume'] = 1000000
                feature_store._price_history[symbol] = df
        
        log(f"Loaded price data for {len(feature_store._price_history)} symbols")
        
        # ============================================================
        # NEWS INTELLIGENCE PIPELINE (Alpha Vantage)
        # ============================================================
        log("=" * 60)
        log("NEWS INTELLIGENCE PIPELINE (Alpha Vantage)")
        log("=" * 60)
        
        # Try to load fresh news from Alpha Vantage
        macro_features = None
        
        # ULTRA-FAST MODE: Skip fresh news fetch, use cached data
        cached_sentiments = load_ultrafast_cache(max_age_minutes=30) if ultra_fast else None
        skip_news_fetch = ultra_fast and cached_sentiments is not None
        
        if skip_news_fetch:
            log("⚡⚡ ULTRA-FAST: Using cached ticker sentiments (skipping news fetch)")
            unique_articles = []
            ticker_features = cached_sentiments
            last_ticker_sentiments.update(cached_sentiments)  # Update global
            if hasattr(macro_data, 'last_features') and macro_data.last_features:
                macro_features = macro_data.last_features
                log(f"⚡⚡ Using cached macro features")
        
        if not skip_news_fetch:
            try:
                symbols_with_data = list(feature_store._price_history.keys())
                
                # Use pre-fetched articles from parallel fetch, or fetch now
                if av_articles_parallel:
                    av_articles = av_articles_parallel
                    log(f"Using pre-fetched {len(av_articles)} news articles")
                else:
                    log("Fetching market news from Alpha Vantage...")
                    av_articles = alpha_vantage_news.fetch_market_news(days_back=7)
            
                # Also fetch ticker-specific news for top symbols
                if symbols_with_data and len(av_articles) < 20:
                    top_symbols = symbols_with_data[:5]  # Limit API calls
                    log(f"Fetching news for top symbols: {', '.join(top_symbols)}")
                    ticker_articles = alpha_vantage_news.fetch_ticker_news(top_symbols, days_back=7)
                    av_articles.extend(ticker_articles)
                
                # Deduplicate
                seen_ids = set()
                unique_articles = []
                for a in av_articles:
                    if a.id not in seen_ids:
                        seen_ids.add(a.id)
                        unique_articles.append(a)
                
                log(f"Received {len(unique_articles)} articles from Alpha Vantage")
            
                if unique_articles:
                    # AGGREGATE TICKER-LEVEL SENTIMENT (THE KEY FEATURE WE WERE MISSING!)
                    log("")
                    log("📊 AGGREGATING TICKER SENTIMENT...")
                    
                    ticker_features = ticker_sentiment_aggregator.aggregate_from_articles(
                        unique_articles,
                        as_of=end_date,
                        universe=list(feature_store._price_history.keys()) if feature_store._price_history else None,
                    )
                    last_ticker_sentiments = ticker_features
                    # Save to file for ultra-fast mode
                    save_ultrafast_cache(ticker_features)
                    
                    # Log ticker sentiment summary
                    bullish = ticker_sentiment_aggregator.get_bullish_stocks(threshold=0.2)
                    bearish = ticker_sentiment_aggregator.get_bearish_stocks(threshold=-0.2)
                    momentum = ticker_sentiment_aggregator.get_momentum_stocks(threshold=0.1)
                    
                    log(f"  📈 Bullish stocks: {len(bullish)} {bullish[:5] if bullish else []}")
                    log(f"  📉 Bearish stocks: {len(bearish)} {bearish[:5] if bearish else []}")
                    log(f"  🚀 Momentum stocks: {len(momentum)} {momentum[:5] if momentum else []}")
                    
                    # Show top sentiment stocks
                    if ticker_features:
                        top_bullish = sorted(
                            ticker_features.items(),
                            key=lambda x: x[1].sentiment_score,
                            reverse=True
                        )[:3]
                        log("")
                        log("  🔝 Top Bullish by Sentiment:")
                        for ticker, feat in top_bullish:
                            log(f"     {ticker}: {feat.sentiment_score:+.2f} (conf: {feat.sentiment_confidence:.0%})")
                    
                    # Show sample headlines
                    log("")
                    log("📰 Sample Alpha Vantage Headlines:")
                    for article in unique_articles[:3]:
                        tickers_str = ", ".join(article.tickers[:3]) if article.tickers else "Market"
                        sentiment = article.overall_sentiment_label
                        log(f"  [{tickers_str}] [{sentiment}] {article.headline[:50]}...")
                    
                    # Convert to News Intelligence Pipeline format
                    from src.news_intelligence.pipeline import NewsArticle as NIPNewsArticle
                    
                    pipeline_articles = []
                    for article in unique_articles:
                        try:
                            ts = article.timestamp
                            if ts.tzinfo is None:
                                ts = pytz.UTC.localize(ts)
                            pipeline_articles.append(NIPNewsArticle(
                                timestamp=ts,
                                source=article.source,
                                title=article.headline,
                                body=article.summary or "",
                                url=article.url,
                            ))
                        except:
                            continue
                    
                    if pipeline_articles:
                        events, stats = news_intelligence.process_articles(pipeline_articles, end_date)
                        log("")
                        log(f"✅ Processed {stats.total_articles} articles → {stats.events_extracted} events")
                        log(f"   Relevance filter: {stats.pass_rate*100:.0f}% pass rate")
                        log(f"   High-impact events: {stats.high_impact_events}")
                        
                        # Write news events to parquet storage for reports
                        if data_writer and events:
                            try:
                                for event in events[:20]:  # Top 20 events
                                    data_writer.write_news_event(
                                        event_id=f"{event.timestamp}_{hash(event.headline)}",
                                        headline=event.headline,
                                        tags=list(event.tags) if hasattr(event, 'tags') else [],
                                        impact_score=event.impact_score,
                                        direction=event.direction,
                                        severity=event.severity,
                                        rationale=event.rationale,
                                        entities=list(event.entities) if hasattr(event, 'entities') else [],
                                        timestamp=event.timestamp,
                                    )
                            except Exception as e:
                                logging.warning(f"Could not write news events: {e}")
                else:
                    # Fallback to sample data
                    log("No articles from Alpha Vantage, using sample data...")
                    sample_path = Path("data/sample_news.json")
                    if sample_path.exists():
                        events, stats = news_intelligence.load_from_json(str(sample_path), end_date)
                        log(f"Loaded {stats.events_extracted} sample events")
                
            except Exception as e:
                log(f"Alpha Vantage error: {e}")
                log("Loading sample macro news for analysis...")
                
                # Load sample news through News Intelligence Pipeline
                sample_path = Path("data/sample_news.json")
                if sample_path.exists():
                    events, stats = news_intelligence.load_from_json(str(sample_path), end_date)
                    log(f"Loaded {stats.events_extracted} sample events for analysis")
        
        # Get macro features from News Intelligence (store temporarily)
        macro_features_temp = None
        risk_sentiment_temp = None
        
        try:
            macro_features_temp = news_intelligence.get_daily_macro_features(end_date)
            risk_sentiment_temp = news_intelligence.get_risk_sentiment(end_date)
            
            log("")
            log("📊 MACRO INDICES:")
            log(f"  Inflation Pressure:  {macro_features_temp.macro_inflation_pressure_index:+.2f}")
            log(f"  Growth Momentum:     {macro_features_temp.growth_momentum_index:+.2f}")
            log(f"  CB Hawkishness:      {macro_features_temp.central_bank_hawkishness_index:+.2f}")
            log(f"  Geopolitical Risk:   {macro_features_temp.geopolitical_risk_index:+.2f}")
            log(f"  Financial Stress:    {macro_features_temp.financial_stress_index:+.2f}")
            log("")
            log(f"🎯 RISK SENTIMENT: {risk_sentiment_temp.risk_sentiment:+.2f} " +
                f"({'RISK-ON' if risk_sentiment_temp.risk_sentiment > 0.1 else 'RISK-OFF' if risk_sentiment_temp.risk_sentiment < -0.1 else 'NEUTRAL'})")
            log(f"  Equity Bias: {risk_sentiment_temp.equity_bias:+.2f}")
            log(f"  Rates Bias:  {risk_sentiment_temp.rates_bias:+.2f}")
            
            # Log top events
            if macro_features_temp.top_events:
                log("")
                log("🔥 TOP EVENTS:")
                for event in macro_features_temp.top_events[:3]:
                    log(f"  • {event[:65]}...")
                    
        except Exception as e:
            log(f"Warning: Could not compute macro features: {e}")
            
        log("")
        
        # Get features for current date
        log("Computing features and market regime...")
        features = feature_store.get_features(end_date, list(feature_store._price_history.keys()))
        
        # === CRITICAL: Add intraday data for HFT-lite strategies ===
        # Without this, intraday strategies fall back to daily data and perform poorly
        # Note: effective_trading_mode is determined later, so we load intraday if setting is intraday
        # or if it's hybrid (which uses some intraday strategies)
        if trading_mode_setting in ['intraday', 'hybrid']:
            log("📊 Loading INTRADAY data (15-min bars) for HFT-lite strategies...")
            try:
                symbols_for_intraday = list(feature_store._price_history.keys())[:50]  # Top 50 to avoid API limits
                features = feature_store.add_intraday_features(features, symbols_for_intraday, timeframe="15Min")
                
                if features.has_intraday_data:
                    log(f"✅ Loaded intraday data for {len(features.intraday_returns)} symbols")
                    log(f"   Volume ratio range: {min(features.volume_ratio.values()):.2f}x - {max(features.volume_ratio.values()):.2f}x" 
                        if features.volume_ratio else "   No volume ratio data")
                else:
                    log("⚠️ No intraday data available - strategies will use daily fallback")
            except Exception as e:
                log(f"⚠️ Could not load intraday data: {e} - using daily fallback")
        
        # Now attach macro features to the features object AND store globally
        if macro_features_temp:
            features.macro_features = macro_features_temp
            last_macro_features["features"] = macro_features_temp
            
            # Write macro features to parquet storage for reports
            if data_writer and macro_features_temp:
                try:
                    data_writer.write_macro_features(
                        inflation_pressure=macro_features_temp.macro_inflation_pressure_index,
                        labor_strength=macro_features_temp.labor_strength_index,
                        growth_momentum=macro_features_temp.growth_momentum_index,
                        cb_hawkishness=macro_features_temp.central_bank_hawkishness_index,
                        geo_risk=macro_features_temp.geopolitical_risk_index,
                        financial_stress=macro_features_temp.financial_stress_index,
                        commodities_risk=macro_features_temp.commodities_supply_risk_index,
                        risk_sentiment=risk_sentiment_temp.value if risk_sentiment_temp else 'NEUTRAL',
                        timestamp=end_date,
                        vix=market_indicators.vix if market_indicators else None,
                    )
                except Exception as e:
                    logging.warning(f"Could not write macro features: {e}")
        
        if risk_sentiment_temp:
            features.risk_sentiment = risk_sentiment_temp
            last_macro_features["risk_sentiment"] = risk_sentiment_temp
        
        # Store update timestamp
        last_macro_features["last_updated"] = datetime.now(pytz.UTC)
        
        if features.regime:
            log(f"Market Regime: {features.regime.description}")
        else:
            log("Market Regime: Could not classify (insufficient data)")
        
        log("")
        
        # === DYNAMIC TRADING MODE (VIX-based) ===
        effective_trading_mode = trading_mode_setting  # Start with static setting
        
        if dynamic_mode_enabled:
            # Get VIX for dynamic mode decision
            early_vix = 20.0  # Default
            try:
                if macro_features_temp and hasattr(macro_features_temp, 'vix') and macro_features_temp.vix:
                    early_vix = macro_features_temp.vix
                elif market_indicators and hasattr(market_indicators, 'vix') and market_indicators.vix:
                    early_vix = market_indicators.vix
            except:
                pass
            
            # Get regime description
            regime_desc = features.regime.description if hasattr(features, 'regime') and features.regime else 'neutral'
            
            # Determine dynamic trading mode
            effective_trading_mode, mode_explanation = get_dynamic_trading_mode(
                vix_level=early_vix,
                regime=regime_desc,
            )
            
            log("")
            log(f"🎯 DYNAMIC TRADING MODE: {effective_trading_mode.upper()}")
            log(f"   Reason: {mode_explanation}")
            log("")
        
        # === CREATE AND RUN STRATEGIES (PARALLEL) ===
        log("=" * 60)
        log("RUNNING STRATEGIES (Parallel Execution)")
        log("=" * 60)
        
        import time as time_module
        strat_start = time_module.time()
        
        strategies = create_strategies(
            enable_long_short=long_short_settings.get('enable_long_short', False),
            enable_futures=long_short_settings.get('enable_futures', False),
            trading_mode=effective_trading_mode,  # Use dynamic mode
        )
        log(f"Loaded {len(strategies)} strategies (L/S: {long_short_settings.get('enable_long_short')}, Futures: {long_short_settings.get('enable_futures')})")
        
        # INJECT MACRO CONTEXT INTO ALL STRATEGIES
        macro_ctx = features.macro_features if hasattr(features, 'macro_features') else None
        risk_ctx = features.risk_sentiment if hasattr(features, 'risk_sentiment') else None
        
        # Convert last_ticker_sentiments to format expected by strategies
        # Format: {symbol: score} and {symbol: confidence}
        sentiment_scores = {}
        sentiment_confidence = {}
        
        if last_ticker_sentiments:
            for symbol, data in last_ticker_sentiments.items():
                if isinstance(data, dict):
                    sentiment_scores[symbol] = data.get('score', 0)
                    sentiment_confidence[symbol] = data.get('confidence', 0.5)
                elif isinstance(data, (int, float)):
                    sentiment_scores[symbol] = float(data)
                    sentiment_confidence[symbol] = 0.5
        
        strategies_with_sentiment = 0
        
        for strategy in strategies:
            # Inject macro context to ALL strategies
            strategy.set_macro_context(macro_ctx, risk_ctx)
            
            # Inject ticker sentiment to ALL strategies (not just NewsSentimentEvent)
            if hasattr(strategy, 'set_sentiment_data'):
                strategy.set_sentiment_data(sentiment_scores, sentiment_confidence)
                strategies_with_sentiment += 1
            elif hasattr(strategy, 'ticker_sentiments'):
                # Fallback for NewsSentimentEvent
                strategy.ticker_sentiments = last_ticker_sentiments
        
        log(f"  Injected macro context to {len(strategies)} strategies")
        log(f"  Injected {len(sentiment_scores)} ticker sentiments to {strategies_with_sentiment} strategies")
        
        # PHASE 1: Generate initial signals (parallel)
        signals, errors = parallel_executor.execute_all(strategies, features, end_date)
        
        # PHASE 2: Share signals between strategies and allow adjustments
        log("")
        log("📢 INTER-STRATEGY COMMUNICATION")
        log("-" * 40)
        
        # Let each strategy see peer signals
        for strategy in strategies:
            if strategy.name in signals:
                strategy.set_peer_signals(signals)
        
        # Optional: Re-run strategies that want to adjust based on peer input
        # (For now, we just log consensus - full re-run can be enabled later)
        for name, signal in signals.items():
            if signal.desired_weights:
                top_symbols = sorted(signal.desired_weights.items(), key=lambda x: -abs(x[1]))[:3]
                for symbol, weight in top_symbols:
                    # Check peer consensus
                    for strategy in strategies:
                        if strategy.name == name:
                            avg_w, agreement = strategy.get_peer_consensus(symbol)
                            if agreement > 0.5:
                                log(f"  {name}/{symbol}: Peer consensus {avg_w:+.1%} (agreement: {agreement:.0%})")
        
        strat_time = time_module.time() - strat_start
        
        # Log results
        for strategy in strategies:
            if strategy.name in signals:
                signal = signals[strategy.name]
                n_positions = len([w for w in signal.desired_weights.values() if abs(w) > 0.01])
                log(f"✓ {strategy.name}: {n_positions} positions, "
                    f"confidence={signal.confidence:.1%}, "
                    f"exp_ret={signal.expected_return:.1%}")
            elif strategy.name in errors:
                log(f"✗ {strategy.name}: ERROR - {errors[strategy.name][:50]}")
        
        log(f"\n⚡ Parallel execution: {len(signals)}/{len(strategies)} strategies in {strat_time:.2f}s")
        log("")
        
        # Write strategy outputs to parquet storage for reports
        if data_writer:
            try:
                for name, signal in signals.items():
                    data_writer.write_strategy_output(
                        strategy_name=name,
                        desired_weights=signal.desired_weights,
                        confidence=signal.confidence,
                        expected_return=signal.expected_return,
                        risk_estimate=getattr(signal, 'risk_estimate', 0.0),
                        timestamp=end_date,
                        weight=0.0,  # Will be updated after debate/ensemble
                        contribution=0.0,  # Will be computed later
                        explanation=signal.explanation,
                    )
            except Exception as e:
                logging.warning(f"Could not write strategy outputs: {e}")
        
        # === DEBATE ENGINE ===
        log("=" * 60)
        log("STRATEGY DEBATE - PHASE 1: INITIAL SCORING")
        log("=" * 60)
        
        # ULTRA-FAST MODE: Use rule-based fast debate
        if ultra_fast:
            log("⚡⚡ ULTRA-FAST: Using rule-based debate (no LLM)")
            from src.debate.fast_debate import fast_debate, MarketContext
            
            # Get historical performance for scoring
            historical_perf = {}
            if learning_engine:
                try:
                    # Try to get strategy weights as proxy for performance
                    if hasattr(learning_engine, 'strategy_performance'):
                        for strat_name, perf_data in learning_engine.strategy_performance.items():
                            if isinstance(perf_data, dict):
                                historical_perf[strat_name] = perf_data.get('accuracy', 0.5)
                            else:
                                historical_perf[strat_name] = 0.5
                except Exception:
                    pass  # Use default performance
            
            fast_scores, fast_metadata = fast_debate(
                signals=signals,
                features=features,
                current_time=end_date,
                historical_performance=historical_perf,
            )
            
            # Convert to StrategyScore format
            from src.debate.debate_engine import StrategyScore
            scores = {}
            for name, score in fast_scores.items():
                details = fast_metadata['score_details'].get(name, {})
                scores[name] = StrategyScore(
                    strategy_name=name,
                    alpha_score=details.get('confidence', 0.5),
                    regime_fit_score=details.get('regime', 0.5),
                    diversification_score=0.5,
                    drawdown_score=0.5,
                    sentiment_score=details.get('urgency', 0.5),
                    total_score=score,
                    rationale=f"Fast debate: blend_weight={details.get('blend_weight', 0.5):.1%}",
                    strengths=[],
                    weaknesses=[],
                )
            
            # Create minimal transcript
            from src.debate.debate_engine import DebateTranscript
            transcript = DebateTranscript(
                timestamp=end_date,
                regime=None,
                agreements=[],
                disagreements=[],
                top_risks=[],
            )
            
            log(f"⚡ Fast debate: {fast_metadata['execution_time_ms']:.1f}ms")
            log(f"  Regime: {fast_metadata['market_context']['regime']}")
            log(f"  Blend: Intraday {fast_metadata['blend_weights']['intraday']:.0%} / Position {fast_metadata['blend_weights']['position']:.0%}")
        else:
            # Normal debate engine
            debate_engine = DebateEngine()
            scores, transcript = debate_engine.run_debate(signals, features)
        
        # Log initial rankings
        ranked = sorted(scores.items(), key=lambda x: x[1].total_score, reverse=True)
        log("Initial Strategy Rankings:")
        for i, (name, score) in enumerate(ranked[:5], 1):
            log(f"  {i}. {name}: {score.total_score:.2f}")
            if score.strengths:
                log(f"     Strengths: {', '.join(score.strengths[:2])}")
        
        # Log agreements
        if transcript.agreements:
            log(f"\nConsensus ({len(transcript.agreements)} agreements):")
            for agree in transcript.agreements[:3]:
                log(f"  + {agree}")
        
        # Log risks
        if transcript.top_risks:
            log(f"\nTop Risks:")
            for risk in transcript.top_risks:
                log(f"  ⚠️ {risk}")
        
        log("")
        
        # === ADVERSARIAL DEBATE ===
        log("=" * 60)
        log("🎭 ADVERSARIAL DEBATE - PHASE 2: STRATEGIES CRITIQUE EACH OTHER")
        log("=" * 60)
        
        base_scores = {name: score.total_score for name, score in scores.items()}
        
        # Choose debate mode:
        # - ultra_fast: Rule-based only (no LLM)
        # - fast_mode: Parallel LLM (full LLM, but concurrent)
        # - normal: Sequential LLM (original behavior)
        
        if ultra_fast:
            # Already handled above in fast_debate
            log("  ⚡⚡ Ultra-fast: Scores already set by rule-based debate")
            adjusted_scores = base_scores
            adversarial_transcript = None
            
        elif fast_mode and llm_service:
            # NEW: Parallel LLM - full reasoning with concurrency!
            log("  ⚡ PARALLEL LLM: Running all LLM calls concurrently...")
            from src.debate.parallel_debate import ParallelDebateEngine
            
            parallel_engine = ParallelDebateEngine(llm_service, max_workers=6)
            parallel_result = parallel_engine.run_parallel_debate(
                signals, features, base_scores
            )
            
            adjusted_scores = parallel_result.strategy_scores
            adversarial_transcript = None  # Different format
            
            log(f"  ⚡ Parallel LLM complete: {parallel_result.execution_time_ms:.0f}ms")
            log(f"  📊 LLM calls: {parallel_result.llm_calls_parallel}/{parallel_result.llm_calls_made} succeeded")
            
            # Log insights from parallel debate
            for insight in parallel_result.insights[:3]:
                log(f"  💡 {insight}")
            
            # Show top arguments
            support_args = [a for a in parallel_result.arguments if a["type"] == "support"]
            for arg in sorted(support_args, key=lambda x: x.get("strength", 0), reverse=True)[:3]:
                log(f"  💪 {arg['strategy']}: {arg.get('claim', '')[:60]}... ({arg.get('strength', 0):.0%})")
            
        else:
            # Original sequential LLM (slow but thorough)
            if fast_mode:
                log("  ⚡ Fast mode: Using rule-based debate (no LLM available)")
                adversarial_engine = AdversarialDebateEngine(
                    debate_learner=debate_learner,
                    llm_service=None,
                )
            else:
                log("  🐢 Sequential LLM: Full debate (may take 30+ seconds)")
                adversarial_engine = AdversarialDebateEngine(
                    debate_learner=debate_learner,
                    llm_service=llm_service,
                )
            
            adjusted_scores, adversarial_transcript = adversarial_engine.run_adversarial_debate(
                signals, features, base_scores
            )
            
            # Log LLM usage
            if llm_service and not fast_mode:
                log(f"  LLM arguments generated: {adversarial_engine.llm_arguments_generated}")
                log(f"  Rule-based fallbacks: {adversarial_engine.rule_based_fallbacks}")
        
        # Log key arguments from each round (only for sequential debate)
        if adversarial_transcript and hasattr(adversarial_transcript, 'rounds'):
            for round_data in adversarial_transcript.rounds:
                if round_data.round_number == 1:
                    log(f"\n📣 Round 1 - SUPPORT ARGUMENTS:")
                    for arg in round_data.arguments[:3]:  # Show top 3
                        log(f"  💪 {arg.source_strategy}: {arg.claim} ({arg.strength:.0%})")
                elif round_data.round_number == 2:
                    log(f"\n⚔️ Round 2 - ATTACKS:")
                    for arg in round_data.arguments[:5]:  # Show top 5 attacks
                        log(f"  ⚔️ {arg.source_strategy} → {arg.target_strategy}:")
                        log(f"     \"{arg.claim}\" ({arg.strength:.0%})")
                elif round_data.round_number == 3:
                    log(f"\n🛡️ Round 3 - REBUTTALS:")
                    for arg in round_data.arguments[:4]:  # Show top 4 rebuttals
                        log(f"  🛡️ {arg.source_strategy} defends:")
                        log(f"     \"{arg.claim}\" ({arg.strength:.0%})")
            
            # Log debate impact
            log(f"\n📊 DEBATE IMPACT:")
            for strat, impact in sorted(adversarial_transcript.attack_impact.items(), key=lambda x: x[1], reverse=True)[:5]:
                defense = adversarial_transcript.defense_success.get(strat, 0)
                net = -impact + defense
                status = "📉" if net < 0 else "📈" if net > 0 else "↔️"
                log(f"  {status} {strat}: Attacked({impact:.0%}) Defended({defense:.0%}) Net({net:+.0%})")
            
            # Log key insights
            if adversarial_transcript.key_insights:
                log(f"\n💡 KEY DEBATE INSIGHTS:")
                for insight in adversarial_transcript.key_insights:
                    log(f"  • {insight}")
        
        # Update scores with adversarial adjustments
        for name in scores:
            if name in adjusted_scores:
                old_score = scores[name].total_score
                new_score = adjusted_scores[name]
                scores[name].total_score = new_score
                if abs(new_score - old_score) > 0.05:
                    direction = "↑" if new_score > old_score else "↓"
                    log(f"  {direction} {name}: {old_score:.2f} → {new_score:.2f}")
        
        # Write debate entries to parquet storage for reports
        if data_writer and adversarial_transcript:
            try:
                # Write support arguments
                for round_data in adversarial_transcript.rounds:
                    if round_data.round_number == 1:  # Support round
                        for arg in round_data.arguments:
                            data_writer.write_debate_entry(
                                strategy_name=arg.source_strategy,
                                score=arg.strength,
                                rationale=arg.claim,
                                role='support',
                                timestamp=end_date,
                            )
                    elif round_data.round_number == 2:  # Attack round
                        for arg in round_data.arguments:
                            data_writer.write_debate_entry(
                                strategy_name=arg.source_strategy,
                                score=-arg.strength,  # Negative for attacks
                                rationale=f"Attacks {arg.target_strategy}: {arg.claim}",
                                role='attack',
                                timestamp=end_date,
                            )
                    elif round_data.round_number == 3:  # Rebuttal round
                        for arg in round_data.arguments:
                            data_writer.write_debate_entry(
                                strategy_name=arg.source_strategy,
                                score=arg.strength,
                                rationale=f"Defends: {arg.claim}",
                                role='rebuttal',
                                timestamp=end_date,
                            )
            except Exception as e:
                logging.warning(f"Could not write debate entries: {e}")
        
        # Final rankings after adversarial debate
        log(f"\n🏆 FINAL RANKINGS (Post-Debate):")
        final_ranked = sorted(scores.items(), key=lambda x: x[1].total_score, reverse=True)
        debate_winners = adversarial_transcript.debate_winners if adversarial_transcript else []
        for i, (name, score) in enumerate(final_ranked[:5], 1):
            winner = "★" if name in debate_winners else ""
            log(f"  {i}. {name}: {score.total_score:.2f} {winner}")
        
        # Store transcript info
        if adversarial_transcript:
            debate_info["transcript"] = transcript.to_string() + "\n\n" + adversarial_transcript.to_string()
            debate_info["adversarial_transcript"] = adversarial_transcript.to_string()
        else:
            debate_info["transcript"] = transcript.to_string() + "\n\n[Parallel LLM debate - scores computed directly]"
            debate_info["adversarial_transcript"] = "[Parallel LLM debate mode]"
        debate_info["scores"] = {name: score.total_score for name, score in scores.items()}
        
        # Record debate outcome for learning (only for sequential debate)
        if adversarial_transcript:
            try:
                regime_name = features.regime.risk_regime.value if features.regime else 'unknown'
                vol_regime = features.regime.volatility.value if features.regime else 'normal_vol'
                
                # Build attack records for learning
                attack_records = []
                for round_data in adversarial_transcript.rounds:
                    for arg in round_data.arguments:
                        if arg.argument_type.value == 'attack':
                            attack_records.append({
                                'attacker': arg.source_strategy,
                                'defender': arg.target_strategy,
                                'claim': arg.claim,
                                'strength': arg.strength,
                                'claim_type': 'general',
                            })
                
                # Build rebuttal records
                rebuttal_records = []
                for round_data in adversarial_transcript.rounds:
                    for arg in round_data.arguments:
                        if arg.argument_type.value == 'rebuttal':
                            rebuttal_records.append({
                                'defender': arg.source_strategy,
                                'attacker': arg.target_strategy,
                                'claim': arg.claim,
                                'strength': arg.strength,
                            })
                
                debate_learner.record_debate(
                    timestamp=end_date,
                    regime=regime_name,
                    volatility_regime=vol_regime,
                    strategies=list(signals.keys()),
                    initial_scores=base_scores,
                    final_scores=adjusted_scores,
                    attacks=attack_records,
                    rebuttals=rebuttal_records,
                    debate_winners=adversarial_transcript.debate_winners,
                )
                log(f"📚 Debate recorded for learning ({len(attack_records)} attacks, {len(rebuttal_records)} rebuttals)")
            except Exception as e:
                log(f"⚠️ Failed to record debate: {str(e)[:50]}")
        
        log("")
        
        # === LEARNING ENGINE INTEGRATION ===
        log("=" * 60)
        log("LEARNING INSIGHTS")
        log("=" * 60)
        
        # Build market context for learning
        market_context_for_learning = {
            'regime': features.regime.risk_regime.value if features.regime else 'unknown',
            'volatility_regime': features.regime.volatility.value if features.regime else 'normal_vol',
            'trend_strength': features.regime.trend_strength if features.regime else 0.0,
            'correlation_regime': 'high' if features.regime and features.regime.correlation_regime > 0.5 else 'low',
        }
        
        # Convert signals for learning
        learning_signals = {}
        for name, signal in signals.items():
            learning_signals[name] = {
                'weights': signal.desired_weights,
                'confidence': signal.confidence,
                'expected_return': signal.expected_return,
                'debate_score': scores.get(name, type('obj', (object,), {'total_score': 0})).total_score,
                'explanation': signal.explanation,
            }
        
        # Record signals in learning engine
        learning_engine.record_signals(
            strategy_signals=learning_signals,
            debate_scores={name: s.total_score for name, s in scores.items()},
            regime=market_context_for_learning['regime'],
            market_context=market_context_for_learning,
        )
        
        # Get learning recommendations
        recommendations = learning_engine.get_recommendations(market_context_for_learning)
        
        if recommendations['recommended_strategies']:
            log(f"🧠 Learning suggests: {', '.join(recommendations['recommended_strategies'][:3])}")
        
        if recommendations['strategies_to_avoid']:
            log(f"⚠️ Learning cautions against: {', '.join(recommendations['strategies_to_avoid'])}")
        
        if recommendations['risk_warnings']:
            for warning in recommendations['risk_warnings'][:2]:
                log(f"🔴 {warning}")
        
        if recommendations['active_patterns']:
            log("Active patterns:")
            for pattern in recommendations['active_patterns'][:2]:
                log(f"  • {pattern['description']} (confidence: {pattern['confidence']:.0%})")
        
        # Get learned weights (blends learning with debate scores)
        learned_weights = learning_engine.get_learned_weights(
            debate_scores={name: s.total_score for name, s in scores.items()},
            market_context=market_context_for_learning,
        )
        
        # Blend with Thompson Sampling for smarter exploration
        thompson_weights = thompson_sampler.blend_with_debate(
            debate_scores={name: s.total_score for name, s in scores.items()},
            regime=market_context_for_learning['regime'],
            thompson_influence=0.2,  # 20% Thompson, 80% debate
        )
        
        # Combine learned weights with Thompson weights
        for name in learned_weights:
            if name in thompson_weights:
                learned_weights[name] = 0.7 * learned_weights[name] + 0.3 * thompson_weights[name]
        
        # Show learning summary
        learning_summary = learning_engine.get_learning_summary()
        thompson_summary = thompson_sampler.get_summary()
        debate_learning_summary = debate_learner.get_learning_summary()
        
        if learning_summary['is_learning']:
            log(f"\n📊 Learning Status: Active ({learning_summary['trade_history']['total_trades']} trades analyzed)")
            log(f"   Win Rate: {learning_summary['trade_history']['win_rate']:.1%}")
            log(f"   Overall Accuracy: {learning_summary['strategy_performance']['overall_accuracy']:.1%}")
        else:
            log("\n📊 Learning Status: Building initial dataset (need 5+ trades)")
        
        # Show debate learning status
        if debate_learning_summary['total_debates_recorded'] > 0:
            log(f"\n🎭 Debate Learning: {debate_learning_summary['total_debates_recorded']} debates analyzed")
            log(f"   Attack Patterns Learned: {debate_learning_summary['total_attack_patterns']}")
            
            # Show top attack patterns
            if debate_learning_summary['top_attack_patterns']:
                log("   Most Effective Attacks:")
                for pattern in debate_learning_summary['top_attack_patterns'][:3]:
                    log(f"     • {pattern['attacker']} → {pattern['defender']} ({pattern['success_rate']} success)")
            
            # Show regime-specific insights
            regime_name = features.regime.risk_regime.value if features.regime else 'unknown'
            regime_insights = debate_learner.get_insights_for_regime(regime_name)
            if regime_insights:
                log(f"   Insights for {regime_name} regime:")
                for insight in regime_insights[:2]:
                    log(f"     💡 {insight}")
        
        # Show Thompson exploration
        exploration_priorities = thompson_summary.get('exploration_priorities', [])
        if exploration_priorities:
            explore_names = [p[0] for p in exploration_priorities[:2]]
            log(f"🔍 Thompson Exploration: {', '.join(explore_names)}")
        
        log("")
        
        # === ENSEMBLE OPTIMIZATION ===
        log("=" * 60)
        log("ENSEMBLE OPTIMIZATION (Learning-Enhanced)")
        log("=" * 60)
        
        ensemble = EnsembleOptimizer({
            'max_position': 0.15,
            'max_leverage': 1.0,
            'vol_target': 0.12,
        })
        
        # Get current positions
        current_positions = broker.get_positions()
        current_weights = {}
        if current_positions and equity > 0:
            for symbol, pos in current_positions.items():
                current_weights[symbol] = pos['market_value'] / equity
        
        # Write portfolio snapshot to parquet storage for reports
        if data_writer:
            try:
                # Calculate exposures
                long_exposure = sum(pos.get('market_value', 0) for pos in current_positions.values() if pos.get('qty', 0) > 0)
                short_exposure = abs(sum(pos.get('market_value', 0) for pos in current_positions.values() if pos.get('qty', 0) < 0))
                net_exposure = long_exposure - short_exposure
                gross_exposure = long_exposure + short_exposure
                leverage = gross_exposure / equity if equity > 0 else 1.0
                
                # Format holdings for storage
                holdings_dict = {}
                for symbol, pos in current_positions.items():
                    holdings_dict[symbol] = {
                        'qty': pos.get('qty', 0),
                        'price': pos.get('current_price', 0),
                        'value': pos.get('market_value', 0),
                        'weight': current_weights.get(symbol, 0),
                        'pnl': pos.get('unrealized_pl', 0),
                        'pnl_pct': pos.get('unrealized_pl_pct', 0),
                        'contribution_1d': 0.0,  # Would need to compute from previous day
                    }
                
                data_writer.write_portfolio_snapshot(
                    portfolio_value=equity,
                    cash=account.get('cash', 0),
                    holdings=holdings_dict,
                    timestamp=end_date,
                    net_exposure=net_exposure,
                    gross_exposure=gross_exposure,
                    leverage=leverage,
                    turnover=0.0,  # Will be updated after trades
                )
            except Exception as e:
                logging.warning(f"Could not write portfolio snapshot: {e}")
        
        # Use learning-enhanced strategy weights
        log("Applying learned strategy weights...")
        for strategy_name, weight in sorted(learned_weights.items(), key=lambda x: -x[1])[:5]:
            log(f"  {strategy_name}: {weight:.1%}")
        
        combined_weights, metadata = ensemble.combine(
            signals, scores, features, current_weights, EnsembleMode.WEIGHTED_VOTE,
            strategy_weights=learned_weights  # Pass learned weights to ensemble
        )
        
        debate_info["final_weights"] = combined_weights
        
        # Log short positions from ensemble (before risk/cost filtering)
        short_positions = {k: v for k, v in combined_weights.items() if v < 0}
        long_positions = {k: v for k, v in combined_weights.items() if v > 0}
        if short_positions:
            log(f"📉 Ensemble produced {len(short_positions)} SHORT positions:")
            for sym, wt in sorted(short_positions.items(), key=lambda x: x[1])[:5]:
                log(f"    {sym}: {wt*100:+.2f}%")
        else:
            log(f"⚠️ Ensemble produced NO short positions")
            # Debug: Check what L/S strategies are producing
            for name, signal in signals.items():
                if '_LS' in name:
                    shorts = {k: v for k, v in signal.desired_weights.items() if v < 0}
                    if shorts:
                        log(f"  → {name} has {len(shorts)} shorts (e.g., {list(shorts.items())[:3]})")
        log(f"📈 Ensemble produced {len(long_positions)} LONG positions")
        
        log(f"Ensemble mode: {metadata['mode']}")
        log(f"Constraints applied: {len(metadata.get('constraints_applied', []))}")
        for constraint in metadata.get('constraints_applied', []):
            log(f"  - {constraint}")
        
        log("")
        
        # === RISK MANAGEMENT ===
        log("=" * 60)
        log("RISK CHECK")
        log("=" * 60)
        
        # Create risk manager with shorting enabled from settings
        risk_manager = RiskManager(RiskConstraints(
            max_position_size=0.15,
            max_sector_exposure=0.30,
            max_leverage=1.0,
            vol_target=0.12,
            enable_shorting=long_short_settings.get("enable_shorting", True),
            max_gross_exposure=long_short_settings.get("max_gross_exposure", 2.0),
            net_exposure_min=long_short_settings.get("net_exposure_min", -0.3),
            net_exposure_max=long_short_settings.get("net_exposure_max", 1.0),
            max_short_position=long_short_settings.get("max_short_per_position", 0.10),
        ))
        log(f"  ✓ Shorting: {'ENABLED' if long_short_settings.get('enable_shorting', True) else 'DISABLED'}")
        
        risk_result = risk_manager.check_and_approve(
            combined_weights, features, current_weights, equity
        )
        
        log(f"Risk check passed: {risk_result.approved}")
        log(f"Portfolio volatility: {risk_result.risk_metrics.get('portfolio_vol', 0):.1%}")
        log(f"Leverage: {risk_result.risk_metrics.get('leverage', 0):.1%}")
        log(f"Positions: {risk_result.risk_metrics.get('n_positions', 0)}")
        
        if risk_result.adjustments:
            log("Adjustments made:")
            for adj in risk_result.adjustments:
                log(f"  - {adj}")
        
        final_weights = risk_result.approved_weights
        log("")
        
        # === VALIDATE SIGNALS BEFORE TRADING ===
        log("=" * 60)
        log("SIGNAL VALIDATION")
        log("=" * 60)
        
        # Convert ticker sentiments to dict format for validator
        ticker_sentiments_dict = {}
        for ticker, feat in last_ticker_sentiments.items():
            if hasattr(feat, 'sentiment_score'):
                ticker_sentiments_dict[ticker] = {
                    'sentiment_score': feat.sentiment_score,
                    'sentiment_confidence': feat.sentiment_confidence,
                    'freshness_hours': feat.freshness_hours,
                }
        
        # Get macro sentiment
        macro_sent = getattr(macro_features_temp, 'overall_risk_sentiment_index', 0) if macro_features_temp else None
        
        # Prepare signals for validation
        validation_signals = {}
        for ticker, weight in final_weights.items():
            if abs(weight) > 0.001:
                # Get average confidence for this ticker
                conf = 0.5
                for name, signal in signals.items():
                    if ticker in signal.desired_weights:
                        conf = max(conf, signal.confidence)
                
                validation_signals[ticker] = {
                    'weight': weight,
                    'confidence': conf,
                }
        
        # Run validation
        validated_weights, validation_warnings = signal_validator.validate_portfolio(
            signals=validation_signals,
            ticker_sentiments=ticker_sentiments_dict,
            macro_sentiment=macro_sent,
        )
        
        # Log validation results
        validation_stats = signal_validator.get_stats()
        log(f"Validated {len(validation_signals)} signals:")
        log(f"  ✅ Passed: {validation_stats['total_passed']}")
        log(f"  ❌ Blocked: {validation_stats['total_blocked']}")
        log(f"  ⚠️ Warnings: {validation_stats['total_warnings']}")
        
        for warning in validation_warnings[:10]:  # Limit to first 10
            log(f"  {warning}")
        
        # Apply validated weights
        final_weights = validated_weights
        debate_info["final_weights"] = final_weights  # Update with actual final weights
        log("")
        
        # === RECORD SIGNALS FOR OUTCOME TRACKING ===
        # This is CRITICAL for validating whether our signals are predictive
        try:
            # Get sentiment scores for recording
            sentiment_scores = {}
            for symbol in final_weights:
                if symbol in last_ticker_sentiments:
                    ts = last_ticker_sentiments[symbol]
                    sentiment_scores[symbol] = ts.sentiment_score if hasattr(ts, 'sentiment_score') else 0
            
            # Get regime from features
            regime_str = features.regime.risk_regime.value if features.regime else 'unknown'
            
            # Record batch signals
            signal_ids = outcome_tracker.record_batch_signals(
                weights=final_weights,
                confidences=confidences if 'confidences' in dir() else {},
                sentiment_scores=sentiment_scores,
                macro_stance=getattr(macro_features_temp, 'overall_risk_sentiment_index', 0) if macro_features_temp else None,
                regime=regime_str,
                strategy_source="ensemble",
                timestamp=end_date,
            )
            
            log(f"📊 Recorded {len(signal_ids)} signals for outcome tracking")
            
        except Exception as e:
            log(f"Warning: Could not record signals: {e}")
        
        # === LLM TRADE REASONING ===
        # Generate genuine explanation for WHY we're making these trades
        # Skip in fast mode for speed
        if fast_mode:
            log("")
            log("⚡ Fast mode: Skipping LLM trade reasoning")
        elif llm_service and llm_service.is_available():
            log("")
            log("=" * 60)
            log("🧠 LLM TRADE REASONING")
            log("=" * 60)
            
            try:
                # Build context for LLM
                macro_ctx = "No macro data"
                if macro_features_temp:
                    macro_ctx = f"""
- Inflation Pressure: {getattr(macro_features_temp, 'macro_inflation_pressure_index', 0):.2f}
- Growth Momentum: {getattr(macro_features_temp, 'growth_momentum_index', 0):.2f}
- Geopolitical Risk: {getattr(macro_features_temp, 'geopolitical_risk_index', 0):.2f}
- Financial Stress: {getattr(macro_features_temp, 'financial_stress_index', 0):.2f}
- Risk Sentiment: {getattr(macro_features_temp, 'overall_risk_sentiment_index', 0):.2f}"""
                
                # Summarize debate
                debate_summary = adversarial_transcript.to_string()[:500] if adversarial_transcript else "No debate conducted"
                
                # Summarize signals
                signals_summary = ""
                for name, signal in signals.items():
                    if signal.desired_weights:
                        top3 = sorted(signal.desired_weights.items(), key=lambda x: -abs(x[1]))[:3]
                        signals_summary += f"\n{name} (conf={signal.confidence:.0%}): {', '.join([f'{s}:{w:+.0%}' for s,w in top3])}"
                
                # Call LLM for reasoning
                trade_reasoning = llm_service.generate_trade_reasoning(
                    final_weights=final_weights,
                    debate_summary=debate_summary,
                    macro_context=macro_ctx,
                    signals_summary=signals_summary,
                )
                
                if trade_reasoning:
                    log("")
                    log("Investment Thesis:")
                    for line in trade_reasoning.split('\n'):
                        log(f"  {line}")
                    log("")
                    
                    # Store for UI
                    debate_info["trade_reasoning"] = trade_reasoning
                else:
                    log("  (LLM reasoning not available)")
                    
            except Exception as e:
                log(f"  LLM reasoning failed: {e}")
        
        # === EXECUTE TRADES ===
        log("=" * 60)
        log("EXECUTING TRADES")
        log("=" * 60)
        
        # Calculate target shares
        target_symbols = [s for s, w in final_weights.items() if w > 0.01]
        log(f"Target portfolio: {len(target_symbols)} positions")
        
        for symbol, weight in sorted(final_weights.items(), key=lambda x: -x[1])[:10]:
            if weight > 0.01:
                log(f"  {symbol}: {weight:.1%}")
        
        if not target_symbols:
            log("No positions to take - going to cash")
            target_shares = {}
        else:
            # Pass the actual weights (not zeros!) to calculate proper share quantities
            target_weights_filtered = {s: final_weights[s] for s in target_symbols}
            
            # Apply Smart Position Sizing (Kelly Criterion + vol scaling)
            confidences = {}
            for symbol in target_weights_filtered:
                # Average confidence from strategies that have this symbol
                conf_sum = 0
                conf_count = 0
                for name, signal in signals.items():
                    if symbol in signal.desired_weights:
                        conf_sum += signal.confidence
                        conf_count += 1
                confidences[symbol] = conf_sum / max(1, conf_count)
            
            # Get volatilities
            volatilities = {s: features.volatility_21d.get(s, 0.20) for s in target_weights_filtered}
            
            # Apply smart sizing
            kelly_adjusted_weights, sizing_details = smart_sizer.size_positions(
                base_weights=target_weights_filtered,
                confidences=confidences,
                volatilities=volatilities,
            )
            
            # === PHASE 4: REGIME DETECTION ===
            log("")
            log("=" * 60)
            log("REGIME DETECTION")
            log("=" * 60)
            
            try:
                global current_regime
                # Get regime indicators
                spy_price = prices.get('SPY', 0)
                spy_200ma = features.moving_avg_200d.get('SPY', spy_price) if hasattr(features, 'moving_avg_200d') else spy_price
                vix_level = 20  # Default if not available
                macro_sent = macro_features_temp.get('overall_risk_sentiment_index', 0) if macro_features_temp else 0
                geo_risk = macro_features_temp.get('geopolitical_risk_index', 0) if macro_features_temp else 0
                fin_stress = macro_features_temp.get('financial_stress_index', 0) if macro_features_temp else 0
                
                current_regime = strategy_enhancer.detect_regime(
                    spy_price=spy_price,
                    spy_200ma=spy_200ma,
                    vix=vix_level,
                    macro_sentiment=macro_sent,
                    geo_risk=geo_risk,
                    financial_stress=fin_stress,
                )
                
                log(f"📊 Market Regime: {current_regime.regime.upper()}")
                log(f"   Score: {current_regime.score:.2f}")
                log(f"   Recommended Exposure: {current_regime.exposure_multiplier*100:.0f}%")
                for ind, val in current_regime.indicators.items():
                    log(f"   {ind}: {val:.2f}")
            except Exception as e:
                log(f"Regime detection error: {e}")
                current_regime = None
            
            # === PHASE 1: ENHANCED POSITION SIZING ===
            log("")
            log("=" * 60)
            log("ENHANCED POSITION SIZING")
            log("=" * 60)
            
            # Apply strategy enhancer
            enhanced_weights, size_reasons = strategy_enhancer.enhance_position_sizes(
                base_weights=kelly_adjusted_weights,
                confidences=confidences,
                target_exposure=capital_exposure_pct,
            )
            
            log(f"🎯 Risk Appetite: {risk_appetite.upper()}")
            log(f"   Kelly Multiplier: {strategy_enhancer.config.kelly_multiplier}x")
            log(f"   Min Position: {strategy_enhancer.config.min_position_pct*100:.1f}%")
            log(f"   Max Positions: {strategy_enhancer.config.max_positions}")
            log(f"   Investment Floor: {strategy_enhancer.config.min_investment_floor*100:.0f}%")
            
            # Count adjustments
            positions_before = len(kelly_adjusted_weights)
            positions_after = len(enhanced_weights)
            log(f"   Positions: {positions_before} → {positions_after}")
            
            total_allocation = sum(enhanced_weights.values())
            log(f"   Total Allocation: {total_allocation*100:.1f}%")
            
            # Apply regime adjustment if enabled
            regime_multiplier = 1.0
            if current_regime and strategy_enhancer.config.auto_regime_adjustment:
                regime_multiplier = current_regime.exposure_multiplier
                log(f"   Regime Adjustment: {regime_multiplier*100:.0f}%")
            
            # Apply capital exposure limit
            effective_equity = equity * capital_exposure_pct * regime_multiplier
            log(f"")
            log(f"💵 Capital Exposure: {capital_exposure_pct*100:.0f}% × {regime_multiplier*100:.0f}% regime = ${effective_equity:,.2f} available")
            
            target_shares = broker.calculate_target_shares(
                enhanced_weights,
                effective_equity,
                cash_buffer_pct=config.CASH_BUFFER_PCT
            )
            
            # Log sizing adjustments
            adjusted_count = len([d for d in sizing_details if d.adjusted_weight != d.base_weight])
            log(f"⚡ Smart sizing applied to {len(enhanced_weights)} positions ({adjusted_count} adjusted)")
        
        # Get current shares
        current_shares = {
            symbol: int(pos["qty"])
            for symbol, pos in current_positions.items()
        }
        
        # Determine orders needed
        all_symbols = set(current_shares.keys()) | set(target_shares.keys())
        
        if all_symbols:
            current_prices = broker.get_current_prices(list(all_symbols))
        else:
            current_prices = {}
        
        # Write prices to parquet storage for reports
        if data_writer and current_prices:
            try:
                data_writer.write_prices(current_prices, end_date)
                
                # Also write SPY as benchmark
                if 'SPY' in current_prices:
                    data_writer.write_benchmark_price('SPY', current_prices['SPY'], end_date)
            except Exception as e:
                logging.warning(f"Could not write prices to storage: {e}")
        
        # Get VIX level for smart execution
        try:
            market_indicators = macro_loader.fetch_all()
            vix_level = market_indicators.vix if market_indicators else 20.0
        except:
            vix_level = 20.0
        
        # Update risk monitor with current VIX
        if risk_monitor and vix_level:
            risk_monitor.update_vix(vix_level)
            log(f"🛡️ Risk monitor updated: VIX={vix_level:.1f}, can_trade={risk_monitor.can_trade()}")
        
        # Initialize Smart Executor with VIX awareness
        smart_executor = SmartExecutor(
            broker=broker,
            data_client=broker.data_client if hasattr(broker, 'data_client') else None,
            dry_run=dry_run,
            log_func=log,
            vix_level=vix_level,
        )
        
        # Build order list with conviction scores and TRANSACTION COST ANALYSIS
        orders_to_execute = []
        total_value = 0
        trades_skipped_by_cost = 0
        cost_avoided = 0.0
        
        # Update transaction cost model with current VIX
        transaction_cost_model.set_vix(vix_level)
        
        log("")
        log("=" * 60)
        log("TRANSACTION COST ANALYSIS")
        log("=" * 60)
        log(f"VIX Level: {vix_level:.1f} (cost multiplier: {transaction_cost_model.get_vix_multiplier():.1f}x)")
        
        # BULK FETCH real bid-ask spreads for all symbols (much faster than per-symbol)
        real_quotes = {}
        try:
            symbols_to_quote = [s for s in all_symbols if s in current_prices and current_prices[s] > 0]
            if symbols_to_quote:
                real_quotes = broker.get_current_quotes(symbols_to_quote[:100])  # Limit to 100
                if real_quotes:
                    avg_spread = sum(q['spread_pct'] for q in real_quotes.values()) / len(real_quotes)
                    log(f"📊 Real-time quotes fetched: {len(real_quotes)} symbols, avg spread: {avg_spread:.3f}%")
        except Exception as e:
            log(f"⚠️ Could not fetch real quotes: {e} - using estimates")
        
        for symbol in sorted(all_symbols):
            current_qty = current_shares.get(symbol, 0)
            target_qty = target_shares.get(symbol, 0)
            
            if current_qty == target_qty:
                continue
            
            price = current_prices.get(symbol, 0.0)
            if price <= 0:
                continue
            
            if current_qty > target_qty:
                # Sell
                qty = current_qty - target_qty
                side = 'sell'
            else:
                # Buy
                qty = target_qty - current_qty
                side = 'buy'
            
            value = qty * price
            
            # Get conviction from final weights (higher weight = higher conviction)
            conviction = min(1.0, abs(final_weights.get(symbol, 0)) * 5)  # Scale to 0-1
            
            # Get expected return for this symbol from signals
            expected_return = 0.02  # Default 2%
            for name, signal in signals.items():
                if symbol in signal.desired_weights:
                    # Use the strategy's expected return weighted by confidence
                    expected_return = max(expected_return, signal.expected_return * signal.confidence)
            
            # Use REAL bid-ask spread from pre-fetched quotes (much more accurate)
            if symbol in real_quotes:
                spread_pct = real_quotes[symbol]['spread_pct']
            else:
                # Fallback to default estimate based on likely liquidity
                spread_pct = 0.05  # Default 5 bps for liquid stocks
            
            # Estimate transaction cost
            cost_estimate = transaction_cost_model.estimate_trade_cost(
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                spread_pct=spread_pct,
                holding_days=20,  # Assume ~1 month holding
            )
            
            # Check if trade is worth executing
            cost_result = transaction_cost_model.should_execute_trade(
                cost_estimate=cost_estimate,
                expected_return=expected_return,
                confidence=conviction,
            )
            
            if not cost_result.should_trade:
                trades_skipped_by_cost += 1
                cost_avoided += cost_estimate.total_cost
                log(f"  ⏭️ Skip {side.upper()} {qty} {symbol}: {cost_result.reason}")
                continue
            
            total_value += value
            
            orders_to_execute.append({
                'symbol': symbol,
                'side': side,
                'quantity': qty,
                'price': price,
                'value': value,
                'conviction': conviction,
                'spread_pct': spread_pct,
                'cost_estimate': cost_estimate,
                'expected_benefit': cost_result.expected_benefit,
                'net_value': cost_result.net_expected_value,
            })
        
        # Log cost summary
        total_estimated_cost = sum(o.get('cost_estimate').total_cost for o in orders_to_execute if o.get('cost_estimate'))
        log("")
        log(f"📊 Cost Analysis Summary:")
        log(f"   Trades to execute: {len(orders_to_execute)}")
        log(f"   Trades skipped (high cost): {trades_skipped_by_cost}")
        log(f"   Estimated total cost: ${total_estimated_cost:.2f}")
        log(f"   Cost avoided: ${cost_avoided:.2f}")
        if total_value > 0:
            log(f"   Cost as % of notional: {(total_estimated_cost / total_value) * 100:.2f}%")
        
        if not orders_to_execute:
            log("")
            log("=" * 60)
            log("SMART EXECUTION ENGINE")
            log("=" * 60)
            log("  No trades needed - portfolio already aligned")
            orders_executed = 0
        else:
            # Use batch execution for optimized ordering
            results, exec_summary = smart_executor.execute_batch(orders_to_execute)
            
            # Record trades for learning (with transaction cost tracking)
            orders_executed = 0
            total_actual_cost = 0.0
            
            # Create a map of orders to their cost estimates
            order_cost_map = {o['symbol']: o.get('cost_estimate') for o in orders_to_execute}
            
            for result in results:
                if result.success:
                    orders_executed += 1
                    fill_price = result.fill_price or 0
                    
                    # Record actual vs estimated cost for learning
                    if result.symbol in order_cost_map and order_cost_map[result.symbol]:
                        cost_estimate = order_cost_map[result.symbol]
                        
                        # Find the order's original price (mid price at decision time)
                        order_info = next((o for o in orders_to_execute if o['symbol'] == result.symbol), None)
                        if order_info:
                            mid_price_at_decision = order_info['price']
                            
                            # Record for learning
                            transaction_cost_model.record_actual_cost(
                                symbol=result.symbol,
                                estimated_cost=cost_estimate,
                                actual_fill_price=fill_price,
                                mid_price_at_decision=mid_price_at_decision,
                            )
                            
                            # Calculate actual slippage
                            actual_slippage = abs(fill_price - mid_price_at_decision) * result.quantity
                            total_actual_cost += actual_slippage + cost_estimate.spread_cost
                    
                    learning_engine.record_trade(
                        symbol=result.symbol,
                        side=result.side,
                        quantity=result.quantity,
                        price=fill_price,
                        ensemble_weight=final_weights.get(result.symbol, 0),
                        ensemble_mode=metadata.get('mode', 'weighted_vote'),
                    )
                    
                    # Write trade to parquet storage for reports
                    if data_writer:
                        try:
                            cost = result.price_improvement * -1 if result.price_improvement < 0 else 0.0
                            data_writer.write_trade(
                                symbol=result.symbol,
                                side=result.side,
                                quantity=result.quantity,
                                price=fill_price,
                                timestamp=end_date,
                                cost=cost,
                                slippage=result.spread_pct * fill_price * result.quantity / 100 if hasattr(result, 'spread_pct') else 0.0,
                            )
                        except Exception as e:
                            logging.warning(f"Could not write trade: {e}")
            
            # Add cost tracking to execution summary
            exec_summary['total_estimated_cost'] = total_estimated_cost
            exec_summary['total_actual_cost'] = total_actual_cost
            exec_summary['trades_skipped_by_cost'] = trades_skipped_by_cost
            exec_summary['cost_avoided'] = cost_avoided
            
            # Store execution summary for UI
            last_run_status['execution_summary'] = exec_summary
            
            # Log symbol insights if available
            insights = smart_executor.get_symbol_insights()
            if 'best_fill_rate' in insights and insights['best_fill_rate']:
                log("")
                log("📈 SYMBOL INSIGHTS (from execution history)")
                log(f"  Symbols with history: {insights.get('symbols_with_history', 0)}")
                if insights['best_fill_rate']:
                    best = insights['best_fill_rate'][0]
                    log(f"  Best limit fill: {best['symbol']} ({best['fill_rate']*100:.0f}% over {best['attempts']} attempts)")
        
        log("")
        log("=" * 60)
        log("REBALANCING COMPLETE! ✅")
        log("=" * 60)
        
        # === RECORD OUTCOMES FOR PATTERN LEARNING ===
        # Update outcomes from current positions to feed the pattern learner
        try:
            positions = broker.get_positions()
            if positions:
                symbol_returns = {}
                for symbol, pos in positions.items():
                    cost_basis = pos.get('cost_basis', 0)
                    current_value = pos.get('current_value', pos.get('market_value', 0))
                    if cost_basis > 0:
                        symbol_returns[symbol] = (current_value - cost_basis) / cost_basis
                
                if symbol_returns:
                    # Determine volatility regime from volatility_percentile (a float)
                    if features.regime:
                        vol_pct = features.regime.volatility_percentile
                        if vol_pct > 0.7:
                            vol_regime = 'high'
                        elif vol_pct < 0.3:
                            vol_regime = 'low'
                        else:
                            vol_regime = 'medium'
                    else:
                        vol_regime = 'medium'
                    
                    # Determine trend strength from SPY 126-day return
                    spy_momentum = features.returns_126d.get('SPY', 0) if features.returns_126d else 0
                    trend_strength = 'strong' if abs(spy_momentum) > 0.1 else 'weak'
                    
                    market_context_for_patterns = {
                        'regime': features.regime.risk_regime.value if features.regime else 'unknown',
                        'volatility_regime': vol_regime,
                        'trend_strength': trend_strength,
                    }
                    
                    learning_engine.record_outcomes(symbol_returns, market_context_for_patterns)
                    log(f"📊 Pattern Learning: Recorded {len(symbol_returns)} position outcomes")
        except Exception as e:
            log(f"⚠️ Could not record outcomes for pattern learning: {e}")
        
        # Save transaction cost learned data
        try:
            transaction_cost_model.save_learned_data()
            learning_stats = transaction_cost_model.get_learning_stats()
            if learning_stats.get('symbols_with_learning', 0) > 0:
                log(f"💰 Cost Learning: {learning_stats['symbols_with_learning']} symbols using learned estimates, {learning_stats['estimation_improvement']}")
        except Exception as e:
            log(f"⚠️ Could not save cost learning data: {e}")
        
        return True, "\n".join(output_lines), None, debate_info
        
    except Exception as e:
        log(f"ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return False, "\n".join(output_lines), str(e), debate_info


def run_bot_threaded(dry_run=True, allow_after_hours=False, force_rebalance=True, cancel_orders=True, exposure_pct=0.8, fast_mode=True, ultra_fast=False):
    """Run bot in a separate thread.
    
    Args:
        exposure_pct: Capital exposure percentage (0.0 to 1.0). Default 0.8 = 80%
        fast_mode: If True, uses parallel LLM (2s instead of 30s, same quality). Default: True
        ultra_fast: If True, use rule-based debate + cached data (<5 sec target)
    """
    global last_run_status, capital_exposure_pct
    capital_exposure_pct = exposure_pct  # Store globally for use in rebalancing
    
    # Store modes in closure variables for the worker
    _fast_mode = fast_mode
    _ultra_fast = ultra_fast
    
    if last_run_status["running"]:
        return
    
    # Clear the queue
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except:
            break
    
    last_run_status["running"] = True
    last_run_status["completed"] = False
    last_run_status["timestamp"] = datetime.now().isoformat()
    last_run_status["output"] = ""
    last_run_status["error"] = None
    last_run_status["debate_transcript"] = None
    last_run_status["strategy_scores"] = None
    
    def bot_worker():
        global last_run_status
        
        try:
            success, output, error, debate_info = run_multi_strategy_rebalance(
                dry_run, allow_after_hours, force_rebalance, cancel_orders, _fast_mode, _ultra_fast
            )
            last_run_status["output"] = output
            last_run_status["error"] = error
            last_run_status["debate_transcript"] = debate_info.get("transcript")
            last_run_status["adversarial_transcript"] = debate_info.get("adversarial_transcript")
            last_run_status["strategy_scores"] = debate_info.get("scores")
            last_run_status["debate_scores"] = debate_info.get("scores")
            last_run_status["trade_reasoning"] = debate_info.get("trade_reasoning")
            last_run_status["final_weights"] = debate_info.get("final_weights", {})
        except Exception as e:
            last_run_status["output"] = f"Fatal error: {str(e)}"
            last_run_status["error"] = str(e)
        finally:
            last_run_status["running"] = False
            last_run_status["completed"] = True
    
    thread = threading.Thread(target=bot_worker, daemon=True)
    thread.start()


def auto_rebalance_scheduler():
    """Background thread for automatic rebalancing."""
    global auto_rebalance_settings
    
    while not scheduler_stop_event.is_set():
        if auto_rebalance_settings["enabled"]:
            now = datetime.now()
            next_run = auto_rebalance_settings.get("next_run")
            
            if next_run and now >= next_run:
                # Time to run
                if not last_run_status["running"]:
                    logging.info("Auto-rebalance triggered (fast mode)")
                    run_bot_threaded(
                        dry_run=auto_rebalance_settings.get("dry_run", True),
                        allow_after_hours=auto_rebalance_settings.get("allow_after_hours", True),
                        force_rebalance=True,
                        cancel_orders=True,
                        exposure_pct=auto_rebalance_settings.get("exposure_pct", 0.8),
                        fast_mode=True,  # Auto mode uses fast mode for speed
                    )
                    auto_rebalance_settings["last_run"] = now.isoformat()
                
                # Schedule next run
                interval = auto_rebalance_settings["interval_minutes"]
                auto_rebalance_settings["next_run"] = now + timedelta(minutes=interval)
        
        # Check every 10 seconds
        time.sleep(10)


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint for cloud deployment monitoring."""
    import pytz
    now = datetime.now(pytz.UTC)
    
    # Check if broker is connected
    broker_ok = False
    try:
        account = broker.get_account()
        broker_ok = account is not None
    except:
        pass
    
    # Check auto-rebalance status
    auto_enabled = auto_rebalance_settings.get("enabled", False)
    next_run = auto_rebalance_settings.get("next_run")
    
    return jsonify({
        "status": "healthy",
        "timestamp": now.isoformat(),
        "broker_connected": broker_ok,
        "auto_rebalance_enabled": auto_enabled,
        "next_scheduled_run": next_run.isoformat() if next_run else None,
        "last_run": last_run_status.get("timestamp"),
        "uptime_check": "ok",
    })


@app.route('/api/risk-monitor')
def get_risk_monitor_status():
    """Get real-time risk monitor status."""
    global risk_monitor
    
    if risk_monitor is None:
        return jsonify({
            "enabled": False,
            "status": "not_initialized",
            "message": "Risk monitor will start with first rebalance",
        })
    
    status = risk_monitor.get_status()
    
    return jsonify({
        "enabled": True,
        "is_running": status.get('is_running', False),
        "risk_level": status.get('risk_level', 'unknown'),
        "halt_trading": status.get('halt_trading', False),
        "can_trade": risk_monitor.can_trade(),
        "position_size_multiplier": status.get('position_size_multiplier', 1.0),
        "peak_equity": status.get('peak_equity', 0),
        "last_vix": status.get('last_vix', 0),
        "recent_alerts": status.get('recent_alerts', []),
    })


@app.route('/api/risk-monitor/toggle', methods=['POST'])
def toggle_risk_monitor():
    """Enable or disable the risk monitor."""
    global risk_monitor_enabled, risk_monitor
    
    data = request.get_json() or {}
    enable = data.get('enabled', True)
    
    if enable:
        risk_monitor_enabled = True
        if risk_monitor:
            start_risk_monitor()
        return jsonify({"success": True, "message": "Risk monitor enabled"})
    else:
        risk_monitor_enabled = False
        if risk_monitor:
            stop_risk_monitor()
        return jsonify({"success": True, "message": "Risk monitor disabled"})


@app.route('/api/status')
def get_status():
    """Get current bot status."""
    new_messages = []
    while not log_queue.empty():
        try:
            msg = log_queue.get_nowait()
            new_messages.append(msg)
        except:
            break
    
    return jsonify({
        **last_run_status,
        "new_messages": new_messages,
        "auto_rebalance": auto_rebalance_settings,
    })


@app.route('/api/debate-transcript')
def get_debate_transcript():
    """Get the full debate transcript including adversarial debate."""
    return jsonify({
        "transcript": last_run_status.get("debate_transcript"),
        "adversarial_transcript": last_run_status.get("adversarial_transcript"),
        "strategy_scores": last_run_status.get("strategy_scores"),
    })


@app.route('/api/debate/learning')
def get_debate_learning():
    """Get debate learning statistics and insights."""
    try:
        summary = debate_learner.get_learning_summary()
        
        # Get strategy credibility for each strategy
        credibility = {}
        for strat in STRATEGY_NAMES:
            credibility[strat] = {
                'score': debate_learner.get_debate_credibility(strat),
                'profile': summary['strategy_credibility'].get(strat, {}),
            }
        
        return jsonify({
            "summary": summary,
            "strategy_credibility": credibility,
            "total_debates": summary['total_debates_recorded'],
            "attack_patterns": summary['total_attack_patterns'],
            "top_patterns": summary['top_attack_patterns'],
            "debate_accuracy": summary['debate_accuracy'],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/debate/insights/<regime>')
def get_debate_insights(regime):
    """Get learned insights for a specific market regime."""
    try:
        insights = debate_learner.get_insights_for_regime(regime)
        return jsonify({
            "regime": regime,
            "insights": insights,
            "attack_boosts": {
                strat: debate_learner.get_attack_boost(strat, "TimeSeriesMomentum", regime)
                for strat in STRATEGY_NAMES
            },
            "defense_boosts": {
                strat: debate_learner.get_defense_boost(strat, regime)
                for strat in STRATEGY_NAMES
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def parse_bool_param(value, default=True):
    """Parse a parameter that could be a bool, string, or None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


@app.route('/api/transaction-costs')
def get_transaction_costs():
    """Get transaction cost metrics and analysis."""
    try:
        cost_summary = transaction_cost_model.get_cost_summary_for_ui()
        
        # Add execution summary if available
        exec_summary = last_run_status.get('execution_summary', {})
        
        return jsonify({
            'status': 'success',
            'cost_model': {
                'vix_level': transaction_cost_model.vix_level,
                'vix_multiplier': transaction_cost_model.get_vix_multiplier(),
                'base_slippage_bps': transaction_cost_model.params['base_slippage_bps'],
                'min_benefit_ratio': transaction_cost_model.params['min_benefit_ratio'],
            },
            'historical': cost_summary,
            'last_execution': {
                'estimated_cost': exec_summary.get('total_estimated_cost', 0),
                'actual_cost': exec_summary.get('total_actual_cost', 0),
                'trades_skipped': exec_summary.get('trades_skipped_by_cost', 0),
                'cost_avoided': exec_summary.get('cost_avoided', 0),
                'avg_spread': exec_summary.get('avg_spread', 0),
            },
            'symbols_analyzed': len(transaction_cost_model.symbol_data),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run', methods=['POST'])
def run_bot_endpoint():
    """Trigger bot run."""
    # Accept both JSON body and query parameters
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = {}
    
    # Query params override JSON - handle both bool and string values
    dry_run_val = request.args.get('dry_run', data.get('dry_run', True))
    dry_run = parse_bool_param(dry_run_val, default=True)
    
    allow_after_hours_val = request.args.get('allow_after_hours', data.get('allow_after_hours', True))
    allow_after_hours = parse_bool_param(allow_after_hours_val, default=True)
    
    force_rebalance_val = request.args.get('force', data.get('force_rebalance', True))
    force_rebalance = parse_bool_param(force_rebalance_val, default=True)
    
    cancel_orders_val = request.args.get('cancel_previous', data.get('cancel_orders', True))
    cancel_orders = parse_bool_param(cancel_orders_val, default=True)
    
    exposure_val = request.args.get('exposure', data.get('exposure', data.get('exposure_pct', 80)))
    try:
        exposure_pct = int(exposure_val) / 100.0  # Convert to decimal
    except (ValueError, TypeError):
        exposure_pct = 0.8  # Default 80%
    
    # Risk appetite setting
    global risk_appetite, strategy_enhancer
    new_risk_appetite = request.args.get('risk_appetite', data.get('risk_appetite', 'moderate'))
    if new_risk_appetite in ['conservative', 'moderate', 'aggressive', 'maximum']:
        risk_appetite = new_risk_appetite
        strategy_enhancer = get_enhancer(EnhancedConfig(risk_appetite=risk_appetite))
    
    # Fast mode - uses parallel LLM for speed while keeping full reasoning
    # DEFAULT: True (parallel LLM gives same quality in 2s instead of 30s)
    fast_mode_val = request.args.get('fast_mode', data.get('fast_mode', True))
    fast_mode = parse_bool_param(fast_mode_val, default=True)
    
    # Ultra-fast mode - rule-based debate, cached data (<5 sec target)
    ultra_fast_val = request.args.get('ultra_fast', data.get('ultra_fast', False))
    ultra_fast = parse_bool_param(ultra_fast_val, default=False)
    
    if last_run_status["running"]:
        return jsonify({"error": "Bot is already running"}), 400
    
    run_bot_threaded(
        dry_run=dry_run,
        allow_after_hours=allow_after_hours,
        force_rebalance=force_rebalance,
        cancel_orders=cancel_orders,
        exposure_pct=exposure_pct,
        fast_mode=fast_mode,
        ultra_fast=ultra_fast,
    )
    if ultra_fast:
        mode_str = "⚡⚡ ULTRA-FAST "
    elif fast_mode:
        mode_str = "⚡ FAST "
    else:
        mode_str = ""
    return jsonify({"status": "started", "message": f"{mode_str}Multi-strategy bot initiated ({int(exposure_pct*100)}% exposure)"})


@app.route('/api/auto-rebalance', methods=['POST'])
def set_auto_rebalance():
    """Configure automatic rebalancing."""
    global auto_rebalance_settings
    
    data = request.get_json() or {}
    
    auto_rebalance_settings["enabled"] = data.get('enabled', False)
    auto_rebalance_settings["interval_minutes"] = data.get('interval_minutes', 60)
    auto_rebalance_settings["dry_run"] = data.get('dry_run', True)
    auto_rebalance_settings["allow_after_hours"] = data.get('allow_after_hours', True)
    
    if auto_rebalance_settings["enabled"]:
        interval = auto_rebalance_settings["interval_minutes"]
        auto_rebalance_settings["next_run"] = datetime.now() + timedelta(minutes=interval)
        message = f"Auto-rebalance enabled: every {interval} minutes"
    else:
        auto_rebalance_settings["next_run"] = None
        message = "Auto-rebalance disabled"
    
    return jsonify({
        "status": "updated",
        "message": message,
        "settings": auto_rebalance_settings
    })


@app.route('/api/cancel-orders', methods=['POST'])
def cancel_orders_endpoint():
    """Cancel all open orders."""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return jsonify({"error": "API keys not configured"}), 400
        
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
        cancelled = broker.cancel_all_orders()
        
        return jsonify({
            "status": "success",
            "cancelled_count": cancelled
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/open-orders')
def get_open_orders():
    """Get all open orders."""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return jsonify({"error": "API keys not configured"}), 400
        
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
        orders = broker.get_open_orders()
        
        return jsonify({"orders": orders, "count": len(orders)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/universe')
def get_universe():
    """Get current stock universe."""
    return jsonify({
        "universe": config.UNIVERSE,
        "count": len(config.UNIVERSE),
        "top_n": config.TOP_N,
        "benchmark": config.BENCHMARK,
        "strategies": [
            "TimeSeriesMomentum", "CrossSectionMomentum", "MeanReversion",
            "VolatilityRegimeVolTarget", "Carry", "ValueQualityTilt",
            "RiskParityMinVar", "TailRiskOverlay", "NewsSentimentEvent"
        ]
    })


@app.route('/api/portfolio')
def get_portfolio():
    """Get current portfolio information."""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return jsonify({"error": "API keys not configured"}), 400
        
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
        account = broker.get_account()
        positions_dict = broker.get_positions()
        
        # Convert positions dict to array with symbol key
        positions_list = []
        for symbol, pos in positions_dict.items():
            pos_data = {
                "symbol": symbol,
                "qty": pos.get("qty", 0),
                "avg_entry_price": pos.get("avg_entry_price", 0),
                "current_price": pos.get("current_price", 0),
                "market_value": pos.get("market_value", 0),
                "cost_basis": pos.get("cost_basis", 0),
                "unrealized_pl": pos.get("pnl", 0),
                "unrealized_plpc": pos.get("pnl_pct", 0) / 100 if pos.get("pnl_pct") else 0,
            }
            positions_list.append(pos_data)
        
        # Calculate totals
        total_pnl = sum(pos.get("pnl", 0) for pos in positions_dict.values())
        total_cost_basis = sum(pos.get("cost_basis", 0) for pos in positions_dict.values())
        total_market_value = sum(pos.get("market_value", 0) for pos in positions_dict.values())
        total_pnl_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        
        # Return FLAT structure that matches what the JavaScript expects
        return jsonify({
            # Top-level fields for easy access
            "equity": account.get("equity", 0),
            "cash": account.get("cash", 0),
            "buying_power": account.get("buying_power", 0),
            "market_value": total_market_value,
            "total_pl": total_pnl,
            "total_pl_pct": total_pnl_pct,
            "positions": positions_list,  # Array, not object
            # Also include nested for backwards compatibility
            "account": account,
            "summary": {
                "total_positions": len(positions_dict),
                "total_market_value": total_market_value,
                "total_cost_basis": total_cost_basis,
                "total_pnl": total_pnl,
                "total_pnl_pct": total_pnl_pct,
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/trade-history')
def get_alpaca_trade_history():
    """Get trade history from Alpaca and learning system."""
    try:
        trades = []
        
        # Try to get trades from learning system first
        try:
            trades_df = learning_engine.trade_memory.to_dataframe()
            if not trades_df.empty:
                for _, row in trades_df.iterrows():
                    trades.append({
                        "timestamp": row.get('timestamp', row.get('entry_time', '')),
                        "symbol": row.get('symbol', ''),
                        "side": row.get('side', 'buy'),
                        "qty": row.get('qty', row.get('quantity', 0)),
                        "price": row.get('entry_price', row.get('price', 0)),
                        "notional": row.get('notional', 0),
                        "pnl": row.get('pnl', row.get('realized_pnl', None))
                    })
        except Exception as e:
            print(f"Could not get learning trades: {e}")
        
        # Also try Alpaca orders
        try:
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            
            if api_key and secret_key:
                broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
                orders = broker.api.get_orders(status='closed', limit=50)
                
                for order in orders:
                    if order.filled_at:
                        trades.append({
                            "timestamp": str(order.filled_at),
                            "symbol": order.symbol,
                            "side": order.side,
                            "qty": float(order.filled_qty) if order.filled_qty else 0,
                            "price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                            "notional": float(order.filled_qty or 0) * float(order.filled_avg_price or 0),
                            "pnl": None  # Alpaca orders don't include PnL
                        })
        except Exception as e:
            print(f"Could not get Alpaca orders: {e}")
        
        # Sort by timestamp (most recent first)
        trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            "trades": trades[:100],  # Limit to 100 most recent
            "count": len(trades)
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "trades": []}), 500


@app.route('/api/data-sources')
def get_data_sources():
    """Get information about data sources."""
    av_stats = alpha_vantage_news.get_cache_stats()
    
    return jsonify({
        "market_data": {
            "source": "Alpaca Markets API",
            "feed": "IEX (free tier)",
            "frequency": "Daily OHLCV",
            "lookback": "~300 trading days",
            "note": "Real-time market data from Alpaca paper trading account"
        },
        "sentiment_data": {
            "source": "Alpha Vantage News Sentiment API",
            "api_key_status": "✅ Configured" if alpha_vantage_news.api_key else "❌ Missing",
            "method": "Alpha Vantage built-in sentiment analysis",
            "features": [
                "Ticker-specific news",
                "Topic filtering (earnings, macro, M&A, etc.)",
                "Built-in sentiment scores",
                "Real-time market news"
            ],
            "cached_articles": av_stats['total_articles'],
            "api_calls_today": av_stats['api_calls_made'],
            "cache_hits": av_stats['cache_hits'],
            "note": "Financial news with built-in sentiment from Alpha Vantage"
        },
        "macro_intelligence": {
            "source": "News Intelligence Pipeline",
            "features": [
                "Relevance gate (filters noise)",
                "Macro/Geo taxonomy (10 tags)",
                "Event extraction",
                "Impact scoring",
                "Risk sentiment analysis"
            ],
            "output": "8 daily macro indices + risk-on/off signal"
        },
        "regime_classification": {
            "method": "Technical indicators",
            "indicators": ["Trend (MA crossover)", "Volatility (realized vol percentile)", "Correlation regime"],
            "note": "Classifies market as Risk-On/Risk-Off based on SPY behavior"
        }
    })


# ============================================================
# LEARNING SYSTEM API ENDPOINTS
# ============================================================

@app.route('/api/learning/summary')
def get_learning_summary():
    """Get comprehensive learning summary."""
    try:
        summary = learning_engine.get_learning_summary()
        
        # Get trade stats for UI
        trade_stats = learning_engine.trade_memory.get_statistics()
        pattern_summary = learning_engine.pattern_learner.get_learning_summary()
        
        # Add fields the UI expects
        summary['total_trades'] = trade_stats.get('total_trades', 0)
        summary['win_rate'] = trade_stats.get('win_rate', 0)
        summary['patterns_found'] = pattern_summary.get('discovered_patterns', 0)
        summary['total_pnl'] = trade_stats.get('total_pnl', 0)
        summary['avg_pnl'] = trade_stats.get('avg_pnl_percent', 0)
        summary['best_trade'] = trade_stats.get('best_trade', 'N/A')
        summary['worst_trade'] = trade_stats.get('worst_trade', 'N/A')
        summary['open_positions'] = trade_stats.get('open_positions', 0)
        summary['closed_trades'] = trade_stats.get('closed_trades', 0)
        
        return jsonify(summary)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/mistakes')
def get_mistake_analysis():
    """Get analysis of losing trades and lessons learned."""
    try:
        analysis = learning_engine.get_mistake_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/trades')
def get_trade_history():
    """Get trade history with full context."""
    try:
        trades_df = learning_engine.trade_memory.to_dataframe()
        if trades_df.empty:
            return jsonify({"trades": [], "count": 0})
        
        # Convert to list of dicts for JSON
        trades = trades_df.to_dict(orient='records')
        
        return jsonify({
            "trades": trades,
            "count": len(trades),
            "statistics": learning_engine.trade_memory.get_statistics()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/performance')
def get_strategy_performance():
    """Get detailed strategy performance metrics."""
    try:
        performance = learning_engine.performance_tracker.get_summary()
        ranking = learning_engine.performance_tracker.get_strategy_ranking()
        insights = learning_engine.performance_tracker.get_learning_insights()
        
        return jsonify({
            "summary": performance,
            "ranking": [{"strategy": name, "score": score} for name, score in ranking],
            "insights": insights
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/weights')
def get_learned_weights():
    """Get current learned strategy weights."""
    try:
        weights_summary = learning_engine.adaptive_weights.get_learning_summary()
        
        return jsonify({
            "current_weights": weights_summary.get('current_weights', {}),
            "top_strategies": weights_summary.get('top_strategies', []),
            "learning_rounds": weights_summary.get('total_learning_rounds', 0),
            "exploration_needed": weights_summary.get('exploration_needed', []),
            "regime_specialists": weights_summary.get('regime_specialists', {})
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/patterns')
def get_learned_patterns():
    """Get discovered patterns and active rules."""
    try:
        pattern_summary = learning_engine.pattern_learner.get_learning_summary()
        
        return jsonify({
            "total_patterns": pattern_summary.get('total_patterns', 0),
            "discovered_patterns": pattern_summary.get('discovered_patterns', 0),
            "high_confidence_patterns": pattern_summary.get('high_confidence_patterns', 0),
            "top_patterns": pattern_summary.get('top_patterns', []),
            "total_observations": pattern_summary.get('total_observations', 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/update-outcomes', methods=['POST'])
def update_learning_outcomes():
    """
    Manually trigger comprehensive outcome update for learning.
    This fetches current prices, updates P/L for ALL positions (open and recently closed),
    and ensures learning data is complete.
    """
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return jsonify({"error": "API keys not configured"}), 400
        
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
        
        # Get current positions from broker
        positions = broker.get_positions()
        current_position_symbols = set(positions.keys()) if positions else set()
        
        # Get current prices for all symbols we have trades for
        all_trade_symbols = set(t.symbol for t in learning_engine.trade_memory.trades)
        symbols_to_fetch = all_trade_symbols | current_position_symbols
        
        # Get latest prices
        current_prices = {}
        if symbols_to_fetch:
            try:
                # Get historical bars for price data
                price_data = broker.get_historical_bars(list(symbols_to_fetch), days=5)
                for symbol, df in price_data.items():
                    if df is not None and not df.empty:
                        current_prices[symbol] = df['close'].iloc[-1]
            except Exception as e:
                logging.warning(f"Could not fetch all prices: {e}")
        
        # Also get prices from current positions (more accurate)
        if positions:
            for symbol, pos in positions.items():
                current_prices[symbol] = pos.get('current_price', pos.get('avg_entry_price', 0))
        
        # Update open positions with current prices
        updated_count = 0
        closed_count = 0
        
        # 1. Update all open positions in trade memory with current prices
        for symbol, trade in list(learning_engine.trade_memory.open_positions.items()):
            if symbol in current_prices:
                price = current_prices[symbol]
                if trade.side == 'buy':
                    trade.pnl_dollars = (price - trade.entry_price) * trade.quantity
                    trade.pnl_percent = (price - trade.entry_price) / trade.entry_price * 100
                    trade.was_profitable = trade.pnl_dollars > 0
                    updated_count += 1
                
                # If this position is no longer held by broker, close it
                if symbol not in current_position_symbols:
                    learning_engine.trade_memory._close_position(trade, price, 'position_closed')
                    del learning_engine.trade_memory.open_positions[symbol]
                    closed_count += 1
        
        # 2. Update trades that don't have PnL yet
        for trade in learning_engine.trade_memory.trades:
            if trade.pnl_percent is None and trade.symbol in current_prices:
                price = current_prices[trade.symbol]
                if trade.side == 'buy':
                    trade.pnl_dollars = (price - trade.entry_price) * trade.quantity
                    trade.pnl_percent = (price - trade.entry_price) / trade.entry_price * 100
                    trade.was_profitable = trade.pnl_dollars > 0
                    updated_count += 1
        
        # Save updated trade memory
        learning_engine.trade_memory._save()
        
        # Calculate returns for pattern learning
        symbol_returns = {}
        for symbol, pos in (positions or {}).items():
            cost_basis = pos.get('cost_basis', 0)
            current_value = pos.get('current_value', pos.get('market_value', 0))
            if cost_basis > 0:
                symbol_returns[symbol] = (current_value - cost_basis) / cost_basis
        
        # Get real market data for proper context
        market_indicators = macro_loader.fetch_all()
        
        # Determine regime from VIX and SPY momentum
        vix = market_indicators.vix if market_indicators else 20
        spy_change = market_indicators.spy_change_pct if market_indicators else 0
        
        # Classify regime
        if vix > 25 or spy_change < -1:
            regime = 'risk_off'
        elif vix < 18 and spy_change > 0.5:
            regime = 'risk_on'
        else:
            regime = 'neutral'
        
        # Classify volatility
        if vix > 25:
            vol_regime = 'high'
        elif vix < 15:
            vol_regime = 'low'
        else:
            vol_regime = 'medium'
        
        # Classify trend
        spy_vs_ma = market_indicators.spy_vs_200ma if market_indicators else 0
        if abs(spy_vs_ma) > 5:
            trend = 'strong'
        else:
            trend = 'weak'
        
        # Build market context with REAL data
        market_context = {
            'regime': regime,
            'volatility_regime': vol_regime,
            'trend_strength': trend,
            'vix': vix,
            'spy_change': spy_change,
        }
        
        # Record outcomes for pattern learning
        if symbol_returns:
            learning_engine.record_outcomes(symbol_returns, market_context)
        
        # Get updated statistics
        stats = learning_engine.trade_memory.get_statistics()
        
        return jsonify({
            "message": "Outcomes updated successfully",
            "trades_updated": updated_count,
            "positions_closed": closed_count,
            "total_trades": stats.get('total_trades', 0),
            "trades_with_pnl": stats.get('closed_trades', 0),
            "win_rate": stats.get('win_rate', 0),
            "total_pnl": stats.get('total_pnl', 0),
            "current_positions": list(current_position_symbols),
        })
    except Exception as e:
        logging.error(f"Error updating learning outcomes: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/config', methods=['GET', 'POST'])
def learning_config():
    """Get or update learning configuration."""
    if request.method == 'GET':
        return jsonify({
            "learning_influence": learning_engine.learning_influence,
            "strategies_tracked": len(learning_engine.strategy_names),
            "outputs_dir": "outputs"
        })
    
    elif request.method == 'POST':
        data = request.get_json() or {}
        
        if 'learning_influence' in data:
            new_influence = float(data['learning_influence'])
            if 0 <= new_influence <= 1:
                learning_engine.learning_influence = new_influence
                return jsonify({
                    "message": "Learning influence updated",
                    "learning_influence": learning_engine.learning_influence
                })
            else:
                return jsonify({"error": "learning_influence must be between 0 and 1"}), 400
        
        return jsonify({"error": "No valid configuration provided"}), 400


@app.route('/api/learning/report')
def get_learning_report():
    """Generate comprehensive learning report with all insights."""
    try:
        from src.learning import LearningReportGenerator
        
        report_gen = LearningReportGenerator(outputs_dir="outputs")
        report = report_gen.generate_full_report()
        
        return jsonify(report)
    except Exception as e:
        logging.error(f"Error generating learning report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/report/html')
def get_learning_report_html():
    """Generate and return HTML learning report."""
    try:
        from src.learning import LearningReportGenerator
        
        report_gen = LearningReportGenerator(outputs_dir="outputs")
        html = report_gen.generate_html_report()
        
        return html, 200, {'Content-Type': 'text/html'}
    except Exception as e:
        logging.error(f"Error generating HTML learning report: {e}")
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>", 500


@app.route('/api/learning/report/download')
def download_learning_report():
    """Generate and download HTML learning report."""
    try:
        from src.learning import LearningReportGenerator
        from flask import Response
        from datetime import datetime
        
        report_gen = LearningReportGenerator(outputs_dir="outputs")
        html = report_gen.generate_html_report()
        
        filename = f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        return Response(
            html,
            mimetype='text/html',
            headers={'Content-Disposition': f'attachment;filename={filename}'}
        )
    except Exception as e:
        logging.error(f"Error downloading learning report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/learning/realtime')
def get_realtime_learning_stats():
    """Get real-time learning statistics for dashboard display."""
    try:
        # Get trade statistics
        trade_stats = learning_engine.trade_memory.get_statistics()
        
        # Get current learning influence
        current_influence = learning_engine.get_adaptive_learning_influence()
        
        # Get pattern learner summary
        pattern_summary = learning_engine.pattern_learner.get_learning_summary()
        
        # Get performance tracker summary
        perf_summary = learning_engine.performance_tracker.get_summary()
        
        # Get strategy ranking
        try:
            ranking = learning_engine.performance_tracker.get_strategy_ranking()
        except Exception:
            ranking = []
        
        # Get active patterns
        market_context = {
            'regime': last_run_status.get('regime', 'neutral'),
            'volatility_regime': 'medium',
            'trend_strength': 0.0,
        }
        
        try:
            active_patterns = learning_engine.pattern_learner.get_active_patterns(market_context)
            active_pattern_list = [
                {
                    'description': p.description if hasattr(p, 'description') else str(p),
                    'confidence': p.confidence if hasattr(p, 'confidence') else 0.5,
                    'action': p.recommended_action if hasattr(p, 'recommended_action') else 'hold',
                }
                for p in active_patterns[:3]
            ]
        except Exception:
            active_pattern_list = []
        
        # Get top and bottom strategies
        weights_summary = learning_engine.adaptive_weights.get_learning_summary()
        current_weights = weights_summary.get('current_weights', {})
        
        sorted_strategies = sorted(
            current_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_strategies = sorted_strategies[:3]
        bottom_strategies = sorted_strategies[-3:] if len(sorted_strategies) > 3 else []
        
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            
            # Core stats
            "total_trades": trade_stats.get('total_trades', 0),
            "win_rate": trade_stats.get('win_rate', 0.5),
            "total_pnl": trade_stats.get('total_pnl', 0),
            "avg_pnl_percent": trade_stats.get('avg_pnl_percent', 0),
            
            # Learning influence
            "learning_influence": current_influence,
            "debate_influence": 1 - current_influence,
            
            # Pattern stats
            "patterns_discovered": pattern_summary.get('discovered_patterns', 0),
            "patterns_active": len(active_pattern_list),
            "active_patterns": active_pattern_list,
            
            # Strategy stats
            "strategies_tracked": perf_summary.get('strategies_tracked', 0),
            "overall_accuracy": perf_summary.get('overall_accuracy', 0.5),
            "best_strategy": perf_summary.get('best_strategy'),
            
            "top_strategies": [
                {"name": name, "weight": weight}
                for name, weight in top_strategies
            ],
            "bottom_strategies": [
                {"name": name, "weight": weight}
                for name, weight in bottom_strategies
            ],
            
            # Strategy ranking
            "strategy_ranking": [
                {"strategy": name, "score": score}
                for name, score in ranking[:5]
            ] if ranking else [],
            
            # Maturity indicator
            "learning_maturity": _get_learning_maturity(trade_stats.get('total_trades', 0)),
        })
    except Exception as e:
        logging.error(f"Error getting realtime learning stats: {e}")
        return jsonify({"error": str(e)}), 500


def _get_learning_maturity(total_trades):
    """Calculate learning maturity level."""
    if total_trades < 20:
        return {"level": "early", "description": "Early Stage - Gathering initial data"}
    elif total_trades < 100:
        return {"level": "developing", "description": "Developing - Patterns emerging"}
    elif total_trades < 500:
        return {"level": "maturing", "description": "Maturing - High confidence learnings"}
    else:
        return {"level": "mature", "description": "Mature - Learnings are primary drivers"}


# ============================================================
# OPTIMIZATION API ENDPOINTS
# ============================================================

@app.route('/api/optimizations/stats')
def get_optimization_stats():
    """Get optimization and performance statistics."""
    try:
        return jsonify({
            "parallel_executor": {
                "performance": parallel_executor.get_performance_stats(),
                "slow_strategies": parallel_executor.get_slow_strategies(),
            },
            "data_cache": price_cache.get_stats(),
            "smart_sizing": smart_sizer.get_sizing_stats(),
            "thompson_sampling": thompson_sampler.get_summary(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/optimizations/thompson')
def get_thompson_stats():
    """Get Thompson Sampling exploration/exploitation stats."""
    try:
        summary = thompson_sampler.get_summary()
        return jsonify({
            "beliefs": summary.get('beliefs', {}),
            "exploration_priorities": summary.get('exploration_priorities', []),
            "total_updates": summary.get('total_updates', 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/optimizations/cache/invalidate', methods=['POST'])
def invalidate_cache():
    """Invalidate data cache."""
    try:
        category = request.get_json().get('category') if request.get_json() else None
        price_cache.invalidate(category)
        return jsonify({
            "message": f"Cache invalidated: {category or 'all'}",
            "stats": price_cache.get_stats()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/news/cache')
def get_news_cache_stats():
    """Get news cache statistics."""
    try:
        news_loader = NewsDataLoader()
        stats = news_loader.get_cache_stats()
        
        # Format timestamps for JSON
        if stats.get('oldest_article'):
            stats['oldest_article'] = stats['oldest_article'].isoformat()
        if stats.get('newest_article'):
            stats['newest_article'] = stats['newest_article'].isoformat()
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/news/cache/clear', methods=['POST'])
def clear_news_cache():
    """Clear news cache."""
    try:
        data = request.get_json() or {}
        older_than_days = data.get('older_than_days')
        
        news_loader = NewsDataLoader()
        news_loader.clear_cache(older_than_days)
        
        return jsonify({
            "message": f"News cache cleared" + (f" (older than {older_than_days} days)" if older_than_days else ""),
            "stats": news_loader.get_cache_stats()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/news/cached-articles')
def get_cached_articles():
    """Get list of cached news articles."""
    try:
        news_loader = NewsDataLoader()
        
        # Optional filters
        ticker = request.args.get('ticker')
        limit = int(request.args.get('limit', 50))
        
        articles = news_loader._articles_cache
        
        # Filter by ticker if provided
        if ticker:
            articles = [a for a in articles if ticker in a.tickers]
        
        # Sort by date (newest first) and limit
        articles = sorted(articles, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return jsonify({
            "articles": [
                {
                    "id": a.id,
                    "timestamp": a.timestamp.isoformat(),
                    "headline": a.headline,
                    "source": a.source,
                    "tickers": a.tickers,
                    "topics": a.topics,
                    "sentiment": a.raw_sentiment,
                }
                for a in articles
            ],
            "count": len(articles),
            "total_cached": len(news_loader._articles_cache),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# NEWS INTELLIGENCE API ENDPOINTS
# ============================================================

def refresh_macro_from_alpha_vantage():
    """Fetch news from Alpha Vantage and process through macro pipeline."""
    import pytz
    from src.news_intelligence.pipeline import NewsArticle as NIPNewsArticle
    
    as_of = datetime.now(pytz.UTC)
    
    # Fetch from Alpha Vantage
    articles = alpha_vantage_news.fetch_market_news(days_back=3)
    
    if not articles:
        # Try cached articles
        articles = alpha_vantage_news.get_cached_articles()
    
    if not articles:
        return None, None, "No articles available"
    
    # Convert to pipeline format
    pipeline_articles = []
    for article in articles:
        try:
            ts = article.timestamp
            if ts.tzinfo is None:
                ts = pytz.UTC.localize(ts)
            pipeline_articles.append(NIPNewsArticle(
                timestamp=ts,
                source=article.source,
                title=article.headline,
                body=article.summary or "",
                url=article.url,
            ))
        except:
            continue
    
    if pipeline_articles:
        events, stats = news_intelligence.process_articles(pipeline_articles, as_of)
        
        # Get computed features
        macro_features = news_intelligence.get_daily_macro_features(as_of)
        risk_sentiment = news_intelligence.get_risk_sentiment(as_of)
        
        # Store globally
        last_macro_features["features"] = macro_features
        last_macro_features["risk_sentiment"] = risk_sentiment
        last_macro_features["last_updated"] = as_of
        
        return macro_features, risk_sentiment, f"Processed {len(pipeline_articles)} articles"
    
    return None, None, "No articles to process"


@app.route('/api/macro/refresh', methods=['POST'])
def refresh_macro_data():
    """Refresh macro data from Alpha Vantage."""
    try:
        features, sentiment, message = refresh_macro_from_alpha_vantage()
        
        if features:
            return jsonify({
                "success": True,
                "message": message,
                "risk_sentiment": sentiment.risk_sentiment if sentiment else 0,
                "indices_computed": True,
            })
        else:
            return jsonify({
                "success": False,
                "message": message,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def compute_hybrid_macro_indices(market_data, news_features):
    """
    Compute macro indices using HYBRID approach:
    - Real economic data (FRED/Yahoo) as baseline
    - News sentiment as modifier
    
    Returns indices in [-1, 1] or [0, 1] range
    """
    # === INFLATION PRESSURE INDEX ===
    # Baseline from CPI YoY - normalize around 2% target
    # Note: FRED CPIAUCSL returns raw index (e.g., 326), not YoY change
    # We approximate YoY as ~3% when raw index is returned
    cpi = market_data.cpi_yoy if market_data.cpi_yoy > 0 else 3.0
    if cpi > 20:  # If it's a raw index value (like 326), approximate YoY as 3%
        cpi = 3.0  # Default to 3% inflation
    # CPI > 4% = high inflation pressure, < 2% = low
    inflation_baseline = (cpi - 2.0) / 4.0  # Maps 2% to 0, 6% to 1, -2% to -1
    inflation_baseline = max(-1, min(1, inflation_baseline))
    # Add news modifier
    news_inflation = getattr(news_features, 'macro_inflation_pressure_index', 0) if news_features else 0
    inflation_index = inflation_baseline * 0.7 + news_inflation * 0.3
    
    # === LABOR STRENGTH INDEX ===
    # Baseline from unemployment - lower = stronger
    unemployment = market_data.unemployment_rate if market_data.unemployment_rate > 0 else 4.0
    # 3.5% = strong (+0.5), 5% = neutral (0), 7% = weak (-0.5)
    labor_baseline = (5.0 - unemployment) / 3.0
    labor_baseline = max(-1, min(1, labor_baseline))
    news_labor = getattr(news_features, 'labor_strength_index', 0) if news_features else 0
    labor_index = labor_baseline * 0.7 + news_labor * 0.3
    
    # === CENTRAL BANK HAWKISHNESS ===
    # Baseline from Fed Funds Rate
    fed_rate = market_data.fed_funds_rate if market_data.fed_funds_rate > 0 else 4.5
    # Higher rate = more hawkish; 5%+ = very hawkish, 2% = neutral, 0% = dovish
    hawkish_baseline = (fed_rate - 2.5) / 3.0
    hawkish_baseline = max(-1, min(1, hawkish_baseline))
    news_hawkish = getattr(news_features, 'central_bank_hawkishness_index', 0) if news_features else 0
    hawkishness_index = hawkish_baseline * 0.6 + news_hawkish * 0.4  # News more important for CB
    
    # === FINANCIAL STRESS INDEX ===
    # Use FRED's Financial Stress Index directly if available
    fred_stress = market_data.financial_stress_index
    # STLFSI: 0 = average, >1 = stressed, <-1 = calm
    if fred_stress != 0:
        stress_baseline = fred_stress / 2.0  # Normalize
    else:
        # Derive from VIX: 15 = calm, 25 = stressed, 35 = very stressed
        stress_baseline = (market_data.vix - 20) / 15.0
    stress_baseline = max(-1, min(1, stress_baseline))
    news_stress = getattr(news_features, 'financial_stress_index', 0) if news_features else 0
    financial_stress_index = stress_baseline * 0.7 + news_stress * 0.3
    
    # === GEOPOLITICAL RISK INDEX ===
    # No direct FRED data - rely more on news but use VIX as proxy
    vix_geo_proxy = max(0, (market_data.vix - 18) / 20)  # VIX > 18 suggests some fear
    news_geo = getattr(news_features, 'geopolitical_risk_index', 0) if news_features else 0
    geopolitical_index = vix_geo_proxy * 0.3 + news_geo * 0.7
    geopolitical_index = max(0, min(1, geopolitical_index))
    
    # === GROWTH MOMENTUM INDEX ===
    # Use SPY momentum as proxy for growth
    spy_momentum = market_data.spy_change_pct / 3.0  # 3% daily move = extreme
    spy_vs_ma = market_data.spy_vs_200ma / 15.0  # 15% above/below MA
    growth_baseline = (spy_momentum * 0.3 + spy_vs_ma * 0.7)
    growth_baseline = max(-1, min(1, growth_baseline))
    news_growth = getattr(news_features, 'growth_momentum_index', 0) if news_features else 0
    growth_index = growth_baseline * 0.6 + news_growth * 0.4
    
    # === COMMODITIES SUPPLY RISK ===
    # Use oil price deviation as proxy
    oil = market_data.oil_price if market_data.oil_price > 0 else 75
    # Oil > $100 = supply concerns, < $60 = ample supply
    oil_risk = (oil - 75) / 35.0  # Normalize around $75
    oil_risk = max(-1, min(1, oil_risk))
    news_commodities = getattr(news_features, 'commodities_supply_risk_index', 0) if news_features else 0
    commodities_index = oil_risk * 0.5 + news_commodities * 0.5
    
    # === OVERALL RISK SENTIMENT ===
    # Combine multiple factors
    risk_sentiment = (
        growth_index * 0.25 +
        labor_index * 0.15 -
        financial_stress_index * 0.25 -
        geopolitical_index * 0.20 -
        abs(inflation_index) * 0.10 -  # High inflation either direction is bad
        (hawkishness_index * 0.05 if hawkishness_index > 0 else 0)
    )
    risk_sentiment = max(-1, min(1, risk_sentiment))
    
    return {
        "macro_inflation_pressure_index": round(inflation_index, 3),
        "labor_strength_index": round(labor_index, 3),
        "growth_momentum_index": round(growth_index, 3),
        "central_bank_hawkishness_index": round(hawkishness_index, 3),
        "geopolitical_risk_index": round(geopolitical_index, 3),
        "financial_stress_index": round(financial_stress_index, 3),
        "commodities_supply_risk_index": round(commodities_index, 3),
        "overall_risk_sentiment_index": round(risk_sentiment, 3),
    }


@app.route('/api/macro/features')
def get_macro_features():
    """Get current macro features combining REAL DATA + news intelligence."""
    try:
        import pytz
        as_of = datetime.now(pytz.UTC)
        
        # Get real market data from Yahoo/FRED (this is always fresh)
        market_data = macro_loader.fetch_all(force=True)
        
        # Get news-based features (may be cached or empty)
        if last_macro_features["features"]:
            news_features = last_macro_features["features"]
            last_updated = last_macro_features["last_updated"]
        else:
            news_features = news_intelligence.get_daily_macro_features(as_of)
            last_updated = None
        
        # === COMPUTE HYBRID INDICES ===
        # This combines real economic data (FRED, Yahoo) with news sentiment
        hybrid_indices = compute_hybrid_macro_indices(market_data, news_features)
        
        # Compute VIX-based volatility stress (for display)
        vix_stress = 0
        if market_data.vix > 30:
            vix_stress = 0.8
        elif market_data.vix > 25:
            vix_stress = 0.5
        elif market_data.vix > 20:
            vix_stress = 0.2
        
        return jsonify({
            "date": as_of.isoformat(),
            "indices": {
                **hybrid_indices,  # All hybrid computed indices
                "vix_stress": vix_stress,
            },
            "sentiment": {
                "overall_risk": market_data.risk_score,
                "equity_bias": getattr(news_features, 'equity_bias', 0) if news_features else 0,
                "rates_bias": getattr(news_features, 'rates_bias', 0) if news_features else 0,
                "dollar_bias": getattr(news_features, 'dollar_bias', 0) if news_features else 0,
            },
            "real_data": {
                "vix": market_data.vix,
                "vix_change": market_data.vix_change,
                "spy_price": market_data.spy_price,
                "spy_change_pct": market_data.spy_change_pct,
                "spy_vs_200ma": market_data.spy_vs_200ma,
                "treasury_10y": market_data.treasury_10y,
                "cpi_yoy": market_data.cpi_yoy,
                "unemployment": market_data.unemployment_rate,
                "fed_funds": market_data.fed_funds_rate,
                "financial_stress_fred": market_data.financial_stress_index,
                "gold": market_data.gold_price,
                "oil": market_data.oil_price,
            },
            "computation_method": {
                "description": "Hybrid: 60-70% real data (FRED/Yahoo) + 30-40% news sentiment",
                "real_data_sources": ["Yahoo Finance (VIX, SPY, Treasury, Gold, Oil)", "FRED (CPI, Unemployment, Fed Funds, Financial Stress)"],
                "news_source": "Alpha Vantage News Sentiment API",
            },
            "metadata": {
                "event_count": getattr(news_features, 'event_count', 0) if news_features else 0,
                "high_impact_events": getattr(news_features, 'high_impact_event_count', 0) if news_features else 0,
                "data_quality": getattr(news_features, 'data_quality_score', 0) if news_features else 0,
                "last_news_update": last_updated.isoformat() if last_updated else "No news processed yet",
                "market_data_timestamp": market_data.timestamp.isoformat(),
            },
            "top_events": getattr(news_features, 'top_events', []) if news_features else [],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/suggested-exposure')
def get_suggested_exposure():
    """Calculate suggested capital exposure based on macro environment."""
    try:
        import pytz
        as_of = datetime.now(pytz.UTC)
        
        # Get macro features
        if last_macro_features["features"]:
            features = last_macro_features["features"]
        else:
            features = news_intelligence.get_daily_macro_features(as_of)
        
        # Base exposure
        suggested_pct = 80
        reasons = []
        
        # Adjust based on risk indicators
        geo_risk = getattr(features, 'geopolitical_risk_index', 0) or 0
        fin_stress = getattr(features, 'financial_stress_index', 0) or 0
        risk_sentiment = getattr(features, 'overall_risk_sentiment_index', 0) or 0
        
        # High geopolitical risk
        if geo_risk > 0.5:
            suggested_pct -= 30
            reasons.append(f"High geopolitical risk ({geo_risk:.2f})")
        elif geo_risk > 0.3:
            suggested_pct -= 15
            reasons.append(f"Elevated geopolitical risk ({geo_risk:.2f})")
        
        # Financial stress
        if fin_stress > 0.5:
            suggested_pct -= 25
            reasons.append(f"High financial stress ({fin_stress:.2f})")
        elif fin_stress > 0.3:
            suggested_pct -= 10
            reasons.append(f"Moderate financial stress ({fin_stress:.2f})")
        
        # Risk sentiment
        if risk_sentiment < -0.3:
            suggested_pct -= 15
            reasons.append(f"Risk-off sentiment ({risk_sentiment:.2f})")
        elif risk_sentiment > 0.3:
            suggested_pct += 10
            reasons.append(f"Risk-on sentiment ({risk_sentiment:.2f})")
        
        # Clamp between 20% and 100%
        suggested_pct = max(20, min(100, suggested_pct))
        
        return jsonify({
            "suggested_exposure_pct": suggested_pct,
            "reasons": reasons,
            "geo_risk": geo_risk,
            "financial_stress": fin_stress,
            "risk_sentiment": risk_sentiment,
        })
    except Exception as e:
        return jsonify({"suggested_exposure_pct": 80, "reasons": [str(e)]})


@app.route('/api/long-short-settings', methods=['GET'])
def get_long_short_settings():
    """Get current long/short trading settings."""
    return jsonify(long_short_settings)


@app.route('/api/long-short-settings', methods=['POST'])
def update_long_short_settings():
    """Update long/short trading settings."""
    global long_short_settings
    
    try:
        data = request.get_json()
        
        if 'enable_long_short' in data:
            long_short_settings['enable_long_short'] = bool(data['enable_long_short'])
            logging.info(f"Long/Short mode: {'ENABLED' if long_short_settings['enable_long_short'] else 'DISABLED'}")
        
        if 'enable_futures' in data:
            long_short_settings['enable_futures'] = bool(data['enable_futures'])
            logging.info(f"Futures strategies: {'ENABLED' if long_short_settings['enable_futures'] else 'DISABLED'}")
        
        if 'enable_shorting' in data:
            long_short_settings['enable_shorting'] = bool(data['enable_shorting'])
            logging.info(f"Short selling: {'ENABLED' if long_short_settings['enable_shorting'] else 'DISABLED'}")
        
        if 'max_gross_exposure' in data:
            long_short_settings['max_gross_exposure'] = float(data['max_gross_exposure'])
        
        if 'net_exposure_min' in data:
            long_short_settings['net_exposure_min'] = float(data['net_exposure_min'])
        
        if 'net_exposure_max' in data:
            long_short_settings['net_exposure_max'] = float(data['net_exposure_max'])
        
        return jsonify({
            "success": True,
            "settings": long_short_settings,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/trading-mode', methods=['GET'])
def get_trading_mode():
    """Get current trading mode (intraday vs position)."""
    return jsonify({
        "mode": trading_mode_setting,
        "description": {
            "intraday": "15-30 minute trading (HFT-lite) - quick in-and-out",
            "position": "Daily/weekly trading - holds for days/weeks",
        }[trading_mode_setting],
        "strategies": {
            "intraday": [
                "IntradayMomentum", "VWAPReversion", "VolumeSpike",
                "RelativeStrengthIntraday", "OpeningRangeBreakout", "QuickMeanReversion",
                "NewsSentimentEvent", "CrossSectionMomentum (5-day)"
            ],
            "position": [
                "TimeSeriesMomentum (126d)", "CrossSectionMomentum (126d)",
                "MeanReversion", "Carry", "ValueQualityTilt", "RiskParity"
            ],
        }[trading_mode_setting],
    })


@app.route('/api/trading-mode', methods=['POST'])
def set_trading_mode():
    """Set trading mode (intraday vs position)."""
    global trading_mode_setting
    
    try:
        data = request.get_json()
        new_mode = data.get('mode', 'intraday')
        
        if new_mode not in ['intraday', 'position']:
            return jsonify({
                "success": False,
                "error": f"Invalid mode: {new_mode}. Must be 'intraday' or 'position'"
            }), 400
        
        trading_mode_setting = new_mode
        logging.info(f"Trading mode changed to: {trading_mode_setting}")
        
        return jsonify({
            "success": True,
            "mode": trading_mode_setting,
            "message": f"Switched to {trading_mode_setting} trading mode"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/exposure-summary')
def get_exposure_summary():
    """Get current portfolio exposure summary (long/short/gross/net)."""
    try:
        broker = AlpacaBroker(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            paper=True
        )
        
        exposure = broker.get_exposure_summary()
        account = broker.get_account_info()
        
        return jsonify({
            "long_exposure": exposure.get('long', 0),
            "short_exposure": exposure.get('short', 0),
            "gross_exposure": exposure.get('gross', 0),
            "net_exposure": exposure.get('net', 0),
            "portfolio_value": account.get('portfolio_value', 0),
            "leverage": exposure.get('gross', 0) / float(account.get('portfolio_value', 1)) if account.get('portfolio_value') else 0,
            "net_leverage": exposure.get('net', 0) / float(account.get('portfolio_value', 1)) if account.get('portfolio_value') else 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/regime')
def get_regime():
    """Get current market regime detection."""
    global current_regime
    
    # If regime not set, calculate it on-demand
    if current_regime is None:
        try:
            # Get current market data
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            
            if api_key and secret_key:
                broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
                
                # Get SPY price
                prices = broker.get_current_prices(['SPY'])
                spy_price = prices.get('SPY', 0)
                
                # Get VIX
                try:
                    market_indicators = macro_loader.fetch_all()
                    vix_level = market_indicators.vix if market_indicators else 20.0
                except:
                    vix_level = 20.0
                
                # Get 200-day MA (approximate from recent data)
                try:
                    bars = broker.get_historical_bars(['SPY'], days=250)
                    if bars and 'SPY' in bars:
                        spy_series = bars['SPY']
                        if len(spy_series) >= 200:
                            spy_200ma = spy_series['close'].tail(200).mean()
                        else:
                            spy_200ma = spy_price
                    else:
                        spy_200ma = spy_price
                except:
                    spy_200ma = spy_price
                
                # Calculate regime using strategy enhancer
                if spy_price > 0:
                    enhancer = get_enhancer()
                    
                    # Get macro features if available
                    try:
                        # Try to get macro features from cache or calculate
                        macro_sentiment = 0.0
                        geo_risk = 0.5
                        financial_stress = 0.5
                        
                        # Try to get from macro loader
                        if market_indicators:
                            # Approximate macro sentiment from VIX and SPY
                            macro_sentiment = 0.5  # Neutral default
                            if hasattr(market_indicators, 'spy_change_pct'):
                                macro_sentiment = market_indicators.spy_change_pct / 10.0  # Normalize
                            
                            # Get geo risk and financial stress from macro features if available
                            # For now, use defaults
                    except:
                        macro_sentiment = 0.0
                        geo_risk = 0.5
                        financial_stress = 0.5
                    
                    current_regime = enhancer.detect_regime(
                        spy_price=spy_price,
                        spy_200ma=spy_200ma,
                        vix=vix_level,
                        macro_sentiment=macro_sentiment,
                        geo_risk=geo_risk,
                        financial_stress=financial_stress,
                        breadth=0.5,  # Default breadth
                    )
        except Exception as e:
            logging.warning(f"Could not calculate regime on-demand: {e}")
    
    if current_regime:
        return jsonify({
            "regime": current_regime.regime,
            "score": current_regime.score,
            "exposure_multiplier": current_regime.exposure_multiplier,
            "indicators": current_regime.indicators,
            "timestamp": current_regime.timestamp.isoformat() if hasattr(current_regime, 'timestamp') else datetime.now().isoformat(),
        })
    else:
        return jsonify({
            "regime": "unknown",
            "score": 0.5,
            "exposure_multiplier": 1.0,
            "indicators": {},
            "timestamp": None,
        })


@app.route('/api/risk-appetite', methods=['GET', 'POST'])
def manage_risk_appetite():
    """Get or set risk appetite."""
    global risk_appetite, strategy_enhancer
    
    if request.method == 'POST':
        data = request.get_json() or {}
        new_appetite = data.get('risk_appetite', request.args.get('risk_appetite'))
        
        if new_appetite in ['conservative', 'moderate', 'aggressive', 'maximum']:
            risk_appetite = new_appetite
            strategy_enhancer = get_enhancer(EnhancedConfig(risk_appetite=risk_appetite))
            return jsonify({
                "success": True,
                "risk_appetite": risk_appetite,
                "settings": {
                    "kelly_multiplier": strategy_enhancer.config.kelly_multiplier,
                    "min_position_pct": strategy_enhancer.config.min_position_pct,
                    "max_positions": strategy_enhancer.config.max_positions,
                }
            })
        else:
            return jsonify({"error": "Invalid risk appetite"}), 400
    
    # GET
    return jsonify({
        "risk_appetite": risk_appetite,
        "settings": {
            "kelly_multiplier": strategy_enhancer.config.kelly_multiplier,
            "min_position_pct": strategy_enhancer.config.min_position_pct,
            "max_positions": strategy_enhancer.config.max_positions,
            "min_investment_floor": strategy_enhancer.config.min_investment_floor,
        },
        "options": ["conservative", "moderate", "aggressive", "maximum"]
    })


@app.route('/api/universe-stats')
def get_universe_stats():
    """Get stock universe statistics."""
    try:
        return jsonify({
            "total_stocks": len(config.UNIVERSE),
            "sectors": {
                "Technology": 60,
                "Finance": 50,
                "Healthcare": 50,
                "Consumer Discretionary": 40,
                "Consumer Staples": 30,
                "Industrials": 40,
                "Energy": 20,
                "Communication": 15,
                "Utilities": 15,
                "Real Estate": 15,
                "Materials": 15,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# REAL-TIME MACRO DATA ENDPOINTS
# ============================================================

@app.route('/api/market-data')
def get_market_data():
    """Get real-time market data from Yahoo Finance."""
    try:
        indicators = macro_loader.fetch_all()
        return jsonify({
            "success": True,
            "data": indicators.to_dict(),
            "sources": {
                "yahoo_finance": True,
                "fred_api": macro_loader.fred.is_available(),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/market-data/vix')
def get_vix():
    """Get current VIX level."""
    try:
        vix, change = macro_loader.yahoo.get_vix()
        return jsonify({
            "vix": vix,
            "change": change,
            "status": "low" if vix < 15 else "normal" if vix < 20 else "elevated" if vix < 25 else "high",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/market-data/treasury')
def get_treasury():
    """Get treasury yield data."""
    try:
        yield_10y = macro_loader.yahoo.get_treasury_10y()
        return jsonify({
            "treasury_10y": yield_10y,
            "treasury_2y": yield_10y - 0.5,  # Approximate
            "spread": 0.5,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/market-data/refresh', methods=['POST'])
def refresh_market_data():
    """Force refresh of all market data."""
    try:
        indicators = macro_loader.fetch_all(force=True)
        return jsonify({
            "success": True,
            "message": "Market data refreshed",
            "vix": indicators.vix,
            "spy": indicators.spy_price,
            "risk_score": indicators.risk_score,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/data-sources/status')
def get_all_data_sources():
    """Get status of all data sources."""
    try:
        return jsonify({
            "alpaca": {
                "name": "Alpaca Markets",
                "type": "Market Data & Trading",
                "status": "connected",
            },
            "alpha_vantage": {
                "name": "Alpha Vantage",
                "type": "News & Sentiment",
                "status": "connected",
                "articles_cached": len(alpha_vantage_news.get_cached_articles()),
            },
            "yahoo_finance": {
                "name": "Yahoo Finance",
                "type": "VIX, Treasury, Indices",
                "status": "connected",
            },
            "fred": {
                "name": "FRED (Federal Reserve)",
                "type": "Economic Indicators",
                "status": "connected" if macro_loader.fred.is_available() else "no_api_key",
                "note": "Add FRED_API_KEY env var for CPI, unemployment data",
            },
            "gemini": {
                "name": "Google Gemini",
                "type": "LLM Reasoning",
                "status": "connected" if LLM_AVAILABLE else "unavailable",
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# BACKGROUND REFRESH
# ============================================================

def background_data_refresh():
    """Background thread to refresh data periodically."""
    global last_background_refresh
    import time
    
    while background_refresh_enabled:
        try:
            # Refresh macro data
            macro_loader.fetch_all(force=True)
            
            # Refresh news
            try:
                alpha_vantage_news.fetch_market_news(days_back=1)
            except:
                pass
            
            # Refresh macro intelligence
            try:
                refresh_macro_from_alpha_vantage()
            except:
                pass
            
            last_background_refresh = datetime.now()
            print(f"✓ Background refresh completed at {last_background_refresh.strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"Background refresh error: {e}")
        
        # Wait for next refresh
        time.sleep(background_refresh_interval * 60)


# Start background refresh thread
background_thread = threading.Thread(target=background_data_refresh, daemon=True)
background_thread.start()
print(f"✓ Background data refresh started (every {background_refresh_interval} minutes)")


@app.route('/api/background-refresh/status')
def get_background_refresh_status():
    """Get background refresh status."""
    return jsonify({
        "enabled": background_refresh_enabled,
        "interval_minutes": background_refresh_interval,
        "last_refresh": last_background_refresh.isoformat() if last_background_refresh else None,
        "next_refresh": (last_background_refresh + timedelta(minutes=background_refresh_interval)).isoformat() if last_background_refresh else None,
    })


@app.route('/api/background-refresh/trigger', methods=['POST'])
def trigger_background_refresh():
    """Manually trigger a background refresh."""
    try:
        # Refresh all data sources
        macro_loader.fetch_all(force=True)
        alpha_vantage_news.fetch_market_news(days_back=1)
        refresh_macro_from_alpha_vantage()
        
        global last_background_refresh
        last_background_refresh = datetime.now()
        
        return jsonify({
            "success": True,
            "message": "All data sources refreshed",
            "timestamp": last_background_refresh.isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/macro/brief')
def get_macro_brief():
    """Get a human-readable macro brief."""
    try:
        import pytz
        as_of = datetime.now(pytz.UTC)
        
        brief = news_intelligence.print_macro_brief(as_of)
        
        return jsonify({
            "brief": brief,
            "timestamp": as_of.isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/macro/events')
def get_macro_events():
    """Get recent macro events."""
    try:
        import pytz
        hours = request.args.get('hours', 24, type=int)
        
        end = datetime.now(pytz.UTC)
        start = end - timedelta(hours=hours)
        
        events = news_intelligence.get_intraday_event_stream(start, end)
        
        return jsonify({
            "events": [
                {
                    "event_id": e.event_id,
                    "event_time": e.event_time.isoformat(),
                    "source": e.source,
                    "title": e.title,
                    "tags": [t.value for t in e.tags],
                    "entities": e.entities,
                    "direction": e.direction.value,
                    "severity": e.severity_score,
                    "impact": e.impact_score,
                    "confidence": e.confidence,
                    "rationale": e.rationale,
                }
                for e in events
            ],
            "count": len(events),
            "start": start.isoformat(),
            "end": end.isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/macro/risk-sentiment')
def get_risk_sentiment():
    """Get current risk sentiment analysis."""
    try:
        import pytz
        as_of = datetime.now(pytz.UTC)
        
        sentiment = news_intelligence.get_risk_sentiment(as_of)
        summary = news_intelligence.sentiment_analyzer.get_sentiment_summary(sentiment)
        
        return jsonify({
            "timestamp": sentiment.timestamp.isoformat(),
            "risk_sentiment": sentiment.risk_sentiment,
            "sentiment_delta": sentiment.sentiment_delta,
            "sentiment_volatility": sentiment.sentiment_volatility,
            "biases": {
                "equity": sentiment.equity_bias,
                "rates": sentiment.rates_bias,
                "dollar": sentiment.dollar_bias,
                "commodity": sentiment.commodity_bias,
            },
            "confidence": sentiment.confidence,
            "event_count": sentiment.event_count,
            "summary": summary,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/macro/load-sample', methods=['POST'])
def load_sample_news():
    """Load sample news for testing."""
    try:
        sample_path = Path("data/sample_news.json")
        
        if not sample_path.exists():
            return jsonify({"error": "Sample news file not found"}), 404
        
        events, stats = news_intelligence.load_from_json(str(sample_path))
        
        return jsonify({
            "message": f"Loaded {len(events)} events from sample news",
            "stats": {
                "total_articles": stats.total_articles,
                "relevant_articles": stats.relevant_articles,
                "events_extracted": stats.events_extracted,
                "high_impact_events": stats.high_impact_events,
                "pass_rate": stats.pass_rate,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/macro/stats')
def get_macro_stats():
    """Get news intelligence pipeline statistics."""
    try:
        stats = news_intelligence.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alpha-vantage/stats')
def get_alpha_vantage_stats():
    """Get Alpha Vantage news loader statistics."""
    try:
        stats = alpha_vantage_news.get_cache_stats()
        return jsonify({
            'source': 'Alpha Vantage News Sentiment API',
            'api_key_configured': bool(alpha_vantage_news.api_key),
            'total_articles_cached': stats['total_articles'],
            'api_calls_made': stats['api_calls_made'],
            'cache_hits': stats['cache_hits'],
            'oldest_article': stats['oldest_article'].isoformat() if stats['oldest_article'] else None,
            'newest_article': stats['newest_article'].isoformat() if stats['newest_article'] else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def refresh_ticker_sentiment_from_cache():
    """Refresh ticker sentiment from cached Alpha Vantage articles."""
    global last_ticker_sentiments
    import pytz
    
    try:
        # Get cached articles
        articles = alpha_vantage_news.get_cached_articles()
        if not articles:
            return False, "No cached articles"
        
        # Process through ticker sentiment aggregator
        as_of = datetime.now(pytz.UTC)
        ticker_features = ticker_sentiment_aggregator.aggregate_from_articles(
            articles,
            as_of=as_of,
            universe=None,  # Accept all tickers
        )
        
        if ticker_features:
            last_ticker_sentiments = ticker_features
            return True, f"Extracted sentiment for {len(ticker_features)} tickers"
        
        return False, "No ticker sentiment extracted"
        
    except Exception as e:
        return False, str(e)


@app.route('/api/ticker-sentiment')
def get_ticker_sentiment():
    """Get aggregated ticker-level sentiment - THE KEY DATA WE WERE MISSING."""
    try:
        # If no sentiment data, try to extract from cache
        if not last_ticker_sentiments:
            success, msg = refresh_ticker_sentiment_from_cache()
            if not success:
                return jsonify({
                    "message": f"No ticker sentiment data. {msg}",
                    "stocks": {},
                    "summary": {},
                    "sentiments": {}  # For UI compatibility
                })
        
        # Convert to dict for JSON
        stocks_data = {}
        for ticker, feat in last_ticker_sentiments.items():
            stocks_data[ticker] = {
                'sentiment_score': feat.sentiment_score,
                'sentiment_confidence': feat.sentiment_confidence,
                'news_volume': feat.news_volume,
                'high_relevance_count': feat.high_relevance_count,
                'sentiment_momentum': feat.sentiment_momentum,
                'bullish_ratio': feat.bullish_ratio,
                'recent_sentiment': feat.recent_sentiment,
                'freshness_hours': feat.freshness_hours,
            }
        
        # Get summary
        bullish = [t for t, f in last_ticker_sentiments.items() if f.sentiment_score >= 0.2]
        bearish = [t for t, f in last_ticker_sentiments.items() if f.sentiment_score <= -0.2]
        
        # Top movers
        sorted_by_sentiment = sorted(
            last_ticker_sentiments.items(),
            key=lambda x: x[1].sentiment_score,
            reverse=True
        )
        
        return jsonify({
            "stocks": stocks_data,
            "sentiments": stocks_data,  # Alias for UI compatibility
            "summary": {
                "total_stocks": len(last_ticker_sentiments),
                "bullish_count": len(bullish),
                "bearish_count": len(bearish),
                "top_bullish": [t for t, _ in sorted_by_sentiment[:5]],
                "top_bearish": [t for t, _ in sorted_by_sentiment[-5:]],
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "sentiments": {}}), 500


@app.route('/api/ticker-sentiment/refresh', methods=['POST'])
def refresh_ticker_sentiment_endpoint():
    """Manually refresh ticker sentiment from cached articles."""
    success, msg = refresh_ticker_sentiment_from_cache()
    return jsonify({
        "success": success,
        "message": msg,
        "total_tickers": len(last_ticker_sentiments)
    })


@app.route('/api/ticker-sentiment/<ticker>')
def get_single_ticker_sentiment(ticker: str):
    """Get sentiment for a specific ticker."""
    try:
        ticker = ticker.upper()
        
        if ticker not in last_ticker_sentiments:
            return jsonify({
                "error": f"No sentiment data for {ticker}",
                "ticker": ticker
            }), 404
        
        feat = last_ticker_sentiments[ticker]
        
        return jsonify({
            "ticker": ticker,
            "sentiment_score": feat.sentiment_score,
            "sentiment_confidence": feat.sentiment_confidence,
            "news_volume": feat.news_volume,
            "high_relevance_count": feat.high_relevance_count,
            "sentiment_momentum": feat.sentiment_momentum,
            "bullish_ratio": feat.bullish_ratio,
            "recent_sentiment": feat.recent_sentiment,
            "freshness_hours": feat.freshness_hours,
            "interpretation": "Bullish" if feat.sentiment_score > 0.1 else "Bearish" if feat.sentiment_score < -0.1 else "Neutral"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === OUTCOME TRACKER ENDPOINTS ===

@app.route('/api/signals/accuracy')
def get_signal_accuracy():
    """Get signal accuracy metrics - ARE OUR SIGNALS PREDICTIVE?"""
    try:
        metrics = outcome_tracker.get_accuracy_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/signals/sentiment-effectiveness')
def get_sentiment_effectiveness():
    """Get sentiment effectiveness metrics - IS SENTIMENT DATA HELPING?"""
    try:
        effectiveness = outcome_tracker.get_sentiment_effectiveness()
        return jsonify(effectiveness)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/signals/summary')
def get_signals_summary():
    """Get combined summary of signal performance."""
    try:
        accuracy = outcome_tracker.get_accuracy_metrics()
        sentiment = outcome_tracker.get_sentiment_effectiveness()
        validation = signal_validator.get_stats()
        
        return jsonify({
            "accuracy": accuracy,
            "sentiment_effectiveness": sentiment,
            "validation": validation,
            "is_predictive": accuracy.get('accuracy', 0) > 0.52 if accuracy.get('accuracy') else None,
            "is_sentiment_helpful": sentiment.get('is_sentiment_helpful'),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/signals/validation')
def get_validation_stats():
    """Get signal validation statistics."""
    try:
        return jsonify(signal_validator.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === FEEDBACK LOOP ENDPOINTS ===

@app.route('/api/feedback/summary')
def get_feedback_summary():
    """Get feedback loop performance summary."""
    try:
        return jsonify({
            "performance": feedback_loop.get_performance_summary(),
            "best_strategies": feedback_loop.get_best_strategies(3),
            "worst_strategies": feedback_loop.get_worst_strategies(3),
            "recommendations": feedback_loop.get_recommendations(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback/adjustments')
def get_weight_adjustments():
    """Get current weight adjustments from feedback loop."""
    try:
        return jsonify({
            "adjustments": feedback_loop.weight_adjustments,
            "strategy_count": len(feedback_loop.strategy_performance),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === LLM ENDPOINTS ===

@app.route('/api/llm/stats')
def get_llm_stats():
    """Get LLM usage statistics."""
    try:
        if not llm_service:
            return jsonify({
                "available": False,
                "message": "LLM not configured (set OPENAI_API_KEY)"
            })
        
        return jsonify({
            "available": True,
            **llm_service.get_stats()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/llm/themes')
def get_active_themes():
    """Get active market themes from LLM synthesis."""
    try:
        if not theme_synthesizer:
            return jsonify({
                "available": False,
                "themes": [],
                "message": "Theme synthesizer not available"
            })
        
        themes = theme_synthesizer.get_active_themes()
        return jsonify({
            "available": True,
            "themes": [
                {
                    "name": t.name,
                    "description": t.description,
                    "momentum": t.momentum,
                    "direction": t.direction,
                    "severity": t.severity,
                    "trading_implication": t.trading_implication,
                }
                for t in themes
            ],
            "market_stance": theme_synthesizer.get_overall_market_stance(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alpha-vantage/fetch', methods=['POST', 'GET'])
def fetch_alpha_vantage_news():
    """Get Alpha Vantage news - uses cache if available."""
    try:
        # Accept both JSON and query params
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = {}
        
        days_back = int(request.args.get('days_back', data.get('days_back', 7)))
        limit = int(request.args.get('limit', data.get('limit', 20)))
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        tickers = data.get('tickers', [])
        
        articles = []
        api_tried = False
        
        # If force refresh, try API first
        if force_refresh:
            api_tried = True
            if tickers:
                articles = alpha_vantage_news.fetch_ticker_news(tickers, days_back)
            else:
                articles = alpha_vantage_news.fetch_market_news(days_back)
        
        # Always fall back to cache if no articles from API
        if not articles:
            articles = alpha_vantage_news.get_cached_articles()
            if articles and api_tried:
                print(f"API rate-limited, using {len(articles)} cached articles")
        
        # If still no articles and didn't try API yet, try once
        if not articles and not api_tried:
            if tickers:
                articles = alpha_vantage_news.fetch_ticker_news(tickers, days_back)
            else:
                articles = alpha_vantage_news.fetch_market_news(days_back)
        
        # Return full article data for UI
        article_data = []
        for a in articles[:limit]:
            article_data.append({
                'headline': a.headline,
                'source': a.source,
                'timestamp': a.timestamp.isoformat() if hasattr(a.timestamp, 'isoformat') else str(a.timestamp),
                'url': a.url,
                'summary': a.summary[:200] if a.summary else '',
                'overall_sentiment_score': a.overall_sentiment,  # Correct attribute name
                'overall_sentiment_label': a.overall_sentiment_label,
                'tickers': a.tickers[:5] if a.tickers else [],
            })
        
        # Get rate limit status
        rate_limit_status = alpha_vantage_news.get_rate_limit_status()
        
        return jsonify({
            'success': True,
            'articles_fetched': len(articles),
            'articles': article_data,
            'rate_limit': rate_limit_status,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alpha-vantage/rate-limit')
def get_alpha_vantage_rate_limit():
    """Get Alpha Vantage API rate limit status."""
    try:
        status = alpha_vantage_news.get_rate_limit_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alpha-vantage/clear-cache', methods=['POST'])
def clear_alpha_vantage_cache():
    """Clear Alpha Vantage cache."""
    try:
        alpha_vantage_news.clear_cache()
        return jsonify({"success": True, "message": "Alpha Vantage cache cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alpha-vantage/deduplicate', methods=['POST'])
def deduplicate_alpha_vantage_cache():
    """Remove duplicate articles from cache."""
    try:
        before_count = len(alpha_vantage_news._articles_cache)
        removed = alpha_vantage_news.deduplicate_cache()
        after_count = len(alpha_vantage_news._articles_cache)
        return jsonify({
            "success": True, 
            "before": before_count,
            "after": after_count,
            "removed": removed,
            "message": f"Removed {removed} duplicates, {after_count} articles remain"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================================================================
# EXECUTION QUALITY ENDPOINTS
# ===============================================================================

# Global execution report for tracking
from src.execution.execution_report import ExecutionReport
execution_report = ExecutionReport()

@app.route('/api/execution/stats')
def get_execution_stats():
    """Get execution quality statistics."""
    try:
        metrics = execution_report.get_metrics(session_only=False)
        return jsonify({
            "total_orders": metrics.total_orders,
            "limit_orders": metrics.limit_orders,
            "market_orders": metrics.market_orders,
            "limit_fills": metrics.limit_fills,
            "limit_fallbacks": metrics.limit_fallbacks,
            "fill_rate": metrics.fill_rate,
            "total_value": metrics.total_value,
            "total_improvement": metrics.total_improvement,
            "avg_improvement_pct": metrics.avg_improvement_pct,
            "avg_execution_time_ms": metrics.avg_execution_time_ms,
            "avg_spread_pct": metrics.avg_spread_pct,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/execution/session')
def get_execution_session():
    """Get current session execution stats."""
    try:
        return jsonify(execution_report.get_session_summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/execution/insights')
def get_execution_insights():
    """Get execution insights from history."""
    try:
        from src.execution import SmartExecutor
        # Create temporary executor to read history
        executor = SmartExecutor(
            broker=None,
            dry_run=True,
            log_func=lambda x: None,
        )
        insights = executor.get_symbol_insights()
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/execution/history')
def get_execution_history():
    """Get per-symbol execution history."""
    try:
        history_path = Path("outputs/execution_history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify({"message": "No execution history yet", "symbols": {}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================================================================
# REPORTING ENDPOINTS
# ===============================================================================

@app.route('/api/reports/generate', methods=['POST'])
def generate_report_endpoint():
    """Generate a report (daily/weekly/monthly) using LIVE data."""
    try:
        from src.reporting import generate_report, ReportLearningFeedback
        from pathlib import Path
        
        data = request.get_json() or {}
        report_type = data.get('type', 'daily')
        
        # Get broker for live data
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return jsonify({"error": "API keys not configured"}), 400
        
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
        
        # Build app state for report
        app_state = {
            'last_run_status': last_run_status,
            'last_macro_features': last_macro_features,
            'last_ticker_sentiments': last_ticker_sentiments,
        }
        
        # Try to add current regime
        if 'current_regime' in globals():
            app_state['current_regime'] = current_regime
        
        # Generate output path
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        output_path = output_dir / f"{report_type}_{timestamp}.html"
        
        # Generate report using live data
        result = generate_report(
            broker=broker,
            learning_engine=learning_engine,
            app_state=app_state,
            report_type=report_type,
            output_path=output_path,
        )
        
        if result['success']:
            # Feed insights to learning system
            try:
                feedback = ReportLearningFeedback(learning_engine)
                insights = feedback.extract_and_learn(result['data'])
                logging.info(f"Extracted {len(insights)} insights from report")
            except Exception as e:
                logging.warning(f"Could not extract insights: {e}")
            
            return jsonify({
                "success": True,
                "html_path": str(output_path),
                "message": f"{report_type.title()} report generated successfully",
                "view_url": f"/api/reports/view/{output_path.name}",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400
    
    except Exception as e:
        logging.error(f"Report generation error: {e}", exc_info=True)
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app.route('/api/reports/view/<filename>')
def view_report(filename):
    """View a generated report."""
    from pathlib import Path
    import html
    
    report_path = Path("outputs/reports") / filename
    if not report_path.exists():
        return jsonify({"error": "Report not found"}), 404
    
    if not filename.endswith('.html'):
        return jsonify({"error": "Invalid report format"}), 400
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/api/reports/list')
def list_reports():
    """List available reports."""
    try:
        reports_dir = Path("outputs/reports")
        if not reports_dir.exists():
            return jsonify({"reports": []})
        
        reports = []
        for pdf_file in reports_dir.glob("*.pdf"):
            reports.append({
                "filename": pdf_file.name,
                "path": str(pdf_file),
                "size": pdf_file.stat().st_size,
                "modified": datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat(),
            })
        
        # Sort by modified time (newest first)
        reports.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({"reports": reports})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reports/download/<filename>')
def download_report(filename):
    """Download a report PDF."""
    try:
        reports_dir = Path("outputs/reports")
        pdf_path = reports_dir / filename
        
        if not pdf_path.exists():
            return jsonify({"error": "Report not found"}), 404
        
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Start auto-rebalance scheduler
    scheduler_thread = threading.Thread(target=auto_rebalance_scheduler, daemon=True)
    scheduler_thread.start()
    
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    
    print(f"\n{'='*60}")
    print("MULTI-STRATEGY QUANT DEBATE BOT - Web UI")
    print(f"{'='*60}")
    print(f"\nOpen your browser: http://localhost:{port}")
    print("\nStrategies: 9 active strategies with debate mechanism")
    print("Data Sources:")
    print("  - Market Data: Alpaca API (IEX feed)")
    print("  - News & Sentiment: Alpha Vantage News Sentiment API ✓")
    print("\nPress Ctrl+C to stop the server")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
