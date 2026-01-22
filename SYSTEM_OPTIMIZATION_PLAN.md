# System Optimization Plan: A-to-Z Critical Analysis

## Executive Summary

After comprehensive analysis of the entire trading system flow, I've identified **12 critical gaps** and **9 optimization opportunities**. This document provides a structured, prioritized plan to transform the system from a prototype into a production-grade HFT-lite platform.

---

## PART 1: COMPLETE FLOW ANALYSIS

### Current Flow (Traced A-to-Z)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRIGGER: API call to /api/run OR scheduled auto-rebalance                  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ A. DATA INGESTION (CURRENT: 25-40 seconds)                                  │
│                                                                             │
│   A1. Market Data (Alpaca)           → 15-20s (SEQUENTIAL, cached)          │
│       • 300+ stocks × 300 days daily bars                                   │
│       • Using price_cache for optimization                                  │
│       • ⚠️ ISSUE: No intraday bars (15-min) for HFT strategies             │
│                                                                             │
│   A2. News Data (Alpha Vantage)      → 10-25s (rate limited)               │
│       • Market news + ticker news                                           │
│       • Sentiment extraction                                                │
│       • ⚠️ ISSUE: Rate limited to 25/day, often using stale cache          │
│                                                                             │
│   A3. Macro Data (News Intelligence) → 2-5s                                 │
│       • Geopolitical risk indices                                          │
│       • Inflation/growth indicators                                         │
│       • ✅ Working well                                                     │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ B. FEATURE ENGINEERING (CURRENT: 1-2 seconds)                               │
│                                                                             │
│   B1. Price Features                                                        │
│       • Returns: 1d, 5d, 21d, 63d, 126d, 252d ✅                            │
│       • Volatility: 21d, 63d rolling ✅                                     │
│       • Moving Averages: 20d, 50d, 200d ✅                                  │
│       • ⚠️ MISSING: Intraday returns (15m, 30m, 60m)                        │
│       • ⚠️ MISSING: VWAP deviation                                         │
│       • ⚠️ MISSING: Volume ratio (current vs average)                      │
│                                                                             │
│   B2. Sentiment Features                                                    │
│       • Ticker sentiment scores ✅                                          │
│       • News recency weighting ✅                                           │
│       • ⚠️ ISSUE: Not used by all strategies                               │
│                                                                             │
│   B3. Regime Detection                                                      │
│       • Market trend (up/down/neutral) ✅                                   │
│       • Volatility regime ✅                                                │
│       • ⚠️ ISSUE: Regime doesn't dynamically switch trading mode           │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ C. STRATEGY SIGNAL GENERATION (CURRENT: <1 second)                          │
│                                                                             │
│   C1. Intraday Strategies (6 strategies)                                   │
│       • IntradayMomentum: Uses fallback_to_daily ⚠️                        │
│       • VWAPReversion: Uses fallback_to_ma ⚠️                              │
│       • VolumeSpike: Uses returns_1d ⚠️                                    │
│       • RelativeStrengthIntraday: Uses returns_1d ⚠️                       │
│       • OpeningRangeBreakout: Uses fallback_to_daily_range ⚠️              │
│       • QuickMeanReversion: Uses returns_1d ⚠️                             │
│       ⚠️ CRITICAL: ALL are using DAILY data as fallback!                   │
│                                                                             │
│   C2. Long/Short Strategies (4 strategies)                                 │
│       • CS_Momentum_LS ✅                                                   │
│       • TS_Momentum_LS ✅                                                   │
│       • MeanReversion_LS ✅                                                 │
│       • QualityValue_LS ✅                                                  │
│       ✅ Working: Generating shorts, ensemble preserving them              │
│                                                                             │
│   C3. Futures Strategies (3 strategies)                                    │
│       • Futures_Carry (ETF proxy) ⚠️ Often fails                           │
│       • Futures_Trend (ETF proxy) ⚠️ Often fails                           │
│       • Futures_Macro (ETF proxy) ⚠️ Often fails                           │
│       ⚠️ ISSUE: Frequent errors due to missing ETF data                    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ D. STRATEGY DEBATE & SCORING (CURRENT: 2-8 seconds)                         │
│                                                                             │
│   D1. Initial Scoring (DebateEngine)                                       │
│       • Alpha score ✅                                                      │
│       • Regime fit score ✅                                                 │
│       • Diversification score ✅                                            │
│       • Drawdown score ✅                                                   │
│                                                                             │
│   D2. Adversarial Debate                                                   │
│       • Support arguments (LLM) ✅                                          │
│       • Attack arguments (LLM) ✅                                           │
│       • ParallelDebateEngine: 28 LLM calls in 2-3s ✅                       │
│       • Fast debate (rule-based) for ultra-fast mode ✅                    │
│                                                                             │
│   D3. Historical Learning Integration                                       │
│       • Debate scores blended with learned weights                         │
│       • ⚠️ ISSUE: Learning influence only 30% (too weak)                   │
│       • ⚠️ ISSUE: Regime-specific weights underutilized                    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ E. ENSEMBLE & WEIGHT COMBINATION (CURRENT: 1-2 seconds)                     │
│                                                                             │
│   E1. Weighted Vote                                                         │
│       • Combines strategy signals by debate score ✅                        │
│       • L/S strategy boost for shorts (2x in conflicts) ✅                 │
│                                                                             │
│   E2. Signal Conflict Resolution                                           │
│       • Nets long vs short signals ✅                                       │
│       • L/S shorts now preserved ✅                                         │
│                                                                             │
│   E3. Constraints Applied                                                   │
│       • Position limits (15%) ✅                                            │
│       • Sector limits (30%) ✅                                              │
│       • Leverage limit (1.0x) ⚠️ Could be higher with shorts               │
│       • Turnover limit (50%) ✅                                             │
│       • Vol targeting ✅                                                    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ F. RISK MANAGEMENT (CURRENT: <1 second)                                     │
│                                                                             │
│   F1. Pre-Trade Checks                                                      │
│       • Max gross exposure (200%) ✅                                        │
│       • Net exposure range (-30% to +100%) ✅                               │
│       • Max single position (15%) ✅                                        │
│                                                                             │
│   F2. GAPS                                                                  │
│       • ⚠️ NO real-time risk monitoring (only at rebalance)                │
│       • ⚠️ NO intraday drawdown circuit breaker                            │
│       • ⚠️ NO VIX-based automatic position reduction                       │
│       • ⚠️ NO correlation-based concentration alerts                       │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ G. TRANSACTION COST ANALYSIS (CURRENT: 5-10 seconds)                        │
│                                                                             │
│   G1. Cost Estimation                                                       │
│       • Spread costs (estimated, not real) ⚠️                              │
│       • Slippage by liquidity tier ✅                                       │
│       • Market impact ✅                                                    │
│       • VIX-adjusted multipliers ✅                                         │
│                                                                             │
│   G2. Benefit/Cost Filter                                                   │
│       • Min ratio threshold: 1.5x ✅                                        │
│       • Skips unprofitable trades ✅                                        │
│                                                                             │
│   G3. GAPS                                                                  │
│       • ⚠️ Using ESTIMATED spreads (0.05%), not REAL bid-ask              │
│       • ⚠️ No real-time quote fetching                                     │
│       • ⚠️ No learning from actual vs estimated costs                      │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ H. TRADE EXECUTION (CURRENT: 5-10 seconds)                                  │
│                                                                             │
│   H1. SmartExecutor                                                         │
│       • Prioritizes high-conviction trades ✅                               │
│       • Position sizing ✅                                                  │
│                                                                             │
│   H2. Order Types                                                           │
│       • Market orders ✅                                                    │
│       • Limit orders available ✅                                           │
│       • ⚠️ NO TWAP/VWAP for large orders                                   │
│                                                                             │
│   H3. GAPS                                                                  │
│       • ⚠️ Fire-and-forget orders (no monitoring)                          │
│       • ⚠️ No partial fill handling                                        │
│       • ⚠️ No order status tracking                                        │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ I. LEARNING & FEEDBACK (CURRENT: <1 second)                                 │
│                                                                             │
│   I1. Outcome Tracking                                                      │
│       • Records signals with predictions ✅                                 │
│       • Tracks actual outcomes ✅                                           │
│       • Calculates accuracy ✅                                              │
│                                                                             │
│   I2. Adaptive Weights                                                      │
│       • EMA performance tracking ✅                                         │
│       • UCB1 exploration bonus ✅                                           │
│       • Regime-specific weights ✅                                          │
│                                                                             │
│   I3. GAPS                                                                  │
│       • ⚠️ Learning influence is only 30% (too weak)                       │
│       • ⚠️ Needs 50+ trades before meaningful influence                    │
│       • ⚠️ Pattern learner recommendations not strongly acted upon         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PART 2: CRITICAL GAPS IDENTIFIED

### 🔴 CRITICAL (Must Fix)

| # | Gap | Location | Impact | Fix Complexity |
|---|-----|----------|--------|----------------|
| 1 | **Intraday strategies use daily data** | `src/strategies/intraday.py` | All 6 intraday strategies are using fallback mode with daily returns instead of real 15-min bars | HIGH |
| 2 | **Sequential data fetching (40s)** | `app.py:456-650` | Bottleneck: price, news, macro fetched sequentially | MEDIUM |
| 3 | **Learning influence too weak (30%)** | `LearningEngine.__init__` | Past performance barely influences decisions | LOW |

### 🟡 HIGH PRIORITY

| # | Gap | Location | Impact | Fix Complexity |
|---|-----|----------|--------|----------------|
| 4 | **No real-time risk monitoring** | Missing | Risk only checked at rebalance time, not continuously | MEDIUM |
| 5 | **Futures strategies often fail** | `src/strategies/futures.py` | Missing ETF data causes errors | LOW |
| 6 | **Regime doesn't drive mode switching** | `app.py:745` | Trading mode is static, doesn't adapt | MEDIUM |
| 7 | **Using estimated spreads, not real** | `TransactionCostModel` | Cost estimates may be 50%+ off | MEDIUM |

### 🟢 MEDIUM PRIORITY

| # | Gap | Location | Impact | Fix Complexity |
|---|-----|----------|--------|----------------|
| 8 | **Fire-and-forget order execution** | `smart_executor.py` | No fill tracking or partial fill handling | MEDIUM |
| 9 | **No TWAP/VWAP for large orders** | `smart_executor.py` | Higher market impact on large trades | HIGH |
| 10 | **Sentiment not in all strategies** | Multiple | Underutilized data source | LOW |
| 11 | **No drawdown circuit breaker** | Missing | No automatic de-risking during crashes | MEDIUM |
| 12 | **VIX doesn't adjust position sizing** | Missing | Should reduce exposure in high-VIX | LOW |

---

## PART 3: OPTIMIZATION PLAN

### PHASE 1: CRITICAL FIXES (This Week)

#### 1.1 Add Real Intraday Data Fetching

**Problem:** All 6 intraday strategies fall back to daily data.

**Solution:** Add 15-minute bar fetching from Alpaca.

```python
# ADD to src/data/market_data.py

def load_intraday_bars(
    self,
    symbols: List[str],
    timeframe: str = "15Min",
    days_back: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch intraday bars from Alpaca.
    
    Args:
        symbols: List of symbols
        timeframe: "1Min", "5Min", "15Min", "30Min", "1Hour"
        days_back: How many days of intraday data
    
    Returns:
        Dict of symbol -> DataFrame with OHLCV columns
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    
    timeframe_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    }
    
    client = StockHistoricalDataClient(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY')
    )
    
    end = datetime.now(pytz.UTC)
    start = end - timedelta(days=days_back)
    
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe_map[timeframe],
        start=start,
        end=end,
    )
    
    bars = client.get_stock_bars(request)
    
    result = {}
    for symbol in symbols:
        if symbol in bars.data:
            df = pd.DataFrame([
                {
                    'timestamp': b.timestamp,
                    'open': b.open,
                    'high': b.high,
                    'low': b.low,
                    'close': b.close,
                    'volume': b.volume,
                    'vwap': b.vwap,
                }
                for b in bars.data[symbol]
            ])
            df.set_index('timestamp', inplace=True)
            result[symbol] = df
    
    return result
```

**Add to Features dataclass:**

```python
# ADD to src/data/feature_store.py Features dataclass

# Intraday features (for HFT-lite strategies)
intraday_returns: Dict[str, float] = field(default_factory=dict)  # Last 15-30 min return
volume_ratio: Dict[str, float] = field(default_factory=dict)  # Current vol vs average
vwap: Dict[str, float] = field(default_factory=dict)  # Current VWAP
opening_high: Dict[str, float] = field(default_factory=dict)  # First 30-min high
opening_low: Dict[str, float] = field(default_factory=dict)  # First 30-min low
```

#### 1.2 Implement Parallel Data Fetching

**Problem:** Data fetching takes 40 seconds (sequential).

**Solution:** Fetch all data sources in parallel using asyncio.

```python
# ADD to app.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

async def fetch_all_data_parallel(broker, config, alpha_vantage_news, price_cache, end_date):
    """
    Fetch price, news, and macro data in parallel.
    Reduces total time from 40s to ~20s.
    """
    executor = ThreadPoolExecutor(max_workers=3)
    loop = asyncio.get_event_loop()
    
    # Define fetch functions
    def fetch_prices():
        cached = price_cache.get_prices(config.UNIVERSE, days=300, end_date=end_date)
        if cached is not None:
            return cached
        return broker.get_historical_bars(config.UNIVERSE, days=300)
    
    def fetch_news():
        return alpha_vantage_news.fetch_market_news(days_back=7)
    
    def fetch_macro():
        # Already fast, but include for completeness
        return news_intelligence.get_cached_macro_features()
    
    # Execute in parallel
    price_future = loop.run_in_executor(executor, fetch_prices)
    news_future = loop.run_in_executor(executor, fetch_news)
    macro_future = loop.run_in_executor(executor, fetch_macro)
    
    # Wait for all
    price_data, news_articles, macro_features = await asyncio.gather(
        price_future, news_future, macro_future
    )
    
    return price_data, news_articles, macro_features
```

#### 1.3 Increase Learning Influence

**Problem:** Learned weights only influence 30% of decisions.

**Solution:** Increase to 50% after 30 trades, 70% after 100 trades.

```python
# MODIFY src/learning/learning_engine.py

def __init__(
    self,
    strategy_names: List[str],
    outputs_dir: str = "outputs",
    learning_influence: float = 0.3,  # Starting influence
):
    # ... existing code ...
    self.base_learning_influence = learning_influence
    
def get_adaptive_learning_influence(self) -> float:
    """
    Dynamically adjust learning influence based on data collected.
    
    More data = more trust in learning = higher influence.
    """
    total_trades = self.trade_memory.get_statistics().get('total_trades', 0)
    
    if total_trades < 10:
        return 0.2  # Low influence, still learning
    elif total_trades < 30:
        return 0.3  # Moderate influence
    elif total_trades < 100:
        return 0.5  # High influence
    else:
        return 0.7  # Strong influence - trust the learning
```

---

### PHASE 2: HIGH PRIORITY FIXES (Next Week)

#### 2.1 Add Real-Time Risk Monitoring

```python
# CREATE src/risk/realtime_monitor.py

import threading
import time
from datetime import datetime
import logging

class RealtimeRiskMonitor:
    """
    Background thread that continuously monitors portfolio risk.
    
    Triggers automatic actions when thresholds are breached:
    - Drawdown > 5%: Alert
    - Drawdown > 10%: Reduce exposure by 50%
    - VIX > 35: Halt new trades, reduce exposure
    """
    
    def __init__(
        self,
        broker,
        check_interval: int = 60,  # Check every 60 seconds
        max_drawdown: float = 0.10,
        vix_halt_threshold: float = 35.0,
    ):
        self.broker = broker
        self.check_interval = check_interval
        self.max_drawdown = max_drawdown
        self.vix_halt_threshold = vix_halt_threshold
        
        self.peak_equity = 0.0
        self.is_running = False
        self.halt_trading = False
        self.alerts = []
        
        self._thread = None
    
    def start(self):
        """Start the background monitoring thread."""
        self.is_running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logging.info("Real-time risk monitor started")
    
    def stop(self):
        """Stop the monitoring thread."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._check_risk()
            except Exception as e:
                logging.error(f"Risk monitor error: {e}")
            
            time.sleep(self.check_interval)
    
    def _check_risk(self):
        """Perform risk checks."""
        account = self.broker.get_account()
        equity = account['equity']
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate drawdown
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Drawdown alert
        if drawdown > 0.05:
            self._alert(f"WARNING: Drawdown at {drawdown:.1%}")
        
        # Drawdown trigger
        if drawdown > self.max_drawdown:
            self._alert(f"CRITICAL: Drawdown {drawdown:.1%} exceeds limit {self.max_drawdown:.1%}")
            self._reduce_exposure(0.5)  # Reduce by 50%
        
        # VIX check
        vix = self._get_vix()
        if vix > self.vix_halt_threshold:
            self._alert(f"HALT: VIX at {vix:.1f} exceeds threshold {self.vix_halt_threshold}")
            self.halt_trading = True
    
    def _reduce_exposure(self, reduction_pct: float):
        """Reduce portfolio exposure by selling positions."""
        positions = self.broker.get_positions()
        
        for pos in positions:
            target_qty = int(pos['qty'] * (1 - reduction_pct))
            if target_qty < pos['qty']:
                sell_qty = pos['qty'] - target_qty
                try:
                    self.broker.submit_order(
                        symbol=pos['symbol'],
                        side='sell',
                        quantity=sell_qty,
                        order_type='market'
                    )
                    logging.info(f"Risk reduction: Sold {sell_qty} shares of {pos['symbol']}")
                except Exception as e:
                    logging.error(f"Failed to reduce {pos['symbol']}: {e}")
    
    def _alert(self, message: str):
        """Log and store alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
        }
        self.alerts.append(alert)
        logging.warning(f"RISK ALERT: {message}")
    
    def _get_vix(self) -> float:
        """Get current VIX level."""
        # This would fetch real VIX - simplified here
        return 18.0  # Placeholder
```

#### 2.2 Fix Futures Strategies

**Problem:** Futures strategies fail due to missing ETF data.

**Solution:** Add graceful degradation and ensure ETFs are in the universe.

```python
# MODIFY config.py

# Ensure these ETF proxies are always in the universe
ETF_PROXIES = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD', 'USO', 'DBC']

UNIVERSE = list(set([
    # ... existing stocks ...
] + ETF_PROXIES))
```

```python
# MODIFY src/strategies/futures.py

def generate_signals(self, features, t):
    """Generate signals with graceful degradation."""
    try:
        # ... existing logic ...
    except Exception as e:
        logger.warning(f"Futures strategy {self.name} failed: {e}, returning empty signals")
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights={},
            expected_return=0.0,
            risk_estimate=0.0,
            confidence=0.0,
            explanation={"error": str(e), "fallback": "empty signals"},
        )
```

#### 2.3 Dynamic Mode Switching Based on Regime

```python
# MODIFY app.py where trading_mode_setting is used

def get_dynamic_trading_mode(features, vix_level: float) -> str:
    """
    Dynamically determine trading mode based on market conditions.
    
    High VIX / volatile = intraday (quick in/out)
    Low VIX / trending = position (hold longer)
    Mixed = hybrid (blend both)
    """
    regime = features.regime
    
    # VIX-based switching
    if vix_level > 30:
        return "intraday"  # High vol = quick trades
    elif vix_level < 15:
        if regime and "trending" in regime.description.lower():
            return "position"  # Low vol + trend = hold
    
    # Default to hybrid
    return "hybrid"  # Use both strategy sets

def create_strategies_dynamic(features, vix_level: float):
    """Create strategies based on dynamic mode."""
    mode = get_dynamic_trading_mode(features, vix_level)
    
    if mode == "intraday":
        return create_intraday_strategies()  # Only intraday
    elif mode == "position":
        return create_position_strategies()  # Only position
    else:
        # Hybrid: use both with blending
        intraday = create_intraday_strategies()
        position = create_position_strategies()
        # Weight intraday more in volatile conditions
        return intraday + position
```

---

### PHASE 3: MEDIUM PRIORITY (Week 3)

#### 3.1 Add Real Bid-Ask Spread Fetching

```python
# ADD to broker_alpaca.py

def get_current_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
    """
    Get real-time bid-ask quotes from Alpaca.
    
    Returns:
        Dict of symbol -> {'bid': float, 'ask': float, 'spread_pct': float}
    """
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import StockLatestQuoteRequest
    
    client = StockHistoricalDataClient(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY')
    )
    
    request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
    quotes = client.get_stock_latest_quote(request)
    
    result = {}
    for symbol in symbols:
        if symbol in quotes:
            q = quotes[symbol]
            bid = q.bid_price
            ask = q.ask_price
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else q.ask_price
            spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 0.05
            
            result[symbol] = {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread_pct': spread_pct,
            }
    
    return result
```

#### 3.2 Add Order Status Monitoring

```python
# MODIFY src/execution/smart_executor.py

def execute_with_monitoring(self, order, max_wait_seconds: int = 30):
    """
    Execute order and monitor for fills.
    
    Returns:
        Dict with fill status and details
    """
    order_id = self.broker.submit_order(order)
    
    for _ in range(max_wait_seconds):
        status = self.broker.get_order(order_id)
        
        if status['status'] == 'filled':
            return {
                'success': True,
                'order_id': order_id,
                'filled_qty': status['filled_qty'],
                'avg_price': status['filled_avg_price'],
                'fill_time': status['filled_at'],
            }
        
        elif status['status'] == 'partially_filled':
            logging.info(f"Partial fill: {status['filled_qty']}/{status['qty']}")
        
        elif status['status'] in ['cancelled', 'rejected']:
            return {
                'success': False,
                'order_id': order_id,
                'reason': status['status'],
            }
        
        time.sleep(1)
    
    # Timeout - cancel and retry with market order
    self.broker.cancel_order(order_id)
    
    if status.get('filled_qty', 0) == 0:
        # No fills - submit as market
        return self.execute_market_order(order)
    else:
        # Partial fills - accept what we got
        return {
            'success': True,
            'order_id': order_id,
            'filled_qty': status['filled_qty'],
            'partial': True,
        }
```

---

## PART 4: IMPLEMENTATION PRIORITY

### Week 1: Critical Fixes
1. ✅ Add intraday bar fetching to `market_data.py`
2. ✅ Add intraday features to `feature_store.py` 
3. ✅ Update intraday strategies to use real intraday data
4. ✅ Implement parallel data fetching
5. ✅ Increase learning influence dynamically

### Week 2: High Priority
6. ✅ Add real-time risk monitor (background thread)
7. ✅ Fix futures strategies (graceful degradation)
8. ✅ Implement dynamic mode switching

### Week 3: Medium Priority
9. ✅ Add real bid-ask spread fetching
10. ✅ Add order status monitoring
11. ✅ Add drawdown circuit breaker to risk monitor

### Week 4: Polish
12. ✅ Add TWAP execution for large orders
13. ✅ Add sentiment to all strategies
14. ✅ Comprehensive testing

---

## PART 5: SUCCESS METRICS

After implementing these optimizations, we should see:

| Metric | Current | Target |
|--------|---------|--------|
| Rebalance time | 40-60s | <20s |
| Intraday strategy accuracy | ~50% (random) | >55% |
| Learning influence | 30% | 50-70% |
| Order fill rate | Unknown | >98% |
| Risk monitoring | Manual | Continuous |
| Cost estimation error | ~50% | <20% |

---

## CONCLUSION

The system has a solid foundation but is operating in a **degraded mode** for intraday trading because:

1. **Intraday strategies have no intraday data** - They're using daily returns, making them essentially random
2. **Data fetching is slow** - 40s is too long for HFT-lite
3. **Learning is too weak** - 30% influence means the system barely learns from mistakes

With the proposed fixes, the system will:
- Use **real 15-minute bars** for intraday decisions
- Fetch data **2x faster** with parallel loading
- **Learn aggressively** from past performance
- **Monitor risk continuously** with automatic de-risking
- **Track order fills** and handle partial fills

This transforms the system from a prototype to a production-grade HFT-lite platform.
