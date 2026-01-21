# 🔬 Mini Quant Fund - Deep Analysis & Maximum Value Extraction Plan

## Critical Self-Assessment

Before proceeding, I must acknowledge gaps in my previous plan:

| Gap | Why It Matters |
|-----|----------------|
| No quantitative alpha targets | "Better" is meaningless without numbers |
| No validation protocol | How do we know LLM is correct before trading? |
| No feedback loops | System doesn't learn from its mistakes |
| No backtesting framework for LLM | Can't prove value without historical test |
| Missing integration flow | LLM output → Strategy → Trade is unclear |
| No monitoring/alerting | Won't know when system breaks |
| Underutilized data sources | Not extracting maximum from Alpha Vantage |

This document addresses ALL of these gaps.

---

## Part 1: What Are We Actually Trying to Achieve?

### The Core Question

**What is the measurable outcome we want?**

Current System Performance (Baseline):
```
Sharpe Ratio: Unknown (no backtested track record)
Win Rate: 0% (no trades executed yet)
Signal Quality: ~70% accuracy on relevance/taxonomy
Regime Detection: Rule-based, untested
```

Target Performance (After Enhancement):
```
Sharpe Ratio: > 1.0 (risk-adjusted returns)
Win Rate: > 55% (slightly better than random)
Signal Quality: > 90% accuracy on all classifications
Regime Detection: LLM-enhanced, validated
Information Ratio: > 0.5 (alpha per unit of tracking error)
```

### The Value Chain

Every component must contribute to this chain:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VALUE CREATION CHAIN                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RAW DATA              SIGNALS              DECISIONS            RETURNS    │
│                                                                              │
│  ┌─────────┐         ┌─────────┐          ┌─────────┐         ┌─────────┐  │
│  │ Alpha   │   →     │ Macro   │    →     │Strategy │   →     │ P&L     │  │
│  │ Vantage │         │ Indices │          │ Weights │         │         │  │
│  │ News    │         │         │          │         │         │         │  │
│  └─────────┘         └─────────┘          └─────────┘         └─────────┘  │
│       │                   │                    │                    │       │
│       │                   │                    │                    │       │
│  ┌─────────┐         ┌─────────┐          ┌─────────┐         ┌─────────┐  │
│  │ Alpaca  │   →     │ Price   │    →     │Position │   →     │ Risk-   │  │
│  │ Market  │         │Features │          │ Sizing  │         │Adjusted │  │
│  │ Data    │         │         │          │         │         │ Alpha   │  │
│  └─────────┘         └─────────┘          └─────────┘         └─────────┘  │
│                                                                              │
│  QUESTION: At each step, are we extracting MAXIMUM value?                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Deep Audit of Current Data Utilization

### Alpha Vantage Data - What We Get vs What We Use

| Data Point | Available | Currently Used | Potential Use |
|------------|-----------|----------------|---------------|
| `title` | ✅ | ✅ For relevance | ✅ |
| `summary` | ✅ | ✅ For extraction | ✅ |
| `url` | ✅ | ❌ Ignored | Could fetch full article |
| `source` | ✅ | ✅ For credibility | ✅ |
| `overall_sentiment_score` | ✅ | ⚠️ Partially | Should use directly as feature |
| `overall_sentiment_label` | ✅ | ⚠️ Partially | Use for validation |
| `ticker_sentiment` | ✅ | ❌ **NOT USED** | **Critical for stock-level signals** |
| `ticker_sentiment_score` | ✅ | ❌ **NOT USED** | **Direct alpha signal** |
| `ticker_sentiment_label` | ✅ | ❌ **NOT USED** | **Trade direction signal** |
| `relevance_score` | ✅ | ❌ **NOT USED** | **Filter low-relevance mentions** |
| `topics` | ✅ | ⚠️ Partially | Full topic taxonomy available |
| `time_published` | ✅ | ✅ For timestamp | ✅ |
| `banner_image` | ✅ | ❌ N/A | Not useful |
| `category_within_source` | ✅ | ❌ Ignored | Could help with classification |

### CRITICAL FINDING: We're ignoring stock-level sentiment!

Alpha Vantage provides **per-ticker sentiment** for each article:

```json
{
  "ticker_sentiment": [
    {
      "ticker": "AAPL",
      "relevance_score": "0.95",
      "ticker_sentiment_score": "0.32",
      "ticker_sentiment_label": "Bullish"
    },
    {
      "ticker": "MSFT",
      "relevance_score": "0.21",
      "ticker_sentiment_score": "-0.12",
      "ticker_sentiment_label": "Somewhat-Bearish"
    }
  ]
}
```

**This is a direct trading signal we're not using!**

### Alpaca Data - What We Get vs What We Use

| Data Point | Available | Currently Used | Potential Use |
|------------|-----------|----------------|---------------|
| OHLCV | ✅ | ✅ Returns, MA | ✅ |
| Volume | ✅ | ⚠️ Basic | Could detect volume anomalies |
| Trade count | ✅ | ❌ Not used | Institutional activity signal |
| VWAP | ✅ | ❌ Not used | Better execution |
| Corporate actions | ✅ | ❌ Not used | Adjust for splits/dividends |
| News from Alpaca | ✅ | ❌ Not used | Redundant with Alpha Vantage |

---

## Part 3: What Should LLM Actually Do?

### Principle: LLM for Reasoning, Not for Routine

| Task | Use LLM? | Reasoning |
|------|----------|-----------|
| Keyword matching | ❌ No | Simple, fast, cheap |
| Relevance classification | ⚠️ Maybe | LLM for edge cases only |
| Multi-label taxonomy | ✅ Yes | Requires context understanding |
| Event extraction | ✅ Yes | Requires semantic parsing |
| Impact reasoning | ✅ Yes | Requires market knowledge |
| Direction inference | ✅ Yes | Requires causal reasoning |
| Debate arguments | ✅ Yes | Requires creative reasoning |
| Trade explanation | ✅ Yes | Requires narrative synthesis |

### The Right LLM Tasks

**Task 1: Semantic Event Extraction (High Value)**

Input: "Fed Chair Powell signaled rates will stay higher for longer amid sticky inflation"

Expected LLM Output:
```json
{
  "event_type": "central_bank_guidance",
  "actor": "Federal Reserve",
  "action": "hawkish_forward_guidance",
  "target": "interest_rates",
  "direction": "rates_higher",
  "duration": "extended_period",
  "trigger": "sticky_inflation",
  "market_implications": {
    "equities": "bearish",
    "bonds": "bearish", 
    "usd": "bullish",
    "gold": "bearish"
  },
  "confidence": 0.92,
  "similar_historical_events": ["Dec 2022 hawkish pivot", "Jun 2023 dot plot"],
  "expected_market_reaction": "Risk-off, yield curve flattening"
}
```

**Why LLM is needed**: Rule-based can't infer "higher for longer" = extended hawkishness

---

**Task 2: Cross-Article Synthesis (High Value)**

Input: 5 articles about different aspects of China economy

Expected LLM Output:
```json
{
  "synthesis": "Multiple signals point to China growth concerns",
  "evidence": [
    "Property sector stress (Article 1)",
    "PMI below 50 (Article 3)", 
    "Export weakness (Article 4)"
  ],
  "theme": "china_slowdown",
  "severity": 0.75,
  "affected_assets": ["FXI", "EEM", "commodities"],
  "trading_implication": "Reduce EM exposure, favor defensive sectors",
  "is_new_theme": false,
  "theme_momentum": "accelerating"
}
```

**Why LLM is needed**: Connecting dots across multiple articles requires reasoning

---

**Task 3: Surprise Quantification (High Value)**

Input: "CPI came in at 3.2% vs 3.4% expected"

Expected LLM Output:
```json
{
  "event": "CPI_release",
  "actual": 3.2,
  "expected": 3.4,
  "surprise_direction": "dovish",
  "surprise_magnitude": "moderate",
  "surprise_score": 0.65,
  "historical_context": "Largest downside surprise in 6 months",
  "market_implications": {
    "immediate": "Risk-on rally, rates lower",
    "medium_term": "Increases probability of Fed cut",
    "positioning_shift": "Short covering in duration"
  }
}
```

**Why LLM is needed**: Understanding expectations vs actual requires context

---

**Task 4: Regime Narrative Generation (Medium Value)**

Input: All macro features + recent events

Expected LLM Output:
```json
{
  "current_regime": "late_cycle_uncertainty",
  "narrative": "Markets are in a late-cycle environment characterized by 
                sticky inflation, tight labor markets, and uncertainty about 
                Fed policy path. Recent data suggests inflation is moderating 
                but not fast enough for imminent cuts.",
  "key_risks": [
    "Inflation re-acceleration",
    "Credit stress from high rates",
    "Geopolitical escalation"
  ],
  "favored_strategies": ["RiskParity", "TailRisk"],
  "disfavored_strategies": ["Momentum", "Carry"],
  "regime_stability": 0.6,
  "change_catalysts": ["FOMC meeting", "CPI release", "earnings season"]
}
```

**Why LLM is needed**: Synthesizing multiple signals into coherent narrative

---

**Task 5: Trade Rationale Generation (Medium Value)**

Input: Portfolio change from rebalance

Expected LLM Output:
```json
{
  "action": "Increased GOOGL position by 2%",
  "rationale": {
    "primary": "Strong momentum in AI narrative",
    "supporting": [
      "Positive earnings sentiment (+0.45 score)",
      "Below historical valuation",
      "Low correlation with existing positions"
    ],
    "risk_consideration": "Concentration in tech increased to 18%"
  },
  "expected_holding_period": "2-4 weeks",
  "exit_triggers": [
    "Sentiment turns negative",
    "Price drops 5% from entry",
    "Tech sector correlation spikes"
  ]
}
```

**Why LLM is needed**: Explaining complex multi-factor decisions

---

## Part 4: What We're NOT Doing That We Should

### Gap 1: No Stock-Level Sentiment Utilization

**Current State**: 
- We get article-level sentiment
- We extract macro themes
- We ignore ticker-specific sentiment from Alpha Vantage

**Should Do**:
- Build daily sentiment score per stock from Alpha Vantage `ticker_sentiment`
- Weight by `relevance_score` 
- Decay older sentiment exponentially
- Feed directly to NewsSentimentEvent strategy

**Implementation**:
```
For each stock in universe:
  daily_sentiment[stock] = Σ (article_sentiment × relevance × recency_weight)
  
Features to add:
  - sentiment_score_1d
  - sentiment_score_7d
  - sentiment_momentum (1d vs 7d)
  - sentiment_volatility
  - news_volume (article count)
```

**Expected Alpha**: This is a direct signal we're not using. Literature shows 1-5 day predictability.

---

### Gap 2: No Earnings Calendar Integration

**Current State**:
- We don't know when companies report earnings
- News before/after earnings has different implications

**Should Do**:
- Track earnings dates for universe
- Adjust sentiment interpretation around earnings
- Pre-earnings: sentiment = expectation
- Post-earnings: sentiment = reaction to surprise

**Expected Alpha**: Earnings events are high-information, timing is everything.

---

### Gap 3: No Event Clustering

**Current State**:
- Each article processed independently
- Miss compounding effect of related news

**Should Do**:
- Cluster articles by theme/entity
- Track theme momentum (accelerating/decelerating)
- Identify narrative shifts

**Example**:
```
Day 1: "Banks report strong earnings" → Positive
Day 2: "Credit concerns emerge" → Negative  
Day 3: "Analyst downgrades banks" → Negative
Day 4: "Bank stocks fall 5%" → Already priced

Current: Treats each as independent
Better: Identify "banking_concern" theme building momentum
```

---

### Gap 4: No Lead-Lag Relationships

**Current State**:
- Treat each asset independently
- Miss sector rotations and spillovers

**Should Do**:
- Track which assets lead/lag on news themes
- Example: Oil news → XOM moves first → CVX follows
- Use lead assets to predict lag assets

---

### Gap 5: No Intraday Signal Decay

**Current State**:
- Daily rebalancing only
- Old news has same weight as new news

**Should Do**:
- Exponential decay on news relevance
- Morning news more important than afternoon
- Track signal freshness

---

## Part 5: Concrete Implementation Specification

### Module 1: Enhanced News Data Loader

**File**: `src/data/enhanced_news.py`

**New Features**:
1. Extract ALL fields from Alpha Vantage
2. Build per-ticker sentiment scores
3. Track article relevance scores
4. Implement topic taxonomy from AV

**Data Structure**:
```python
@dataclass
class EnhancedNewsArticle:
    # From Alpha Vantage directly
    timestamp: datetime
    title: str
    summary: str
    source: str
    overall_sentiment: float
    overall_sentiment_label: str
    
    # Per-ticker signals (THE CRITICAL DATA)
    ticker_sentiments: Dict[str, TickerSentiment]
    
    # Topic classification from AV
    topics: List[str]
    
    # Our enrichment via LLM
    macro_event: Optional[MacroEvent]
    impact_score: float
    is_actionable: bool

@dataclass
class TickerSentiment:
    ticker: str
    relevance_score: float  # How relevant article is to this ticker
    sentiment_score: float  # -1 to +1
    sentiment_label: str    # Bearish/Bullish/Neutral
```

---

### Module 2: Stock-Level Sentiment Aggregator

**File**: `src/data/ticker_sentiment.py`

**Purpose**: Aggregate article-level sentiment into stock-level daily features

**Algorithm**:
```
For each stock in universe:
    articles = get_articles_mentioning(stock, last_7_days)
    
    if len(articles) == 0:
        sentiment_score = 0  # No news is neutral
        news_volume = 0
    else:
        weighted_sentiment = 0
        total_weight = 0
        
        for article in articles:
            ticker_data = article.ticker_sentiments[stock]
            
            # Weight by relevance and recency
            recency = exp(-hours_ago / 24)  # Half-life of 24 hours
            weight = ticker_data.relevance_score * recency
            
            weighted_sentiment += ticker_data.sentiment_score * weight
            total_weight += weight
        
        sentiment_score = weighted_sentiment / total_weight
        news_volume = len(articles)

Output Features:
  - sentiment_score: float (-1 to +1)
  - sentiment_confidence: float (based on relevance and volume)
  - news_volume: int
  - sentiment_momentum: float (today vs 7d average)
  - sentiment_dispersion: float (std of individual scores)
```

---

### Module 3: LLM Event Extractor

**File**: `src/llm/event_extractor.py`

**When to Call LLM**:
- Article passes relevance threshold (>0.5 overall sentiment or >0.7 ticker relevance)
- Article has macro topic (central_bank, economy, etc.)
- Article has high-impact keywords (rate, inflation, gdp, etc.)

**LLM Call Spec**:
```python
def extract_event(article: EnhancedNewsArticle) -> MacroEvent:
    """
    Call LLM only for articles that pass pre-filter.
    Pre-filter uses Alpha Vantage's built-in relevance.
    """
    
    # Pre-filter (NO LLM needed)
    if article.overall_sentiment == 0 and len(article.ticker_sentiments) < 3:
        return None  # Not significant enough for LLM
    
    if not any(topic in MACRO_TOPICS for topic in article.topics):
        return None  # Not a macro article
    
    # LLM extraction (for ~20% of articles)
    prompt = build_event_prompt(article)
    response = llm_service.extract(prompt, EVENT_SCHEMA)
    
    return parse_event(response)
```

**Cost Control**:
- Only ~20% of articles go to LLM
- Average 50 articles/day → 10 LLM calls
- Cost: ~$0.30/day

---

### Module 4: LLM Theme Synthesizer

**File**: `src/llm/theme_synthesizer.py`

**Purpose**: Identify themes across multiple articles

**When to Call**:
- Once per hour during market hours
- Batch process new articles

**Algorithm**:
```python
def synthesize_themes(articles: List[EnhancedNewsArticle]) -> List[Theme]:
    """
    Group articles by similarity, identify emerging themes.
    """
    
    # Step 1: Group by topic (no LLM)
    topic_groups = group_by_topic(articles)
    
    # Step 2: For each group with 3+ articles, synthesize theme
    themes = []
    for topic, group in topic_groups.items():
        if len(group) >= 3:
            prompt = build_synthesis_prompt(group)
            response = llm_service.reason(prompt)
            themes.append(parse_theme(response))
    
    return themes
```

---

### Module 5: Validation Protocol

**Critical Question**: How do we know LLM outputs are correct?

**Validation Approach 1: Consistency Check**
```python
def validate_event(event: MacroEvent, article: EnhancedNewsArticle) -> bool:
    """
    Check if LLM event is consistent with Alpha Vantage sentiment.
    """
    
    # If LLM says "hawkish" but AV says "Bullish for equities" → Inconsistent
    if event.direction == "hawkish" and article.overall_sentiment > 0.3:
        logger.warning("LLM-AV sentiment mismatch")
        return False
    
    # If LLM says "risk_off" but AV says "Bullish" → Inconsistent
    if event.direction == "risk_off" and article.overall_sentiment > 0.5:
        return False
    
    return True
```

**Validation Approach 2: Market Confirmation**
```python
def validate_with_market(event: MacroEvent, time_after_hours: int = 24):
    """
    After N hours, check if market moved in predicted direction.
    """
    
    expected_direction = event.market_implications["equities"]
    actual_move = get_spy_return(event.timestamp, event.timestamp + hours(time_after_hours))
    
    if expected_direction == "bullish" and actual_move > 0.002:
        return "confirmed"
    elif expected_direction == "bearish" and actual_move < -0.002:
        return "confirmed"
    else:
        return "not_confirmed"
```

**Validation Approach 3: Human Review (Periodic)**
```
Weekly: Sample 20 LLM outputs
- Score accuracy of event extraction
- Score relevance classification
- Calculate precision/recall
- Adjust prompts based on errors
```

---

### Module 6: Feedback Loop

**Critical Missing Piece**: System should learn from outcomes

**Implementation**:
```python
class OutcomeTracker:
    """Track whether our signals led to profitable trades."""
    
    def record_signal(self, signal: Signal):
        """Record signal when generated."""
        self.signals.append({
            "timestamp": signal.timestamp,
            "ticker": signal.ticker,
            "direction": signal.direction,
            "confidence": signal.confidence,
            "macro_features": signal.macro_features,
            "outcome": None  # Filled later
        })
    
    def record_outcome(self, signal_id: str, return_1d: float, return_5d: float):
        """Record actual outcome 1d and 5d later."""
        signal = self.signals[signal_id]
        signal["outcome"] = {
            "return_1d": return_1d,
            "return_5d": return_5d,
            "was_correct": (signal.direction == "long" and return_5d > 0) or
                          (signal.direction == "short" and return_5d < 0)
        }
    
    def analyze_performance(self) -> dict:
        """Analyze which signals work."""
        completed = [s for s in self.signals if s["outcome"]]
        
        return {
            "total_signals": len(completed),
            "accuracy": mean(s["outcome"]["was_correct"] for s in completed),
            "avg_return_when_correct": ...,
            "avg_return_when_wrong": ...,
            "best_performing_features": ...,
            "worst_performing_features": ...
        }
```

---

## Part 6: Priority Ranking

### Must Have (Week 1-2)

| Task | Value | Effort | Priority |
|------|-------|--------|----------|
| Use Alpha Vantage ticker_sentiment | Very High | Low | **P0** |
| Build per-stock sentiment aggregator | Very High | Medium | **P0** |
| Add sentiment features to strategies | Very High | Medium | **P0** |
| Implement outcome tracking | High | Medium | **P1** |
| Add validation checks | High | Low | **P1** |

### Should Have (Week 3-4)

| Task | Value | Effort | Priority |
|------|-------|--------|----------|
| LLM event extraction | High | High | **P2** |
| Theme synthesis | Medium | High | **P2** |
| Feedback loop | High | Medium | **P2** |

### Nice to Have (Week 5+)

| Task | Value | Effort | Priority |
|------|-------|--------|----------|
| LLM debate arguments | Medium | Medium | **P3** |
| Trade explanation | Low | Medium | **P3** |
| Regime narrative | Low | High | **P3** |

---

## Part 7: Success Criteria

### Quantitative Metrics

| Metric | Current | Week 2 Target | Week 4 Target |
|--------|---------|---------------|---------------|
| News features per stock | 0 | 5 | 8 |
| Signal accuracy (backtest) | Unknown | 52% | 55% |
| Sharpe ratio (backtest) | Unknown | 0.5 | 0.8 |
| LLM cost per day | $0 | $0.50 | $2.00 |
| Articles processed | 200 | 200 | 200 |
| Ticker sentiments extracted | 0 | 500 | 500 |

### Qualitative Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Can explain why trade was made | ❌ | ✅ |
| Can identify regime changes | ⚠️ | ✅ |
| Can synthesize themes | ❌ | ✅ |
| Can validate predictions | ❌ | ✅ |

---

## Part 8: What Could Go Wrong

### Risk 1: Alpha Vantage Rate Limits
- **Problem**: Free tier = 25 calls/day, we need more
- **Mitigation**: 
  - Aggressive caching (already implemented)
  - Only refresh every few hours
  - Consider premium tier ($50/month)

### Risk 2: LLM Hallucination
- **Problem**: LLM invents events that didn't happen
- **Mitigation**:
  - Always validate against Alpha Vantage raw data
  - Never trade on LLM-only signal
  - Require human review for large positions

### Risk 3: Overfitting to News
- **Problem**: News-based signals may not persist
- **Mitigation**:
  - Combine with price-based signals (already done)
  - Limit news-based position sizing
  - Track signal decay over time

### Risk 4: Stale Data
- **Problem**: Using old news as if it's new
- **Mitigation**:
  - Timestamp validation
  - Decay function on old news
  - Alert on data staleness

---

## Part 9: The Actual Next Steps

### Tomorrow (Day 1)
1. Modify Alpha Vantage loader to extract `ticker_sentiment` field
2. Create TickerSentiment data class
3. Log what we're receiving (verify data quality)

### This Week (Days 2-5)
1. Build sentiment aggregator per stock
2. Add sentiment features to feature store
3. Modify NewsSentimentEvent strategy to use direct sentiment
4. Backtest with new features vs without

### Next Week (Days 6-10)
1. Add LLM event extraction for macro articles
2. Implement validation protocol
3. Add outcome tracking
4. Create monitoring dashboard

### Week 3+
1. LLM theme synthesis
2. Feedback loop implementation
3. Performance analysis

---

## Part 10: Final Checklist

Before considering the plan complete, verify:

- [ ] Are we using ALL available data from Alpha Vantage?
- [ ] Do we have per-stock sentiment (not just article-level)?
- [ ] Can we track signal → outcome → learning?
- [ ] Do we have validation before trading on signals?
- [ ] Is LLM used only where it adds value?
- [ ] Do we have fallback for LLM failures?
- [ ] Can we explain every trade in plain English?
- [ ] Do we have cost controls on LLM usage?
- [ ] Are we monitoring system health?
- [ ] Can we backtest new features properly?

---

## Conclusion

The previous plan was too focused on LLM and missed the **most valuable low-hanging fruit**: Alpha Vantage already provides ticker-level sentiment that we're completely ignoring.

**Revised Priority**:
1. **First**: Extract and use ALL Alpha Vantage data (no LLM needed)
2. **Second**: Build proper feedback loops and validation
3. **Third**: Add LLM for complex reasoning tasks

The goal is not "use more LLM" but "extract maximum value from our data sources."

---

*Document Version: 2.0*
*Created: 2026-01-20*
*Status: Ready for Implementation Review*
