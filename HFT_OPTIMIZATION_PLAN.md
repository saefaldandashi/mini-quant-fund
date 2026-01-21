# HFT-Lite Optimization Plan

## Current Problems

| Issue | Impact | Priority |
|-------|--------|----------|
| Rebalance takes 45-90 seconds | Cannot trade every 15-30 min | **CRITICAL** |
| No strategy blending | Miss optimal opportunities | **HIGH** |
| News not time-aware | Stale news treated same as breaking | **HIGH** |
| Debate engine too slow | LLM calls block execution | **CRITICAL** |
| Features are daily, not intraday | Strategies use wrong data | **MEDIUM** |

---

## Solution Architecture

### 1. ULTRA-FAST EXECUTION MODE

**Target: <5 seconds per rebalance**

```
┌─────────────────────────────────────────────────────────────────────┐
│ FAST PATH (for intraday trading)                                    │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │ Pre-cached  │ → │ Rule-based  │ → │ Immediate   │               │
│  │ Data        │   │ Debate      │   │ Execution   │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
│       0.5s              0.5s              2-3s                      │
│                                                                     │
│  TOTAL: ~3-4 seconds                                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ SLOW PATH (for analysis/position trading)                          │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │ Fresh Data  │ → │ LLM Debate  │ → │ Careful     │               │
│  │ Fetch       │   │ (full)      │   │ Execution   │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
│       20-30s           15-30s            5-10s                      │
│                                                                     │
│  TOTAL: ~45-90 seconds                                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
- Background thread continuously updates cached data (every 30 seconds)
- Rule-based debate scoring (no LLM) for intraday
- LLM debate runs async in background, results cached
- Execution uses pre-computed spread analysis

---

### 2. DYNAMIC STRATEGY BLENDING

**Blend intraday + position based on market conditions**

```python
def get_strategy_blend_weights(vix, time_of_day, news_velocity):
    """
    Returns blend weights for intraday vs position strategies.
    """
    # Base weights
    intraday_weight = 0.5
    position_weight = 0.5
    
    # VIX adjustment (high vol = more intraday)
    if vix > 30:
        intraday_weight += 0.3
    elif vix > 25:
        intraday_weight += 0.2
    elif vix < 15:
        position_weight += 0.2
    
    # Time of day (open/close = more intraday)
    hour = time_of_day.hour
    if 9 <= hour <= 10:  # First hour
        intraday_weight += 0.2  # ORB strategy
    elif 15 <= hour <= 16:  # Last hour
        intraday_weight += 0.1  # End-of-day momentum
    
    # News velocity (many articles = breaking news = intraday)
    if news_velocity > 10:  # >10 articles in last 30 min
        intraday_weight += 0.2
    
    # Normalize
    total = intraday_weight + position_weight
    return {
        'intraday': intraday_weight / total,
        'position': position_weight / total,
    }
```

---

### 3. TIME-AWARE NEWS INTEGRATION

**News recency affects weight**

```python
def get_news_urgency_weight(article_time, current_time):
    """
    More recent news gets higher weight for intraday trading.
    """
    age_minutes = (current_time - article_time).total_seconds() / 60
    
    if age_minutes < 5:
        return 2.0   # Breaking news: 2x weight
    elif age_minutes < 15:
        return 1.5   # Very recent: 1.5x weight
    elif age_minutes < 30:
        return 1.2   # Recent: 1.2x weight
    elif age_minutes < 60:
        return 1.0   # Normal: 1x weight
    elif age_minutes < 240:
        return 0.5   # Old: 0.5x weight
    else:
        return 0.2   # Stale: 0.2x weight
```

---

### 4. FAST DEBATE ENGINE (Rule-Based)

**For intraday, use rules instead of LLM**

```python
def fast_debate_score(strategy, features, time_horizon='intraday'):
    """
    Fast rule-based scoring (no LLM calls).
    """
    score = 0.5  # Base score
    
    # Confidence boost
    score += strategy.confidence * 0.2
    
    # Urgency boost (for intraday)
    if hasattr(strategy, 'urgency') and strategy.urgency == 'immediate':
        score += 0.15
    
    # Regime alignment
    if features.regime == 'high_vol' and strategy.name in INTRADAY_STRATEGIES:
        score += 0.1
    
    # Historical performance (from learning system)
    if strategy.name in learned_performance:
        score += learned_performance[strategy.name] * 0.2
    
    return min(1.0, max(0.0, score))
```

---

### 5. IMPLEMENTATION PRIORITY

| Step | Change | Time to Implement |
|------|--------|-------------------|
| 1 | Add ultra-fast mode (skip LLM) | 30 min |
| 2 | Background data caching (continuous) | 45 min |
| 3 | Dynamic strategy blending | 30 min |
| 4 | News recency weighting | 20 min |
| 5 | Fast rule-based debate | 30 min |
| 6 | Intraday features (VWAP, etc.) | 1 hour |

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Rebalance time | 45-90 sec | **3-5 sec** |
| Strategy blend | Fixed | **Dynamic** |
| News freshness | Ignored | **Weighted** |
| LLM usage | Every trade | **Background only** |
| Trades per hour | ~1 | **4-6** |
