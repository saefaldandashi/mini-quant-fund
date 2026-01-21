# Strategy Enhancement Plan
## Mini Quant Fund - Comprehensive Upgrade

**Date:** January 20, 2026  
**Objective:** Transform the current conservative system into a higher-conviction, more profitable trading engine while maintaining appropriate risk controls.

---

## Executive Summary

The current system invests only ~25-30% of available capital due to:
1. Low strategy confidence scores
2. Conservative Kelly Criterion sizing
3. Unfocused stock universe (79 stocks)
4. Underutilized sentiment data
5. Passive learning system

This plan addresses each issue with concrete, measurable improvements.

---

## Phase 1: Position Sizing Overhaul
**Goal:** Increase capital deployment from ~30% to 60-80% of target exposure

### 1.1 Risk Appetite Setting
Add a user-controllable risk level that adjusts the entire system:

| Setting | Kelly Multiplier | Min Position | Max Positions | Description |
|---------|------------------|--------------|---------------|-------------|
| Conservative | 0.25× | 1% | 30 | Current behavior |
| Moderate | 0.50× | 2% | 20 | Balanced approach |
| Aggressive | 0.75× | 3% | 15 | Higher conviction |
| Maximum | 1.00× | 5% | 10 | Full Kelly, concentrated |

**Implementation:**
- Add slider/dropdown in UI for risk appetite
- Multiply all position sizes by the Kelly multiplier
- Enforce minimum position sizes (no more 0.5% positions)
- Cap number of positions to force concentration

### 1.2 Minimum Investment Floor
If total allocated < 50% of target exposure, scale up all positions proportionally.

**Example:**
- Target exposure: $80,000
- Strategies allocate: $25,000 (31%)
- Floor: 50% minimum → Scale to $40,000
- All positions increase by 1.6×

### 1.3 Confidence Calibration
Current confidence scores are too conservative. Recalibrate:

| Current Range | Issue | New Calibration |
|---------------|-------|-----------------|
| 0.0 - 0.3 | Too common | Map to 0.0 - 0.2 |
| 0.3 - 0.6 | Should be higher | Map to 0.3 - 0.7 |
| 0.6 - 1.0 | Rare | Map to 0.7 - 1.0 |

---

## Phase 2: Sentiment-Driven Stock Selection
**Goal:** Focus on stocks where we have information edge from Alpha Vantage

### 2.1 Sentiment Score Thresholds
Only trade stocks meeting these criteria:

| Criteria | Threshold | Rationale |
|----------|-----------|-----------|
| Relevance Score | > 0.3 | News must be about the stock |
| Sentiment Magnitude | > 0.15 or < -0.15 | Clear bullish/bearish signal |
| News Volume | ≥ 3 articles/week | Sufficient data |
| Sentiment Consistency | > 70% same direction | Not mixed signals |

**Result:** Universe shrinks from 79 → ~20-30 actionable stocks

### 2.2 Sentiment Momentum
Track 7-day sentiment change:

| Momentum | Action |
|----------|--------|
| Improving rapidly (Δ > +0.2) | Boost position by 25% |
| Stable positive | Normal position |
| Declining | Reduce or avoid |
| Negative momentum | Short candidate (if enabled) |

### 2.3 News Recency Weighting
Recent news matters more:

| Age | Weight |
|-----|--------|
| < 24 hours | 100% |
| 1-3 days | 70% |
| 3-7 days | 40% |
| > 7 days | 10% |

---

## Phase 3: Focused Universe Strategy
**Goal:** Trade fewer stocks with higher conviction

### 3.1 Dynamic Universe Filtering
Daily filter the 79-stock universe down to "tradeable" stocks:

**Filter Pipeline:**
```
79 stocks
  ↓ Sentiment filter (has recent news + clear signal) → ~40 stocks
  ↓ Liquidity filter (avg volume > $10M/day) → ~35 stocks
  ↓ Momentum filter (price > 20-day MA) → ~25 stocks
  ↓ Top conviction (highest combined score) → 10-15 stocks
```

### 3.2 Sector Concentration Limits
Prevent over-concentration while allowing conviction:

| Sector | Max Allocation |
|--------|---------------|
| Technology | 35% |
| Financial | 25% |
| Healthcare | 20% |
| Consumer | 20% |
| Energy | 15% |
| Other | 15% |

### 3.3 Position Size Tiers

| Conviction Level | Position Size | Max Stocks |
|------------------|---------------|------------|
| Very High (>0.8) | 8-10% | 3 |
| High (0.6-0.8) | 5-7% | 5 |
| Medium (0.4-0.6) | 3-5% | 7 |
| Low (<0.4) | 0% (don't trade) | 0 |

---

## Phase 4: Enhanced Regime Detection
**Goal:** Automatically adjust strategy based on market conditions

### 4.1 Market Regime Classification
Compute daily regime score:

| Indicator | Weight | Bullish | Bearish |
|-----------|--------|---------|---------|
| SPY vs 200-day MA | 25% | Above | Below |
| VIX Level | 20% | < 18 | > 25 |
| Macro Risk Sentiment | 20% | > 0.2 | < -0.2 |
| Geopolitical Index | 15% | < 0.3 | > 0.6 |
| Financial Stress | 10% | < 0.2 | > 0.5 |
| Breadth (% above 50-MA) | 10% | > 60% | < 40% |

**Regime Output:**
- **Risk-On (score > 0.6):** Full exposure, favor momentum
- **Neutral (0.4-0.6):** Balanced approach
- **Risk-Off (score < 0.4):** Reduce exposure to 50%, favor defensive

### 4.2 Automatic Exposure Adjustment

| Regime | Base Exposure | Strategy Bias |
|--------|---------------|---------------|
| Strong Bull | 100% of target | Momentum 2×, Mean Reversion 0.5× |
| Mild Bull | 80% of target | Balanced |
| Neutral | 60% of target | Quality/Value focus |
| Mild Bear | 40% of target | Defensive, high dividend |
| Strong Bear | 20% of target | Cash is king |

### 4.3 Volatility Scaling
Dynamic position sizing based on realized volatility:

```
Position Size = Base Size × (Target Vol / Realized Vol)

Example:
- Target volatility: 15%
- Current SPY realized vol: 25%
- Scaling factor: 15/25 = 0.6
- All positions reduced by 40%
```

---

## Phase 5: Debate System Improvements
**Goal:** Make strategy debates more decisive and meaningful

### 5.1 Weighted Voting by Track Record
Strategies with better recent performance get more votes:

| Performance (30-day) | Vote Weight |
|---------------------|-------------|
| Top 3 strategies | 1.5× |
| Middle 3 strategies | 1.0× |
| Bottom 3 strategies | 0.5× |

### 5.2 Consensus Requirements

| Agreement Level | Action |
|-----------------|--------|
| 7+ strategies agree | Full position (1.0×) |
| 5-6 strategies agree | Reduced position (0.7×) |
| 4 strategies agree | Small position (0.4×) |
| < 4 agree | No position |

### 5.3 Veto Power for Risk Strategies
TailRisk and RiskParity strategies can veto positions:

- If TailRisk flags high tail risk → Position capped at 3%
- If RiskParity flags correlation risk → Position reduced 30%

### 5.4 LLM Debate Enhancement
Use Gemini more substantively:

| Current | Enhanced |
|---------|----------|
| Template-based arguments | LLM generates unique arguments per stock |
| Generic rebuttals | Context-aware counter-arguments |
| No memory | Remember past debate outcomes |

---

## Phase 6: Active Learning & Adaptation
**Goal:** System gets smarter over time based on results

### 6.1 Daily Performance Attribution
After market close, compute:

- Which strategy's picks made money?
- Which lost money?
- What was the sentiment at entry vs outcome?

### 6.2 Strategy Weight Adjustment
Weekly rolling adjustment:

```
New Weight = Old Weight × (1 + Performance Score × Learning Rate)

Example:
- Momentum strategy: +5% returns this week
- Learning rate: 0.1
- New weight: 1.0 × (1 + 0.05 × 0.1) = 1.005 → +0.5% boost
```

### 6.3 Sentiment Effectiveness Tracking
Track: "When sentiment was X, what happened?"

| Sentiment | Predicted | Actual (5-day) | Effectiveness |
|-----------|-----------|----------------|---------------|
| Strong Bullish (>0.4) | +3% | +2.1% | 70% accurate |
| Mild Bullish (0.2-0.4) | +1% | +0.8% | 80% accurate |
| Neutral | 0% | -0.2% | 60% accurate |
| Bearish (<-0.2) | -2% | -1.5% | 75% accurate |

Use this to calibrate how much to trust sentiment signals.

### 6.4 Pattern Recognition
Log and learn from patterns:

```
Pattern: "Tech stock + Strong earnings beat + Bullish sentiment"
Historical outcome: +4.2% average (5-day), 78% win rate
Action: Boost position size by 20% when pattern detected
```

---

## Phase 7: UI/UX Improvements
**Goal:** Make the system more usable and transparent

### 7.1 New Controls
- [ ] Risk Appetite slider (Conservative → Aggressive)
- [ ] Minimum Investment % floor
- [ ] Target number of positions
- [ ] Regime override (force risk-on/off)

### 7.2 New Displays
- [ ] Why each stock was selected (top 3 reasons)
- [ ] Strategy contribution breakdown per position
- [ ] Regime indicator with color coding
- [ ] Sentiment heatmap for universe
- [ ] Performance attribution chart

### 7.3 Alerts
- [ ] Regime change notification
- [ ] Large position P/L alerts
- [ ] Strategy weight shift alerts
- [ ] News event alerts for holdings

---

## Implementation Priority

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| Phase 1: Position Sizing | Medium | HIGH | 🔴 First |
| Phase 2: Sentiment-Driven | Medium | HIGH | 🔴 First |
| Phase 3: Focused Universe | Low | MEDIUM | 🟡 Second |
| Phase 4: Regime Detection | Medium | HIGH | 🟡 Second |
| Phase 5: Debate Improvements | High | MEDIUM | 🟢 Third |
| Phase 6: Active Learning | High | HIGH | 🟢 Third |
| Phase 7: UI Improvements | Low | LOW | 🔵 Ongoing |

---

## Expected Outcomes

### Before Enhancements
- Capital deployed: ~25-30% of target
- Number of positions: 24-30
- Average position size: ~1%
- Regime awareness: Basic
- Learning: Passive

### After Enhancements
- Capital deployed: 60-80% of target
- Number of positions: 10-15
- Average position size: 4-8%
- Regime awareness: Active, auto-adjusting
- Learning: Continuous improvement

### Risk Considerations
- Higher concentration = higher volatility
- More aggressive sizing = larger potential losses
- Mitigation: Regime detection + volatility scaling + stop-losses

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Capital Utilization | 30% | 70% | Avg invested / target |
| Position Conviction | 1% avg | 5% avg | Average position size |
| Regime Accuracy | N/A | 65%+ | Correct regime calls |
| Learning Improvement | N/A | +2%/quarter | Strategy weight optimization |
| Sentiment Edge | Unknown | +0.5%/trade | Returns when following sentiment |

---

## Next Steps

1. **Review this plan** - Provide feedback on priorities
2. **Phase 1 Implementation** - Position sizing changes (1-2 hours)
3. **Phase 2 Implementation** - Sentiment integration (2-3 hours)
4. **Testing** - Run simulations with new settings
5. **Iterate** - Adjust based on results

---

*This plan is designed to be implemented incrementally. Each phase can be tested independently before moving to the next.*
