# Remaining Fixes Plan - Prioritized

## ✅ Phase 1: Critical Fixes (COMPLETE)
All 6 critical fixes have been implemented, tested, and verified:
1. ✅ UnboundLocalError prevention
2. ✅ Double Kelly removal
3. ✅ Lower minimum position thresholds
4. ✅ Network reliability (retry logic)
5. ✅ Transaction cost filter relaxation
6. ✅ Capital exposure reduction (less aggressive)

## 🎯 Phase 2: High-Impact Fixes (NEXT PRIORITY)

### Fix #7: Data Quality - Filter Delisted/Missing Symbols
**Priority**: HIGH  
**Impact**: Prevents errors, improves reliability  
**Complexity**: LOW  
**Status**: Ready to implement

**Problem**: 
- Logs show errors for delisted symbols (ANSS, $TASI.SR, $ADI.AD)
- Missing price data causes warnings
- These errors don't crash but create noise

**Solution**:
- Add symbol validation before data fetching
- Filter out delisted/invalid symbols
- Better error handling for missing data
- Cache "bad symbols" to avoid repeated failures

**Files to modify**:
- `src/data/market_data.py` - Add symbol validation
- `app.py` - Filter symbols before processing
- `broker_alpaca.py` - Handle missing prices gracefully

**Expected Impact**:
- Cleaner logs
- Fewer warnings
- More reliable data fetching

---

### Fix #8: Real-Time Risk Monitoring Enhancement
**Priority**: HIGH  
**Impact**: Better risk management, prevents large losses  
**Complexity**: MEDIUM  
**Status**: Ready to implement

**Problem**:
- Risk only checked at rebalance time
- No continuous monitoring during trading hours
- Large drawdowns can occur between rebalances

**Solution**:
- Enhance existing `RealtimeRiskMonitor` 
- Add more frequent checks (every 5-10 minutes)
- Implement automatic position reduction on drawdown
- Add VIX-based exposure adjustments

**Files to modify**:
- `src/risk/realtime_monitor.py` - Enhance monitoring
- `app.py` - Integrate with rebalance flow

**Expected Impact**:
- Better risk control
- Automatic de-risking during drawdowns
- VIX-aware position sizing

---

### Fix #9: Parallel Data Fetching
**Priority**: MEDIUM-HIGH  
**Impact**: Faster rebalance (40s → 15s)  
**Complexity**: MEDIUM  
**Status**: Ready to implement

**Problem**:
- Data fetching is sequential (price, news, macro)
- Takes 30-40 seconds per rebalance
- Slows down the entire process

**Solution**:
- Use parallel fetching for independent data sources
- Price data, news, and macro can be fetched simultaneously
- Use `concurrent.futures.ThreadPoolExecutor`

**Files to modify**:
- `app.py` - Parallelize data fetching (lines ~1300-1350)

**Expected Impact**:
- 50-60% faster rebalance
- Better responsiveness
- More time for strategy execution

---

### Fix #10: Learning Engine Influence Increase
**Priority**: MEDIUM  
**Impact**: Better strategy selection based on performance  
**Complexity**: LOW  
**Status**: Ready to implement

**Problem**:
- Learning engine influence is only 30%
- Past performance barely affects decisions
- Winning strategies don't get enough weight

**Solution**:
- Increase learning influence to 50-70%
- More aggressively favor winning strategies
- Reduce allocation to consistently losing strategies

**Files to modify**:
- `src/learning/learning_engine.py` - Adjust blend weights

**Expected Impact**:
- Better strategy selection
- Improved performance over time
- Adaptive to market conditions

---

## 🔧 Phase 3: Medium Priority Fixes

### Fix #11: Real Bid-Ask Spreads (Not Estimated)
**Priority**: MEDIUM  
**Impact**: More accurate transaction costs  
**Complexity**: MEDIUM  
**Status**: Partially implemented (get_current_quotes exists)

**Problem**:
- Transaction costs use estimated spreads
- Can be 50%+ off from reality
- May reject good trades or accept bad ones

**Solution**:
- Use `broker.get_current_quotes()` more extensively
- Fall back to estimates only when quotes unavailable
- Cache spreads to reduce API calls

**Files to modify**:
- `src/execution/transaction_costs.py` - Use real quotes
- `app.py` - Fetch quotes before cost calculation

**Expected Impact**:
- More accurate cost estimates
- Better trade decisions
- Improved execution quality

---

### Fix #12: Regime-Driven Trading Mode Switching
**Priority**: MEDIUM  
**Impact**: Better adaptation to market conditions  
**Complexity**: MEDIUM  
**Status**: Ready to implement

**Problem**:
- Trading mode is static (intraday vs swing)
- Doesn't adapt to market regime
- Missing opportunities in different regimes

**Solution**:
- Use regime detection to switch modes
- High volatility → intraday
- Low volatility → swing
- Crisis mode → defensive

**Files to modify**:
- `app.py` - Dynamic mode switching based on regime
- `src/optimizations/strategy_enhancer.py` - Regime detection

**Expected Impact**:
- Better adaptation to market
- Improved performance in different regimes
- More intelligent strategy selection

---

### Fix #13: Order Execution Monitoring
**Priority**: MEDIUM  
**Impact**: Better fill quality  
**Complexity**: MEDIUM  
**Status**: Partially implemented (wait_for_fill exists)

**Problem**:
- Fire-and-forget order execution
- No tracking of partial fills
- May miss optimal fill opportunities

**Solution**:
- Enhance order monitoring
- Track partial fills
- Implement order adjustments for large orders

**Files to modify**:
- `broker_alpaca.py` - Enhance order monitoring
- `src/execution/smart_executor.py` - Better fill tracking

**Expected Impact**:
- Better fill prices
- Reduced market impact
- More reliable execution

---

## 📋 Phase 4: Lower Priority (Future)

### Fix #14: TWAP/VWAP Execution for Large Orders
**Priority**: LOW-MEDIUM  
**Impact**: Lower market impact on large trades  
**Complexity**: HIGH  
**Status**: Future enhancement

### Fix #15: Sentiment in All Strategies
**Priority**: LOW  
**Impact**: Better signal quality  
**Complexity**: LOW  
**Status**: Future enhancement

### Fix #16: Drawdown Circuit Breaker
**Priority**: LOW (already have loss awareness)  
**Impact**: Additional safety  
**Complexity**: LOW  
**Status**: Future enhancement

---

## 🎯 Recommended Next Steps

### Immediate (This Week):
1. **Fix #7: Data Quality** - Quick win, high impact
2. **Fix #8: Real-Time Risk** - Important for safety
3. **Fix #9: Parallel Fetching** - Improves speed

### Short Term (Next 2 Weeks):
4. **Fix #10: Learning Influence** - Improves performance
5. **Fix #11: Real Spreads** - Better accuracy

### Medium Term (Next Month):
6. **Fix #12: Regime Switching** - Better adaptation
7. **Fix #13: Order Monitoring** - Better execution

---

## 📊 Impact vs Complexity Matrix

| Fix | Impact | Complexity | Priority |
|-----|--------|------------|----------|
| #7: Data Quality | High | Low | ⭐⭐⭐ |
| #8: Real-Time Risk | High | Medium | ⭐⭐⭐ |
| #9: Parallel Fetching | High | Medium | ⭐⭐⭐ |
| #10: Learning Influence | Medium | Low | ⭐⭐ |
| #11: Real Spreads | Medium | Medium | ⭐⭐ |
| #12: Regime Switching | Medium | Medium | ⭐⭐ |
| #13: Order Monitoring | Medium | Medium | ⭐⭐ |

---

## 🚀 Implementation Order

**Week 1:**
- Fix #7: Data Quality (1-2 hours)
- Fix #8: Real-Time Risk (2-3 hours)

**Week 2:**
- Fix #9: Parallel Fetching (2-3 hours)
- Fix #10: Learning Influence (1 hour)

**Week 3:**
- Fix #11: Real Spreads (2-3 hours)
- Fix #12: Regime Switching (3-4 hours)

**Week 4:**
- Fix #13: Order Monitoring (2-3 hours)
- Testing and optimization

---

## ✅ Success Criteria

After implementing Phase 2 fixes:
- ✅ No delisted symbol errors
- ✅ Faster rebalance (< 20s)
- ✅ Better risk monitoring
- ✅ Improved strategy selection
- ✅ More accurate cost estimates

---

**Status**: Ready to begin Phase 2 fixes  
**Next Action**: Start with Fix #7 (Data Quality) - quick win, high impact
