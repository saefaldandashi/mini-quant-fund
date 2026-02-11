# Rebalance Testing Guide

## Quick Test (Cloud Server)

### Step 1: Pull Latest Code
```bash
git pull origin main
```

### Step 2: Verify Fixes Are In Code
```bash
python3 verify_fixes.py
```

Expected output:
```
✅ ALL FIXES VERIFIED IN CODE
```

### Step 3: Test Rebalance Execution
```bash
python3 test_rebalance_cloud.py
```

This will:
- Verify all fixes are in code
- Check server.log for recent errors
- Optionally run a test rebalance
- Report any issues found

### Step 4: Monitor Live Rebalance

If testing on cloud, you can trigger a rebalance via API:
```bash
curl -X POST http://your-server:5000/api/run?force=true
```

Then monitor:
```bash
tail -f server.log | grep -E "ERROR|UnboundLocalError|REBALANCE|enhanced_weights"
```

## What to Look For

### ✅ Success Indicators:
1. No `UnboundLocalError` in logs
2. Rebalance completes without crashing
3. Position sizes are 2-3x larger than before
4. More positions created (due to lower thresholds)
5. No errors in server.log during execution

### ❌ Error Indicators:
1. `UnboundLocalError: cannot access local variable 'enhanced_weights'`
   - **Solution**: Code not updated - run `git pull origin main` and restart
   
2. Rebalance fails immediately
   - **Solution**: Check API keys, network connectivity
   
3. Position sizes still very small
   - **Solution**: Verify `use_kelly=False` in app.py line 234
   - Verify thresholds in config.py are lowered

4. No positions created
   - **Solution**: Check minimum thresholds, verify signals are being generated

## Solution Architecture for Issues

### Issue: UnboundLocalError Still Occurs

**Root Cause**: Code not updated on server

**Solution**:
1. Verify code is up to date:
   ```bash
   git status
   git pull origin main
   ```
2. Verify fix is in code:
   ```bash
   grep -n "CRITICAL FIX.*enhanced_weights" app.py
   ```
   Should show line 3603
3. Restart application
4. Run test again

### Issue: Position Sizes Still Small

**Root Cause**: Double Kelly or thresholds not applied

**Solution**:
1. Verify Kelly fix:
   ```bash
   grep "use_kelly" app.py
   ```
   Should show `use_kelly=False`
2. Verify thresholds:
   ```bash
   grep "min_position_pct" config.py
   ```
   Moderate should be 0.02 (2%)
3. Check if StrategyEnhancer is applying Kelly:
   - Look for "Kelly Multiplier" in rebalance output
   - Should show multiplier being applied once

### Issue: Rebalance Crashes with Other Errors

**Root Cause**: Various (network, data, etc.)

**Solution**:
1. Check server.log for specific error
2. Review error message
3. Common fixes:
   - Network errors → Add retry logic (Fix #4)
   - Missing data → Remove delisted symbols (Fix #5)
   - LLM errors → Check quota/rate limits

## Continuous Monitoring

After fixes are deployed, monitor:

1. **Success Rate**: Should be >95%
   ```bash
   grep "REBALANCE COMPLETE" server.log | wc -l
   grep "REBALANCE FAILED" server.log | wc -l
   ```

2. **Position Sizes**: Should be larger
   - Check rebalance output for position percentages
   - Should see 3-5%+ positions (not 0.5-1%)

3. **Position Count**: Should be more
   - Check "Target portfolio: X positions" in logs
   - Should see 10-20 positions (not 2-5)

4. **Errors**: Should be minimal
   ```bash
   tail -1000 server.log | grep ERROR | wc -l
   ```

## Next Fixes (If Needed)

If issues persist after these fixes:

1. **Fix #4**: Network reliability (retry logic)
2. **Fix #5**: Data quality (remove delisted symbols)
3. **Fix #6**: Transaction cost filter (relax thresholds)
4. **Fix #7**: Capital exposure (less aggressive reduction)
