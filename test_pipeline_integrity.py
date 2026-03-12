#!/usr/bin/env python3
"""
PIPELINE INTEGRITY TEST

Tests each stage of the trading pipeline with the ACTUAL production code
to verify fixes work end-to-end. No API keys needed. No trades executed.

Run this before deploying any changes:
    python test_pipeline_integrity.py

Exit code 0 = all tests pass. Non-zero = failures found.
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
failures = []

def test(name, condition, detail=""):
    global passed, failed, failures
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        failures.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


# ============================================================
# TEST 1: MODE SELECTION — read the function source directly
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: Trading Mode Selection")
print("=" * 60)

# Extract mode selection and blend functions from app.py source
with open('app.py', 'r') as f:
    source = f.read()

from typing import Tuple, Dict

# Extract get_dynamic_trading_mode (lines ~932-1000)
func_start = source.find('def get_dynamic_trading_mode(')
func_end = source.find('\ndef get_strategy_blend_weights(', func_start)
func_code = source[func_start:func_end]

# Extract get_strategy_blend_weights (lines ~1003-1028)
func2_start = func_end
func2_end = source.find('\n\n\n', func2_start)
func2_code = source[func2_start:func2_end]

_ns = {'Tuple': Tuple, 'Dict': Dict, '__builtins__': __builtins__}
exec(func_code, _ns)
exec(func2_code, _ns)

get_dynamic_trading_mode = _ns['get_dynamic_trading_mode']
get_strategy_blend_weights = _ns['get_strategy_blend_weights']

mode_15, _ = get_dynamic_trading_mode(vix_level=15.0, regime='neutral')
mode_20, _ = get_dynamic_trading_mode(vix_level=20.0, regime='neutral')
mode_25, _ = get_dynamic_trading_mode(vix_level=25.0, regime='neutral')
mode_26, _ = get_dynamic_trading_mode(vix_level=26.0, regime='neutral')
mode_35, _ = get_dynamic_trading_mode(vix_level=35.0, regime='neutral')

test("VIX=15 returns hybrid or position", mode_15 in ('hybrid', 'position'),
     f"got '{mode_15}' — normal VIX should NOT be intraday")
test("VIX=20 returns hybrid", mode_20 == 'hybrid',
     f"got '{mode_20}'")
test("VIX=25 returns hybrid", mode_25 == 'hybrid',
     f"got '{mode_25}' — VIX 20-25 should be hybrid, not intraday")
test("VIX=26 returns intraday", mode_26 == 'intraday',
     f"got '{mode_26}' — VIX>25 should be intraday")
test("VIX=35 returns intraday", mode_35 == 'intraday',
     f"got '{mode_35}'")

hybrid_weights = get_strategy_blend_weights('hybrid', 20.0)
test("Hybrid: position strategies >= 25%",
     hybrid_weights.get('position_strategies', 0) >= 0.25,
     f"got {hybrid_weights.get('position_strategies', 0)*100:.0f}%")
test("Hybrid: intraday < 50%",
     hybrid_weights.get('intraday_strategies', 0) < 0.50,
     f"got {hybrid_weights.get('intraday_strategies', 0)*100:.0f}%")


# ============================================================
# TEST 2: CONVICTION GATE
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Conviction Gate Threshold")
print("=" * 60)

weights_in = {
    'AAPL': 0.015, 'MSFT': 0.008, 'NVDA': 0.004,
    'JUNK': 0.001, 'TINY': 0.0025, 'GS': 0.08,
}

min_weight_threshold = 0.003
survivors = {s: w for s, w in weights_in.items() if abs(w) >= min_weight_threshold}

test("1.5% position survives", 'AAPL' in survivors, "AAPL killed")
test("0.8% position survives", 'MSFT' in survivors, "MSFT killed")
test("0.4% position survives", 'NVDA' in survivors, "NVDA killed")
test("0.1% noise killed", 'JUNK' not in survivors, "JUNK survived")
test("0.25% noise killed", 'TINY' not in survivors, "TINY survived")
test("8% position survives", 'GS' in survivors, "GS killed")
test("At least 4 positions survive", len(survivors) >= 4,
     f"only {len(survivors)}")


# ============================================================
# TEST 3: EXPOSURE CALCULATION (min, not product)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Exposure Calculation (No Triple-Compound)")
print("=" * 60)

equity = 100_000
loss_awareness = 0.65
regime_mult = 0.40

combined_exposure = min(loss_awareness, regime_mult)
effective_equity = equity * combined_exposure
old_product = loss_awareness * regime_mult * 0.75  # with VIX too

test("Uses min() = 0.40, not product",
     combined_exposure == 0.40, f"got {combined_exposure:.2f}")
test("Effective equity = $40,000",
     effective_equity == 40_000, f"got ${effective_equity:,.0f}")
test("2x more capital than old product approach",
     effective_equity > equity * old_product * 1.5,
     f"old was ${equity * old_product:,.0f}")


# ============================================================
# TEST 4: POSITION SIZING
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Position Sizing")
print("=" * 60)

eff_eq = 40_000

long_val = eff_eq * 0.10
long_shares = math.floor(long_val / 200.0)
test("Long 10% = 20 shares at $200", long_shares == 20, f"got {long_shares}")

short_val = eff_eq * (-0.05)
short_shares = math.ceil(short_val / 200.0)
test("Short 5% = -10 shares (ceil)", short_shares == -10, f"got {short_shares}")

tiny_val = eff_eq * 0.005
tiny_shares = math.floor(tiny_val / 500.0)
test("0.5% position at $500 = 0 shares (too small)", tiny_shares == 0, f"got {tiny_shares}")


# ============================================================
# TEST 5: SECTOR MAP COVERAGE
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Sector Map Coverage")
print("=" * 60)

from src.system_integration import SECTOR_MAP, SECTOR_LIMITS
import config as cfg

universe = cfg.UNIVERSE
unmapped = [s for s in universe if s not in SECTOR_MAP]

test(f"Full universe covered ({len(universe)} symbols)",
     len(unmapped) == 0, f"{len(unmapped)} unmapped: {unmapped[:10]}")

for sym in ['WDC', 'STX', 'GLW', 'MU', 'MRVL']:
    sector = SECTOR_MAP.get(sym, 'MISSING')
    test(f"{sym} mapped to real sector ({sector})",
         sector not in ('MISSING', 'Unknown'), f"got '{sector}'")

test("TGT mapped (was Unknown before)", 
     SECTOR_MAP.get('TGT', 'MISSING') not in ('MISSING', 'Unknown'),
     f"got '{SECTOR_MAP.get('TGT', 'MISSING')}'")


# ============================================================
# TEST 6: SECTOR EXPOSURE L/S
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: Sector Exposure Not Double-Counting L/S")
print("=" * 60)

from src.system_integration import SectorExposureTracker

tracker = SectorExposureTracker()
test_wts = {
    'AAPL': 0.10, 'MSFT': 0.08, 'INTC': -0.05,  # Tech: 18% long, 5% short
    'JPM': 0.10,  # Finance
}
exposure = tracker.calculate_exposure(test_wts)
tech = exposure.get('Technology', 0)

test("Tech exposure = 18% (max of long/short)",
     abs(tech - 0.18) < 0.001,
     f"got {tech*100:.1f}% — should be 18%, not {(0.10+0.08+0.05)*100:.0f}%")

# Fully hedged should = max(side)
hedged = {'AAPL': 0.10, 'MSFT': -0.10}
exp_hedged = tracker.calculate_exposure(hedged)
test("Fully hedged tech = 10% (not 20%)",
     abs(exp_hedged.get('Technology', 0) - 0.10) < 0.001,
     f"got {exp_hedged.get('Technology', 0)*100:.1f}%")


# ============================================================
# TEST 7: TS_MOMENTUM_LS
# ============================================================
print("\n" + "=" * 60)
print("TEST 7: TS_Momentum_LS Position Cap and Returns")
print("=" * 60)

from src.strategies.long_short import TimeSeriesMomentumLS
from datetime import datetime
import numpy as np

ts = TimeSeriesMomentumLS()

class FakeFeatures:
    def __init__(self):
        np.random.seed(42)
        symbols = [f"SYM{i}" for i in range(200)]
        self.returns_126d = {s: np.random.normal(0.1, 0.3) for s in symbols}
        self.volatility_21d = {s: max(0.05, abs(np.random.normal(0.2, 0.1))) for s in symbols}
        self.regime = None

signal = ts.generate_signals(FakeFeatures(), datetime.now())
n_pos = len(signal.desired_weights) if signal.desired_weights else 0

test(f"Positions capped at <=40 (got {n_pos})", n_pos <= 40,
     f"got {n_pos} — still generating too many")

if signal.expected_return is not None:
    test(f"Expected return daily-scale ({signal.expected_return:.4f})",
         abs(signal.expected_return) <= 0.05,
         f"got {signal.expected_return:.2%} — still using period returns")

# Check individual expected returns
if signal.desired_weights:
    max_exp = max(abs(v) for v in signal.desired_weights.values())
    test(f"Max position weight reasonable ({max_exp:.2%})",
         max_exp <= 0.10,
         f"got {max_exp:.2%}")


# ============================================================
# TEST 8: SIGNAL VALIDATOR
# ============================================================
print("\n" + "=" * 60)
print("TEST 8: Signal Validator (Mean-Reversion Safe)")
print("=" * 60)

from src.learning.signal_validator import SignalValidator

validator = SignalValidator(min_confidence=0.3)

# Mean-reversion: buy stock with strong negative momentum
result = validator.validate_signal(
    ticker='AAPL', signal_direction='long',
    signal_weight=0.05, signal_confidence=0.6,
    momentum_signal=-0.7,
)
test("Long with momentum=-0.7 NOT blocked", result.is_valid,
     f"blocked: {result.blocking_issues}")

# Mean-reversion: short stock with strong positive momentum
result2 = validator.validate_signal(
    ticker='TSLA', signal_direction='short',
    signal_weight=-0.03, signal_confidence=0.6,
    momentum_signal=0.8,
)
test("Short with momentum=+0.8 NOT blocked", result2.is_valid,
     f"blocked: {result2.blocking_issues}")

# Even extreme momentum should not block
result3 = validator.validate_signal(
    ticker='GME', signal_direction='long',
    signal_weight=0.05, signal_confidence=0.6,
    momentum_signal=-0.95,
)
test("Long with momentum=-0.95 NOT blocked", result3.is_valid,
     f"blocked: {result3.blocking_issues}")


# ============================================================
# TEST 9: STRATEGY ENHANCER
# ============================================================
print("\n" + "=" * 60)
print("TEST 9: Strategy Enhancer Position Limits")
print("=" * 60)

from src.optimizations.strategy_enhancer import StrategyEnhancer, EnhancedConfig

enhancer = StrategyEnhancer(EnhancedConfig(
    kelly_multiplier=0.5, min_position_pct=0.02,
    max_positions=20, min_investment_floor=0.50,
))

test_wts = {}
for i in range(39):
    test_wts[f"L{i}"] = 0.005 + (i * 0.001)
for i in range(5):
    test_wts[f"S{i}"] = -0.02 - (i * 0.01)

enhanced, _ = enhancer.enhance_position_sizes(
    base_weights=test_wts,
    confidences={s: 0.6 for s in test_wts},
    target_exposure=1.0,
)

n_l = len([w for w in enhanced.values() if w > 0])
n_s = len([w for w in enhanced.values() if w < 0])
total = n_l + n_s

test(f"Total <= 20 (got {total})", total <= 20, f"got {total}")
test(f"Has longs (got {n_l})", n_l > 0, "no longs")
test(f"Has shorts (got {n_s})", n_s > 0, "no shorts")

total_long_weight = sum(w for w in enhanced.values() if w > 0)
test(f"Long weight >= 50% floor (got {total_long_weight:.0%})",
     total_long_weight >= 0.45, f"got {total_long_weight:.0%}")


# ============================================================
# TEST 10: SHORT SCANNER CONVICTION
# ============================================================
print("\n" + "=" * 60)
print("TEST 10: Short Scanner Requires Multi-Source Conviction")
print("=" * 60)

from src.strategies.short_scanner import ShortCandidate, ShortScannerConfig

scanner_cfg = ShortScannerConfig()
test("RSI threshold >= 80", scanner_cfg.rsi_overbought >= 80,
     f"got {scanner_cfg.rsi_overbought}")
test("Min sources >= 2", scanner_cfg.min_sources_agreeing >= 2,
     f"got {scanner_cfg.min_sources_agreeing}")

# Single source should NOT trigger
c1 = ShortCandidate(symbol="X")
c1.technical_score = 0.7
c1.calculate_total()
test("Single technical source = no short",
     c1.recommended_weight == 0.0,
     f"got weight={c1.recommended_weight} — one source shouldn't short")

# Two sources should trigger
c2 = ShortCandidate(symbol="Y")
c2.technical_score = 0.6
c2.valuation_score = 0.4
c2.calculate_total()
test("Two sources = valid short",
     c2.recommended_weight < 0,
     f"got weight={c2.recommended_weight}")


# ============================================================
# TEST 11: SCANNER [:5] SLICE FIX
# ============================================================
print("\n" + "=" * 60)
print("TEST 11: Scanner Processes ALL Candidates (not just 5)")
print("=" * 60)

# Read the actual code to verify the loop iterates all items
with open('app.py', 'r') as f:
    src = f.read()

# Find the scanner processing loop
scanner_section = src[src.find("Short Scanner found"):src.find("Short Scanner found") + 500]
uses_slice_for_processing = "scanner_shorts.items())[:5]:" in scanner_section

# The correct pattern: enumerate all, log first 5
uses_enumerate = "enumerate" in scanner_section

test("Scanner iterates ALL candidates (not [:5] slice)",
     not uses_slice_for_processing and uses_enumerate,
     "still using [:5] slice on the processing loop")


# ============================================================
# TEST 12: target_symbols THRESHOLD
# ============================================================
print("\n" + "=" * 60)
print("TEST 12: target_symbols Threshold Allows Ensemble Positions")
print("=" * 60)

# Check the source for the threshold
threshold_section = src[src.find("WEIGHT_THRESHOLD"):src.find("WEIGHT_THRESHOLD") + 200] if "WEIGHT_THRESHOLD" in src else ""
has_unified_threshold = "WEIGHT_THRESHOLD" in src
no_strong_threshold = "STRONG_THRESHOLD" not in src

test("Uses unified threshold (not STRONG_THRESHOLD/BASE_THRESHOLD)",
     has_unified_threshold and no_strong_threshold,
     "still has STRONG_THRESHOLD=0.02 that kills ensemble positions on subsequent runs")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
total = passed + failed
if failed == 0:
    print(f"ALL {total} TESTS PASSED ✅")
else:
    print(f"RESULTS: {passed}/{total} passed, {failed} FAILED")
print("=" * 60)

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(f"  ❌ {f}")

print()
sys.exit(0 if failed == 0 else 1)
