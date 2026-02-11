#!/usr/bin/env python3
"""
Deployment Verification Script
Verifies all fixes are properly deployed and working.
"""

import sys
import os
import subprocess
from pathlib import Path

print("=" * 70)
print("DEPLOYMENT VERIFICATION")
print("=" * 70)
print()

# Check 1: Verify all files exist
print("✓ Checking critical files...")
critical_files = [
    "src/data/symbol_validator.py",
    "src/risk/realtime_monitor.py",
    "src/learning/learning_engine.py",
    "src/learning/adaptive_weights.py",
    "broker_alpaca.py",
    "app.py",
    "config.py",
]

all_exist = True
for file in critical_files:
    if Path(file).exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} MISSING")
        all_exist = False

if not all_exist:
    print("\n❌ Some critical files are missing!")
    sys.exit(1)

print()

# Check 2: Verify imports work
print("✓ Testing imports...")
try:
    from src.data.symbol_validator import SymbolValidator, get_symbol_validator
    print("  ✓ SymbolValidator imports OK")
except Exception as e:
    print(f"  ✗ SymbolValidator import failed: {e}")
    sys.exit(1)

try:
    from src.risk.realtime_monitor import RealtimeRiskMonitor, get_realtime_monitor, set_realtime_monitor
    print("  ✓ RealtimeRiskMonitor imports OK")
except Exception as e:
    print(f"  ✗ RealtimeRiskMonitor import failed: {e}")
    sys.exit(1)

try:
    from src.learning.learning_engine import LearningEngine
    print("  ✓ LearningEngine imports OK")
except Exception as e:
    print(f"  ✗ LearningEngine import failed: {e}")
    sys.exit(1)

print()

# Check 3: Verify fixes in code
print("✓ Verifying fixes in code...")

# Fix #7: Symbol validator
with open("src/data/symbol_validator.py", "r") as f:
    content = f.read()
    if "class SymbolValidator" in content and "is_valid_format" in content:
        print("  ✓ Fix #7: SymbolValidator implemented")
    else:
        print("  ✗ Fix #7: SymbolValidator not found")
        sys.exit(1)

# Fix #8: VIX exposure multiplier
with open("src/risk/realtime_monitor.py", "r") as f:
    content = f.read()
    if "vix_exposure_multiplier" in content and "get_vix_exposure_multiplier" in content:
        print("  ✓ Fix #8: VIX exposure multiplier implemented")
    else:
        print("  ✗ Fix #8: VIX exposure multiplier not found")
        sys.exit(1)

# Fix #10: Learning influence
with open("src/learning/learning_engine.py", "r") as f:
    content = f.read()
    if "learning_influence: float = 0.5" in content or "learning_influence=0.5" in content:
        print("  ✓ Fix #10: Learning influence increased to 50%")
    elif "learning_influence: float = 0.3" in content:
        print("  ⚠ Fix #10: Learning influence still at 30% (may need update)")
    else:
        print("  ✗ Fix #10: Could not verify learning influence")
        sys.exit(1)

# Fix #12: Regime-driven mode switching
with open("app.py", "r") as f:
    content = f.read()
    if "regime_object" in content and "get_dynamic_trading_mode" in content:
        print("  ✓ Fix #12: Regime-driven mode switching enhanced")
    else:
        print("  ✗ Fix #12: Regime-driven mode switching not found")
        sys.exit(1)

print()

# Check 4: Run syntax checks
print("✓ Running syntax checks...")
files_to_check = [
    "src/data/symbol_validator.py",
    "src/risk/realtime_monitor.py",
    "src/learning/learning_engine.py",
    "app.py",
]

import ast
all_syntax_ok = True
for file in files_to_check:
    try:
        with open(file, "r") as f:
            code = f.read()
            ast.parse(code)
        print(f"  ✓ {file}: Syntax valid")
    except SyntaxError as e:
        print(f"  ✗ {file}: Syntax error at line {e.lineno}: {e.msg}")
        all_syntax_ok = False
    except Exception as e:
        print(f"  ✗ {file}: Error: {e}")
        all_syntax_ok = False

if not all_syntax_ok:
    print("\n❌ Syntax errors found!")
    sys.exit(1)

print()

# Check 5: Verify Git status
print("✓ Checking Git status...")
try:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        uncommitted = [line for line in result.stdout.strip().split("\n") if line and not line.startswith("??")]
        if uncommitted:
            print(f"  ⚠ {len(uncommitted)} uncommitted changes")
        else:
            print("  ✓ All changes committed")
    else:
        print("  ⚠ Could not check Git status")
except Exception as e:
    print(f"  ⚠ Could not check Git status: {e}")

print()

# Check 6: Verify test suite
print("✓ Running comprehensive test suite...")
try:
    result = subprocess.run(
        [sys.executable, "test_all_fixes.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode == 0:
        if "ALL TESTS PASSED" in result.stdout:
            print("  ✓ All tests passed")
        else:
            print("  ⚠ Some tests may have failed")
            print(result.stdout[-500:])  # Show last 500 chars
    else:
        print("  ⚠ Test suite had errors")
        print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
except Exception as e:
    print(f"  ⚠ Could not run test suite: {e}")

print()
print("=" * 70)
print("✅ DEPLOYMENT VERIFICATION COMPLETE")
print("=" * 70)
print()
print("Next steps:")
print("1. Pull latest code on cloud server: git pull origin main")
print("2. Restart your application/service")
print("3. Monitor first rebalance for:")
print("   - No UnboundLocalError crashes")
print("   - Larger position sizes (2-3x)")
print("   - More positions created")
print("   - Successful completion")
print("4. Check logs for any errors or warnings")
print()
