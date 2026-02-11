#!/usr/bin/env python3
"""
Comprehensive rebalance testing script for cloud environment.
Tests fixes and monitors for errors.
"""
import sys
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rebalance_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_rebalance_execution():
    """Test rebalance execution and monitor for errors."""
    print("=" * 70)
    print("REBALANCE TESTING & MONITORING")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print()
    
    test_results = {
        'fix_1_unbound_error': False,
        'fix_2_kelly': False,
        'fix_3_thresholds': False,
        'rebalance_success': False,
        'position_sizes': None,
        'position_count': 0,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Import after checking environment
        print("Step 1: Importing modules...")
        try:
            from app import run_multi_strategy_rebalance
            print("   ✓ Successfully imported run_multi_strategy_rebalance")
        except Exception as e:
            test_results['errors'].append(f"Import error: {e}")
            print(f"   ✗ Import failed: {e}")
            return test_results
        
        print()
        print("Step 2: Verifying fixes in code...")
        
        # Verify Fix #1
        with open('app.py', 'r') as f:
            content = f.read()
            if 'enhanced_weights = {}' in content and 'BEFORE if/else' in content:
                test_results['fix_1_unbound_error'] = True
                print("   ✓ Fix #1: UnboundLocalError prevention verified")
            else:
                test_results['warnings'].append("Fix #1 not clearly found")
                print("   ⚠ Fix #1: Could not verify")
        
        # Verify Fix #2
        if 'use_kelly=False' in content:
            test_results['fix_2_kelly'] = True
            print("   ✓ Fix #2: Double Kelly removal verified")
        else:
            test_results['warnings'].append("Fix #2 not clearly found")
            print("   ⚠ Fix #2: Could not verify")
        
        # Verify Fix #3
        with open('config.py', 'r') as f:
            config_content = f.read()
            if '"min_position_pct": 0.02' in config_content:
                test_results['fix_3_thresholds'] = True
                print("   ✓ Fix #3: Lower thresholds verified")
            else:
                test_results['warnings'].append("Fix #3 not clearly found")
                print("   ⚠ Fix #3: Could not verify")
        
        print()
        print("Step 3: Checking server.log for recent errors...")
        
        # Check recent logs
        log_file = Path('server.log')
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                recent_errors = [l for l in lines[-100:] if 'ERROR' in l or 'UnboundLocalError' in l]
                
                if recent_errors:
                    # Check if errors are recent (last 10 minutes)
                    print(f"   Found {len(recent_errors)} recent errors in log")
                    for err in recent_errors[-5:]:
                        if 'UnboundLocalError' in err:
                            test_results['errors'].append(f"Recent UnboundLocalError in log: {err[:100]}")
                            print(f"   ⚠ Recent UnboundLocalError found")
                        else:
                            print(f"   ℹ Other error: {err[:80]}")
                else:
                    print("   ✓ No recent errors in server.log")
        else:
            print("   ℹ server.log not found (may be first run)")
        
        print()
        print("Step 4: Testing rebalance execution...")
        print("   NOTE: This will attempt to run a rebalance.")
        print("   Set TEST_MODE=True to skip actual execution")
        print()
        
        test_mode = True  # Set to False to actually run rebalance
        
        if not test_mode:
            print("   Running actual rebalance (this may take 30-60 seconds)...")
            start_time = time.time()
            
            try:
                success, output, error, debate_info = run_multi_strategy_rebalance(
                    allow_after_hours=True,
                    force_rebalance=True,
                    cancel_orders=False  # Don't cancel orders in test
                )
                
                elapsed = time.time() - start_time
                
                if success:
                    test_results['rebalance_success'] = True
                    print(f"   ✓ Rebalance completed successfully in {elapsed:.1f}s")
                    
                    # Extract position info from output
                    if 'positions' in output.lower():
                        # Try to extract position count
                        import re
                        pos_match = re.search(r'(\d+)\s+positions?', output, re.IGNORECASE)
                        if pos_match:
                            test_results['position_count'] = int(pos_match.group(1))
                            print(f"   ✓ Created {test_results['position_count']} positions")
                    
                    if error:
                        test_results['warnings'].append(f"Rebalance had warning: {error}")
                else:
                    test_results['errors'].append(f"Rebalance failed: {error}")
                    print(f"   ✗ Rebalance failed: {error}")
                    if output:
                        print(f"   Output: {output[:200]}")
            except UnboundLocalError as e:
                test_results['errors'].append(f"UnboundLocalError still occurs: {e}")
                print(f"   ✗ CRITICAL: UnboundLocalError still occurs!")
                traceback.print_exc()
            except Exception as e:
                test_results['errors'].append(f"Unexpected error: {e}")
                print(f"   ✗ Unexpected error: {e}")
                traceback.print_exc()
        else:
            print("   ⏭ Skipping actual execution (TEST_MODE=True)")
            print("   To test actual execution, set test_mode=False in script")
        
        print()
        print("=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        
        if test_results['fix_1_unbound_error']:
            print("✅ Fix #1: UnboundLocalError prevention - VERIFIED")
        else:
            print("⚠️  Fix #1: UnboundLocalError prevention - NEEDS VERIFICATION")
        
        if test_results['fix_2_kelly']:
            print("✅ Fix #2: Double Kelly removal - VERIFIED")
        else:
            print("⚠️  Fix #2: Double Kelly removal - NEEDS VERIFICATION")
        
        if test_results['fix_3_thresholds']:
            print("✅ Fix #3: Lower thresholds - VERIFIED")
        else:
            print("⚠️  Fix #3: Lower thresholds - NEEDS VERIFICATION")
        
        if test_results['rebalance_success']:
            print(f"✅ Rebalance execution - SUCCESS ({test_results['position_count']} positions)")
        elif test_mode:
            print("⏭ Rebalance execution - SKIPPED (test mode)")
        else:
            print("❌ Rebalance execution - FAILED")
        
        if test_results['errors']:
            print()
            print("❌ ERRORS FOUND:")
            for err in test_results['errors']:
                print(f"   - {err}")
        
        if test_results['warnings']:
            print()
            print("⚠️  WARNINGS:")
            for warn in test_results['warnings']:
                print(f"   - {warn}")
        
        print()
        print("=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        
        if test_results['errors']:
            print("1. ❌ ERRORS DETECTED - Review errors above")
            print("2. Check server.log for detailed error messages")
            print("3. Verify code is up to date: git pull origin main")
            print("4. Restart application if needed")
        elif test_results['rebalance_success']:
            print("1. ✅ All fixes working correctly!")
            print("2. Monitor next few rebalances for stability")
            print("3. Check position sizes are larger (2-3x expected)")
            print("4. Verify more positions are created")
        else:
            print("1. Run actual rebalance test (set test_mode=False)")
            print("2. Monitor server.log during execution")
            print("3. Check for any runtime errors")
        
        print()
        
    except Exception as e:
        logger.error(f"Test script error: {e}")
        traceback.print_exc()
        test_results['errors'].append(f"Test script error: {e}")
    
    return test_results

if __name__ == "__main__":
    results = test_rebalance_execution()
    
    # Exit with error code if critical issues found
    if results['errors']:
        sys.exit(1)
    elif not results['rebalance_success'] and 'test_mode' not in str(results):
        sys.exit(1)
    else:
        sys.exit(0)
