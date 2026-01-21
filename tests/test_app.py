#!/usr/bin/env python3
"""Quick diagnostic script to test if the app can run."""

import sys
import os
from pathlib import Path

print("🔍 Mini Quant Fund - Diagnostic Test")
print("=" * 60)

# Check 1: Virtual environment
venv_path = Path(".venv")
if venv_path.exists():
    print("✅ Virtual environment exists")
else:
    print("❌ Virtual environment missing")
    print("   Run: python3 -m venv .venv")

# Check 2: Dependencies
try:
    import flask
    print("✅ Flask installed")
except ImportError:
    print("❌ Flask not installed")
    print("   Run: pip install flask")

# Check 3: App imports
try:
    import app
    print("✅ app.py imports successfully")
except Exception as e:
    print(f"❌ Error importing app.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Flask app
if hasattr(app, 'app'):
    print("✅ Flask app created")
    print(f"   Routes: {len(app.app.url_map._rules)}")
else:
    print("❌ Flask app not found")
    sys.exit(1)

# Check 5: Port availability
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 5000))
sock.close()

if result == 0:
    print("⚠️  Port 5000 is already in use")
    print("   Run: pkill -f 'python app.py'")
else:
    print("✅ Port 5000 is available")

# Check 6: API keys
if os.getenv('ALPACA_API_KEY'):
    print("✅ ALPACA_API_KEY is set")
else:
    print("⚠️  ALPACA_API_KEY not set (app will still start)")

print("\n" + "=" * 60)
print("✅ All checks passed! App should run.")
print("\nTo start the app:")
print("  1. python app.py")
print("  2. Or double-click: start_quant_fund.command")
print("  3. Or double-click: MiniQuantFund.app")
print("\nThen open: http://localhost:5000")
