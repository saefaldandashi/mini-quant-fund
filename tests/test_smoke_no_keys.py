"""
Smoke test: verify bot exits gracefully with missing API keys.
"""
import os
import pytest
import subprocess
import sys


def test_bot_exits_with_missing_keys():
    """Test that bot exits with clear error message when keys are missing."""
    # Remove any existing keys from environment
    env = os.environ.copy()
    env.pop("ALPACA_API_KEY", None)
    env.pop("ALPACA_SECRET_KEY", None)
    env["DRY_RUN"] = "1"
    env["LOG_LEVEL"] = "INFO"
    env["FORCE_REBALANCE"] = "1"  # Bypass daily limit to test key check
    
    # Run bot.py
    result = subprocess.run(
        [sys.executable, "bot.py"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=10
    )
    
    # Should exit with code 1
    assert result.returncode == 1
    
    # Should contain error message about missing keys
    assert "ALPACA_API_KEY" in result.stderr or "ALPACA_API_KEY" in result.stdout
    assert "ALPACA_SECRET_KEY" in result.stderr or "ALPACA_SECRET_KEY" in result.stdout
