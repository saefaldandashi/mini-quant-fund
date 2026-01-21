"""
Unit tests for utils module.
"""
import os
import pytest
from datetime import datetime

from utils import (
    get_last_rebalance_month,
    get_current_month,
    save_rebalance_month,
    is_already_rebalanced_this_month,
    get_env_bool,
)


def test_get_current_month():
    """Test current month format."""
    month = get_current_month()
    assert len(month) == 7  # YYYY-MM
    assert month[4] == "-"
    assert month[:4].isdigit()
    assert month[5:].isdigit()


def test_save_and_read_rebalance_month(tmp_path, monkeypatch):
    """Test saving and reading rebalance month."""
    # Change to temp directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Import after changing directory so it uses the temp path
        import importlib
        import utils
        importlib.reload(utils)
        
        test_month = "2024-03"
        utils.save_rebalance_month(test_month)
        
        assert os.path.exists(utils.config.STATE_FILE)
        
        month = utils.get_last_rebalance_month()
        assert month == test_month
    finally:
        os.chdir(original_cwd)


def test_is_already_rebalanced_this_month(tmp_path, monkeypatch):
    """Test monthly idempotency check."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        import importlib
        import utils
        importlib.reload(utils)
        
        # No state file - should return False
        assert not utils.is_already_rebalanced_this_month()
        
        # Save current month
        current_month = utils.get_current_month()
        utils.save_rebalance_month(current_month)
        
        # Should return True
        assert utils.is_already_rebalanced_this_month()
        
        # Save different month
        utils.save_rebalance_month("2020-01")
        assert not utils.is_already_rebalanced_this_month()
    finally:
        os.chdir(original_cwd)


def test_get_env_bool(monkeypatch):
    """Test environment variable boolean parsing."""
    # Test True values
    for val in ["1", "TRUE", "true", "YES", "yes", "ON", "on"]:
        monkeypatch.setenv("TEST_VAR", val)
        assert get_env_bool("TEST_VAR") == True
    
    # Test False values
    for val in ["0", "FALSE", "false", "NO", "no", "OFF", "off", ""]:
        monkeypatch.setenv("TEST_VAR", val)
        assert get_env_bool("TEST_VAR") == False
    
    # Test default
    monkeypatch.delenv("TEST_VAR", raising=False)
    assert get_env_bool("TEST_VAR", default=True) == True
    assert get_env_bool("TEST_VAR", default=False) == False
