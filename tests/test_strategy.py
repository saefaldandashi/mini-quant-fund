"""
Unit tests for strategy module.
"""
import pytest
import pandas as pd
import numpy as np

from strategy import (
    calculate_moving_average,
    calculate_momentum,
    determine_market_regime,
    select_top_momentum_stocks,
)


def test_calculate_moving_average():
    """Test moving average calculation."""
    # Create test data: 10 days of prices
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    
    ma = calculate_moving_average(prices, window=5)
    expected = (105 + 106 + 107 + 108 + 109) / 5
    assert abs(ma - expected) < 0.01
    
    # Test with insufficient data
    with pytest.raises(ValueError):
        calculate_moving_average(prices, window=20)


def test_calculate_momentum():
    """Test momentum calculation."""
    # Create test data: 127 days (enough for 126-day lookback)
    base_price = 100.0
    prices = pd.Series([base_price * (1.0 + i * 0.001) for i in range(127)])
    
    momentum = calculate_momentum(prices, lookback_days=126)
    
    # Expected: (last_price / first_price) - 1
    expected = (prices.iloc[-1] / prices.iloc[0]) - 1.0
    assert abs(momentum - expected) < 0.0001
    
    # Test with insufficient data
    short_prices = pd.Series([100, 101, 102])
    with pytest.raises(ValueError):
        calculate_momentum(short_prices, lookback_days=126)


def test_determine_market_regime():
    """Test market regime determination."""
    # Create SPY data: 200 days
    # Last 50 days above MA, first 150 below
    base_price = 400.0
    prices = pd.Series([base_price - 10] * 150 + [base_price + 10] * 50)
    
    risk_on, last_close, ma_value = determine_market_regime(prices, ma_days=200)
    
    assert last_close == prices.iloc[-1]
    assert ma_value == prices.mean()
    assert risk_on == (last_close > ma_value)
    
    # Test with insufficient data
    short_prices = pd.Series([400, 401, 402])
    with pytest.raises(ValueError):
        determine_market_regime(short_prices, ma_days=200)


def test_select_top_momentum_stocks():
    """Test top momentum stock selection."""
    # Create test data for 3 stocks
    base_price = 100.0
    
    # Stock A: high momentum (20% gain)
    prices_a = pd.Series([base_price] * 100 + [base_price * 1.20])
    
    # Stock B: medium momentum (10% gain)
    prices_b = pd.Series([base_price] * 100 + [base_price * 1.10])
    
    # Stock C: low momentum (5% gain)
    prices_c = pd.Series([base_price] * 100 + [base_price * 1.05])
    
    universe_data = {
        "STOCK_A": prices_a,
        "STOCK_B": prices_b,
        "STOCK_C": prices_c,
    }
    
    top_stocks = select_top_momentum_stocks(
        universe_data,
        lookback_days=100,
        top_n=2
    )
    
    assert len(top_stocks) == 2
    assert top_stocks[0][0] == "STOCK_A"  # Highest momentum
    assert top_stocks[1][0] == "STOCK_B"  # Second highest
    
    # Verify momentum values
    assert top_stocks[0][1] > top_stocks[1][1]


def test_select_top_momentum_stocks_insufficient_data():
    """Test that stocks with insufficient data are skipped."""
    universe_data = {
        "STOCK_A": pd.Series([100, 101, 102]),  # Insufficient
        "STOCK_B": pd.Series([100.0] * 127),  # Sufficient
    }
    
    top_stocks = select_top_momentum_stocks(
        universe_data,
        lookback_days=126,
        top_n=5
    )
    
    # Should only return STOCK_B
    assert len(top_stocks) == 1
    assert top_stocks[0][0] == "STOCK_B"
