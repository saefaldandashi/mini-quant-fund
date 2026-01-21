"""
Strategy logic: momentum calculation, moving averages, and stock selection.
"""
import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

import config


def calculate_moving_average(prices: pd.Series, window: int) -> float:
    """
    Calculate moving average of prices.
    
    Args:
        prices: Series of closing prices
        window: Window size for MA
        
    Returns:
        Moving average value
    """
    if len(prices) < window:
        raise ValueError(f"Insufficient data: need {window} days, got {len(prices)}")
    
    return prices.tail(window).mean()


def calculate_momentum(
    prices: pd.Series,
    lookback_days: int = config.LOOKBACK_DAYS
) -> float:
    """
    Calculate momentum as (current_price / price_N_days_ago) - 1.
    
    Args:
        prices: Series of closing prices (sorted by date, ascending)
        lookback_days: Number of days to look back
        
    Returns:
        Momentum value (e.g., 0.15 for 15% gain)
    """
    if len(prices) < lookback_days + 1:
        raise ValueError(
            f"Insufficient data: need {lookback_days + 1} days, got {len(prices)}"
        )
    
    current_price = prices.iloc[-1]
    past_price = prices.iloc[-lookback_days - 1]
    
    if past_price <= 0:
        raise ValueError(f"Invalid past price: {past_price}")
    
    momentum = (current_price / past_price) - 1.0
    return momentum


def determine_market_regime(
    spy_prices: pd.Series,
    ma_days: int = config.MA_DAYS
) -> Tuple[bool, float, float]:
    """
    Determine if market is risk-on (SPY > 200-day MA) or risk-off.
    
    Args:
        spy_prices: Series of SPY closing prices (sorted by date, ascending)
        ma_days: Days for moving average
        
    Returns:
        Tuple of (risk_on: bool, last_close: float, ma_value: float)
    """
    if len(spy_prices) < ma_days:
        raise ValueError(f"Insufficient SPY data: need {ma_days} days, got {len(spy_prices)}")
    
    last_close = spy_prices.iloc[-1]
    ma_value = calculate_moving_average(spy_prices, ma_days)
    
    risk_on = last_close > ma_value
    
    logging.info(
        f"Market regime: SPY close={last_close:.2f}, MA{ma_days}={ma_value:.2f}, "
        f"risk_on={risk_on}"
    )
    
    return risk_on, last_close, ma_value


def select_top_momentum_stocks(
    universe_data: Dict[str, pd.Series],
    lookback_days: int = config.LOOKBACK_DAYS,
    top_n: int = config.TOP_N
) -> List[Tuple[str, float]]:
    """
    Select top N stocks by momentum from universe.
    
    Args:
        universe_data: Dict mapping symbol to Series of closing prices
        lookback_days: Days to look back for momentum
        top_n: Number of top stocks to select
        
    Returns:
        List of (symbol, momentum) tuples, sorted by momentum descending
    """
    momentum_scores = []
    
    for symbol, prices in universe_data.items():
        try:
            if len(prices) < lookback_days + 1:
                logging.warning(f"Insufficient data for {symbol}: {len(prices)} days")
                continue
            
            momentum = calculate_momentum(prices, lookback_days)
            momentum_scores.append((symbol, momentum))
            logging.debug(f"{symbol}: momentum = {momentum:.4f}")
        except Exception as e:
            logging.warning(f"Error calculating momentum for {symbol}: {e}")
            continue
    
    if not momentum_scores:
        raise ValueError("No valid momentum scores calculated")
    
    # Sort by momentum descending
    momentum_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_stocks = momentum_scores[:top_n]
    
    logging.info(f"Top {top_n} stocks by momentum:")
    for symbol, momentum in top_stocks:
        logging.info(f"  {symbol}: {momentum:.4f} ({momentum*100:.2f}%)")
    
    return top_stocks


def calculate_target_positions(
    top_stocks: List[Tuple[str, float]],
    equity: float,
    cash_buffer_pct: float = config.CASH_BUFFER_PCT
) -> Dict[str, int]:
    """
    Calculate target share quantities for equal-weight portfolio.
    
    Args:
        top_stocks: List of (symbol, momentum) tuples
        equity: Total account equity
        cash_buffer_pct: Percentage to keep as cash buffer
        
    Returns:
        Dict mapping symbol to target share quantity (whole shares)
    """
    if not top_stocks:
        return {}
    
    investable_equity = equity * (1.0 - cash_buffer_pct)
    num_positions = len(top_stocks)
    target_notional_per_stock = investable_equity / num_positions
    
    target_positions = {}
    
    for symbol, _ in top_stocks:
        # We'll need current price to calculate shares, but for now return 0
        # This will be filled in by the broker when we have prices
        target_positions[symbol] = 0
    
    logging.info(
        f"Target positions: {num_positions} stocks, "
        f"${target_notional_per_stock:.2f} per stock (from ${investable_equity:.2f} investable)"
    )
    
    return target_positions
