"""
Utility functions for retries, state management, and logging.
"""
import os
import time
import logging
from datetime import datetime
from typing import Callable, Any, Optional
from functools import wraps

import config


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with timestamp and level."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logging.error(f"{func.__name__} failed after {max_retries} attempts")
            
            raise last_exception
        return wrapper
    return decorator


def get_last_rebalance_date() -> Optional[str]:
    """
    Read the last rebalance date from state file.
    
    Returns:
        YYYY-MM-DD string if file exists, None otherwise
    """
    if not os.path.exists(config.STATE_FILE):
        return None
    
    try:
        with open(config.STATE_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                return content
    except Exception as e:
        logging.warning(f"Error reading state file: {e}")
    
    return None


def get_current_date() -> str:
    """Get current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def save_rebalance_date(date: str) -> None:
    """Save the rebalance date to state file."""
    try:
        with open(config.STATE_FILE, 'w') as f:
            f.write(date)
        logging.info(f"Saved rebalance state: {date}")
    except Exception as e:
        logging.error(f"Error saving state file: {e}")
        raise


def is_already_rebalanced_today() -> bool:
    """
    Check if we've already rebalanced today.
    
    Returns:
        True if already rebalanced today, False otherwise
    """
    last_date = get_last_rebalance_date()
    current_date = get_current_date()
    
    if last_date == current_date:
        logging.info(f"Already rebalanced today ({current_date}). Exiting.")
        return True
    
    return False


# Backward compatibility aliases
def get_last_rebalance_month() -> Optional[str]:
    """Deprecated: Use get_last_rebalance_date() instead."""
    date = get_last_rebalance_date()
    if date:
        return date[:7]  # Return YYYY-MM
    return None


def get_current_month() -> str:
    """Deprecated: Use get_current_date() instead."""
    return get_current_date()[:7]  # Return YYYY-MM


def save_rebalance_month(month: str) -> None:
    """Deprecated: Use save_rebalance_date() instead."""
    # For backward compatibility, append -01 to make it a valid date
    save_rebalance_date(f"{month}-01")


def is_already_rebalanced_this_month() -> bool:
    """Deprecated: Use is_already_rebalanced_today() instead."""
    return is_already_rebalanced_today()


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    
    value = value.strip().upper()
    if value in ("1", "TRUE", "YES", "ON"):
        return True
    elif value in ("0", "FALSE", "NO", "OFF", ""):
        return False
    return default
