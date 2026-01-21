"""
Pytest configuration and shared fixtures.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def reproducible_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(None)


@pytest.fixture
def sample_universe():
    """Sample stock universe."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']


@pytest.fixture
def sample_date_range():
    """Sample date range for testing."""
    return datetime(2024, 1, 1), datetime(2024, 6, 1)
