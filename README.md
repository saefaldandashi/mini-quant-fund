# Multi-Strategy Quant Debate Bot

A research-grade, multi-strategy trading system with a unique "debate" mechanism for portfolio allocation. The bot implements 10 investing strategies that propose trades, then uses an ensemble/debate layer to decide final allocations based on market data, regime analysis, and sentiment.

## 🎯 Key Features

- **10 Trading Strategies** - From momentum to mean-reversion to risk parity
- **Debate Engine** - Strategies "debate" and the best proposals win
- **Regime Classification** - Adapts to trend, volatility, and risk regimes
- **Sentiment Analysis** - Incorporates news sentiment into decisions
- **Risk Management** - Position limits, sector exposure, volatility targeting
- **Walk-Forward Backtesting** - With transaction costs and attribution
- **Paper Trading Mode** - Generate signals without execution
- **Extensible Architecture** - Easy to add new strategies and data sources

## 📊 Implemented Strategies

| Strategy | Description |
|----------|-------------|
| **TimeSeriesMomentum** | Long assets with positive momentum, scaled by volatility |
| **CrossSectionMomentum** | Rank assets, long winners (12-1 month momentum) |
| **MeanReversion** | Fade oversold/overbought moves relative to MA |
| **VolatilityRegimeVolTarget** | Adjust exposure to maintain constant portfolio risk |
| **CarryStrategy** | Favor high-yield assets (dividend proxy for equities) |
| **ValueQualityTilt** | Tilt towards value and quality factors |
| **RiskParityMinVar** | Equal risk contribution or minimum variance |
| **TailRiskOverlay** | Reduce exposure when tail risk is elevated |
| **NewsSentimentEvent** | Trade on sentiment signals from news |
| **MLMetaEnsemble** | Learn optimal strategy weights over time |

## 🏗️ Project Structure

```
mini fund tool/
├── src/
│   ├── data/
│   │   ├── market_data.py      # OHLCV data loading
│   │   ├── news_data.py        # News ingestion & entity linking
│   │   ├── sentiment.py        # Sentiment analysis
│   │   ├── regime.py           # Market regime classification
│   │   └── feature_store.py    # Timestamped feature computation
│   │
│   ├── strategies/
│   │   ├── base.py             # Strategy base class
│   │   ├── momentum.py         # Time-series & cross-section momentum
│   │   ├── mean_reversion.py   # Mean reversion
│   │   ├── volatility.py       # Vol targeting
│   │   ├── carry.py            # Carry strategy
│   │   ├── value_quality.py    # Value/quality factors
│   │   ├── risk_parity.py      # Risk parity & min variance
│   │   ├── tail_risk.py        # Tail risk overlay
│   │   ├── sentiment_event.py  # News sentiment
│   │   └── ml_ensemble.py      # Meta-learning ensemble
│   │
│   ├── debate/
│   │   ├── debate_engine.py    # Strategy evaluation & scoring
│   │   └── ensemble.py         # Weight combination methods
│   │
│   ├── risk/
│   │   └── risk_manager.py     # Constraint enforcement
│   │
│   ├── backtest/
│   │   └── backtester.py       # Walk-forward backtesting
│   │
│   └── cli/
│       └── main.py             # Command-line interface
│
├── configs/
│   └── default.yaml            # Default configuration
│
├── tests/                      # Unit tests
├── outputs/                    # Backtest results
├── app.py                      # Web UI (from original bot)
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project
cd "mini fund tool"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Set API Keys (Optional)

For real market data, set your Alpaca API keys:

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

Or create a `.env` file:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

### Run Backtest

```bash
# Run backtest with default config
python -m src.cli.main backtest

# With custom dates
python -m src.cli.main backtest --start-date 2024-01-01 --end-date 2024-12-31

# With custom config
python -m src.cli.main backtest --config configs/default.yaml
```

### Paper Trading (Signal Generation)

```bash
# Generate current trading signals
python -m src.cli.main papertrade
```

### Run Tests

```bash
pytest tests/ -v
```

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Strategy settings
strategies:
  time_series_momentum:
    enabled: true
    lookback: 126
    vol_target: 0.10
    
# Risk constraints
risk:
  max_position: 0.15
  max_sector: 0.30
  vol_target: 0.12
  
# Backtest settings
backtest:
  frequency: weekly
  initial_capital: 100000
  ensemble_mode: weighted_vote  # or convex_optimization, stacking
```

## 🔄 The Debate Process

At each rebalance:

1. **Signal Generation**: Each strategy produces `SignalOutput` with weights, expected return, risk, confidence, and explanation

2. **Debate Scoring**: The DebateEngine scores each strategy on:
   - Alpha potential (expected return vs risk)
   - Regime fit (how well it matches current market conditions)
   - Diversification contribution
   - Drawdown awareness
   - Sentiment alignment

3. **Consensus Building**: 
   - Identifies agreements (multiple strategies agree on positions)
   - Highlights disagreements
   - Lists top risks

4. **Ensemble Combination**:
   - **Weighted Vote**: Score-weighted average of signals
   - **Convex Optimization**: Maximize return - λ·risk
   - **Stacking**: Meta-model weighting

5. **Risk Check**: Final weights pass through RiskManager for constraint enforcement

## 📈 Regime Classification

The bot classifies market regimes on three dimensions:

- **Trend**: Strong Up → Neutral → Strong Down
- **Volatility**: Low → Normal → High → Extreme
- **Risk**: Risk-On → Neutral → Risk-Off

Strategies adapt their signals based on regime. For example:
- Momentum strategies reduce confidence in neutral/choppy markets
- Mean reversion increases confidence in range-bound markets
- Tail risk overlay reduces exposure in high-vol risk-off environments

## 🛡️ Risk Management

Built-in constraints:
- **Position Size**: Max 15% per stock (configurable)
- **Sector Exposure**: Max 30% per sector
- **Leverage**: Max 1.0x (no leverage by default)
- **Turnover**: Max 50% per rebalance
- **Volatility Target**: 12% annualized
- **Drawdown Trigger**: Reduce exposure at 15% drawdown

## 📊 Sample Output

```
==================================================
BACKTEST RESULTS
==================================================
Total Return: 18.45%
CAGR: 15.32%
Sharpe Ratio: 1.24
Sortino Ratio: 1.89
Max Drawdown: 8.72%
Avg Turnover: 23.45%
==================================================
STRATEGY CONTRIBUTIONS:
  RiskParityMinVar: 18.5%
  CrossSectionMomentum: 16.2%
  TimeSeriesMomentum: 14.8%
  ...
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_strategies.py -v
```

## 📝 Adding a New Strategy

1. Create a new file in `src/strategies/`:

```python
from .base import Strategy, SignalOutput

class MyNewStrategy(Strategy):
    def __init__(self, config=None):
        super().__init__("MyNewStrategy", config)
        self._required_features = ['prices', 'returns_21d']
    
    def generate_signals(self, features, t):
        weights = {}
        # Your logic here
        return SignalOutput(
            strategy_name=self.name,
            timestamp=t,
            desired_weights=weights,
            expected_return=...,
            risk_estimate=...,
            confidence=...,
            explanation={...},
        )
```

2. Register in `src/strategies/__init__.py`
3. Add to `src/cli/main.py` in `create_strategies()`
4. Add config section in `configs/default.yaml`

## 🔌 Adding a New Data Source

1. Create adapter in `src/data/market_data.py`:

```python
def _load_my_source(self, symbol, start, end):
    # Fetch data from your source
    return df  # DataFrame with open, high, low, close, volume
```

2. Add to `load_ohlcv()` method
3. Update config with new source name

## ⚠️ Important Notes

- **No Live Trading**: Default mode is backtest + paper trading only
- **No Look-Ahead Bias**: All features are point-in-time aligned
- **Reproducibility**: Use random seeds for consistent results
- **Transaction Costs**: Included in backtests (default 10bps + 5bps slippage)

## 🎨 Web UI

The project includes a web UI from the original mini fund tool:

```bash
python app.py
# Visit http://localhost:5000
```

## 📄 License

MIT License - see LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

Built with ❤️ for quantitative research and education.
