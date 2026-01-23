"""
Configuration constants for the mini quant fund bot.
Enhanced with 300 stocks and strategy enhancement settings.
"""

# ============================================================
# STOCK UNIVERSE - Top 300 US Large/Mid Cap Stocks
# ============================================================
UNIVERSE = [
    # === TECHNOLOGY (60 stocks) ===
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "CSCO", "ACN",
    "INTC", "IBM", "QCOM", "TXN", "AMAT", "LRCX", "MU", "INTU", "NOW", "ADI",
    "KLAC", "SNPS", "CDNS", "PANW", "MRVL", "FTNT", "NXPI", "MCHP", "TEL", "HPQ",
    "KEYS", "ON", "CTSH", "GLW", "ANSS", "ZBRA", "CDW", "AKAM", "EPAM", "FFIV",
    "JNPR", "NTAP", "WDC", "STX", "SWKS", "QRVO", "SEDG", "ENPH", "FSLR", "RUN",
    "ANET", "CRWD", "DDOG", "ZS", "OKTA", "NET", "MDB", "SNOW", "PLTR", "PATH",
    
    # === FINANCE (50 stocks) ===
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "BK", "COF", "CME", "ICE", "MCO", "SPGI", "MSCI", "FIS",
    "FISV", "ADP", "PAYX", "GPN", "FLT", "SYF", "DFS", "CFG", "KEY", "RF",
    "HBAN", "MTB", "FITB", "ZION", "CMA", "NTRS", "STT", "BRO", "AJG", "MMC",
    "AON", "WTW", "CINF", "L", "ALL", "TRV", "PGR", "CB", "MET", "PRU",
    
    # === HEALTHCARE (50 stocks) ===
    "UNH", "LLY", "JNJ", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "VRTX", "REGN", "MRNA", "BIIB", "ILMN", "DXCM", "ISRG", "SYK",
    "MDT", "BSX", "EW", "ZBH", "IDXX", "IQV", "A", "MTD", "WAT", "HOLX",
    "ALGN", "TECH", "BIO", "CRL", "PKI", "ELV", "HUM", "CI", "CNC", "MOH",
    "CVS", "MCK", "ABC", "CAH", "WBA", "VTRS", "ZTS", "CTLT", "DGX", "LH",
    
    # === CONSUMER DISCRETIONARY (40 stocks) ===
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR",
    "HLT", "CMG", "YUM", "DRI", "ORLY", "AZO", "BBY", "DHI", "LEN", "PHM",
    "NVR", "GRMN", "POOL", "ULTA", "RCL", "CCL", "NCLH", "EXPE", "LVS", "WYNN",
    "MGM", "F", "GM", "APTV", "BWA", "LEA", "RL", "TPR", "VFC", "PVH",
    
    # === CONSUMER STAPLES (30 stocks) ===
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "K", "CAG", "SJM", "HSY", "HRL", "TSN", "MNST", "KDP", "STZ",
    "BF.B", "TAP", "EL", "CHD", "CLX", "KHC", "CPB", "MKC", "SYY", "ADM",
    
    # === INDUSTRIALS (40 stocks) ===
    "CAT", "DE", "HON", "UNP", "UPS", "RTX", "LMT", "BA", "GE", "GD",
    "NOC", "TXT", "HII", "LHX", "TDG", "AXON", "ETN", "EMR", "ROK", "AME",
    "ITW", "PH", "DOV", "FAST", "ODFL", "JBHT", "CSX", "NSC", "FDX", "CHRW",
    "WAB", "GWW", "CTAS", "CPRT", "PCAR", "CARR", "OTIS", "JCI", "LII", "TT",
    
    # === ENERGY (20 stocks) ===
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD",
    "DVN", "HAL", "BKR", "FANG", "HES", "KMI", "WMB", "OKE", "TRGP", "LNG",
    
    # === COMMUNICATION SERVICES (15 stocks) ===
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR",
    "EA", "TTWO", "MTCH", "OMC", "IPG",
    
    # === UTILITIES (15 stocks) ===
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PEG", "ED",
    "WEC", "ES", "AWK", "ATO", "NI",
    
    # === REAL ESTATE (15 stocks) ===
    "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    "EQR", "VTR", "ARE", "MAA", "UDR",
    
    # === MATERIALS (15 stocks) ===
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "STLD", "VMC",
    "MLM", "ALB", "CF", "MOS", "IFF",
]

# Benchmark for market regime filter
BENCHMARK = "SPY"

# ============================================================
# STRATEGY PARAMETERS
# ============================================================
TOP_N = 15  # Increased from 5 to allow more positions
LOOKBACK_DAYS = 126  # ~6 months
MA_DAYS = 200  # 200-day moving average

# State file for daily idempotency
STATE_FILE = "state_last_rebalance.txt"

# Cash buffer (use 95% of equity, keep 5% buffer)
CASH_BUFFER_PCT = 0.05

# ============================================================
# POSITION SIZING SETTINGS (Phase 1)
# ============================================================
RISK_APPETITE_SETTINGS = {
    "conservative": {
        "kelly_multiplier": 0.25,
        "min_position_pct": 0.01,  # 1% minimum
        "max_positions": 30,
        "description": "Low risk, many small positions"
    },
    "moderate": {
        "kelly_multiplier": 0.50,
        "min_position_pct": 0.02,  # 2% minimum
        "max_positions": 20,
        "description": "Balanced risk and reward"
    },
    "aggressive": {
        "kelly_multiplier": 0.75,
        "min_position_pct": 0.03,  # 3% minimum
        "max_positions": 15,
        "description": "Higher conviction, concentrated"
    },
    "maximum": {
        "kelly_multiplier": 1.00,
        "min_position_pct": 0.05,  # 5% minimum
        "max_positions": 10,
        "description": "Full Kelly, highly concentrated"
    }
}

# Minimum investment floor (scale up if below this)
MIN_INVESTMENT_FLOOR_PCT = 0.50  # At least 50% of target exposure

# ============================================================
# SENTIMENT THRESHOLDS (Phase 2)
# ============================================================
SENTIMENT_THRESHOLDS = {
    "min_relevance": 0.25,  # Article must be this relevant to ticker
    "min_magnitude": 0.10,  # Sentiment must be at least this strong
    "min_news_count": 2,    # Minimum articles per week
    "recency_decay": {
        "1_day": 1.0,
        "3_days": 0.7,
        "7_days": 0.4,
        "older": 0.1
    }
}

# ============================================================
# UNIVERSE FILTERING (Phase 3)
# ============================================================
UNIVERSE_FILTER_SETTINGS = {
    "max_tradeable_stocks": 25,  # Filter down to this many
    "require_sentiment": True,   # Must have recent news
    "require_momentum": True,    # Must be above 20-day MA
    "min_volume_usd": 5_000_000, # $5M daily volume minimum
}

# Sector concentration limits
SECTOR_LIMITS = {
    "Technology": 0.35,
    "Finance": 0.25,
    "Healthcare": 0.25,
    "Consumer": 0.25,
    "Industrials": 0.20,
    "Energy": 0.15,
    "Communication": 0.15,
    "Utilities": 0.10,
    "Real Estate": 0.10,
    "Materials": 0.10,
}

# Position size tiers based on conviction
CONVICTION_TIERS = {
    "very_high": {"min_score": 0.8, "position_pct": 0.10, "max_stocks": 3},
    "high": {"min_score": 0.6, "position_pct": 0.07, "max_stocks": 5},
    "medium": {"min_score": 0.4, "position_pct": 0.04, "max_stocks": 10},
    "low": {"min_score": 0.0, "position_pct": 0.00, "max_stocks": 0},  # Don't trade
}

# ============================================================
# REGIME DETECTION (Phase 4)
# ============================================================
REGIME_SETTINGS = {
    "indicators": {
        "spy_vs_200ma": 0.25,      # Weight for SPY vs 200-day MA
        "vix_level": 0.20,         # Weight for VIX
        "macro_sentiment": 0.20,   # Weight for macro risk sentiment
        "geo_risk": 0.15,          # Weight for geopolitical risk
        "financial_stress": 0.10, # Weight for financial stress
        "breadth": 0.10,           # Weight for market breadth
    },
    "thresholds": {
        "risk_on": 0.6,    # Above this = risk on
        "risk_off": 0.4,   # Below this = risk off
    },
    "exposure_adjustments": {
        "strong_bull": 1.00,   # Full exposure
        "mild_bull": 0.80,
        "neutral": 0.60,
        "mild_bear": 0.40,
        "strong_bear": 0.20,
    }
}

# VIX thresholds
VIX_THRESHOLDS = {
    "low": 15,      # Below = bullish
    "normal": 20,   # Normal range
    "elevated": 25, # Caution
    "high": 30,     # Reduce exposure
    "extreme": 40,  # Minimal exposure
}

# ============================================================
# DEBATE SYSTEM (Phase 5)
# ============================================================
DEBATE_SETTINGS = {
    "min_consensus": 4,  # Minimum strategies agreeing to trade
    "consensus_scaling": {
        7: 1.0,   # 7+ agree = full position
        5: 0.7,   # 5-6 agree = 70% position
        4: 0.4,   # 4 agree = 40% position
    },
    "track_record_boost": {
        "top_3": 1.5,    # Top performers get 1.5x weight
        "middle_3": 1.0,
        "bottom_3": 0.5,
    }
}

# ============================================================
# LEARNING SYSTEM (Phase 6)
# ============================================================
LEARNING_SETTINGS = {
    "learning_rate": 0.10,  # How fast to adjust weights
    "lookback_days": 30,    # Days to consider for performance
    "min_trades": 5,        # Min trades before adjusting
}

# ============================================================
# LEVERAGE SETTINGS
# ============================================================
LEVERAGE_SETTINGS = {
    # USER-CONFIGURABLE MAX LEVERAGE
    # Options: 1.0 (no leverage), 2.0 (overnight max), 4.0 (intraday PDT max)
    "max_leverage": 2.0,  # Default: 2x (Alpaca overnight max for RegT)
    
    # LEVERAGE MODE
    # "manual" = always use max_leverage
    # "dynamic" = calculate optimal leverage based on conditions
    "mode": "dynamic",
    
    # DYNAMIC LEVERAGE PARAMETERS
    "dynamic": {
        # VIX-based adjustments (percentage of max leverage)
        "vix_factors": {
            15: 1.00,   # VIX < 15: 100% of max leverage
            20: 0.80,   # VIX 15-20: 80% of max leverage
            25: 0.60,   # VIX 20-25: 60% of max leverage
            30: 0.40,   # VIX 25-30: 40% of max leverage
            100: 0.20,  # VIX > 30: 20% of max leverage
        },
        # Drawdown-based adjustments
        "drawdown_factors": {
            3: 1.00,    # DD < 3%: 100% leverage
            5: 0.80,    # DD 3-5%: 80% leverage
            10: 0.50,   # DD 5-10%: 50% leverage
            15: 0.25,   # DD 10-15%: 25% leverage
            100: 0.00,  # DD > 15%: no leverage
        },
        # Confidence adjustment range
        "min_confidence_factor": 0.7,
        "max_confidence_factor": 1.2,
    },
    
    # RISK CONTROLS
    "risk_controls": {
        # Daily loss limits
        "daily_loss_halt_pct": 3.0,     # Halt new leveraged trades at 3% daily loss
        "daily_loss_delever_pct": 5.0,  # Close leveraged positions at 5% daily loss
        
        # Margin buffer
        "min_margin_buffer_pct": 25.0,  # Always keep 25% margin buffer
        "margin_alert_pct": 70.0,       # Alert when 70% margin used
        "margin_auto_reduce_pct": 85.0, # Auto-reduce at 85% margin used
        
        # Position limits with leverage
        "max_leveraged_position_pct": 15.0,  # Max 15% per position when leveraged
        "max_leveraged_sector_pct": 30.0,    # Max 30% per sector when leveraged
    },
    
    # MARGIN COSTS
    "margin_interest_rate": 0.07,  # 7% annual margin interest rate
}

# ============================================================
# SHORT SELLING SETTINGS
# ============================================================
SHORT_SETTINGS = {
    # Enable/disable shorting
    "enabled": True,
    
    # Maximum exposure
    "max_short_exposure_pct": 40.0,  # Max 40% of portfolio in shorts
    "max_single_short_pct": 5.0,     # Max 5% per short position
    
    # Short signal sources
    "signal_sources": {
        "news_driven": True,       # Shorts from negative news
        "valuation_based": True,   # Shorts from high valuation
        "technical_breakdown": True,  # Shorts from technical breakdowns
    },
    
    # Valuation thresholds for short signals
    "valuation_thresholds": {
        "pe_vs_sector_ratio": 2.0,      # Short if P/E > 2x sector average
        "price_above_200ma_pct": 50.0,  # Short if price > 50% above 200 DMA
    },
    
    # Technical breakdown thresholds
    "technical_thresholds": {
        "below_200ma_days": 5,          # Must be below 200 DMA for 5 days
        "rsi_overbought": 70,           # RSI above 70 = overbought
        "death_cross_lookback": 20,     # Days to detect death cross
    },
    
    # Borrow cost limits
    "max_borrow_rate_pct": 10.0,  # Skip if borrow rate > 10% annually
}
