"""
Cross-Correlation Analysis Engine

Analyzes correlations across multiple asset classes to generate alpha signals:
1. Commodities → Stock Sectors (Oil → Energy, Copper → Industrials)
2. Currencies → Multinationals (DXY strength → Domestic vs International)
3. International Markets → US Stocks (Europe/Asia lead → US follow)
4. Bond Markets → Equities (Yield curve → Financials, Credit spreads → Risk)
5. VIX Term Structure → Market Direction (Contango/Backwardation)

These cross-correlations reveal predictive relationships that single-asset
analysis misses.
"""

import logging
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class categories."""
    EQUITY = "equity"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    BOND = "bond"
    VOLATILITY = "volatility"
    INTERNATIONAL = "international"


class CorrelationRegime(Enum):
    """Correlation regime states."""
    STRONG_POSITIVE = "strong_positive"    # > 0.7
    MODERATE_POSITIVE = "moderate_positive"  # 0.3 to 0.7
    NEUTRAL = "neutral"                      # -0.3 to 0.3
    MODERATE_NEGATIVE = "moderate_negative"  # -0.7 to -0.3
    STRONG_NEGATIVE = "strong_negative"      # < -0.7


@dataclass
class CorrelationPair:
    """A tracked correlation between two assets."""
    asset_a: str
    asset_b: str
    asset_a_class: AssetClass
    asset_b_class: AssetClass
    
    # Current correlations at different lookbacks
    corr_5d: float = 0.0
    corr_20d: float = 0.0
    corr_60d: float = 0.0
    
    # Historical average (baseline)
    historical_avg: float = 0.0
    historical_std: float = 0.1
    
    # Regime
    current_regime: CorrelationRegime = CorrelationRegime.NEUTRAL
    
    # Lead-lag relationship
    lead_lag_days: int = 0  # Positive = asset_a leads, Negative = asset_b leads
    lead_lag_strength: float = 0.0
    
    # Anomaly detection
    z_score: float = 0.0  # How unusual is current correlation?
    is_anomaly: bool = False
    
    last_updated: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "asset_a_class": self.asset_a_class.value,
            "asset_b_class": self.asset_b_class.value,
            "corr_5d": self.corr_5d,
            "corr_20d": self.corr_20d,
            "corr_60d": self.corr_60d,
            "historical_avg": self.historical_avg,
            "current_regime": self.current_regime.value,
            "lead_lag_days": self.lead_lag_days,
            "z_score": self.z_score,
            "is_anomaly": self.is_anomaly,
            "last_updated": self.last_updated,
        }


@dataclass
class CrossAssetSignal:
    """Signal generated from cross-asset analysis."""
    signal_type: str  # "sector_rotation", "risk_sentiment", "currency_impact", etc.
    direction: str    # "bullish", "bearish", "neutral"
    strength: float   # 0-1
    
    affected_symbols: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    
    driver_asset: str = ""
    driver_move: float = 0.0
    
    rationale: str = ""
    confidence: float = 0.5
    
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "signal_type": self.signal_type,
            "direction": self.direction,
            "strength": self.strength,
            "affected_symbols": self.affected_symbols,
            "affected_sectors": self.affected_sectors,
            "driver_asset": self.driver_asset,
            "driver_move": self.driver_move,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


class CrossCorrelationEngine:
    """
    Comprehensive cross-asset correlation analysis.
    
    Tracks and learns correlations between:
    - Commodities and equity sectors
    - Currencies and multinational stocks  
    - International markets and US stocks
    - Bonds/credit and equity risk
    - Volatility structure and market direction
    
    Uses these to generate predictive signals.
    """
    
    # Asset mappings
    COMMODITY_SECTOR_MAP = {
        # Commodity → Affected Sectors
        "CL=F": ["XLE", "OIH", "XOP"],      # Crude Oil → Energy
        "GC=F": ["GDX", "GLD", "GOLD"],     # Gold → Gold Miners
        "HG=F": ["XLB", "FCX", "SCCO"],     # Copper → Materials
        "SI=F": ["SLV", "PAAS", "AG"],      # Silver → Silver Miners
        "NG=F": ["XLE", "LNG", "CHK"],      # Natural Gas → Energy/Utilities
        "ZC=F": ["MON", "DE", "ADM"],       # Corn → Agriculture
        "ZS=F": ["BG", "ADM", "INGR"],      # Soybeans → Agriculture
    }
    
    COMMODITY_STOCK_MAP = {
        # Commodity → Individual Stocks
        "CL=F": ["XOM", "CVX", "COP", "SLB", "HAL", "OXY", "PSX", "VLO", "MPC"],
        "GC=F": ["NEM", "GOLD", "AEM", "KGC", "AU"],
        "HG=F": ["FCX", "SCCO", "TECK", "RIO", "BHP"],
    }
    
    CURRENCY_IMPACT_MAP = {
        # Currency strength → Stock impact
        "DXY": {
            "strong": {  # Strong dollar
                "bearish": ["multinational", "emerging_markets", "commodities"],
                "bullish": ["domestic_focus", "importers"],
            },
            "weak": {  # Weak dollar
                "bullish": ["multinational", "exporters", "commodities"],
                "bearish": ["importers"],
            }
        },
        "USDJPY": {
            "rising": {"bullish": ["japanese_adr"], "bearish": []},
            "falling": {"bearish": ["japanese_adr"], "bullish": []},
        },
        "EURUSD": {
            "rising": {"bullish": ["european_adr", "luxury"], "bearish": []},
            "falling": {"bearish": ["european_adr"], "bullish": []},
        },
    }
    
    INTERNATIONAL_MARKET_MAP = {
        # International Index → Correlated US Sectors/Stocks
        "EWG": "germany",   # Germany → Industrials, Autos
        "EWJ": "japan",     # Japan → Tech, Autos
        "FXI": "china",     # China → Materials, Tech Supply Chain
        "EWZ": "brazil",    # Brazil → Commodities
        "EWY": "korea",     # Korea → Semiconductors
        "EWT": "taiwan",    # Taiwan → Semiconductors
    }
    
    SECTOR_ETFS = [
        "XLK",  # Technology
        "XLF",  # Financials
        "XLE",  # Energy
        "XLV",  # Healthcare
        "XLI",  # Industrials
        "XLB",  # Materials
        "XLY",  # Consumer Discretionary
        "XLP",  # Consumer Staples
        "XLU",  # Utilities
        "XLRE", # Real Estate
        "XLC",  # Communication Services
    ]
    
    MULTINATIONAL_STOCKS = [
        "AAPL", "MSFT", "GOOGL", "META", "AMZN",  # Big tech
        "JNJ", "PG", "KO", "PEP", "MCD",           # Consumer multinationals
        "CAT", "DE", "BA", "GE", "HON",            # Industrial multinationals
    ]
    
    DOMESTIC_FOCUS_STOCKS = [
        "WMT", "HD", "LOW", "TGT", "COST",  # Retailers
        "JPM", "BAC", "WFC", "C", "USB",     # Regional banks
        "VZ", "T", "TMUS",                    # Telecoms
    ]
    
    def __init__(
        self,
        storage_path: str = "outputs/cross_correlations.json",
        lookback_days: int = 252,
    ):
        self.storage_path = Path(storage_path)
        self.lookback_days = lookback_days
        
        # Correlation tracking
        self.correlation_pairs: Dict[str, CorrelationPair] = {}
        self.price_history: Dict[str, pd.Series] = {}
        
        # Signal generation
        self.active_signals: List[CrossAssetSignal] = []
        self.signal_history: List[Dict] = []
        
        # Learning
        self.learned_correlations: Dict[str, Dict] = {}
        
        self._load()
    
    def _load(self):
        """Load saved correlation data."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.learned_correlations = data.get('learned_correlations', {})
                self.signal_history = data.get('signal_history', [])[-1000:]  # Keep last 1000
                logger.info(f"Loaded {len(self.learned_correlations)} learned correlations")
            except Exception as e:
                logger.warning(f"Could not load correlations: {e}")
    
    def _save(self):
        """Save correlation data."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'learned_correlations': self.learned_correlations,
                    'signal_history': self.signal_history[-1000:],
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save correlations: {e}")
    
    def update_prices(self, prices: Dict[str, pd.Series]):
        """Update price history for correlation calculation."""
        for symbol, series in prices.items():
            if len(series) > 0:
                self.price_history[symbol] = series
        logger.info(f"Updated prices for {len(prices)} symbols")
    
    def calculate_correlations(self) -> Dict[str, CorrelationPair]:
        """Calculate all tracked correlations."""
        pairs = {}
        
        # 1. Commodity → Sector correlations
        for commodity, sectors in self.COMMODITY_SECTOR_MAP.items():
            for sector in sectors:
                pair = self._calculate_pair_correlation(
                    commodity, sector,
                    AssetClass.COMMODITY, AssetClass.EQUITY
                )
                if pair:
                    key = f"{commodity}_{sector}"
                    pairs[key] = pair
        
        # 2. International → US market correlations
        for intl_etf, region in self.INTERNATIONAL_MARKET_MAP.items():
            # Correlate with SPY
            pair = self._calculate_pair_correlation(
                intl_etf, "SPY",
                AssetClass.INTERNATIONAL, AssetClass.EQUITY
            )
            if pair:
                pairs[f"{intl_etf}_SPY"] = pair
            
            # Correlate with sector ETFs
            for sector in self.SECTOR_ETFS:
                pair = self._calculate_pair_correlation(
                    intl_etf, sector,
                    AssetClass.INTERNATIONAL, AssetClass.EQUITY
                )
                if pair:
                    pairs[f"{intl_etf}_{sector}"] = pair
        
        # 3. VIX → Market correlations
        vix_spy = self._calculate_pair_correlation(
            "^VIX", "SPY",
            AssetClass.VOLATILITY, AssetClass.EQUITY
        )
        if vix_spy:
            pairs["VIX_SPY"] = vix_spy
        
        # 4. Bond → Equity correlations
        bond_pairs = [
            ("TLT", "SPY"),   # Long bonds vs market
            ("TLT", "XLF"),   # Long bonds vs financials
            ("HYG", "SPY"),   # High yield vs market (credit risk)
            ("LQD", "XLF"),   # Investment grade vs financials
        ]
        for bond, equity in bond_pairs:
            pair = self._calculate_pair_correlation(
                bond, equity,
                AssetClass.BOND, AssetClass.EQUITY
            )
            if pair:
                pairs[f"{bond}_{equity}"] = pair
        
        self.correlation_pairs = pairs
        logger.info(f"Calculated {len(pairs)} correlation pairs")
        return pairs
    
    def _calculate_pair_correlation(
        self,
        asset_a: str,
        asset_b: str,
        class_a: AssetClass,
        class_b: AssetClass,
    ) -> Optional[CorrelationPair]:
        """Calculate correlation between two assets."""
        if asset_a not in self.price_history or asset_b not in self.price_history:
            return None
        
        try:
            series_a = self.price_history[asset_a]
            series_b = self.price_history[asset_b]
            
            # Align series
            combined = pd.DataFrame({'a': series_a, 'b': series_b}).dropna()
            
            if len(combined) < 20:
                return None
            
            # Calculate returns
            returns_a = combined['a'].pct_change().dropna()
            returns_b = combined['b'].pct_change().dropna()
            
            if len(returns_a) < 20:
                return None
            
            # Rolling correlations at different lookbacks
            corr_5d = returns_a.tail(5).corr(returns_b.tail(5)) if len(returns_a) >= 5 else 0
            corr_20d = returns_a.tail(20).corr(returns_b.tail(20)) if len(returns_a) >= 20 else 0
            corr_60d = returns_a.tail(60).corr(returns_b.tail(60)) if len(returns_a) >= 60 else 0
            
            # Handle NaN
            corr_5d = 0 if np.isnan(corr_5d) else corr_5d
            corr_20d = 0 if np.isnan(corr_20d) else corr_20d
            corr_60d = 0 if np.isnan(corr_60d) else corr_60d
            
            # Historical baseline (full series)
            historical_corr = returns_a.corr(returns_b)
            historical_corr = 0 if np.isnan(historical_corr) else historical_corr
            
            # Calculate rolling correlation std for anomaly detection
            rolling_corr = returns_a.rolling(20).corr(returns_b)
            historical_std = rolling_corr.std() if len(rolling_corr.dropna()) > 10 else 0.1
            historical_std = 0.1 if np.isnan(historical_std) or historical_std < 0.01 else historical_std
            
            # Z-score (how unusual is current correlation?)
            z_score = (corr_20d - historical_corr) / historical_std if historical_std > 0 else 0
            
            # Detect regime
            regime = self._classify_correlation_regime(corr_20d)
            
            # Lead-lag analysis
            lead_lag, lag_strength = self._calculate_lead_lag(returns_a, returns_b)
            
            return CorrelationPair(
                asset_a=asset_a,
                asset_b=asset_b,
                asset_a_class=class_a,
                asset_b_class=class_b,
                corr_5d=round(corr_5d, 4),
                corr_20d=round(corr_20d, 4),
                corr_60d=round(corr_60d, 4),
                historical_avg=round(historical_corr, 4),
                historical_std=round(historical_std, 4),
                current_regime=regime,
                lead_lag_days=lead_lag,
                lead_lag_strength=round(lag_strength, 4),
                z_score=round(z_score, 2),
                is_anomaly=abs(z_score) > 2.0,
                last_updated=datetime.now().isoformat(),
            )
            
        except Exception as e:
            logger.debug(f"Error calculating correlation {asset_a}/{asset_b}: {e}")
            return None
    
    def _classify_correlation_regime(self, corr: float) -> CorrelationRegime:
        """Classify correlation into regime."""
        if corr > 0.7:
            return CorrelationRegime.STRONG_POSITIVE
        elif corr > 0.3:
            return CorrelationRegime.MODERATE_POSITIVE
        elif corr > -0.3:
            return CorrelationRegime.NEUTRAL
        elif corr > -0.7:
            return CorrelationRegime.MODERATE_NEGATIVE
        else:
            return CorrelationRegime.STRONG_NEGATIVE
    
    def _calculate_lead_lag(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        max_lag: int = 5,
    ) -> Tuple[int, float]:
        """
        Calculate lead-lag relationship.
        Returns (lag_days, correlation_at_lag).
        Positive lag = asset_a leads asset_b.
        """
        best_lag = 0
        best_corr = abs(returns_a.corr(returns_b))
        
        for lag in range(1, max_lag + 1):
            # Asset A leads (A[t] correlates with B[t+lag])
            shifted_b = returns_b.shift(-lag)
            corr_lead = abs(returns_a.corr(shifted_b.dropna()))
            if not np.isnan(corr_lead) and corr_lead > best_corr:
                best_corr = corr_lead
                best_lag = lag
            
            # Asset B leads (A[t+lag] correlates with B[t])
            shifted_a = returns_a.shift(-lag)
            corr_lag = abs(shifted_a.corr(returns_b.dropna()))
            if not np.isnan(corr_lag) and corr_lag > best_corr:
                best_corr = corr_lag
                best_lag = -lag
        
        return best_lag, best_corr
    
    def generate_signals(
        self,
        current_prices: Dict[str, float] = None,
    ) -> List[CrossAssetSignal]:
        """
        Generate trading signals from cross-asset analysis.
        
        Signal types:
        1. Commodity-driven sector rotation
        2. Currency impact on multinationals
        3. International market lead signals
        4. Credit spread risk signals
        5. VIX term structure signals
        """
        signals = []
        
        # 1. Commodity-driven signals
        signals.extend(self._generate_commodity_signals(current_prices))
        
        # 2. Currency impact signals
        signals.extend(self._generate_currency_signals(current_prices))
        
        # 3. International lead signals
        signals.extend(self._generate_international_signals(current_prices))
        
        # 4. Credit/Bond signals
        signals.extend(self._generate_credit_signals(current_prices))
        
        # 5. Correlation breakdown signals (anomalies)
        signals.extend(self._generate_anomaly_signals())
        
        self.active_signals = signals
        
        # Record for learning
        for sig in signals:
            self.signal_history.append({
                **sig.to_dict(),
                "timestamp": datetime.now().isoformat(),
            })
        
        self._save()
        
        logger.info(f"Generated {len(signals)} cross-asset signals")
        return signals
    
    def _generate_commodity_signals(
        self,
        current_prices: Dict[str, float] = None,
    ) -> List[CrossAssetSignal]:
        """Generate signals from commodity price movements."""
        signals = []
        
        # Calculate commodity returns
        commodity_returns = {}
        for commodity in self.COMMODITY_SECTOR_MAP.keys():
            if commodity in self.price_history:
                series = self.price_history[commodity]
                if len(series) >= 5:
                    ret_1d = (series.iloc[-1] / series.iloc[-2] - 1) if len(series) >= 2 else 0
                    ret_5d = (series.iloc[-1] / series.iloc[-5] - 1) if len(series) >= 5 else 0
                    commodity_returns[commodity] = {
                        'ret_1d': ret_1d,
                        'ret_5d': ret_5d,
                    }
        
        # Generate signals for significant commodity moves
        for commodity, returns in commodity_returns.items():
            ret_5d = returns['ret_5d']
            
            # Significant move threshold
            if abs(ret_5d) > 0.03:  # 3% move in 5 days
                direction = "bullish" if ret_5d > 0 else "bearish"
                strength = min(1.0, abs(ret_5d) / 0.10)  # Cap at 10% move = 1.0
                
                affected_sectors = self.COMMODITY_SECTOR_MAP.get(commodity, [])
                affected_stocks = self.COMMODITY_STOCK_MAP.get(commodity, [])
                
                # Get correlation strength for confidence
                confidence = 0.5
                for sector in affected_sectors:
                    pair_key = f"{commodity}_{sector}"
                    if pair_key in self.correlation_pairs:
                        pair = self.correlation_pairs[pair_key]
                        confidence = max(confidence, abs(pair.corr_20d))
                
                signal = CrossAssetSignal(
                    signal_type="commodity_sector_rotation",
                    direction=direction,
                    strength=strength,
                    affected_symbols=affected_stocks,
                    affected_sectors=affected_sectors,
                    driver_asset=commodity,
                    driver_move=ret_5d,
                    rationale=f"{commodity} moved {ret_5d*100:.1f}% over 5 days, "
                             f"historically correlated with {affected_sectors}",
                    confidence=confidence,
                    expires_at=datetime.now() + timedelta(days=5),
                )
                signals.append(signal)
        
        return signals
    
    def _generate_currency_signals(
        self,
        current_prices: Dict[str, float] = None,
    ) -> List[CrossAssetSignal]:
        """Generate signals from currency movements."""
        signals = []
        
        # Check DXY (Dollar Index)
        if "DXY" in self.price_history or "UUP" in self.price_history:
            dxy_symbol = "DXY" if "DXY" in self.price_history else "UUP"
            series = self.price_history[dxy_symbol]
            
            if len(series) >= 20:
                # 20-day momentum
                ret_20d = (series.iloc[-1] / series.iloc[-20] - 1)
                
                if abs(ret_20d) > 0.02:  # 2% move
                    if ret_20d > 0:
                        # Strong dollar
                        signal = CrossAssetSignal(
                            signal_type="currency_impact",
                            direction="bearish",
                            strength=min(1.0, abs(ret_20d) / 0.05),
                            affected_symbols=self.MULTINATIONAL_STOCKS,
                            affected_sectors=["XLK", "XLI"],  # Tech and Industrials
                            driver_asset=dxy_symbol,
                            driver_move=ret_20d,
                            rationale=f"Strong dollar ({ret_20d*100:.1f}% in 20d) hurts multinational earnings",
                            confidence=0.65,
                        )
                        signals.append(signal)
                        
                        # Bullish for domestic
                        signal2 = CrossAssetSignal(
                            signal_type="currency_impact",
                            direction="bullish",
                            strength=min(1.0, abs(ret_20d) / 0.05) * 0.5,  # Weaker signal
                            affected_symbols=self.DOMESTIC_FOCUS_STOCKS,
                            affected_sectors=["XLP", "XLU"],
                            driver_asset=dxy_symbol,
                            driver_move=ret_20d,
                            rationale=f"Strong dollar benefits domestic-focused companies",
                            confidence=0.55,
                        )
                        signals.append(signal2)
                    else:
                        # Weak dollar - bullish for multinationals
                        signal = CrossAssetSignal(
                            signal_type="currency_impact",
                            direction="bullish",
                            strength=min(1.0, abs(ret_20d) / 0.05),
                            affected_symbols=self.MULTINATIONAL_STOCKS,
                            affected_sectors=["XLK", "XLI"],
                            driver_asset=dxy_symbol,
                            driver_move=ret_20d,
                            rationale=f"Weak dollar ({ret_20d*100:.1f}% in 20d) boosts multinational earnings",
                            confidence=0.65,
                        )
                        signals.append(signal)
        
        return signals
    
    def _generate_international_signals(
        self,
        current_prices: Dict[str, float] = None,
    ) -> List[CrossAssetSignal]:
        """Generate signals from international market moves (lead-lag)."""
        signals = []
        
        for intl_etf, region in self.INTERNATIONAL_MARKET_MAP.items():
            pair_key = f"{intl_etf}_SPY"
            
            if pair_key in self.correlation_pairs:
                pair = self.correlation_pairs[pair_key]
                
                # If international market leads US
                if pair.lead_lag_days > 0 and pair.lead_lag_strength > 0.5:
                    # International moves may predict US
                    if intl_etf in self.price_history:
                        series = self.price_history[intl_etf]
                        if len(series) >= 3:
                            ret_3d = (series.iloc[-1] / series.iloc[-3] - 1)
                            
                            if abs(ret_3d) > 0.02:
                                direction = "bullish" if ret_3d > 0 else "bearish"
                                
                                signal = CrossAssetSignal(
                                    signal_type="international_lead",
                                    direction=direction,
                                    strength=min(1.0, abs(ret_3d) / 0.05),
                                    affected_symbols=[],
                                    affected_sectors=["SPY"],
                                    driver_asset=intl_etf,
                                    driver_move=ret_3d,
                                    rationale=f"{region.upper()} market ({intl_etf}) leads US by ~{pair.lead_lag_days} days. "
                                             f"Recent {ret_3d*100:.1f}% move may predict US direction.",
                                    confidence=pair.lead_lag_strength * 0.8,
                                    expires_at=datetime.now() + timedelta(days=pair.lead_lag_days + 1),
                                )
                                signals.append(signal)
        
        return signals
    
    def _generate_credit_signals(
        self,
        current_prices: Dict[str, float] = None,
    ) -> List[CrossAssetSignal]:
        """Generate signals from credit/bond market."""
        signals = []
        
        # HYG (High Yield) vs SPY - credit spread proxy
        hyg_spy_key = "HYG_SPY"
        if hyg_spy_key in self.correlation_pairs:
            pair = self.correlation_pairs[hyg_spy_key]
            
            # Correlation breakdown = risk signal
            if pair.is_anomaly and pair.z_score < -2.0:
                # Negative z-score = correlation lower than usual
                # HYG and SPY diverging = potential credit stress
                signal = CrossAssetSignal(
                    signal_type="credit_stress",
                    direction="bearish",
                    strength=min(1.0, abs(pair.z_score) / 3.0),
                    affected_symbols=[],
                    affected_sectors=["XLF", "SPY"],  # Financials most affected
                    driver_asset="HYG",
                    driver_move=0,
                    rationale=f"HYG/SPY correlation breakdown (z={pair.z_score:.1f}) signals credit stress",
                    confidence=0.70,
                )
                signals.append(signal)
        
        # TLT (Long Bonds) relative to XLF (Financials)
        tlt_xlf_key = "TLT_XLF"
        if tlt_xlf_key in self.correlation_pairs:
            pair = self.correlation_pairs[tlt_xlf_key]
            
            # Strong inverse correlation (rising bonds = falling financials)
            if pair.current_regime == CorrelationRegime.STRONG_NEGATIVE:
                if "TLT" in self.price_history:
                    tlt_ret = self.price_history["TLT"].pct_change().tail(5).sum()
                    
                    if abs(tlt_ret) > 0.02:
                        direction = "bearish" if tlt_ret > 0 else "bullish"
                        
                        signal = CrossAssetSignal(
                            signal_type="rate_impact",
                            direction=direction,
                            strength=min(1.0, abs(tlt_ret) / 0.05),
                            affected_symbols=["JPM", "BAC", "WFC", "C", "GS"],
                            affected_sectors=["XLF"],
                            driver_asset="TLT",
                            driver_move=tlt_ret,
                            rationale=f"Bond prices {'up' if tlt_ret > 0 else 'down'} {abs(tlt_ret)*100:.1f}% "
                                     f"→ {'lower' if tlt_ret > 0 else 'higher'} rates impact financials",
                            confidence=0.60,
                        )
                        signals.append(signal)
        
        return signals
    
    def _generate_anomaly_signals(self) -> List[CrossAssetSignal]:
        """Generate signals from correlation anomalies."""
        signals = []
        
        for key, pair in self.correlation_pairs.items():
            if pair.is_anomaly:
                # Correlation significantly different from historical
                direction = "neutral"
                if pair.z_score > 2.0:
                    direction = "caution"  # Unusually high correlation = crowded trade
                elif pair.z_score < -2.0:
                    direction = "opportunity"  # Divergence = potential mean reversion
                
                signal = CrossAssetSignal(
                    signal_type="correlation_anomaly",
                    direction=direction,
                    strength=min(1.0, abs(pair.z_score) / 3.0),
                    affected_symbols=[],
                    affected_sectors=[pair.asset_b] if pair.asset_b_class == AssetClass.EQUITY else [],
                    driver_asset=pair.asset_a,
                    driver_move=pair.z_score,
                    rationale=f"{pair.asset_a}/{pair.asset_b} correlation anomaly "
                             f"(z={pair.z_score:.1f}, current={pair.corr_20d:.2f} vs hist={pair.historical_avg:.2f})",
                    confidence=0.55,
                )
                signals.append(signal)
        
        return signals
    
    def get_symbol_signals(self, symbol: str) -> List[CrossAssetSignal]:
        """Get all active signals affecting a specific symbol."""
        return [
            sig for sig in self.active_signals
            if symbol in sig.affected_symbols or symbol in sig.affected_sectors
        ]
    
    def get_sector_exposure(self, symbol: str) -> Dict[str, float]:
        """
        Get cross-asset exposure factors for a symbol.
        Returns multipliers for position sizing.
        """
        exposures = {
            "commodity_sensitivity": 0.0,
            "currency_sensitivity": 0.0,
            "international_sensitivity": 0.0,
            "credit_sensitivity": 0.0,
        }
        
        signals = self.get_symbol_signals(symbol)
        
        for sig in signals:
            if sig.signal_type == "commodity_sector_rotation":
                exposures["commodity_sensitivity"] += sig.strength * (1 if sig.direction == "bullish" else -1)
            elif sig.signal_type == "currency_impact":
                exposures["currency_sensitivity"] += sig.strength * (1 if sig.direction == "bullish" else -1)
            elif sig.signal_type == "international_lead":
                exposures["international_sensitivity"] += sig.strength * (1 if sig.direction == "bullish" else -1)
            elif sig.signal_type in ["credit_stress", "rate_impact"]:
                exposures["credit_sensitivity"] += sig.strength * (1 if sig.direction == "bullish" else -1)
        
        return exposures
    
    def get_summary(self) -> Dict:
        """Get summary of current cross-correlation state."""
        return {
            "total_pairs": len(self.correlation_pairs),
            "anomalies": len([p for p in self.correlation_pairs.values() if p.is_anomaly]),
            "active_signals": len(self.active_signals),
            "signal_breakdown": {
                sig_type: len([s for s in self.active_signals if s.signal_type == sig_type])
                for sig_type in set(s.signal_type for s in self.active_signals)
            },
            "strongest_signals": [
                {
                    "type": s.signal_type,
                    "direction": s.direction,
                    "driver": s.driver_asset,
                    "confidence": s.confidence,
                }
                for s in sorted(self.active_signals, key=lambda x: -x.strength)[:5]
            ],
        }


# Singleton instance
_cross_correlation_engine: Optional[CrossCorrelationEngine] = None


def get_cross_correlation_engine() -> CrossCorrelationEngine:
    """Get singleton instance of cross-correlation engine."""
    global _cross_correlation_engine
    if _cross_correlation_engine is None:
        _cross_correlation_engine = CrossCorrelationEngine()
    return _cross_correlation_engine
