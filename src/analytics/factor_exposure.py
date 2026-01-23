"""
Sector and Factor Exposure Management

Tracks and limits portfolio exposure to:
1. Sectors (Tech, Healthcare, Financials, etc.)
2. Factors (Momentum, Value, Size, Quality, Volatility)
3. Geographic exposure (US, International, Emerging)
4. Macro sensitivities (Rate sensitivity, Dollar sensitivity)

Prevents concentration and ensures diversification.
"""

import logging
import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class Sector(Enum):
    """GICS Sectors."""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    ENERGY = "energy"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATION = "communication"
    UNKNOWN = "unknown"


class Factor(Enum):
    """Investment factors."""
    MOMENTUM = "momentum"
    VALUE = "value"
    SIZE = "size"  # Small vs Large cap
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    GROWTH = "growth"
    DIVIDEND = "dividend"


@dataclass
class ExposureLimits:
    """Exposure limits configuration."""
    # Sector limits
    max_single_sector_pct: float = 30.0  # Max 30% in one sector
    min_sectors: int = 3  # Must be in at least 3 sectors
    
    # Factor limits
    max_factor_tilt: float = 0.5  # Max factor z-score tilt
    
    # Concentration limits
    max_single_position_pct: float = 10.0  # Max 10% in one stock
    max_top_5_pct: float = 40.0  # Max 40% in top 5 positions
    
    # Geographic limits
    max_international_pct: float = 30.0
    max_emerging_pct: float = 15.0
    
    # Correlation limits
    max_avg_correlation: float = 0.6  # Average pairwise correlation


@dataclass
class SymbolExposure:
    """Exposure characteristics for a symbol."""
    symbol: str
    
    # Sector
    sector: Sector = Sector.UNKNOWN
    
    # Factor loadings (-1 to 1, where 0 = neutral)
    momentum_loading: float = 0.0
    value_loading: float = 0.0
    size_loading: float = 0.0  # Positive = large cap
    quality_loading: float = 0.0
    volatility_loading: float = 0.0  # Positive = high vol
    growth_loading: float = 0.0
    
    # Geographic
    us_revenue_pct: float = 100.0
    international_revenue_pct: float = 0.0
    emerging_revenue_pct: float = 0.0
    
    # Sensitivities
    rate_sensitivity: float = 0.0  # Positive = benefits from higher rates
    dollar_sensitivity: float = 0.0  # Positive = benefits from strong dollar
    commodity_sensitivity: float = 0.0


@dataclass
class PortfolioExposure:
    """Aggregate portfolio exposure."""
    # Sector breakdown
    sector_weights: Dict[str, float] = field(default_factory=dict)
    sector_count: int = 0
    
    # Factor tilts
    factor_tilts: Dict[str, float] = field(default_factory=dict)
    
    # Concentration
    top_position_weight: float = 0.0
    top_5_weight: float = 0.0
    herfindahl_index: float = 0.0  # Concentration measure
    
    # Geographic
    us_weight: float = 100.0
    international_weight: float = 0.0
    emerging_weight: float = 0.0
    
    # Violations
    violations: List[str] = field(default_factory=list)
    
    def is_compliant(self) -> bool:
        return len(self.violations) == 0
    
    def to_dict(self) -> Dict:
        return {
            "sector_weights": self.sector_weights,
            "sector_count": self.sector_count,
            "factor_tilts": self.factor_tilts,
            "top_position_weight": self.top_position_weight,
            "top_5_weight": self.top_5_weight,
            "herfindahl_index": self.herfindahl_index,
            "us_weight": self.us_weight,
            "international_weight": self.international_weight,
            "violations": self.violations,
            "is_compliant": self.is_compliant(),
        }


class FactorExposureManager:
    """
    Manages portfolio factor and sector exposure.
    
    Features:
    - Sector classification and limits
    - Factor loading calculation
    - Exposure rebalancing suggestions
    - Concentration monitoring
    """
    
    # Sector mapping for common stocks
    SECTOR_MAP = {
        # Technology
        "AAPL": Sector.TECHNOLOGY, "MSFT": Sector.TECHNOLOGY, "GOOGL": Sector.TECHNOLOGY,
        "META": Sector.TECHNOLOGY, "NVDA": Sector.TECHNOLOGY, "AMD": Sector.TECHNOLOGY,
        "INTC": Sector.TECHNOLOGY, "CSCO": Sector.TECHNOLOGY, "ORCL": Sector.TECHNOLOGY,
        "CRM": Sector.TECHNOLOGY, "ADBE": Sector.TECHNOLOGY, "NOW": Sector.TECHNOLOGY,
        "IBM": Sector.TECHNOLOGY, "AVGO": Sector.TECHNOLOGY, "TXN": Sector.TECHNOLOGY,
        "QCOM": Sector.TECHNOLOGY, "MU": Sector.TECHNOLOGY, "AMAT": Sector.TECHNOLOGY,
        
        # Communication
        "AMZN": Sector.CONSUMER_DISCRETIONARY, "NFLX": Sector.COMMUNICATION,
        "DIS": Sector.COMMUNICATION, "VZ": Sector.COMMUNICATION, "T": Sector.COMMUNICATION,
        "TMUS": Sector.COMMUNICATION, "CMCSA": Sector.COMMUNICATION,
        
        # Healthcare
        "JNJ": Sector.HEALTHCARE, "UNH": Sector.HEALTHCARE, "PFE": Sector.HEALTHCARE,
        "ABBV": Sector.HEALTHCARE, "MRK": Sector.HEALTHCARE, "LLY": Sector.HEALTHCARE,
        "TMO": Sector.HEALTHCARE, "ABT": Sector.HEALTHCARE, "DHR": Sector.HEALTHCARE,
        "BMY": Sector.HEALTHCARE, "AMGN": Sector.HEALTHCARE, "GILD": Sector.HEALTHCARE,
        
        # Financials
        "JPM": Sector.FINANCIALS, "BAC": Sector.FINANCIALS, "WFC": Sector.FINANCIALS,
        "GS": Sector.FINANCIALS, "MS": Sector.FINANCIALS, "C": Sector.FINANCIALS,
        "BLK": Sector.FINANCIALS, "SCHW": Sector.FINANCIALS, "AXP": Sector.FINANCIALS,
        "V": Sector.FINANCIALS, "MA": Sector.FINANCIALS, "USB": Sector.FINANCIALS,
        
        # Consumer Discretionary
        "TSLA": Sector.CONSUMER_DISCRETIONARY, "HD": Sector.CONSUMER_DISCRETIONARY,
        "MCD": Sector.CONSUMER_DISCRETIONARY, "NKE": Sector.CONSUMER_DISCRETIONARY,
        "SBUX": Sector.CONSUMER_DISCRETIONARY, "TGT": Sector.CONSUMER_DISCRETIONARY,
        "LOW": Sector.CONSUMER_DISCRETIONARY, "TJX": Sector.CONSUMER_DISCRETIONARY,
        
        # Consumer Staples
        "PG": Sector.CONSUMER_STAPLES, "KO": Sector.CONSUMER_STAPLES,
        "PEP": Sector.CONSUMER_STAPLES, "WMT": Sector.CONSUMER_STAPLES,
        "COST": Sector.CONSUMER_STAPLES, "PM": Sector.CONSUMER_STAPLES,
        "MO": Sector.CONSUMER_STAPLES, "CL": Sector.CONSUMER_STAPLES,
        
        # Industrials
        "CAT": Sector.INDUSTRIALS, "DE": Sector.INDUSTRIALS, "BA": Sector.INDUSTRIALS,
        "HON": Sector.INDUSTRIALS, "UNP": Sector.INDUSTRIALS, "UPS": Sector.INDUSTRIALS,
        "GE": Sector.INDUSTRIALS, "MMM": Sector.INDUSTRIALS, "LMT": Sector.INDUSTRIALS,
        "RTX": Sector.INDUSTRIALS, "NOC": Sector.INDUSTRIALS,
        
        # Energy
        "XOM": Sector.ENERGY, "CVX": Sector.ENERGY, "COP": Sector.ENERGY,
        "SLB": Sector.ENERGY, "EOG": Sector.ENERGY, "OXY": Sector.ENERGY,
        "PSX": Sector.ENERGY, "VLO": Sector.ENERGY, "MPC": Sector.ENERGY,
        
        # Materials
        "LIN": Sector.MATERIALS, "APD": Sector.MATERIALS, "ECL": Sector.MATERIALS,
        "DD": Sector.MATERIALS, "NEM": Sector.MATERIALS, "FCX": Sector.MATERIALS,
        
        # Utilities
        "NEE": Sector.UTILITIES, "DUK": Sector.UTILITIES, "SO": Sector.UTILITIES,
        "D": Sector.UTILITIES, "AEP": Sector.UTILITIES, "XEL": Sector.UTILITIES,
        
        # Real Estate
        "AMT": Sector.REAL_ESTATE, "PLD": Sector.REAL_ESTATE, "CCI": Sector.REAL_ESTATE,
        "EQIX": Sector.REAL_ESTATE, "SPG": Sector.REAL_ESTATE, "O": Sector.REAL_ESTATE,
        
        # ETFs
        "SPY": Sector.UNKNOWN, "QQQ": Sector.TECHNOLOGY, "IWM": Sector.UNKNOWN,
        "XLK": Sector.TECHNOLOGY, "XLF": Sector.FINANCIALS, "XLE": Sector.ENERGY,
        "XLV": Sector.HEALTHCARE, "XLI": Sector.INDUSTRIALS, "XLB": Sector.MATERIALS,
        "XLY": Sector.CONSUMER_DISCRETIONARY, "XLP": Sector.CONSUMER_STAPLES,
        "XLU": Sector.UTILITIES, "XLRE": Sector.REAL_ESTATE, "XLC": Sector.COMMUNICATION,
    }
    
    # Factor characteristics by sector (simplified)
    SECTOR_FACTOR_DEFAULTS = {
        Sector.TECHNOLOGY: {"momentum": 0.3, "growth": 0.5, "volatility": 0.3, "value": -0.3},
        Sector.HEALTHCARE: {"quality": 0.3, "growth": 0.2, "volatility": 0.2},
        Sector.FINANCIALS: {"value": 0.3, "rate_sensitivity": 0.5, "volatility": 0.2},
        Sector.CONSUMER_DISCRETIONARY: {"momentum": 0.2, "growth": 0.3, "volatility": 0.2},
        Sector.CONSUMER_STAPLES: {"quality": 0.4, "dividend": 0.4, "volatility": -0.3},
        Sector.INDUSTRIALS: {"value": 0.2, "quality": 0.2, "volatility": 0.1},
        Sector.MATERIALS: {"commodity_sensitivity": 0.6, "value": 0.2, "volatility": 0.3},
        Sector.ENERGY: {"commodity_sensitivity": 0.8, "value": 0.3, "volatility": 0.4},
        Sector.UTILITIES: {"dividend": 0.5, "rate_sensitivity": -0.4, "volatility": -0.4},
        Sector.REAL_ESTATE: {"rate_sensitivity": -0.5, "dividend": 0.4, "volatility": 0.1},
        Sector.COMMUNICATION: {"growth": 0.2, "momentum": 0.1, "volatility": 0.2},
    }
    
    def __init__(
        self,
        limits: Optional[ExposureLimits] = None,
        storage_path: str = "outputs/factor_exposure.json",
    ):
        self.limits = limits or ExposureLimits()
        self.storage_path = Path(storage_path)
        
        # Symbol exposure cache
        self.symbol_exposures: Dict[str, SymbolExposure] = {}
        
        # Factor history for calculating loadings
        self.factor_returns: Dict[str, List[float]] = {}
        
        self._load()
    
    def _load(self):
        """Load saved exposure data."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                # Could load learned factor loadings here
                logger.info("Loaded factor exposure data")
            except Exception as e:
                logger.warning(f"Could not load factor exposure: {e}")
    
    def _save(self):
        """Save exposure data."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'symbol_count': len(self.symbol_exposures),
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save factor exposure: {e}")
    
    def get_symbol_exposure(self, symbol: str) -> SymbolExposure:
        """Get or create exposure profile for a symbol."""
        if symbol in self.symbol_exposures:
            return self.symbol_exposures[symbol]
        
        # Create new exposure with defaults
        sector = self.SECTOR_MAP.get(symbol, Sector.UNKNOWN)
        
        # Get factor defaults for sector
        sector_factors = self.SECTOR_FACTOR_DEFAULTS.get(sector, {})
        
        exposure = SymbolExposure(
            symbol=symbol,
            sector=sector,
            momentum_loading=sector_factors.get("momentum", 0.0),
            value_loading=sector_factors.get("value", 0.0),
            size_loading=0.3 if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"] else 0.0,
            quality_loading=sector_factors.get("quality", 0.0),
            volatility_loading=sector_factors.get("volatility", 0.0),
            growth_loading=sector_factors.get("growth", 0.0),
            rate_sensitivity=sector_factors.get("rate_sensitivity", 0.0),
            commodity_sensitivity=sector_factors.get("commodity_sensitivity", 0.0),
        )
        
        self.symbol_exposures[symbol] = exposure
        return exposure
    
    def calculate_portfolio_exposure(
        self,
        weights: Dict[str, float],
    ) -> PortfolioExposure:
        """
        Calculate aggregate portfolio exposure.
        
        Args:
            weights: Dict of symbol -> weight (e.g., 0.10 for 10%)
        """
        if not weights:
            return PortfolioExposure()
        
        # Normalize weights to absolute values for exposure calculation
        abs_weights = {s: abs(w) for s, w in weights.items() if abs(w) > 0.001}
        total_abs = sum(abs_weights.values())
        if total_abs > 0:
            norm_weights = {s: w / total_abs for s, w in abs_weights.items()}
        else:
            norm_weights = {}
        
        # Sector breakdown
        sector_weights = {}
        for symbol, weight in norm_weights.items():
            exposure = self.get_symbol_exposure(symbol)
            sector = exposure.sector.value
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        
        # Factor tilts (weighted average of factor loadings)
        factor_tilts = {
            "momentum": 0.0,
            "value": 0.0,
            "size": 0.0,
            "quality": 0.0,
            "volatility": 0.0,
            "growth": 0.0,
        }
        
        for symbol, weight in norm_weights.items():
            exposure = self.get_symbol_exposure(symbol)
            factor_tilts["momentum"] += exposure.momentum_loading * weight
            factor_tilts["value"] += exposure.value_loading * weight
            factor_tilts["size"] += exposure.size_loading * weight
            factor_tilts["quality"] += exposure.quality_loading * weight
            factor_tilts["volatility"] += exposure.volatility_loading * weight
            factor_tilts["growth"] += exposure.growth_loading * weight
        
        # Concentration metrics
        sorted_weights = sorted(norm_weights.values(), reverse=True)
        top_position = sorted_weights[0] if sorted_weights else 0.0
        top_5 = sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)
        
        # Herfindahl index (sum of squared weights)
        hhi = sum(w**2 for w in norm_weights.values())
        
        # Geographic exposure (simplified)
        us_weight = 0.0
        intl_weight = 0.0
        emerging_weight = 0.0
        
        for symbol, weight in norm_weights.items():
            exposure = self.get_symbol_exposure(symbol)
            us_weight += (exposure.us_revenue_pct / 100) * weight
            intl_weight += (exposure.international_revenue_pct / 100) * weight
            emerging_weight += (exposure.emerging_revenue_pct / 100) * weight
        
        # Check for violations
        violations = []
        
        # Sector violations
        for sector, pct in sector_weights.items():
            if pct > self.limits.max_single_sector_pct / 100:
                violations.append(
                    f"Sector {sector} at {pct*100:.1f}% exceeds {self.limits.max_single_sector_pct}% limit"
                )
        
        if len([s for s in sector_weights.values() if s > 0.01]) < self.limits.min_sectors:
            violations.append(f"Less than {self.limits.min_sectors} sectors represented")
        
        # Concentration violations
        if top_position > self.limits.max_single_position_pct / 100:
            violations.append(
                f"Top position at {top_position*100:.1f}% exceeds {self.limits.max_single_position_pct}% limit"
            )
        
        if top_5 > self.limits.max_top_5_pct / 100:
            violations.append(
                f"Top 5 positions at {top_5*100:.1f}% exceeds {self.limits.max_top_5_pct}% limit"
            )
        
        # Factor tilt violations
        for factor, tilt in factor_tilts.items():
            if abs(tilt) > self.limits.max_factor_tilt:
                violations.append(
                    f"Factor {factor} tilt of {tilt:.2f} exceeds {self.limits.max_factor_tilt} limit"
                )
        
        return PortfolioExposure(
            sector_weights=sector_weights,
            sector_count=len([s for s in sector_weights.values() if s > 0.01]),
            factor_tilts=factor_tilts,
            top_position_weight=top_position,
            top_5_weight=top_5,
            herfindahl_index=hhi,
            us_weight=us_weight,
            international_weight=intl_weight,
            emerging_weight=emerging_weight,
            violations=violations,
        )
    
    def adjust_weights_for_compliance(
        self,
        weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Adjust weights to comply with exposure limits.
        Returns (adjusted_weights, adjustments_made).
        """
        adjusted = weights.copy()
        adjustments = []
        
        # Check current exposure
        exposure = self.calculate_portfolio_exposure(adjusted)
        
        if exposure.is_compliant():
            return adjusted, []
        
        # Iteratively adjust until compliant (max 5 iterations)
        for _ in range(5):
            exposure = self.calculate_portfolio_exposure(adjusted)
            
            if exposure.is_compliant():
                break
            
            # Fix sector concentration
            for sector, pct in exposure.sector_weights.items():
                if pct > self.limits.max_single_sector_pct / 100:
                    # Find symbols in this sector and reduce
                    reduction_factor = (self.limits.max_single_sector_pct / 100) / pct
                    
                    for symbol in adjusted:
                        sym_exposure = self.get_symbol_exposure(symbol)
                        if sym_exposure.sector.value == sector:
                            old_weight = adjusted[symbol]
                            adjusted[symbol] = old_weight * reduction_factor
                            adjustments.append(
                                f"Reduced {symbol} from {old_weight*100:.1f}% to {adjusted[symbol]*100:.1f}% (sector limit)"
                            )
            
            # Fix single position concentration
            if exposure.top_position_weight > self.limits.max_single_position_pct / 100:
                top_symbol = max(adjusted.keys(), key=lambda s: abs(adjusted.get(s, 0)))
                old_weight = adjusted[top_symbol]
                adjusted[top_symbol] = self.limits.max_single_position_pct / 100 * (1 if old_weight > 0 else -1)
                adjustments.append(
                    f"Capped {top_symbol} from {old_weight*100:.1f}% to {adjusted[top_symbol]*100:.1f}% (position limit)"
                )
        
        return adjusted, adjustments
    
    def get_diversification_score(
        self,
        weights: Dict[str, float],
    ) -> float:
        """
        Calculate a diversification score (0-1, higher is better).
        """
        exposure = self.calculate_portfolio_exposure(weights)
        
        score = 0.0
        
        # Sector diversification (more sectors = better)
        sector_score = min(1.0, exposure.sector_count / 5)  # Max at 5 sectors
        score += sector_score * 0.25
        
        # Concentration (lower HHI = better)
        # HHI of 0.1 is well diversified, 0.5 is concentrated
        hhi_score = max(0, 1 - (exposure.herfindahl_index - 0.05) / 0.45)
        score += hhi_score * 0.25
        
        # Factor balance (lower tilts = better)
        avg_tilt = np.mean([abs(t) for t in exposure.factor_tilts.values()])
        tilt_score = max(0, 1 - avg_tilt / 0.5)
        score += tilt_score * 0.25
        
        # Compliance bonus
        if exposure.is_compliant():
            score += 0.25
        
        return min(1.0, score)
    
    def get_rebalancing_suggestions(
        self,
        weights: Dict[str, float],
    ) -> List[Dict]:
        """
        Get suggestions for improving portfolio diversification.
        """
        exposure = self.calculate_portfolio_exposure(weights)
        suggestions = []
        
        # Sector suggestions
        over_sectors = [s for s, w in exposure.sector_weights.items() 
                       if w > self.limits.max_single_sector_pct / 100]
        under_sectors = [s.value for s in Sector if s.value not in exposure.sector_weights 
                        and s != Sector.UNKNOWN]
        
        if over_sectors:
            suggestions.append({
                "type": "sector_concentration",
                "message": f"Reduce exposure to: {', '.join(over_sectors)}",
                "priority": "high",
            })
        
        if len(under_sectors) > 3:
            suggestions.append({
                "type": "sector_diversification",
                "message": f"Consider adding exposure to: {', '.join(under_sectors[:3])}",
                "priority": "medium",
            })
        
        # Factor suggestions
        for factor, tilt in exposure.factor_tilts.items():
            if abs(tilt) > 0.3:
                direction = "positive" if tilt > 0 else "negative"
                suggestions.append({
                    "type": "factor_tilt",
                    "message": f"High {direction} {factor} tilt ({tilt:.2f}). Consider balancing.",
                    "priority": "low" if abs(tilt) < 0.4 else "medium",
                })
        
        # Concentration suggestions
        if exposure.top_5_weight > 0.5:
            suggestions.append({
                "type": "concentration",
                "message": f"Top 5 positions at {exposure.top_5_weight*100:.0f}%. Spread risk.",
                "priority": "high" if exposure.top_5_weight > 0.6 else "medium",
            })
        
        return suggestions
    
    def get_summary(self) -> Dict:
        """Get summary of factor exposure management."""
        return {
            "symbols_tracked": len(self.symbol_exposures),
            "limits": {
                "max_single_sector": self.limits.max_single_sector_pct,
                "max_single_position": self.limits.max_single_position_pct,
                "max_factor_tilt": self.limits.max_factor_tilt,
            },
        }


# Singleton instance
_factor_manager: Optional[FactorExposureManager] = None


def get_factor_manager() -> FactorExposureManager:
    """Get singleton instance of factor exposure manager."""
    global _factor_manager
    if _factor_manager is None:
        _factor_manager = FactorExposureManager()
    return _factor_manager
