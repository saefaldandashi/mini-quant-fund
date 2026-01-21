"""
Source Credibility System for news weighting.

Tier-based weighting for news sources.
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


class SourceTier(Enum):
    """News source credibility tiers."""
    TIER_1 = "tier_1"  # Highest credibility: Reuters, Bloomberg, WSJ, FT, central banks
    TIER_2 = "tier_2"  # High credibility: CNBC, Economist, major financial outlets
    TIER_3 = "tier_3"  # Medium credibility: General news, regional outlets
    TIER_4 = "tier_4"  # Low credibility: Blogs, unknown sources
    UNKNOWN = "unknown"  # Default for unrecognized sources


@dataclass
class SourceInfo:
    """Information about a news source."""
    name: str
    tier: SourceTier
    weight: float  # 0-1 credibility weight
    category: str  # e.g., "wire", "financial", "government", "blog"


class SourceCredibility:
    """
    Assign credibility weights to news sources.
    
    Configurable source mappings with default weights.
    """
    
    # Default source configurations
    TIER_1_SOURCES: Dict[str, str] = {
        # Wire services
        'reuters': 'wire',
        'associated press': 'wire',
        'ap news': 'wire',
        
        # Financial news
        'bloomberg': 'financial',
        'wall street journal': 'financial',
        'wsj': 'financial',
        'financial times': 'financial',
        'ft': 'financial',
        'barrons': 'financial',
        
        # Central banks & Government
        'federal reserve': 'government',
        'fed': 'government',
        'treasury': 'government',
        'bls': 'government',
        'bureau of labor': 'government',
        'bea': 'government',
        'bureau of economic': 'government',
        'ecb': 'government',
        'bank of england': 'government',
        'imf': 'government',
        'world bank': 'government',
    }
    
    TIER_2_SOURCES: Dict[str, str] = {
        'cnbc': 'financial',
        'economist': 'financial',
        'forbes': 'financial',
        'marketwatch': 'financial',
        'seeking alpha': 'financial',
        'benzinga': 'financial',
        'yahoo finance': 'financial',
        'business insider': 'financial',
        'cnn business': 'general',
        'nytimes': 'general',
        'new york times': 'general',
        'washington post': 'general',
        'bbc': 'general',
        'guardian': 'general',
        'axios': 'general',
        'politico': 'general',
    }
    
    TIER_3_SOURCES: Dict[str, str] = {
        'cnn': 'general',
        'fox business': 'general',
        'msnbc': 'general',
        'investorplace': 'financial',
        'motley fool': 'financial',
        'zacks': 'financial',
        'thestreet': 'financial',
        'investor': 'financial',
    }
    
    # Weight multipliers for each tier
    TIER_WEIGHTS: Dict[SourceTier, float] = {
        SourceTier.TIER_1: 1.0,
        SourceTier.TIER_2: 0.8,
        SourceTier.TIER_3: 0.5,
        SourceTier.TIER_4: 0.3,
        SourceTier.UNKNOWN: 0.2,
    }
    
    def __init__(
        self,
        custom_sources: Optional[Dict[str, Dict]] = None,
        default_tier: SourceTier = SourceTier.UNKNOWN,
    ):
        """
        Initialize source credibility system.
        
        Args:
            custom_sources: Optional custom source configurations
                Format: {"source_name": {"tier": "tier_1", "category": "financial"}}
            default_tier: Default tier for unknown sources
        """
        self.default_tier = default_tier
        
        # Build source lookup
        self._sources: Dict[str, SourceInfo] = {}
        
        # Add default sources
        for name, category in self.TIER_1_SOURCES.items():
            self._sources[name.lower()] = SourceInfo(
                name=name,
                tier=SourceTier.TIER_1,
                weight=self.TIER_WEIGHTS[SourceTier.TIER_1],
                category=category,
            )
        
        for name, category in self.TIER_2_SOURCES.items():
            self._sources[name.lower()] = SourceInfo(
                name=name,
                tier=SourceTier.TIER_2,
                weight=self.TIER_WEIGHTS[SourceTier.TIER_2],
                category=category,
            )
        
        for name, category in self.TIER_3_SOURCES.items():
            self._sources[name.lower()] = SourceInfo(
                name=name,
                tier=SourceTier.TIER_3,
                weight=self.TIER_WEIGHTS[SourceTier.TIER_3],
                category=category,
            )
        
        # Add custom sources
        if custom_sources:
            for name, config in custom_sources.items():
                tier = SourceTier(config.get('tier', 'tier_3'))
                self._sources[name.lower()] = SourceInfo(
                    name=name,
                    tier=tier,
                    weight=self.TIER_WEIGHTS[tier],
                    category=config.get('category', 'custom'),
                )
    
    def get_source_info(self, source_name: str) -> SourceInfo:
        """
        Get source information and credibility.
        
        Args:
            source_name: Name of the news source
            
        Returns:
            SourceInfo with tier and weight
        """
        if not source_name:
            return SourceInfo(
                name="unknown",
                tier=self.default_tier,
                weight=self.TIER_WEIGHTS[self.default_tier],
                category="unknown",
            )
        
        source_lower = source_name.lower().strip()
        
        # Direct match
        if source_lower in self._sources:
            return self._sources[source_lower]
        
        # Fuzzy match - check if source contains known name
        for known_name, info in self._sources.items():
            if known_name in source_lower or source_lower in known_name:
                return info
        
        # Check for domain patterns
        domain_match = self._extract_domain(source_name)
        if domain_match and domain_match in self._sources:
            return self._sources[domain_match]
        
        # Return unknown
        return SourceInfo(
            name=source_name,
            tier=self.default_tier,
            weight=self.TIER_WEIGHTS[self.default_tier],
            category="unknown",
        )
    
    def get_weight(self, source_name: str) -> float:
        """Get credibility weight for a source (0-1)."""
        return self.get_source_info(source_name).weight
    
    def get_tier(self, source_name: str) -> SourceTier:
        """Get tier for a source."""
        return self.get_source_info(source_name).tier
    
    def _extract_domain(self, source: str) -> Optional[str]:
        """Extract domain name from URL or source string."""
        # Remove common prefixes
        patterns = [
            r'https?://(?:www\.)?([^/]+)',
            r'www\.([^/]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, source.lower())
            if match:
                domain = match.group(1).split('.')[0]
                return domain
        
        return None
    
    def add_source(
        self,
        name: str,
        tier: SourceTier,
        category: str = "custom",
    ):
        """Add or update a source configuration."""
        self._sources[name.lower()] = SourceInfo(
            name=name,
            tier=tier,
            weight=self.TIER_WEIGHTS[tier],
            category=category,
        )
    
    def get_all_sources(self) -> Dict[str, SourceInfo]:
        """Get all configured sources."""
        return self._sources.copy()
