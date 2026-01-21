"""
Macro/Geo/Finance Taxonomy for news classification.

Assigns structured tags to news articles for downstream processing.
"""

from enum import Enum
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


class TaxonomyTag(Enum):
    """Market-moving news taxonomy tags."""
    
    # Macro Economic
    MACRO_INFLATION = "macro_inflation"  # CPI, PCE, inflation expectations, prices
    MACRO_GROWTH = "macro_growth"  # GDP, PMI, retail sales, industrial production
    MACRO_LABOR = "macro_labor"  # Unemployment, payrolls, wages, participation
    
    # Central Bank & Policy
    CENTRAL_BANK = "central_bank"  # Rate decisions, guidance, minutes, QT/QE
    FISCAL_POLICY = "fiscal_policy"  # Stimulus, taxes, budget, debt ceiling
    
    # Stress & Risk
    FINANCIAL_STRESS = "financial_stress"  # Banking stress, liquidity, credit events
    GEOPOLITICS = "geopolitics"  # War, sanctions, trade restrictions, elections
    
    # Commodities & Trade
    COMMODITIES_ENERGY = "commodities_energy"  # OPEC, supply disruptions, shipping
    FX_TRADE = "fx_trade"  # Tariffs, trade war, capital controls, currency
    
    # Regulatory
    REGULATION = "regulation"  # Major regulatory changes affecting sectors
    
    # Catch-all for relevant but unclassified
    MARKET_GENERAL = "market_general"  # General market news


@dataclass
class TaxonomyMatch:
    """Result of taxonomy classification."""
    tag: TaxonomyTag
    confidence: float  # 0-1
    matched_keywords: List[str]
    matched_rules: List[str]


class MacroTaxonomy:
    """
    Classify news articles into macro/geo/finance taxonomy.
    
    Uses keyword matching + rules as baseline.
    Extension point for ML classifier.
    """
    
    # Keyword patterns for each tag (lowercase, can be regex)
    TAG_KEYWORDS: Dict[TaxonomyTag, List[str]] = {
        TaxonomyTag.MACRO_INFLATION: [
            r'\bcpi\b', r'\bpce\b', r'inflation', r'consumer price', 
            r'producer price', r'ppi\b', r'price index', r'deflation',
            r'disinflation', r'inflation expectation', r'core inflation',
            r'headline inflation', r'cost of living', r'price pressure',
            r'wage inflation', r'sticky inflation', r'transitory',
        ],
        
        TaxonomyTag.MACRO_GROWTH: [
            r'\bgdp\b', r'gross domestic', r'\bpmi\b', r'purchasing manager',
            r'retail sales', r'industrial production', r'manufacturing',
            r'economic growth', r'contraction', r'expansion', r'recession',
            r'soft landing', r'hard landing', r'business activity',
            r'new orders', r'factory output', r'services sector',
            r'housing starts', r'durable goods', r'trade balance',
        ],
        
        TaxonomyTag.MACRO_LABOR: [
            r'unemployment', r'jobless', r'payroll', r'nonfarm payroll',
            r'job growth', r'job loss', r'labor market', r'hiring',
            r'layoff', r'firing', r'participation rate', r'wage growth',
            r'average hourly earning', r'initial claim', r'continuing claim',
            r'jolts', r'job opening', r'quit rate', r'labor force',
        ],
        
        TaxonomyTag.CENTRAL_BANK: [
            r'federal reserve', r'\bfed\b', r'\bfomc\b', r'powell',
            r'interest rate', r'rate decision', r'rate hike', r'rate cut',
            r'monetary policy', r'hawkish', r'dovish', r'tightening',
            r'quantitative', r'\bqe\b', r'\bqt\b', r'balance sheet',
            r'\becb\b', r'\bboj\b', r'bank of england', r'bank of japan',
            r'central bank', r'policy rate', r'terminal rate', r'dot plot',
            r'fed minutes', r'policy pivot', r'forward guidance',
            r'lagarde', r'yellen', r'brainard', r'waller', r'kashkari',
        ],
        
        TaxonomyTag.FISCAL_POLICY: [
            r'stimulus', r'fiscal', r'government spending', r'budget',
            r'debt ceiling', r'deficit', r'national debt', r'tax cut',
            r'tax hike', r'infrastructure bill', r'spending bill',
            r'appropriation', r'treasury secretary', r'government shutdown',
        ],
        
        TaxonomyTag.FINANCIAL_STRESS: [
            r'bank run', r'bank failure', r'banking crisis', r'liquidity',
            r'credit crunch', r'credit event', r'default', r'bankruptcy',
            r'insolvency', r'contagion', r'systemic risk', r'bail',
            r'deposit outflow', r'stress test', r'capital requirement',
            r'credit spread', r'yield spread', r'ted spread', r'libor',
            r'financial stability', r'emergency lending', r'discount window',
        ],
        
        TaxonomyTag.GEOPOLITICS: [
            r'war\b', r'conflict', r'military', r'invasion', r'sanction',
            r'geopolitical', r'ukraine', r'russia', r'taiwan', r'china',
            r'middle east', r'israel', r'iran', r'north korea', r'nato',
            r'escalation', r'de-escalation', r'ceasefire', r'peace talk',
            r'territorial', r'nuclear', r'missile', r'drone strike',
            r'election', r'coup', r'instability', r'regime change',
            r'diplomatic', r'embassy', r'hostage', r'terrorism',
        ],
        
        TaxonomyTag.COMMODITIES_ENERGY: [
            r'\bopec\b', r'oil price', r'crude', r'brent', r'wti\b',
            r'natural gas', r'lng\b', r'petroleum', r'energy price',
            r'supply disruption', r'production cut', r'output cut',
            r'inventory', r'stockpile', r'strategic reserve', r'spr\b',
            r'shipping lane', r'suez', r'panama canal', r'strait',
            r'refinery', r'pipeline', r'energy crisis', r'blackout',
            r'oil embargo', r'energy security', r'renewable',
        ],
        
        TaxonomyTag.FX_TRADE: [
            r'tariff', r'trade war', r'trade deficit', r'trade surplus',
            r'import', r'export', r'trade barrier', r'trade deal',
            r'trade negotiation', r'currency', r'exchange rate',
            r'capital control', r'capital flow', r'devaluation',
            r'dollar strength', r'dollar weakness', r'yen\b', r'euro\b',
            r'yuan', r'renminbi', r'forex', r'fx\b', r'carry trade',
        ],
        
        TaxonomyTag.REGULATION: [
            r'\bsec\b', r'securities and exchange', r'regulator',
            r'antitrust', r'\bftc\b', r'\bdoj\b', r'department of justice',
            r'investigation', r'probe', r'subpoena', r'enforcement',
            r'fine\b', r'penalty', r'settlement', r'consent decree',
            r'legislation', r'law passed', r'bill signed', r'executive order',
            r'compliance', r'deregulation', r'oversight', r'watchdog',
        ],
        
        TaxonomyTag.MARKET_GENERAL: [
            r'stock market', r'wall street', r'\bs&p\b', r'\bnasdaq\b',
            r'dow jones', r'market rally', r'market selloff', r'correction',
            r'bear market', r'bull market', r'volatility', r'\bvix\b',
            r'trading volume', r'market cap', r'investor sentiment',
        ],
    }
    
    # Contextual rules that boost confidence
    BOOST_RULES: Dict[TaxonomyTag, List[Tuple[str, float]]] = {
        TaxonomyTag.CENTRAL_BANK: [
            (r'rate (decision|hike|cut|hold)', 0.3),
            (r'(fomc|fed) (meeting|statement|minutes)', 0.3),
            (r'(hawkish|dovish) (surprise|tone|shift)', 0.2),
        ],
        TaxonomyTag.MACRO_INFLATION: [
            (r'cpi (beat|miss|surprise|higher|lower)', 0.3),
            (r'inflation (unexpectedly|surprisingly)', 0.2),
            (r'(core|headline) (cpi|pce|inflation)', 0.2),
        ],
        TaxonomyTag.GEOPOLITICS: [
            (r'(military|armed) (action|strike|conflict)', 0.3),
            (r'sanction.*(expand|escalat|new)', 0.2),
            (r'(war|conflict).*(escalat|intensif)', 0.3),
        ],
        TaxonomyTag.FINANCIAL_STRESS: [
            (r'bank.*(fail|run|crisis|collapse)', 0.3),
            (r'(liquidity|credit) (crisis|crunch|squeeze)', 0.3),
            (r'(emergency|bailout|rescue)', 0.2),
        ],
    }
    
    # Words that indicate this is NOT market-relevant news
    IRRELEVANT_PATTERNS: List[str] = [
        r'\b(football|basketball|baseball|hockey|soccer|nfl|nba|mlb|nhl)\b',
        r'\b(touchdown|quarterback|playoff|championship|super bowl)\b',
        r'\b(celebrity|kardashian|hollywood|movie|film|actor|actress)\b',
        r'\b(recipe|cooking|restaurant|chef|dining)\b',
        r'\b(weather forecast|horoscope|zodiac|astrology)\b',
        r'\b(wedding|divorce|dating|relationship|affair)\b',
        r'\b(local police|traffic accident|house fire)\b',
        r'\b(high school|elementary|middle school)\b',
        r'\bstate (legislature|assembly|senate)\b(?!.*fiscal)',
    ]
    
    def __init__(
        self,
        min_confidence: float = 0.3,
        use_ml_classifier: bool = False,
    ):
        """
        Initialize taxonomy classifier.
        
        Args:
            min_confidence: Minimum confidence to assign a tag
            use_ml_classifier: Whether to use ML classifier (future extension)
        """
        self.min_confidence = min_confidence
        self.use_ml_classifier = use_ml_classifier
        
        # Compile regex patterns for efficiency
        self._compiled_keywords: Dict[TaxonomyTag, List[re.Pattern]] = {}
        for tag, patterns in self.TAG_KEYWORDS.items():
            self._compiled_keywords[tag] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self._compiled_boosts: Dict[TaxonomyTag, List[Tuple[re.Pattern, float]]] = {}
        for tag, rules in self.BOOST_RULES.items():
            self._compiled_boosts[tag] = [
                (re.compile(p, re.IGNORECASE), boost) for p, boost in rules
            ]
        
        self._compiled_irrelevant = [
            re.compile(p, re.IGNORECASE) for p in self.IRRELEVANT_PATTERNS
        ]
    
    def is_relevant(self, title: str, body: str = "") -> Tuple[bool, float]:
        """
        Check if article is relevant for macro/geo/finance analysis.
        
        Returns:
            Tuple of (is_relevant, relevance_score)
        """
        text = f"{title} {body}".lower()
        
        # Check for irrelevant patterns (hard reject)
        for pattern in self._compiled_irrelevant:
            if pattern.search(text):
                return False, 0.0
        
        # Check for any taxonomy matches
        matches = self.classify(title, body)
        
        if not matches:
            return False, 0.0
        
        # Calculate relevance score
        max_confidence = max(m.confidence for m in matches)
        
        if max_confidence >= self.min_confidence:
            return True, max_confidence
        
        return False, max_confidence
    
    def classify(
        self,
        title: str,
        body: str = "",
        max_tags: int = 3,
    ) -> List[TaxonomyMatch]:
        """
        Classify article into taxonomy tags.
        
        Args:
            title: Article title/headline
            body: Article body text
            max_tags: Maximum number of tags to return
            
        Returns:
            List of TaxonomyMatch objects, sorted by confidence
        """
        text = f"{title} {body}"
        matches: List[TaxonomyMatch] = []
        
        for tag, patterns in self._compiled_keywords.items():
            matched_keywords = []
            
            # Count keyword matches
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    matched_keywords.extend(found)
            
            if not matched_keywords:
                continue
            
            # Base confidence from keyword density
            unique_matches = len(set(matched_keywords))
            base_confidence = min(0.7, 0.2 + unique_matches * 0.15)
            
            # Apply boost rules
            matched_rules = []
            if tag in self._compiled_boosts:
                for pattern, boost in self._compiled_boosts[tag]:
                    if pattern.search(text):
                        base_confidence += boost
                        matched_rules.append(pattern.pattern)
            
            # Title match bonus
            title_lower = title.lower()
            for pattern in patterns[:5]:  # Check top patterns
                if pattern.search(title_lower):
                    base_confidence += 0.15
                    break
            
            confidence = min(1.0, base_confidence)
            
            if confidence >= self.min_confidence:
                matches.append(TaxonomyMatch(
                    tag=tag,
                    confidence=confidence,
                    matched_keywords=list(set(matched_keywords))[:5],
                    matched_rules=matched_rules,
                ))
        
        # Sort by confidence and limit
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:max_tags]
    
    def get_primary_tag(self, title: str, body: str = "") -> Optional[TaxonomyTag]:
        """Get the highest confidence tag, or None if not relevant."""
        matches = self.classify(title, body, max_tags=1)
        return matches[0].tag if matches else None
