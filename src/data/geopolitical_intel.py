"""
Geopolitical Intelligence Layer

Monitors global events that impact markets:
- Military tensions and conflicts
- Regional instability
- Flight/travel disruptions (early warning signals)
- Diplomatic crises
- Sanctions and trade tensions
- Regional market reactions

Data Sources:
1. NewsAPI - Real-time global news
2. GDELT Project - Global event database (conflicts, protests)
3. RSS feeds - Reuters, BBC, Al Jazeera
4. Regional market data - Middle East, Asia indices
5. Flight data - Disruption signals

Now with ADVANCED RELEVANCE FILTERING:
- Rule-based market-moving detection
- Hard discard for irrelevant content
- Scoring: relevance, impact, credibility, novelty
- Direction inference for market impact
"""

import os
import json
import logging
import hashlib
import requests
import feedparser
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import re

# Import the advanced relevance filter
try:
    from src.data.news_relevance_filter import get_news_filter, NewsEvent, NewsCategory, MarketDirection
    HAS_RELEVANCE_FILTER = True
except ImportError:
    HAS_RELEVANCE_FILTER = False
    logging.warning("NewsRelevanceFilter not available, using basic filtering")

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass
class GeopoliticalEvent:
    """Represents a geopolitical event that may impact markets."""
    event_id: str
    timestamp: datetime
    headline: str
    summary: str
    source: str
    event_type: str  # military, diplomatic, economic, civil_unrest, natural_disaster
    severity: float  # 0-1 scale
    regions: List[str]  # affected regions
    keywords: List[str]
    market_impact_score: float  # estimated market impact 0-1
    url: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "event_type": self.event_type,
            "severity": self.severity,
            "regions": self.regions,
            "keywords": self.keywords,
            "market_impact_score": self.market_impact_score,
            "url": self.url,
        }


@dataclass
class GeopoliticalRiskAssessment:
    """
    Comprehensive geopolitical risk assessment with multi-timeframe analysis.
    
    NEW FEATURES:
    - Tail risk indicator for black swan events
    - Multi-timeframe risk (immediate/situational/structural)
    - Active conflict tracking
    - Graduated safe haven signals
    """
    timestamp: datetime
    overall_risk_score: float  # 0-1 (combined weighted score)
    risk_level: str  # low, guarded, moderate, elevated, high, critical
    active_events: List[GeopoliticalEvent]
    regional_risks: Dict[str, float]  # region -> risk score
    recommended_exposure_adjustment: float  # multiplier 0.0-1.0
    key_concerns: List[str]
    safe_haven_signal: bool  # should rotate to safe havens?
    
    # NEW: Multi-timeframe risk assessment
    immediate_risk: float = 0.0       # Last 6 hours, fast decay - intraday trading
    situational_risk: float = 0.0     # Last 48 hours - swing trading
    structural_risk: float = 0.0      # Ongoing conflicts - position management
    
    # NEW: Tail risk for black swan probability
    tail_risk_score: float = 0.0      # 0-1, probability of catastrophic escalation
    tail_risk_triggers: List[str] = field(default_factory=list)  # Active tail risk triggers
    
    # NEW: Active conflicts being tracked
    active_conflicts: List[str] = field(default_factory=list)  # e.g., ["ukraine_russia", "israel_hamas"]
    
    # NEW: Graduated safe haven signal
    safe_haven_level: str = "none"    # none, hedging, moderate, strong
    
    # NEW: Market validation
    market_confirms_risk: bool = False  # True if VIX/gold/bonds confirm the risk
    vix_level: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level,
            "active_events_count": len(self.active_events),
            "active_events": [e.to_dict() for e in self.active_events[:10]],
            "regional_risks": self.regional_risks,
            "recommended_exposure_adjustment": self.recommended_exposure_adjustment,
            "key_concerns": self.key_concerns,
            "safe_haven_signal": self.safe_haven_signal,
            # NEW fields
            "immediate_risk": self.immediate_risk,
            "situational_risk": self.situational_risk,
            "structural_risk": self.structural_risk,
            "tail_risk_score": self.tail_risk_score,
            "tail_risk_triggers": self.tail_risk_triggers,
            "active_conflicts": self.active_conflicts,
            "safe_haven_level": self.safe_haven_level,
            "market_confirms_risk": self.market_confirms_risk,
            "vix_level": self.vix_level,
        }


class GeopoliticalIntelligence:
    """
    Comprehensive geopolitical risk monitoring system.
    
    Monitors multiple data sources to detect events that could impact markets.
    Uses SENTIMENT-AWARE keyword classification to distinguish between
    escalation (risk-increasing) and de-escalation (risk-decreasing) events.
    """
    
    # ==========================================================================
    # SEVERITY TIERS - Different risk levels for different event intensities
    # ==========================================================================
    SEVERITY_TIERS = {
        "CRITICAL": {"range": (0.8, 1.0), "description": "Imminent major market impact"},
        "HIGH": {"range": (0.6, 0.8), "description": "Significant market-moving event"},
        "ELEVATED": {"range": (0.4, 0.6), "description": "Moderate concern, monitor closely"},
        "GUARDED": {"range": (0.2, 0.4), "description": "Low-level concern"},
        "LOW": {"range": (0.0, 0.2), "description": "Normal conditions"},
    }
    
    # ==========================================================================
    # ESCALATION KEYWORDS - Events that INCREASE geopolitical risk
    # ==========================================================================
    ESCALATION_KEYWORDS = {
        # ------------------------------------------------------------------
        # CATEGORY 1: MILITARY & DEFENSE - Active Combat & Tensions
        # ------------------------------------------------------------------
        "military_combat": [
            # Active Combat
            "attack", "attacked", "attacking", "strike", "airstrike", "strikes", "struck",
            "bombing", "bombed", "bomb blast", "shelling", "shelled", "artillery fire",
            "invasion", "invaded", "invading", "assault", "offensive", "ground offensive",
            "raid", "raided", "ambush", "firefight", "clashes", "skirmish",
            # Weapons & Deployment
            "missiles", "missile launch", "ballistic", "troops deployed", "troop buildup",
            "mobilization", "warships", "naval fleet", "carrier group", "fighter jets",
            "aircraft scrambled", "drones", "drone strike", "uav attack", "tanks",
            "armored vehicles", "nuclear", "nuclear threat", "atomic", "hypersonic",
            "icbm", "cruise missile", "chemical weapons", "biological weapons",
            # Military Exercises & Tensions (can escalate)
            "military drills", "military exercises", "war games", "naval exercises",
            "live fire exercise", "show of force", "gunboat diplomacy",
            # Casualties & Damage
            "casualties", "killed", "deaths", "fatalities", "wounded", "injured",
            "victims", "destroyed", "destruction", "devastation", "civilian deaths",
            "collateral damage", "mass casualty", "massacre", "war crimes", "atrocities",
            # Military Escalation
            "escalation", "escalating", "escalates", "retaliation", "retaliatory",
            "counterattack", "counter-offensive", "declared war", "war declaration",
            "martial law", "state of emergency", "full-scale", "all-out",
            "military operation", "special operation",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 2: DIPLOMATIC BREAKDOWN
        # ------------------------------------------------------------------
        "diplomatic_breakdown": [
            # Diplomatic Breakdown
            "breakdown", "collapsed", "failed", "talks failed", "negotiations collapsed",
            "walked out", "stormed out", "deadlock", "stalemate", "impasse",
            "diplomatic rift", "severed ties", "expelled diplomats", "embassy closed",
            "recalled ambassador",
            # Threats & Ultimatums
            "ultimatum", "final warning", "threatens", "threatened", "threatening",
            "red line", "crossed red line", "provocation", "provocative",
            "hostile", "hostility", "rejects", "rejected", "refuses",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 3: POLITICAL INSTABILITY
        # ------------------------------------------------------------------
        "political_instability": [
            # Government Collapse
            "coup", "coup attempt", "military takeover", "regime change",
            "government overthrown", "government collapse", "government falls",
            "assassination", "attempted assassination",
            "political crisis", "constitutional crisis", "impeachment", "ousted",
            "removed from power", "state of siege", "power vacuum",
            "prime minister resigns", "president resigns", "cabinet resigns",
            # Civil Unrest & Protests
            "mass protests", "protests escalate", "riots", "civil unrest",
            "revolution", "uprising", "violent protests", "demonstrations",
            "general strike", "nationwide strike", "civil disobedience",
            "protesters killed", "police crackdown", "state of emergency",
            # Disputed Elections
            "disputed election", "election fraud", "election violence",
            "contested results", "refuses to concede", "electoral crisis",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 4: SANCTIONS & RESTRICTIONS (Imposed)
        # ------------------------------------------------------------------
        "sanctions_imposed": [
            "sanctions", "sanctions imposed", "new sanctions", "sanctions package",
            "sanctions regime", "asset freeze", "assets frozen", "travel ban",
            "visa restrictions", "blacklist", "blacklisted", "entity list",
            "embargo", "trade embargo", "blockade", "naval blockade",
            # Financial Restrictions
            "swift ban", "swift exclusion", "banking sanctions", "financial sanctions",
            "secondary sanctions", "treasury designation", "ofac",
            # Sector Sanctions
            "oil embargo", "energy sanctions", "technology sanctions", "chip ban",
            "arms embargo", "weapons ban", "luxury goods ban",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 5: ENERGY & COMMODITY DISRUPTION
        # ------------------------------------------------------------------
        "energy_disruption": [
            # Infrastructure Attacks
            "pipeline attack", "pipeline sabotage", "oil facility attack",
            "refinery fire", "refinery explosion", "power plant attack",
            "pipeline explosion", "tanker seized", "ship attacked",
            # Supply Disruptions
            "supply disruption", "supply cut", "production halt",
            "output cut", "export ban", "export halt", "production cut",
            "opec cut", "supply shortage", "oil embargo", "gas embargo",
            "fuel shortage", "energy crisis", "power outage", "blackout",
            # Strategic Chokepoints (CRITICAL - high severity boost)
            "strait of hormuz blocked", "hormuz blocked", "hormuz closed",
            "suez canal blocked", "suez blocked", "suez closed",
            "strait of malacca", "bab el-mandeb", "panama canal blocked",
            "shipping blocked", "strait closed", "chokepoint",
            "red sea attacks", "houthi attacks", "shipping rerouted",
            # Price Shocks
            "oil spike", "price surge", "oil prices surge", "gas prices surge",
            "nord stream", "port blocked", "terminal closed",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 6: TERRORISM & SECURITY THREATS
        # ------------------------------------------------------------------
        "terrorism": [
            "terrorist attack", "terror attack", "suicide bombing",
            "mass shooting", "gunman", "hostage", "hostages taken", "kidnapping",
            "abduction", "attack claimed by",
            "terror threat", "threat level raised", "security alert", "high alert",
            "intelligence warning", "imminent threat", "credible threat",
            # Terror Groups (need word boundaries - handled in check_keyword)
            "islamic state", "al-qaeda", "taliban", "hezbollah",
            "hamas", "boko haram", "al-shabaab", "terror cell", "extremist",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 7: CYBER ATTACKS
        # ------------------------------------------------------------------
        "cyber_attacks": [
            "cyberattack", "cyber attack", "hacked", "ransomware", "malware",
            "data breach", "data stolen", "critical infrastructure attack",
            "power grid attack", "grid down", "ddos attack", "systems down",
            "state-sponsored hack", "technology war", "chip war",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 8: FINANCIAL STRESS & CRISIS
        # ------------------------------------------------------------------
        "financial_crisis": [
            "default", "debt default", "sovereign default", "bankruptcy",
            "insolvency", "bank run", "financial panic", "currency crisis",
            "currency collapse", "hyperinflation", "stagflation", "recession",
            "depression", "economic collapse", "market crash", "circuit breaker",
            "bank failure", "banking crisis", "deposit outflows", "liquidity stress",
            "capital adequacy", "loan losses", "debt restructuring", "credit downgrade",
            "debt distress", "capital shortfall", "liquidity crisis",
            "sovereign debt crisis", "distressed debt", "contagion risk",
            "funding stress", "margin calls", "systemic risk",
            "debt ceiling", "government shutdown", "fiscal cliff", "credit rating downgrade",
            "treasury auction", "bond sell-off", "yield spike",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 9: TRADE WAR & ECONOMIC WARFARE
        # ------------------------------------------------------------------
        "trade_war": [
            "tariffs", "tariff increase", "import duties", "trade war",
            "trade dispute", "trade tensions", "protectionism", "trade barriers",
            "dumping", "anti-dumping", "retaliatory tariffs", "trade deficit widens",
            "economic warfare", "economic attack", "currency manipulation",
            "technology ban", "export controls", "supply chain disruption",
            "decoupling", "economic decoupling", "investment ban", "capital controls",
            "rare earth ban", "rare earth export", "semiconductor ban", "chip export",
            "trade retaliation", "counter-tariffs", "trade escalation",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 9b: REGULATORY & POLICY SHOCKS
        # ------------------------------------------------------------------
        "regulatory_shock": [
            # Nationalization & Expropriation
            "nationalization", "nationalize", "nationalized", "expropriation",
            "seize assets", "assets seized", "government takeover",
            "forced sale", "compulsory acquisition",
            # Capital Controls
            "capital controls", "currency controls", "currency not convertible",
            "bank holiday", "deposit freeze", "withdrawal limits",
            "foreign exchange ban", "fx controls",
            # Regulatory Crackdowns
            "regulatory crackdown", "antitrust action", "break up",
            "banned", "outlawed", "prohibited", "license revoked",
            "sector ban", "industry ban", "sudden regulation",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 10: NATURAL DISASTERS & CLIMATE
        # ------------------------------------------------------------------
        "natural_disasters": [
            "earthquake", "tsunami", "hurricane", "typhoon", "cyclone", "tornado",
            "flooding", "floods", "flash flood", "wildfire", "forest fire",
            "volcanic eruption", "drought", "famine", "heatwave", "cold snap",
            "emergency declared", "mass evacuation", "infrastructure damage",
            "supply chain impact", "crop failure", "harvest destroyed",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 11: PANDEMIC & HEALTH EMERGENCY
        # ------------------------------------------------------------------
        "pandemic": [
            "outbreak", "epidemic", "pandemic", "new variant", "mutation",
            "cases surge", "hospitalization spike", "lockdown", "quarantine",
            "travel ban", "border closed", "health emergency", "who declares",
            "death toll rises", "fatality rate", "overwhelmed hospitals",
            "icu capacity", "vaccine resistant", "immune escape",
        ],
    }
    
    # ==========================================================================
    # DE-ESCALATION KEYWORDS - Events that DECREASE geopolitical risk
    # ==========================================================================
    DE_ESCALATION_KEYWORDS = {
        # ------------------------------------------------------------------
        # CATEGORY 1: PEACE & MILITARY DE-ESCALATION
        # ------------------------------------------------------------------
        "peace_process": [
            "ceasefire", "truce", "armistice", "peace deal", "peace agreement",
            "peace treaty", "peace talks", "peace process", "cessation of hostilities",
            "laying down arms", "end of war", "war ends", "conflict resolved",
            # Military Withdrawal
            "withdrawal", "withdrawing", "pullback", "troops withdrawn", "pulling out",
            "de-escalation", "de-escalating", "stand down", "stepping back",
            "demobilization", "demobilizing", "retreating", "retreat",
            # Diplomatic Military
            "military hotline", "deconfliction", "buffer zone", "safe zone",
            "peacekeepers", "peacekeeping force", "un observers", "monitoring mission",
            # Disarmament
            "denuclearize", "denuclearization", "disarmament", "weapons dismantled",
            "nuclear deal", "arms reduction", "treaty signed",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 2: POSITIVE DIPLOMACY
        # ------------------------------------------------------------------
        "diplomatic_progress": [
            "talks", "negotiations", "dialogue", "agreement", "accord", "deal reached",
            "breakthrough", "progress", "momentum", "willing to negotiate",
            "open to talks", "constructive", "productive talks", "framework agreement",
            "memorandum of understanding", "mou signed",
            # Positive Signals
            "signals willingness", "willing to", "open to", "agree to",
            "resume talks", "restart negotiations", "return to table",
            "diplomatic channels", "positive signal",
            # Relationship Improvement
            "normalization", "normalizing ties", "thaw", "warming relations",
            "cooperation", "collaboration", "partnership", "alliance strengthened",
            "reconciliation", "rapprochement", "diplomatic solution",
            "peaceful resolution", "mutual understanding",
            # Confidence Building
            "confidence building measures", "goodwill gesture", "olive branch",
            "prisoner exchange", "hostage release", "humanitarian corridor",
            "back channel", "quiet diplomacy",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 3: SANCTIONS RELIEF
        # ------------------------------------------------------------------
        "sanctions_relief": [
            "sanctions lifted", "sanctions removed", "sanctions relief",
            "sanctions waiver", "sanctions exemption", "delisted",
            "removed from blacklist", "unfrozen", "assets released", "embargo lifted",
            "restrictions eased", "trade resumed",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 4: ECONOMIC STABILITY & RECOVERY
        # ------------------------------------------------------------------
        "economic_recovery": [
            "trade deal", "trade agreement", "tariffs reduced", "tariffs lifted",
            "free trade", "trade liberalization", "market access", "trade opening",
            "trade surplus", "export growth", "stimulus", "economic support",
            "bailout", "rescue package", "recovery", "economic rebound", "growth",
            "gdp growth", "investment", "foreign investment", "stabilization",
            "market calm", "confidence returns",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 5: ENERGY STABILITY
        # ------------------------------------------------------------------
        "energy_stability": [
            "production increase", "output boost", "supply restored",
            "shipments resume", "pipeline reopened", "opec increase", "quota raised",
            "strategic reserve release", "alternative supply", "prices stabilize",
            "market calm", "inventory build", "stockpiles rise", "demand eases",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 6: SECURITY IMPROVEMENT
        # ------------------------------------------------------------------
        "security_improvement": [
            "threat neutralized", "threat eliminated", "terror leader killed",
            "captured", "cell dismantled", "network disrupted", "threat level lowered",
            "all clear", "situation resolved", "hostages freed", "hostages released",
            "security upgraded", "systems restored", "back online",
            "vulnerability patched", "cyber cooperation", "security enhanced",
        ],
        
        # ------------------------------------------------------------------
        # CATEGORY 7: HEALTH IMPROVEMENT
        # ------------------------------------------------------------------
        "health_improvement": [
            "cases declining", "curve flattening", "restrictions lifted", "reopening",
            "vaccine rollout", "vaccination campaign", "herd immunity", "endemic",
            "treatment approved", "cure found", "outbreak contained",
        ],
    }
    
    # ==========================================================================
    # CONTEXT-DEPENDENT KEYWORDS - Need surrounding context to classify
    # ==========================================================================
    CONTEXT_KEYWORDS = {
        # Format: keyword -> (escalation_context_words, de_escalation_context_words)
        "war": (
            ["declares", "begins", "erupts", "intensifies", "casualties", "attacks"],
            ["talks", "ends", "ceasefire", "peace", "negotiations", "resolved"]
        ),
        "military": (
            ["action", "strike", "offensive", "buildup", "deploys", "attacks"],
            ["talks", "diplomacy", "withdrawal", "cooperation", "training"]
        ),
        "conflict": (
            ["escalates", "intensifies", "spreads", "violence", "fighting"],
            ["resolution", "resolved", "ends", "de-escalation", "peace"]
        ),
        "forces": (
            ["attack", "advance", "invade", "strike", "mobilize"],
            ["withdraw", "return", "retreat", "stand down", "leave"]
        ),
        "troops": (
            ["deploy", "advance", "attack", "mobilize", "buildup"],
            ["withdraw", "return home", "pullback", "leave", "retreat"]
        ),
        "weapons": (
            ["used", "deployed", "fired", "launch", "strike"],
            ["inspectors", "destroyed", "dismantled", "ban", "treaty"]
        ),
        "nuclear": (
            ["threat", "test", "program", "weapons", "arsenal", "attack", "launch", "warhead"],
            ["deal", "agreement", "disarmament", "inspection", "treaty", "talks", "denuclearize", "willingness"]
        ),
        "sanctions": (
            ["imposed", "new", "expanded", "tightened", "added", "announces", "slaps"],
            ["lifted", "removed", "eased", "waived", "suspended", "impact", "working", "effect"]
        ),
        "trade": (
            ["war", "dispute", "tensions", "tariffs", "barriers"],
            ["deal", "agreement", "talks", "cooperation", "opening"]
        ),
        "summit": (
            ["canceled", "failed", "collapsed", "postponed", "no progress"],
            ["agreed", "successful", "productive", "breakthrough", "signed"]
        ),
        "talks": (
            ["collapse", "fail", "stall", "breakdown", "deadlock"],
            ["progress", "breakthrough", "agree", "productive", "resume"]
        ),
        "crisis": (
            ["deepens", "worsens", "escalates", "spreads", "intensifies"],
            ["eases", "resolved", "contained", "stabilizes", "over"]
        ),
        "tensions": (
            ["rise", "escalate", "increase", "mount", "grow"],
            ["ease", "reduce", "calm", "de-escalate", "subside"]
        ),
        # NEW: Business/corporate terms that shouldn't be geopolitical
        "threatens": (
            ["military", "attack", "war", "retaliation", "strike", "invasion", "missile"],
            ["delay", "lawsuit", "legal", "business", "revenue", "profit", "stock"]
        ),
        "raid": (
            ["military", "air", "airstrike", "drone", "commando", "special forces", "terrorist"],
            ["police", "immigration", "ice", "tax", "fbi", "sec", "antitrust"]
        ),
    }
    
    # ==========================================================================
    # BUSINESS-CONTEXT EXCLUSIONS - Words that negate geopolitical classification
    # If these appear near a keyword, reduce its geopolitical weight
    # ==========================================================================
    BUSINESS_CONTEXT_WORDS = [
        "stock", "shares", "profit", "revenue", "earnings", "quarterly",
        "market cap", "valuation", "ipo", "merger", "acquisition",
        "ceo", "cfo", "board", "shareholders", "dividend", "buyback",
        "software", "app", "platform", "startup", "tech company",
        "lawsuit", "legal", "court", "attorney", "settlement",
    ]
    
    # ==========================================================================
    # REGIONAL-SPECIFIC HIGH-RISK KEYWORDS
    # ==========================================================================
    REGIONAL_KEYWORDS = {
        "middle_east": [
            "iran", "tehran", "ayatollah", "revolutionary guard", "irgc",
            "israel", "idf", "netanyahu", "gaza", "west bank", "hamas",
            "hezbollah", "houthis", "saudi", "riyadh", "mbs", "aramco",
            "strait of hormuz", "persian gulf", "nuclear deal", "jcpoa",
            "yemen", "lebanon", "syria", "iraq",
        ],
        "russia_ukraine": [
            "putin", "kremlin", "moscow", "zelensky", "kyiv", "kiev",
            "crimea", "donbas", "donetsk", "luhansk", "nato expansion",
            "article 5", "wagner", "prigozhin", "nord stream", "gazprom",
            "russian forces", "ukrainian forces",
        ],
        "asia_pacific": [
            "china", "beijing", "xi jinping", "ccp", "taiwan", "taipei",
            "strait", "south china sea", "spratlys", "north korea",
            "pyongyang", "kim jong un", "india-china", "lac", "galwan",
            "aukus", "quad",
        ],
        "europe": [
            "nato", "eu", "brussels", "brexit", "eurozone", "balkans",
            "serbia", "kosovo", "ecb", "european commission",
        ],
    }
    
    # ==========================================================================
    # ORIGINAL HIGH_IMPACT_KEYWORDS (for backward compatibility & economic data)
    # ==========================================================================
    HIGH_IMPACT_KEYWORDS = {
        # Central Banks & Monetary Policy
        "central_bank": [
            "federal reserve", "fed", "fomc", "ecb", "european central bank",
            "bank of england", "boe", "bank of japan", "boj", "pboc",
            "jerome powell", "powell", "lagarde", "christine lagarde",
            "interest rate hike", "interest rate cut", "rate hike", "rate cut",
            "rate decision", "policy tightening", "policy easing",
            "quantitative easing", "qe", "quantitative tightening", "qt",
            "hawkish", "dovish", "monetary policy", "fomc meeting",
        ],
        
        # Macroeconomic Indicators
        "economic": [
            "cpi", "pce", "core cpi", "core pce", "inflation", "core inflation",
            "unemployment", "unemployment rate", "jobless claims", "initial claims",
            "nonfarm payrolls", "non-farm payrolls", "nfp", "wage growth",
            "gdp", "gdp growth", "recession", "economic slowdown", "contraction",
            "pmi", "ism", "retail sales", "consumer spending", "consumer confidence",
        ],
        
        # Financial Markets
        "financial_stress": [
            "yield curve", "curve inversion", "credit spreads", "market volatility",
            "vix", "volatility spike", "risk-off", "risk-on", "flight to safety",
            "market selloff", "market rally", "liquidity crunch",
        ],
        
        # Shipping & Supply Chain
        "shipping": [
            "shipping", "shipping disruption", "port congestion", "container shortage",
            "freight rates", "suez canal", "panama canal", "strait of hormuz",
            "red sea", "south china sea", "supply chain", "logistics",
        ],
        
        # Energy
        "energy": [
            "opec", "opec+", "oil", "crude oil", "brent", "wti", "natural gas",
            "lng", "energy supply", "energy crisis", "production cut", "output cut",
            "gold", "safe haven", "commodities",
        ],
    }
    
    # SEVERITY MODIFIERS - Words that boost impact score
    SEVERITY_MODIFIERS = [
        "unexpectedly", "sharply", "emergency", "historic", "unprecedented",
        "sudden", "significant", "severe", "escalates", "intensifies",
        "collapses", "surges", "spikes", "plunges", "widens", "narrows",
        "crisis", "rare", "record", "shock", "imminent", "breaking",
        "urgent", "major", "massive", "critical", "dramatic",
    ]
    
    # URGENCY WORDS - Words that indicate immediate market relevance
    URGENCY_WORDS = [
        "breaking", "urgent", "imminent", "immediate", "critical",
        "major", "massive", "flash", "just in", "developing",
    ]
    
    # HARD DISCARD - Events that should NEVER be classified as market-moving
    # These are local news, accidents, and irrelevant events
    HARD_DISCARD_KEYWORDS = [
        # Local disasters (not market-moving unless massive scale)
        "ferry sinks", "ferry sank", "ferry capsizes", "ferry accident", "boat sinks",
        "bus crash", "train derailment", "plane crash", "car accident",
        "building collapse", "house fire", "apartment fire", "factory fire",
        "people dead", "people killed", "bodies found", "bodies recovered",
        "missing persons", "search and rescue", "rescue operation", "rescue workers",
        "rescuers save", "survivors found", "survivors rescued", "death toll",
        "people onboard", "passengers aboard", "passengers rescued",
        # Crime/Local
        "arrested", "murder", "robbery", "burglary", "theft", "assault",
        "shooting", "stabbing", "drug bust", "gang violence",
        "local police", "city council", "town hall",
        # Domestic US Politics (not geopolitical unless trade/fiscal/regulatory)
        "ice raids", "immigration enforcement", "border patrol", "deportation",
        "immigration policy", "migrants detained", "asylum seekers",
        "gun control", "abortion", "congressional hearing",
        "campaign trail", "polling",
        # Sports/Entertainment
        "world cup", "olympics", "olympic gold", "olympic medal", "olympic games",
        "skating gold", "hockey gold", "swimming gold", "gymnast",
        "football", "soccer", "basketball", "baseball", "cricket",
        "tennis", "golf", "rugby", "boxing", "wrestling", "marathon",
        "celebrity", "kardashian", "hollywood", "movie", "film festival",
        "concert", "grammy", "oscar", "emmy", "red carpet", "netflix",
        "prince andrew", "prince harry", "royal family", "monarchy",
        "bug of the year", "insect", "moth named", "beetle",
        # Historical / Opinion / Lifestyle
        "spanish armada", "teaches us about", "lessons from history",
        "recipe", "cooking", "restaurant", "travel tips", "fashion",
        "diet", "workout", "yoga", "meditation", "home decor",
        # Nature / Weather (not infrastructure)
        "avalanche", "avalanches", "weather forecast", "sunny", "cloudy",
        "chance of rain", "wildflower", "wildlife", "zoo ",
        # UK fluff / campaigns
        "savvy squirrel",
        # Crime / Missing persons / Local
        "missing person", "cold case", "serial killer", "true crime",
        "record label investment", "eyed record label",
        # Social media / Tech regulation (unless antitrust)
        "social media ban", "under-15", "age verification", "content moderation",
        "parental controls", "screen time", "online safety",
        # Podcasts / Media listings
        "podcast", "(podcast)", "listen now", "episode recap",
        # Obituaries
        "obituary", "funeral", "passed away", "dies at",
        # Business/Corporate (not geopolitical)
        "software sell-off", "stock buyback", "quarterly earnings", "annual report",
        "profit warning", "revenue growth", "market share", "product launch",
        "ceo resigns", "board meeting", "shareholder vote", "ipo filing",
    ]
    
    # MINIMUM SEVERITY THRESHOLD - Events below this are discarded
    MIN_SEVERITY_THRESHOLD = 0.25
    
    # ==========================================================================
    # NEW: FAILED NEGOTIATION PATTERNS (P0 Fix)
    # These OVERRIDE de-escalation keywords - "peace talks" with failure = ESCALATION
    # ==========================================================================
    NEGOTIATION_FAILURE_PATTERNS = [
        "without breakthrough", "no breakthrough", "talks fail", "talks failed",
        "talks collapse", "talks collapsed", "negotiations fail", "negotiations failed",
        "no progress", "little progress", "stalled", "deadlock", "impasse",
        "without agreement", "no agreement reached", "talks end without",
        "collapsed", "broke down", "breakdown", "walked out", "stormed out",
        "rejected", "refuses to", "ultimatum", "no deal", "deal falls through",
        "suspended", "postponed indefinitely", "stalemate continues",
    ]
    
    # ==========================================================================
    # NEW: CONCRETE DE-ESCALATION OUTCOMES (P1 Fix)
    # Only these actually reduce risk - "talks happening" doesn't count
    # ==========================================================================
    CONCRETE_DE_ESCALATION_OUTCOMES = [
        "ceasefire signed", "ceasefire begins", "ceasefire holds",
        "agreement reached", "agreement signed", "deal struck", "deal signed",
        "treaty signed", "accord reached", "peace deal",
        "troops withdrawn", "troops withdrawing", "pullback begins",
        "sanctions lifted", "sanctions removed", "embargo lifted",
        "hostages freed", "hostages released", "prisoners released",
        "prisoners exchanged", "humanitarian corridor opens",
        "weapons inspectors", "inspections begin", "disarmament",
        "border reopened", "flights resume", "trade resumes",
    ]
    
    # ==========================================================================
    # NEW: ACTIVE CONFLICTS - Structural Risk Floor (P0 Fix)
    # These ongoing conflicts should NEVER score below their floor
    # ==========================================================================
    ACTIVE_CONFLICTS = {
        "ukraine_russia": {
            "start_date": "2022-02-24",
            "min_risk_floor": 0.10,  # Priced in — markets have adapted. Only escalation spikes matter.
            "affected_regions": ["russia", "europe"],
            "escalation_triggers": ["nuclear", "nato direct", "article 5", "tactical nuclear"],
            "keywords": ["ukraine", "russia", "kyiv", "moscow", "zelensky", "putin", "crimea", "donbas"],
        },
        "israel_hamas": {
            "start_date": "2023-10-07",
            "min_risk_floor": 0.08,  # Priced in — ongoing for 2+ years
            "affected_regions": ["middle_east"],
            "escalation_triggers": ["iran strikes israel", "hezbollah full", "regional war"],
            "keywords": ["gaza", "israel", "hamas", "netanyahu", "hezbollah", "iran"],
        },
        "iran_tensions": {
            "start_date": "2024-01-01",
            "min_risk_floor": 0.06,  # Priced in — chronic background
            "affected_regions": ["middle_east"],
            "escalation_triggers": ["iran nuclear", "strait of hormuz", "iran attacks"],
            "keywords": ["iran", "tehran", "irgc", "revolutionary guard", "houthis"],
        },
        "taiwan_tensions": {
            "start_date": "2022-08-01",
            "min_risk_floor": 0.05,  # Priced in — no active hostilities
            "affected_regions": ["asia"],
            "escalation_triggers": ["china invades taiwan", "blockade taiwan", "taiwan strait crisis"],
            "keywords": ["taiwan", "taipei", "china", "strait", "pla"],
        },
    }
    
    # ==========================================================================
    # NEW: KEYWORD SEVERITY WEIGHTS (P1 Fix)
    # Different keywords have different severity - "nuclear attack" >> "tensions"
    # ==========================================================================
    KEYWORD_SEVERITY_WEIGHTS = {
        # CRITICAL (10x weight)
        "nuclear attack": 10.0, "nuclear war": 10.0, "nuclear strike": 10.0,
        "article 5": 10.0, "nato direct": 9.0,
        
        # VERY HIGH (7-8x weight)
        "invasion": 8.0, "invades": 8.0, "declares war": 8.0, "war declared": 8.0,
        "nuclear threat": 7.0, "nuclear test": 7.0,
        "strait of hormuz blocked": 8.0, "suez blocked": 7.0,
        
        # HIGH (4-6x weight)
        "attack": 5.0, "airstrike": 5.0, "missile strike": 5.0,
        "sanctions imposed": 4.0, "embargo": 4.0,
        "coup": 6.0, "assassination": 6.0,
        "default": 5.0, "bank run": 5.0,
        
        # MEDIUM (2-3x weight)
        "escalation": 3.0, "escalating": 3.0, "retaliation": 3.0,
        "military buildup": 3.0, "troops deployed": 3.0,
        "tariffs": 2.0, "trade war": 2.0,
        
        # LOW (1x weight - default)
        "tensions": 1.0, "concerns": 1.0, "dispute": 1.0,
    }
    
    # ==========================================================================
    # NEW: TIME DECAY BY EVENT TYPE (P1 Fix)
    # Military events persist longer than terrorism
    # ==========================================================================
    TIME_DECAY_HALF_LIFE = {
        "military_combat": 24,       # War events persist 24h half-life
        "diplomatic_breakdown": 48,  # Diplomatic rifts last longer
        "political_instability": 36, # Political events persist
        "sanctions_imposed": 72,     # Sanctions last weeks
        "energy_disruption": 24,     # Energy shocks are urgent
        "terrorism": 6,              # Security threats are fast-moving
        "cyber_attacks": 12,         # Cyber events moderate
        "financial_crisis": 12,      # Markets price in quickly
        "trade_war": 168,            # Trade wars last weeks
        "regulatory_shock": 48,      # Regulatory changes persist
        "natural_disasters": 24,     # Disasters are urgent
        "pandemic": 48,              # Health events persist
        "general": 12,               # Default
    }
    
    # ==========================================================================
    # NEW: CRITICAL EVENT COMBINATIONS (P3 Fix)
    # Certain keyword combinations indicate extreme risk
    # NOTE: These require ALL terms to appear in SAME headline to trigger
    # ==========================================================================
    CRITICAL_COMBINATIONS = {
        # NUCLEAR - only when imminent threat language
        ("nuclear", "attack", "imminent"): 10.0,
        ("nuclear", "strike", "threatens"): 9.0,
        ("tactical", "nuclear", "deploy"): 9.0,
        # NATO ARTICLE 5 - only if invoked
        ("nato", "article 5", "invoke"): 10.0,
        ("nato", "direct", "intervention"): 8.0,
        # IRAN-ISRAEL DIRECT CONFLICT
        ("iran", "strikes", "israel"): 8.0,
        ("israel", "strikes", "iran"): 8.0,
        # TAIWAN INVASION
        ("china", "invades", "taiwan"): 10.0,
        ("pla", "blockade", "taiwan"): 8.0,
        # ENERGY CHOKEPOINTS BLOCKED
        ("hormuz", "blocked", "closed"): 8.0,
        ("suez", "blocked", "crisis"): 7.0,
        # FINANCIAL CONTAGION
        ("bank", "run", "spreads"): 6.0,
        ("sovereign", "default", "contagion"): 6.0,
    }
    
    # ==========================================================================
    # NEW: REGIONAL CONTAGION MATRIX (P2 Fix)
    # Risk in one region spills over to others
    # ==========================================================================
    REGIONAL_CONTAGION = {
        "middle_east": {
            "europe": 0.25,      # Energy dependence
            "asia": 0.15,        # Shipping routes
            "americas": 0.10,    # Oil prices
        },
        "russia": {
            "europe": 0.40,      # Direct neighbor + energy
            "asia": 0.15,        # China alignment
            "americas": 0.10,    # Geopolitical impact
        },
        "asia": {
            "americas": 0.20,    # Trade dependence
            "europe": 0.10,      # Supply chains
        },
        "europe": {
            "americas": 0.15,    # Economic ties
            "asia": 0.10,        # Trade
        },
    }
    
    # ==========================================================================
    # NEW: STEEPER EXPOSURE ADJUSTMENT (P0 Fix)
    # More aggressive risk reduction at moderate levels
    # ==========================================================================
    RISK_TO_EXPOSURE = [
        (0.80, 1.00, "critical", 0.40),   # Only for genuine crises (nuclear, Article 5)
        (0.65, 0.80, "high", 0.60),       # Active military escalation with market impact
        (0.50, 0.65, "elevated", 0.75),   # Significant new developments
        (0.35, 0.50, "moderate", 0.85),   # Heightened activity but markets coping
        (0.20, 0.35, "guarded", 0.92),    # Background noise — barely affect exposure
        (0.00, 0.20, "low", 1.00),        # Full exposure
    ]
    
    # ==========================================================================
    # NEW: TAIL RISK TRIGGERS (P1 Fix)
    # Events that indicate black swan probability
    # ==========================================================================
    TAIL_RISK_TRIGGERS = {
        "nuclear_escalation": ["nuclear threat", "nuclear test", "nuclear capable", "tactical nuclear", "nuclear option"],
        "nato_direct": ["article 5", "nato troops", "nato direct involvement", "nato intervention"],
        "iran_israel_war": ["iran strikes israel", "israel strikes iran", "regional war middle east"],
        "taiwan_invasion": ["china invades taiwan", "taiwan strait crisis", "pla blockade"],
        "financial_contagion": ["lehman moment", "systemic collapse", "banking crisis spreads", "contagion"],
        "energy_crisis": ["strait of hormuz closed", "oil embargo", "energy blackout", "grid collapse"],
        "us_fiscal": ["debt ceiling breach", "us default", "government shutdown", "treasury default"],
        "supply_chain_collapse": ["suez canal blocked", "panama canal closed", "global shipping halt", "port shutdown"],
    }
    
    # Regions and their market indices
    REGIONAL_INDICES = {
        "middle_east": ["TASI.SR", "ADI.AD", "DFMGI.DU", "EGX30.CA"],  # Saudi, UAE, Egypt
        "asia": ["^N225", "^HSI", "000001.SS", "^KS11", "^TWII"],  # Japan, HK, China, Korea, Taiwan
        "europe": ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E"],  # UK, Germany, France, Euro
        "emerging": ["EEM", "^BVSP", "^BSESN", "^JKSE"],  # EM ETF, Brazil, India, Indonesia
    }
    
    # RSS feeds for geopolitical news
    # General world news
    RSS_FEEDS = {
        "reuters_world": "https://feeds.reuters.com/Reuters/worldNews",
        "bbc_world": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "ap_world": "https://rsshub.app/apnews/topics/world-news",
        "guardian_world": "https://www.theguardian.com/world/rss",
    }
    
    # TARGETED RSS FEEDS - Topic-specific for higher relevance
    TARGETED_RSS_FEEDS = {
        # Middle East specific
        "aljazeera_middleeast": "https://www.aljazeera.com/xml/rss/all.xml",
        "bbc_middleeast": "http://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
        
        # Asia specific  
        "bbc_asia": "http://feeds.bbci.co.uk/news/world/asia/rss.xml",
        
        # US Politics & Foreign Policy
        "bbc_uspolitics": "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
        
        # Business/Economy (for trade wars, sanctions)
        "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
        "bbc_business": "http://feeds.bbci.co.uk/news/business/rss.xml",
        
        # Defense & Military (specialized)
        "defense_news": "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
        
        # Europe (for NATO, EU decisions)
        "bbc_europe": "http://feeds.bbci.co.uk/news/world/europe/rss.xml",
        
        # Russia/Ukraine
        "guardian_russia": "https://www.theguardian.com/world/russia/rss",
        
        # CENTRAL BANKS & MONETARY POLICY
        "fed_news": "https://www.federalreserve.gov/feeds/press_all.xml",
        "ecb_press": "https://www.ecb.europa.eu/rss/press.html",
        "ft_central_banks": "https://www.ft.com/central-banks?format=rss",
        
        # ENERGY & COMMODITIES
        "reuters_energy": "https://feeds.reuters.com/reuters/energyNews",
        "oilprice_news": "https://oilprice.com/rss/main",
        
        # FINANCIAL MARKETS & ECONOMY
        "reuters_markets": "https://feeds.reuters.com/reuters/companyNews",
        "ft_markets": "https://www.ft.com/markets?format=rss",
        "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "bloomberg_markets": "https://feeds.bloomberg.com/markets/news.rss",
        
        # TRADE & TARIFFS
        "reuters_trade": "https://feeds.reuters.com/reuters/USTradingDesk",
        
        # SHIPPING & LOGISTICS
        "splash247": "https://splash247.com/feed/",
        "gcaptain": "https://gcaptain.com/feed/",
        "lloyds_list": "https://lloydslist.maritimeintelligence.informa.com/rss",
    }
    
    def __init__(self, cache_dir: str = "outputs/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "geopolitical_cache.json"
        self.filtered_cache_file = self.cache_dir / "filtered_events_cache.json"  # PERSISTENT FILTERED EVENTS
        
        # API keys
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        
        # Event cache
        self.events_cache: List[GeopoliticalEvent] = []
        self.last_assessment: Optional[GeopoliticalRiskAssessment] = None
        self.last_update: Optional[datetime] = None
        self.last_filtered_update: Optional[datetime] = None  # Track filtered events update time
        
        # Advanced relevance filter (rule-based, deterministic)
        self.relevance_filter = get_news_filter() if HAS_RELEVANCE_FILTER else None
        self.filtered_events: List[NewsEvent] = []  # High-quality filtered events
        
        # Story tracking for staleness decay
        self._story_first_seen: Dict[str, str] = {}  # topic_key -> ISO date first seen
        
        # Load caches (both raw and filtered)
        self._load_cache()
        self._load_filtered_cache()  # LOAD PERSISTENT FILTERED EVENTS
        
        logging.info(f"GeopoliticalIntelligence: Loaded {len(self.events_cache)} raw events, "
                    f"{len(self.filtered_events)} filtered events from cache")
    
    def _load_cache(self):
        """Load cached events from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Only load events from last 48 hours
                cutoff = datetime.now(pytz.UTC) - timedelta(hours=48)
                
                self.events_cache = []
                for event_data in data.get("events", []):
                    try:
                        ts = datetime.fromisoformat(event_data["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=pytz.UTC)
                        if ts > cutoff:
                            self.events_cache.append(GeopoliticalEvent(
                                event_id=event_data["event_id"],
                                timestamp=ts,
                                headline=event_data["headline"],
                                summary=event_data.get("summary", ""),
                                source=event_data["source"],
                                event_type=event_data["event_type"],
                                severity=event_data["severity"],
                                regions=event_data["regions"],
                                keywords=event_data.get("keywords", []),
                                market_impact_score=event_data["market_impact_score"],
                                url=event_data.get("url"),
                            ))
                    except Exception:
                        continue
                
                if data.get("last_update"):
                    self.last_update = datetime.fromisoformat(data["last_update"])
                
                self._story_first_seen = data.get("story_first_seen", {})
                    
        except Exception as e:
            logging.warning(f"Could not load geopolitical cache: {e}")
    
    def _save_cache(self):
        """Save events to disk cache."""
        try:
            data = {
                "events": [e.to_dict() for e in self.events_cache],
                "last_update": datetime.now(pytz.UTC).isoformat(),
                "story_first_seen": self._story_first_seen,
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save geopolitical cache: {e}")
    
    def _load_filtered_cache(self):
        """
        Load high-quality filtered events from persistent cache.
        This ensures filtered news survives page refreshes and server restarts.
        """
        try:
            if not self.filtered_cache_file.exists():
                logging.info("No filtered events cache file found")
                return
                
            with open(self.filtered_cache_file, 'r') as f:
                data = json.load(f)
            
            # Only load events from last 48 hours
            cutoff = datetime.now(pytz.UTC) - timedelta(hours=48)
            
            # Import NewsEvent and enums at module level is safer
            from src.data.news_relevance_filter import NewsEvent, NewsCategory, MarketDirection
            
            # Build lookup maps for enums (case-insensitive)
            category_map = {c.name.lower(): c for c in NewsCategory}
            category_map.update({c.value.lower(): c for c in NewsCategory})
            direction_map = {d.name.lower(): d for d in MarketDirection}
            direction_map.update({d.value.lower(): d for d in MarketDirection})
            
            self.filtered_events = []
            loaded_count = 0
            skipped_count = 0
            
            for event_data in data.get("events", []):
                try:
                    # Parse timestamp
                    ts_str = event_data.get("timestamp")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=pytz.UTC)
                    else:
                        ts = datetime.now(pytz.UTC)
                    
                    # Skip old events
                    if ts < cutoff:
                        skipped_count += 1
                        continue
                    
                    # Parse category (case-insensitive)
                    cat_str = str(event_data.get("category", "irrelevant")).lower()
                    category = category_map.get(cat_str, NewsCategory.IRRELEVANT)
                    
                    # Parse direction (case-insensitive)
                    dir_str = str(event_data.get("direction", "neutral")).lower()
                    direction = direction_map.get(dir_str, MarketDirection.NEUTRAL)
                    
                    event = NewsEvent(
                        event_id=event_data.get("event_id", ""),
                        timestamp=ts,
                        headline=event_data.get("headline", ""),
                        summary=event_data.get("summary", ""),
                        source=event_data.get("source", ""),
                        url=event_data.get("url"),
                        category=category,
                        tags=event_data.get("tags", []),
                        matched_keywords=event_data.get("matched_keywords", []),
                        relevance_score=float(event_data.get("relevance_score", 0.5)),
                        impact_score=float(event_data.get("impact_score", 0.5)),
                        credibility_score=float(event_data.get("credibility_score", 0.5)),
                        novelty_score=float(event_data.get("novelty_score", 0.5)),
                        final_score=float(event_data.get("final_score", 0.5)),
                        direction=direction,
                        direction_confidence=float(event_data.get("direction_confidence", 0.5)),
                        affected_assets=event_data.get("affected_assets", []),
                        affected_regions=event_data.get("affected_regions", []),
                        rationale=event_data.get("rationale", ""),  # Include rationale field
                    )
                    self.filtered_events.append(event)
                    loaded_count += 1
                    
                except Exception as e:
                    logging.debug(f"Could not restore filtered event: {e}")
                    continue
            
            if data.get("last_update"):
                try:
                    self.last_filtered_update = datetime.fromisoformat(
                        data["last_update"].replace('Z', '+00:00')
                    )
                except:
                    self.last_filtered_update = datetime.now(pytz.UTC)
            
            logging.info(f"Loaded {loaded_count} filtered events from cache "
                        f"(skipped {skipped_count} old events)")
                    
        except Exception as e:
            logging.warning(f"Could not load filtered events cache: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_filtered_cache(self):
        """
        Save high-quality filtered events to persistent cache.
        Called after every update to ensure no data loss.
        """
        try:
            events_data = []
            for event in self.filtered_events:
                try:
                    # Convert NewsEvent to dict
                    event_dict = event.to_dict() if hasattr(event, 'to_dict') else {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp),
                        "headline": event.headline,
                        "summary": event.summary,
                        "source": event.source,
                        "url": event.url,
                        "category": event.category.name if hasattr(event.category, 'name') else str(event.category),
                        "tags": event.tags,
                        "matched_keywords": event.matched_keywords,
                        "relevance_score": event.relevance_score,
                        "impact_score": event.impact_score,
                        "credibility_score": event.credibility_score,
                        "novelty_score": event.novelty_score,
                        "final_score": event.final_score,
                        "direction": event.direction.name if hasattr(event.direction, 'name') else str(event.direction),
                        "direction_confidence": event.direction_confidence,
                        "affected_assets": event.affected_assets,
                        "affected_regions": event.affected_regions,
                    }
                    events_data.append(event_dict)
                except Exception as e:
                    logging.debug(f"Could not serialize filtered event: {e}")
                    continue
            
            data = {
                "events": events_data,
                "last_update": datetime.now(pytz.UTC).isoformat(),
                "total_count": len(events_data),
            }
            
            with open(self.filtered_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"Saved {len(events_data)} filtered events to cache")
            
        except Exception as e:
            logging.warning(f"Could not save filtered events cache: {e}")
    
    def get_cached_filtered_events_age(self) -> Optional[float]:
        """Get the age of cached filtered events in minutes."""
        if self.last_filtered_update:
            age = datetime.now(pytz.UTC) - self.last_filtered_update
            return age.total_seconds() / 60
        return None
    
    def _generate_event_id(self, headline: str, source: str) -> str:
        """Generate unique event ID from headline and source."""
        content = f"{headline[:100]}_{source}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _classify_event(self, text: str) -> Tuple[str, float, List[str]]:
        """
        Classify event type and severity from text using SENTIMENT-AWARE classification.
        Returns: (event_type, severity, matched_keywords)
        
        IMPORTANT: 
        - Returns severity=0 for hard-discard events
        - Escalation events INCREASE severity
        - De-escalation events DECREASE severity (but still tracked)
        - Context-dependent keywords are analyzed with surrounding words
        """
        text_lower = " " + text.lower() + " "  # Add spaces for word boundary matching
        
        # STEP 1: Check hard discard FIRST - reject irrelevant content immediately
        for discard_kw in self.HARD_DISCARD_KEYWORDS:
            if discard_kw in text_lower:
                return "irrelevant", 0.0, []
        
        # STEP 1b: Check for BUSINESS CONTEXT - reduce score if this is corporate news
        business_context_count = sum(1 for bw in self.BUSINESS_CONTEXT_WORDS if bw in text_lower)
        is_business_context = business_context_count >= 2  # 2+ business words = likely corporate news
        
        # STEP 2: Classify as ESCALATION or DE-ESCALATION
        escalation_score = 0.0
        de_escalation_score = 0.0
        matched_keywords = []
        event_type = "general"
        
        # Short keywords that need word boundaries to avoid false matches
        # e.g., "war" shouldn't match "warns", "isis" shouldn't match "crisis"
        short_keywords = [
            "war", "fed", "ecb", "boj", "qe", "qt", "oil", "lng", "gdp", "cpi", "pmi", "ism",
            "isis", "isil", "hamas", "coup", "riot", "ban", "cut", "halt", "run",
            "eu", "uk", "us", "un", "imf", "wto", "who",
        ]
        
        def check_keyword(kw: str, text: str) -> bool:
            """Check if keyword exists with proper word boundaries for short words."""
            if kw in short_keywords or len(kw) <= 3:
                return (f" {kw} " in text or f" {kw}," in text or 
                        f" {kw}." in text or f" {kw}:" in text or
                        f" {kw})" in text or f"({kw} " in text)
            return kw in text
        
        # STEP 2a: Check ESCALATION keywords
        escalation_matches = {}
        for category, keywords in self.ESCALATION_KEYWORDS.items():
            matches = []
            for kw in keywords:
                if check_keyword(kw, text_lower):
                    matches.append(kw)
            if matches:
                escalation_matches[category] = matches
                escalation_score += len(matches) * 0.15  # Each escalation keyword adds 0.15
        
        # STEP 2b: Check DE-ESCALATION keywords
        de_escalation_matches = {}
        for category, keywords in self.DE_ESCALATION_KEYWORDS.items():
            matches = []
            for kw in keywords:
                if check_keyword(kw, text_lower):
                    matches.append(kw)
            if matches:
                de_escalation_matches[category] = matches
                de_escalation_score += len(matches) * 0.12  # De-escalation reduces risk
        
        # STEP 2c: Handle CONTEXT-DEPENDENT keywords
        for context_kw, (escalation_ctx, de_escalation_ctx) in self.CONTEXT_KEYWORDS.items():
            if check_keyword(context_kw, text_lower):
                # Check surrounding context (20 chars before/after)
                kw_pos = text_lower.find(context_kw)
                if kw_pos != -1:
                    context_window = text_lower[max(0, kw_pos-30):kw_pos+len(context_kw)+30]
                    
                    # Check if escalation context words are present
                    has_escalation = any(ctx in context_window for ctx in escalation_ctx)
                    has_de_escalation = any(ctx in context_window for ctx in de_escalation_ctx)
                    
                    if has_escalation and not has_de_escalation:
                        escalation_score += 0.2
                        matched_keywords.append(f"{context_kw} (escalation)")
                    elif has_de_escalation and not has_escalation:
                        de_escalation_score += 0.15
                        matched_keywords.append(f"{context_kw} (de-escalation)")
                    # If both or neither, treat as neutral (don't add to either score)
        
        # STEP 3: Determine event type from matched categories
        all_escalation_kws = [kw for kws in escalation_matches.values() for kw in kws]
        all_de_escalation_kws = [kw for kws in de_escalation_matches.values() for kw in kws]
        matched_keywords.extend(all_escalation_kws[:5])  # Limit to prevent huge lists
        matched_keywords.extend(all_de_escalation_kws[:3])
        
        # Map category to event type
        category_to_type = {
            "military_combat": "military",
            "diplomatic_breakdown": "diplomatic",
            "political_instability": "civil_unrest",
            "sanctions_imposed": "diplomatic",
            "energy_disruption": "energy",
            "terrorism": "military",
            "cyber_attacks": "infrastructure",
            "financial_crisis": "financial_stress",
            "trade_war": "economic",
            "natural_disasters": "infrastructure",
            "pandemic": "economic",
            "peace_process": "diplomatic",
            "diplomatic_progress": "diplomatic",
            "sanctions_relief": "diplomatic",
            "economic_recovery": "economic",
            "energy_stability": "energy",
            "security_improvement": "military",
            "health_improvement": "economic",
        }
        
        # Find the category with most matches
        max_category = None
        max_count = 0
        for cat, kws in {**escalation_matches, **de_escalation_matches}.items():
            if len(kws) > max_count:
                max_count = len(kws)
                max_category = cat
        
        if max_category:
            event_type = category_to_type.get(max_category, "general")
        
        # Also check legacy HIGH_IMPACT_KEYWORDS for economic/central bank news
        for etype, keywords in self.HIGH_IMPACT_KEYWORDS.items():
            matches = [kw for kw in keywords if check_keyword(kw, text_lower)]
            if len(matches) > max_count:
                event_type = etype
                matched_keywords.extend(matches[:3])
        
        # STEP 4: Calculate NET SEVERITY (escalation - de-escalation)
        net_risk = escalation_score - (de_escalation_score * 0.7)  # De-escalation dampens but doesn't fully negate
        
        # Check for severity modifiers
        severity_count = sum(1 for w in self.SEVERITY_MODIFIERS if w in text_lower)
        urgency_count = sum(1 for w in self.URGENCY_WORDS if w in text_lower)
        
        # Base severity calculation
        if escalation_score > 0 and de_escalation_score > 0:
            # Mixed signals - use net but with dampening
            base_severity = max(0.15, min(0.6, net_risk))
        elif escalation_score > 0:
            # Pure escalation - higher severity
            base_severity = min(0.9, 0.3 + net_risk)
        elif de_escalation_score > 0:
            # Pure de-escalation - lower severity but still track
            base_severity = max(0.15, 0.35 - de_escalation_score * 0.3)
        else:
            # No sentiment keywords matched, check legacy keywords
            if matched_keywords:
                base_severity = 0.35
            else:
                base_severity = 0.1
        
        # Add bonuses
        severity_bonus = min(0.2, severity_count * 0.06)
        urgency_bonus = min(0.15, urgency_count * 0.05)
        
        # Critical event types get base bonus
        critical_types = ["central_bank", "financial_stress", "military_combat"]
        if event_type in critical_types:
            base_severity += 0.1
        
        # CRITICAL INFRASTRUCTURE BOOST - Chokepoints, power grid, etc.
        critical_infra_keywords = [
            "strait of hormuz", "suez canal", "panama canal", "strait of malacca",
            "power grid", "grid down", "blackout", "banking system",
            "nuclear plant", "dam collapse", "dam breach",
        ]
        if any(cik in text_lower for cik in critical_infra_keywords):
            base_severity += 0.25  # Major boost for critical infrastructure
        
        # POLITICAL CRISIS BOOST
        political_crisis_keywords = [
            "coup", "government overthrown", "revolution", "civil war",
            "nationalization", "expropriation", "capital controls",
        ]
        if any(pck in text_lower for pck in political_crisis_keywords):
            base_severity += 0.15
        
        # Regional high-risk bonus (if mentions volatile regions)
        for region, region_kws in self.REGIONAL_KEYWORDS.items():
            if any(rkw in text_lower for rkw in region_kws):
                base_severity += 0.05  # Small regional risk bonus
                break
        
        severity = min(1.0, base_severity + severity_bonus + urgency_bonus)
        
        # STEP 5: Apply de-escalation cap
        # If predominantly de-escalation, cap severity at GUARDED level
        if de_escalation_score > escalation_score * 1.5:
            severity = min(severity, 0.35)  # Cap at GUARDED level
        
        # STEP 6: Apply BUSINESS CONTEXT dampening
        # If this appears to be corporate/business news, reduce geopolitical severity
        if is_business_context:
            severity *= 0.4  # Heavy dampening for business news
            if matched_keywords:
                matched_keywords.append("(business_context)")
        
        return event_type, severity, matched_keywords
    
    def _identify_regions(self, text: str) -> List[str]:
        """Identify affected regions from text."""
        text_lower = text.lower()
        regions = []
        
        region_keywords = {
            "middle_east": ["middle east", "saudi", "uae", "emirates", "bahrain", 
                           "qatar", "iran", "iraq", "israel", "gaza", "lebanon",
                           "syria", "jordan", "kuwait", "oman", "yemen", "gulf"],
            "asia": ["china", "japan", "korea", "taiwan", "hong kong", "india",
                    "indonesia", "singapore", "vietnam", "thailand", "asia",
                    "pacific", "philippines", "malaysia"],
            "europe": ["europe", "european", "uk", "britain", "germany", "france",
                      "italy", "spain", "nato", "eu", "brussels"],
            "americas": ["us", "united states", "america", "canada", "mexico",
                        "brazil", "latin america"],
            "russia": ["russia", "russian", "moscow", "ukraine", "putin"],
            "africa": ["africa", "african", "egypt", "south africa", "nigeria"],
        }
        
        for region, keywords in region_keywords.items():
            if any(kw in text_lower for kw in keywords):
                regions.append(region)
        
        return regions if regions else ["global"]
    
    def _is_market_relevant(self, text: str, event_type: str, severity: float,
                            regions: List[str]) -> bool:
        """
        Gate check: does this event have a plausible channel to move US equities?
        Filters out articles that match escalation keywords but aren't market-moving.
        
        An event is relevant if it involves at least one of:
        - Major economy trade/fiscal policy (US, China, EU, Japan)
        - Oil/energy supply disruption or pricing
        - Central bank action or financial system stress
        - Military action involving a major power or critical chokepoint
        - Sanctions affecting global trade
        - Commodity supply disruption
        """
        text_lower = text.lower()
        
        # Always relevant: involves a major economy directly
        major_economy_keywords = [
            "united states", "u.s.", "federal reserve", "fed ", "treasury",
            "china", "chinese", "beijing", "pboc",
            "european union", "ecb", "euro zone", "eurozone",
            "japan", "boj", "bank of japan",
            "uk ", "bank of england", "g7", "g20",
        ]
        if any(kw in text_lower for kw in major_economy_keywords):
            return True
        
        # Always relevant: oil/energy/commodity supply
        commodity_channels = [
            "oil price", "crude oil", "oil market", "opec", "oil export",
            "oil import", "natural gas", "lng ", "pipeline", "refiner",
            "barrel", "brent", "wti ", "gasoline", "diesel",
            "strait of hormuz", "suez canal", "panama canal",
            "shipping lane", "trade route", "supply chain",
            "commodity", "copper", "lithium", "rare earth", "semiconductor",
            "wheat", "grain export", "food price", "fertilizer",
        ]
        if any(kw in text_lower for kw in commodity_channels):
            return True
        
        # Always relevant: financial system / central bank / sanctions
        financial_channels = [
            "central bank", "interest rate", "rate cut", "rate hike",
            "inflation", "deflation", "recession", "gdp ",
            "sanctions", "embargo", "tariff", "trade war", "trade deal",
            "debt crisis", "sovereign debt", "bond yield", "credit rating",
            "banking crisis", "bank run", "default", "bailout",
            "currency", "dollar", "yuan", "yen ", "euro ",
            "stock market", "equity market", "bond market",
            "capital flight", "capital controls",
        ]
        if any(kw in text_lower for kw in financial_channels):
            return True
        
        # Always relevant: military involving major powers or nuclear
        major_military = [
            "nato", "nuclear", "us military", "us forces", "pentagon",
            "china military", "taiwan strait", "south china sea",
            "russia nato", "article 5", "missile defense",
            "carrier group", "aircraft carrier", "hypersonic",
            "cyber attack", "critical infrastructure",
        ]
        if any(kw in text_lower for kw in major_military):
            return True
        
        # High-severity events in market-connected regions pass through
        market_connected_regions = {"americas", "asia", "europe", "russia", "middle_east"}
        if severity >= 0.6 and any(r in market_connected_regions for r in regions):
            return True
        
        # High-impact event types always pass if severity is decent
        high_impact_types = {"central_bank", "financial_stress", "energy", "economic"}
        if event_type in high_impact_types and severity >= 0.35:
            return True
        
        # Everything else: not market-relevant
        return False
    
    def _calculate_market_impact(self, event_type: str, severity: float, 
                                  regions: List[str]) -> float:
        """Calculate estimated market impact score."""
        # Base impact by event type
        type_impact = {
            "military": 0.9,
            "diplomatic": 0.6,
            "economic": 0.7,
            "civil_unrest": 0.5,
            "infrastructure": 0.7,
            "general": 0.3,
            # NEW event types
            "central_bank": 0.85,      # Fed/ECB decisions are highly impactful
            "energy": 0.8,             # Oil/gas prices affect everything
            "financial_stress": 0.9,   # Banking crises are critical
            "shipping": 0.7,           # Supply chain disruptions
        }
        
        base_impact = type_impact.get(event_type, 0.3)
        
        # Adjust by regions (more regions = more global impact)
        region_multiplier = min(1.5, 1.0 + (len(regions) - 1) * 0.1)
        
        # High-impact regions for US markets
        high_impact_regions = ["middle_east", "asia", "russia", "americas"]
        if any(r in regions for r in high_impact_regions):
            region_multiplier *= 1.2
        
        impact = base_impact * severity * region_multiplier
        return min(1.0, impact)
    
    def fetch_rss_news(self, hours_back: int = 24) -> List[GeopoliticalEvent]:
        """Fetch news from RSS feeds."""
        events = []
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
        
        def fetch_feed(name: str, url: str) -> List[GeopoliticalEvent]:
            feed_events = []
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:  # Limit per feed
                    try:
                        # Parse timestamp
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            ts = datetime(*entry.published_parsed[:6], tzinfo=pytz.UTC)
                        else:
                            ts = datetime.now(pytz.UTC)
                        
                        if ts < cutoff:
                            continue
                        
                        headline = entry.title
                        summary = entry.get('summary', entry.get('description', ''))[:500]
                        
                        # Classify event
                        full_text = f"{headline} {summary}"
                        event_type, severity, keywords = self._classify_event(full_text)
                        
                        # Skip low-severity or irrelevant events
                        # Uses class-level threshold for consistency
                        if severity < self.MIN_SEVERITY_THRESHOLD or event_type == "irrelevant":
                            continue
                        
                        regions = self._identify_regions(full_text)
                        
                        if not self._is_market_relevant(full_text, event_type, severity, regions):
                            continue
                        
                        market_impact = self._calculate_market_impact(event_type, severity, regions)
                        
                        event = GeopoliticalEvent(
                            event_id=self._generate_event_id(headline, name),
                            timestamp=ts,
                            headline=headline,
                            summary=summary,
                            source=name,
                            event_type=event_type,
                            severity=severity,
                            regions=regions,
                            keywords=keywords,
                            market_impact_score=market_impact,
                            url=entry.get('link'),
                        )
                        feed_events.append(event)
                        
                    except Exception as e:
                        continue
                        
            except Exception as e:
                logging.warning(f"Failed to fetch RSS feed {name}: {e}")
            
            return feed_events
        
        # Combine general and targeted feeds
        all_feeds = {**self.RSS_FEEDS, **self.TARGETED_RSS_FEEDS}
        
        # Fetch all feeds in parallel (increased workers for more feeds)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fetch_feed, name, url): name 
                for name, url in all_feeds.items()
            }
            
            for future in as_completed(futures, timeout=45):
                try:
                    feed_events = future.result()
                    events.extend(feed_events)
                except Exception:
                    continue
        
        return events
    
    def fetch_newsapi(self, hours_back: int = 24) -> List[GeopoliticalEvent]:
        """Fetch news from NewsAPI (if API key available)."""
        if not self.newsapi_key:
            return []
        
        events = []
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
        
        # Queries for geopolitical events
        queries = [
            "military strike OR airstrike OR troops deploy",
            "sanctions OR embargo OR diplomatic crisis",
            "flights cancelled conflict OR airspace closed",
            "middle east tension OR gulf crisis",
            "trade war OR tariffs escalation",
        ]
        
        try:
            for query in queries[:3]:  # Limit API calls
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "apiKey": self.newsapi_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "from": cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get("articles", []):
                        try:
                            ts_str = article.get("publishedAt", "")
                            if ts_str:
                                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            else:
                                continue
                            
                            headline = article.get("title", "")
                            summary = article.get("description", "")[:500]
                            source = article.get("source", {}).get("name", "newsapi")
                            
                            full_text = f"{headline} {summary}"
                            event_type, severity, keywords = self._classify_event(full_text)
                            
                            if severity < 0.3 or event_type == "irrelevant":
                                continue
                            
                            regions = self._identify_regions(full_text)
                            
                            if not self._is_market_relevant(full_text, event_type, severity, regions):
                                continue
                            
                            market_impact = self._calculate_market_impact(event_type, severity, regions)
                            
                            event = GeopoliticalEvent(
                                event_id=self._generate_event_id(headline, source),
                                timestamp=ts,
                                headline=headline,
                                summary=summary,
                                source=source,
                                event_type=event_type,
                                severity=severity,
                                regions=regions,
                                keywords=keywords,
                                market_impact_score=market_impact,
                                url=article.get("url"),
                            )
                            events.append(event)
                            
                        except Exception:
                            continue
                            
        except Exception as e:
            logging.warning(f"NewsAPI fetch failed: {e}")
        
        return events
    
    def fetch_regional_market_data(self) -> Dict[str, Dict]:
        """
        Fetch regional market data to detect panic selling.
        Returns dict of region -> {change_pct, is_panic}
        """
        if not yf:
            return {}
        
        regional_data = {}
        
        for region, indices in self.REGIONAL_INDICES.items():
            try:
                # Get data for first available index in region
                for index in indices[:2]:  # Try first 2
                    try:
                        ticker = yf.Ticker(index)
                        hist = ticker.history(period="5d")
                        
                        if len(hist) >= 2:
                            current = hist['Close'].iloc[-1]
                            prev = hist['Close'].iloc[-2]
                            change_pct = ((current / prev) - 1) * 100
                            
                            # Detect panic (>2% single-day drop)
                            is_panic = change_pct < -2.0
                            
                            regional_data[region] = {
                                "index": index,
                                "change_pct": round(change_pct, 2),
                                "is_panic": is_panic,
                                "current": round(current, 2),
                            }
                            break
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                logging.warning(f"Could not get {region} market data: {e}")
        
        return regional_data
    
    def update_events(self, hours_back: int = 24) -> int:
        """
        Fetch new events from all sources and apply advanced relevance filtering.
        Returns number of new events added.
        """
        all_events = []
        raw_articles = []  # For relevance filter
        
        # Fetch from multiple sources
        logging.info("Fetching geopolitical events from RSS feeds...")
        rss_events = self.fetch_rss_news(hours_back)
        all_events.extend(rss_events)
        
        # Also collect raw articles for advanced filtering
        for event in rss_events:
            raw_articles.append({
                "headline": event.headline,
                "summary": event.summary,
                "source": event.source,
                "timestamp": event.timestamp,
                "url": event.url,
            })
        
        logging.info("Fetching geopolitical events from NewsAPI...")
        newsapi_events = self.fetch_newsapi(hours_back)
        all_events.extend(newsapi_events)
        
        for event in newsapi_events:
            raw_articles.append({
                "headline": event.headline,
                "summary": event.summary,
                "source": event.source,
                "timestamp": event.timestamp,
                "url": event.url,
            })
        
        # ============================================================
        # ADVANCED RELEVANCE FILTERING (Rule-based, deterministic)
        # ============================================================
        if self.relevance_filter and raw_articles:
            logging.info(f"Applying advanced relevance filter to {len(raw_articles)} articles...")
            
            # Filter articles
            filtered = self.relevance_filter.filter_batch(raw_articles)
            
            # Merge new filtered events with existing cached ones (deduplicate by event_id)
            existing_ids = {e.event_id for e in self.filtered_events}
            new_filtered = [e for e in filtered if e.event_id not in existing_ids]
            self.filtered_events = new_filtered + self.filtered_events
            
            # Keep only last 48 hours worth and limit to 500 events max
            cutoff = datetime.now(pytz.UTC) - timedelta(hours=48)
            self.filtered_events = [
                e for e in self.filtered_events 
                if e.timestamp and (
                    e.timestamp > cutoff if hasattr(e.timestamp, '__gt__') else True
                )
            ][:500]
            
            # Log filter stats
            stats = self.relevance_filter.get_stats()
            logging.info(f"Filter results: {stats['accepted']} accepted, "
                        f"{stats['rejected']} rejected "
                        f"({stats['acceptance_rate']*100:.1f}% acceptance rate)")
            
            # Save filtered events to BOTH locations (original timestamped + persistent cache)
            if filtered:
                self.relevance_filter.save_events(filtered)
            
            # CRITICAL: Save to persistent cache for survival across restarts
            self.last_filtered_update = datetime.now(pytz.UTC)
            self._save_filtered_cache()
            logging.info(f"Persistent filtered cache: {len(self.filtered_events)} total events")
        
        # Deduplicate by event_id
        existing_ids = {e.event_id for e in self.events_cache}
        new_events = [e for e in all_events if e.event_id not in existing_ids]
        
        # Add new events
        self.events_cache.extend(new_events)
        
        # Sort by timestamp (newest first)
        self.events_cache.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Keep only last 48 hours
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=48)
        self.events_cache = [e for e in self.events_cache 
                            if e.timestamp.replace(tzinfo=pytz.UTC) > cutoff]
        
        # Limit cache size
        self.events_cache = self.events_cache[:200]
        
        self.last_update = datetime.now(pytz.UTC)
        self._save_cache()
        
        logging.info(f"Geopolitical: Found {len(new_events)} new events, "
                    f"{len(self.events_cache)} total cached, "
                    f"{len(self.filtered_events)} high-quality filtered")
        
        return len(new_events)
    
    def get_filtered_events(self, auto_refresh_if_empty: bool = True, max_age_minutes: int = 60) -> List:
        """
        Get high-quality filtered events (from advanced filter).
        
        Args:
            auto_refresh_if_empty: If True, triggers refresh when no events available
            max_age_minutes: If events are older than this, trigger refresh
            
        Returns:
            List of NewsEvent objects
        """
        # Check if we need to refresh (empty or stale)
        should_refresh = False
        
        if not self.filtered_events and auto_refresh_if_empty:
            logging.info("No filtered events in cache, triggering refresh...")
            should_refresh = True
        elif self.last_filtered_update:
            age_minutes = self.get_cached_filtered_events_age()
            if age_minutes and age_minutes > max_age_minutes:
                logging.info(f"Filtered events are {age_minutes:.1f} min old (max: {max_age_minutes}), refreshing...")
                should_refresh = True
        
        if should_refresh:
            try:
                self.update_events(hours_back=24)
            except Exception as e:
                logging.warning(f"Auto-refresh failed: {e}")
        
        return self.filtered_events
    
    def get_filtered_events_summary(self) -> dict:
        """Get a summary of filtered events for UI display."""
        events = self.filtered_events
        
        # Group by category
        by_category = {}
        for event in events:
            cat = event.category.name if hasattr(event.category, 'name') else str(event.category)
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(event)
        
        return {
            "total_events": len(events),
            "categories": {cat: len(evts) for cat, evts in by_category.items()},
            "last_update": self.last_filtered_update.isoformat() if self.last_filtered_update else None,
            "cache_age_minutes": self.get_cached_filtered_events_age(),
        }
    
    def get_filter_stats(self) -> dict:
        """Get statistics from the relevance filter."""
        if self.relevance_filter:
            return self.relevance_filter.get_stats()
        return {}
    
    def _extract_story_keys(self, event) -> set:
        """Extract topic identifiers for an event for story-level matching."""
        text_lower = f"{event.headline} {event.summary}".lower()
        keys = set()
        
        # High-signal: known conflict IDs (e.g. "ukraine_russia", "israel_hamas")
        for conflict_id, conf in self.ACTIVE_CONFLICTS.items():
            if any(kw in text_lower for kw in conf.get("keywords", [])):
                keys.add(conflict_id)
        
        # Medium-signal: event's own classified keywords
        for kw in getattr(event, 'keywords', []):
            if kw and '(' not in kw and len(kw) > 2:
                keys.add(kw.lower().strip())
        
        # Medium-signal: significant headline words (3+ chars, not stopwords)
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                     'can', 'has', 'her', 'was', 'one', 'our', 'out', 'his',
                     'how', 'its', 'may', 'new', 'now', 'say', 'says', 'she',
                     'too', 'use', 'who', 'why', 'will', 'with', 'from', 'have',
                     'been', 'more', 'than', 'that', 'this', 'what', 'when',
                     'after', 'about', 'over', 'into', 'also', 'could', 'would',
                     'should', 'some', 'them', 'they', 'were', 'which', 'their'}
        headline_words = set()
        for word in event.headline.lower().split():
            clean = ''.join(c for c in word if c.isalnum())
            if len(clean) >= 4 and clean not in stopwords:
                headline_words.add(clean)
        keys.update(headline_words)
        
        if not keys:
            keys.add(f"_misc_{event.event_id[:8]}")
        
        return keys
    
    def _group_into_stories(self, events: list) -> list:
        """
        Group events about the same story and keep only the most severe per group.
        Prevents 10 articles about the same topic from inflating the score 10x.
        
        Key design choices to prevent snowballing:
        - Each event is matched against a story's SEED keys (not accumulated keys)
        - Requires >= 2 overlapping keys to merge (conflict ID alone is not enough)
        - Best match wins when multiple stories qualify
        """
        if len(events) <= 1:
            return events
        
        try:
            # (seed_keys, accumulated_keys, all_member_events, best_event)
            stories = []
            
            for event in events:
                event_keys = self._extract_story_keys(event)
                
                best_match_idx = -1
                best_match_score = 0
                
                for i, (seed_keys, _, _, _) in enumerate(stories):
                    overlap = event_keys & seed_keys
                    
                    if len(overlap) >= 2 and len(overlap) > best_match_score:
                        best_match_score = len(overlap)
                        best_match_idx = i
                
                if best_match_idx >= 0:
                    seed, acc, members, best = stories[best_match_idx]
                    acc.update(event_keys)
                    members.append(event)
                    if event.market_impact_score > best.market_impact_score:
                        stories[best_match_idx] = (seed, acc, members, event)
                else:
                    stories.append((event_keys.copy(), event_keys.copy(), [event], event))
            
            deduped = [best for _, _, _, best in stories]
            
            if events and not deduped:
                return events
            
            if len(deduped) < len(events):
                logging.info(f"Story grouping: {len(events)} events -> {len(deduped)} stories")
            
            return deduped
            
        except Exception as e:
            logging.warning(f"Story grouping failed, using raw events: {e}")
            return events
    
    def _get_staleness_multiplier(self, event) -> float:
        """
        Returns a decay multiplier (0.15-1.0) based on how long a story topic
        has been in the news. New stories get full weight; old stories get discounted.
        """
        try:
            text_lower = f"{event.headline} {event.summary}".lower()
            topic_key = None
            has_escalation_trigger = False
            
            # Identify topic and check for escalation triggers
            for conflict_id, conf in self.ACTIVE_CONFLICTS.items():
                if any(kw in text_lower for kw in conf.get("keywords", [])):
                    topic_key = conflict_id
                    if any(trigger in text_lower for trigger in conf.get("escalation_triggers", [])):
                        has_escalation_trigger = True
                    break
            
            if not topic_key:
                topic_key = f"{getattr(event, 'event_type', 'general')}_{'-'.join(sorted(getattr(event, 'regions', ['global'])))}"
            
            today = datetime.now(pytz.UTC).date().isoformat()
            
            # New escalation trigger resets the clock
            if has_escalation_trigger:
                self._story_first_seen[topic_key] = today
                return 1.0
            
            if topic_key not in self._story_first_seen:
                self._story_first_seen[topic_key] = today
                return 1.0
            
            first_seen = datetime.fromisoformat(self._story_first_seen[topic_key]).date()
            days_active = (datetime.now(pytz.UTC).date() - first_seen).days
            
            if days_active <= 3:
                return 1.0
            elif days_active <= 7:
                return 0.6
            elif days_active <= 14:
                return 0.3
            else:
                return 0.15
                
        except Exception:
            return 1.0
    
    def _log_prediction(self, overall_risk: float, risk_level: str,
                        vix_level: float, spy_change: float,
                        escalation_count: int, de_escalation_count: int):
        """
        Log daily geo risk prediction alongside market data for future calibration.
        Only logs once per day.
        """
        try:
            log_file = Path("outputs/geo_prediction_log.json")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            entries = []
            if log_file.exists():
                with open(log_file, 'r') as f:
                    entries = json.load(f)
            
            today = datetime.now(pytz.UTC).date().isoformat()
            
            # Only log once per day
            if entries and entries[-1].get("date") == today:
                return
            
            entries.append({
                "date": today,
                "overall_risk_score": round(overall_risk, 4),
                "risk_level": risk_level,
                "spy_change_pct": round(spy_change, 4),
                "vix": round(vix_level, 2),
                "escalation_count": escalation_count,
                "de_escalation_count": de_escalation_count,
            })
            
            # Keep rolling year
            entries = entries[-365:]
            
            with open(log_file, 'w') as f:
                json.dump(entries, f, indent=2)
                
        except Exception as e:
            logging.debug(f"Could not log geo prediction: {e}")
    
    def get_risk_assessment(self, refresh: bool = False) -> GeopoliticalRiskAssessment:
        """
        COMPREHENSIVE Geopolitical Risk Assessment v2.0
        
        MAJOR IMPROVEMENTS:
        1. Failed negotiations = ESCALATION (not de-escalation)
        2. Structural risk floor for active conflicts
        3. Steeper exposure adjustment curve
        4. Event-type-specific time decay
        5. Concrete de-escalation requirement
        6. Tail risk indicator
        7. Keyword severity weighting
        8. VIX integration
        9. Regional contagion modeling
        10. Multi-timeframe risk (immediate/situational/structural)
        """
        # Update events if needed
        if refresh or not self.last_update or \
           (datetime.now(pytz.UTC) - self.last_update) > timedelta(minutes=30):
            self.update_events()
        
        now = datetime.now(pytz.UTC)
        
        # Get events from different time windows
        cutoff_6h = now - timedelta(hours=6)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_48h = now - timedelta(hours=48)
        
        events_6h = [e for e in self.events_cache if e.timestamp.replace(tzinfo=pytz.UTC) > cutoff_6h]
        events_24h = [e for e in self.events_cache if e.timestamp.replace(tzinfo=pytz.UTC) > cutoff_24h]
        events_48h = [e for e in self.events_cache if e.timestamp.replace(tzinfo=pytz.UTC) > cutoff_48h]
        
        # ================================================================
        # STEP 1: CLASSIFY EVENTS WITH ENHANCED LOGIC
        # Key fix: Failed negotiations = ESCALATION
        # ================================================================
        escalation_events = []
        de_escalation_events = []
        neutral_events = []
        
        for event in events_24h:
            text = f"{event.headline} {event.summary}"
            text_lower = text.lower()
            
            # FIX P0: Check for FAILED NEGOTIATION patterns FIRST
            has_failure_pattern = any(fp in text_lower for fp in self.NEGOTIATION_FAILURE_PATTERNS)
            
            # Count escalation vs de-escalation with SEVERITY WEIGHTING
            escalation_score = 0.0
            de_escalation_score = 0.0
            
            for category, keywords in self.ESCALATION_KEYWORDS.items():
                for kw in keywords:
                    if kw in text_lower:
                        # Apply severity weight if available
                        weight = self.KEYWORD_SEVERITY_WEIGHTS.get(kw, 1.0)
                        escalation_score += weight
            
            for category, keywords in self.DE_ESCALATION_KEYWORDS.items():
                for kw in keywords:
                    if kw in text_lower:
                        de_escalation_score += 1.0
            
            # FIX P0: If has failure pattern, convert de-escalation to escalation
            if has_failure_pattern and de_escalation_score > 0:
                # "Peace talks without breakthrough" = ESCALATION
                escalation_score += de_escalation_score * 1.5  # Boost escalation
                de_escalation_score = 0  # Cancel de-escalation
            
            # FIX P1: Only count de-escalation if CONCRETE outcome present
            has_concrete_de_escalation = any(
                cde in text_lower for cde in self.CONCRETE_DE_ESCALATION_OUTCOMES
            )
            if not has_concrete_de_escalation:
                de_escalation_score *= 0.3  # Heavily discount non-concrete de-escalation
            
            # Classify based on net sentiment
            if escalation_score > de_escalation_score * 1.5:
                escalation_events.append(event)
            elif de_escalation_score > escalation_score * 1.5 and has_concrete_de_escalation:
                de_escalation_events.append(event)
            else:
                neutral_events.append(event)
        
        # ================================================================
        # STEP 1b: STORY-LEVEL DEDUPLICATION
        # Collapse multiple articles about the same topic into one per story
        # ================================================================
        escalation_events = self._group_into_stories(escalation_events)
        de_escalation_events = self._group_into_stories(de_escalation_events)
        
        # ================================================================
        # STEP 2: CHECK FOR CRITICAL COMBINATIONS (P3 Fix)
        # NOTE: Combinations must appear in SAME headline to be meaningful
        # ================================================================
        combination_boost = 0.0
        combination_triggers = []
        
        for event in events_24h:
            event_text = f"{event.headline} {event.summary}".lower()
            for combo, boost in self.CRITICAL_COMBINATIONS.items():
                if all(term in event_text for term in combo):
                    combination_boost += boost * 0.015  # Each combo adds max 0.15
                    trigger_name = " + ".join(combo)
                    if trigger_name not in combination_triggers:
                        combination_triggers.append(trigger_name)
        
        # Cap combination boost at 0.25 total
        combination_boost = min(0.25, combination_boost)
        
        # ================================================================
        # STEP 3: IDENTIFY ACTIVE CONFLICTS (P0 Fix - Structural Risk Floor)
        # ================================================================
        active_conflict_ids = []
        structural_risk_floor = 0.03  # Base minimum — near zero when nothing is happening
        conflict_escalation_boost = 0.0
        
        for conflict_id, config in self.ACTIVE_CONFLICTS.items():
            # Check if any events mention this conflict
            conflict_keywords = config.get("keywords", [])
            has_conflict_mention = any(
                any(kw in f"{e.headline} {e.summary}".lower() for kw in conflict_keywords)
                for e in events_48h
            )
            
            if has_conflict_mention:
                active_conflict_ids.append(conflict_id)
                # Apply structural risk floor
                structural_risk_floor = max(structural_risk_floor, config["min_risk_floor"])
                
                # Check for escalation triggers within this conflict
                for event in events_48h:
                    event_text = f"{event.headline} {event.summary}".lower()
                    for trigger in config.get("escalation_triggers", []):
                        if trigger in event_text:
                            conflict_escalation_boost += 0.10  # Major escalation trigger
                            if f"{conflict_id}: {trigger}" not in combination_triggers:
                                combination_triggers.append(f"{conflict_id}: {trigger}")
        
        # Cap conflict escalation boost
        combination_boost += min(0.20, conflict_escalation_boost)
        
        # ================================================================
        # STEP 4: CALCULATE REGIONAL RISKS WITH CONTAGION (P2 Fix)
        # ================================================================
        regional_risks = {}
        regional_escalation = {}
        regional_de_escalation = {}
        
        BASELINE_EVENTS_PER_REGION = {
            "americas": 15, "europe": 10, "asia": 8,
            "middle_east": 5, "russia": 5, "africa": 4, "global": 10,
        }
        DEFAULT_BASELINE = 5
        
        # Count escalation events per region with severity weighting
        for event in escalation_events:
            if event.market_impact_score > 0.3:
                for region in event.regions:
                    # Weight by severity
                    weight = 1 + event.severity
                    regional_escalation[region] = regional_escalation.get(region, 0) + weight
        
        # Count de-escalation events per region
        for event in de_escalation_events:
            if event.market_impact_score > 0.25:
                for region in event.regions:
                    regional_de_escalation[region] = regional_de_escalation.get(region, 0) + 1
        
        # Calculate initial regional risks
        all_regions = set(list(regional_escalation.keys()) + list(regional_de_escalation.keys()))
        
        for region in all_regions:
            esc_count = regional_escalation.get(region, 0)
            de_esc_count = regional_de_escalation.get(region, 0)
            baseline = BASELINE_EVENTS_PER_REGION.get(region, DEFAULT_BASELINE)
            
            net_escalation = max(0, esc_count - de_esc_count * 0.3)  # Reduced de-esc dampening
            
            if net_escalation <= baseline:
                regional_risk = 0.15 + (net_escalation / baseline) * 0.25
            elif net_escalation <= baseline * 2:
                excess = (net_escalation - baseline) / baseline
                regional_risk = 0.4 + excess * 0.2
            elif net_escalation <= baseline * 3:
                excess = (net_escalation - baseline * 2) / baseline
                regional_risk = 0.6 + excess * 0.15
            else:
                regional_risk = 0.75 + min(0.1, (net_escalation - baseline * 3) * 0.015)
            
            regional_risks[region] = round(min(0.70, regional_risk), 2)  # Cap at 70% before contagion
        
        # Apply CONTAGION from one region to others (P2 Fix)
        contagion_applied = {}
        for source_region, targets in self.REGIONAL_CONTAGION.items():
            source_risk = regional_risks.get(source_region, 0)
            if source_risk > 0.3:  # Only spread if source has significant risk
                for target_region, spillover_factor in targets.items():
                    spillover = source_risk * spillover_factor
                    if target_region not in regional_risks:
                        regional_risks[target_region] = 0.15
                    contagion_applied[target_region] = contagion_applied.get(target_region, 0) + spillover
        
        # Add contagion to regional risks (capped at 50% after contagion)
        for region, spillover in contagion_applied.items():
            regional_risks[region] = min(0.50, regional_risks.get(region, 0.0) + spillover)
        
        # Final cap on all regional risks at 60%
        for region in regional_risks:
            regional_risks[region] = min(0.60, regional_risks[region])
        
        # ================================================================
        # STEP 5: APPLY STRUCTURAL RISK FLOOR FOR ACTIVE CONFLICTS
        # ================================================================
        for conflict_id in active_conflict_ids:
            config = self.ACTIVE_CONFLICTS.get(conflict_id, {})
            for region in config.get("affected_regions", []):
                if region in regional_risks:
                    regional_risks[region] = max(regional_risks[region], config["min_risk_floor"])
                else:
                    regional_risks[region] = config["min_risk_floor"]
        
        # ================================================================
        # STEP 6: CALCULATE MULTI-TIMEFRAME RISK (P2 Fix)
        # ================================================================
        
        # IMMEDIATE RISK (last 6 hours) - for intraday trading
        # Uses blend of max and average to avoid single-event spikes
        immediate_scores = []
        for event in events_6h:
            if event in escalation_events:
                age_hours = (now - event.timestamp.replace(tzinfo=pytz.UTC)).total_seconds() / 3600
                decay = 2 ** (-age_hours / 3)  # 3-hour half-life for immediate
                staleness = self._get_staleness_multiplier(event)
                immediate_scores.append(event.market_impact_score * decay * staleness * 0.7)
        
        if immediate_scores:
            max_imm = max(immediate_scores)
            avg_imm = sum(immediate_scores) / len(immediate_scores)
            immediate_risk = 0.5 * max_imm + 0.5 * avg_imm
        else:
            immediate_risk = 0.02  # No events = near-zero, not 10%
        
        # SITUATIONAL RISK (last 48 hours) - for swing trading
        situational_scores = []
        for event in events_48h:
            if event in escalation_events:
                age_hours = (now - event.timestamp.replace(tzinfo=pytz.UTC)).total_seconds() / 3600
                half_life = self.TIME_DECAY_HALF_LIFE.get(event.event_type, 12)
                decay = 2 ** (-age_hours / half_life)
                staleness = self._get_staleness_multiplier(event)
                situational_scores.append(event.market_impact_score * event.severity * decay * staleness * 0.6)
        
        if situational_scores:
            max_sit = max(situational_scores)
            avg_sit = sum(situational_scores) / len(situational_scores)
            situational_risk = 0.4 * max_sit + 0.6 * avg_sit
        else:
            situational_risk = 0.03  # No events = near-zero, not 15%
        
        # STRUCTURAL RISK - reflects PRICED-IN background, kept low
        # Only spikes if there are actual escalation triggers within those conflicts
        structural_risk = structural_risk_floor
        if conflict_escalation_boost > 0:
            structural_risk = min(0.50, structural_risk + conflict_escalation_boost)
        elif len(active_conflict_ids) >= 3:
            structural_risk = min(0.15, structural_risk + 0.02)  # Tiny bump for many conflicts, not 45%
        
        # ================================================================
        # STEP 7: CALCULATE TAIL RISK (P1 Fix)
        # Tail risk = probability of catastrophic escalation
        # Requires specific warning language, not just general mentions
        # ================================================================
        tail_risk_score = 0.0
        tail_risk_triggers = []
        
        # Check each headline individually for tail risk triggers
        for event in events_24h:
            event_text = f"{event.headline} {event.summary}".lower()
            for trigger_name, trigger_keywords in self.TAIL_RISK_TRIGGERS.items():
                for kw in trigger_keywords:
                    if kw in event_text:
                        # Only count each trigger category once
                        if trigger_name not in tail_risk_triggers:
                            tail_risk_score += 0.12  # Each trigger category adds 12%
                            tail_risk_triggers.append(trigger_name)
                        break  # Don't double-count same trigger in same event
        
        # Boost tail risk if multiple active conflicts detected
        if len(active_conflict_ids) >= 3:
            tail_risk_score += 0.10  # 3+ conflicts = 10% additional tail risk
        
        tail_risk_score = min(0.75, tail_risk_score)  # Cap at 75%
        
        # ================================================================
        # STEP 8: VIX INTEGRATION (P2 Fix)
        # ================================================================
        vix_level = 20.0  # Default
        vix_multiplier = 1.0
        market_confirms_risk = False
        spy_change = 0.0
        
        try:
            from src.data.macro_data import MacroDataLoader
            macro = MacroDataLoader()
            indicators = macro.fetch_all()
            vix_level = indicators.vix or 20.0
            spy_change = getattr(indicators, 'spy_change_pct', 0) or 0
            
            # Market confirmation: only amplify when market ACTUALLY confirms risk
            if vix_level > 30:
                vix_multiplier = 1.20
                market_confirms_risk = True
            elif vix_level > 25:
                vix_multiplier = 1.10
                market_confirms_risk = True
            elif vix_level > 20:
                vix_multiplier = 1.0
            elif vix_level > 16:
                vix_multiplier = 0.75  # Market calm — geo risk is overblown
            else:
                vix_multiplier = 0.60  # Market very calm — heavy discount
            
            # Market adaptation: if SPY is positive, markets are ignoring geo risk
            if spy_change > 0.5 and vix_level < 22:
                vix_multiplier *= 0.70  # Additional 30% discount when market rallying
                logging.info(f"Geo risk market adaptation: SPY +{spy_change:.2f}%, VIX {vix_level:.1f} — discounting geo risk")
            elif spy_change > 0 and vix_level < 20:
                vix_multiplier *= 0.85  # 15% discount when market mildly positive
        except Exception as e:
            logging.debug(f"Could not get VIX: {e}")
        
        # ================================================================
        # STEP 9: CALCULATE OVERALL RISK
        # ================================================================
        
        # Blend multi-timeframe risks
        base_risk = (
            0.30 * immediate_risk +      # 30% weight to immediate
            0.40 * situational_risk +    # 40% weight to situational
            0.30 * structural_risk       # 30% weight to structural
        )
        
        # Add regional contribution — reduced weight since regional scores inflate
        # from routine news coverage, not actual market-moving events
        if regional_risks:
            max_regional = max(regional_risks.values())
            avg_regional = sum(regional_risks.values()) / len(regional_risks)
            regional_contribution = 0.4 * max_regional + 0.6 * avg_regional
            base_risk = 0.75 * base_risk + 0.25 * regional_contribution
        
        # Add combination boost
        base_risk += combination_boost
        
        # Apply VIX multiplier
        overall_risk = base_risk * vix_multiplier
        
        # Ensure minimum from structural floor
        overall_risk = max(overall_risk, structural_risk_floor)
        
        # If tail risk is high, boost overall
        if tail_risk_score > 0.3:
            overall_risk = max(overall_risk, 0.4 + tail_risk_score * 0.3)
        
        # Cap at 95%
        overall_risk = min(0.95, overall_risk)
        
        # Get market data for panic detection
        market_data = self.fetch_regional_market_data()
        market_panic_count = sum(1 for d in market_data.values() if d.get("is_panic"))
        
        if market_panic_count >= 2:
            overall_risk = min(0.95, overall_risk + 0.15)
            market_confirms_risk = True
        
        # ================================================================
        # STEP 10: DETERMINE RISK LEVEL AND EXPOSURE (P0 Fix - Steeper Curve)
        # ================================================================
        risk_level = "low"
        exposure_adj = 1.0
        
        for min_risk, max_risk, level, exposure in self.RISK_TO_EXPOSURE:
            if min_risk <= overall_risk < max_risk:
                risk_level = level
                exposure_adj = exposure
                break
        
        if overall_risk >= 0.80:
            risk_level = "critical"
            exposure_adj = 0.40
        
        # ================================================================
        # STEP 11: GRADUATED SAFE HAVEN SIGNAL (P1 Fix)
        # ================================================================
        if overall_risk >= 0.60 or tail_risk_score >= 0.50:
            safe_haven_level = "strong"
            safe_haven_signal = True
        elif overall_risk >= 0.45 or tail_risk_score >= 0.30:
            safe_haven_level = "moderate"
            safe_haven_signal = True
        elif overall_risk >= 0.35 and (tail_risk_score >= 0.15 or len(active_conflict_ids) >= 2):
            safe_haven_level = "hedging"
            safe_haven_signal = False
        else:
            safe_haven_level = "none"
            safe_haven_signal = False
        
        # Override for specific triggers
        if any(t in tail_risk_triggers for t in ["nuclear_escalation", "nato_direct"]):
            safe_haven_level = "strong"
            safe_haven_signal = True
        
        # ================================================================
        # STEP 12: EXTRACT KEY CONCERNS
        # ================================================================
        key_concerns = []
        
        # Add tail risk triggers first
        for trigger in tail_risk_triggers[:3]:
            key_concerns.append(f"⚠️ TAIL RISK: {trigger.replace('_', ' ').title()}")
        
        # Add critical combinations
        for combo in combination_triggers[:2]:
            key_concerns.append(f"🔥 CRITICAL: {combo}")
        
        # Add top escalation events
        significant_events = [e for e in escalation_events if e.market_impact_score > 0.35]
        for event in sorted(significant_events, 
                           key=lambda x: x.market_impact_score * x.severity, 
                           reverse=True)[:5]:
            if len(key_concerns) < 8:
                key_concerns.append(f"{event.event_type.upper()}: {event.headline[:70]}...")
        
        # ================================================================
        # BUILD AND RETURN ASSESSMENT
        # ================================================================
        self.last_assessment = GeopoliticalRiskAssessment(
            timestamp=now,
            overall_risk_score=round(overall_risk, 3),
            risk_level=risk_level,
            active_events=significant_events[:20],
            regional_risks=regional_risks,
            recommended_exposure_adjustment=exposure_adj,
            key_concerns=key_concerns,
            safe_haven_signal=safe_haven_signal,
            # NEW fields
            immediate_risk=round(immediate_risk, 3),
            situational_risk=round(situational_risk, 3),
            structural_risk=round(structural_risk, 3),
            tail_risk_score=round(tail_risk_score, 3),
            tail_risk_triggers=tail_risk_triggers,
            active_conflicts=active_conflict_ids,
            safe_haven_level=safe_haven_level,
            market_confirms_risk=market_confirms_risk,
            vix_level=vix_level,
        )
        
        # Log daily prediction for calibration and persist story tracking
        self._log_prediction(
            overall_risk=overall_risk,
            risk_level=risk_level,
            vix_level=vix_level,
            spy_change=spy_change,
            escalation_count=len(escalation_events),
            de_escalation_count=len(de_escalation_events),
        )
        self._save_cache()
        
        return self.last_assessment
    
    def get_context_for_llm(self) -> str:
        """
        Generate comprehensive context string for LLM debate/reasoning.
        Now includes multi-timeframe risk, tail risk, and active conflicts.
        """
        assessment = self.get_risk_assessment()
        
        context_parts = [
            f"## Geopolitical Risk Assessment v2.0",
            f"Overall Risk: {assessment.risk_level.upper()} ({assessment.overall_risk_score:.0%})",
            f"Recommended Exposure: {assessment.recommended_exposure_adjustment:.0%}",
            f"",
            f"### Multi-Timeframe Analysis:",
            f"  - Immediate Risk (6h): {assessment.immediate_risk:.0%}",
            f"  - Situational Risk (48h): {assessment.situational_risk:.0%}",
            f"  - Structural Risk (ongoing): {assessment.structural_risk:.0%}",
        ]
        
        # Tail risk section
        if assessment.tail_risk_score > 0.1:
            context_parts.append(f"\n### ⚠️ TAIL RISK: {assessment.tail_risk_score:.0%}")
            if assessment.tail_risk_triggers:
                for trigger in assessment.tail_risk_triggers:
                    context_parts.append(f"  - {trigger.replace('_', ' ').title()}")
        
        # Active conflicts
        if assessment.active_conflicts:
            context_parts.append(f"\n### Active Conflicts Being Monitored:")
            for conflict in assessment.active_conflicts:
                config = self.ACTIVE_CONFLICTS.get(conflict, {})
                floor = config.get("min_risk_floor", 0)
                context_parts.append(f"  - {conflict.replace('_', ' ').title()} (min floor: {floor:.0%})")
        
        # Safe haven signal
        context_parts.append(f"\n### Safe Haven Signal: {assessment.safe_haven_level.upper()}")
        if assessment.safe_haven_level == "strong":
            context_parts.append("  → ROTATE to gold, treasuries, defensive sectors")
        elif assessment.safe_haven_level == "moderate":
            context_parts.append("  → Add 10-15% allocation to safe havens")
        elif assessment.safe_haven_level == "hedging":
            context_parts.append("  → Consider put protection or VIX calls")
        
        # Market validation
        if assessment.market_confirms_risk:
            context_parts.append(f"\n### Market Validation: CONFIRMED")
            context_parts.append(f"  VIX: {assessment.vix_level:.1f}")
        
        if assessment.regional_risks:
            context_parts.append("\n### Regional Risks:")
            for region, risk in sorted(assessment.regional_risks.items(), 
                                       key=lambda x: x[1], reverse=True):
                context_parts.append(f"  - {region}: {risk:.0%}")
        
        if assessment.key_concerns:
            context_parts.append("\n### Key Concerns:")
            for concern in assessment.key_concerns:
                context_parts.append(f"  - {concern}")
        
        if assessment.active_events:
            context_parts.append(f"\n### Recent Events ({len(assessment.active_events)}):")
            for event in assessment.active_events[:5]:
                age_hours = (datetime.now(pytz.UTC) - 
                            event.timestamp.replace(tzinfo=pytz.UTC)).total_seconds() / 3600
                context_parts.append(
                    f"  - [{event.event_type}] {event.headline[:60]}... "
                    f"(severity: {event.severity:.0%}, {age_hours:.0f}h ago)"
                )
        
        return "\n".join(context_parts)
    
    def get_exposure_multiplier(self) -> float:
        """Get recommended exposure multiplier based on geopolitical risk."""
        assessment = self.get_risk_assessment()
        return assessment.recommended_exposure_adjustment
    
    def analyze_headline(self, headline: str) -> dict:
        """
        Analyze a single headline for debugging/transparency.
        Returns detailed classification including:
        - Event type
        - Severity
        - Escalation vs De-escalation breakdown
        - Matched keywords
        - Explanation
        """
        text_lower = " " + headline.lower() + " "
        
        # Check hard discard
        for discard_kw in self.HARD_DISCARD_KEYWORDS:
            if discard_kw in text_lower:
                return {
                    "headline": headline,
                    "classification": "DISCARDED",
                    "reason": f"Matched discard keyword: '{discard_kw}'",
                    "severity": 0,
                    "is_market_relevant": False,
                }
        
        # Count escalation keywords
        escalation_matches = []
        for category, keywords in self.ESCALATION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    escalation_matches.append(f"{category}:{kw}")
        
        # Count de-escalation keywords
        de_escalation_matches = []
        for category, keywords in self.DE_ESCALATION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    de_escalation_matches.append(f"{category}:{kw}")
        
        # Get full classification
        event_type, severity, keywords = self._classify_event(headline)
        
        # Determine net sentiment
        if len(escalation_matches) > len(de_escalation_matches) * 1.2:
            sentiment = "ESCALATION"
        elif len(de_escalation_matches) > len(escalation_matches) * 1.2:
            sentiment = "DE-ESCALATION"
        else:
            sentiment = "NEUTRAL/MIXED"
        
        # Build explanation
        explanation = []
        if escalation_matches:
            explanation.append(f"Escalation signals: {len(escalation_matches)}")
        if de_escalation_matches:
            explanation.append(f"De-escalation signals: {len(de_escalation_matches)}")
        if severity < 0.25:
            explanation.append("Low severity - minimal market impact expected")
        elif severity >= 0.75:
            explanation.append("High severity - significant market impact possible")
        
        return {
            "headline": headline,
            "event_type": event_type,
            "severity": round(severity, 3),
            "sentiment": sentiment,
            "escalation_keywords": escalation_matches[:5],  # Limit output
            "de_escalation_keywords": de_escalation_matches[:5],
            "matched_keywords": keywords[:5],
            "is_market_relevant": severity >= self.MIN_SEVERITY_THRESHOLD,
            "explanation": " | ".join(explanation) if explanation else "Standard news",
        }
    
    def get_sentiment_summary(self) -> dict:
        """
        Get a summary of current escalation vs de-escalation balance.
        Useful for UI display and debugging.
        """
        now = datetime.now(pytz.UTC)
        cutoff_24h = now - timedelta(hours=24)
        recent_events = [e for e in self.events_cache 
                        if e.timestamp.replace(tzinfo=pytz.UTC) > cutoff_24h]
        
        escalation_count = 0
        de_escalation_count = 0
        neutral_count = 0
        
        for event in recent_events:
            analysis = self.analyze_headline(event.headline)
            if analysis["sentiment"] == "ESCALATION":
                escalation_count += 1
            elif analysis["sentiment"] == "DE-ESCALATION":
                de_escalation_count += 1
            else:
                neutral_count += 1
        
        total = max(1, escalation_count + de_escalation_count + neutral_count)
        
        return {
            "total_events_24h": len(recent_events),
            "escalation_events": escalation_count,
            "de_escalation_events": de_escalation_count,
            "neutral_events": neutral_count,
            "escalation_ratio": round(escalation_count / total, 2),
            "de_escalation_ratio": round(de_escalation_count / total, 2),
            "net_sentiment": "RISK-ON" if de_escalation_count > escalation_count * 1.2 else (
                "RISK-OFF" if escalation_count > de_escalation_count * 1.2 else "NEUTRAL"
            ),
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


# Singleton instance
_geo_intel_instance: Optional[GeopoliticalIntelligence] = None

def get_geopolitical_intel() -> GeopoliticalIntelligence:
    """Get singleton instance of GeopoliticalIntelligence."""
    global _geo_intel_instance
    if _geo_intel_instance is None:
        _geo_intel_instance = GeopoliticalIntelligence()
    return _geo_intel_instance
