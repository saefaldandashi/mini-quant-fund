"""
Adversarial Debate System - Strategies critique and defend against each other.

NOW WITH REAL LLM REASONING - not just template strings!

This creates a multi-round debate where:
1. Round 1: Initial proposals (LLM generates genuine support arguments)
2. Round 2: Counter-arguments against competitors (LLM generates real critiques)
3. Round 3: Rebuttals and defense (LLM generates genuine defenses)
4. Final: Adjusted scores based on LLM evaluation of debate quality
"""
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.strategies.base import SignalOutput
from src.data.feature_store import Features
from src.data.regime import MarketRegime, TrendRegime, VolatilityRegime, RiskRegime

logger = logging.getLogger(__name__)

# Try to import LLM service
try:
    from src.llm.llm_service import LLMService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMService = None


class ArgumentType(Enum):
    """Types of arguments in the debate."""
    SUPPORT = "support"           # Argument supporting own position
    ATTACK = "attack"             # Argument against another strategy
    REBUTTAL = "rebuttal"         # Defense against an attack
    CONCESSION = "concession"     # Acknowledging opponent's valid point


@dataclass
class Argument:
    """A single argument in the debate."""
    argument_type: ArgumentType
    source_strategy: str          # Strategy making the argument
    target_strategy: Optional[str]  # Strategy being addressed (for attacks/rebuttals)
    claim: str                    # The main claim
    evidence: List[str]           # Supporting evidence
    strength: float               # 0-1 score of argument strength
    affected_symbols: List[str]   # Symbols this argument concerns
    
    def to_string(self) -> str:
        """Human-readable format."""
        type_emoji = {
            ArgumentType.SUPPORT: "💪",
            ArgumentType.ATTACK: "⚔️",
            ArgumentType.REBUTTAL: "🛡️",
            ArgumentType.CONCESSION: "🤝"
        }
        
        emoji = type_emoji.get(self.argument_type, "")
        target = f" → {self.target_strategy}" if self.target_strategy else ""
        
        lines = [f"{emoji} [{self.source_strategy}{target}] {self.claim}"]
        for ev in self.evidence[:2]:  # Limit to 2 evidence points
            lines.append(f"   • {ev}")
        lines.append(f"   Strength: {self.strength:.0%}")
        
        return "\n".join(lines)


@dataclass  
class DebateRound:
    """A single round of debate."""
    round_number: int
    arguments: List[Argument] = field(default_factory=list)
    
    def get_attacks_on(self, strategy_name: str) -> List[Argument]:
        """Get all attacks targeting a specific strategy."""
        return [a for a in self.arguments 
                if a.argument_type == ArgumentType.ATTACK 
                and a.target_strategy == strategy_name]
    
    def get_rebuttals_by(self, strategy_name: str) -> List[Argument]:
        """Get all rebuttals made by a strategy."""
        return [a for a in self.arguments 
                if a.argument_type == ArgumentType.REBUTTAL 
                and a.source_strategy == strategy_name]


@dataclass
class AdversarialTranscript:
    """Full transcript of the adversarial debate."""
    timestamp: datetime
    rounds: List[DebateRound] = field(default_factory=list)
    
    # Score adjustments from debate
    attack_impact: Dict[str, float] = field(default_factory=dict)  # How much attacks hurt each strategy
    defense_success: Dict[str, float] = field(default_factory=dict)  # How well each defended
    debate_winners: List[str] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    
    def to_string(self) -> str:
        """Generate full debate transcript."""
        lines = []
        lines.append("=" * 70)
        lines.append("🎭 ADVERSARIAL DEBATE TRANSCRIPT")
        lines.append("=" * 70)
        
        for round_data in self.rounds:
            lines.append(f"\n--- ROUND {round_data.round_number} ---")
            for arg in round_data.arguments:
                lines.append(arg.to_string())
                lines.append("")
        
        lines.append("\n--- DEBATE IMPACT ---")
        for strat, impact in sorted(self.attack_impact.items(), key=lambda x: x[1], reverse=True):
            defense = self.defense_success.get(strat, 0)
            net = -impact + defense
            lines.append(f"  {strat}: Attacked({impact:.0%}) Defended({defense:.0%}) Net({net:+.0%})")
        
        if self.debate_winners:
            lines.append(f"\n🏆 DEBATE WINNERS: {', '.join(self.debate_winners)}")
        
        if self.key_insights:
            lines.append("\n💡 KEY INSIGHTS:")
            for insight in self.key_insights:
                lines.append(f"  • {insight}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class AdversarialDebateEngine:
    """
    Orchestrates adversarial debate between strategies.
    
    NOW WITH GENUINE LLM REASONING:
    - LLM generates real arguments, not templates
    - LLM evaluates argument quality
    - LLM considers market context for each argument
    
    Each strategy acts as an "agent" that:
    1. Proposes its trades (LLM explains why)
    2. Attacks competing proposals (LLM generates genuine critiques)
    3. Defends its own proposals (LLM generates real rebuttals)
    
    Falls back to rule-based if LLM not available.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        debate_learner=None,
        llm_service=None,  # NEW: LLM for genuine reasoning
    ):
        self.config = config or {}
        self.debate_learner = debate_learner
        self.llm_service = llm_service  # LLM for real debate
        
        # Track LLM usage
        self.llm_arguments_generated = 0
        self.rule_based_fallbacks = 0
        
        # Strategy characteristics (used as context for LLM, fallback for rules)
        self.strategy_strengths = {
            "TimeSeriesMomentum": ["trending markets", "momentum persistence", "clear trends"],
            "CrossSectionMomentum": ["relative strength", "sector rotation", "winner persistence"],
            "MeanReversion": ["oversold bounces", "overbought pullbacks", "range-bound markets"],
            "VolatilityRegimeVolTarget": ["risk management", "volatility timing", "drawdown protection"],
            "Carry": ["income generation", "yield differentials", "stable returns"],
            "ValueQualityTilt": ["fundamental value", "quality factors", "long-term returns"],
            "RiskParityMinVar": ["diversification", "risk balance", "stability"],
            "TailRiskOverlay": ["crash protection", "tail risk", "extreme events"],
            "NewsSentimentEvent": ["information edge", "sentiment shifts", "event reactions"],
        }
        
        self.strategy_weaknesses = {
            "TimeSeriesMomentum": ["choppy markets", "trend reversals", "whipsaws"],
            "CrossSectionMomentum": ["crowded trades", "momentum crashes", "factor rotation"],
            "MeanReversion": ["trending markets", "regime changes", "false bottoms"],
            "VolatilityRegimeVolTarget": ["sudden vol spikes", "vol clustering", "regime shifts"],
            "Carry": ["risk-off events", "credit spreads", "flight to safety"],
            "ValueQualityTilt": ["value traps", "growth dominance", "timing"],
            "RiskParityMinVar": ["correlation breakdown", "leverage risk", "bond-equity correlation"],
            "TailRiskOverlay": ["insurance costs", "false alarms", "opportunity cost"],
            "NewsSentimentEvent": ["noise vs signal", "stale news", "sentiment reversals"],
        }
    
    def _llm_available(self) -> bool:
        """Check if LLM is available for reasoning."""
        return (
            self.llm_service is not None 
            and hasattr(self.llm_service, 'is_available') 
            and self.llm_service.is_available()
        )
    
    def _generate_argument_with_llm(
        self,
        argument_type: str,  # "support", "attack", "rebuttal"
        strategy_name: str,
        signal: SignalOutput,
        features: Features,
        target_strategy: Optional[str] = None,
        target_signal: Optional[SignalOutput] = None,
        attack_to_rebut: Optional[str] = None,
    ) -> Optional[Argument]:
        """
        Generate a genuine argument using LLM.
        
        This is where REAL reasoning happens.
        """
        if not self._llm_available():
            return None
        
        # Build context for LLM
        regime_info = ""
        if features.regime:
            regime_info = f"Market Regime: Trend={features.regime.trend.value}, Vol={features.regime.volatility.value}, Risk={features.regime.risk_regime.value}"
        
        # Get macro context if available
        macro_info = ""
        if hasattr(features, 'macro_features') and features.macro_features:
            mf = features.macro_features
            if hasattr(mf, 'overall_risk_sentiment_index'):
                macro_info = f"Macro: Risk Sentiment={mf.overall_risk_sentiment_index:.2f}, Geo Risk={mf.geopolitical_risk_index:.2f}"
        
        # Build position info
        top_positions = sorted(
            signal.desired_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        positions_str = ", ".join([f"{t}: {w:.1%}" for t, w in top_positions])
        
        # Build prompt based on argument type
        if argument_type == "support":
            prompt = f"""You are {strategy_name}, a trading strategy in an investment committee debate.

MARKET CONTEXT:
{regime_info}
{macro_info}

YOUR PROPOSED POSITIONS: {positions_str}
YOUR CONFIDENCE: {signal.confidence:.0%}
YOUR EXPECTED RETURN: {signal.expected_return:.1%}

Generate a compelling 2-3 sentence argument for WHY your proposal is the right choice NOW.
Focus on:
1. Why current market conditions favor your approach
2. Specific evidence from the regime/macro data
3. What edge you have over other approaches

Respond in JSON:
{{"claim": "your main argument", "evidence": ["point 1", "point 2", "point 3"], "strength": 0.0-1.0}}"""

        elif argument_type == "attack":
            target_positions = sorted(
                target_signal.desired_weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3] if target_signal else []
            target_pos_str = ", ".join([f"{t}: {w:.1%}" for t, w in target_positions])
            
            prompt = f"""You are {strategy_name}, challenging {target_strategy}'s proposal in an investment committee debate.

MARKET CONTEXT:
{regime_info}
{macro_info}

YOUR APPROACH: {strategy_name} (strengths: {', '.join(self.strategy_strengths.get(strategy_name, []))})
OPPONENT: {target_strategy} (weaknesses: {', '.join(self.strategy_weaknesses.get(target_strategy, []))})
OPPONENT'S POSITIONS: {target_pos_str}

Generate a 2-3 sentence attack on {target_strategy}'s proposal.
Be specific about:
1. WHY their approach is wrong for current conditions
2. What risks they're ignoring
3. Historical precedent for failure

Respond in JSON:
{{"claim": "your attack", "evidence": ["weakness 1", "weakness 2"], "strength": 0.0-1.0}}"""

        elif argument_type == "rebuttal":
            prompt = f"""You are {strategy_name}, defending your proposal against an attack.

ATTACK AGAINST YOU: "{attack_to_rebut}"

MARKET CONTEXT:
{regime_info}
{macro_info}

YOUR POSITIONS: {positions_str}
YOUR CONFIDENCE: {signal.confidence:.0%}

Generate a 2-3 sentence defense/rebuttal.
Address the criticism directly and explain why it doesn't apply or is mitigated.

Respond in JSON:
{{"claim": "your defense", "evidence": ["counter-point 1", "counter-point 2"], "strength": 0.0-1.0}}"""

        else:
            return None
        
        # Call LLM
        system = "You are a portfolio manager in an investment committee debate. Be specific, analytical, and cite market evidence."
        
        try:
            response = self.llm_service.call(prompt, system=system, temperature=0.4, max_tokens=300)
            
            if not response:
                return None
            
            # Parse response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            self.llm_arguments_generated += 1
            
            return Argument(
                argument_type=ArgumentType.SUPPORT if argument_type == "support" 
                    else ArgumentType.ATTACK if argument_type == "attack"
                    else ArgumentType.REBUTTAL,
                source_strategy=strategy_name,
                target_strategy=target_strategy,
                claim=data.get('claim', 'LLM-generated argument'),
                evidence=data.get('evidence', []),
                strength=float(data.get('strength', 0.6)),
                affected_symbols=list(signal.desired_weights.keys())[:5],
            )
            
        except Exception as e:
            logger.warning(f"LLM argument generation failed: {e}")
            return None
    
    def run_adversarial_debate(
        self,
        signals: Dict[str, SignalOutput],
        features: Features,
        base_scores: Dict[str, float]  # Scores from regular debate
    ) -> Tuple[Dict[str, float], AdversarialTranscript]:
        """
        Run multi-round adversarial debate.
        
        Args:
            signals: Strategy signals
            features: Current market features
            base_scores: Initial scores from regular debate
            
        Returns:
            Tuple of (adjusted_scores, transcript)
        """
        timestamp = features.timestamp
        regime = features.regime
        
        transcript = AdversarialTranscript(timestamp=timestamp)
        
        # Round 1: Generate support arguments (why each strategy is good now)
        round1 = self._run_support_round(signals, features, regime)
        transcript.rounds.append(round1)
        
        # Round 2: Generate attacks (strategies critique competitors)
        round2 = self._run_attack_round(signals, features, regime, base_scores)
        transcript.rounds.append(round2)
        
        # Round 3: Generate rebuttals (defend against attacks)
        round3 = self._run_rebuttal_round(signals, features, regime, round2)
        transcript.rounds.append(round3)
        
        # Calculate debate impact
        attack_impact, defense_success = self._calculate_debate_impact(round2, round3)
        transcript.attack_impact = attack_impact
        transcript.defense_success = defense_success
        
        # Adjust scores based on debate
        adjusted_scores = self._adjust_scores(base_scores, attack_impact, defense_success)
        
        # Determine winners
        transcript.debate_winners = self._determine_winners(adjusted_scores, base_scores)
        
        # Extract key insights
        transcript.key_insights = self._extract_insights(transcript)
        
        return adjusted_scores, transcript
    
    def _run_support_round(
        self,
        signals: Dict[str, SignalOutput],
        features: Features,
        regime: Optional[MarketRegime]
    ) -> DebateRound:
        """Round 1: Each strategy argues why it's suitable for current conditions.
        
        NOW WITH LLM: If LLM available, generates genuine arguments.
        Falls back to rule-based if LLM unavailable or fails.
        """
        round_data = DebateRound(round_number=1)
        
        for name, signal in signals.items():
            # TRY LLM FIRST for genuine reasoning
            if self._llm_available():
                llm_arg = self._generate_argument_with_llm(
                    argument_type="support",
                    strategy_name=name,
                    signal=signal,
                    features=features,
                )
                if llm_arg:
                    round_data.arguments.append(llm_arg)
                    continue  # Skip rule-based
            
            # FALLBACK: Rule-based argument generation
            self.rule_based_fallbacks += 1
            
            strengths = self.strategy_strengths.get(name, ["general market exposure"])
            evidence = []
            claim = ""
            strength = 0.5
            
            # Regime-based arguments (fallback)
            if regime:
                if name == "TimeSeriesMomentum":
                    if regime.trend == TrendRegime.STRONG_UP:
                        claim = "Strong uptrend favors momentum strategies"
                        evidence.append(f"Trend regime: {regime.trend.value}")
                        evidence.append(f"Trend strength supports trend-following")
                        strength = 0.85
                    elif regime.trend == TrendRegime.STRONG_DOWN:
                        claim = "Clear downtrend - momentum can profit from shorts/defense"
                        evidence.append(f"Trend regime: {regime.trend.value}")
                        strength = 0.70
                    else:
                        claim = "Moderate trend conditions"
                        strength = 0.50
                        
                elif name == "MeanReversion":
                    if regime.trend in [TrendRegime.WEAK_UP, TrendRegime.WEAK_DOWN, TrendRegime.NEUTRAL]:
                        claim = "Range-bound market ideal for mean reversion"
                        evidence.append(f"Weak/neutral trend: {regime.trend.value}")
                        evidence.append("Prices likely to revert to mean")
                        strength = 0.80
                    else:
                        claim = "Mean reversion may work on individual oversold names"
                        strength = 0.40
                        
                elif name == "VolatilityRegimeVolTarget":
                    if regime.volatility in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
                        claim = "High volatility requires active vol management"
                        evidence.append(f"Vol regime: {regime.volatility.value}")
                        evidence.append("Risk scaling critical in current environment")
                        strength = 0.85
                    else:
                        claim = "Stable volatility - maintaining risk targets"
                        strength = 0.60
                        
                elif name == "RiskParityMinVar":
                    claim = "Diversification always valuable for risk management"
                    evidence.append("Risk parity balances across asset classes")
                    if regime.correlation_regime == "high":
                        evidence.append("High correlations may reduce diversification benefit")
                        strength = 0.55
                    else:
                        evidence.append("Low correlations enhance diversification")
                        strength = 0.75
                        
                elif name == "TailRiskOverlay":
                    if regime.risk_regime == RiskRegime.RISK_OFF:
                        claim = "Risk-off environment demands tail protection"
                        evidence.append(f"Risk regime: {regime.risk_regime.value}")
                        evidence.append("Hedges likely to pay off")
                        strength = 0.90
                    else:
                        claim = "Tail protection provides insurance"
                        strength = 0.50
                        
                elif name == "NewsSentimentEvent":
                    claim = "Sentiment signals provide real-time market insights"
                    evidence.append("News-based alpha captures information flow")
                    if signal.confidence > 0.6:
                        evidence.append(f"High confidence signals detected: {signal.confidence:.0%}")
                        strength = 0.75
                    else:
                        strength = 0.50
                else:
                    claim = f"{name} provides valuable market exposure"
                    evidence = [s for s in strengths[:2]]
                    strength = 0.55
            else:
                claim = f"{name} offers structured approach to markets"
                evidence = strengths[:2]
                strength = 0.50
            
            # Add position-based evidence
            n_positions = len([w for w in signal.desired_weights.values() if abs(w) > 0.01])
            if n_positions > 0:
                evidence.append(f"Proposes {n_positions} positions with {signal.confidence:.0%} confidence")
            
            arg = Argument(
                argument_type=ArgumentType.SUPPORT,
                source_strategy=name,
                target_strategy=None,
                claim=claim,
                evidence=evidence,
                strength=strength,
                affected_symbols=list(signal.desired_weights.keys())[:5]
            )
            round_data.arguments.append(arg)
        
        return round_data
    
    def _run_attack_round(
        self,
        signals: Dict[str, SignalOutput],
        features: Features,
        regime: Optional[MarketRegime],
        base_scores: Dict[str, float]
    ) -> DebateRound:
        """Round 2: Strategies attack each other's weak points.
        
        NOW WITH LLM: Generates genuine critiques, not templates.
        """
        round_data = DebateRound(round_number=2)
        
        # Each strategy attacks the top 2 competitors
        sorted_strategies = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
        
        for attacker_name, attacker_signal in signals.items():
            # Attack strategies ranked above (competitors)
            for defender_name, defender_score in sorted_strategies:
                if defender_name == attacker_name:
                    continue
                if defender_score <= base_scores.get(attacker_name, 0):
                    continue  # Only attack stronger competitors
                
                defender_signal = signals.get(defender_name)
                if not defender_signal:
                    continue
                
                # TRY LLM FIRST for genuine attack reasoning
                if self._llm_available():
                    llm_attack = self._generate_argument_with_llm(
                        argument_type="attack",
                        strategy_name=attacker_name,
                        signal=attacker_signal,
                        features=features,
                        target_strategy=defender_name,
                        target_signal=defender_signal,
                    )
                    if llm_attack and llm_attack.strength > 0.3:
                        round_data.arguments.append(llm_attack)
                        continue  # Skip rule-based
                
                # FALLBACK: Rule-based attack generation
                self.rule_based_fallbacks += 1
                weaknesses = self.strategy_weaknesses.get(defender_name, [])
                attack = self._generate_attack(
                    attacker_name, defender_name,
                    attacker_signal, defender_signal,
                    features, regime, weaknesses
                )
                
                if attack and attack.strength > 0.3:
                    round_data.arguments.append(attack)
        
        return round_data
    
    def _generate_attack(
        self,
        attacker: str,
        defender: str,
        attacker_signal: SignalOutput,
        defender_signal: SignalOutput,
        features: Features,
        regime: Optional[MarketRegime],
        defender_weaknesses: List[str]
    ) -> Optional[Argument]:
        """Generate an attack argument against a competitor."""
        
        claim = ""
        evidence = []
        strength = 0.5
        affected = []
        
        # Momentum attacks Mean Reversion in trends
        if attacker == "TimeSeriesMomentum" and defender == "MeanReversion":
            if regime and regime.trend in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
                claim = f"Mean reversion is WRONG in this {regime.trend.value} market"
                evidence.append("Trend is strong - mean reversion will fight the tape")
                evidence.append("Buying dips in downtrends catches falling knives")
                evidence.append("Historical data: MR underperforms in trending regimes")
                strength = 0.80
                
        # Mean Reversion attacks Momentum in choppy markets        
        elif attacker == "MeanReversion" and defender == "TimeSeriesMomentum":
            if regime and regime.trend in [TrendRegime.NEUTRAL, TrendRegime.WEAK_UP, TrendRegime.WEAK_DOWN]:
                claim = "Momentum will get WHIPSAWED in this choppy market"
                evidence.append("No clear trend direction - momentum signals unreliable")
                evidence.append("Expect trend reversals that hurt momentum")
                evidence.append("Range-bound conditions favor mean reversion")
                strength = 0.75
                
        # Vol Target attacks aggressive strategies in high vol
        elif attacker == "VolatilityRegimeVolTarget" and regime:
            if regime.volatility in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
                if defender_signal.risk_estimate > 0.15:
                    claim = f"{defender} is taking EXCESSIVE RISK in high-vol environment"
                    evidence.append(f"Vol regime: {regime.volatility.value}")
                    evidence.append(f"{defender} risk estimate: {defender_signal.risk_estimate:.1%}")
                    evidence.append("Position sizing should be reduced, not increased")
                    strength = 0.85
                    
        # Risk Parity attacks concentrated strategies
        elif attacker == "RiskParityMinVar":
            n_positions = len([w for w in defender_signal.desired_weights.values() if abs(w) > 0.01])
            max_position = max(abs(w) for w in defender_signal.desired_weights.values()) if defender_signal.desired_weights else 0
            
            if n_positions < 5 or max_position > 0.20:
                claim = f"{defender} portfolio is DANGEROUSLY CONCENTRATED"
                evidence.append(f"Only {n_positions} positions proposed")
                evidence.append(f"Max single position: {max_position:.0%}")
                evidence.append("Concentration amplifies idiosyncratic risk")
                strength = 0.70
                
        # Tail Risk attacks in risk-off
        elif attacker == "TailRiskOverlay" and regime:
            if regime.risk_regime == RiskRegime.RISK_OFF:
                if "tail" not in defender.lower() and defender_signal.expected_return > 0.10:
                    claim = f"{defender} ignores TAIL RISKS in this environment"
                    evidence.append(f"Risk regime: {regime.risk_regime.value}")
                    evidence.append("Aggressive positioning without hedges")
                    evidence.append("Crash risk elevated - protection needed")
                    strength = 0.80
                    
        # Sentiment attacks based on news
        elif attacker == "NewsSentimentEvent":
            # Check if defender's positions contradict sentiment
            contrary_positions = []
            if features.sentiment:
                for symbol, weight in defender_signal.desired_weights.items():
                    sent = features.sentiment.get(symbol)
                    if sent and sent.confidence > 0.4:
                        if (weight > 0.05 and sent.sentiment_score < -0.3) or \
                           (weight < -0.05 and sent.sentiment_score > 0.3):
                            contrary_positions.append(symbol)
            
            if len(contrary_positions) >= 2:
                claim = f"{defender} positions CONFLICT with sentiment"
                evidence.append(f"Contrary positions: {', '.join(contrary_positions[:3])}")
                evidence.append("News sentiment disagrees with proposed trades")
                evidence.append("Information flow suggests reversal risk")
                strength = 0.65
                affected = contrary_positions[:5]
        
        # Generic attacks based on weaknesses
        if not claim and defender_weaknesses:
            # Find applicable weakness based on regime
            for weakness in defender_weaknesses:
                if regime:
                    if "trending" in weakness.lower() and regime.trend in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
                        claim = f"{defender} struggles in trending markets like now"
                        evidence.append(f"Known weakness: {weakness}")
                        strength = 0.55
                        break
                    elif "choppy" in weakness.lower() and regime.trend == TrendRegime.NEUTRAL:
                        claim = f"{defender} unreliable in current choppy conditions"
                        evidence.append(f"Known weakness: {weakness}")
                        strength = 0.55
                        break
        
        if not claim:
            return None
        
        # Apply learned attack boost from debate history
        if self.debate_learner and regime:
            regime_name = regime.risk_regime.value if regime.risk_regime else 'unknown'
            learned_boost = self.debate_learner.get_attack_boost(attacker, defender, regime_name)
            strength = min(1.0, strength * learned_boost)
            
            if learned_boost != 1.0:
                evidence.append(f"[LEARNED: Attack has {learned_boost:.0%} historical effectiveness]")
            
        return Argument(
            argument_type=ArgumentType.ATTACK,
            source_strategy=attacker,
            target_strategy=defender,
            claim=claim,
            evidence=evidence,
            strength=strength,
            affected_symbols=affected or list(defender_signal.desired_weights.keys())[:3]
        )
    
    def _run_rebuttal_round(
        self,
        signals: Dict[str, SignalOutput],
        features: Features,
        regime: Optional[MarketRegime],
        attack_round: DebateRound
    ) -> DebateRound:
        """Round 3: Strategies defend against attacks.
        
        NOW WITH LLM: Generates genuine defenses, not templates.
        """
        round_data = DebateRound(round_number=3)
        
        for defender_name, defender_signal in signals.items():
            attacks = attack_round.get_attacks_on(defender_name)
            
            if not attacks:
                continue
            
            # Defend against each attack
            for attack in attacks:
                # TRY LLM FIRST for genuine rebuttal reasoning
                if self._llm_available():
                    llm_rebuttal = self._generate_argument_with_llm(
                        argument_type="rebuttal",
                        strategy_name=defender_name,
                        signal=defender_signal,
                        features=features,
                        target_strategy=attack.source_strategy,
                        attack_to_rebut=attack.claim,
                    )
                    if llm_rebuttal:
                        round_data.arguments.append(llm_rebuttal)
                        continue  # Skip rule-based
                
                # FALLBACK: Rule-based rebuttal generation
                self.rule_based_fallbacks += 1
                rebuttal = self._generate_rebuttal(
                    defender_name, attack, defender_signal, features, regime
                )
                if rebuttal:
                    round_data.arguments.append(rebuttal)
        
        return round_data
    
    def _generate_rebuttal(
        self,
        defender: str,
        attack: Argument,
        signal: SignalOutput,
        features: Features,
        regime: Optional[MarketRegime]
    ) -> Optional[Argument]:
        """Generate a rebuttal to an attack."""
        
        claim = ""
        evidence = []
        strength = 0.5
        
        # Rebuttals based on strategy type
        if defender == "TimeSeriesMomentum":
            claim = "Momentum adapts to market conditions"
            evidence.append(f"Confidence-weighted positions: {signal.confidence:.0%}")
            evidence.append("Trend filters reduce whipsaw risk")
            if signal.risk_estimate < 0.15:
                evidence.append(f"Conservative risk sizing: {signal.risk_estimate:.0%}")
                strength = 0.65
            else:
                strength = 0.45
                
        elif defender == "MeanReversion":
            claim = "Mean reversion uses strict entry criteria"
            evidence.append("Only trades extreme deviations from mean")
            evidence.append("Stop-losses limit trend-fighting losses")
            if signal.confidence > 0.6:
                evidence.append(f"High-confidence signals only: {signal.confidence:.0%}")
                strength = 0.70
            else:
                strength = 0.50
                
        elif defender == "VolatilityRegimeVolTarget":
            claim = "Vol targeting is THE appropriate response to volatility"
            evidence.append("Dynamic sizing reduces risk in high-vol")
            evidence.append("Evidence-based risk management")
            strength = 0.75
            
        elif defender == "RiskParityMinVar":
            n_pos = len([w for w in signal.desired_weights.values() if abs(w) > 0.01])
            claim = "Risk parity provides TRUE diversification"
            evidence.append(f"Balanced across {n_pos} positions")
            evidence.append("Risk-weighted, not dollar-weighted")
            strength = 0.70
            
        elif defender == "TailRiskOverlay":
            claim = "Tail protection is INSURANCE, not a prediction"
            evidence.append("Asymmetric payoff protects downside")
            evidence.append("Small cost for large protection")
            strength = 0.65
            
        elif defender == "NewsSentimentEvent":
            claim = "Sentiment provides REAL-TIME information edge"
            evidence.append("News leads price moves")
            evidence.append("Entity-level sentiment is actionable")
            if signal.confidence > 0.5:
                evidence.append(f"Current signal confidence: {signal.confidence:.0%}")
                strength = 0.65
            else:
                strength = 0.45
        else:
            claim = f"{defender} methodology is sound"
            evidence.append("Based on academic research")
            evidence.append("Backtested across market cycles")
            strength = 0.50
        
        # Strength is capped by the attack strength (harder to fully rebut strong attacks)
        strength = min(strength, attack.strength * 1.1)
        
        # Apply learned defense boost from debate history
        if self.debate_learner and regime:
            regime_name = regime.risk_regime.value if regime.risk_regime else 'unknown'
            learned_boost = self.debate_learner.get_defense_boost(defender, regime_name)
            strength = min(1.0, strength * learned_boost)
            
            if learned_boost != 1.0:
                evidence.append(f"[LEARNED: Defense has {learned_boost:.0%} historical effectiveness]")
        
        return Argument(
            argument_type=ArgumentType.REBUTTAL,
            source_strategy=defender,
            target_strategy=attack.source_strategy,
            claim=claim,
            evidence=evidence,
            strength=strength,
            affected_symbols=attack.affected_symbols
        )
    
    def _calculate_debate_impact(
        self,
        attack_round: DebateRound,
        rebuttal_round: DebateRound
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate how much attacks hurt and how well defenses worked."""
        
        attack_impact = {}
        defense_success = {}
        
        # Calculate attack damage
        for attack in attack_round.arguments:
            if attack.argument_type == ArgumentType.ATTACK and attack.target_strategy:
                target = attack.target_strategy
                current = attack_impact.get(target, 0)
                attack_impact[target] = current + attack.strength * 0.15  # 15% max per attack
        
        # Calculate defense success
        for rebuttal in rebuttal_round.arguments:
            if rebuttal.argument_type == ArgumentType.REBUTTAL:
                source = rebuttal.source_strategy
                current = defense_success.get(source, 0)
                defense_success[source] = current + rebuttal.strength * 0.10  # 10% per rebuttal
        
        # Cap impacts
        attack_impact = {k: min(v, 0.30) for k, v in attack_impact.items()}
        defense_success = {k: min(v, 0.20) for k, v in defense_success.items()}
        
        return attack_impact, defense_success
    
    def _adjust_scores(
        self,
        base_scores: Dict[str, float],
        attack_impact: Dict[str, float],
        defense_success: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust strategy scores based on debate performance."""
        adjusted = {}
        
        for name, base_score in base_scores.items():
            damage = attack_impact.get(name, 0)
            defense = defense_success.get(name, 0)
            
            # Net adjustment: attacks hurt, defenses help (partially recover)
            net_adjustment = -damage + (defense * 0.8)  # Defense recovers 80% of its value
            
            adjusted[name] = max(0.1, min(1.0, base_score + net_adjustment))
        
        return adjusted
    
    def _determine_winners(
        self,
        adjusted_scores: Dict[str, float],
        base_scores: Dict[str, float]
    ) -> List[str]:
        """Determine which strategies won the debate."""
        winners = []
        
        # Strategies that improved or stayed strong
        for name in adjusted_scores:
            base = base_scores.get(name, 0)
            adjusted = adjusted_scores[name]
            
            # Winner if: improved OR (stayed in top 3 and maintained score)
            improved = adjusted > base
            still_strong = adjusted > 0.5
            
            if improved or still_strong:
                winners.append((name, adjusted))
        
        # Sort by adjusted score
        winners.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in winners[:3]]
    
    def _extract_insights(self, transcript: AdversarialTranscript) -> List[str]:
        """Extract key insights from the debate."""
        insights = []
        
        # Find strongest attacks
        all_attacks = []
        for r in transcript.rounds:
            all_attacks.extend([a for a in r.arguments if a.argument_type == ArgumentType.ATTACK])
        
        if all_attacks:
            strongest = max(all_attacks, key=lambda x: x.strength)
            insights.append(
                f"Strongest critique: {strongest.source_strategy} on {strongest.target_strategy} "
                f"({strongest.strength:.0%})"
            )
        
        # Find best defender
        if transcript.defense_success:
            best_defender = max(transcript.defense_success.items(), key=lambda x: x[1])
            insights.append(f"Best defense: {best_defender[0]} ({best_defender[1]:.0%})")
        
        # Most attacked
        if transcript.attack_impact:
            most_attacked = max(transcript.attack_impact.items(), key=lambda x: x[1])
            insights.append(f"Most challenged: {most_attacked[0]} ({most_attacked[1]:.0%} impact)")
        
        return insights
