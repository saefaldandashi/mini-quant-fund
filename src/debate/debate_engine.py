"""
Debate Engine for strategy signal evaluation and consensus building.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging

from src.strategies.base import Strategy, SignalOutput
from src.data.feature_store import Features
from src.data.regime import MarketRegime, TrendRegime, VolatilityRegime, RiskRegime

logger = logging.getLogger(__name__)


@dataclass
class StrategyScore:
    """Score and evaluation for a single strategy's proposal."""
    strategy_name: str
    
    # Core scores (0 to 1)
    alpha_score: float  # Expected return vs risk
    regime_fit_score: float  # How well strategy fits current regime
    diversification_score: float  # Contribution to diversification
    drawdown_score: float  # Respects drawdown constraints
    sentiment_score: float  # Agreement with sentiment
    
    # Composite score
    total_score: float
    
    # Rationale
    rationale: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class DebateTranscript:
    """Record of the debate process at a given timestamp."""
    timestamp: datetime
    regime: Optional[MarketRegime]
    
    # Strategy evaluations
    strategy_scores: Dict[str, StrategyScore] = field(default_factory=dict)
    
    # Agreements and disagreements
    agreements: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)
    
    # Top risks identified
    top_risks: List[str] = field(default_factory=list)
    
    # Final decision
    winning_strategies: List[str] = field(default_factory=list)
    final_weights: Dict[str, float] = field(default_factory=dict)
    constraints_applied: List[str] = field(default_factory=list)
    
    # Summary
    summary: str = ""
    
    def to_string(self) -> str:
        """Generate human-readable debate transcript."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"DEBATE TRANSCRIPT - {self.timestamp}")
        lines.append("=" * 60)
        
        # Regime
        if self.regime:
            lines.append(f"\nMARKET REGIME: {self.regime.description}")
        
        # Strategy evaluations
        lines.append("\n--- STRATEGY EVALUATIONS ---")
        for name, score in sorted(self.strategy_scores.items(), 
                                   key=lambda x: x[1].total_score, reverse=True):
            lines.append(f"\n{name}: {score.total_score:.2f}")
            lines.append(f"  Rationale: {score.rationale}")
            if score.strengths:
                lines.append(f"  Strengths: {', '.join(score.strengths)}")
            if score.weaknesses:
                lines.append(f"  Weaknesses: {', '.join(score.weaknesses)}")
        
        # Agreements/Disagreements
        if self.agreements:
            lines.append(f"\n--- AGREEMENTS ---")
            for a in self.agreements:
                lines.append(f"  + {a}")
        
        if self.disagreements:
            lines.append(f"\n--- DISAGREEMENTS ---")
            for d in self.disagreements:
                lines.append(f"  - {d}")
        
        # Risks
        if self.top_risks:
            lines.append(f"\n--- TOP RISKS ---")
            for i, risk in enumerate(self.top_risks, 1):
                lines.append(f"  {i}. {risk}")
        
        # Final decision
        lines.append(f"\n--- FINAL DECISION ---")
        lines.append(f"Winning strategies: {', '.join(self.winning_strategies)}")
        if self.constraints_applied:
            lines.append(f"Constraints applied: {', '.join(self.constraints_applied)}")
        
        lines.append(f"\n{self.summary}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class DebateEngine:
    """
    Orchestrates debate between strategies to produce consensus allocation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize debate engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Scoring weights
        self.alpha_weight = self.config.get('alpha_weight', 0.25)
        self.regime_weight = self.config.get('regime_weight', 0.25)
        self.diversification_weight = self.config.get('diversification_weight', 0.20)
        self.drawdown_weight = self.config.get('drawdown_weight', 0.15)
        self.sentiment_weight = self.config.get('sentiment_weight', 0.15)
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
    
    def run_debate(
        self,
        signals: Dict[str, SignalOutput],
        features: Features,
        current_drawdown: float = 0.0,
        max_drawdown: float = 0.20
    ) -> Tuple[Dict[str, StrategyScore], DebateTranscript]:
        """
        Run the debate process for all strategy signals.
        
        Args:
            signals: Dict of strategy name -> SignalOutput
            features: Current features
            current_drawdown: Current portfolio drawdown
            max_drawdown: Maximum allowed drawdown
            
        Returns:
            Tuple of (strategy_scores, debate_transcript)
        """
        timestamp = features.timestamp
        regime = features.regime
        
        # Score each strategy
        strategy_scores = {}
        for name, signal in signals.items():
            score = self._score_strategy(
                signal, features, regime, current_drawdown, max_drawdown
            )
            strategy_scores[name] = score
        
        # Identify agreements and disagreements
        agreements, disagreements = self._find_consensus(signals)
        
        # Identify top risks
        top_risks = self._identify_risks(signals, features, current_drawdown)
        
        # Select winning strategies
        winning = self._select_winners(strategy_scores)
        
        # Build transcript
        transcript = DebateTranscript(
            timestamp=timestamp,
            regime=regime,
            strategy_scores=strategy_scores,
            agreements=agreements,
            disagreements=disagreements,
            top_risks=top_risks,
            winning_strategies=winning,
        )
        
        # Generate summary
        transcript.summary = self._generate_summary(transcript)
        
        return strategy_scores, transcript
    
    def _score_strategy(
        self,
        signal: SignalOutput,
        features: Features,
        regime: Optional[MarketRegime],
        current_drawdown: float,
        max_drawdown: float
    ) -> StrategyScore:
        """Score a single strategy's proposal."""
        strengths = []
        weaknesses = []
        
        # 1. Alpha score: expected return vs risk
        if signal.risk_estimate > 0:
            sharpe = signal.expected_return / signal.risk_estimate
            alpha_score = min(1.0, max(0.0, (sharpe + 1.0) / 3.0))  # Map Sharpe to 0-1
        else:
            alpha_score = 0.5
        
        if alpha_score > 0.7:
            strengths.append(f"Strong risk-adjusted return (Sharpe: {sharpe:.2f})")
        elif alpha_score < 0.3:
            weaknesses.append(f"Weak risk-adjusted return")
        
        # 2. Regime fit score
        regime_fit_score = signal.regime_fit
        
        if regime:
            # Adjust based on strategy type and regime
            if regime.risk_regime == RiskRegime.RISK_OFF:
                # Penalize aggressive strategies in risk-off
                if signal.expected_return > 0.15:
                    regime_fit_score *= 0.7
                    weaknesses.append("Aggressive in risk-off environment")
                    
            if regime.volatility == VolatilityRegime.EXTREME:
                # Prefer defensive strategies
                if signal.risk_estimate < 0.10:
                    regime_fit_score *= 1.2
                    strengths.append("Conservative in high-vol environment")
        
        regime_fit_score = min(1.0, regime_fit_score)
        
        if regime_fit_score > 0.7:
            strengths.append(f"Good regime fit")
        elif regime_fit_score < 0.4:
            weaknesses.append(f"Poor regime fit")
        
        # 3. Diversification score
        diversification_score = signal.diversification_score
        n_positions = len([w for w in signal.desired_weights.values() if abs(w) > 0.01])
        
        if n_positions > 10:
            diversification_score = min(1.0, diversification_score * 1.1)
            strengths.append(f"Well diversified ({n_positions} positions)")
        elif n_positions < 3:
            diversification_score *= 0.7
            weaknesses.append(f"Concentrated ({n_positions} positions)")
        
        # 4. Drawdown score
        drawdown_score = 1.0
        
        if current_drawdown > max_drawdown * 0.5:
            # Be more cautious when in drawdown
            if signal.risk_estimate > 0.15:
                drawdown_score = 0.5
                weaknesses.append("High risk while in drawdown")
            else:
                strengths.append("Conservative during drawdown")
        
        # 5. Sentiment alignment score
        sentiment_score = 0.5
        
        if features.sentiment:
            aligned = 0
            total = 0
            
            for symbol, weight in signal.desired_weights.items():
                sent = features.sentiment.get(symbol)
                if sent and sent.confidence > 0.3:
                    total += 1
                    # Check if position direction aligns with sentiment
                    if (weight > 0 and sent.sentiment_score > 0) or \
                       (weight < 0 and sent.sentiment_score < 0):
                        aligned += 1
            
            if total > 0:
                sentiment_score = aligned / total
                
                if sentiment_score > 0.7:
                    strengths.append("Aligned with sentiment")
                elif sentiment_score < 0.3:
                    weaknesses.append("Against sentiment")
        
        # 6. MACRO INTELLIGENCE alignment score (NEW)
        macro_score = 0.5
        
        macro_features = getattr(features, 'macro_features', None)
        risk_sentiment = getattr(features, 'risk_sentiment', None)
        
        if macro_features or risk_sentiment:
            # Check if strategy aligns with macro environment
            net_exposure = sum(signal.desired_weights.values())  # Long bias
            
            if risk_sentiment:
                # Risk-on environment: prefer long exposure
                if risk_sentiment.risk_sentiment > 0.2 and net_exposure > 0:
                    macro_score = 0.7
                    strengths.append("Long in risk-on macro")
                elif risk_sentiment.risk_sentiment < -0.2 and net_exposure < 0.3:
                    macro_score = 0.7
                    strengths.append("Defensive in risk-off macro")
                elif risk_sentiment.risk_sentiment < -0.3 and net_exposure > 0.5:
                    macro_score = 0.3
                    weaknesses.append("Aggressive despite risk-off macro")
            
            if macro_features:
                # High geopolitical/financial stress: prefer conservative
                stress_level = max(
                    macro_features.geopolitical_risk_index,
                    macro_features.financial_stress_index
                )
                
                if stress_level > 0.5:
                    if signal.risk_estimate < 0.12:
                        macro_score = min(1.0, macro_score + 0.2)
                        strengths.append("Conservative during macro stress")
                    else:
                        macro_score = max(0.2, macro_score - 0.2)
                        weaknesses.append("High risk during macro stress")
        
        # Composite score (adjusted weights to include macro)
        total_score = (
            self.alpha_weight * 0.9 * alpha_score +
            self.regime_weight * 0.9 * regime_fit_score +
            self.diversification_weight * diversification_score +
            self.drawdown_weight * drawdown_score +
            self.sentiment_weight * 0.8 * sentiment_score +
            0.15 * macro_score  # Macro intelligence gets 15% weight
        )
        
        # Multiply by confidence
        total_score *= signal.confidence
        
        # Build rationale
        rationale = self._build_rationale(
            signal, alpha_score, regime_fit_score, diversification_score
        )
        
        return StrategyScore(
            strategy_name=signal.strategy_name,
            alpha_score=alpha_score,
            regime_fit_score=regime_fit_score,
            diversification_score=diversification_score,
            drawdown_score=drawdown_score,
            sentiment_score=sentiment_score,
            total_score=total_score,
            rationale=rationale,
            strengths=strengths,
            weaknesses=weaknesses,
        )
    
    def _build_rationale(
        self,
        signal: SignalOutput,
        alpha: float,
        regime: float,
        div: float
    ) -> str:
        """Build natural language rationale."""
        parts = []
        
        # Main thesis
        if signal.expected_return > 0.10:
            parts.append(f"Expects {signal.expected_return:.1%} return")
        elif signal.expected_return > 0:
            parts.append(f"Modest return expectation ({signal.expected_return:.1%})")
        else:
            parts.append("Defensive positioning")
        
        # Risk profile
        if signal.risk_estimate < 0.10:
            parts.append("low risk")
        elif signal.risk_estimate < 0.20:
            parts.append("moderate risk")
        else:
            parts.append("elevated risk")
        
        # Confidence
        if signal.confidence > 0.7:
            parts.append("high conviction")
        elif signal.confidence < 0.4:
            parts.append("low conviction")
        
        return f"{', '.join(parts)}."
    
    def _find_consensus(
        self,
        signals: Dict[str, SignalOutput]
    ) -> Tuple[List[str], List[str]]:
        """Find areas of agreement and disagreement between strategies."""
        agreements = []
        disagreements = []
        
        # Collect positions across strategies
        positions: Dict[str, List[Tuple[str, float]]] = {}
        
        for name, signal in signals.items():
            for symbol, weight in signal.desired_weights.items():
                if abs(weight) > 0.01:
                    if symbol not in positions:
                        positions[symbol] = []
                    positions[symbol].append((name, weight))
        
        # Find consensus
        for symbol, pos_list in positions.items():
            if len(pos_list) >= 3:
                # Multiple strategies have opinions
                bullish = sum(1 for _, w in pos_list if w > 0)
                bearish = sum(1 for _, w in pos_list if w < 0)
                
                if bullish >= 3 and bearish == 0:
                    agreements.append(f"Strong consensus: LONG {symbol} ({bullish} strategies)")
                elif bearish >= 3 and bullish == 0:
                    agreements.append(f"Strong consensus: SHORT {symbol} ({bearish} strategies)")
                elif bullish >= 2 and bearish >= 2:
                    disagreements.append(f"Conflicting views on {symbol} ({bullish} long, {bearish} short)")
        
        return agreements, disagreements
    
    def _identify_risks(
        self,
        signals: Dict[str, SignalOutput],
        features: Features,
        current_drawdown: float
    ) -> List[str]:
        """Identify top risks in current proposals."""
        risks = []
        
        # 1. Regime risk
        if features.regime:
            if features.regime.volatility == VolatilityRegime.EXTREME:
                risks.append("EXTREME VOLATILITY: Consider reducing overall exposure")
            elif features.regime.volatility == VolatilityRegime.HIGH:
                risks.append("Elevated volatility: Monitor positions closely")
            
            if features.regime.risk_regime == RiskRegime.RISK_OFF:
                risks.append("Risk-off environment: Defensive positioning recommended")
        
        # 2. Concentration risk
        combined_weights: Dict[str, float] = {}
        for signal in signals.values():
            for symbol, weight in signal.desired_weights.items():
                combined_weights[symbol] = combined_weights.get(symbol, 0) + weight
        
        max_weight = max(abs(w) for w in combined_weights.values()) if combined_weights else 0
        if max_weight > 0.3:
            top_symbol = max(combined_weights, key=lambda x: abs(combined_weights[x]))
            risks.append(f"Concentration risk: {top_symbol} has {max_weight:.1%} combined weight")
        
        # 3. Drawdown risk
        if current_drawdown > 0.10:
            risks.append(f"Current drawdown: {current_drawdown:.1%} - Reduce risk")
        
        # Limit to top 3
        return risks[:3]
    
    def _select_winners(
        self,
        scores: Dict[str, StrategyScore]
    ) -> List[str]:
        """Select winning strategies based on scores."""
        ranked = sorted(scores.items(), key=lambda x: x[1].total_score, reverse=True)
        
        # Select strategies with score > 0.5 or top 3
        winners = []
        for name, score in ranked:
            if score.total_score > 0.5 or len(winners) < 3:
                winners.append(name)
        
        return winners
    
    def _generate_summary(self, transcript: DebateTranscript) -> str:
        """Generate debate summary."""
        parts = []
        
        # Top strategy
        if transcript.winning_strategies:
            parts.append(f"Preferred strategy: {transcript.winning_strategies[0]}")
        
        # Key insight
        if transcript.agreements:
            parts.append(f"Key consensus: {transcript.agreements[0]}")
        elif transcript.disagreements:
            parts.append(f"Key disagreement: {transcript.disagreements[0]}")
        
        # Risk warning
        if transcript.top_risks:
            parts.append(f"Primary risk: {transcript.top_risks[0]}")
        
        return " | ".join(parts)
