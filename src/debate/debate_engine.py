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

    # NEW: Urgency & Opportunity (0 to 1)
    urgency_score: float = 0.5       # How time-sensitive is this signal
    opportunity_score: float = 0.5   # How large is the expected move
    conviction_score: float = 0.5    # How strong is the overall conviction


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
    
    # NEW: Aggregate urgency & opportunity
    aggregate_urgency: float = 0.5    # Overall market urgency (avg of winners)
    aggregate_opportunity: float = 0.5  # Overall opportunity magnitude
    performance_mode: str = "NORMAL"  # Current performance state for context
    
    def to_string(self) -> str:
        """Generate human-readable debate transcript."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"DEBATE TRANSCRIPT - {self.timestamp}")
        lines.append("=" * 60)
        
        # Regime
        if self.regime:
            lines.append(f"\nMARKET REGIME: {self.regime.description}")
        
        # Performance mode & urgency
        lines.append(f"\nPERFORMANCE MODE: {self.performance_mode}")
        lines.append(f"AGGREGATE URGENCY: {self.aggregate_urgency:.0%}")
        lines.append(f"AGGREGATE OPPORTUNITY: {self.aggregate_opportunity:.0%}")
        
        # Strategy evaluations
        lines.append("\n--- STRATEGY EVALUATIONS ---")
        for name, score in sorted(self.strategy_scores.items(), 
                                   key=lambda x: x[1].total_score, reverse=True):
            lines.append(f"\n{name}: {score.total_score:.2f} [urgency={score.urgency_score:.0%}, opportunity={score.opportunity_score:.0%}]")
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
    
    Enhanced with:
    - Urgency scoring (time-sensitive signals get priority)
    - Opportunity magnitude (bigger expected moves get priority)  
    - Performance-state awareness (bias toward conviction when winning)
    - Conviction escalation (high agreement = bigger allocations)
    """
    
    # Performance states that bias the debate toward action
    OFFENSIVE_STATES = {"SURGE", "MOMENTUM", "EXCELLENT", "PRESSING"}
    DEFENSIVE_STATES = {"CONCERNING", "BAD", "CRITICAL"}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize debate engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Scoring weights (base — adjusted dynamically by performance state)
        self.alpha_weight = self.config.get('alpha_weight', 0.25)
        self.regime_weight = self.config.get('regime_weight', 0.25)
        self.diversification_weight = self.config.get('diversification_weight', 0.20)
        self.drawdown_weight = self.config.get('drawdown_weight', 0.15)
        self.sentiment_weight = self.config.get('sentiment_weight', 0.15)
        
        # NEW: Urgency & opportunity weights
        self.urgency_weight = self.config.get('urgency_weight', 0.10)
        self.opportunity_weight = self.config.get('opportunity_weight', 0.10)
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
    
    def _get_performance_adjusted_weights(self, performance_state: str) -> Dict[str, float]:
        """
        Dynamically adjust scoring weights based on performance state.
        
        When winning: bias toward alpha/urgency/opportunity (offense)
        When losing: bias toward drawdown/diversification (defense)
        When neutral: balanced
        """
        if performance_state in self.OFFENSIVE_STATES:
            # WINNING: Prioritize alpha, urgency, opportunity — reduce caution
            offense_factor = 1.3 if performance_state == "SURGE" else 1.2 if performance_state == "MOMENTUM" else 1.1
            return {
                'alpha': self.alpha_weight * offense_factor,
                'regime': self.regime_weight * 0.9,
                'diversification': self.diversification_weight * 0.7,  # Less diversification drag
                'drawdown': self.drawdown_weight * 0.5,               # Much less caution
                'sentiment': self.sentiment_weight,
                'urgency': self.urgency_weight * offense_factor,
                'opportunity': self.opportunity_weight * offense_factor,
            }
        elif performance_state in self.DEFENSIVE_STATES:
            # LOSING: Prioritize safety
            defense_factor = 1.3 if performance_state == "CRITICAL" else 1.15
            return {
                'alpha': self.alpha_weight * 0.8,
                'regime': self.regime_weight * 1.1,
                'diversification': self.diversification_weight * defense_factor,
                'drawdown': self.drawdown_weight * defense_factor,
                'sentiment': self.sentiment_weight,
                'urgency': self.urgency_weight * 0.5,
                'opportunity': self.opportunity_weight * 0.7,
            }
        else:
            # NEUTRAL / GOOD: balanced
            return {
                'alpha': self.alpha_weight,
                'regime': self.regime_weight,
                'diversification': self.diversification_weight,
                'drawdown': self.drawdown_weight,
                'sentiment': self.sentiment_weight,
                'urgency': self.urgency_weight,
                'opportunity': self.opportunity_weight,
            }
    
    def run_debate(
        self,
        signals: Dict[str, SignalOutput],
        features: Features,
        current_drawdown: float = 0.0,
        max_drawdown: float = 0.20,
        performance_state: str = "NORMAL",
        current_pnl_pct: float = 0.0,
    ) -> Tuple[Dict[str, StrategyScore], DebateTranscript]:
        """
        Run the debate process for all strategy signals.
        
        Args:
            signals: Dict of strategy name -> SignalOutput
            features: Current features
            current_drawdown: Current portfolio drawdown
            max_drawdown: Maximum allowed drawdown
            performance_state: Current performance state (SURGE/MOMENTUM/NORMAL/etc.)
            current_pnl_pct: Current day P&L percentage (positive = profitable)
            
        Returns:
            Tuple of (strategy_scores, debate_transcript)
        """
        timestamp = features.timestamp
        regime = features.regime
        
        # Get performance-adjusted weights
        weights = self._get_performance_adjusted_weights(performance_state)
        
        # Score each strategy
        strategy_scores = {}
        for name, signal in signals.items():
            score = self._score_strategy(
                signal, features, regime, current_drawdown, max_drawdown,
                performance_state=performance_state,
                current_pnl_pct=current_pnl_pct,
                scoring_weights=weights,
            )
            strategy_scores[name] = score
        
        # Conviction escalation: boost scores when multiple strategies agree with high confidence
        strategy_scores = self._apply_conviction_escalation(strategy_scores, signals, performance_state)
        
        # Identify agreements and disagreements
        agreements, disagreements = self._find_consensus(signals)
        
        # Identify top risks
        top_risks = self._identify_risks(signals, features, current_drawdown)
        
        # Select winning strategies
        winning = self._select_winners(strategy_scores)
        
        # Calculate aggregate urgency & opportunity from winners
        if winning:
            winner_scores = [strategy_scores[w] for w in winning if w in strategy_scores]
            agg_urgency = np.mean([s.urgency_score for s in winner_scores]) if winner_scores else 0.5
            agg_opportunity = np.mean([s.opportunity_score for s in winner_scores]) if winner_scores else 0.5
        else:
            agg_urgency = 0.5
            agg_opportunity = 0.5
        
        # Build transcript
        transcript = DebateTranscript(
            timestamp=timestamp,
            regime=regime,
            strategy_scores=strategy_scores,
            agreements=agreements,
            disagreements=disagreements,
            top_risks=top_risks,
            winning_strategies=winning,
            aggregate_urgency=agg_urgency,
            aggregate_opportunity=agg_opportunity,
            performance_mode=performance_state,
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
        max_drawdown: float,
        performance_state: str = "NORMAL",
        current_pnl_pct: float = 0.0,
        scoring_weights: Optional[Dict[str, float]] = None,
    ) -> StrategyScore:
        """Score a single strategy's proposal with urgency, opportunity, and performance awareness."""
        strengths = []
        weaknesses = []
        
        w = scoring_weights or {
            'alpha': self.alpha_weight, 'regime': self.regime_weight,
            'diversification': self.diversification_weight, 'drawdown': self.drawdown_weight,
            'sentiment': self.sentiment_weight, 'urgency': self.urgency_weight,
            'opportunity': self.opportunity_weight,
        }
        
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
            if regime.risk_regime == RiskRegime.RISK_OFF:
                if signal.expected_return > 0.15:
                    regime_fit_score *= 0.7
                    weaknesses.append("Aggressive in risk-off environment")
                    
            if regime.volatility == VolatilityRegime.EXTREME:
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
        n_positions = len([wt for wt in signal.desired_weights.values() if abs(wt) > 0.01])
        
        if n_positions > 10:
            diversification_score = min(1.0, diversification_score * 1.1)
            strengths.append(f"Well diversified ({n_positions} positions)")
        elif n_positions < 3:
            diversification_score *= 0.7
            weaknesses.append(f"Concentrated ({n_positions} positions)")
        
        # 4. Drawdown score — NOW ASYMMETRIC: penalizes in drawdown, REWARDS in profit
        drawdown_score = 1.0
        
        if current_drawdown > max_drawdown * 0.5:
            # In drawdown: be cautious
            if signal.risk_estimate > 0.15:
                drawdown_score = 0.5
                weaknesses.append("High risk while in drawdown")
            else:
                strengths.append("Conservative during drawdown")
        elif current_pnl_pct > 0.01:
            # IN PROFIT: boost strategies proportional to how well we're doing
            profit_boost = min(0.3, current_pnl_pct * 5.0)  # Up to 30% boost
            drawdown_score = 1.0 + profit_boost
            strengths.append(f"Profit tailwind (+{profit_boost:.0%} boost)")
        
        # 5. Sentiment alignment score
        sentiment_score = 0.5
        
        if features.sentiment:
            aligned = 0
            total = 0
            
            for symbol, weight in signal.desired_weights.items():
                sent = features.sentiment.get(symbol)
                if sent and sent.confidence > 0.3:
                    total += 1
                    if (weight > 0 and sent.sentiment_score > 0) or \
                       (weight < 0 and sent.sentiment_score < 0):
                        aligned += 1
            
            if total > 0:
                sentiment_score = aligned / total
                
                if sentiment_score > 0.7:
                    strengths.append("Aligned with sentiment")
                elif sentiment_score < 0.3:
                    weaknesses.append("Against sentiment")
        
        # 6. MACRO INTELLIGENCE alignment score
        macro_score = 0.5
        
        macro_features = getattr(features, 'macro_features', None)
        risk_sentiment = getattr(features, 'risk_sentiment', None)
        
        if macro_features or risk_sentiment:
            net_exposure = sum(signal.desired_weights.values())
            
            if risk_sentiment:
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
        
        # 7. NEW — URGENCY SCORE: How time-sensitive is this signal?
        urgency_score = self._calculate_urgency(signal, features, regime)
        
        if urgency_score > 0.7:
            strengths.append(f"High urgency signal ({urgency_score:.0%})")
        
        # 8. NEW — OPPORTUNITY MAGNITUDE: How big is the expected move?
        opportunity_score = self._calculate_opportunity_magnitude(signal)
        
        if opportunity_score > 0.7:
            strengths.append(f"Large opportunity ({opportunity_score:.0%})")
        elif opportunity_score < 0.3:
            weaknesses.append(f"Small expected move")
        
        # Composite score (performance-aware weights)
        total_score = (
            w['alpha'] * 0.9 * alpha_score +
            w['regime'] * 0.9 * regime_fit_score +
            w['diversification'] * diversification_score +
            w['drawdown'] * drawdown_score +
            w['sentiment'] * 0.8 * sentiment_score +
            0.12 * macro_score +
            w['urgency'] * urgency_score +
            w['opportunity'] * opportunity_score
        )
        
        # Multiply by confidence
        total_score *= signal.confidence
        
        # Conviction score: composite of confidence + alpha + urgency
        conviction_score = (signal.confidence * 0.4 + alpha_score * 0.3 + urgency_score * 0.3)
        
        # Build rationale
        rationale = self._build_rationale(
            signal, alpha_score, regime_fit_score, diversification_score,
            urgency_score=urgency_score, opportunity_score=opportunity_score,
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
            urgency_score=urgency_score,
            opportunity_score=opportunity_score,
            conviction_score=conviction_score,
        )
    
    def _calculate_urgency(
        self,
        signal: SignalOutput,
        features: Features,
        regime: Optional[MarketRegime],
    ) -> float:
        """
        Calculate how time-sensitive this signal is (0-1).
        
        High urgency when:
        - Intraday strategy with short holding period
        - High confidence (the edge may decay quickly)
        - Extreme regime (volatility creates fleeting opportunities)
        - News-driven signals (information decays rapidly)
        - Large expected returns (big moves happen fast)
        """
        urgency = 0.3  # Base urgency
        
        # 1. Holding period urgency: shorter = more urgent
        if signal.holding_period_minutes > 0:
            if signal.holding_period_minutes <= 30:
                urgency += 0.35  # Very short-term: very urgent
            elif signal.holding_period_minutes <= 120:
                urgency += 0.20  # Intraday: moderately urgent
            elif signal.holding_period_minutes <= 480:
                urgency += 0.10  # Same-day: somewhat urgent
        
        # 2. Confidence urgency: high confidence edges decay fast
        if signal.confidence > 0.75:
            urgency += 0.15
        elif signal.confidence > 0.60:
            urgency += 0.08
        
        # 3. Regime urgency: extreme regimes create fleeting opportunities
        if regime:
            if regime.volatility == VolatilityRegime.EXTREME:
                urgency += 0.15  # Extreme vol = act NOW or miss it
            elif regime.volatility == VolatilityRegime.HIGH:
                urgency += 0.08
            
            if regime.trend in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
                urgency += 0.05  # Strong trends: jumping on early matters
        
        # 4. Expected return urgency: big moves are time-sensitive
        if abs(signal.expected_return) > 0.15:
            urgency += 0.10
        elif abs(signal.expected_return) > 0.08:
            urgency += 0.05
        
        # 5. News/sentiment urgency: check if strategy is sentiment-driven
        strategy_name = signal.strategy_name.lower()
        if 'news' in strategy_name or 'sentiment' in strategy_name or 'event' in strategy_name:
            urgency += 0.15  # News edges decay within hours
        
        return min(1.0, urgency)
    
    def _calculate_opportunity_magnitude(self, signal: SignalOutput) -> float:
        """
        Calculate how large the expected move is (0-1).
        
        Maps absolute expected return to a 0-1 scale, rewarding
        strategies that see BIG moves rather than marginal ones.
        """
        abs_return = abs(signal.expected_return)
        
        # Non-linear mapping: reward big expected moves disproportionately
        # 0% → 0.0, 5% → 0.25, 10% → 0.50, 20% → 0.75, 30%+ → 1.0
        if abs_return >= 0.30:
            opportunity = 1.0
        elif abs_return >= 0.20:
            opportunity = 0.75 + 0.25 * ((abs_return - 0.20) / 0.10)
        elif abs_return >= 0.10:
            opportunity = 0.50 + 0.25 * ((abs_return - 0.10) / 0.10)
        elif abs_return >= 0.05:
            opportunity = 0.25 + 0.25 * ((abs_return - 0.05) / 0.05)
        else:
            opportunity = abs_return / 0.05 * 0.25
        
        # Boost if Sharpe-like ratio is high (big return with controlled risk)
        if signal.risk_estimate > 0:
            efficiency = abs_return / signal.risk_estimate
            if efficiency > 2.0:
                opportunity = min(1.0, opportunity * 1.2)  # 20% boost for efficient alpha
        
        # Bonus for concentrated conviction (fewer, bigger positions)
        max_weight = max(abs(w) for w in signal.desired_weights.values()) if signal.desired_weights else 0
        if max_weight > 0.10:
            opportunity = min(1.0, opportunity + 0.05)  # Concentrated = big conviction
        
        return min(1.0, max(0.0, opportunity))
    
    def _apply_conviction_escalation(
        self,
        strategy_scores: Dict[str, StrategyScore],
        signals: Dict[str, SignalOutput],
        performance_state: str,
    ) -> Dict[str, StrategyScore]:
        """
        Boost scores when multiple high-conviction strategies agree on the same assets.
        
        This is DIFFERENT from the ensemble's confluence boost:
        - Ensemble boosts POSITION WEIGHTS for agreeing signals
        - This boosts STRATEGY SCORES so high-agreement strategies win the debate
        
        When in offensive performance states, the escalation is stronger.
        """
        if len(signals) < 2:
            return strategy_scores
        
        # Build per-asset consensus: how many high-confidence strategies agree?
        asset_consensus: Dict[str, Dict[str, int]] = {}  # asset -> {long: n, short: n}
        high_confidence_threshold = 0.55
        
        for name, signal in signals.items():
            if signal.confidence < high_confidence_threshold:
                continue
            for asset, weight in signal.desired_weights.items():
                if abs(weight) < 0.01:
                    continue
                if asset not in asset_consensus:
                    asset_consensus[asset] = {'long': 0, 'short': 0, 'strategies': []}
                if weight > 0:
                    asset_consensus[asset]['long'] += 1
                else:
                    asset_consensus[asset]['short'] += 1
                asset_consensus[asset]['strategies'].append(name)
        
        # Find high-consensus assets (3+ strategies agree on direction)
        consensus_assets = {}
        for asset, counts in asset_consensus.items():
            if counts['long'] >= 3 and counts['short'] == 0:
                consensus_assets[asset] = counts['strategies']
            elif counts['short'] >= 3 and counts['long'] == 0:
                consensus_assets[asset] = counts['strategies']
        
        if not consensus_assets:
            return strategy_scores
        
        # Calculate conviction boost based on performance state
        if performance_state in self.OFFENSIVE_STATES:
            base_boost = 0.12  # 12% boost per consensus asset when winning
        else:
            base_boost = 0.06  # 6% boost normally
        
        # Boost strategies participating in consensus
        boosted_strategies = set()
        for asset, participating in consensus_assets.items():
            for strat_name in participating:
                boosted_strategies.add(strat_name)
        
        for name in boosted_strategies:
            if name in strategy_scores:
                score = strategy_scores[name]
                # Count how many consensus assets this strategy participates in
                n_consensus = sum(1 for asset, strats in consensus_assets.items() if name in strats)
                boost = min(0.25, base_boost * n_consensus)  # Cap at 25%
                
                old_total = score.total_score
                score.total_score = min(1.0, score.total_score * (1.0 + boost))
                score.conviction_score = min(1.0, score.conviction_score + boost)
                
                if boost > 0.05:
                    score.strengths.append(
                        f"Conviction escalation: {n_consensus} consensus assets (+{boost:.0%})"
                    )
                    logger.debug(f"Conviction escalation: {name} {old_total:.2f} → {score.total_score:.2f}")
        
        return strategy_scores
    
    def _build_rationale(
        self,
        signal: SignalOutput,
        alpha: float,
        regime: float,
        div: float,
        urgency_score: float = 0.5,
        opportunity_score: float = 0.5,
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
        
        # Urgency & opportunity
        if urgency_score > 0.7:
            parts.append("URGENT")
        if opportunity_score > 0.7:
            parts.append("BIG opportunity")
        
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
        """Generate debate summary with urgency and opportunity context."""
        parts = []
        
        # Performance mode
        if transcript.performance_mode in self.OFFENSIVE_STATES:
            parts.append(f"MODE: {transcript.performance_mode} (offensive bias)")
        elif transcript.performance_mode in self.DEFENSIVE_STATES:
            parts.append(f"MODE: {transcript.performance_mode} (defensive bias)")
        
        # Top strategy
        if transcript.winning_strategies:
            parts.append(f"Preferred: {transcript.winning_strategies[0]}")
        
        # Urgency & opportunity
        if transcript.aggregate_urgency > 0.65:
            parts.append(f"URGENCY: HIGH ({transcript.aggregate_urgency:.0%})")
        if transcript.aggregate_opportunity > 0.65:
            parts.append(f"OPPORTUNITY: LARGE ({transcript.aggregate_opportunity:.0%})")
        
        # Key insight
        if transcript.agreements:
            parts.append(f"Consensus: {transcript.agreements[0]}")
        elif transcript.disagreements:
            parts.append(f"Disagreement: {transcript.disagreements[0]}")
        
        # Risk warning
        if transcript.top_risks:
            parts.append(f"Risk: {transcript.top_risks[0]}")
        
        return " | ".join(parts)
