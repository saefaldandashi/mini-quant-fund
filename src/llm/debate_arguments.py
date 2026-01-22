"""
LLM Debate Argument Generator - Generates sophisticated debate arguments.

Uses LLM to generate:
1. Support arguments - Why a strategy's proposal is good
2. Attack arguments - Why a competitor's proposal is weak
3. Rebuttal arguments - Defense against attacks
4. Trade explanations - Why we're making this trade
"""
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .llm_service import LLMService

logger = logging.getLogger(__name__)


@dataclass
class DebateArgument:
    """A structured debate argument."""
    argument_type: str  # support, attack, rebuttal
    strategy: str
    target: Optional[str]  # For attacks
    argument: str
    strength: float  # 0-1
    key_points: List[str]
    market_evidence: List[str]
    confidence: float


class LLMDebateArgumentGenerator:
    """
    Generates sophisticated debate arguments using LLM.
    
    Only uses LLM for complex debates where rule-based arguments
    are insufficient.
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
    ):
        self.llm_service = llm_service
        
        # Stats
        self.arguments_generated = 0
        self.llm_calls = 0
        
        # Recent trades context (set externally)
        self.recent_trades: List[Dict] = []
    
    def set_recent_trades(self, trades: List[Dict]) -> None:
        """
        Set recent trades context for LLM prompts.
        
        Args:
            trades: List of recent trade dicts with:
                - symbol, side, quantity, entry_price, pnl, was_profitable
        """
        self.recent_trades = trades[:10]  # Keep last 10
    
    def _format_recent_trades(self) -> str:
        """Format recent trades for prompt injection."""
        if not self.recent_trades:
            return "No recent trades."
        
        lines = ["RECENT TRADE HISTORY (learn from these):"]
        for trade in self.recent_trades[:10]:
            symbol = trade.get('symbol', '?')
            side = trade.get('side', '?')
            pnl = trade.get('pnl_percent', 0) or 0
            was_win = trade.get('was_profitable', False)
            strategy = trade.get('entry_strategy', 'unknown')
            outcome = "✓ WIN" if was_win else "✗ LOSS"
            lines.append(f"  {outcome}: {side.upper()} {symbol} ({strategy}) → {pnl:+.1%}")
        
        return "\n".join(lines)
    
    def generate_support_argument(
        self,
        strategy_name: str,
        signal: Dict,
        features: Dict,
        regime: str = "unknown",
    ) -> DebateArgument:
        """Generate a support argument for a strategy's proposal."""
        
        # Build context
        top_positions = sorted(
            signal.get('desired_weights', {}).items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        position_str = ", ".join([f"{t}: {w:.1%}" for t, w in top_positions])
        
        # Try LLM if available
        if self.llm_service and self.llm_service.is_available():
            return self._generate_support_llm(
                strategy_name, signal, features, regime, position_str
            )
        
        # Fall back to rule-based
        return self._generate_support_rules(
            strategy_name, signal, features, regime, position_str
        )
    
    def _generate_support_rules(
        self,
        strategy_name: str,
        signal: Dict,
        features: Dict,
        regime: str,
        position_str: str,
    ) -> DebateArgument:
        """Generate support argument using rules."""
        
        confidence = signal.get('confidence', 0.5)
        exp_return = signal.get('expected_return', 0)
        
        # Build argument based on strategy type
        if "Momentum" in strategy_name:
            argument = f"Momentum signals are strong with expected return of {exp_return:.1%}. Current market regime ({regime}) supports trend-following."
            key_points = ["Strong price trends", "Low reversal probability", "Historical momentum persistence"]
        
        elif "MeanReversion" in strategy_name:
            argument = f"Mean reversion opportunity detected. Prices have deviated from fair value, expecting reversion with {exp_return:.1%} expected return."
            key_points = ["Price deviation from mean", "Historical reversion patterns", "Contrarian opportunity"]
        
        elif "Volatility" in strategy_name:
            argument = f"Volatility regime suggests optimal positioning. Adjusting exposure for risk-adjusted returns."
            key_points = ["Volatility regime identified", "Risk-adjusted positioning", "Dynamic hedging"]
        
        elif "Sentiment" in strategy_name:
            argument = f"News sentiment supports this positioning. Confidence: {confidence:.0%}."
            key_points = ["Positive sentiment signals", "News flow confirmation", "Sentiment momentum"]
        
        elif "RiskParity" in strategy_name:
            argument = f"Risk parity allocation optimizes for diversification. Balanced risk across assets."
            key_points = ["Risk diversification", "Correlation benefits", "Drawdown protection"]
        
        else:
            argument = f"{strategy_name} proposes positions with {confidence:.0%} confidence and {exp_return:.1%} expected return."
            key_points = ["Strategy-specific signals", "Quantitative analysis", "Historical validation"]
        
        return DebateArgument(
            argument_type="support",
            strategy=strategy_name,
            target=None,
            argument=argument,
            strength=confidence,
            key_points=key_points,
            market_evidence=[f"Current regime: {regime}", f"Expected return: {exp_return:.1%}"],
            confidence=confidence,
        )
    
    def _generate_support_llm(
        self,
        strategy_name: str,
        signal: Dict,
        features: Dict,
        regime: str,
        position_str: str,
    ) -> DebateArgument:
        """Generate support argument using LLM."""
        
        self.llm_calls += 1
        
        # Include recent trades for learning context
        trades_context = self._format_recent_trades()
        
        prompt = f"""You are a portfolio manager defending your trading strategy in an investment committee debate.

STRATEGY: {strategy_name}
PROPOSED POSITIONS: {position_str}
CONFIDENCE: {signal.get('confidence', 0.5):.0%}
EXPECTED RETURN: {signal.get('expected_return', 0):.1%}
CURRENT REGIME: {regime}

{trades_context}

Learn from recent trades above. If similar positions lost money, explain why this time is different.
Generate a compelling 2-3 sentence argument defending this position. Include:
1. Why now is the right time (considering past trade outcomes)
2. What market evidence supports this
3. Why the risk/reward is attractive

Respond in JSON:
{{
    "argument": "your compelling argument",
    "key_points": ["point 1", "point 2", "point 3"],
    "market_evidence": ["evidence 1", "evidence 2"],
    "confidence": 0.0-1.0
}}"""

        response = self.llm_service.call(prompt, temperature=0.4, max_tokens=300)
        
        if not response:
            return self._generate_support_rules(strategy_name, signal, features, regime, position_str)
        
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            self.arguments_generated += 1
            
            return DebateArgument(
                argument_type="support",
                strategy=strategy_name,
                target=None,
                argument=data.get('argument', 'LLM-generated argument'),
                strength=float(data.get('confidence', 0.7)),
                key_points=data.get('key_points', []),
                market_evidence=data.get('market_evidence', []),
                confidence=float(data.get('confidence', 0.7)),
            )
            
        except Exception as e:
            logger.warning(f"Could not parse LLM support argument: {e}")
            return self._generate_support_rules(strategy_name, signal, features, regime, position_str)
    
    def generate_attack_argument(
        self,
        attacker: str,
        target: str,
        target_signal: Dict,
        regime: str = "unknown",
        attacker_strength: float = 0.5,
    ) -> DebateArgument:
        """Generate an attack argument against a competitor's proposal."""
        
        if self.llm_service and self.llm_service.is_available():
            return self._generate_attack_llm(attacker, target, target_signal, regime, attacker_strength)
        
        return self._generate_attack_rules(attacker, target, target_signal, regime, attacker_strength)
    
    def _generate_attack_rules(
        self,
        attacker: str,
        target: str,
        target_signal: Dict,
        regime: str,
        attacker_strength: float,
    ) -> DebateArgument:
        """Generate attack argument using rules."""
        
        target_confidence = target_signal.get('confidence', 0.5)
        target_return = target_signal.get('expected_return', 0)
        
        # Build attack based on strategy types
        if "Momentum" in target:
            if "MeanReversion" in attacker:
                argument = f"Momentum strategies fail in ranging markets. Current signals suggest mean reversion is more appropriate."
                key_points = ["Momentum decay", "Trend exhaustion", "Reversion opportunity"]
            else:
                argument = f"Momentum signals may be late. Price trends are extended and vulnerable to reversal."
                key_points = ["Late entry risk", "Trend exhaustion", "Crowded trade"]
        
        elif "MeanReversion" in target:
            if "Momentum" in attacker:
                argument = f"Mean reversion is fighting the trend. Strong momentum suggests trends will persist."
                key_points = ["Trend strength", "Early reversion attempt", "Momentum dominance"]
            else:
                argument = f"Mean reversion timing is difficult. Deviations can persist longer than expected."
                key_points = ["Timing risk", "Extended deviations", "Catching falling knives"]
        
        elif "Sentiment" in target:
            argument = f"Sentiment can be noisy. News-driven signals need confirmation from price action."
            key_points = ["Noise in sentiment", "Confirmation needed", "Lagging indicator risk"]
        
        else:
            argument = f"{target}'s proposal has execution risks. Confidence of {target_confidence:.0%} may be overstated."
            key_points = ["Execution risk", "Model uncertainty", "Regime sensitivity"]
        
        return DebateArgument(
            argument_type="attack",
            strategy=attacker,
            target=target,
            argument=argument,
            strength=attacker_strength,
            key_points=key_points,
            market_evidence=[f"Target confidence: {target_confidence:.0%}", f"Current regime: {regime}"],
            confidence=attacker_strength,
        )
    
    def _generate_attack_llm(
        self,
        attacker: str,
        target: str,
        target_signal: Dict,
        regime: str,
        attacker_strength: float,
    ) -> DebateArgument:
        """Generate attack argument using LLM."""
        
        self.llm_calls += 1
        
        target_positions = sorted(
            target_signal.get('desired_weights', {}).items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        position_str = ", ".join([f"{t}: {w:.1%}" for t, w in target_positions])
        
        # Include recent trades for learning context
        trades_context = self._format_recent_trades()
        
        prompt = f"""You are a portfolio manager challenging a competitor's strategy in an investment committee debate.

YOUR STRATEGY: {attacker}
OPPOSING STRATEGY: {target}
OPPONENT'S POSITIONS: {position_str}
OPPONENT'S CONFIDENCE: {target_signal.get('confidence', 0.5):.0%}
CURRENT REGIME: {regime}

{trades_context}

Use recent trade outcomes above to identify patterns of failure in similar positions.
Generate a 2-3 sentence challenge to the opponent's proposal. Be specific about:
1. Why their approach is flawed in current conditions (cite recent losses if relevant)
2. What risks they're overlooking
3. Historical precedent for failure

Respond in JSON:
{{
    "argument": "your challenge",
    "key_weaknesses": ["weakness 1", "weakness 2"],
    "risks_overlooked": ["risk 1", "risk 2"],
    "strength": 0.0-1.0
}}"""

        response = self.llm_service.call(prompt, temperature=0.5, max_tokens=300)
        
        if not response:
            return self._generate_attack_rules(attacker, target, target_signal, regime, attacker_strength)
        
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            self.arguments_generated += 1
            
            return DebateArgument(
                argument_type="attack",
                strategy=attacker,
                target=target,
                argument=data.get('argument', 'LLM-generated attack'),
                strength=float(data.get('strength', attacker_strength)),
                key_points=data.get('key_weaknesses', []),
                market_evidence=data.get('risks_overlooked', []),
                confidence=float(data.get('strength', attacker_strength)),
            )
            
        except Exception as e:
            logger.warning(f"Could not parse LLM attack argument: {e}")
            return self._generate_attack_rules(attacker, target, target_signal, regime, attacker_strength)
    
    def generate_trade_explanation(
        self,
        ticker: str,
        action: str,  # buy, sell
        weight: float,
        signals: Dict[str, Dict],
        sentiment: Optional[Dict] = None,
        regime: str = "unknown",
    ) -> str:
        """Generate a human-readable explanation for a trade."""
        
        if self.llm_service and self.llm_service.is_available():
            return self._generate_explanation_llm(ticker, action, weight, signals, sentiment, regime)
        
        return self._generate_explanation_rules(ticker, action, weight, signals, sentiment, regime)
    
    def _generate_explanation_rules(
        self,
        ticker: str,
        action: str,
        weight: float,
        signals: Dict[str, Dict],
        sentiment: Optional[Dict],
        regime: str,
    ) -> str:
        """Generate trade explanation using rules."""
        
        # Find strategies that recommended this trade
        supporting = []
        for name, sig in signals.items():
            if ticker in sig.get('desired_weights', {}):
                w = sig['desired_weights'][ticker]
                if (action == 'buy' and w > 0) or (action == 'sell' and w < 0):
                    supporting.append(name)
        
        explanation = f"{'Buying' if action == 'buy' else 'Selling'} {ticker} with {weight:.1%} weight. "
        
        if supporting:
            explanation += f"Supported by: {', '.join(supporting)}. "
        
        if sentiment:
            sent_score = sentiment.get('sentiment_score', 0)
            if sent_score > 0.2:
                explanation += f"Positive sentiment ({sent_score:.2f}). "
            elif sent_score < -0.2:
                explanation += f"Negative sentiment ({sent_score:.2f}). "
        
        explanation += f"Current regime: {regime}."
        
        return explanation
    
    def _generate_explanation_llm(
        self,
        ticker: str,
        action: str,
        weight: float,
        signals: Dict[str, Dict],
        sentiment: Optional[Dict],
        regime: str,
    ) -> str:
        """Generate trade explanation using LLM."""
        
        self.llm_calls += 1
        
        supporting = []
        for name, sig in signals.items():
            if ticker in sig.get('desired_weights', {}):
                w = sig['desired_weights'][ticker]
                if (action == 'buy' and w > 0) or (action == 'sell' and w < 0):
                    supporting.append(f"{name} ({sig.get('confidence', 0):.0%} confidence)")
        
        sent_info = ""
        if sentiment:
            sent_info = f"Sentiment: {sentiment.get('sentiment_score', 0):.2f}"
        
        prompt = f"""Explain this trade decision in 2-3 sentences for an investment committee.

TRADE: {'Buy' if action == 'buy' else 'Sell'} {ticker}
WEIGHT: {weight:.1%}
SUPPORTING STRATEGIES: {', '.join(supporting) if supporting else 'Multiple'}
{sent_info}
REGIME: {regime}

Write a clear, professional explanation focusing on:
1. Primary reason for the trade
2. Supporting evidence
3. Risk consideration

Respond with just the explanation, no JSON."""

        response = self.llm_service.call(prompt, temperature=0.3, max_tokens=150)
        
        if not response:
            return self._generate_explanation_rules(ticker, action, weight, signals, sentiment, regime)
        
        self.arguments_generated += 1
        return response.content.strip()
    
    def get_stats(self) -> Dict:
        """Get generator statistics."""
        return {
            'arguments_generated': self.arguments_generated,
            'llm_calls': self.llm_calls,
            'llm_available': self.llm_service is not None and self.llm_service.is_available() if self.llm_service else False,
        }
