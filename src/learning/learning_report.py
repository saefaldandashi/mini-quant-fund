"""
Learning Report Generator - Creates comprehensive learning reports.

This module generates detailed reports that explain:
1. What the system has learned
2. How it's using these learnings
3. How learnings influence future strategies

The reports are designed to be human-readable and actionable.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class LearningInsight:
    """A single learning insight."""
    category: str  # 'pattern', 'strategy', 'execution', 'mistake'
    title: str
    description: str
    confidence: float
    impact: str  # 'high', 'medium', 'low'
    action: str  # What we're doing about it
    evidence: List[str]  # Supporting data points


@dataclass
class StrategyLearning:
    """Learning summary for a specific strategy."""
    strategy_name: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    current_weight: float
    weight_trend: str  # 'increasing', 'decreasing', 'stable'
    best_regime: Optional[str]
    worst_regime: Optional[str]
    recommendation: str


@dataclass
class PatternLearning:
    """A learned market pattern."""
    pattern_id: str
    description: str
    times_observed: int
    success_rate: float
    avg_return: float
    recommended_strategies: List[str]
    strategies_to_avoid: List[str]
    current_status: str  # 'active', 'dormant'


@dataclass
class ExecutionLearning:
    """Execution-related learnings."""
    total_trades: int
    avg_slippage_bps: float
    symbols_with_learning: int
    best_execution_symbols: List[str]
    worst_execution_symbols: List[str]
    cost_savings_estimate: float


class LearningReportGenerator:
    """Generates comprehensive learning reports."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = outputs_dir
        self.logger = logging.getLogger(__name__)
    
    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive learning report.
        
        Returns:
            Dict containing all learning insights and recommendations
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_version": "2.0",
            "summary": self._generate_summary(),
            "strategy_learnings": self._analyze_strategy_performance(),
            "pattern_learnings": self._analyze_patterns(),
            "execution_learnings": self._analyze_execution(),
            "mistake_analysis": self._analyze_mistakes(),
            "key_insights": self._extract_key_insights(),
            "future_strategy_influence": self._explain_future_influence(),
            "recommendations": self._generate_recommendations(),
            "learning_evolution": self._get_learning_evolution(),
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary of learning state."""
        trade_stats = self._load_json("trade_memory.json")
        pattern_data = self._load_json("patterns.json")
        weights_data = self._load_json("learned_weights.json")
        perf_data = self._load_json("strategy_performance.json")
        
        trades = trade_stats.get("trades", [])
        total_trades = len(trades)
        
        # Calculate win rate
        trades_with_pnl = [t for t in trades if t.get("pnl_percent") is not None]
        wins = sum(1 for t in trades_with_pnl if t.get("pnl_percent", 0) > 0)
        losses = sum(1 for t in trades_with_pnl if t.get("pnl_percent", 0) < 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
        
        # Calculate total PnL
        total_pnl = sum(t.get("pnl_amount", 0) or 0 for t in trades)
        
        # Count active patterns
        patterns = pattern_data.get("patterns", {})
        active_patterns = sum(1 for p in patterns.values() if p.get("times_observed", 0) >= 5)
        
        # Learning maturity
        if total_trades < 20:
            maturity = "Early Stage"
            maturity_desc = "Still gathering initial data. Learnings have low confidence."
        elif total_trades < 100:
            maturity = "Developing"
            maturity_desc = "Building statistical significance. Patterns emerging."
        elif total_trades < 500:
            maturity = "Maturing"
            maturity_desc = "Strong confidence in learnings. Actively influencing decisions."
        else:
            maturity = "Mature"
            maturity_desc = "High confidence. Learnings are primary decision drivers."
        
        return {
            "total_trades_analyzed": total_trades,
            "trades_with_outcome": len(trades_with_pnl),
            "pnl_tracking_rate": len(trades_with_pnl) / total_trades if total_trades > 0 else 0,
            "overall_win_rate": win_rate,
            "total_pnl": total_pnl,
            "patterns_discovered": len(patterns),
            "active_patterns": active_patterns,
            "strategies_tracked": len(weights_data.get("weights", {})),
            "learning_maturity": maturity,
            "maturity_description": maturity_desc,
            "current_learning_influence": self._calculate_learning_influence(total_trades, win_rate),
        }
    
    def _calculate_learning_influence(self, trades: int, win_rate: float) -> Dict[str, Any]:
        """Calculate how much influence learning has on decisions."""
        if trades < 10:
            influence_pct = 20
        elif trades < 30:
            influence_pct = 35
        elif trades < 100:
            influence_pct = 50
        else:
            influence_pct = 65
        
        # Adjust based on win rate
        if win_rate > 0.55 and trades >= 30:
            influence_pct = min(70, influence_pct * 1.15)
        elif win_rate < 0.45 and trades >= 30:
            influence_pct = max(20, influence_pct * 0.85)
        
        return {
            "current_percentage": influence_pct,
            "debate_weight": 100 - influence_pct,
            "explanation": f"Learning influences {influence_pct}% of strategy weight decisions. "
                          f"The debate engine provides the remaining {100-influence_pct}%. "
                          f"This ratio will {'increase' if influence_pct < 65 else 'stabilize'} "
                          f"as more data is collected."
        }
    
    def _analyze_strategy_performance(self) -> List[Dict[str, Any]]:
        """Analyze and report on each strategy's learned performance."""
        perf_data = self._load_json("strategy_performance.json")
        weights_data = self._load_json("learned_weights.json")
        
        metrics = perf_data.get("metrics", {})
        weights = weights_data.get("weights", {})
        
        strategy_learnings = []
        
        for name, metric in metrics.items():
            total_pred = metric.get("total_predictions", 0)
            correct_pred = metric.get("correct_predictions", 0)
            accuracy = correct_pred / total_pred if total_pred > 0 else 0.5
            
            # Get weight info
            weight_info = weights.get(name, {})
            current_weight = weight_info.get("base_weight", 0.1)
            times_selected = weight_info.get("times_selected", 0)
            regime_weights = weight_info.get("regime_weights", {})
            
            # Determine best/worst regimes
            if regime_weights:
                best_regime = max(regime_weights.items(), key=lambda x: x[1])[0]
                worst_regime = min(regime_weights.items(), key=lambda x: x[1])[0]
            else:
                best_regime = None
                worst_regime = None
            
            # Determine weight trend
            if times_selected < 5:
                weight_trend = "insufficient_data"
            elif weight_info.get("cumulative_reward", 0) > 0.1:
                weight_trend = "increasing"
            elif weight_info.get("cumulative_reward", 0) < -0.1:
                weight_trend = "decreasing"
            else:
                weight_trend = "stable"
            
            # Generate recommendation
            if accuracy > 0.55 and total_pred >= 20:
                recommendation = f"Strong performer. Consider increasing weight in {best_regime or 'all'} regimes."
            elif accuracy < 0.45 and total_pred >= 20:
                recommendation = f"Underperforming. Reduce exposure, especially in {worst_regime or 'volatile'} conditions."
            elif total_pred < 10:
                recommendation = "Insufficient data. Continue monitoring."
            else:
                recommendation = "Neutral performance. Maintain current weight."
            
            strategy_learnings.append({
                "strategy_name": name,
                "total_predictions": total_pred,
                "correct_predictions": correct_pred,
                "accuracy": accuracy,
                "accuracy_display": f"{accuracy*100:.1f}%",
                "current_weight": current_weight,
                "weight_display": f"{current_weight*100:.1f}%",
                "times_selected": times_selected,
                "weight_trend": weight_trend,
                "best_regime": best_regime,
                "worst_regime": worst_regime,
                "regime_weights": regime_weights,
                "recommendation": recommendation,
            })
        
        # Sort by accuracy (highest first)
        strategy_learnings.sort(key=lambda x: x["accuracy"], reverse=True)
        
        return strategy_learnings
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze discovered patterns and their impact."""
        pattern_data = self._load_json("patterns.json")
        
        patterns = pattern_data.get("patterns", {})
        observations = pattern_data.get("observations", [])
        
        pattern_list = []
        
        for pattern_id, pattern in patterns.items():
            times_observed = pattern.get("times_observed", 0)
            times_profitable = pattern.get("times_profitable", 0)
            success_rate = times_profitable / times_observed if times_observed > 0 else 0
            
            pattern_list.append({
                "pattern_id": pattern_id,
                "description": pattern.get("description", ""),
                "conditions": pattern.get("conditions", {}),
                "times_observed": times_observed,
                "success_rate": success_rate,
                "success_rate_display": f"{success_rate*100:.1f}%",
                "avg_return": pattern.get("avg_return", 0),
                "avg_return_display": f"{pattern.get('avg_return', 0)*100:.2f}%",
                "confidence": pattern.get("confidence", 0),
                "recommended_action": pattern.get("recommended_action", "hold"),
                "recommended_strategies": pattern.get("recommended_strategies", []),
                "strategies_to_avoid": pattern.get("strategies_to_avoid", []),
                "status": "active" if times_observed >= 5 and pattern.get("confidence", 0) > 0.5 else "dormant",
            })
        
        # Sort by confidence
        pattern_list.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Group by status
        active_patterns = [p for p in pattern_list if p["status"] == "active"]
        dormant_patterns = [p for p in pattern_list if p["status"] == "dormant"]
        
        return {
            "total_patterns": len(patterns),
            "active_patterns": len(active_patterns),
            "dormant_patterns": len(dormant_patterns),
            "total_observations": len(observations),
            "patterns": pattern_list,
            "active_pattern_details": active_patterns[:5],
            "explanation": self._explain_pattern_usage(active_patterns),
        }
    
    def _explain_pattern_usage(self, active_patterns: List[Dict]) -> str:
        """Explain how patterns are being used."""
        if not active_patterns:
            return "No patterns have achieved sufficient confidence yet. The system needs more observations."
        
        explanation = f"Currently using {len(active_patterns)} active patterns to influence trading decisions:\n\n"
        
        for p in active_patterns[:3]:
            explanation += f"• **{p['description']}**: When detected, "
            if p['recommended_action'] == 'increase_exposure':
                explanation += "increases exposure via " + ", ".join(p['recommended_strategies'][:2])
            elif p['recommended_action'] == 'reduce_exposure':
                explanation += "reduces exposure and avoids " + ", ".join(p['strategies_to_avoid'][:2])
            else:
                explanation += "maintains current positions"
            explanation += f" (confidence: {p['confidence']*100:.0f}%)\n"
        
        return explanation
    
    def _analyze_execution(self) -> Dict[str, Any]:
        """Analyze execution quality learnings."""
        cost_data_path = os.path.join(self.outputs_dir, "transaction_cost_learned.json")
        
        if os.path.exists(cost_data_path):
            with open(cost_data_path, 'r') as f:
                cost_data = json.load(f)
        else:
            cost_data = {"symbol_data": {}}
        
        symbol_data = cost_data.get("symbol_data", {})
        
        if not symbol_data:
            return {
                "status": "learning",
                "message": "Execution learning requires more trade data.",
                "symbols_with_learning": 0,
                "total_trades_analyzed": 0,
            }
        
        # Analyze symbols
        symbols_sorted = sorted(
            symbol_data.items(),
            key=lambda x: x[1].get("avg_slippage_bps", 0)
        )
        
        best_symbols = [s[0] for s in symbols_sorted[:5]]
        worst_symbols = [s[0] for s in symbols_sorted[-5:]]
        
        avg_slippage = sum(s.get("avg_slippage_bps", 0) for s in symbol_data.values()) / len(symbol_data) if symbol_data else 0
        
        return {
            "status": "active",
            "symbols_with_learning": len(symbol_data),
            "avg_slippage_bps": avg_slippage,
            "avg_slippage_display": f"{avg_slippage:.1f} bps",
            "best_execution_symbols": best_symbols,
            "worst_execution_symbols": worst_symbols,
            "explanation": f"The system has learned execution characteristics for {len(symbol_data)} symbols. "
                          f"Average slippage is {avg_slippage:.1f} basis points. "
                          f"This learning is used to estimate transaction costs before trades and skip "
                          f"unprofitable trades.",
        }
    
    def _analyze_mistakes(self) -> Dict[str, Any]:
        """Analyze trading mistakes and lessons learned."""
        trade_data = self._load_json("trade_memory.json")
        trades = trade_data.get("trades", [])
        
        # Get losing trades
        losing_trades = [t for t in trades if t.get("pnl_percent") and t.get("pnl_percent", 0) < 0]
        
        if len(losing_trades) < 5:
            return {
                "status": "insufficient_data",
                "message": "Need more losing trades to analyze patterns.",
                "total_losses": len(losing_trades),
            }
        
        # Analyze by regime
        regime_losses = {}
        for t in losing_trades:
            ctx = t.get("market_context", {})
            regime = ctx.get("regime", "unknown") if isinstance(ctx, dict) else "unknown"
            regime_losses[regime] = regime_losses.get(regime, 0) + 1
        
        # Analyze by strategy
        strategy_losses = {}
        for t in losing_trades:
            signals = t.get("strategy_signals", [])
            for sig in signals:
                name = sig.get("strategy_name") or sig.get("name", "unknown")
                if sig.get("debate_score", 0) > 0.3:
                    strategy_losses[name] = strategy_losses.get(name, 0) + 1
        
        worst_regime = max(regime_losses.items(), key=lambda x: x[1])[0] if regime_losses else None
        worst_strategy = max(strategy_losses.items(), key=lambda x: x[1])[0] if strategy_losses else None
        
        # Calculate average loss
        avg_loss = sum(t.get("pnl_percent", 0) for t in losing_trades) / len(losing_trades)
        
        lessons = []
        if worst_regime and worst_regime != "unknown":
            lessons.append({
                "lesson": f"Most losses occur in {worst_regime} regime",
                "action": f"System now reduces exposure by 30% in {worst_regime} conditions",
                "confidence": "high" if regime_losses.get(worst_regime, 0) >= 10 else "medium",
            })
        
        if worst_strategy:
            lessons.append({
                "lesson": f"{worst_strategy} is associated with most losing trades",
                "action": f"Strategy weight is reduced by 15% from learned patterns",
                "confidence": "high" if strategy_losses.get(worst_strategy, 0) >= 5 else "medium",
            })
        
        return {
            "status": "active",
            "total_losses_analyzed": len(losing_trades),
            "avg_loss_percent": avg_loss,
            "avg_loss_display": f"{avg_loss:.2f}%",
            "worst_regime": worst_regime,
            "worst_strategy": worst_strategy,
            "regime_distribution": regime_losses,
            "strategy_distribution": strategy_losses,
            "lessons_learned": lessons,
        }
    
    def _extract_key_insights(self) -> List[Dict[str, Any]]:
        """Extract the most important learning insights."""
        insights = []
        
        # Get data
        summary = self._generate_summary()
        patterns = self._analyze_patterns()
        strategies = self._analyze_strategy_performance()
        mistakes = self._analyze_mistakes()
        
        # Insight 1: Overall performance
        if summary["overall_win_rate"] >= 0.52:
            insights.append({
                "title": "Positive Learning Trend",
                "description": f"Overall win rate of {summary['overall_win_rate']*100:.1f}% indicates strategies are effective.",
                "impact": "high",
                "action": "Increasing learning influence to capitalize on success.",
            })
        elif summary["overall_win_rate"] < 0.48 and summary["total_trades_analyzed"] >= 30:
            insights.append({
                "title": "Underperformance Detected",
                "description": f"Win rate of {summary['overall_win_rate']*100:.1f}% is below target.",
                "impact": "high",
                "action": "Reducing learning influence, relying more on debate engine.",
            })
        
        # Insight 2: Best performing strategy
        if strategies and strategies[0]["total_predictions"] >= 20:
            best = strategies[0]
            insights.append({
                "title": f"Top Strategy: {best['strategy_name']}",
                "description": f"Achieving {best['accuracy_display']} accuracy with {best['total_predictions']} predictions.",
                "impact": "high",
                "action": f"Weight increased to {best['weight_display']}. Prioritized in {best['best_regime'] or 'all'} regimes.",
            })
        
        # Insight 3: Active patterns
        active = patterns.get("active_pattern_details", [])
        if active:
            top_pattern = active[0]
            insights.append({
                "title": f"Active Pattern: {top_pattern['description']}",
                "description": f"Observed {top_pattern['times_observed']} times with {top_pattern['success_rate_display']} success rate.",
                "impact": "medium",
                "action": f"When detected: {top_pattern['recommended_action'].replace('_', ' ')}.",
            })
        
        # Insight 4: Mistake learning
        if mistakes.get("status") == "active" and mistakes.get("lessons_learned"):
            lesson = mistakes["lessons_learned"][0]
            insights.append({
                "title": f"Mistake Pattern: {lesson['lesson']}",
                "description": f"Identified from {mistakes['total_losses_analyzed']} losing trades.",
                "impact": "medium",
                "action": lesson['action'],
            })
        
        return insights
    
    def _explain_future_influence(self) -> Dict[str, Any]:
        """Explain how learnings will influence future strategies."""
        summary = self._generate_summary()
        strategies = self._analyze_strategy_performance()
        patterns = self._analyze_patterns()
        
        influence_explanation = {
            "overview": (
                "The learning system continuously refines trading decisions based on historical outcomes. "
                "Here's how each component influences future strategies:"
            ),
            "components": [
                {
                    "name": "Strategy Weight Adjustment",
                    "current_state": f"Currently influencing {summary['current_learning_influence']['current_percentage']}% of weight allocation",
                    "how_it_works": (
                        "Strategies with higher win rates receive higher weights. "
                        "The system tracks each strategy's performance across market regimes "
                        "and adjusts weights dynamically."
                    ),
                    "example": self._get_weight_example(strategies),
                },
                {
                    "name": "Pattern Recognition",
                    "current_state": f"{patterns['active_patterns']} patterns actively influencing decisions",
                    "how_it_works": (
                        "When market conditions match a learned pattern, the system "
                        "automatically adjusts strategy selection and exposure levels."
                    ),
                    "example": self._get_pattern_example(patterns),
                },
                {
                    "name": "Mistake Avoidance",
                    "current_state": "Actively learning from losing trades",
                    "how_it_works": (
                        "The system analyzes losing trades to identify common conditions "
                        "and reduces exposure when those conditions are detected."
                    ),
                    "example": "If losses cluster in risk_off regimes, exposure is automatically reduced by 30% when that regime is detected.",
                },
                {
                    "name": "Execution Optimization",
                    "current_state": "Learning slippage patterns for each symbol",
                    "how_it_works": (
                        "Before each trade, estimated transaction costs are calculated using "
                        "learned slippage data. Trades where costs exceed expected benefit are skipped."
                    ),
                    "example": "A signal with 0.5% expected return will be skipped if learned slippage for that symbol is 0.8%.",
                },
            ],
            "feedback_loop": (
                "All trade outcomes feed back into the system. Each rebalance cycle:\n"
                "1. Records all signals and trade decisions\n"
                "2. Tracks actual vs expected returns\n"
                "3. Updates strategy weights and pattern confidence\n"
                "4. Adjusts future decision-making accordingly"
            ),
        }
        
        return influence_explanation
    
    def _get_weight_example(self, strategies: List[Dict]) -> str:
        """Generate an example of weight adjustment."""
        if not strategies:
            return "Collecting data to provide examples."
        
        best = strategies[0]
        worst = strategies[-1] if len(strategies) > 1 else None
        
        example = f"• {best['strategy_name']}: {best['accuracy_display']} accuracy → Weight: {best['weight_display']}"
        if worst and worst != best:
            example += f"\n• {worst['strategy_name']}: {worst['accuracy_display']} accuracy → Weight: {worst['weight_display']}"
        
        return example
    
    def _get_pattern_example(self, patterns: Dict) -> str:
        """Generate an example of pattern influence."""
        active = patterns.get("active_pattern_details", [])
        if not active:
            return "No active patterns yet. Continue trading to discover patterns."
        
        p = active[0]
        return (
            f"Pattern: '{p['description']}'\n"
            f"When detected → {p['recommended_action'].replace('_', ' ')}\n"
            f"Boost: {', '.join(p['recommended_strategies'][:2])}\n"
            f"Avoid: {', '.join(p['strategies_to_avoid'][:2])}"
        )
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        summary = self._generate_summary()
        
        # Recommendation based on data sufficiency
        if summary["pnl_tracking_rate"] < 0.5:
            recommendations.append({
                "priority": "high",
                "category": "data_quality",
                "title": "Improve Outcome Tracking",
                "description": f"Only {summary['pnl_tracking_rate']*100:.0f}% of trades have PnL tracked.",
                "action": "Run 'Update Outcomes' more frequently to improve learning data quality.",
            })
        
        if summary["total_trades_analyzed"] < 50:
            recommendations.append({
                "priority": "medium",
                "category": "data_volume",
                "title": "Increase Trade Volume",
                "description": f"Only {summary['total_trades_analyzed']} trades analyzed. Need 50+ for reliable patterns.",
                "action": "Continue trading to build statistical significance.",
            })
        
        if summary["learning_maturity"] in ["Early Stage", "Developing"]:
            recommendations.append({
                "priority": "medium",
                "category": "learning_maturity",
                "title": "Patience Required",
                "description": "Learning system is still developing confidence.",
                "action": "Learning influence will automatically increase as data accumulates.",
            })
        
        return recommendations
    
    def _get_learning_evolution(self) -> Dict[str, Any]:
        """Track how learning has evolved over time."""
        learning_dir = os.path.join(self.outputs_dir, "learning")
        
        if not os.path.exists(learning_dir):
            return {"status": "no_history"}
        
        # Load performance trends if available
        trends_path = os.path.join(learning_dir, "performance_trends.json")
        if os.path.exists(trends_path):
            with open(trends_path, 'r') as f:
                trends = json.load(f)
        else:
            trends = {}
        
        return {
            "status": "active",
            "trends_available": bool(trends),
            "last_updated": trends.get("last_updated"),
            "improvement_metrics": trends.get("improvement_metrics", {}),
        }
    
    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file from outputs directory."""
        path = os.path.join(self.outputs_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def generate_html_report(self) -> str:
        """Generate an HTML version of the learning report."""
        report = self.generate_full_report()
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning System Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --border: #30363d;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 40px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: var(--accent-blue); margin-bottom: 10px; font-size: 2.5em; }}
        h2 {{ color: var(--text-primary); margin: 40px 0 20px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }}
        h3 {{ color: var(--accent-blue); margin: 20px 0 10px; }}
        .subtitle {{ color: var(--text-secondary); margin-bottom: 30px; }}
        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-blue);
        }}
        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}
        .warning {{ color: var(--accent-yellow); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: var(--bg-tertiary); color: var(--accent-blue); }}
        tr:hover {{ background: var(--bg-tertiary); }}
        .insight-card {{
            background: var(--bg-tertiary);
            border-left: 4px solid var(--accent-blue);
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        .insight-title {{ color: var(--text-primary); font-weight: bold; margin-bottom: 5px; }}
        .insight-action {{ color: var(--accent-green); font-style: italic; margin-top: 10px; }}
        .pattern-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            margin: 2px;
        }}
        .badge-active {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }}
        .badge-dormant {{ background: rgba(139, 148, 158, 0.2); color: var(--text-secondary); }}
        .progress-bar {{
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: var(--accent-blue);
            transition: width 0.3s;
        }}
        .recommendation {{
            background: var(--bg-tertiary);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .recommendation.high {{ border-left: 4px solid var(--accent-red); }}
        .recommendation.medium {{ border-left: 4px solid var(--accent-yellow); }}
        .recommendation.low {{ border-left: 4px solid var(--accent-green); }}
        pre {{
            background: var(--bg-tertiary);
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Learning System Report</h1>
        <p class="subtitle">Generated: {report['generated_at']}</p>
        
        <h2>📊 Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-value">{report['summary']['total_trades_analyzed']}</div>
                <div class="metric-label">Trades Analyzed</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if report['summary']['overall_win_rate'] >= 0.5 else 'negative'}">{report['summary']['overall_win_rate']*100:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report['summary']['active_patterns']}</div>
                <div class="metric-label">Active Patterns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report['summary']['current_learning_influence']['current_percentage']}%</div>
                <div class="metric-label">Learning Influence</div>
            </div>
        </div>
        
        <div class="card">
            <h3>Learning Maturity: {report['summary']['learning_maturity']}</h3>
            <p>{report['summary']['maturity_description']}</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {min(100, report['summary']['total_trades_analyzed'] / 5)}%"></div>
            </div>
            <p class="metric-label">{report['summary']['trades_with_outcome']} of {report['summary']['total_trades_analyzed']} trades have outcomes tracked ({report['summary']['pnl_tracking_rate']*100:.0f}%)</p>
        </div>
        
        <h2>🎯 Key Insights</h2>
        {''.join(f'''
        <div class="insight-card">
            <div class="insight-title">{insight['title']}</div>
            <p>{insight['description']}</p>
            <p class="insight-action">→ {insight['action']}</p>
        </div>
        ''' for insight in report['key_insights'])}
        
        <h2>📈 Strategy Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Accuracy</th>
                    <th>Predictions</th>
                    <th>Weight</th>
                    <th>Trend</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>
                {''.join(f'''
                <tr>
                    <td>{s['strategy_name']}</td>
                    <td class="{'positive' if s['accuracy'] >= 0.52 else 'negative' if s['accuracy'] < 0.48 else ''}">{s['accuracy_display']}</td>
                    <td>{s['total_predictions']}</td>
                    <td>{s['weight_display']}</td>
                    <td>{'📈' if s['weight_trend'] == 'increasing' else '📉' if s['weight_trend'] == 'decreasing' else '➡️'}</td>
                    <td style="font-size: 0.9em;">{s['recommendation'][:50]}...</td>
                </tr>
                ''' for s in report['strategy_learnings'][:10])}
            </tbody>
        </table>
        
        <h2>🔍 Pattern Learnings</h2>
        <div class="card">
            <p><strong>{report['pattern_learnings']['active_patterns']}</strong> active patterns from <strong>{report['pattern_learnings']['total_observations']}</strong> observations</p>
            <pre>{report['pattern_learnings']['explanation']}</pre>
        </div>
        
        <h2>⚙️ How Learning Influences Future Strategies</h2>
        <div class="card">
            <p>{report['future_strategy_influence']['overview']}</p>
        </div>
        {''.join(f'''
        <div class="card">
            <h3>{comp['name']}</h3>
            <p><strong>Current State:</strong> {comp['current_state']}</p>
            <p>{comp['how_it_works']}</p>
            <pre>{comp['example']}</pre>
        </div>
        ''' for comp in report['future_strategy_influence']['components'])}
        
        <h2>📋 Recommendations</h2>
        {''.join(f'''
        <div class="recommendation {rec['priority']}">
            <h4>{rec['title']}</h4>
            <p>{rec['description']}</p>
            <p class="insight-action">→ {rec['action']}</p>
        </div>
        ''' for rec in report['recommendations'])}
        
        <div style="margin-top: 60px; padding-top: 20px; border-top: 1px solid var(--border); color: var(--text-secondary); text-align: center;">
            <p>Mini Quant Fund - Learning Report v{report['report_version']}</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def save_report(self, output_path: Optional[str] = None) -> str:
        """Save the report to a file."""
        if output_path is None:
            output_path = os.path.join(
                self.outputs_dir, 
                "reports", 
                f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        html = self.generate_html_report()
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
