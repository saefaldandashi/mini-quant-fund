"""
Parallel LLM Debate Engine

Runs LLM calls concurrently for massive speed improvement.
Instead of sequential 30+ seconds, achieves ~5-8 seconds with full LLM reasoning.
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import time
import threading

logger = logging.getLogger(__name__)


@dataclass
class ParallelDebateResult:
    """Result from parallel debate."""
    strategy_scores: Dict[str, float]
    arguments: List[Dict]
    insights: List[str]
    execution_time_ms: float
    llm_calls_made: int
    llm_calls_parallel: int


class ParallelDebateEngine:
    """
    Debate engine that runs LLM calls in parallel.
    
    Key insight: Each strategy's argument is independent - 
    we don't need to wait for one before starting another.
    """
    
    def __init__(self, llm_service, max_workers: int = 6):
        """
        Args:
            llm_service: LLM service for generating arguments
            max_workers: Maximum parallel LLM calls (default 6)
        """
        self.llm_service = llm_service
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Strategy knowledge for prompts
        self.strategy_strengths = {
            "IntradayMomentum": ["captures quick moves", "fast reaction", "intraday trends"],
            "VWAPReversion": ["mean reversion to fair value", "institutional levels", "liquidity"],
            "VolumeSpike": ["detects unusual activity", "early signals", "momentum ignition"],
            "RelativeStrengthIntraday": ["sector rotation", "relative performance", "pair trades"],
            "OpeningRangeBreakout": ["opening volatility", "direction bias", "range expansion"],
            "QuickMeanReversion": ["oversold bounces", "quick profits", "high win rate"],
            "TimeSeriesMomentum": ["trend following", "risk-adjusted returns", "crisis alpha"],
            "CrossSectionMomentum": ["relative strength", "factor exposure", "diversification"],
            "MeanReversion": ["value opportunities", "contrarian plays", "patience pays"],
            "NewsSentimentEvent": ["real-time news", "sentiment shifts", "event-driven"],
            "CS_Momentum_LS": ["market neutral", "long-short pairs", "reduced beta"],
            "TS_Momentum_LS": ["trend capture", "downside protection", "crisis performance"],
        }
    
    def _llm_available(self) -> bool:
        """Check if LLM is available."""
        return (
            self.llm_service is not None 
            and hasattr(self.llm_service, 'is_available') 
            and self.llm_service.is_available()
        )
    
    def _call_llm_sync(self, prompt: str, system: str) -> Optional[Dict]:
        """Synchronous LLM call for use in thread pool."""
        try:
            response = self.llm_service.call(
                prompt, 
                system=system, 
                temperature=0.4, 
                max_tokens=250  # Reduced for speed
            )
            
            if not response or not response.content:
                return None
            
            # Parse JSON response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content)
            
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None
    
    def _build_debate_prompt(
        self,
        strategy_name: str,
        signal: Any,
        market_context: str,
        debate_type: str = "support",
        target_strategy: str = None,
    ) -> Tuple[str, str]:
        """Build prompt for debate argument."""
        
        # Get top positions
        top_positions = sorted(
            signal.desired_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        positions_str = ", ".join([f"{t}: {w:.1%}" for t, w in top_positions])
        
        strengths = self.strategy_strengths.get(strategy_name, ["market exposure"])
        
        if debate_type == "support":
            prompt = f"""You are {strategy_name} strategy in an investment committee.

MARKET: {market_context}
YOUR POSITIONS: {positions_str}
CONFIDENCE: {signal.confidence:.0%}
STRENGTHS: {', '.join(strengths)}

In 2 sentences: Why is YOUR approach right for THIS market NOW?
Focus on specific market conditions supporting you.

JSON: {{"claim": "your argument", "strength": 0.0-1.0, "key_factor": "main reason"}}"""

        elif debate_type == "critique":
            prompt = f"""You are {strategy_name} critiquing {target_strategy} in an investment committee.

MARKET: {market_context}
YOUR APPROACH: {strategy_name}
OPPONENT: {target_strategy}

In 2 sentences: What risk is {target_strategy} IGNORING in current conditions?

JSON: {{"claim": "your critique", "risk": "specific risk", "strength": 0.0-1.0}}"""
        
        system = "You're a portfolio manager. Be specific, cite market conditions. Respond only in JSON."
        
        return prompt, system
    
    def run_parallel_debate(
        self,
        signals: Dict[str, Any],
        features: Any,
        base_scores: Dict[str, float],
    ) -> ParallelDebateResult:
        """
        Run the full debate with parallel LLM calls.
        
        This is the main optimization - instead of sequential calls,
        we fire all LLM requests simultaneously.
        """
        start_time = time.time()
        
        # Build market context once (shared across all prompts)
        market_context = self._build_market_context(features)
        
        # Collect all LLM tasks
        all_tasks = []
        task_metadata = []
        
        # PHASE 1: Support arguments (all in parallel)
        for name, signal in signals.items():
            prompt, system = self._build_debate_prompt(
                name, signal, market_context, "support"
            )
            all_tasks.append((prompt, system))
            task_metadata.append({"type": "support", "strategy": name})
        
        # PHASE 2: Critiques (top 5 strategies critique each other)
        top_strategies = sorted(
            base_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        for i, (attacker, _) in enumerate(top_strategies):
            for j, (target, _) in enumerate(top_strategies):
                if i != j and abs(i - j) <= 2:  # Only adjacent critiques
                    prompt, system = self._build_debate_prompt(
                        attacker, 
                        signals[attacker], 
                        market_context, 
                        "critique",
                        target_strategy=target
                    )
                    all_tasks.append((prompt, system))
                    task_metadata.append({
                        "type": "critique", 
                        "attacker": attacker, 
                        "target": target
                    })
        
        # Execute all LLM calls in parallel!
        logger.info(f"⚡ Launching {len(all_tasks)} parallel LLM calls...")
        
        results = []
        if self._llm_available() and all_tasks:
            # Submit all tasks to thread pool
            futures = [
                self.executor.submit(self._call_llm_sync, prompt, system)
                for prompt, system in all_tasks
            ]
            
            # Wait for all to complete (with timeout)
            concurrent.futures.wait(futures, timeout=15.0)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=0.1)
                    results.append(result)
                except:
                    results.append(None)
        else:
            results = [None] * len(all_tasks)
        
        # Process results
        arguments = []
        attack_impact = {}
        defense_strength = {}
        insights = []
        
        for i, (meta, result) in enumerate(zip(task_metadata, results)):
            if result is None:
                continue
                
            if meta["type"] == "support":
                strategy = meta["strategy"]
                arguments.append({
                    "type": "support",
                    "strategy": strategy,
                    "claim": result.get("claim", ""),
                    "strength": result.get("strength", 0.5),
                    "key_factor": result.get("key_factor", ""),
                })
                
            elif meta["type"] == "critique":
                attacker = meta["attacker"]
                target = meta["target"]
                strength = result.get("strength", 0.3)
                
                arguments.append({
                    "type": "critique",
                    "attacker": attacker,
                    "target": target,
                    "claim": result.get("claim", ""),
                    "risk": result.get("risk", ""),
                    "strength": strength,
                })
                
                # Track attack impact
                attack_impact[target] = attack_impact.get(target, 0) + strength * 0.1
                
                # Generate insight
                if strength > 0.7:
                    insights.append(f"⚠️ {attacker} warns: {target} ignores {result.get('risk', 'risk')}")
        
        # Calculate adjusted scores
        adjusted_scores = {}
        for name, base_score in base_scores.items():
            impact = attack_impact.get(name, 0)
            
            # Find support strength
            support = 0.5
            for arg in arguments:
                if arg["type"] == "support" and arg["strategy"] == name:
                    support = arg.get("strength", 0.5)
                    break
            
            # Adjust score: boost by support, penalize by attacks
            # CAP PENALTY: Never reduce score by more than 50%
            effective_impact = min(impact, 0.5)  # Cap attack impact at 50%
            adjusted = base_score * (1 + support * 0.2) * (1 - effective_impact)
            
            # Ensure minimum score floor (strategies shouldn't be zeroed out)
            adjusted_scores[name] = max(0.15, min(1.0, adjusted))
        
        # PRESERVE relative scores - don't force sum to 1.0
        # Instead, scale so that the best strategy keeps its score
        if adjusted_scores:
            max_score = max(adjusted_scores.values())
            if max_score > 0:
                # Scale so max score stays near 1.0 (but keep minimum floor)
                scale = 1.0 / max_score if max_score > 1.0 else 1.0
                adjusted_scores = {k: max(0.15, v * scale) for k, v in adjusted_scores.items()}
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        llm_success = sum(1 for r in results if r is not None)
        
        logger.info(f"⚡ Parallel debate complete: {elapsed_ms:.0f}ms, {llm_success}/{len(all_tasks)} LLM calls succeeded")
        
        return ParallelDebateResult(
            strategy_scores=adjusted_scores,
            arguments=arguments,
            insights=insights,
            execution_time_ms=elapsed_ms,
            llm_calls_made=len(all_tasks),
            llm_calls_parallel=llm_success,
        )
    
    def _build_market_context(self, features) -> str:
        """Build concise market context string WITH cross-asset data."""
        parts = []
        
        if features.regime:
            r = features.regime
            parts.append(f"Trend={r.trend.value}, Vol={r.volatility.value}")
        
        if hasattr(features, 'vix') and features.vix:
            parts.append(f"VIX={features.vix:.1f}")
        
        if hasattr(features, 'returns_1d') and features.returns_1d:
            spy_ret = features.returns_1d.get('SPY', 0)
            parts.append(f"SPY={spy_ret*100:+.1f}%")
        
        # Add macro features (oil, gold, etc.) if available
        if hasattr(features, 'macro_features') and features.macro_features:
            mf = features.macro_features
            if hasattr(mf, 'oil_price') and mf.oil_price:
                parts.append(f"Oil=${mf.oil_price:.0f}")
            if hasattr(mf, 'gold_price') and mf.gold_price:
                parts.append(f"Gold=${mf.gold_price:.0f}")
            if hasattr(mf, 'dxy') and mf.dxy:
                parts.append(f"DXY={mf.dxy:.1f}")
            if hasattr(mf, 'geopolitical_risk_index'):
                parts.append(f"GeoRisk={mf.geopolitical_risk_index:.1f}")
        
        # Add cross-asset signals if available
        if hasattr(features, 'cross_asset_signals') and features.cross_asset_signals:
            ca = features.cross_asset_signals
            cross_parts = []
            if 'oil_signal' in ca:
                cross_parts.append(f"Oil→{'↑' if ca['oil_signal'] > 0 else '↓'}")
            if 'dxy_signal' in ca:
                cross_parts.append(f"USD→{'↑' if ca['dxy_signal'] > 0 else '↓'}")
            if 'europe_lead' in ca:
                cross_parts.append(f"EU→{'↑' if ca['europe_lead'] > 0 else '↓'}")
            if cross_parts:
                parts.append(f"X-Asset[{','.join(cross_parts)}]")
        
        return ", ".join(parts) if parts else "Normal conditions"
    
    def shutdown(self):
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=False)


# Background pre-computation for even faster execution
class BackgroundDebateCache:
    """
    Runs LLM debate in background continuously.
    Rebalance can use pre-computed results instantly.
    """
    
    def __init__(self, debate_engine: ParallelDebateEngine, refresh_interval_seconds: int = 300):
        self.debate_engine = debate_engine
        self.refresh_interval = refresh_interval_seconds
        self.cached_result: Optional[ParallelDebateResult] = None
        self.cache_timestamp: Optional[datetime] = None
        self.lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self, signals_provider, features_provider, base_scores_provider):
        """Start background debate refresh."""
        self._running = True
        self._signals_provider = signals_provider
        self._features_provider = features_provider
        self._base_scores_provider = base_scores_provider
        
        self._thread = threading.Thread(target=self._background_worker, daemon=True)
        self._thread.start()
        logger.info(f"🔄 Background debate cache started (refresh every {self.refresh_interval}s)")
    
    def stop(self):
        """Stop background refresh."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _background_worker(self):
        """Background thread that continuously updates debate cache."""
        while self._running:
            try:
                signals = self._signals_provider()
                features = self._features_provider()
                base_scores = self._base_scores_provider()
                
                if signals and features and base_scores:
                    result = self.debate_engine.run_parallel_debate(
                        signals, features, base_scores
                    )
                    
                    with self.lock:
                        self.cached_result = result
                        self.cache_timestamp = datetime.now()
                    
                    logger.info(f"🔄 Background debate updated: {len(result.strategy_scores)} strategies")
                
            except Exception as e:
                logger.warning(f"Background debate failed: {e}")
            
            time.sleep(self.refresh_interval)
    
    def get_cached_result(self, max_age_seconds: int = 600) -> Optional[ParallelDebateResult]:
        """Get cached result if fresh enough."""
        with self.lock:
            if not self.cached_result or not self.cache_timestamp:
                return None
            
            age = (datetime.now() - self.cache_timestamp).total_seconds()
            if age > max_age_seconds:
                return None
            
            return self.cached_result
