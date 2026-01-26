"""
Multi-Provider LLM Service

Queries multiple LLM providers in parallel and aggregates their responses.
This provides:
- Redundancy: If one provider fails, others still work
- Consensus: Multiple perspectives for higher conviction
- Speed: All providers queried simultaneously
"""

import os
import logging
import concurrent.futures
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time

from .llm_service import LLMService, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class MultiProviderResponse:
    """Aggregated response from multiple LLM providers."""
    responses: Dict[str, str]  # provider -> response
    consensus: Optional[str]  # Aggregated/consensus response
    providers_succeeded: int
    providers_failed: int
    total_latency_ms: int
    total_cost_usd: float


class MultiProviderLLM:
    """
    Queries multiple LLM providers in parallel.
    
    Use cases:
    - High-conviction trade decisions (get 3 AI opinions)
    - Risk assessment (multiple perspectives)
    - Debate synthesis (aggregate different viewpoints)
    """
    
    PROVIDERS = [
        ("gemini", "gemini-2.0-flash", "GEMINI_API_KEY"),
        ("openai", "gpt-4-turbo", "OPENAI_API_KEY"),
        ("anthropic", "claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
    ]
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize available providers
        self.providers: Dict[str, LLMService] = {}
        
        for provider, model, env_key in self.PROVIDERS:
            api_key = os.environ.get(env_key)
            if api_key:
                try:
                    llm = LLMService(provider=provider, model=model, api_key=api_key)
                    if llm.is_available():
                        self.providers[provider] = llm
                        logger.info(f"✅ Multi-LLM: {provider} available")
                except Exception as e:
                    logger.warning(f"Failed to init {provider}: {e}")
        
        logger.info(f"MultiProviderLLM initialized with {len(self.providers)} providers: {list(self.providers.keys())}")
    
    def _call_provider(
        self,
        provider_name: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple:
        """Call a single provider (for use in thread pool)."""
        try:
            llm = self.providers[provider_name]
            response = llm.call(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                use_cache=False,  # Fresh responses for multi-provider
            )
            
            if response:
                return (provider_name, response.content, response.latency_ms, response.cost_usd, True)
            return (provider_name, None, 0, 0, False)
            
        except Exception as e:
            logger.warning(f"Provider {provider_name} failed: {e}")
            return (provider_name, None, 0, 0, False)
    
    def call_all(
        self,
        prompt: str,
        system: str = "You are a financial analyst.",
        temperature: float = 0.3,
        max_tokens: int = 500,
        require_consensus: bool = False,
    ) -> MultiProviderResponse:
        """
        Query all available LLM providers in parallel.
        
        Args:
            prompt: The prompt to send
            system: System message
            temperature: Sampling temperature
            max_tokens: Max response tokens
            require_consensus: If True, synthesizes a consensus response
            
        Returns:
            MultiProviderResponse with all results
        """
        if not self.providers:
            logger.warning("No LLM providers available!")
            return MultiProviderResponse(
                responses={},
                consensus=None,
                providers_succeeded=0,
                providers_failed=0,
                total_latency_ms=0,
                total_cost_usd=0,
            )
        
        start_time = time.time()
        
        # Submit all calls in parallel
        futures = {
            self.executor.submit(
                self._call_provider,
                provider_name,
                prompt,
                system,
                temperature,
                max_tokens,
            ): provider_name
            for provider_name in self.providers
        }
        
        # Collect results
        responses = {}
        total_latency = 0
        total_cost = 0.0
        succeeded = 0
        failed = 0
        
        # Wait for all to complete (with timeout)
        done, not_done = concurrent.futures.wait(futures, timeout=30.0)
        
        for future in done:
            try:
                provider, content, latency, cost, success = future.result(timeout=0.1)
                if success and content:
                    responses[provider] = content
                    total_latency = max(total_latency, latency)  # Wall clock is max
                    total_cost += cost
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Future failed: {e}")
        
        # Handle timeouts
        for future in not_done:
            failed += 1
            future.cancel()
        
        # Build consensus if requested and we have multiple responses
        consensus = None
        if require_consensus and len(responses) >= 2:
            consensus = self._synthesize_consensus(responses, prompt)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Multi-LLM call: {succeeded}/{succeeded+failed} succeeded, {elapsed_ms}ms")
        
        return MultiProviderResponse(
            responses=responses,
            consensus=consensus,
            providers_succeeded=succeeded,
            providers_failed=failed,
            total_latency_ms=elapsed_ms,
            total_cost_usd=total_cost,
        )
    
    def _synthesize_consensus(self, responses: Dict[str, str], original_prompt: str) -> str:
        """Synthesize a consensus from multiple LLM responses."""
        # Use the first available provider to synthesize
        synthesizer = list(self.providers.values())[0]
        
        responses_text = "\n\n".join([
            f"=== {provider.upper()} ===\n{content}"
            for provider, content in responses.items()
        ])
        
        synthesis_prompt = f"""Multiple AI analysts provided these perspectives on the same question:

{responses_text}

ORIGINAL QUESTION: {original_prompt}

Synthesize a CONSENSUS view that:
1. Highlights points ALL analysts agree on
2. Notes key disagreements (if any)
3. Provides a balanced final recommendation

Keep response concise (3-5 sentences)."""

        response = synthesizer.call(
            prompt=synthesis_prompt,
            system="You are synthesizing multiple expert opinions into a balanced consensus.",
            temperature=0.2,
            max_tokens=300,
            use_cache=False,
        )
        
        return response.content if response else None
    
    def get_trade_consensus(
        self,
        symbol: str,
        market_context: str,
        signals: Dict[str, Any],
    ) -> MultiProviderResponse:
        """
        Get consensus from all LLMs on a specific trade decision.
        
        This is the KEY function for high-conviction trades.
        """
        prompt = f"""TRADE DECISION: {symbol}

MARKET CONTEXT:
{market_context}

STRATEGY SIGNALS:
{json.dumps(signals, indent=2, default=str)}

As a quant portfolio manager, provide your assessment:
1. Should we LONG, SHORT, or AVOID {symbol}?
2. Conviction level (1-10)?
3. Key risk to monitor?
4. Suggested position size (% of portfolio)?

Be specific and decisive."""

        return self.call_all(
            prompt=prompt,
            system="You are a quantitative portfolio manager making a trade decision.",
            temperature=0.3,
            max_tokens=400,
            require_consensus=True,
        )
    
    def get_risk_assessment(
        self,
        portfolio: Dict[str, float],
        market_conditions: str,
    ) -> MultiProviderResponse:
        """Get multi-LLM risk assessment for current portfolio."""
        
        positions_str = "\n".join([
            f"  {symbol}: {weight:+.1%}"
            for symbol, weight in sorted(portfolio.items(), key=lambda x: -abs(x[1]))[:15]
        ])
        
        prompt = f"""PORTFOLIO RISK ASSESSMENT

CURRENT POSITIONS:
{positions_str}

MARKET CONDITIONS:
{market_conditions}

Assess:
1. Top 3 risks to this portfolio RIGHT NOW
2. Which positions are most vulnerable?
3. Hedging recommendations?
4. Overall risk score (1-10)?"""

        return self.call_all(
            prompt=prompt,
            system="You are a risk manager evaluating portfolio exposure.",
            temperature=0.2,
            max_tokens=500,
            require_consensus=True,
        )
    
    def is_available(self) -> bool:
        """Check if at least one provider is available."""
        return len(self.providers) > 0
    
    def get_stats(self) -> Dict:
        """Get provider statistics."""
        return {
            "providers_available": list(self.providers.keys()),
            "provider_count": len(self.providers),
            "provider_stats": {
                name: llm.get_stats() for name, llm in self.providers.items()
            }
        }
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=False)


# Singleton for easy access
_multi_llm_instance: Optional[MultiProviderLLM] = None

def get_multi_llm() -> MultiProviderLLM:
    """Get or create the multi-provider LLM instance."""
    global _multi_llm_instance
    if _multi_llm_instance is None:
        _multi_llm_instance = MultiProviderLLM()
    return _multi_llm_instance
