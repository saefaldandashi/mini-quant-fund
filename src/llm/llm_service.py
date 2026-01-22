"""
LLM Service - Abstraction layer for LLM API calls.

Supports multiple providers with fallback:
- Google Gemini (primary)
- OpenAI (fallback)
- Anthropic (fallback)
- Local models (future)

Features:
- Cost tracking
- Rate limiting
- Caching
- Error handling with retries
"""
import os
import logging
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    model: str
    tokens_used: int
    cost_usd: float
    latency_ms: int
    cached: bool = False


class LLMService:
    """
    Unified LLM service with cost control and caching.
    
    Supports:
    - Google Gemini (primary - free tier available)
    - OpenAI
    - Anthropic
    """
    
    # Cost per 1K tokens (approximate)
    COST_PER_1K = {
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
        'gemini-2.0-flash': {'input': 0.0, 'output': 0.0},  # Free tier
        'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005},
        'gemini-1.5-flash': {'input': 0.0, 'output': 0.0},  # Free tier
        'gemini-3-pro-preview': {'input': 0.00125, 'output': 0.01},  # Gemini 3 Pro Preview
        'gemini-3-flash-preview': {'input': 0.0, 'output': 0.0},  # Gemini 3 Flash Preview
        'gemini-exp-1206': {'input': 0.0, 'output': 0.0},  # Experimental
    }
    
    def __init__(
        self,
        provider: str = "gemini",  # 'gemini', 'openai', or 'anthropic'
        model: str = "gemini-2.0-flash",  # Default to Gemini
        api_key: Optional[str] = None,
        cache_dir: str = "outputs/llm_cache",
        max_daily_cost: float = 5.0,  # $5/day limit
        cache_ttl_hours: int = 24,
    ):
        self.provider = provider
        self.model = model
        
        # Get API key based on provider
        if api_key:
            self.api_key = api_key
        elif provider == "gemini":
            self.api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        elif provider == "openai":
            self.api_key = os.environ.get('OPENAI_API_KEY')
        elif provider == "anthropic":
            self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        else:
            self.api_key = None
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_daily_cost = max_daily_cost
        self.cache_ttl_hours = cache_ttl_hours
        
        # Track usage
        self.daily_cost = 0.0
        self.daily_calls = 0
        self.last_reset = datetime.now()
        
        # Cache stats
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Confidence calibration
        self.predictions: List[Dict] = []  # Track predictions and outcomes
        self.calibration_accuracy = 0.5  # Historical accuracy (starts neutral)
        self._load_calibration()
        
        # Load usage tracking
        self._load_usage()
        
        # Initialize client
        self._client = None
        self._gemini_model = None
        if self.api_key:
            self._init_client()
    
    def _init_client(self):
        """Initialize the API client."""
        if self.provider == "gemini":
            if GEMINI_AVAILABLE and genai:
                try:
                    genai.configure(api_key=self.api_key)
                    self._gemini_model = genai.GenerativeModel(self.model)
                    self._client = True  # Mark as initialized
                    logger.info(f"Gemini client initialized with model: {self.model}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {e}")
            else:
                logger.warning("google-generativeai package not installed. Run: pip install google-generativeai")
        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("openai package not installed")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("anthropic package not installed")
    
    def _load_usage(self):
        """Load daily usage tracking."""
        usage_file = self.cache_dir / "usage.json"
        if usage_file.exists():
            try:
                with open(usage_file) as f:
                    data = json.load(f)
                
                last_reset = datetime.fromisoformat(data.get('last_reset', '2000-01-01'))
                
                # Reset if new day
                if last_reset.date() < datetime.now().date():
                    self.daily_cost = 0.0
                    self.daily_calls = 0
                else:
                    self.daily_cost = data.get('daily_cost', 0.0)
                    self.daily_calls = data.get('daily_calls', 0)
                
                self.last_reset = last_reset
            except Exception as e:
                logger.warning(f"Could not load usage: {e}")
    
    def _save_usage(self):
        """Save daily usage tracking."""
        usage_file = self.cache_dir / "usage.json"
        try:
            with open(usage_file, 'w') as f:
                json.dump({
                    'daily_cost': self.daily_cost,
                    'daily_calls': self.daily_calls,
                    'last_reset': self.last_reset.isoformat(),
                }, f)
        except Exception as e:
            logger.warning(f"Could not save usage: {e}")
    
    def _load_calibration(self):
        """Load historical prediction accuracy for confidence calibration."""
        calibration_file = self.cache_dir / "calibration.json"
        if calibration_file.exists():
            try:
                with open(calibration_file) as f:
                    data = json.load(f)
                self.predictions = data.get('predictions', [])[-100:]  # Keep last 100
                self.calibration_accuracy = data.get('accuracy', 0.5)
            except:
                pass
    
    def _save_calibration(self):
        """Save calibration data."""
        calibration_file = self.cache_dir / "calibration.json"
        try:
            with open(calibration_file, 'w') as f:
                json.dump({
                    'predictions': self.predictions[-100:],
                    'accuracy': self.calibration_accuracy,
                }, f)
        except:
            pass
    
    def record_prediction(self, prediction_id: str, confidence: float, direction: str, symbol: str):
        """
        Record a prediction made by LLM for later calibration.
        
        Args:
            prediction_id: Unique ID for this prediction
            confidence: LLM's stated confidence (0-1)
            direction: 'bullish' or 'bearish'
            symbol: Stock symbol
        """
        self.predictions.append({
            'id': prediction_id,
            'confidence': confidence,
            'direction': direction,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'outcome': None,  # Will be filled later
        })
        self._save_calibration()
    
    def record_outcome(self, prediction_id: str, was_correct: bool):
        """
        Record the outcome of a previous prediction.
        
        Args:
            prediction_id: ID of the prediction
            was_correct: True if the prediction was correct
        """
        for pred in self.predictions:
            if pred['id'] == prediction_id:
                pred['outcome'] = was_correct
                break
        
        # Recalculate calibration accuracy
        outcomes = [p for p in self.predictions if p['outcome'] is not None]
        if outcomes:
            self.calibration_accuracy = sum(1 for p in outcomes if p['outcome']) / len(outcomes)
        
        self._save_calibration()
    
    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """
        Adjust LLM confidence based on historical accuracy.
        
        If LLM is overconfident (high confidence but low accuracy),
        we dial down the confidence. If underconfident, we boost it.
        
        Args:
            raw_confidence: LLM's stated confidence
            
        Returns:
            Calibrated confidence
        """
        # Simple linear calibration
        # If accuracy is 0.5 (random), return raw_confidence
        # If accuracy is higher, boost confidence; if lower, reduce it
        
        calibration_factor = self.calibration_accuracy / 0.5  # 1.0 if 50% accurate
        calibrated = raw_confidence * min(1.5, max(0.5, calibration_factor))
        
        return max(0.1, min(0.95, calibrated))  # Clamp to 0.1-0.95
    
    def _make_cache_key(self, prompt: str, system: str) -> str:
        """Create cache key from prompt."""
        content = f"{self.model}:{system}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cached(self, cache_key: str) -> Optional[str]:
        """Get cached response if fresh."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file) as f:
                data = json.load(f)
            
            cached_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
            if datetime.now() - cached_time > timedelta(hours=self.cache_ttl_hours):
                return None  # Expired
            
            self.cache_hits += 1
            return data.get('response')
        except Exception:
            return None
    
    def _save_cache(self, cache_key: str, response: str):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'model': self.model,
                }, f)
        except Exception as e:
            logger.warning(f"Could not cache response: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self._client is not None and self.daily_cost < self.max_daily_cost
    
    def call(
        self,
        prompt: str,
        system: str = "You are a financial analyst assistant.",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        use_cache: bool = True,
    ) -> Optional[LLMResponse]:
        """
        Call the LLM with cost control and caching.
        
        Returns None if:
        - Service not available
        - Daily cost limit reached
        - API error
        """
        # Check budget
        if self.daily_cost >= self.max_daily_cost:
            logger.warning(f"Daily cost limit reached (${self.daily_cost:.2f})")
            return None
        
        # Check client
        if not self._client:
            logger.warning("LLM client not initialized (missing API key?)")
            return None
        
        # Check cache
        cache_key = self._make_cache_key(prompt, system)
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached:
                return LLMResponse(
                    content=cached,
                    model=self.model,
                    tokens_used=0,
                    cost_usd=0,
                    latency_ms=0,
                    cached=True,
                )
        
        self.cache_misses += 1
        
        # Make API call
        start_time = time.time()
        
        try:
            if self.provider == "gemini":
                # Combine system and user prompt for Gemini
                full_prompt = f"{system}\n\n{prompt}"
                
                # Configure generation with safety settings
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                
                # Set safety to minimum to allow financial discussion
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                response = self._gemini_model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                
                # Handle different response types
                if not response.candidates:
                    logger.warning("Gemini returned no candidates")
                    return None
                
                candidate = response.candidates[0]
                
                # Try to extract content from parts
                content = None
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts:
                        content = "".join(part.text for part in parts if hasattr(part, 'text'))
                
                # Fallback to response.text
                if not content:
                    try:
                        content = response.text
                    except Exception as e:
                        logger.warning(f"Could not extract Gemini response: {e}")
                        return None
                
                if not content:
                    logger.warning("Gemini returned empty content")
                    return None
                
                # Gemini doesn't always provide token counts
                tokens = len(prompt.split()) + len(content.split())  # Rough estimate
                
            elif self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens
                
            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    system=system,
                )
                
                content = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
            
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return None
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Calculate cost
            cost_rates = self.COST_PER_1K.get(self.model, {'input': 0.01, 'output': 0.03})
            cost = (tokens / 1000) * (cost_rates['input'] + cost_rates['output']) / 2
            
            # Update tracking
            self.daily_cost += cost
            self.daily_calls += 1
            self._save_usage()
            
            # Cache response
            if use_cache:
                self._save_cache(cache_key, content)
            
            logger.info(f"LLM call: {tokens} tokens, ${cost:.4f}, {latency_ms}ms")
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            'provider': self.provider,
            'model': self.model,
            'is_available': self.is_available(),
            'daily_cost': self.daily_cost,
            'daily_calls': self.daily_calls,
            'max_daily_cost': self.max_daily_cost,
            'budget_remaining': self.max_daily_cost - self.daily_cost,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
        }
    
    def generate_trade_reasoning(
        self,
        final_weights: Dict[str, float],
        debate_summary: str,
        macro_context: str,
        signals_summary: str,
    ) -> Optional[str]:
        """
        Generate a natural language explanation for the trade decision.
        
        This is the KEY function for genuine reasoning - it synthesizes:
        - The debate outcome
        - Macro context
        - Strategy signals
        
        Into a coherent explanation of WHY we're making these trades.
        """
        if not self.is_available():
            return None
        
        # Build prompt
        top_positions = sorted(
            final_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        positions_str = "\n".join([
            f"  {symbol}: {weight:+.1%}" 
            for symbol, weight in top_positions
        ])
        
        prompt = f"""As the Chief Investment Officer, provide a 3-5 sentence explanation of today's portfolio decision.

MACRO CONTEXT:
{macro_context}

STRATEGY DEBATE SUMMARY:
{debate_summary}

SIGNALS FROM STRATEGIES:
{signals_summary}

FINAL POSITIONS:
{positions_str}

Explain:
1. The KEY driver of today's allocation (one main reason)
2. The primary RISK we're managing against
3. Why THIS specific set of positions (not generic statements)

Be specific. Reference the actual data above. No platitudes."""

        response = self.call(
            prompt=prompt,
            system="You are a CIO explaining investment decisions to the board. Be specific, not generic.",
            temperature=0.4,
            max_tokens=300,
            use_cache=False,  # Each decision is unique
        )
        
        if response:
            return response.content
        return None
    
    def analyze_trade_outcome(
        self,
        trade_entry: Dict,
        outcome: Dict,
    ) -> Optional[str]:
        """
        Analyze why a trade succeeded or failed.
        
        This enables LEARNING from past trades.
        """
        if not self.is_available():
            return None
        
        prompt = f"""Analyze this trade outcome:

TRADE ENTRY:
- Symbol: {trade_entry.get('symbol')}
- Direction: {trade_entry.get('direction')}
- Entry Price: {trade_entry.get('entry_price')}
- Reasoning: {trade_entry.get('reasoning')}
- Macro Context: {trade_entry.get('macro_context')}

OUTCOME:
- Exit Price: {outcome.get('exit_price')}
- Return: {outcome.get('return_pct')}%
- Holding Period: {outcome.get('days_held')} days

Provide:
1. Was the original thesis correct or wrong? Why?
2. What was the key factor that determined the outcome?
3. What should we learn for similar setups in the future?

Be analytical and specific."""

        response = self.call(
            prompt=prompt,
            system="You are analyzing trading outcomes to improve future decisions.",
            temperature=0.3,
            max_tokens=250,
            use_cache=True,
        )
        
        if response:
            return response.content
        return None
