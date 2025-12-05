"""
OpenAI LLM Client Implementation.

Date: 2025-11-18
Author: backend-developer
Description: Lightweight wrapper around OpenAI API for prompt analysis and optimization.
"""

import json
import time
from typing import Optional, Dict, Any

from loguru import logger
from openai import OpenAI

from ..llm_client import LLMClient, LLMResponse, UsageStats
from ...config import LLMConfig, LLMProvider
from ...utils.token_tracker import TokenUsageTracker
from ...utils.prompt_cache import PromptCache


class OpenAIClient(LLMClient):
    """OpenAI client for LLM-based prompt optimization.

    This client provides a lightweight wrapper around the OpenAI API with:
    - Automatic token usage tracking and cost calculation
    - Response caching to reduce API calls
    - Comprehensive error handling and retry logic
    - Cost limit enforcement
    - Performance metrics collection

    Example:
        >>> from optimizer.config import LLMConfig, LLMProvider
        >>> config = LLMConfig(
        ...     provider=LLMProvider.OPENAI,
        ...     model="gpt-4-turbo-preview",
        ...     api_key_env="OPENAI_API_KEY"
        ... )
        >>> client = OpenAIClient(config)
        >>> response = client.analyze_prompt("Summarize this document")
        >>> print(response.content)
    """

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI client.

        Args:
            config: LLM configuration

        Raises:
            ValueError: If provider is not OPENAI or config is invalid

        Example:
            >>> config = LLMConfig(provider=LLMProvider.OPENAI)
            >>> client = OpenAIClient(config)
        """
        if config.provider != LLMProvider.OPENAI:
            raise ValueError(
                f"OpenAIClient requires provider=OPENAI, got {config.provider}"
            )

        # Validate configuration
        config.validate_config()

        self._config = config
        self._client = OpenAI(
            api_key=config.get_api_key(),
            base_url=config.base_url
        )

        # Initialize tools
        self._tracker = TokenUsageTracker()
        self._cache = PromptCache(
            ttl_seconds=config.cache_ttl,
            max_size=1000
        ) if config.enable_cache else None

        self._logger = logger.bind(module="optimizer.openai_client")
        self._logger.info(
            f"Initialized OpenAI client with model={config.model}, "
            f"cache_enabled={config.enable_cache}"
        )

    def analyze_prompt(self, prompt: str) -> LLMResponse:
        """Analyze prompt quality using GPT-4.

        This method uses GPT-4 to analyze prompt quality across multiple dimensions:
        - Clarity: How clear and unambiguous the instructions are
        - Efficiency: How concise and well-structured the prompt is
        - Structure: How well-organized the prompt is

        Args:
            prompt: Prompt text to analyze

        Returns:
            LLMResponse with JSON analysis results

        Raises:
            Exception: If API call fails after retries

        Example:
            >>> response = client.analyze_prompt("Summarize the document")
            >>> analysis = json.loads(response.content)
            >>> print(f"Clarity score: {analysis['clarity_score']}")
        """
        system_prompt = """You are a prompt quality analysis expert. Analyze the following prompt and rate it across these dimensions (0-100):

1. Clarity: Are the instructions clear and unambiguous?
2. Efficiency: Is the prompt concise without unnecessary verbosity?
3. Structure: Is the prompt well-organized and properly formatted?

Respond ONLY with valid JSON in this exact format:
{
  "clarity_score": 85,
  "efficiency_score": 75,
  "structure_score": 90,
  "overall_score": 83,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"]
}

Ensure:
- All scores are integers between 0-100
- overall_score is the average of the three dimension scores
- issues and suggestions are arrays of strings (can be empty)
- Response is valid JSON only, no additional text"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this prompt:\n\n{prompt}"}
        ]

        return self._call_openai(messages, "analysis")

    def optimize_prompt(
        self,
        prompt: str,
        strategy: str,
        current_analysis: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Optimize prompt using GPT-4.

        Args:
            prompt: Original prompt text
            strategy: Optimization strategy (llm_guided, llm_clarity, llm_efficiency)
            current_analysis: Optional analysis results to guide optimization

        Returns:
            LLMResponse with optimized prompt

        Raises:
            Exception: If API call fails after retries

        Example:
            >>> response = client.optimize_prompt(
            ...     "Write summary",
            ...     strategy="llm_clarity"
            ... )
            >>> print(response.content)
        """
        # Check cache first
        if self._cache:
            cached = self._cache.get(prompt, strategy)
            if cached:
                self._logger.info(f"Cache hit for strategy={strategy}")
                return LLMResponse(
                    content=cached,
                    tokens_used=0,
                    cost=0.0,
                    model=self._config.model,
                    provider="openai",
                    latency_ms=0,
                    cached=True,
                    metadata={"strategy": strategy}
                )

        # Build optimization instruction
        system_prompt = self._build_optimization_prompt(strategy, current_analysis)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Optimize this prompt:\n\n{prompt}"}
        ]

        response = self._call_openai(messages, "optimization")

        # Cache the result
        if self._cache:
            self._cache.set(prompt, response.content, strategy)

        return response

    def _call_openai(self, messages: list, operation: str) -> LLMResponse:
        """Call OpenAI API with error handling and tracking.

        Args:
            messages: Chat messages for the API
            operation: Operation type for logging (analysis/optimization)

        Returns:
            LLMResponse with API response

        Raises:
            Exception: If API call fails after retries
        """
        start_time = time.time()

        try:
            self._logger.debug(f"Calling OpenAI API for operation={operation}")

            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage information
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Track token usage and calculate cost
            cost = self._tracker.track_usage(
                self._config.model,
                prompt_tokens,
                completion_tokens,
                latency_ms
            )

            # Check cost limits
            if self._config.cost_limits:
                request_limit = self._config.cost_limits.get("max_cost_per_request")
                if request_limit and cost > request_limit:
                    self._logger.warning(
                        f"Request cost ${cost:.4f} exceeds limit ${request_limit}"
                    )

                daily_limit = self._config.cost_limits.get("max_cost_per_day")
                if daily_limit:
                    daily_cost = self._tracker.get_daily_cost()
                    if daily_cost > daily_limit:
                        self._logger.error(
                            f"Daily cost ${daily_cost:.4f} exceeds limit ${daily_limit}"
                        )

            # Extract content
            content = response.choices[0].message.content

            self._logger.info(
                f"OpenAI API call succeeded: operation={operation}, "
                f"tokens={total_tokens}, cost=${cost:.4f}, latency={latency_ms:.0f}ms"
            )

            return LLMResponse(
                content=content,
                tokens_used=total_tokens,
                cost=cost,
                model=self._config.model,
                provider="openai",
                latency_ms=latency_ms,
                cached=False,
                metadata={
                    "operation": operation,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": response.choices[0].finish_reason
                }
            )

        except Exception as e:
            self._logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    def _build_optimization_prompt(
        self,
        strategy: str,
        analysis: Optional[Dict[str, Any]]
    ) -> str:
        """Build system prompt for optimization based on strategy.

        Args:
            strategy: Optimization strategy
            analysis: Optional current analysis results

        Returns:
            System prompt string
        """
        base = "You are an expert prompt engineer. "

        if strategy == "llm_guided":
            instruction = (
                "Optimize the following prompt comprehensively. Focus on:\n"
                "1. Improving clarity and specificity\n"
                "2. Enhancing structure and organization\n"
                "3. Removing redundancy while maintaining completeness\n\n"
                "Respond ONLY with the optimized prompt, no explanations."
            )
        elif strategy == "llm_clarity":
            instruction = (
                "Optimize the following prompt for maximum clarity. Focus on:\n"
                "1. Making instructions explicit and unambiguous\n"
                "2. Adding necessary context and constraints\n"
                "3. Using precise language\n\n"
                "Respond ONLY with the optimized prompt, no explanations."
            )
        elif strategy == "llm_efficiency":
            instruction = (
                "Optimize the following prompt for efficiency. Focus on:\n"
                "1. Removing unnecessary verbosity\n"
                "2. Consolidating redundant instructions\n"
                "3. Maintaining clarity while being concise\n\n"
                "Respond ONLY with the optimized prompt, no explanations."
            )
        else:
            instruction = "Optimize the following prompt. Respond ONLY with the optimized prompt."

        # Add analysis context if available
        if analysis:
            context = f"\n\nCurrent quality scores:\n{json.dumps(analysis, indent=2)}\n"
            instruction = context + instruction

        return base + instruction

    def get_usage_stats(self) -> UsageStats:
        """Get usage statistics.

        Returns:
            UsageStats with request counts, tokens, costs, and cache metrics

        Example:
            >>> stats = client.get_usage_stats()
            >>> print(f"Total cost: ${stats.total_cost:.4f}")
            >>> print(f"Cache hit rate: {stats.cache_hits / (stats.cache_hits + stats.cache_misses):.2%}")
        """
        tracker_stats = self._tracker.get_stats()
        cache_stats = self._cache.get_stats() if self._cache else {}

        return UsageStats(
            total_requests=tracker_stats.get("request_count", 0),
            total_tokens=tracker_stats.get("total_tokens", 0),
            total_cost=tracker_stats.get("total_cost", 0.0),
            cache_hits=cache_stats.get("hits", 0),
            cache_misses=cache_stats.get("misses", 0),
            average_latency_ms=tracker_stats.get("average_latency_ms", 0.0)
        )

    def reset_stats(self) -> None:
        """Reset usage statistics.

        Example:
            >>> client.reset_stats()
        """
        self._tracker.reset()
        if self._cache:
            self._cache.clear()
        self._logger.info("Usage statistics reset")
