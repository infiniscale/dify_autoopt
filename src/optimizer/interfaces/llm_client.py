"""
Optimizer Module - LLM Client Interface

Date: 2025-11-17
Author: backend-developer
Description: Abstract interface for LLM-based prompt analysis and optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """LLM调用响应的统一封装

    Attributes:
        content: 生成的内容
        tokens_used: 使用的token数
        cost: 本次调用成本（美元）
        model: 使用的模型
        provider: 提供商
        latency_ms: 响应延迟（毫秒）
        cached: 是否来自缓存
        metadata: 额外元数据
        created_at: 创建时间
    """
    content: str = Field(..., description="生成的内容")
    tokens_used: int = Field(..., description="使用的token数")
    cost: float = Field(..., description="本次调用成本（美元）")
    model: str = Field(..., description="使用的模型")
    provider: str = Field(..., description="提供商")
    latency_ms: float = Field(..., description="响应延迟（毫秒）")
    cached: bool = Field(default=False, description="是否来自缓存")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    created_at: datetime = Field(default_factory=datetime.now)


class UsageStats(BaseModel):
    """使用统计

    Attributes:
        total_requests: 总请求数
        total_tokens: 总token数
        total_cost: 总成本
        cache_hits: 缓存命中次数
        cache_misses: 缓存未命中次数
        average_latency_ms: 平均延迟（毫秒）
    """
    total_requests: int = Field(default=0, description="总请求数")
    total_tokens: int = Field(default=0, description="总token数")
    total_cost: float = Field(default=0.0, description="总成本")
    cache_hits: int = Field(default=0, description="缓存命中次数")
    cache_misses: int = Field(default=0, description="缓存未命中次数")
    average_latency_ms: float = Field(default=0.0, description="平均延迟（毫秒）")


class LLMClient(ABC):
    """Abstract interface for LLM-based prompt analysis.

    This interface enables future integration with different LLM providers
    (OpenAI, Anthropic, etc.) without modifying core optimizer logic.

    Implementations:
        - StubLLMClient (MVP): Rule-based stub for testing.
        - OpenAIClient: GPT-4 based analysis.
        - AnthropicClient (Future): Claude based analysis.

    Example:
        >>> client = StubLLMClient()
        >>> response = client.analyze_prompt("Your prompt text")
        >>> print(response.content)
    """

    @abstractmethod
    def analyze_prompt(self, prompt: str) -> LLMResponse:
        """Analyze prompt quality using LLM.

        Args:
            prompt: Prompt text to analyze.

        Returns:
            LLMResponse containing analysis results.

        Raises:
            AnalysisError: If LLM call fails or analysis cannot be performed.

        Example:
            >>> response = client.analyze_prompt("Summarize the document")
            >>> analysis = json.loads(response.content)
            >>> print(analysis['clarity_score'])
        """
        pass

    @abstractmethod
    def optimize_prompt(
        self,
        prompt: str,
        strategy: str,
        current_analysis: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate optimized prompt using LLM.

        Args:
            prompt: Original prompt text.
            strategy: Optimization strategy (llm_guided, llm_clarity, llm_efficiency).
            current_analysis: Current analysis results (optional).

        Returns:
            LLMResponse containing optimized prompt.

        Raises:
            OptimizationError: If LLM call fails or optimization cannot be performed.

        Example:
            >>> response = client.optimize_prompt(
            ...     "Write a summary",
            ...     strategy="llm_clarity"
            ... )
            >>> print(response.content)
        """
        pass

    @abstractmethod
    def get_usage_stats(self) -> UsageStats:
        """Get usage statistics.

        Returns:
            UsageStats containing total requests, tokens, costs, and cache metrics.

        Example:
            >>> stats = client.get_usage_stats()
            >>> print(f"Total cost: ${stats.total_cost:.4f}")
        """
        pass

    @abstractmethod
    def reset_stats(self) -> None:
        """Reset usage statistics.

        Example:
            >>> client.reset_stats()
        """
        pass


class StubLLMClient(LLMClient):
    """Stub implementation for MVP (rule-based).

    This implementation does NOT call external LLM APIs. Instead, it delegates
    to rule-based PromptAnalyzer and OptimizationEngine for deterministic testing.

    Used for:
        - Testing without API costs.
        - MVP deployment before LLM integration.
        - Deterministic behavior in CI/CD.

    Example:
        >>> client = StubLLMClient()
        >>> response = client.analyze_prompt("Your prompt")
        >>> # Returns analysis based on heuristics, not LLM
    """

    def __init__(self) -> None:
        """Initialize stub client with usage tracking."""
        self._total_requests = 0

    def analyze_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Analyze prompt using rule-based heuristics.

        This method delegates to PromptAnalyzer for MVP implementation.

        Args:
            prompt: Prompt text to analyze.
            context: Optional context (ignored in new interface, kept for backward compatibility).

        Returns:
            LLMResponse with analysis results in JSON format.

        Note:
            This is a stub implementation. It does NOT call external LLM APIs.
        """
        import json
        from ..prompt_analyzer import PromptAnalyzer
        from ..models import Prompt

        self._total_requests += 1

        # Create temporary Prompt object
        temp_prompt = Prompt(
            id="temp",
            workflow_id=context.get("workflow_id", "stub") if context else "stub",
            node_id=context.get("node_id", "stub") if context else "stub",
            node_type="llm",
            text=prompt,
            role="system",
            variables=[],
            context=context or {},
            extracted_at=datetime.now(),
        )

        # Use rule-based analyzer
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze_prompt(temp_prompt)

        # Convert to JSON format
        # Note: PromptAnalysis doesn't have structure_score, calculate as average
        structure_score = int((analysis.clarity_score + analysis.efficiency_score) / 2)

        result = {
            "overall_score": analysis.overall_score,
            "clarity_score": analysis.clarity_score,
            "efficiency_score": analysis.efficiency_score,
            "structure_score": structure_score,
            "issues": [issue.description for issue in analysis.issues],
            "suggestions": [sugg.description for sugg in analysis.suggestions]
        }

        return LLMResponse(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            tokens_used=0,
            cost=0.0,
            model="stub",
            provider="stub",
            latency_ms=0,
            cached=False,
            metadata={"method": "rule-based"}
        )

    def optimize_prompt(
        self,
        prompt: str,
        strategy: str,
        current_analysis: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate optimized prompt using rule-based transformations.

        This method delegates to OptimizationEngine for MVP implementation.

        Args:
            prompt: Original prompt text.
            strategy: Strategy name (llm_clarity, llm_efficiency, llm_guided, or legacy names).
            current_analysis: Current analysis results (optional, ignored in stub).
            context: Optional context (ignored, for backward compatibility).

        Returns:
            LLMResponse containing optimized prompt.

        Raises:
            InvalidStrategyError: If strategy name is invalid.

        Note:
            This is a stub implementation. It does NOT call external LLM APIs.
        """
        from ..optimization_engine import OptimizationEngine
        from ..prompt_analyzer import PromptAnalyzer
        from ..exceptions import InvalidStrategyError

        self._total_requests += 1

        # Map both new LLM strategies and legacy rule-based strategies
        strategy_mapping = {
            # New LLM-style names
            "llm_clarity": "clarity_focus",
            "llm_efficiency": "efficiency_focus",
            "llm_guided": "structure_focus",
            # Legacy rule-based names (for backward compatibility)
            "clarity_focus": "clarity_focus",
            "efficiency_focus": "efficiency_focus",
            "structure_focus": "structure_focus"
        }

        mapped_strategy = strategy_mapping.get(strategy)
        if not mapped_strategy:
            # Accept both new and old strategy names in error message
            valid_strategies = list(strategy_mapping.keys())
            raise InvalidStrategyError(strategy, valid_strategies)

        # Use rule-based engine
        analyzer = PromptAnalyzer()
        engine = OptimizationEngine(analyzer)

        # Apply strategy
        if mapped_strategy == "clarity_focus":
            optimized = engine.apply_clarity_focus(prompt)
        elif mapped_strategy == "efficiency_focus":
            optimized = engine.apply_efficiency_focus(prompt)
        elif mapped_strategy == "structure_focus":
            optimized = engine.apply_structure_optimization(prompt)
        else:
            optimized = prompt

        return LLMResponse(
            content=optimized,
            tokens_used=0,
            cost=0.0,
            model="stub",
            provider="stub",
            latency_ms=0,
            cached=False,
            metadata={"strategy": strategy, "method": "rule-based"}
        )

    def get_usage_stats(self) -> UsageStats:
        """Get usage statistics.

        Returns:
            UsageStats with stub metrics (no real API usage).
        """
        return UsageStats(
            total_requests=self._total_requests,
            total_tokens=0,
            total_cost=0.0,
            cache_hits=0,
            cache_misses=0,
            average_latency_ms=0.0
        )

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_requests = 0
