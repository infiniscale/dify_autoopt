"""
Optimizer Module - LLM Client Interface

Date: 2025-11-17
Author: backend-developer
Description: Abstract interface for LLM-based prompt analysis and optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMClient(ABC):
    """Abstract interface for LLM-based prompt analysis.

    This interface enables future integration with different LLM providers
    (OpenAI, Anthropic, etc.) without modifying core optimizer logic.

    Implementations:
        - StubLLMClient (MVP): Rule-based stub for testing.
        - OpenAIClient (Future): GPT-4 based analysis.
        - AnthropicClient (Future): Claude based analysis.

    Example:
        >>> client = StubLLMClient()
        >>> analysis = client.analyze_prompt("Your prompt text", context={"workflow_id": "wf_001"})
        >>> print(analysis.clarity_score)
    """

    @abstractmethod
    def analyze_prompt(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> "PromptAnalysis":  # type: ignore
        """Analyze prompt quality using LLM.

        Args:
            prompt: Prompt text to analyze.
            context: Optional context (workflow_id, node_type, etc.).

        Returns:
            PromptAnalysis with scores and suggestions.

        Raises:
            AnalysisError: If LLM call fails or analysis cannot be performed.

        Example:
            >>> analysis = client.analyze_prompt(
            ...     "Summarize the document",
            ...     context={"workflow_id": "wf_001", "node_id": "llm_1"}
            ... )
        """
        pass

    @abstractmethod
    def optimize_prompt(
        self, prompt: str, strategy: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate optimized prompt using LLM.

        Args:
            prompt: Original prompt text.
            strategy: Optimization strategy hint (clarity_focus, efficiency_focus, etc.).
            context: Optional context information.

        Returns:
            Optimized prompt text.

        Raises:
            OptimizationError: If LLM call fails or optimization cannot be performed.

        Example:
            >>> optimized = client.optimize_prompt(
            ...     "Write a summary",
            ...     strategy="clarity_focus"
            ... )
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
        >>> analysis = client.analyze_prompt("Your prompt")
        >>> # Returns analysis based on heuristics, not LLM
    """

    def analyze_prompt(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> "PromptAnalysis":  # type: ignore
        """Analyze prompt using rule-based heuristics.

        This method delegates to PromptAnalyzer for MVP implementation.

        Args:
            prompt: Prompt text to analyze.
            context: Optional context (workflow_id, node_type, etc.).

        Returns:
            PromptAnalysis with scores based on heuristics.

        Note:
            This is a stub implementation. It does NOT call external LLM APIs.
        """
        # Import here to avoid circular dependency
        from ..prompt_analyzer import PromptAnalyzer
        from ..models import Prompt
        from datetime import datetime

        # Create temporary Prompt object
        temp_prompt = Prompt(
            id="temp",
            workflow_id=context.get("workflow_id", "unknown") if context else "unknown",
            node_id=context.get("node_id", "unknown") if context else "unknown",
            node_type="llm",
            text=prompt,
            role="system",
            variables=[],
            context=context or {},
            extracted_at=datetime.now(),
        )

        # Use rule-based analyzer
        analyzer = PromptAnalyzer()
        return analyzer.analyze_prompt(temp_prompt)

    def optimize_prompt(
        self, prompt: str, strategy: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate optimized prompt using rule-based transformations.

        This method delegates to OptimizationEngine for MVP implementation.

        Args:
            prompt: Original prompt text.
            strategy: Strategy name (clarity_focus, efficiency_focus, structure_focus).
            context: Optional context information.

        Returns:
            Optimized prompt text.

        Raises:
            InvalidStrategyError: If strategy name is invalid.

        Note:
            This is a stub implementation. It does NOT call external LLM APIs.
        """
        # Import here to avoid circular dependency
        from ..optimization_engine import OptimizationEngine
        from ..prompt_analyzer import PromptAnalyzer
        from ..exceptions import InvalidStrategyError

        # Validate strategy
        valid_strategies = ["clarity_focus", "efficiency_focus", "structure_focus"]
        if strategy not in valid_strategies:
            raise InvalidStrategyError(strategy, valid_strategies)

        # Use rule-based engine
        analyzer = PromptAnalyzer()
        engine = OptimizationEngine(analyzer)

        # Apply strategy
        if strategy == "clarity_focus":
            return engine.apply_clarity_focus(prompt)
        elif strategy == "efficiency_focus":
            return engine.apply_efficiency_focus(prompt)
        elif strategy == "structure_focus":
            return engine.apply_structure_optimization(prompt)
        else:
            # Fallback (should not reach here due to validation)
            return prompt
