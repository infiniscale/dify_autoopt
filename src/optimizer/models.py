"""
Optimizer Module - Data Models

Date: 2025-11-17
Author: backend-developer
Description: Pydantic V2 models for prompts, analysis, optimization, and versioning.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# Enumerations
# ============================================================================


class IssueSeverity(str, Enum):
    """Severity level for prompt issues."""

    CRITICAL = "critical"  # Severe issue blocking optimization
    WARNING = "warning"  # Issue should be addressed
    INFO = "info"  # Informational notice


class IssueType(str, Enum):
    """Type of prompt issue detected."""

    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    VAGUE_LANGUAGE = "vague_language"
    MISSING_STRUCTURE = "missing_structure"
    REDUNDANCY = "redundancy"
    POOR_FORMATTING = "poor_formatting"
    AMBIGUOUS_INSTRUCTIONS = "ambiguous_instructions"


class SuggestionType(str, Enum):
    """Type of improvement suggestion."""

    ADD_STRUCTURE = "add_structure"
    CLARIFY_INSTRUCTIONS = "clarify_instructions"
    REDUCE_LENGTH = "reduce_length"
    ADD_EXAMPLES = "add_examples"
    IMPROVE_FORMATTING = "improve_formatting"


class OptimizationStrategy(str, Enum):
    """Optimization strategy names.

    Rule-based strategies (MVP):
        - CLARITY_FOCUS: Improve readability and structure using rule-based heuristics
        - EFFICIENCY_FOCUS: Reduce length and remove redundancy using rules
        - STRUCTURE_FOCUS: Add formatting and organization using rules
        - AUTO: Auto-select based on analysis

    LLM-powered strategies (Production):
        - LLM_GUIDED: Full LLM rewrite with context understanding
        - LLM_CLARITY: Semantic restructuring for maximum clarity
        - LLM_EFFICIENCY: Intelligent compression preserving meaning
        - HYBRID: Combine rule-based + LLM refinement
    """

    # Rule-based strategies (MVP)
    CLARITY_FOCUS = "clarity_focus"
    EFFICIENCY_FOCUS = "efficiency_focus"
    STRUCTURE_FOCUS = "structure_focus"
    AUTO = "auto"  # Auto-select based on analysis

    # LLM-powered strategies (Production)
    LLM_GUIDED = "llm_guided"
    LLM_CLARITY = "llm_clarity"
    LLM_EFFICIENCY = "llm_efficiency"
    HYBRID = "hybrid"


# ============================================================================
# Core Models
# ============================================================================


class OptimizationChange(BaseModel):
    """Structured change record for audit trail.

    Attributes:
        rule_id: Rule identifier (e.g., 'REMOVE_FILLER', 'ADD_STRUCTURE')
        description: Human-readable description of the change
        location: Optional (start, end) character positions in text
        before: Optional text before the change
        after: Optional text after the change

    Example:
        >>> change = OptimizationChange(
        ...     rule_id="REMOVE_FILLER",
        ...     description="Removed filler word 'very'",
        ...     location=(42, 47),
        ...     before="very important",
        ...     after="important"
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    rule_id: str = Field(..., description="Rule identifier (e.g., 'REMOVE_FILLER')")
    description: str = Field(..., description="Human-readable description")
    location: Optional[Tuple[int, int]] = Field(
        None, description="(start, end) character positions"
    )
    before: Optional[str] = Field(None, description="Text before change")
    after: Optional[str] = Field(None, description="Text after change")


class Prompt(BaseModel):
    """Extracted prompt with metadata.

    Represents a prompt extracted from a workflow node, including all
    necessary context for analysis and optimization.

    Attributes:
        id: Unique identifier (workflow_id + node_id).
        workflow_id: Parent workflow identifier.
        node_id: Node identifier in workflow DSL.
        node_type: Type of node (llm, code, etc.).
        text: Actual prompt content.
        role: Message role (system, user, assistant).
        variables: List of variable placeholders found in text.
        context: Additional node metadata (dependencies, config, etc.).
        extracted_at: Timestamp of extraction.

    Example:
        >>> prompt = Prompt(
        ...     id="wf_001_llm_1",
        ...     workflow_id="wf_001",
        ...     node_id="llm_1",
        ...     node_type="llm",
        ...     text="Summarize: {{document}}",
        ...     role="user",
        ...     variables=["document"],
        ...     context={"label": "Summarizer"},
        ...     extracted_at=datetime.now()
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(..., description="Unique prompt identifier")
    workflow_id: str = Field(..., description="Parent workflow ID")
    node_id: str = Field(..., description="Node ID in workflow")
    node_type: str = Field(..., description="Node type (llm, code, etc.)")
    text: str = Field(..., description="Prompt content")
    role: str = Field("system", description="Message role (system, user, assistant)")
    variables: List[str] = Field(
        default_factory=list, description="Variable placeholders"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Extraction timestamp"
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure prompt text is non-empty and within size limits.

        Args:
            v: Prompt text to validate.

        Returns:
            Validated text.

        Raises:
            ValueError: If text is empty, whitespace only, or exceeds size limit.
        """
        if not v or not v.strip():
            raise ValueError("Prompt text cannot be empty")

        # Enforce reasonable limits (LLM context windows)
        # GPT-4: ~8k tokens ≈ 32k chars
        # GPT-4-32k: ~32k tokens ≈ 128k chars
        # Claude 2: ~100k tokens ≈ 400k chars
        MAX_PROMPT_LENGTH = 100_000  # ~25k tokens (safe for most LLMs)

        if len(v) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt exceeds maximum length: "
                f"{len(v):,} > {MAX_PROMPT_LENGTH:,} characters. "
                f"Please split into smaller prompts or use a model with larger context."
            )

        return v

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, v: List[str]) -> List[str]:
        """Ensure variable names are valid.

        Args:
            v: List of variable names.

        Returns:
            Validated list.

        Raises:
            ValueError: If any variable name is empty.
        """
        for var in v:
            if not var or not var.strip():
                raise ValueError(f"Invalid variable name: '{var}'")
        return v


class PromptIssue(BaseModel):
    """Detected issue in a prompt.

    Attributes:
        severity: Issue severity level (critical, warning, info).
        type: Issue type classification.
        description: Human-readable issue description.
        location: Optional location info (line number, section, etc.).
        suggestion: Optional immediate suggestion to fix the issue.

    Example:
        >>> issue = PromptIssue(
        ...     severity=IssueSeverity.WARNING,
        ...     type=IssueType.VAGUE_LANGUAGE,
        ...     description="Prompt uses vague terms like 'some' and 'maybe'",
        ...     location="Line 3",
        ...     suggestion="Replace vague language with specific instructions"
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    severity: IssueSeverity = Field(..., description="Issue severity")
    type: IssueType = Field(..., description="Issue type")
    description: str = Field(..., description="Issue description")
    location: Optional[str] = Field(None, description="Location in prompt")
    suggestion: Optional[str] = Field(None, description="Quick fix suggestion")


class PromptSuggestion(BaseModel):
    """Improvement suggestion for a prompt.

    Attributes:
        type: Suggestion type classification.
        description: Human-readable suggestion description.
        priority: Priority level (1-10, higher is more important).

    Example:
        >>> suggestion = PromptSuggestion(
        ...     type=SuggestionType.ADD_STRUCTURE,
        ...     description="Add section headers to organize instructions",
        ...     priority=8
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: SuggestionType = Field(..., description="Suggestion type")
    description: str = Field(..., description="Suggestion description")
    priority: int = Field(..., ge=1, le=10, description="Priority (1-10)")


class PromptAnalysis(BaseModel):
    """Analysis result for a prompt.

    Contains quality scores, detected issues, and improvement suggestions.

    Attributes:
        prompt_id: ID of analyzed prompt.
        overall_score: Overall quality score (0-100).
        clarity_score: Clarity score (0-100).
        efficiency_score: Efficiency score (0-100).
        issues: List of detected issues.
        suggestions: List of improvement suggestions.
        metadata: Additional metrics (token_count, sentence_count, etc.).
        analyzed_at: Timestamp of analysis.

    Example:
        >>> analysis = PromptAnalysis(
        ...     prompt_id="wf_001_llm_1",
        ...     overall_score=75.0,
        ...     clarity_score=80.0,
        ...     efficiency_score=70.0,
        ...     issues=[...],
        ...     suggestions=[...],
        ...     metadata={"token_count": 120},
        ...     analyzed_at=datetime.now()
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    prompt_id: str = Field(..., description="Analyzed prompt ID")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    clarity_score: float = Field(..., ge=0.0, le=100.0, description="Clarity score")
    efficiency_score: float = Field(
        ..., ge=0.0, le=100.0, description="Efficiency score"
    )
    issues: List[PromptIssue] = Field(
        default_factory=list, description="Detected issues"
    )
    suggestions: List[PromptSuggestion] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metrics"
    )
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Analysis timestamp"
    )

    @field_validator("overall_score", "clarity_score", "efficiency_score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are in valid range [0, 100].

        Args:
            v: Score value.

        Returns:
            Validated score.

        Raises:
            ValueError: If score is out of range.
        """
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"Score must be between 0 and 100, got {v}")
        return v


class OptimizationResult(BaseModel):
    """Result of prompt optimization.

    Contains original and optimized prompts, along with improvement metrics.

    Attributes:
        prompt_id: ID of optimized prompt.
        original_prompt: Original prompt text.
        optimized_prompt: Optimized prompt text.
        strategy: Strategy used for optimization.
        improvement_score: Score delta (optimized - original).
        confidence: Confidence in improvement (0.0-1.0).
        changes: List of structured changes made (OptimizationChange objects).
        metadata: Additional optimization metadata.
        optimized_at: Timestamp of optimization.

    Example:
        >>> result = OptimizationResult(
        ...     prompt_id="wf_001_llm_1",
        ...     original_prompt="Summarize the document",
        ...     optimized_prompt="Summarize the document in 3-5 bullet points",
        ...     strategy=OptimizationStrategy.CLARITY_FOCUS,
        ...     improvement_score=10.5,
        ...     confidence=0.85,
        ...     changes=[OptimizationChange(...)],
        ...     metadata={},
        ...     optimized_at=datetime.now()
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    prompt_id: str = Field(..., description="Optimized prompt ID")
    original_prompt: str = Field(..., description="Original prompt text")
    optimized_prompt: str = Field(..., description="Optimized prompt text")
    strategy: OptimizationStrategy = Field(..., description="Optimization strategy")
    improvement_score: float = Field(..., description="Score delta (can be negative)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in improvement"
    )
    changes: List[OptimizationChange] = Field(
        default_factory=list, description="Structured list of changes made"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    optimized_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Optimization timestamp"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Ensure confidence is in valid range [0.0, 1.0].

        Args:
            v: Confidence value.

        Returns:
            Validated confidence.

        Raises:
            ValueError: If confidence is out of range.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class PromptVersion(BaseModel):
    """Version record for a prompt.

    Tracks prompt evolution over time with semantic versioning.

    Attributes:
        prompt_id: ID of versioned prompt.
        version: Semantic version number (e.g., "1.2.0").
        prompt: The Prompt object at this version.
        analysis: PromptAnalysis for this version.
        optimization_result: OptimizationResult (if applicable).
        parent_version: Parent version number (for tracking lineage).
        created_at: Version creation timestamp.
        metadata: Additional version metadata.

    Example:
        >>> version = PromptVersion(
        ...     prompt_id="wf_001_llm_1",
        ...     version="1.0.0",
        ...     prompt=prompt_obj,
        ...     analysis=analysis_obj,
        ...     optimization_result=None,  # Baseline version
        ...     parent_version=None,
        ...     created_at=datetime.now(),
        ...     metadata={"author": "baseline"}
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    prompt_id: str = Field(..., description="Versioned prompt ID")
    version: str = Field(..., description="Semantic version (e.g., 1.2.0)")
    prompt: Prompt = Field(..., description="Prompt at this version")
    analysis: PromptAnalysis = Field(..., description="Analysis for this version")
    optimization_result: Optional[OptimizationResult] = Field(
        None, description="Optimization result (if optimized)"
    )
    parent_version: Optional[str] = Field(None, description="Parent version number")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Version creation timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("version")
    @classmethod
    def validate_semantic_version(cls, v: str) -> str:
        """Ensure version follows semantic versioning format.

        Args:
            v: Version string.

        Returns:
            Validated version string.

        Raises:
            ValueError: If version format is invalid.
        """
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(f"Version must be in format 'X.Y.Z', got '{v}'")

        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            if major < 0 or minor < 0 or patch < 0:
                raise ValueError("Version components must be non-negative")
        except ValueError as e:
            raise ValueError(f"Invalid version format '{v}': {str(e)}")

        return v

    def get_version_number(self) -> tuple:
        """Get version as tuple for comparison.

        Returns:
            Tuple of (major, minor, patch) integers.

        Example:
            >>> version = PromptVersion(version="1.2.3", ...)
            >>> version.get_version_number()
            (1, 2, 3)
        """
        parts = self.version.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    def is_baseline(self) -> bool:
        """Check if this is a baseline version (no parent).

        Returns:
            True if this is a baseline version.

        Example:
            >>> version.is_baseline()
            True
        """
        return self.parent_version is None


class OptimizationConfig(BaseModel):
    """Configuration for optimization engine.

    Attributes:
        strategies: List of strategies to try (or single strategy).
        min_confidence: Minimum confidence threshold to accept optimization.
        max_iterations: Maximum optimization iterations.
        score_threshold: Minimum score to skip optimization (0-100).
        analysis_rules: Custom analysis rules (optional).
        metadata: Additional configuration metadata.

    Example:
        >>> config = OptimizationConfig(
        ...     strategies=[OptimizationStrategy.CLARITY_FOCUS],
        ...     min_confidence=0.7,
        ...     max_iterations=3,
        ...     score_threshold=80.0,
        ...     analysis_rules={},
        ...     metadata={}
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    strategies: List[OptimizationStrategy] = Field(
        default_factory=lambda: [OptimizationStrategy.AUTO],
        description="Optimization strategies to use",
    )
    min_confidence: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to accept optimization",
    )
    max_iterations: int = Field(3, ge=1, description="Maximum optimization iterations")
    score_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Minimum score threshold to skip optimization (0-100). "
                    "Prompts with overall_score below this value will be optimized.",
    )
    analysis_rules: Dict[str, Any] = Field(
        default_factory=dict, description="Custom analysis rules"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("strategies")
    @classmethod
    def validate_strategies_not_empty(
        cls, v: List[OptimizationStrategy]
    ) -> List[OptimizationStrategy]:
        """Ensure at least one strategy is specified.

        Args:
            v: List of strategies.

        Returns:
            Validated list.

        Raises:
            ValueError: If strategies list is empty.
        """
        if not v:
            raise ValueError("At least one strategy must be specified")
        return v


# ============================================================================
# Test-Driven Optimization Models (Phase 1)
# ============================================================================


class ErrorDistribution(BaseModel):
    """Error type distribution from test execution.

    Provides detailed breakdown of error types to inform optimization strategies.
    For example, high timeout errors suggest clarity issues (prompt too complex),
    while high API errors suggest infrastructure issues.

    Attributes:
        timeout_errors: Number of timeout errors (suggests complex/long prompts).
        api_errors: Number of API/network errors (infrastructure issues).
        validation_errors: Number of validation failures (output format issues).
        llm_errors: Number of LLM-specific errors (rate limit, content filter).
        total_errors: Total error count (must equal sum of individual errors).

    Example:
        >>> errors = ErrorDistribution(
        ...     timeout_errors=5,
        ...     api_errors=2,
        ...     validation_errors=1,
        ...     llm_errors=0,
        ...     total_errors=8
        ... )
        >>> assert errors.total_errors == 8
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    timeout_errors: int = Field(default=0, ge=0, description="Timeout error count")
    api_errors: int = Field(default=0, ge=0, description="API error count")
    validation_errors: int = Field(
        default=0, ge=0, description="Validation error count"
    )
    llm_errors: int = Field(default=0, ge=0, description="LLM-specific error count")
    total_errors: int = Field(default=0, ge=0, description="Total error count")

    @field_validator("total_errors")
    @classmethod
    def validate_total_errors(cls, v: int, info) -> int:
        """Ensure total_errors matches sum of individual errors.

        Args:
            v: Total errors value.
            info: Field validation info.

        Returns:
            Validated total_errors.

        Raises:
            ValueError: If total doesn't match sum of components.
        """
        individual_sum = (
            info.data.get("timeout_errors", 0)
            + info.data.get("api_errors", 0)
            + info.data.get("validation_errors", 0)
            + info.data.get("llm_errors", 0)
        )
        if v != individual_sum:
            raise ValueError(
                f"total_errors ({v}) must equal sum of individual errors ({individual_sum})"
            )
        return v


class TestExecutionReport(BaseModel):
    """Aggregated test execution metrics for optimization decisions.

    This model serves as the contract between executor and optimizer modules.
    It is intentionally decoupled from executor's internal models (RunExecutionResult)
    to allow independent evolution of both modules.

    Design Rationale:
        - Placed in optimizer module (consumer defines contract)
        - Conversion via factory method from executor results
        - Strongly typed to catch errors at compile time
        - Includes percentile metrics for latency analysis

    Attributes:
        workflow_id: Workflow identifier.
        run_id: Test run identifier (from executor).
        total_tests: Total number of tests executed.
        successful_tests: Number of successful tests.
        failed_tests: Number of failed tests.
        success_rate: Success rate (0.0-1.0).
        avg_response_time_ms: Average response time in milliseconds.
        p95_response_time_ms: 95th percentile response time.
        p99_response_time_ms: 99th percentile response time.
        total_tokens: Total tokens consumed across all tests.
        avg_tokens_per_request: Average tokens per request.
        total_cost: Total cost in USD.
        cost_per_request: Average cost per request in USD.
        error_distribution: Error type distribution.
        executed_at: Execution timestamp.
        metadata: Additional execution metadata.

    Example:
        >>> report = TestExecutionReport(
        ...     workflow_id="wf_001",
        ...     run_id="run_001",
        ...     total_tests=100,
        ...     successful_tests=95,
        ...     failed_tests=5,
        ...     success_rate=0.95,
        ...     avg_response_time_ms=1200.0,
        ...     total_tokens=50000,
        ...     avg_tokens_per_request=500.0,
        ...     total_cost=2.5,
        ...     cost_per_request=0.025
        ... )
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Identifiers
    workflow_id: str = Field(..., description="Workflow ID")
    run_id: str = Field(..., description="Test run ID from executor")

    # Test counts
    total_tests: int = Field(..., ge=1, description="Total tests executed")
    successful_tests: int = Field(..., ge=0, description="Successful tests")
    failed_tests: int = Field(..., ge=0, description="Failed tests")

    # Success metrics
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0.0-1.0)")

    # Performance metrics
    avg_response_time_ms: float = Field(
        ..., ge=0.0, description="Average response time (ms)"
    )
    p95_response_time_ms: Optional[float] = Field(
        None, ge=0.0, description="95th percentile response time (ms)"
    )
    p99_response_time_ms: Optional[float] = Field(
        None, ge=0.0, description="99th percentile response time (ms)"
    )

    # Token and cost metrics
    total_tokens: int = Field(..., ge=0, description="Total tokens consumed")
    avg_tokens_per_request: float = Field(
        ..., ge=0.0, description="Average tokens per request"
    )
    total_cost: float = Field(..., ge=0.0, description="Total cost (USD)")
    cost_per_request: float = Field(
        ..., ge=0.0, description="Average cost per request (USD)"
    )

    # Error analysis
    error_distribution: ErrorDistribution = Field(
        default_factory=ErrorDistribution, description="Error type distribution"
    )

    # Metadata
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Execution timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("success_rate")
    @classmethod
    def validate_success_rate_consistency(cls, v: float, info) -> float:
        """Ensure success_rate matches successful_tests / total_tests.

        Args:
            v: Success rate value.
            info: Field validation info.

        Returns:
            Validated success_rate.

        Raises:
            ValueError: If success_rate inconsistent with test counts.
        """
        total = info.data.get("total_tests")
        successful = info.data.get("successful_tests")

        if total and successful is not None:
            expected_rate = successful / total
            # Allow small floating point differences
            if abs(v - expected_rate) > 0.001:
                raise ValueError(
                    f"success_rate ({v:.3f}) inconsistent with "
                    f"successful_tests/total_tests ({expected_rate:.3f})"
                )
        return v

    @classmethod
    def from_executor_result(cls, executor_result: Any) -> "TestExecutionReport":
        """Convert executor's RunExecutionResult to TestExecutionReport.

        This factory method decouples executor and optimizer modules by
        providing a conversion layer. It extracts and transforms executor's
        internal statistics into optimizer's contract format.

        Args:
            executor_result: RunExecutionResult from executor.models

        Returns:
            TestExecutionReport for optimizer consumption

        Raises:
            ValueError: If executor_result is None or missing required fields
            ImportError: If executor module is not available

        Example:
            >>> from src.executor.executor_service import ExecutorService
            >>> service = ExecutorService()
            >>> exec_result = service.scheduler.run_manifest(manifest)
            >>> test_report = TestExecutionReport.from_executor_result(exec_result)
            >>> print(f"Success rate: {test_report.success_rate:.2%}")
        """
        if executor_result is None:
            raise ValueError("executor_result cannot be None")

        # Import here to avoid circular dependency
        try:
            from src.executor.models import TaskStatus
        except ImportError as e:
            raise ImportError(
                "executor module required for conversion. "
                "Ensure src.executor is available."
            ) from e

        stats = executor_result.statistics

        # Calculate percentiles from task_results
        response_times = [
            tr.execution_time * 1000  # Convert seconds to ms
            for tr in executor_result.task_results
            if tr.execution_time > 0
        ]

        p95 = None
        p99 = None
        if response_times:
            response_times_sorted = sorted(response_times)
            p95_idx = int(len(response_times_sorted) * 0.95)
            p99_idx = int(len(response_times_sorted) * 0.99)
            if p95_idx < len(response_times_sorted):
                p95 = response_times_sorted[p95_idx]
            if p99_idx < len(response_times_sorted):
                p99 = response_times_sorted[p99_idx]

        # Analyze error distribution
        # Note: In executor, failed_tasks, timeout_tasks, and error_tasks are separate counts
        # We map timeout_tasks -> timeout_errors, error_tasks -> api_errors
        # total_errors should only be sum of these components, not failed_tasks
        error_dist = ErrorDistribution(
            timeout_errors=stats.timeout_tasks,
            api_errors=stats.error_tasks,
            validation_errors=0,  # Executor doesn't track this separately
            llm_errors=0,  # Would need to parse error messages
            total_errors=stats.timeout_tasks + stats.error_tasks,
        )

        return cls(
            workflow_id=executor_result.workflow_id,
            run_id=executor_result.run_id,
            total_tests=stats.total_tasks,
            successful_tests=stats.succeeded_tasks,
            failed_tests=stats.failed_tasks + stats.timeout_tasks + stats.error_tasks,
            success_rate=stats.success_rate,
            avg_response_time_ms=stats.avg_execution_time * 1000,  # Convert to ms
            p95_response_time_ms=p95,
            p99_response_time_ms=p99,
            total_tokens=stats.total_tokens,
            avg_tokens_per_request=(
                stats.total_tokens / stats.completed_tasks
                if stats.completed_tasks > 0
                else 0.0
            ),
            total_cost=stats.total_cost,
            cost_per_request=(
                stats.total_cost / stats.completed_tasks
                if stats.completed_tasks > 0
                else 0.0
            ),
            error_distribution=error_dist,
            executed_at=executor_result.finished_at,
            metadata={
                "total_retries": stats.total_retries,
                "cancelled_tasks": stats.cancelled_tasks,
            },
        )

    def has_timeout_errors(self) -> bool:
        """Check if execution had timeout errors.

        Returns:
            True if any timeout errors occurred.

        Example:
            >>> if report.has_timeout_errors():
            ...     print("Consider optimizing prompt for clarity")
        """
        return self.error_distribution.timeout_errors > 0

    def has_api_errors(self) -> bool:
        """Check if execution had API errors.

        Returns:
            True if any API errors occurred.

        Example:
            >>> if report.has_api_errors():
            ...     print("Infrastructure issues detected")
        """
        return self.error_distribution.api_errors > 0

    def get_error_rate(self) -> float:
        """Calculate overall error rate.

        Returns:
            Error rate as float (0.0-1.0).

        Example:
            >>> error_rate = report.get_error_rate()
            >>> print(f"Error rate: {error_rate:.2%}")
        """
        return 1.0 - self.success_rate

    def get_timeout_error_rate(self) -> float:
        """Calculate timeout error rate.

        Returns:
            Timeout error rate as float (0.0-1.0).

        Example:
            >>> if report.get_timeout_error_rate() > 0.1:
            ...     print("More than 10% timeouts - prompt may be too complex")
        """
        if self.total_tests == 0:
            return 0.0
        return self.error_distribution.timeout_errors / self.total_tests
