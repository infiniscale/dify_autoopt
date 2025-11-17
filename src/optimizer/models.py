"""
Optimizer Module - Data Models

Date: 2025-11-17
Author: backend-developer
Description: Pydantic V2 models for prompts, analysis, optimization, and versioning.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
    """Optimization strategy names."""

    CLARITY_FOCUS = "clarity_focus"
    EFFICIENCY_FOCUS = "efficiency_focus"
    STRUCTURE_FOCUS = "structure_focus"
    AUTO = "auto"  # Auto-select based on analysis


# ============================================================================
# Core Models
# ============================================================================


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
        """Ensure prompt text is not empty.

        Args:
            v: Prompt text to validate.

        Returns:
            Validated text.

        Raises:
            ValueError: If text is empty or whitespace only.
        """
        if not v or not v.strip():
            raise ValueError("Prompt text cannot be empty")
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
        changes: List of change descriptions.
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
        ...     changes=["Added specific output format"],
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
    changes: List[str] = Field(default_factory=list, description="List of changes")
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
        analysis_rules: Custom analysis rules (optional).
        metadata: Additional configuration metadata.

    Example:
        >>> config = OptimizationConfig(
        ...     strategies=[OptimizationStrategy.CLARITY_FOCUS],
        ...     min_confidence=0.7,
        ...     max_iterations=3,
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
