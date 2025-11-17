"""
Optimizer Module - Prompt Optimization and Version Management

Date: 2025-11-17
Author: backend-developer
Description: Public API for prompt extraction, analysis, optimization, and versioning.

This module provides:
    - PromptExtractor: Extract prompts from workflow DSL
    - PromptAnalyzer: Analyze prompt quality
    - OptimizationEngine: Generate optimized prompts
    - VersionManager: Track prompt versions
    - OptimizerService: High-level orchestration facade
    - Data models: Prompt, PromptAnalysis, OptimizationResult, etc.
    - Convenience functions: optimize_workflow()

Example Usage:
    >>> from src.optimizer import OptimizerService, optimize_workflow
    >>>
    >>> # High-level API
    >>> patches = optimize_workflow(
    ...     workflow_id="wf_001",
    ...     catalog=workflow_catalog,
    ...     strategy="clarity_focus"
    ... )
    >>>
    >>> # Low-level API
    >>> service = OptimizerService(catalog=workflow_catalog)
    >>> report = service.analyze_workflow("wf_001")
    >>> print(f"Average score: {report['average_score']}")
"""

from typing import Any, Dict, List, Optional

from src.config.models import PromptPatch, WorkflowCatalog

# Core services
from .optimizer_service import OptimizerService
from .prompt_analyzer import PromptAnalyzer
from .prompt_extractor import PromptExtractor
from .optimization_engine import OptimizationEngine
from .version_manager import VersionManager
from .prompt_patch_engine import PromptPatchEngine

# Data models
from .models import (
    Prompt,
    PromptAnalysis,
    PromptIssue,
    PromptSuggestion,
    OptimizationResult,
    PromptVersion,
    OptimizationConfig,
    IssueSeverity,
    IssueType,
    SuggestionType,
    OptimizationStrategy,
)

# Interfaces
from .interfaces.llm_client import LLMClient, StubLLMClient
from .interfaces.storage import VersionStorage, InMemoryStorage

# Exceptions
from .exceptions import (
    OptimizerError,
    ExtractionError,
    WorkflowNotFoundError,
    NodeNotFoundError,
    DSLParseError,
    AnalysisError,
    ScoringError,
    OptimizationError,
    InvalidStrategyError,
    OptimizationFailedError,
    VersionError,
    VersionConflictError,
    VersionNotFoundError,
    ValidationError,
    ConfigError,
)


__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Services
    "OptimizerService",
    "PromptExtractor",
    "PromptAnalyzer",
    "OptimizationEngine",
    "VersionManager",
    "PromptPatchEngine",
    # Models
    "Prompt",
    "PromptAnalysis",
    "PromptIssue",
    "PromptSuggestion",
    "OptimizationResult",
    "PromptVersion",
    "OptimizationConfig",
    "IssueSeverity",
    "IssueType",
    "SuggestionType",
    "OptimizationStrategy",
    # Interfaces
    "LLMClient",
    "StubLLMClient",
    "VersionStorage",
    "InMemoryStorage",
    # Exceptions
    "OptimizerError",
    "ExtractionError",
    "WorkflowNotFoundError",
    "NodeNotFoundError",
    "DSLParseError",
    "AnalysisError",
    "ScoringError",
    "OptimizationError",
    "InvalidStrategyError",
    "OptimizationFailedError",
    "VersionError",
    "VersionConflictError",
    "VersionNotFoundError",
    "ValidationError",
    "ConfigError",
    # Convenience functions
    "optimize_workflow",
    "analyze_workflow",
]


# ============================================================================
# Convenience Functions
# ============================================================================


def optimize_workflow(
    workflow_id: str,
    catalog: WorkflowCatalog,
    strategy: str = "auto",
    baseline_metrics: Optional[Dict[str, Any]] = None,
    config: Optional[OptimizationConfig] = None,
    llm_client: Optional[LLMClient] = None,
    storage: Optional[VersionStorage] = None,
) -> List[PromptPatch]:
    """Convenience function for workflow optimization.

    High-level API that handles the complete optimization cycle:
        1. Extract prompts from workflow
        2. Analyze prompt quality
        3. Optimize low-scoring prompts
        4. Generate PromptPatch objects

    Args:
        workflow_id: Workflow to optimize.
        catalog: WorkflowCatalog containing workflow metadata.
        strategy: Optimization strategy (auto, clarity_focus, efficiency_focus, structure_focus).
        baseline_metrics: Optional baseline performance metrics.
        config: Optional optimization configuration.
        llm_client: Optional LLM client (defaults to StubLLMClient).
        storage: Optional version storage (defaults to InMemoryStorage).

    Returns:
        List of PromptPatch objects to apply in test plan.

    Raises:
        WorkflowNotFoundError: If workflow doesn't exist.
        OptimizerError: If optimization fails.

    Example:
        >>> from src.config import ConfigLoader
        >>> from src.optimizer import optimize_workflow
        >>>
        >>> loader = ConfigLoader()
        >>> catalog = loader.load_workflow_catalog("config/workflows.yaml")
        >>>
        >>> patches = optimize_workflow(
        ...     workflow_id="wf_001",
        ...     catalog=catalog,
        ...     strategy="clarity_focus"
        ... )
        >>>
        >>> print(f"Generated {len(patches)} prompt patches")
    """
    service = OptimizerService(
        catalog=catalog,
        llm_client=llm_client,
        storage=storage,
    )

    return service.run_optimization_cycle(
        workflow_id=workflow_id,
        baseline_metrics=baseline_metrics,
        strategy=strategy,
        config=config,
    )


def analyze_workflow(
    workflow_id: str,
    catalog: WorkflowCatalog,
) -> Dict[str, Any]:
    """Convenience function for workflow analysis.

    Analyzes all prompts in a workflow without optimizing.

    Args:
        workflow_id: Workflow to analyze.
        catalog: WorkflowCatalog containing workflow metadata.

    Returns:
        Analysis report dictionary with:
            - workflow_id: Workflow identifier
            - prompt_count: Number of prompts found
            - average_score: Average quality score
            - prompts: List of individual prompt analyses
            - needs_optimization: Boolean flag

    Raises:
        WorkflowNotFoundError: If workflow doesn't exist.

    Example:
        >>> from src.optimizer import analyze_workflow
        >>>
        >>> report = analyze_workflow("wf_001", catalog)
        >>> print(f"Average score: {report['average_score']:.1f}")
        >>> print(f"Needs optimization: {report['needs_optimization']}")
        >>>
        >>> for prompt in report['prompts']:
        ...     print(f"{prompt['node_id']}: {prompt['overall_score']:.1f}")
    """
    service = OptimizerService(catalog=catalog)
    return service.analyze_workflow(workflow_id)
