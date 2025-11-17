"""
Optimizer Module - Optimizer Service

Date: 2025-11-17
Author: backend-developer
Description: High-level facade service for prompt optimization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config.models import (
    PromptPatch,
    PromptSelector,
    PromptStrategy,
    WorkflowCatalog,
)

from .exceptions import OptimizerError, WorkflowNotFoundError
from .interfaces.llm_client import LLMClient, StubLLMClient
from .interfaces.storage import InMemoryStorage, VersionStorage
from .models import (
    OptimizationConfig,
    OptimizationResult,
    OptimizationStrategy,
    Prompt,
    PromptAnalysis,
)
from .optimization_engine import OptimizationEngine
from .prompt_analyzer import PromptAnalyzer
from .prompt_extractor import PromptExtractor
from .version_manager import VersionManager


class OptimizerService:
    """High-level optimizer service (facade).

    Orchestrates all optimizer components to provide a simple API
    for prompt extraction, analysis, optimization, and version management.

    Workflow:
        1. Extract prompts from workflow DSL
        2. Analyze prompt quality
        3. Optimize low-scoring prompts
        4. Create version records
        5. Generate PromptPatch objects for test plan

    Attributes:
        _catalog: WorkflowCatalog for workflow access.
        _extractor: PromptExtractor for prompt extraction.
        _analyzer: PromptAnalyzer for quality analysis.
        _engine: OptimizationEngine for optimization.
        _version_manager: VersionManager for version tracking.
        _logger: Loguru logger instance.

    Example:
        >>> service = OptimizerService(catalog)
        >>> patches = service.run_optimization_cycle(
        ...     workflow_id="wf_001",
        ...     strategy="clarity_focus"
        ... )
    """

    def __init__(
        self,
        catalog: Optional[WorkflowCatalog] = None,
        llm_client: Optional[LLMClient] = None,
        storage: Optional[VersionStorage] = None,
        custom_logger: Optional[Any] = None,
    ) -> None:
        """Initialize OptimizerService.

        Args:
            catalog: WorkflowCatalog for workflow access (optional).
            llm_client: LLM client for analysis (defaults to StubLLMClient).
            storage: Version storage backend (defaults to InMemoryStorage).
            custom_logger: Optional custom logger instance.
        """
        self._catalog = catalog
        self._logger = custom_logger or logger.bind(module="optimizer.service")

        # Initialize components
        self._extractor = PromptExtractor(custom_logger=self._logger)
        self._analyzer = PromptAnalyzer(
            llm_client=llm_client or StubLLMClient(),
            custom_logger=self._logger,
        )
        self._engine = OptimizationEngine(
            analyzer=self._analyzer,
            llm_client=llm_client or StubLLMClient(),
            custom_logger=self._logger,
        )
        self._version_manager = VersionManager(
            storage=storage or InMemoryStorage(),
            custom_logger=self._logger,
        )

        self._logger.info("OptimizerService initialized")

    def run_optimization_cycle(
        self,
        workflow_id: str,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        strategy: str = "auto",
        config: Optional[OptimizationConfig] = None,
    ) -> List[PromptPatch]:
        """Run complete optimization cycle for a workflow.

        This is the main entry point for optimization. It:
            1. Extracts prompts from workflow
            2. Analyzes each prompt
            3. Optimizes prompts that need improvement
            4. Creates version records
            5. Generates PromptPatch objects

        Args:
            workflow_id: Workflow to optimize.
            baseline_metrics: Optional baseline performance metrics.
            strategy: Optimization strategy (auto, clarity_focus, efficiency_focus, structure_focus).
            config: Optional optimization configuration.

        Returns:
            List of PromptPatch objects to apply in test plan.

        Raises:
            WorkflowNotFoundError: If workflow doesn't exist.
            OptimizerError: If optimization fails.

        Example:
            >>> patches = service.run_optimization_cycle(
            ...     workflow_id="wf_001",
            ...     strategy="clarity_focus"
            ... )
            >>> # Apply patches to test plan
            >>> test_plan.workflows[0].prompt_optimization[0].nodes.extend(patches)
        """
        self._logger.info(
            f"Starting optimization cycle for workflow '{workflow_id}' "
            f"with strategy '{strategy}'"
        )

        try:
            # 1. Extract prompts
            prompts = self._extract_prompts(workflow_id)

            if not prompts:
                self._logger.info(f"No prompts found in workflow '{workflow_id}'")
                return []

            self._logger.info(f"Extracted {len(prompts)} prompts")

            # 2. Analyze and optimize each prompt
            patches: List[PromptPatch] = []

            for prompt in prompts:
                # Analyze
                analysis = self._analyzer.analyze_prompt(prompt)

                # Create baseline version
                self._version_manager.create_version(
                    prompt=prompt,
                    analysis=analysis,
                    optimization_result=None,
                    parent_version=None,
                )

                # Determine if optimization is needed
                if self._should_optimize(analysis, baseline_metrics, config):
                    # Select strategy
                    selected_strategy = (
                        self._select_strategy(analysis, baseline_metrics)
                        if strategy == "auto"
                        else strategy
                    )

                    # Optimize
                    result = self.optimize_single_prompt(prompt, selected_strategy)

                    # Create optimized version
                    self._version_manager.create_version(
                        prompt=Prompt(
                            id=prompt.id,
                            workflow_id=prompt.workflow_id,
                            node_id=prompt.node_id,
                            node_type=prompt.node_type,
                            text=result.optimized_prompt,
                            role=prompt.role,
                            variables=prompt.variables,
                            context=prompt.context,
                            extracted_at=prompt.extracted_at,
                        ),
                        analysis=self._analyzer.analyze_prompt(
                            Prompt(
                                id=prompt.id,
                                workflow_id=prompt.workflow_id,
                                node_id=prompt.node_id,
                                node_type=prompt.node_type,
                                text=result.optimized_prompt,
                                role=prompt.role,
                                variables=prompt.variables,
                                context=prompt.context,
                                extracted_at=prompt.extracted_at,
                            )
                        ),
                        optimization_result=result,
                        parent_version="1.0.0",
                    )

                    # Generate patch
                    patch = self._create_prompt_patch(prompt, result)
                    patches.append(patch)

                    self._logger.info(
                        f"Optimized prompt '{prompt.id}': "
                        f"improvement={result.improvement_score:.1f}, "
                        f"confidence={result.confidence:.2f}"
                    )
                else:
                    self._logger.debug(
                        f"Prompt '{prompt.id}' does not need optimization "
                        f"(score={analysis.overall_score:.1f})"
                    )

            self._logger.info(
                f"Optimization cycle complete: generated {len(patches)} patches"
            )

            return patches

        except WorkflowNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Optimization cycle failed: {str(e)}")
            raise OptimizerError(
                message=f"Optimization cycle failed for workflow '{workflow_id}'",
                error_code="OPT-SVC-001",
                context={"workflow_id": workflow_id, "error": str(e)},
            )

    def optimize_single_prompt(
        self,
        prompt: Prompt,
        strategy: str = "auto",
    ) -> OptimizationResult:
        """Optimize a single prompt.

        Args:
            prompt: Prompt to optimize.
            strategy: Strategy to use (auto, clarity_focus, etc.).

        Returns:
            OptimizationResult with optimized prompt.

        Raises:
            OptimizerError: If optimization fails.

        Example:
            >>> result = service.optimize_single_prompt(prompt, "clarity_focus")
            >>> print(result.optimized_prompt)
        """
        self._logger.info(f"Optimizing single prompt '{prompt.id}'")

        try:
            # Analyze first if strategy is auto
            if strategy == "auto":
                analysis = self._analyzer.analyze_prompt(prompt)
                strategy = self._select_strategy(analysis, None)

            # Optimize
            result = self._engine.optimize(prompt, strategy)

            return result

        except Exception as e:
            self._logger.error(f"Single prompt optimization failed: {str(e)}")
            raise OptimizerError(
                message=f"Failed to optimize prompt '{prompt.id}'",
                error_code="OPT-SVC-002",
                context={"prompt_id": prompt.id, "error": str(e)},
            )

    def analyze_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Analyze all prompts in a workflow without optimizing.

        Args:
            workflow_id: Workflow to analyze.

        Returns:
            Analysis report dictionary.

        Raises:
            WorkflowNotFoundError: If workflow doesn't exist.

        Example:
            >>> report = service.analyze_workflow("wf_001")
            >>> print(report["average_score"])
        """
        self._logger.info(f"Analyzing workflow '{workflow_id}'")

        # Extract prompts
        prompts = self._extract_prompts(workflow_id)

        if not prompts:
            return {
                "workflow_id": workflow_id,
                "prompt_count": 0,
                "average_score": 0.0,
                "prompts": [],
            }

        # Analyze each prompt
        analyses: List[Dict[str, Any]] = []
        total_score = 0.0

        for prompt in prompts:
            analysis = self._analyzer.analyze_prompt(prompt)
            total_score += analysis.overall_score

            analyses.append(
                {
                    "prompt_id": prompt.id,
                    "node_id": prompt.node_id,
                    "overall_score": analysis.overall_score,
                    "clarity_score": analysis.clarity_score,
                    "efficiency_score": analysis.efficiency_score,
                    "issues_count": len(analysis.issues),
                    "suggestions_count": len(analysis.suggestions),
                    "issues": [
                        {
                            "severity": issue.severity.value,
                            "type": issue.type.value,
                            "description": issue.description,
                        }
                        for issue in analysis.issues
                    ],
                }
            )

        average_score = total_score / len(prompts)

        report = {
            "workflow_id": workflow_id,
            "prompt_count": len(prompts),
            "average_score": average_score,
            "prompts": analyses,
            "needs_optimization": average_score < 80.0,
        }

        self._logger.info(
            f"Analysis complete: {len(prompts)} prompts, "
            f"average score={average_score:.1f}"
        )

        return report

    def _extract_prompts(self, workflow_id: str) -> List[Prompt]:
        """Extract prompts from workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            List of extracted prompts.

        Raises:
            WorkflowNotFoundError: If workflow doesn't exist in catalog.
        """
        if self._catalog is None:
            raise OptimizerError(
                message="Cannot extract prompts: WorkflowCatalog not provided",
                error_code="OPT-SVC-003",
            )

        # Get workflow from catalog
        workflow = self._catalog.get_workflow(workflow_id)
        if workflow is None:
            raise WorkflowNotFoundError(workflow_id)

        # Load DSL file
        dsl_path = workflow.dsl_path_resolved
        if not dsl_path.exists():
            raise OptimizerError(
                message=f"DSL file not found: {dsl_path}",
                error_code="OPT-SVC-004",
                context={"workflow_id": workflow_id, "dsl_path": str(dsl_path)},
            )

        # Load and parse DSL
        dsl_dict = self._extractor.load_dsl_file(dsl_path)

        # Extract prompts
        prompts = self._extractor.extract_from_workflow(dsl_dict, workflow_id)

        return prompts

    def _should_optimize(
        self,
        analysis: PromptAnalysis,
        baseline_metrics: Optional[Dict[str, Any]],
        config: Optional[OptimizationConfig],
    ) -> bool:
        """Determine if a prompt should be optimized.

        Args:
            analysis: Prompt analysis.
            baseline_metrics: Baseline performance metrics.
            config: Optimization configuration.

        Returns:
            True if optimization is recommended.
        """
        # Use config threshold if provided, otherwise default
        score_threshold = config.score_threshold if config else 80.0

        # Optimize if score is below threshold
        if analysis.overall_score < score_threshold:
            return True

        # Optimize if there are critical issues
        critical_issues = [
            issue for issue in analysis.issues if issue.severity.value == "critical"
        ]
        if critical_issues:
            return True

        # Optimize based on baseline metrics (if provided)
        if baseline_metrics and "success_rate" in baseline_metrics:
            if baseline_metrics["success_rate"] < 0.8:
                return True

        return False

    def _select_strategy(
        self,
        analysis: PromptAnalysis,
        baseline_metrics: Optional[Dict[str, Any]],
    ) -> str:
        """Auto-select optimization strategy based on analysis.

        Args:
            analysis: Prompt analysis.
            baseline_metrics: Optional baseline metrics.

        Returns:
            Strategy name.
        """
        # Prioritize based on scores
        clarity_score = analysis.clarity_score
        efficiency_score = analysis.efficiency_score

        # If clarity is significantly lower, focus on clarity
        if clarity_score < efficiency_score - 10:
            return "clarity_focus"

        # If efficiency is significantly lower, focus on efficiency
        if efficiency_score < clarity_score - 10:
            return "efficiency_focus"

        # If both are low, focus on structure
        if clarity_score < 70 and efficiency_score < 70:
            return "structure_focus"

        # Default to clarity focus
        return "clarity_focus"

    def _create_prompt_patch(
        self,
        prompt: Prompt,
        result: OptimizationResult,
    ) -> PromptPatch:
        """Create PromptPatch from optimization result.

        Args:
            prompt: Original prompt.
            result: Optimization result.

        Returns:
            PromptPatch object for test plan.
        """
        # Create selector to target this specific node
        selector = PromptSelector(by_id=prompt.node_id)

        # Create strategy (replace mode with optimized content)
        strategy = PromptStrategy(
            mode="replace",
            content=result.optimized_prompt,
        )

        # Create and return patch
        patch = PromptPatch(
            selector=selector,
            strategy=strategy,
        )

        return patch

    def get_version_history(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get version history for a prompt.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            List of version dictionaries.

        Example:
            >>> history = service.get_version_history("wf_001_llm_1")
            >>> for v in history:
            ...     print(f"v{v['version']}: {v['score']}")
        """
        versions = self._version_manager.get_version_history(prompt_id)

        history = [
            {
                "version": v.version,
                "score": v.analysis.overall_score,
                "clarity": v.analysis.clarity_score,
                "efficiency": v.analysis.efficiency_score,
                "created_at": v.created_at.isoformat(),
                "author": v.metadata.get("author", "unknown"),
                "is_optimized": v.optimization_result is not None,
            }
            for v in versions
        ]

        return history
