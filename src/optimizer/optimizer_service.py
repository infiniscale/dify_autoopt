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

from .config import LLMConfig, LLMProvider
from .exceptions import OptimizerError, WorkflowNotFoundError
from .interfaces.llm_client import LLMClient, StubLLMClient
from .interfaces.llm_providers import OpenAIClient
from .interfaces.storage import InMemoryStorage, VersionStorage
from .models import (
    OptimizationConfig,
    OptimizationResult,
    OptimizationStrategy,
    Prompt,
    PromptAnalysis,
    TestExecutionReport,
)
from .optimization_engine import OptimizationEngine
from .prompt_analyzer import PromptAnalyzer
from .prompt_extractor import PromptExtractor
from .scoring_rules import ScoringRules
from .version_manager import VersionManager
from .utils.variable_extractor import VariableExtractor


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
        scoring_rules: Optional[ScoringRules] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> None:
        """Initialize OptimizerService.

        Args:
            catalog: WorkflowCatalog for workflow access (optional).
            llm_client: LLM client for analysis (optional, deprecated - use llm_config instead).
            storage: Version storage backend (defaults to InMemoryStorage).
            custom_logger: Optional custom logger instance.
            scoring_rules: Optional ScoringRules for configurable thresholds.
            llm_config: LLM configuration for LLM-driven optimization (optional).
        """
        self._catalog = catalog
        self._logger = custom_logger or logger.bind(module="optimizer.service")
        self._scoring_rules = scoring_rules or ScoringRules()

        # Initialize LLM client (priority: explicit client > config > stub)
        if llm_client is not None:
            # Backward compatibility: explicit client parameter takes precedence
            self._llm_client = llm_client
            self._llm_config = None
            self._logger.info("Using explicitly provided LLM client")
        elif llm_config is not None:
            # Create LLM client from config
            self._llm_config = llm_config
            self._llm_client = self._create_llm_client(llm_config)
        else:
            # Default to StubLLMClient
            self._llm_client = StubLLMClient()
            self._llm_config = None
            self._logger.info("Using default StubLLMClient (rule-based optimization)")

        # Initialize components
        self._extractor = PromptExtractor(custom_logger=self._logger)
        self._analyzer = PromptAnalyzer(
            llm_client=self._llm_client,
            custom_logger=self._logger,
        )
        self._engine = OptimizationEngine(
            analyzer=self._analyzer,
            llm_client=self._llm_client,
            custom_logger=self._logger,
        )
        self._version_manager = VersionManager(
            storage=storage or InMemoryStorage(),
            scoring_rules=self._scoring_rules,
            custom_logger=self._logger,
        )

        # Analysis cache to avoid re-analyzing identical prompts
        self._analysis_cache: Dict[str, PromptAnalysis] = {}

        self._logger.info(
            f"OptimizerService initialized "
            f"(LLM enabled: {not isinstance(self._llm_client, StubLLMClient)})"
        )

    def _create_llm_client(self, config: LLMConfig) -> LLMClient:
        """Create LLM client instance based on configuration.

        Args:
            config: LLM configuration.

        Returns:
            LLMClient instance (StubLLMClient for unsupported providers).

        Example:
            >>> config = LLMConfig(provider=LLMProvider.OPENAI)
            >>> client = self._create_llm_client(config)
        """
        if config.provider == LLMProvider.STUB:
            self._logger.info("Using STUB provider (rule-based optimization)")
            return StubLLMClient()
        elif config.provider == LLMProvider.OPENAI:
            self._logger.info(f"Initializing OpenAI client with model {config.model}")
            try:
                return OpenAIClient(config)
            except Exception as e:
                self._logger.error(f"Failed to initialize OpenAI client: {e}")
                self._logger.warning("Falling back to StubLLMClient")
                return StubLLMClient()
        elif config.provider == LLMProvider.ANTHROPIC:
            # TODO: Phase 2 implementation
            self._logger.warning("Anthropic provider not yet implemented, using STUB")
            return StubLLMClient()
        elif config.provider == LLMProvider.LOCAL:
            # TODO: Phase 2 implementation
            self._logger.warning("Local LLM provider not yet implemented, using STUB")
            return StubLLMClient()
        else:
            self._logger.warning(f"Unknown provider {config.provider}, using STUB")
            return StubLLMClient()


    def run_optimization_cycle(
        self,
        workflow_id: str,
        test_results: Optional[TestExecutionReport] = None,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
        config: Optional[OptimizationConfig] = None,
    ) -> List[PromptPatch]:
        """Run complete optimization cycle for a workflow.

        This is the main entry point for optimization. It:
            1. Extracts prompts from workflow
            2. Analyzes each prompt
            3. Optimizes prompts that need improvement
            4. Creates version records
            5. Generates PromptPatch objects

        Supports both rule-based and LLM strategies:
        - Rule strategies: clarity_focus, efficiency_focus, structure_focus, auto
        - LLM strategies: llm_guided, llm_clarity, llm_efficiency, hybrid

        Behavior:
            - If `strategy` is provided: Single-strategy, single-iteration mode (backward compatible)
            - If `config` is provided: Multi-strategy, multi-iteration mode (new feature)
            - If both provided: `strategy` parameter takes precedence (backward compatibility priority)
            - If neither provided: Default config with AUTO strategy

        Test-Driven Optimization (NEW):
            - If `test_results` provided: Automatic optimization based on test metrics
            - If `baseline_metrics` provided: Auto-converted to test_results for backward compatibility
            - Test metrics inform both optimization triggers and strategy selection

        Args:
            workflow_id: Workflow to optimize.
            test_results: Test execution report from executor (primary parameter).
            baseline_metrics: (DEPRECATED) Legacy baseline metrics dict, use test_results instead.
            strategy: (LEGACY) Single strategy name (auto, clarity_focus, efficiency_focus, structure_focus).
            config: (NEW) Full optimization configuration with strategies, iterations, confidence.

        Returns:
            List of PromptPatch objects to apply in test plan.

        Raises:
            WorkflowNotFoundError: If workflow doesn't exist.
            OptimizerError: If optimization fails.

        Example:
            >>> # Test-driven mode (automatic optimization)
            >>> from src.executor import ExecutorService
            >>> from src.optimizer.models import TestExecutionReport
            >>> executor = ExecutorService()
            >>> test_result = executor.scheduler.run_manifest(manifest)
            >>> test_report = TestExecutionReport.from_executor_result(test_result)
            >>> patches = service.run_optimization_cycle(
            ...     workflow_id="wf_001",
            ...     test_results=test_report
            ... )
            >>> # Legacy mode (backward compatible)
            >>> patches = service.run_optimization_cycle(
            ...     workflow_id="wf_001",
            ...     strategy="clarity_focus"
            ... )
            >>> # New mode with config
            >>> config = OptimizationConfig(
            ...     strategies=[OptimizationStrategy.CLARITY_FOCUS],
            ...     max_iterations=3,
            ...     min_confidence=0.7
            ... )
            >>> patches = service.run_optimization_cycle("wf_001", config=config)
        """
        self._logger.info(
            f"Starting optimization cycle for workflow '{workflow_id}' "
            f"(LLM enabled: {not isinstance(self._llm_client, StubLLMClient)})"
        )

        # Step 0: Auto-convert legacy baseline_metrics to test_results (backward compatibility)
        effective_test_results = test_results
        if test_results is None and baseline_metrics is not None:
            self._logger.info("Converting legacy baseline_metrics to test_results")
            effective_test_results = self._scoring_rules._convert_legacy_baseline_metrics(
                baseline_metrics
            )

        # Step 1: Resolve effective configuration
        if strategy is not None:
            # Backward compatibility mode: strategy parameter overrides config
            self._logger.info(
                f"Using legacy single-strategy mode with strategy='{strategy}'"
            )
            # Handle "auto" strategy by selecting based on analysis
            if strategy == "auto":
                # For "auto", we'll use the default config with AUTO strategy enum
                effective_config = OptimizationConfig(
                    strategies=[OptimizationStrategy.AUTO],
                    max_iterations=1,  # Legacy behavior: single run
                    min_confidence=0.0,  # Legacy behavior: accept all results
                    score_threshold=config.score_threshold if config else 80.0,
                )
            else:
                effective_config = OptimizationConfig(
                    strategies=[OptimizationStrategy(strategy)],
                    max_iterations=1,  # Legacy behavior: single run
                    min_confidence=0.0,  # Legacy behavior: accept all results
                    score_threshold=config.score_threshold if config else 80.0,
                )
        elif config is not None:
            # New multi-strategy mode
            self._logger.info(
                f"Using multi-strategy mode with {len(config.strategies)} strategies, "
                f"{config.max_iterations} iterations, min_confidence={config.min_confidence}"
            )
            effective_config = config
        else:
            # Default configuration
            self._logger.info("Using default configuration")
            effective_config = OptimizationConfig()  # Uses model defaults

        try:
            # Step 2: Extract prompts
            prompts = self._extract_prompts(workflow_id)

            if not prompts:
                self._logger.info(f"No prompts found in workflow '{workflow_id}'")
                return []

            self._logger.info(f"Extracted {len(prompts)} prompts")

            # Step 3: Optimize each prompt
            patches: List[PromptPatch] = []

            for prompt in prompts:
                # 3.1: Analyze baseline (with caching)
                baseline_analysis = self._get_or_analyze(prompt)

                # Create baseline version
                self._version_manager.create_version(
                    prompt=prompt,
                    analysis=baseline_analysis,
                    optimization_result=None,
                    parent_version=None,
                )

                # 3.2: Check if optimization needed
                if not self._should_optimize(
                    baseline_analysis, effective_test_results, effective_config
                ):
                    self._logger.debug(
                        f"Prompt '{prompt.id}' does not need optimization "
                        f"(score={baseline_analysis.overall_score:.1f})"
                    )
                    continue

                # 3.3: Try all strategies
                best_result = None

                for strategy_enum in effective_config.strategies:
                    strategy_name = strategy_enum.value

                    # Handle AUTO strategy: select based on baseline analysis
                    if strategy_name == "auto":
                        strategy_name = self._select_strategy(
                            baseline_analysis, effective_test_results
                        )
                        self._logger.info(
                            f"Auto-selected strategy '{strategy_name}' for prompt '{prompt.id}'"
                        )
                    else:
                        self._logger.info(
                            f"Trying strategy '{strategy_name}' for prompt '{prompt.id}'"
                        )

                    # Optimize with iterations
                    result = self._optimize_with_iterations(
                        prompt=prompt,
                        strategy=strategy_name,
                        max_iterations=effective_config.max_iterations,
                        min_confidence=effective_config.min_confidence,
                    )

                    # Update best result
                    if result and self._is_better_result(
                        result, best_result, effective_config.min_confidence
                    ):
                        best_result = result
                        self._logger.info(
                            f"New best result: strategy='{strategy_name}', "
                            f"confidence={result.confidence:.2f}, "
                            f"improvement={result.improvement_score:.1f}"
                        )

                # 3.4: Accept best result if meets confidence threshold
                if (
                    best_result
                    and best_result.confidence >= effective_config.min_confidence
                ):
                    # Create optimized version
                    optimized_prompt = Prompt(
                        id=prompt.id,
                        workflow_id=prompt.workflow_id,
                        node_id=prompt.node_id,
                        node_type=prompt.node_type,
                        text=best_result.optimized_prompt,
                        role=prompt.role,
                        variables=self._extract_variables(
                            best_result.optimized_prompt
                        ),
                        context=prompt.context,
                        extracted_at=prompt.extracted_at,
                    )

                    # Validate that all variables preserved
                    self._validate_optimized_prompt(prompt, best_result.optimized_prompt)

                    optimized_analysis = self._analyzer.analyze_prompt(
                        optimized_prompt
                    )

                    self._version_manager.create_version(
                        prompt=optimized_prompt,
                        analysis=optimized_analysis,
                        optimization_result=best_result,
                        parent_version="1.0.0",
                    )

                    # Generate patch
                    patch = self._create_prompt_patch(prompt, best_result)
                    patches.append(patch)

                    self._logger.info(
                        f"Optimized prompt '{prompt.id}': "
                        f"strategy={best_result.strategy.value}, "
                        f"improvement={best_result.improvement_score:.1f}, "
                        f"confidence={best_result.confidence:.2f}"
                    )
                else:
                    best_conf = best_result.confidence if best_result else 0.0
                    self._logger.warning(
                        f"No acceptable optimization found for prompt '{prompt.id}' "
                        f"(best confidence: {best_conf:.2f})"
                    )

            self._logger.info(
                f"Optimization cycle complete: generated {len(patches)} patches"
            )

            # Record LLM usage statistics if available
            if not isinstance(self._llm_client, StubLLMClient):
                try:
                    stats = self._llm_client.get_usage_stats()
                    cache_hit_rate = (
                        stats.cache_hits / (stats.cache_hits + stats.cache_misses)
                        if (stats.cache_hits + stats.cache_misses) > 0
                        else 0.0
                    )
                    self._logger.info(
                        f"LLM usage: {stats.total_requests} requests, "
                        f"{stats.total_tokens} tokens, ${stats.total_cost:.4f} cost, "
                        f"cache hit rate: {cache_hit_rate:.2%}, "
                        f"avg latency: {stats.average_latency_ms:.0f}ms"
                    )
                except Exception as e:
                    self._logger.warning(f"Failed to retrieve LLM stats: {e}")

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

    def _optimize_with_iterations(
        self,
        prompt: Prompt,
        strategy: str,
        max_iterations: int,
        min_confidence: float,
    ) -> Optional[OptimizationResult]:
        """Iteratively optimize a prompt with a single strategy.

        Attempts optimization up to max_iterations times, stopping early if:
        - Confidence threshold is met (success)
        - No improvement detected (convergence)

        Args:
            prompt: Original prompt to optimize
            strategy: Optimization strategy to use
            max_iterations: Maximum number of iterations
            min_confidence: Minimum confidence threshold for early exit

        Returns:
            Best OptimizationResult across all iterations, or None if all failed
        """
        current_prompt = prompt
        best_result = None
        best_score = -float("inf")

        for iteration in range(max_iterations):
            self._logger.debug(
                f"Iteration {iteration + 1}/{max_iterations} "
                f"for prompt '{prompt.id}' with strategy '{strategy}'"
            )

            # Optimize current prompt
            result = self._engine.optimize(current_prompt, strategy)

            # Track best result
            if result.improvement_score > best_score:
                best_score = result.improvement_score
                best_result = result

            # Check if confidence threshold met (success)
            if result.confidence >= min_confidence:
                self._logger.info(
                    f"Confidence threshold met at iteration {iteration + 1}: "
                    f"{result.confidence:.2f} >= {min_confidence:.2f}"
                )
                return result

            # Check for convergence (no improvement)
            if iteration > 0 and result.improvement_score <= 0:
                self._logger.info(
                    f"No improvement detected at iteration {iteration + 1}, "
                    f"stopping early"
                )
                break

            # Prepare for next iteration using optimized prompt
            current_prompt = Prompt(
                id=prompt.id,
                workflow_id=prompt.workflow_id,
                node_id=prompt.node_id,
                node_type=prompt.node_type,
                text=result.optimized_prompt,
                role=prompt.role,
                variables=self._extract_variables(result.optimized_prompt),
                context=prompt.context,
                extracted_at=prompt.extracted_at,
            )

        # Return best result even if confidence not met
        if best_result:
            self._logger.warning(
                f"Max iterations reached without meeting confidence threshold. "
                f"Best confidence: {best_result.confidence:.2f}"
            )

        return best_result

    def _get_or_analyze(self, prompt: Prompt) -> PromptAnalysis:
        """Get cached analysis or analyze if not cached.

        Uses MD5 hash of prompt text as cache key to avoid re-analyzing
        identical prompts during optimization cycles.

        Args:
            prompt: Prompt to analyze

        Returns:
            PromptAnalysis (from cache or newly computed)
        """
        import hashlib

        # Use prompt text hash as cache key
        cache_key = hashlib.md5(prompt.text.encode('utf-8')).hexdigest()

        if cache_key not in self._analysis_cache:
            self._analysis_cache[cache_key] = self._analyzer.analyze_prompt(prompt)
            self._logger.debug(
                f"Analysis cache miss for prompt '{prompt.id}' (key={cache_key[:8]}...)"
            )
        else:
            self._logger.debug(
                f"Analysis cache hit for prompt '{prompt.id}' (key={cache_key[:8]}...)"
            )

        return self._analysis_cache[cache_key]

    def _extract_variables(self, text: str) -> List[str]:
        """Extract variables using centralized extractor.

        Args:
            text: Prompt text

        Returns:
            List of variable names
        """
        return VariableExtractor.extract(text)

    def _validate_optimized_prompt(self, original: Prompt, optimized_text: str) -> None:
        """Ensure optimization preserved all variables.

        Args:
            original: Original prompt with required variables
            optimized_text: Optimized prompt text

        Raises:
            OptimizerError: If optimization lost required variables
        """
        missing = VariableExtractor.validate_variables(
            optimized_text,
            original.variables
        )
        if missing:
            raise OptimizerError(
                message=f"Optimization lost required variables: {missing}",
                error_code="OPT-VAR-001",
                context={"prompt_id": original.id, "missing": missing}
            )

    def _is_better_result(
        self,
        candidate: OptimizationResult,
        current_best: Optional[OptimizationResult],
        min_confidence: float,
    ) -> bool:
        """Determine if candidate result is better than current best.

        Selection priority:
            1. Meets confidence threshold (min_confidence)
            2. Higher optimized_score (primary metric)
            3. Higher confidence (tie-breaker)

        Args:
            candidate: New optimization result to evaluate
            current_best: Current best result (or None)
            min_confidence: Minimum confidence threshold

        Returns:
            True if candidate is better than current_best
        """
        if current_best is None:
            return True

        # Priority 1: Prefer results that meet confidence threshold
        candidate_meets_threshold = candidate.confidence >= min_confidence
        current_meets_threshold = current_best.confidence >= min_confidence

        if candidate_meets_threshold and not current_meets_threshold:
            return True
        if current_meets_threshold and not candidate_meets_threshold:
            return False

        # Priority 2: Higher optimized score
        candidate_new_score = candidate.metadata.get("optimized_score", 0.0)
        current_new_score = current_best.metadata.get("optimized_score", 0.0)

        if candidate_new_score > current_new_score + 1.0:  # 1-point tolerance
            return True
        if current_new_score > candidate_new_score + 1.0:
            return False

        # Priority 3: Higher confidence (tie-breaker)
        return candidate.confidence > current_best.confidence

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
            OptimizerError: If catalog not initialized or extraction fails.
        """
        # Enhanced catalog validation
        if self._catalog is None:
            raise ValueError(
                "WorkflowCatalog not initialized. "
                "Optimizer requires catalog for workflow data."
            )

        try:
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

        except AttributeError as e:
            raise RuntimeError(
                f"Catalog integration error: {e}. "
                f"Check WorkflowCatalog initialization."
            ) from e

    def _should_optimize(
        self,
        analysis: PromptAnalysis,
        test_results: Optional[TestExecutionReport],
        config: OptimizationConfig = None,
    ) -> bool:
        """Determine if a prompt should be optimized using ScoringRules.

        Args:
            analysis: Prompt analysis.
            test_results: Test execution report (optional). Also accepts legacy dict format for backward compatibility.
            config: Optimization configuration (uses default if None).

        Returns:
            True if optimization is recommended.
        """
        # Handle legacy dict format (backward compatibility for direct calls)
        if test_results is not None and isinstance(test_results, dict):
            # Legacy baseline_metrics dict passed directly
            test_results_converted = self._scoring_rules._convert_legacy_baseline_metrics(test_results)
            return self._scoring_rules.should_optimize(analysis, test_results_converted, None, config)

        return self._scoring_rules.should_optimize(analysis, test_results, None, config)

    def _select_strategy(
        self,
        analysis: PromptAnalysis,
        test_results: Optional[TestExecutionReport],
    ) -> str:
        """Auto-select optimization strategy using ScoringRules.

        Args:
            analysis: Prompt analysis.
            test_results: Test execution report (optional). Also accepts legacy dict format for backward compatibility.

        Returns:
            Strategy name.
        """
        # Handle legacy dict format (backward compatibility for direct calls)
        if test_results is not None and isinstance(test_results, dict):
            # Legacy baseline_metrics dict passed directly
            test_results_converted = self._scoring_rules._convert_legacy_baseline_metrics(test_results)
            return self._scoring_rules.select_strategy(analysis, test_results_converted, None)

        return self._scoring_rules.select_strategy(analysis, test_results, None)

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

    def get_llm_stats(self) -> Optional[Dict[str, Any]]:
        """Get LLM usage statistics.

        Returns comprehensive statistics about LLM API usage including:
        - Total requests made
        - Total tokens used
        - Total cost incurred
        - Cache hit rate
        - Average latency

        Returns:
            Dictionary with statistics, or None if LLM client not available.

        Example:
            >>> stats = service.get_llm_stats()
            >>> if stats:
            ...     print(f"Total cost: ${stats['total_cost']:.4f}")
            ...     print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        """
        if isinstance(self._llm_client, StubLLMClient):
            return None

        try:
            stats = self._llm_client.get_usage_stats()
            return {
                "total_requests": stats.total_requests,
                "total_tokens": stats.total_tokens,
                "total_cost": stats.total_cost,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "cache_hit_rate": (
                    stats.cache_hits / (stats.cache_hits + stats.cache_misses)
                    if (stats.cache_hits + stats.cache_misses) > 0
                    else 0.0
                ),
                "average_latency_ms": stats.average_latency_ms,
            }
        except Exception as e:
            self._logger.error(f"Failed to get LLM stats: {e}")
            return None

    def reset_llm_stats(self) -> None:
        """Reset LLM usage statistics.

        Clears all accumulated usage statistics including request counts,
        token counts, costs, and cache metrics. Useful for starting fresh
        measurements for a new optimization cycle.

        Example:
            >>> service.reset_llm_stats()
            >>> # Run optimization cycle
            >>> patches = service.run_optimization_cycle("wf_001")
            >>> stats = service.get_llm_stats()
            >>> # Stats now only reflect the recent cycle
        """
        if not isinstance(self._llm_client, StubLLMClient):
            try:
                self._llm_client.reset_stats()
                self._logger.info("LLM statistics reset")
            except Exception as e:
                self._logger.warning(f"Failed to reset LLM stats: {e}")
