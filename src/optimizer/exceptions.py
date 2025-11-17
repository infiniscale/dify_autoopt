"""
Optimizer Module - Custom Exceptions

Date: 2025-11-17
Author: backend-developer
Description: Defines custom exception hierarchy for the optimizer module.
"""

from typing import Any, Dict, Optional


class OptimizerError(Exception):
    """Base exception for optimizer module.

    All optimizer exceptions inherit from this class.

    Attributes:
        message: Human-readable error message.
        error_code: Optional error code for categorization.
        context: Optional dictionary with additional context information.

    Example:
        >>> try:
        ...     raise OptimizerError("Something went wrong", error_code="OPT-001")
        ... except OptimizerError as e:
        ...     print(f"Error: {e.message}, Code: {e.error_code}")
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize OptimizerError.

        Args:
            message: Human-readable error message.
            error_code: Optional error code for categorization.
            context: Optional dictionary with additional context information.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


# ============================================================================
# Extraction Errors
# ============================================================================


class ExtractionError(OptimizerError):
    """Base exception for prompt extraction failures.

    Raised when extracting prompts from workflow DSL fails.

    Typical causes:
        - Workflow not found in catalog
        - DSL file not accessible
        - Malformed DSL structure
    """

    pass


class WorkflowNotFoundError(ExtractionError):
    """Workflow ID not found in catalog.

    Raised when:
        - workflow_id doesn't exist in WorkflowCatalog
        - catalog.get_workflow() returns None

    Example:
        >>> raise WorkflowNotFoundError("wf_001")
    """

    def __init__(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize WorkflowNotFoundError.

        Args:
            workflow_id: The workflow ID that was not found.
            context: Optional additional context.
        """
        super().__init__(
            message=f"Workflow '{workflow_id}' not found in catalog",
            error_code="OPT-EXT-001",
            context={"workflow_id": workflow_id, **(context or {})},
        )
        self.workflow_id = workflow_id


class NodeNotFoundError(ExtractionError):
    """Node ID not found in workflow.

    Raised when:
        - node_id doesn't exist in workflow DSL
        - node path is invalid

    Example:
        >>> raise NodeNotFoundError("llm_node_1", "wf_001")
    """

    def __init__(
        self,
        node_id: str,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize NodeNotFoundError.

        Args:
            node_id: The node ID that was not found.
            workflow_id: The workflow ID where the node was expected.
            context: Optional additional context.
        """
        super().__init__(
            message=f"Node '{node_id}' not found in workflow '{workflow_id}'",
            error_code="OPT-EXT-002",
            context={
                "node_id": node_id,
                "workflow_id": workflow_id,
                **(context or {}),
            },
        )
        self.node_id = node_id
        self.workflow_id = workflow_id


class DSLParseError(ExtractionError):
    """DSL YAML parsing failed.

    Raised when:
        - YAML syntax is invalid
        - DSL file is malformed
        - Required fields are missing

    Example:
        >>> raise DSLParseError("/path/to/dsl.yml", "Invalid YAML syntax at line 10")
    """

    def __init__(
        self,
        dsl_path: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize DSLParseError.

        Args:
            dsl_path: Path to the DSL file that failed to parse.
            reason: Human-readable reason for the parse failure.
            context: Optional additional context.
        """
        super().__init__(
            message=f"Failed to parse DSL file '{dsl_path}': {reason}",
            error_code="OPT-EXT-003",
            context={"dsl_path": dsl_path, "reason": reason, **(context or {})},
        )
        self.dsl_path = dsl_path
        self.reason = reason


# ============================================================================
# Analysis Errors
# ============================================================================


class AnalysisError(OptimizerError):
    """Base exception for prompt analysis failures.

    Raised when analyzing prompt quality fails.
    """

    pass


class ScoringError(AnalysisError):
    """Scoring calculation failed.

    Raised when:
        - Readability calculation fails
        - Token estimation fails
        - Metric calculation throws exception

    Example:
        >>> raise ScoringError("clarity", "Division by zero in readability calculation")
    """

    def __init__(
        self,
        metric_name: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ScoringError.

        Args:
            metric_name: Name of the metric that failed to calculate.
            reason: Human-readable reason for the failure.
            context: Optional additional context.
        """
        super().__init__(
            message=f"Failed to calculate {metric_name} score: {reason}",
            error_code="OPT-ANA-001",
            context={"metric_name": metric_name, "reason": reason, **(context or {})},
        )
        self.metric_name = metric_name
        self.reason = reason


# ============================================================================
# Optimization Errors
# ============================================================================


class OptimizationError(OptimizerError):
    """Base exception for optimization process failures.

    Raised when the optimization process fails.
    """

    pass


class InvalidStrategyError(OptimizationError):
    """Invalid strategy name provided.

    Raised when:
        - strategy not in ['clarity_focus', 'efficiency_focus', 'structure_focus']
        - Unknown strategy requested

    Example:
        >>> raise InvalidStrategyError("unknown_strategy")
    """

    def __init__(
        self,
        strategy: str,
        valid_strategies: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize InvalidStrategyError.

        Args:
            strategy: The invalid strategy name that was provided.
            valid_strategies: List of valid strategy names.
            context: Optional additional context.
        """
        if valid_strategies is None:
            valid_strategies = ["clarity_focus", "efficiency_focus", "structure_focus"]

        super().__init__(
            message=(
                f"Invalid strategy: '{strategy}'. " f"Must be one of {valid_strategies}"
            ),
            error_code="OPT-OPT-001",
            context={
                "strategy": strategy,
                "valid_strategies": valid_strategies,
                **(context or {}),
            },
        )
        self.strategy = strategy
        self.valid_strategies = valid_strategies


class OptimizationFailedError(OptimizationError):
    """Optimization execution failed.

    Raised when:
        - Transformation logic throws exception
        - Re-analysis fails
        - Confidence calculation fails

    Example:
        >>> raise OptimizationFailedError("prompt_001", "clarity_focus", "Transform failed")
    """

    def __init__(
        self,
        prompt_id: str,
        strategy: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize OptimizationFailedError.

        Args:
            prompt_id: ID of the prompt that failed to optimize.
            strategy: Strategy being applied when failure occurred.
            reason: Human-readable reason for the failure.
            context: Optional additional context.
        """
        super().__init__(
            message=(
                f"Optimization failed for prompt '{prompt_id}' "
                f"using strategy '{strategy}': {reason}"
            ),
            error_code="OPT-OPT-002",
            context={
                "prompt_id": prompt_id,
                "strategy": strategy,
                "reason": reason,
                **(context or {}),
            },
        )
        self.prompt_id = prompt_id
        self.strategy = strategy
        self.reason = reason


# ============================================================================
# Version Management Errors
# ============================================================================


class VersionError(OptimizerError):
    """Base exception for version management failures.

    Raised when version operations fail.
    """

    pass


class VersionConflictError(VersionError):
    """Version conflict occurred.

    Raised when:
        - Parent version doesn't exist
        - Version already exists
        - Invalid version number format

    Example:
        >>> raise VersionConflictError("prompt_001", "1.1.0", "Version already exists")
    """

    def __init__(
        self,
        prompt_id: str,
        version: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize VersionConflictError.

        Args:
            prompt_id: ID of the prompt with version conflict.
            version: Version number involved in the conflict.
            reason: Human-readable reason for the conflict.
            context: Optional additional context.
        """
        super().__init__(
            message=(
                f"Version conflict for prompt '{prompt_id}' "
                f"at version '{version}': {reason}"
            ),
            error_code="OPT-VER-001",
            context={
                "prompt_id": prompt_id,
                "version": version,
                "reason": reason,
                **(context or {}),
            },
        )
        self.prompt_id = prompt_id
        self.version = version
        self.reason = reason


class VersionNotFoundError(VersionError):
    """Version not found in storage.

    Raised when:
        - Requested version doesn't exist
        - prompt_id not in storage

    Example:
        >>> raise VersionNotFoundError("prompt_001", "1.5.0")
    """

    def __init__(
        self,
        prompt_id: str,
        version: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize VersionNotFoundError.

        Args:
            prompt_id: ID of the prompt.
            version: Version number that was not found.
            context: Optional additional context.
        """
        super().__init__(
            message=f"Version '{version}' not found for prompt '{prompt_id}'",
            error_code="OPT-VER-002",
            context={
                "prompt_id": prompt_id,
                "version": version,
                **(context or {}),
            },
        )
        self.prompt_id = prompt_id
        self.version = version


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(OptimizerError):
    """Data validation failed.

    Raised when:
        - Model validation fails
        - Input data is invalid
        - Configuration is malformed

    Example:
        >>> raise ValidationError("OptimizationConfig", "scoring_weights must sum to 1.0")
    """

    def __init__(
        self,
        model_name: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            model_name: Name of the model or data structure that failed validation.
            reason: Human-readable reason for the validation failure.
            context: Optional additional context.
        """
        super().__init__(
            message=f"Validation failed for {model_name}: {reason}",
            error_code="OPT-VAL-001",
            context={"model_name": model_name, "reason": reason, **(context or {})},
        )
        self.model_name = model_name
        self.reason = reason


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigError(OptimizerError):
    """Configuration error.

    Raised when:
        - Configuration file is missing
        - Configuration values are invalid
        - Required configuration is not provided

    Example:
        >>> raise ConfigError("Missing required configuration: llm_client.api_key")
    """

    def __init__(
        self,
        reason: str,
        config_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ConfigError.

        Args:
            reason: Human-readable reason for the configuration error.
            config_key: Optional specific configuration key that caused the error.
            context: Optional additional context.
        """
        super().__init__(
            message=f"Configuration error: {reason}",
            error_code="OPT-CFG-001",
            context={"reason": reason, "config_key": config_key, **(context or {})},
        )
        self.reason = reason
        self.config_key = config_key
