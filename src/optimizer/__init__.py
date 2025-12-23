"""
Optimizer package

Lightweight optimizer utilities live here. Modules should remain import-safe
and avoid side effects at import time.
"""

from .loop import run_optimize_loop
from .prompt_optimizer import (
    PromptOptimizer,
    PromptPatch,
    PromptAction,
    PromptState,
    apply_actions,
    optimize_prompt,
    OptimizationReport,
    DetectedIssue,
    ReferenceSpec,
    ExecutionSample,
)
from .yaml_loader import WorkflowYamlLoader, load_workflow_yaml

__all__ = [
    "WorkflowYamlLoader",
    "load_workflow_yaml",
    "PromptOptimizer",
    "PromptPatch",
    "PromptAction",
    "PromptState",
    "apply_actions",
    "optimize_prompt",
    "OptimizationReport",
    "DetectedIssue",
    "ReferenceSpec",
    "ExecutionSample",
    "run_optimize_loop",
]
