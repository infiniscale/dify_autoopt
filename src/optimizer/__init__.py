"""
Optimizer package

Lightweight optimizer utilities live here. Modules should remain import-safe
and avoid side effects at import time.
"""

from .yaml_loader import WorkflowYamlLoader, load_workflow_yaml
from .prompt_optimizer import (
    PromptOptimizer,
    PromptPatch,
    OptimizationReport,
    DetectedIssue,
    ReferenceSpec,
    ExecutionSample,
)
from .loop import run_optimize_loop

__all__ = [
    "WorkflowYamlLoader",
    "load_workflow_yaml",
    "PromptOptimizer",
    "PromptPatch",
    "OptimizationReport",
    "DetectedIssue",
    "ReferenceSpec",
    "ExecutionSample",
    "run_optimize_loop",
]
