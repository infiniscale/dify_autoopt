"""
Workflow Module - Minimal discovery and runner

Provides minimal utilities to discover workflows from a WorkflowCatalog and
to run a stubbed workflow execution for integration with the CLI.
"""

from typing import List, Optional

from src.config.models import WorkflowCatalog, WorkflowEntry

from .discovery import discover_workflows
from .runner import run_workflow

__all__ = [
    "discover_workflows",
    "run_workflow",
    "WorkflowCatalog",
    "WorkflowEntry",
]

