"""
Workflow Module - Minimal discovery and runner

Provides minimal utilities to discover workflows from a WorkflowCatalog and
to run a stubbed workflow execution for integration with the CLI.
"""

from typing import List, Optional

from src.config.models import WorkflowCatalog, WorkflowEntry

from .discovery import discover_workflows
from .runner import run_workflow
from .apps import list_all_apps
from .export import export_app_dsl
from .imports import import_app_yaml
from .publish import publish_workflow
from .execute import execute_workflow_v1, execute_workflow_from_config

__all__ = [
    "discover_workflows",
    "run_workflow",
    "list_all_apps",
    "export_app_dsl",
    "import_app_yaml",
    "publish_workflow",
    "execute_workflow_v1",
    "execute_workflow_from_config",
    "WorkflowCatalog",
    "WorkflowEntry",
]
