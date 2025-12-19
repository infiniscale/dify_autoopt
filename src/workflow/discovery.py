"""
Workflow Discovery - Minimal Implementation

Select workflows from a WorkflowCatalog by id or tag.
"""

from typing import List, Optional

from src.config.models import WorkflowCatalog, WorkflowEntry


def discover_workflows(
        catalog: WorkflowCatalog,
        workflow_id: Optional[str] = None,
        tag: Optional[str] = None,
) -> List[WorkflowEntry]:
    """Discover workflows by id or tag; default returns all.

    Args:
        catalog: WorkflowCatalog instance
        workflow_id: Optional specific id to select
        tag: Optional tag to filter

    Returns:
        List of WorkflowEntry
    """
    if workflow_id:
        wf = catalog.get_workflow(workflow_id)
        return [wf] if wf else []

    if tag:
        return catalog.get_workflows_by_tag(tag)

    return list(catalog.workflows)
