"""
YAML Configuration Module - Workflow Catalog Models

Date: 2025-11-13
Author: Rebirthli
Description: Pydantic models for workflow_repository.yaml
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


class NodeMeta(BaseModel):
    """Metadata for a workflow node"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    node_id: str = Field(..., description="Node unique identifier")
    label: Optional[str] = Field(None, description="Human-readable label")
    type: str = Field(..., description="Node type (llm, tool, code, if_else, etc.)")
    path: str = Field(..., description="JSON pointer path (e.g., '/graph/nodes/3')")
    prompt_fields: List[str] = Field(default_factory=list, description="Editable prompt field paths")


class WorkflowEntry(BaseModel):
    """Single workflow entry in the catalog"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    id: str = Field(..., description="Stable workflow identifier")
    label: str = Field(..., description="Human-readable workflow name")
    type: str = Field(..., description="Workflow type: 'workflow' or 'chatflow'")
    version: Optional[str] = Field(None, description="Workflow version")
    dsl_path: Path = Field(..., description="Path to DSL YAML file")
    checksum: Optional[str] = Field(None, description="DSL file checksum (SHA256)")
    nodes: List[NodeMeta] = Field(default_factory=list, description="Node structure index")
    resources: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Resource dependencies (knowledge_bases, tools)"
    )
    tags: List[str] = Field(default_factory=list, description="Workflow tags")

    @property
    def dsl_path_resolved(self) -> Path:
        """Get resolved absolute path"""
        return self.dsl_path.resolve() if not self.dsl_path.is_absolute() else self.dsl_path


class WorkflowCatalog(BaseModel):
    """Root workflow repository model"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    meta: Dict[str, Any] = Field(..., description="Metadata (source, last_synced, version)")
    workflows: List[WorkflowEntry] = Field(..., description="List of workflow entries")

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowEntry]:
        """Get workflow by ID"""
        for workflow in self.workflows:
            if workflow.id == workflow_id:
                return workflow
        return None

    def get_workflows_by_tag(self, tag: str) -> List[WorkflowEntry]:
        """Get workflows with a specific tag"""
        return [wf for wf in self.workflows if tag in wf.tags]
