"""
YAML Configuration Module - RunManifest Models

Date: 2025-11-13
Author: Rebirthli
Description: Pydantic models for RunManifest (execution manifest)
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field

from .common import RateLimit, ModelEvaluator
from .test_plan import ExecutionPolicy, ConversationFlow


class TestCase(BaseModel):
    """Single test case"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    workflow_id: str = Field(..., description="Workflow identifier")
    dataset: str = Field(..., description="Dataset name")
    scenario: str = Field(..., description="Scenario type")
    parameters: Dict[str, Any] = Field(..., description="Test input parameters")
    conversation_flow: Optional[ConversationFlow] = Field(None, description="Chatflow conversation")
    prompt_variant: Optional[str] = Field(None, description="Prompt variant ID")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


@dataclass
class RunManifest:
    """
    Execution manifest for downstream modules

    This uses dataclass for simplicity as it's primarily a data transfer object.
    Contains all information needed to execute a test run.
    """
    workflow_id: str
    workflow_version: str
    prompt_variant: Optional[str]
    dsl_payload: str  # Modified DSL YAML text
    cases: List[TestCase]
    execution_policy: ExecutionPolicy
    rate_limits: RateLimit
    evaluator: ModelEvaluator
    metadata: Dict[str, Any]
