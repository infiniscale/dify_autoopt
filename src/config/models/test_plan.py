"""
YAML Configuration Module - Test Plan Models

Date: 2025-11-13
Author: Rebirthli
Description: Pydantic models for test_plan.yaml
"""

from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import RateLimit


class PromptSelector(BaseModel):
    """Selector for targeting nodes in a workflow"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    by_id: Optional[str] = Field(None, description="Select by node ID")
    by_label: Optional[str] = Field(None, description="Select by node label (fuzzy match)")
    by_type: Optional[str] = Field(None, description="Select by node type")
    by_path: Optional[str] = Field(None, description="Select by JSON pointer path")
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional constraints (workflow_type, if_missing)"
    )


class PromptTemplate(BaseModel):
    """Template configuration for prompt patching"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    file: Optional[str] = Field(None, description="Path to template file")
    inline: Optional[str] = Field(None, description="Inline template content")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")


class PromptStrategy(BaseModel):
    """Strategy for applying prompt patch"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    mode: str = Field(..., description="Patch mode: replace|prepend|append|template")
    content: Optional[str] = Field(None, description="Content for replace/prepend/append")
    template: Optional[PromptTemplate] = Field(None, description="Template configuration")
    fallback_value: Optional[str] = Field(None, description="Fallback if strategy fails")

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, value: str) -> str:
        allowed = ['replace', 'prepend', 'append', 'template']
        if value not in allowed:
            raise ValueError(f"Invalid mode: {value}. Must be one of {allowed}")
        return value


class PromptPatch(BaseModel):
    """Single prompt patch configuration"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    selector: PromptSelector = Field(..., description="Node selector")
    strategy: PromptStrategy = Field(..., description="Patch strategy")


class PromptVariant(BaseModel):
    """Prompt optimization variant"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    variant_id: str = Field(..., description="Variant unique identifier")
    description: Optional[str] = Field(None, description="Variant description")
    weight: float = Field(1.0, ge=0.0, description="Variant sampling weight")
    nodes: List[PromptPatch] = Field(..., description="List of prompt patches")
    fallback_variant: Optional[str] = Field(None, description="Fallback variant ID")


class InputParameter(BaseModel):
    """Input parameter configuration for test data"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    type: str = Field(..., description="Parameter type: string|int|float|bool|file|json")
    values: Optional[List[Any]] = Field(None, description="List of possible values")
    range: Optional[Dict[str, float]] = Field(None, description="Numeric range (min, max, step)")
    file_pool: Optional[List[str]] = Field(None, description="Pool of file paths")
    json_template: Optional[str] = Field(None, description="JSON template string")
    default: Optional[Any] = Field(None, description="Default value")


class ConversationStep(BaseModel):
    """Single conversation step for chatflow testing"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    role: str = Field(..., description="Role: user|assistant|system|tool")
    message: str = Field(..., description="Message content")
    wait_for_response: bool = Field(True, description="Wait for response before next step")


class ConversationFlow(BaseModel):
    """Multi-turn conversation flow for chatflow testing"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    title: Optional[str] = Field(None, description="Flow title")
    steps: List[ConversationStep] = Field(..., description="Conversation steps")
    expected_outcome: Optional[str] = Field(None, description="Expected outcome description")


class Dataset(BaseModel):
    """Test dataset configuration"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    name: str = Field(..., description="Dataset name")
    scenario: str = Field(..., description="Scenario type: normal|boundary|error|custom")
    description: Optional[str] = Field(None, description="Dataset description")
    parameters: Dict[str, InputParameter] = Field(
        default_factory=dict,
        description="Input parameters"
    )
    conversation_flows: List[ConversationFlow] = Field(
        default_factory=list,
        description="Conversation flows (for chatflow)"
    )
    pairwise_dimensions: List[str] = Field(
        default_factory=list,
        description="Parameters to use for pairwise testing"
    )
    weight: float = Field(1.0, ge=0.0, description="Dataset sampling weight")


class WorkflowPlanEntry(BaseModel):
    """Workflow entry in test plan"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    catalog_id: str = Field(..., description="Reference to WorkflowCatalog ID")
    enabled: bool = Field(True, description="Whether workflow is enabled for testing")
    dataset_refs: List[str] = Field(..., description="Referenced dataset names")
    weight: float = Field(1.0, ge=0.0, description="Workflow priority weight")
    prompt_optimization: Optional[List[PromptVariant]] = Field(
        None,
        description="Prompt optimization variants"
    )


class RetryPolicy(BaseModel):
    """Retry policy configuration"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    max_attempts: int = Field(3, ge=1, description="Maximum retry attempts")
    backoff_seconds: float = Field(2.0, ge=0, description="Backoff time in seconds")
    backoff_multiplier: float = Field(1.5, ge=1.0, description="Backoff multiplier")


class ExecutionPolicy(BaseModel):
    """Execution policy configuration"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    concurrency: int = Field(..., ge=1, description="Concurrent execution count")
    batch_size: int = Field(5, ge=1, description="Batch size for execution")
    rate_control: RateLimit = Field(..., description="Rate control limits")
    backoff_seconds: float = Field(2.0, ge=0, description="Backoff between batches")
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy, description="Retry policy")
    stop_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stop conditions (max_failures, timeout)"
    )


class TestDataConfig(BaseModel):
    """Test data configuration"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    datasets: List[Dataset] = Field(..., description="Test datasets")
    strategy: Dict[str, Any] = Field(
        default_factory=dict,
        description="Test generation strategy (pairwise_mode, sampling_method)"
    )


class TestPlan(BaseModel):
    """Root test plan model"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    meta: Dict[str, Any] = Field(..., description="Metadata (plan_id, owner, description)")
    workflows: List[WorkflowPlanEntry] = Field(..., description="Workflow entries")
    test_data: TestDataConfig = Field(..., description="Test data configuration")
    execution: ExecutionPolicy = Field(..., description="Execution policy")
    validation: Dict[str, Any] = Field(default_factory=dict, description="Validation rules")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output configuration")

    @property
    def datasets(self) -> List[Dataset]:
        """Extract datasets from test_data"""
        return self.test_data.datasets

    def get_dataset(self, name: str) -> Optional[Dataset]:
        """Get dataset by name"""
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None
