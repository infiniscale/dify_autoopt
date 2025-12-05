# Optimizer Module - Software Requirements Specification (SRS)

**Project:** dify_autoopt
**Module:** src/optimizer
**Version:** 1.0.0 MVP
**Date:** 2025-11-17
**Author:** Requirements Analyst
**Status:** Draft for Review

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-17 | Requirements Analyst | Initial SRS creation |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Models](#2-data-models)
3. [Functional Requirements](#3-functional-requirements)
4. [Interface Requirements](#4-interface-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Constraints and Assumptions](#6-constraints-and-assumptions)
7. [Acceptance Criteria](#7-acceptance-criteria)
8. [Appendix](#8-appendix)

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document defines the detailed functional, non-functional, and interface requirements for the **Optimizer Module** of the dify_autoopt project. The optimizer module is responsible for:

- Extracting prompts from Dify workflow DSL files
- Analyzing prompt quality using rule-based heuristics
- Generating optimized prompt variants
- Managing prompt version history
- Providing orchestration services for the complete optimization cycle

This document is intended for:
- **Backend Developers** implementing the module
- **QA Engineers** designing test cases
- **Project Managers** tracking feature completeness
- **Integration Engineers** connecting optimizer with other modules

### 1.2 Scope

**In Scope (MVP):**
- Prompt extraction from LLM nodes in workflow DSL
- Rule-based prompt quality analysis (clarity, efficiency)
- Three core optimization strategies (clarity focus, efficiency focus, structure optimization)
- In-memory version management with linear history
- Integration with existing modules (config, executor, collector)
- CLI command interface for optimization workflows
- Comprehensive test coverage (target: 90%)

**Out of Scope (Post-MVP):**
- LLM-based prompt analysis (e.g., using GPT-4 for quality evaluation)
- A/B testing framework integration
- Persistent database storage for version history
- Git-like branching and merging for prompts
- Real-time optimization during test execution
- Multi-language prompt support (MVP is English-only)
- Automated rollback mechanisms
- Ensemble optimization (combining multiple strategies)

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| **DSL** | Domain Specific Language - YAML-based workflow definition format used by Dify |
| **LLM** | Large Language Model |
| **Prompt** | Text instruction provided to an LLM node in a workflow |
| **Prompt Patch** | Modification applied to a prompt via PromptPatchEngine |
| **Optimization Cycle** | Complete flow: Extract → Analyze → Optimize → Version → Generate Patch |
| **Baseline Metrics** | Performance metrics collected before optimization |
| **Variant** | Alternative version of a prompt (e.g., original, optimized_v1) |
| **Strategy** | Optimization approach (e.g., clarity_focus, efficiency_focus) |
| **MVP** | Minimum Viable Product |
| **SRS** | Software Requirements Specification |
| **TDD** | Test-Driven Development |
| **RTM** | Requirements Traceability Matrix |

### 1.4 References

| Document | Location |
|----------|----------|
| Optimizer Execution Blueprint | `docs/optimizer_execution_blueprint.md` |
| Optimizer Summary | `docs/optimizer_summary.md` |
| Optimizer Design README | `src/optimizer/README.md` |
| Config Module Models | `src/config/models/` |
| Executor Module Models | `src/executor/models.py` |
| Collector Module Models | `src/collector/models.py` |
| Project README | `README.md` |

---

## 2. Data Models

All data models MUST be defined using **Pydantic V2** with the following conventions:
- Use `BaseModel` from `pydantic`
- Set `model_config = ConfigDict(extra='forbid', validate_assignment=True)`
- Use `Field(...)` for required fields with descriptions
- Use `Field(default=...)` or `Field(default_factory=...)` for optional fields
- Implement custom validators using `@field_validator` decorator
- Follow existing naming patterns from config/executor/collector modules

### 2.1 Prompt Model

**Purpose:** Represents an extracted prompt with complete metadata for analysis and optimization.

**File Location:** `src/optimizer/models.py`

```python
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

class Prompt(BaseModel):
    """
    Extracted prompt with metadata from workflow DSL.

    Attributes:
        id: Unique identifier format: "{workflow_id}_{node_id}"
        workflow_id: Parent workflow identifier
        node_id: Node identifier in workflow graph
        node_type: Type of node (e.g., "llm", "code")
        text: Actual prompt content (raw text)
        role: Message role ("system", "user", "assistant")
        variables: List of detected variable placeholders (e.g., ["{{var1}}", "{{var2}}"])
        context: Node context metadata (label, position, dependencies)
        extracted_at: Timestamp of extraction
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    id: str = Field(..., description="Unique prompt identifier: workflow_id_node_id")
    workflow_id: str = Field(..., min_length=1, description="Workflow identifier")
    node_id: str = Field(..., min_length=1, description="Node identifier")
    node_type: str = Field(..., description="Node type (llm, code, tool)")
    text: str = Field(..., description="Prompt content")
    role: str = Field(..., description="Message role: system|user|assistant")
    variables: List[str] = Field(default_factory=list, description="Variable placeholders")
    context: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")

    @field_validator('role')
    @classmethod
    def validate_role(cls, value: str) -> str:
        allowed = ['system', 'user', 'assistant']
        if value not in allowed:
            raise ValueError(f"Invalid role: {value}. Must be one of {allowed}")
        return value

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, value: str) -> str:
        if '_' not in value:
            raise ValueError(f"Prompt ID must follow format 'workflow_id_node_id', got: {value}")
        return value
```

**Validation Rules:**
- `id` MUST follow format `{workflow_id}_{node_id}` (validated by presence of underscore)
- `workflow_id` and `node_id` MUST NOT be empty strings
- `role` MUST be one of: `system`, `user`, `assistant`
- `text` MAY be empty (for placeholder prompts)
- `variables` MUST be a list of strings matching pattern `{{variable_name}}`
- `extracted_at` defaults to current timestamp if not provided

**Field Constraints:**
- `id`: max_length=255, unique within extraction session
- `workflow_id`: max_length=100
- `node_id`: max_length=100
- `text`: max_length=50000 (to handle very long prompts)
- `variables`: max_items=100 (reasonable limit for variable count)

---

### 2.2 PromptAnalysis Model

**Purpose:** Contains comprehensive quality analysis results for a prompt.

**File Location:** `src/optimizer/models.py`

```python
class PromptIssue(BaseModel):
    """
    Represents a single detected issue in a prompt.

    Attributes:
        severity: Issue severity level
        category: Issue category (clarity, efficiency, safety, structure)
        message: Human-readable issue description
        location: Optional location hint (line number, character range)
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    severity: str = Field(..., description="Severity: critical|warning|info")
    category: str = Field(..., description="Category: clarity|efficiency|safety|structure")
    message: str = Field(..., description="Issue description")
    location: Optional[str] = Field(None, description="Location hint in prompt text")

    @field_validator('severity')
    @classmethod
    def validate_severity(cls, value: str) -> str:
        allowed = ['critical', 'warning', 'info']
        if value not in allowed:
            raise ValueError(f"Invalid severity: {value}. Must be one of {allowed}")
        return value


class PromptSuggestion(BaseModel):
    """
    Represents an improvement suggestion for a prompt.

    Attributes:
        category: Suggestion category
        priority: Suggestion priority (1=highest, 5=lowest)
        suggestion: Human-readable suggestion text
        example: Optional example of improved text
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    category: str = Field(..., description="Category: clarity|efficiency|safety|structure")
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest, 5=lowest)")
    suggestion: str = Field(..., description="Improvement suggestion")
    example: Optional[str] = Field(None, description="Example of improved prompt")


class PromptAnalysis(BaseModel):
    """
    Comprehensive analysis results for a prompt.

    Scoring Metrics:
    - clarity_score: Readability and instruction clarity (0-100)
    - efficiency_score: Token usage efficiency (0-100)
    - overall_score: Weighted average of all scores (0-100)

    Attributes:
        prompt_id: Reference to analyzed prompt
        clarity_score: Clarity/readability score (0-100)
        efficiency_score: Token efficiency score (0-100)
        overall_score: Weighted composite score (0-100)
        issues: List of detected issues
        suggestions: List of improvement suggestions
        metrics: Raw metrics used for scoring
        analyzed_at: Analysis timestamp
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    prompt_id: str = Field(..., description="Reference to Prompt.id")
    clarity_score: float = Field(..., ge=0.0, le=100.0, description="Clarity score (0-100)")
    efficiency_score: float = Field(..., ge=0.0, le=100.0, description="Efficiency score (0-100)")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score (0-100)")
    issues: List[PromptIssue] = Field(default_factory=list, description="Detected issues")
    suggestions: List[PromptSuggestion] = Field(default_factory=list, description="Improvement suggestions")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Raw analysis metrics")
    analyzed_at: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")

    @field_validator('overall_score')
    @classmethod
    def validate_overall_score(cls, value: float, info) -> float:
        # Overall score should be weighted average of component scores
        # This is informational validation, not enforced
        return value
```

**Validation Rules:**
- All scores MUST be in range [0.0, 100.0]
- `overall_score` SHOULD be calculated as weighted average (default: 0.6 * clarity + 0.4 * efficiency)
- `issues` and `suggestions` MAY be empty lists
- `metrics` MUST contain at least: `token_count`, `avg_sentence_length`, `variable_count`

**Scoring Algorithm (Detailed in Section 3.2):**
```python
# Default weights (configurable via OptimizationConfig)
CLARITY_WEIGHT = 0.6
EFFICIENCY_WEIGHT = 0.4

overall_score = (clarity_score * CLARITY_WEIGHT) + (efficiency_score * EFFICIENCY_WEIGHT)
```

**Raw Metrics Dictionary:**
```python
{
    "token_count": int,           # Estimated token count
    "character_count": int,        # Total characters
    "sentence_count": int,         # Number of sentences
    "avg_sentence_length": float,  # Average words per sentence
    "variable_count": int,         # Number of variables
    "readability_index": float,    # Flesch Reading Ease or similar
    "information_density": float,  # Content-to-token ratio estimate
}
```

---

### 2.3 OptimizationResult Model

**Purpose:** Encapsulates the result of applying an optimization strategy to a prompt.

**File Location:** `src/optimizer/models.py`

```python
class OptimizationResult(BaseModel):
    """
    Result of optimization process.

    Represents the transformation of an original prompt into an optimized version,
    along with comparative analysis and confidence metrics.

    Attributes:
        prompt_id: Reference to original prompt
        original_prompt: Original prompt text
        optimized_prompt: Optimized prompt text
        strategy_used: Strategy identifier (clarity_focus, efficiency_focus, structure)
        improvement_score: Delta in overall_score (optimized - original)
        original_analysis: Analysis of original prompt
        optimized_analysis: Analysis of optimized prompt
        confidence: Confidence level in improvement (0.0-1.0)
        optimized_at: Optimization timestamp
        metadata: Additional optimization metadata
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    prompt_id: str = Field(..., description="Reference to Prompt.id")
    original_prompt: str = Field(..., description="Original prompt text")
    optimized_prompt: str = Field(..., description="Optimized prompt text")
    strategy_used: str = Field(..., description="Optimization strategy applied")
    improvement_score: float = Field(..., description="Delta in overall_score (can be negative)")
    original_analysis: PromptAnalysis = Field(..., description="Original prompt analysis")
    optimized_analysis: PromptAnalysis = Field(..., description="Optimized prompt analysis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in improvement (0-1)")
    optimized_at: datetime = Field(default_factory=datetime.now, description="Optimization timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('strategy_used')
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        allowed = ['clarity_focus', 'efficiency_focus', 'structure_optimization']
        if value not in allowed:
            raise ValueError(f"Invalid strategy: {value}. Must be one of {allowed}")
        return value

    @property
    def is_improvement(self) -> bool:
        """Check if optimization resulted in improvement."""
        return self.improvement_score > 0

    @property
    def improvement_percentage(self) -> float:
        """Calculate improvement as percentage of original score."""
        if self.original_analysis.overall_score == 0:
            return 0.0
        return (self.improvement_score / self.original_analysis.overall_score) * 100
```

**Validation Rules:**
- `improvement_score` CAN be negative (indicates optimization degraded quality)
- `confidence` MUST be in range [0.0, 1.0]
- `strategy_used` MUST be one of the defined strategies
- `optimized_prompt` MUST differ from `original_prompt` (non-trivial optimization)

**Confidence Calculation:**
```python
# Confidence based on:
# 1. Magnitude of improvement
# 2. Number of issues resolved
# 3. Consistency of metric improvements

base_confidence = min(abs(improvement_score) / 20.0, 1.0)  # Score improvement
issues_resolved = len(original_analysis.issues) - len(optimized_analysis.issues)
issue_factor = min(issues_resolved / 5.0, 0.5)  # Up to 0.5 bonus

confidence = min(base_confidence + issue_factor, 1.0)
```

---

### 2.4 PromptVersion Model

**Purpose:** Represents a single version in the prompt version history.

**File Location:** `src/optimizer/models.py`

```python
class PromptVersion(BaseModel):
    """
    Single version record in prompt history.

    Follows semantic versioning principles:
    - Major version: Breaking changes to prompt structure
    - Minor version: Optimization improvements
    - Patch version: Bug fixes or minor tweaks

    Attributes:
        prompt_id: Reference to prompt (base identifier without version)
        version: Semantic version string (e.g., "1.2.0")
        text: Prompt text for this version
        analysis: Quality analysis for this version
        created_at: Version creation timestamp
        author: Creator of this version (baseline, optimizer, manual)
        parent_version: Parent version number (None for v1.0.0)
        change_summary: Human-readable change description
        tags: Optional version tags (stable, experimental, rollback)
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    prompt_id: str = Field(..., description="Base prompt identifier")
    version: str = Field(..., description="Semantic version (e.g., 1.2.0)")
    text: str = Field(..., description="Prompt text for this version")
    analysis: PromptAnalysis = Field(..., description="Quality analysis")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    author: str = Field(..., description="Creator: baseline|optimizer|manual")
    parent_version: Optional[str] = Field(None, description="Parent version number")
    change_summary: Optional[str] = Field(None, description="Change description")
    tags: List[str] = Field(default_factory=list, description="Version tags")

    @field_validator('version')
    @classmethod
    def validate_version_format(cls, value: str) -> str:
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(pattern, value):
            raise ValueError(f"Version must follow semantic versioning (x.y.z), got: {value}")
        return value

    @field_validator('author')
    @classmethod
    def validate_author(cls, value: str) -> str:
        allowed = ['baseline', 'optimizer', 'manual']
        if value not in allowed:
            raise ValueError(f"Invalid author: {value}. Must be one of {allowed}")
        return value

    @property
    def version_tuple(self) -> tuple:
        """Parse version string into (major, minor, patch) tuple."""
        parts = self.version.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
```

**Validation Rules:**
- `version` MUST follow semantic versioning format `X.Y.Z` where X, Y, Z are non-negative integers
- `author` MUST be one of: `baseline`, `optimizer`, `manual`
- `parent_version` MUST reference an existing version OR be None (for first version)
- First version MUST be `1.0.0` with `parent_version=None`

**Version Numbering Strategy:**
- **Baseline version:** Always `1.0.0`
- **Optimizer improvements:** Increment minor version (1.0.0 → 1.1.0 → 1.2.0)
- **Manual edits:** Increment patch version (1.1.0 → 1.1.1)
- **Major refactors:** Increment major version (1.x.x → 2.0.0) - NOT used in MVP

---

### 2.5 OptimizationConfig Model

**Purpose:** Configuration parameters for the optimization engine.

**File Location:** `src/optimizer/models.py`

```python
class OptimizationConfig(BaseModel):
    """
    Configuration for optimization engine behavior.

    Controls strategy selection, scoring weights, and acceptance thresholds.

    Attributes:
        default_strategy: Default optimization strategy
        improvement_threshold: Minimum improvement to accept (percentage)
        max_optimization_iterations: Maximum optimization passes
        enable_version_tracking: Whether to track versions
        scoring_weights: Weights for score components
        confidence_threshold: Minimum confidence to accept optimization
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    default_strategy: str = Field(
        "clarity_focus",
        description="Default strategy: clarity_focus|efficiency_focus|structure_optimization"
    )
    improvement_threshold: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Minimum improvement percentage to accept"
    )
    max_optimization_iterations: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum optimization iterations"
    )
    enable_version_tracking: bool = Field(
        True,
        description="Enable version management"
    )
    scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {"clarity": 0.6, "efficiency": 0.4},
        description="Component score weights (must sum to 1.0)"
    )
    confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to accept optimization"
    )

    @field_validator('scoring_weights')
    @classmethod
    def validate_weights_sum(cls, value: Dict[str, float]) -> Dict[str, float]:
        total = sum(value.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating-point errors
            raise ValueError(f"Scoring weights must sum to 1.0, got: {total}")

        required_keys = {'clarity', 'efficiency'}
        if not required_keys.issubset(value.keys()):
            raise ValueError(f"Scoring weights must include keys: {required_keys}")

        return value

    @field_validator('default_strategy')
    @classmethod
    def validate_default_strategy(cls, value: str) -> str:
        allowed = ['clarity_focus', 'efficiency_focus', 'structure_optimization']
        if value not in allowed:
            raise ValueError(f"Invalid default_strategy: {value}. Must be one of {allowed}")
        return value
```

**Validation Rules:**
- `scoring_weights` values MUST sum to approximately 1.0 (±0.01 tolerance)
- `scoring_weights` MUST include keys: `clarity`, `efficiency`
- `improvement_threshold` in range [0.0, 100.0] (percentage)
- `confidence_threshold` in range [0.0, 1.0]

**Default Configuration:**
```yaml
default_strategy: clarity_focus
improvement_threshold: 5.0
max_optimization_iterations: 3
enable_version_tracking: true
scoring_weights:
  clarity: 0.6
  efficiency: 0.4
confidence_threshold: 0.5
```

---

### 2.6 Supporting Models

**File Location:** `src/optimizer/models.py`

```python
class VersionComparison(BaseModel):
    """
    Comparison result between two prompt versions.

    Used by VersionManager.compare_versions() method.
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    prompt_id: str = Field(..., description="Base prompt identifier")
    version_a: str = Field(..., description="First version number")
    version_b: str = Field(..., description="Second version number")
    text_diff: str = Field(..., description="Unified diff of prompt texts")
    score_delta: float = Field(..., description="overall_score difference (b - a)")
    clarity_delta: float = Field(..., description="clarity_score difference")
    efficiency_delta: float = Field(..., description="efficiency_score difference")
    issues_resolved: int = Field(..., description="Number of issues resolved")
    issues_introduced: int = Field(..., description="Number of new issues")
    recommendation: str = Field(..., description="Recommendation: prefer_a|prefer_b|equivalent")
    compared_at: datetime = Field(default_factory=datetime.now, description="Comparison timestamp")


class OptimizationReport(BaseModel):
    """
    Aggregated report for optimization results across multiple prompts.

    Generated by OptimizerService.get_optimization_report() method.
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    workflow_id: str = Field(..., description="Workflow identifier")
    total_prompts: int = Field(..., ge=0, description="Total prompts analyzed")
    optimized_prompts: int = Field(..., ge=0, description="Number of prompts optimized")
    avg_improvement: float = Field(..., description="Average improvement score")
    total_issues_resolved: int = Field(..., ge=0, description="Total issues resolved")
    strategy_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each strategy used"
    )
    optimization_results: List[OptimizationResult] = Field(
        default_factory=list,
        description="Individual optimization results"
    )
    generated_at: datetime = Field(default_factory=datetime.now, description="Report generation timestamp")
```

---

## 3. Functional Requirements

### 3.1 PromptExtractor Component

**Component ID:** OPT-COMP-001
**File Location:** `src/optimizer/prompt_extractor.py`
**Responsibility:** Extract prompts from workflow DSL YAML files

#### 3.1.1 Public Interface

```python
class PromptExtractor:
    """
    Extracts prompts from workflow DSL files.

    Responsible for:
    - Parsing workflow DSL YAML
    - Identifying LLM nodes
    - Extracting prompt text and metadata
    - Detecting variable placeholders
    """

    def __init__(
        self,
        workflow_catalog: WorkflowCatalog,
        yaml_parser: Optional[YamlParser] = None
    ):
        """
        Initialize PromptExtractor.

        Args:
            workflow_catalog: Catalog containing workflow metadata
            yaml_parser: YAML parser (optional, defaults to YamlParser())
        """
        pass

    def extract_from_workflow(
        self,
        workflow_id: str
    ) -> List[Prompt]:
        """
        Extract all prompts from a workflow.

        Args:
            workflow_id: Workflow identifier from catalog

        Returns:
            List of Prompt objects with metadata

        Raises:
            WorkflowNotFoundError: If workflow_id not in catalog
            PromptExtractionError: If DSL parsing fails
        """
        pass

    def extract_from_node(
        self,
        workflow_id: str,
        node_id: str
    ) -> Optional[Prompt]:
        """
        Extract prompt from a specific node.

        Args:
            workflow_id: Workflow identifier
            node_id: Node identifier

        Returns:
            Prompt object or None if node has no prompt

        Raises:
            NodeNotFoundError: If node_id not found
            PromptExtractionError: If extraction fails
        """
        pass

    def _parse_dsl_file(
        self,
        dsl_path: Path
    ) -> Dict[str, Any]:
        """
        Parse DSL YAML file into dictionary.

        Internal method. Uses yaml_parser.
        """
        pass

    def _extract_variables(
        self,
        prompt_text: str
    ) -> List[str]:
        """
        Extract variable placeholders from prompt text.

        Detects patterns like: {{variable_name}}, {variable}, $variable

        Args:
            prompt_text: Prompt text to analyze

        Returns:
            List of unique variable names (without delimiters)
        """
        pass

    def _build_node_context(
        self,
        node_data: Dict[str, Any],
        node_meta: NodeMeta
    ) -> Dict[str, Any]:
        """
        Build context dictionary for a node.

        Returns:
            Context dict with keys: label, type, position, dependencies
        """
        pass
```

#### 3.1.2 Input Data Format

**From WorkflowCatalog:**
```python
workflow_entry = WorkflowEntry(
    id="wf_001",
    label="Customer Support Agent",
    type="workflow",
    dsl_path=Path("workflows/wf_001.yml"),
    nodes=[
        NodeMeta(
            node_id="llm_node_1",
            label="Intent Classification",
            type="llm",
            path="/graph/nodes/0",
            prompt_fields=["data.prompt_template.0.text"]
        )
    ]
)
```

**From DSL YAML (sample structure):**
```yaml
graph:
  nodes:
    - id: llm_node_1
      data:
        title: Intent Classification
        type: llm
        model:
          provider: openai
          name: gpt-4
        prompt_template:
          - role: system
            text: |
              You are a customer support intent classifier.
              Classify the user's message into one of these categories:
              - product_inquiry
              - technical_support
              - billing_question
              - complaint

              User message: {{user_message}}

              Respond with the category name only.
```

#### 3.1.3 Output Data Format

```python
[
    Prompt(
        id="wf_001_llm_node_1",
        workflow_id="wf_001",
        node_id="llm_node_1",
        node_type="llm",
        text="You are a customer support intent classifier...",
        role="system",
        variables=["user_message"],
        context={
            "label": "Intent Classification",
            "position": 0,
            "model_provider": "openai",
            "model_name": "gpt-4"
        },
        extracted_at=datetime(2025, 11, 17, 10, 30, 0)
    )
]
```

#### 3.1.4 Error Handling

**Exception Hierarchy:**
```python
class PromptExtractionError(OptimizerException):
    """Base exception for extraction errors."""
    pass

class WorkflowNotFoundError(PromptExtractionError):
    """Workflow ID not found in catalog."""
    pass

class NodeNotFoundError(PromptExtractionError):
    """Node ID not found in workflow."""
    pass

class DSLParseError(PromptExtractionError):
    """DSL YAML parsing failed."""
    pass
```

**Error Handling Strategy:**
- **Workflow not found:** Raise `WorkflowNotFoundError` with workflow_id in message
- **DSL file missing:** Raise `PromptExtractionError` with file path
- **Invalid YAML syntax:** Raise `DSLParseError` with line number if available
- **Node has no prompt:** Return `None` from `extract_from_node()`, log warning
- **Malformed prompt field:** Log warning, skip node, continue extraction
- **Variable detection failure:** Log warning, return empty list for variables

**Logging Requirements:**
```python
logger.info(f"Starting prompt extraction for workflow: {workflow_id}")
logger.debug(f"Loading DSL from: {dsl_path}")
logger.info(f"Extracted {len(prompts)} prompts from workflow")
logger.warning(f"Node {node_id} has no extractable prompt, skipping")
logger.error(f"Failed to parse DSL file: {error}", exc_info=True)
```

#### 3.1.5 Performance Requirements

- Extract prompts from workflow with 100 nodes in < 2 seconds
- Memory usage: < 50MB for workflow with 1000 nodes
- Support DSL files up to 10MB in size

---

### 3.2 PromptAnalyzer Component

**Component ID:** OPT-COMP-002
**File Location:** `src/optimizer/prompt_analyzer.py`
**Responsibility:** Analyze prompt quality using rule-based heuristics

#### 3.2.1 Public Interface

```python
class PromptAnalyzer:
    """
    Analyzes prompt quality using rule-based heuristics.

    Scoring dimensions:
    - Clarity: Readability, instruction clarity, structure
    - Efficiency: Token usage, information density
    - Overall: Weighted combination
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize PromptAnalyzer.

        Args:
            config: Optimization configuration (optional)
        """
        pass

    def analyze_prompt(
        self,
        prompt: Prompt
    ) -> PromptAnalysis:
        """
        Perform comprehensive quality analysis on a prompt.

        Args:
            prompt: Prompt object to analyze

        Returns:
            PromptAnalysis with scores, issues, and suggestions

        Raises:
            PromptAnalysisError: If analysis fails
        """
        pass

    def calculate_clarity_score(
        self,
        prompt_text: str
    ) -> float:
        """
        Calculate clarity score (0-100).

        Factors:
        - Readability (Flesch Reading Ease or similar)
        - Average sentence length (shorter is clearer)
        - Instruction explicitness (presence of action verbs)
        - Structure quality (markdown formatting, bullet points)

        Returns:
            Clarity score in range [0.0, 100.0]
        """
        pass

    def calculate_efficiency_score(
        self,
        prompt_text: str
    ) -> float:
        """
        Calculate efficiency score (0-100).

        Factors:
        - Token count (fewer is better)
        - Information density (content-to-token ratio)
        - Redundancy detection (repeated phrases)

        Returns:
            Efficiency score in range [0.0, 100.0]
        """
        pass

    def detect_issues(
        self,
        prompt_text: str,
        metrics: Dict[str, float]
    ) -> List[PromptIssue]:
        """
        Detect common prompt quality issues.

        Issue Categories:
        - Clarity: Vague instructions, ambiguous language
        - Efficiency: Redundant text, overly verbose
        - Safety: Potentially harmful content patterns
        - Structure: Poor formatting, missing context

        Returns:
            List of detected issues
        """
        pass

    def generate_suggestions(
        self,
        prompt_text: str,
        issues: List[PromptIssue],
        metrics: Dict[str, float]
    ) -> List[PromptSuggestion]:
        """
        Generate improvement suggestions based on analysis.

        Returns:
            List of actionable suggestions
        """
        pass
```

#### 3.2.2 Scoring Algorithm Specifications

**Clarity Score Calculation:**
```python
def calculate_clarity_score(self, prompt_text: str) -> float:
    """
    Clarity Score = (
        readability_score * 0.4 +
        instruction_clarity * 0.3 +
        structure_score * 0.3
    )
    """

    # 1. Readability (based on Flesch Reading Ease)
    # Scale: 0-100 (higher is more readable)
    readability = self._calculate_readability(prompt_text)

    # 2. Instruction Clarity
    # Check for: action verbs, specific terms, no ambiguous words
    instruction_clarity = self._assess_instruction_clarity(prompt_text)

    # 3. Structure Score
    # Check for: markdown formatting, bullet points, clear sections
    structure_score = self._assess_structure(prompt_text)

    clarity = (
        readability * 0.4 +
        instruction_clarity * 0.3 +
        structure_score * 0.3
    )

    return min(max(clarity, 0.0), 100.0)
```

**Efficiency Score Calculation:**
```python
def calculate_efficiency_score(self, prompt_text: str) -> float:
    """
    Efficiency Score = (
        token_efficiency * 0.6 +
        information_density * 0.4
    )
    """

    # 1. Token Efficiency
    # Penalize very long prompts (> 500 tokens)
    # Formula: max(0, 100 - (token_count - 500) / 10)
    token_count = self._estimate_token_count(prompt_text)
    if token_count <= 500:
        token_efficiency = 100.0
    else:
        token_efficiency = max(0, 100 - (token_count - 500) / 10)

    # 2. Information Density
    # Ratio of unique words to total words
    # Formula: (unique_words / total_words) * 100
    information_density = self._calculate_information_density(prompt_text)

    efficiency = (
        token_efficiency * 0.6 +
        information_density * 0.4
    )

    return min(max(efficiency, 0.0), 100.0)
```

**Overall Score Calculation:**
```python
def calculate_overall_score(
    self,
    clarity_score: float,
    efficiency_score: float,
    config: OptimizationConfig
) -> float:
    """
    Overall Score = weighted average using config.scoring_weights
    """
    weights = config.scoring_weights

    overall = (
        clarity_score * weights['clarity'] +
        efficiency_score * weights['efficiency']
    )

    return min(max(overall, 0.0), 100.0)
```

#### 3.2.3 Issue Detection Rules

**Rule-Based Issue Detection:**

| Issue Category | Detection Pattern | Severity | Message Template |
|----------------|-------------------|----------|------------------|
| **Vague Instructions** | Lacks action verbs (analyze, classify, summarize) | warning | "Prompt lacks clear action verbs. Add explicit instructions." |
| **Ambiguous Language** | Contains words: maybe, possibly, try, might | warning | "Ambiguous language detected: '{word}'. Be more specific." |
| **Missing Examples** | No examples when dealing with classification/categorization | info | "Consider adding examples to clarify expected output format." |
| **Excessive Length** | Token count > 1000 | warning | "Prompt is very long ({tokens} tokens). Consider simplification." |
| **No Structure** | No markdown formatting, no bullet points | info | "Add structure (bullet points, sections) for clarity." |
| **Redundant Text** | Repeated phrases (> 3 times) | warning | "Redundant phrase detected: '{phrase}' appears {count} times." |
| **Missing Context** | No variable context or explanation | warning | "Variable {{var}} used without context. Add explanation." |

**Implementation:**
```python
def detect_issues(
    self,
    prompt_text: str,
    metrics: Dict[str, float]
) -> List[PromptIssue]:
    issues = []

    # Rule 1: Check for action verbs
    if not self._has_action_verbs(prompt_text):
        issues.append(PromptIssue(
            severity="warning",
            category="clarity",
            message="Prompt lacks clear action verbs. Add explicit instructions.",
            location=None
        ))

    # Rule 2: Check for ambiguous words
    ambiguous_words = self._find_ambiguous_words(prompt_text)
    if ambiguous_words:
        for word in ambiguous_words:
            issues.append(PromptIssue(
                severity="warning",
                category="clarity",
                message=f"Ambiguous language detected: '{word}'. Be more specific.",
                location=None
            ))

    # Rule 3: Check token count
    if metrics.get('token_count', 0) > 1000:
        issues.append(PromptIssue(
            severity="warning",
            category="efficiency",
            message=f"Prompt is very long ({metrics['token_count']} tokens). Consider simplification.",
            location=None
        ))

    # ... (continue for all rules)

    return issues
```

#### 3.2.4 Suggestion Generation

**Template-Based Suggestions:**

```python
SUGGESTION_TEMPLATES = {
    "add_structure": PromptSuggestion(
        category="structure",
        priority=2,
        suggestion="Use markdown formatting to organize content into sections",
        example="# Task\n- Point 1\n- Point 2\n\n## Context\n..."
    ),
    "simplify_language": PromptSuggestion(
        category="clarity",
        priority=1,
        suggestion="Simplify complex sentences for better readability",
        example="Instead of: 'Utilize the aforementioned methodology'\nUse: 'Use the method above'"
    ),
    "add_examples": PromptSuggestion(
        category="clarity",
        priority=2,
        suggestion="Add examples to clarify expected output format",
        example="Example:\nInput: {input}\nExpected Output: {output}"
    ),
    "reduce_redundancy": PromptSuggestion(
        category="efficiency",
        priority=1,
        suggestion="Remove redundant phrases to improve efficiency",
        example="Remove repeated phrases like 'please note that'"
    ),
}
```

#### 3.2.5 Error Handling

```python
class PromptAnalysisError(OptimizerException):
    """Base exception for analysis errors."""
    pass

class ScoringError(PromptAnalysisError):
    """Scoring calculation failed."""
    pass
```

**Error Handling Strategy:**
- **Empty prompt text:** Return analysis with all scores = 0, add critical issue
- **Token counting fails:** Log warning, use character count / 4 as estimate
- **Readability calculation fails:** Use fallback score of 50.0
- **Issue detection fails:** Log error, continue with empty issues list
- **Suggestion generation fails:** Log error, continue with empty suggestions

---

### 3.3 OptimizationEngine Component

**Component ID:** OPT-COMP-003
**File Location:** `src/optimizer/optimization_engine.py`
**Responsibility:** Generate optimized prompt variants using defined strategies

#### 3.3.1 Public Interface

```python
class OptimizationEngine:
    """
    Generates optimized prompt variants using optimization strategies.

    Strategies:
    - clarity_focus: Improve readability and instruction clarity
    - efficiency_focus: Reduce token count and redundancy
    - structure_optimization: Improve formatting and organization
    """

    def __init__(
        self,
        analyzer: PromptAnalyzer,
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize OptimizationEngine.

        Args:
            analyzer: PromptAnalyzer instance
            config: Optimization configuration
        """
        pass

    def optimize(
        self,
        prompt: Prompt,
        strategy: str,
        baseline_metrics: Optional[PerformanceMetrics] = None
    ) -> OptimizationResult:
        """
        Optimize a prompt using specified strategy.

        Args:
            prompt: Prompt to optimize
            strategy: Strategy name (clarity_focus, efficiency_focus, structure_optimization)
            baseline_metrics: Optional baseline performance metrics

        Returns:
            OptimizationResult with optimized prompt and analysis

        Raises:
            OptimizationError: If optimization fails
            InvalidStrategyError: If strategy name is invalid
        """
        pass

    def apply_clarity_focus(
        self,
        prompt_text: str
    ) -> str:
        """
        Apply clarity-focused optimization strategy.

        Transformations:
        - Simplify complex sentences
        - Replace jargon with plain language
        - Add structure (markdown, bullet points)
        - Make instructions more explicit

        Returns:
            Optimized prompt text
        """
        pass

    def apply_efficiency_focus(
        self,
        prompt_text: str
    ) -> str:
        """
        Apply efficiency-focused optimization strategy.

        Transformations:
        - Remove redundant phrases
        - Compress verbose explanations
        - Eliminate filler words
        - Consolidate repeated instructions

        Returns:
            Optimized prompt text
        """
        pass

    def apply_structure_optimization(
        self,
        prompt_text: str
    ) -> str:
        """
        Apply structure optimization strategy.

        Transformations:
        - Add markdown headers
        - Convert lists to bullet points
        - Separate sections clearly
        - Add context blocks

        Returns:
            Optimized prompt text
        """
        pass

    def _calculate_confidence(
        self,
        original_analysis: PromptAnalysis,
        optimized_analysis: PromptAnalysis,
        improvement_score: float
    ) -> float:
        """
        Calculate confidence in optimization result.

        Factors:
        - Magnitude of improvement
        - Number of issues resolved
        - Consistency across metrics

        Returns:
            Confidence score in range [0.0, 1.0]
        """
        pass
```

#### 3.3.2 Optimization Strategy Specifications

**Strategy 1: Clarity Focus**

**Goal:** Improve readability and instruction clarity
**Target Metrics:** clarity_score increase by 10-20 points

**Transformation Rules:**

| Rule | Pattern | Transformation | Example |
|------|---------|----------------|---------|
| **Simplify Sentences** | Sentence > 30 words | Split into 2-3 shorter sentences | "You are an AI assistant that should analyze the user's input and provide a detailed response..." → "You are an AI assistant. Analyze the user's input. Provide a detailed response." |
| **Replace Jargon** | Technical terms | Replace with simpler alternatives | "Utilize" → "Use", "Ascertain" → "Find out" |
| **Add Structure** | Unstructured text | Add markdown headers and bullets | Plain text → "## Task\n- Step 1\n- Step 2" |
| **Explicit Instructions** | Vague verbs | Replace with specific action verbs | "Handle" → "Classify", "Deal with" → "Process" |
| **Add Examples** | Classification tasks | Insert example input/output | Add: "Example:\nInput: hello\nOutput: greeting" |

**Implementation Pattern:**
```python
def apply_clarity_focus(self, prompt_text: str) -> str:
    optimized = prompt_text

    # Step 1: Split long sentences
    optimized = self._split_long_sentences(optimized, max_words=30)

    # Step 2: Replace jargon
    optimized = self._replace_jargon(optimized, self.JARGON_MAP)

    # Step 3: Add structure
    optimized = self._add_markdown_structure(optimized)

    # Step 4: Make instructions explicit
    optimized = self._strengthen_action_verbs(optimized)

    return optimized
```

---

**Strategy 2: Efficiency Focus**

**Goal:** Reduce token count and improve information density
**Target Metrics:** efficiency_score increase by 10-20 points, token_count reduction by 15-30%

**Transformation Rules:**

| Rule | Pattern | Transformation | Example |
|------|---------|----------------|---------|
| **Remove Redundancy** | Repeated phrases | Remove duplicates | "Please note that... please note that..." → "Please note that..." (once) |
| **Eliminate Filler** | Filler words/phrases | Remove unnecessary words | "In order to" → "To", "At this point in time" → "Now" |
| **Compress Verbose** | Wordy explanations | Use concise alternatives | "Due to the fact that" → "Because" |
| **Consolidate Lists** | Multiple similar items | Merge related items | "Do A, do B, do C" → "Do A, B, and C" |
| **Remove Qualifiers** | Hedge words | Remove weakening words | "somewhat", "rather", "quite", "very" |

**Filler Word List:**
```python
FILLER_PHRASES = [
    "in order to",
    "due to the fact that",
    "at this point in time",
    "for the purpose of",
    "in the event that",
    "it should be noted that",
    "it is important to note",
]

FILLER_WORDS = [
    "really", "very", "quite", "rather", "somewhat",
    "just", "actually", "basically", "literally"
]
```

**Implementation Pattern:**
```python
def apply_efficiency_focus(self, prompt_text: str) -> str:
    optimized = prompt_text

    # Step 1: Remove redundant phrases
    optimized = self._remove_redundant_phrases(optimized)

    # Step 2: Replace verbose phrases
    optimized = self._replace_verbose_phrases(optimized, self.VERBOSE_MAP)

    # Step 3: Remove filler words
    optimized = self._remove_filler_words(optimized, self.FILLER_WORDS)

    # Step 4: Consolidate repetitive lists
    optimized = self._consolidate_lists(optimized)

    return optimized
```

---

**Strategy 3: Structure Optimization**

**Goal:** Improve prompt organization and formatting
**Target Metrics:** clarity_score increase by 5-15 points (through better structure)

**Transformation Rules:**

| Rule | Pattern | Transformation | Example |
|------|---------|----------------|---------|
| **Add Headers** | Sections without titles | Insert markdown headers | Plain text → "## Task\n{text}\n\n## Context\n{text}" |
| **Bulletize Lists** | Line-separated items | Convert to bullet points | "Item 1\nItem 2" → "- Item 1\n- Item 2" |
| **Separate Sections** | Mixed content | Add blank lines between sections | No separation → Double newline separation |
| **Format Code** | Code snippets | Wrap in code blocks | Code → "```\ncode\n```" |
| **Highlight Variables** | Variables in text | Use consistent formatting | {{var}} or {var} → **{{var}}** |

**Implementation Pattern:**
```python
def apply_structure_optimization(self, prompt_text: str) -> str:
    optimized = prompt_text

    # Step 1: Detect sections (based on content analysis)
    sections = self._detect_sections(optimized)

    # Step 2: Add markdown headers
    optimized = self._add_section_headers(optimized, sections)

    # Step 3: Convert lists to bullets
    optimized = self._bulletize_lists(optimized)

    # Step 4: Format code blocks
    optimized = self._format_code_blocks(optimized)

    # Step 5: Ensure proper spacing
    optimized = self._normalize_spacing(optimized)

    return optimized
```

#### 3.3.3 Confidence Calculation

```python
def _calculate_confidence(
    self,
    original_analysis: PromptAnalysis,
    optimized_analysis: PromptAnalysis,
    improvement_score: float
) -> float:
    """
    Calculate confidence based on multiple factors.
    """

    # Factor 1: Magnitude of improvement (0-0.4)
    improvement_factor = min(abs(improvement_score) / 20.0, 0.4)

    # Factor 2: Issues resolved (0-0.3)
    issues_resolved = len(original_analysis.issues) - len(optimized_analysis.issues)
    issue_factor = min(max(issues_resolved, 0) / 5.0, 0.3)

    # Factor 3: Metric consistency (0-0.3)
    # Check if clarity AND efficiency both improved
    clarity_improved = optimized_analysis.clarity_score > original_analysis.clarity_score
    efficiency_improved = optimized_analysis.efficiency_score > original_analysis.efficiency_score

    if clarity_improved and efficiency_improved:
        consistency_factor = 0.3
    elif clarity_improved or efficiency_improved:
        consistency_factor = 0.15
    else:
        consistency_factor = 0.0

    confidence = improvement_factor + issue_factor + consistency_factor

    return min(max(confidence, 0.0), 1.0)
```

#### 3.3.4 Error Handling

```python
class OptimizationError(OptimizerException):
    """Base exception for optimization errors."""
    pass

class InvalidStrategyError(OptimizationError):
    """Invalid strategy name provided."""
    pass

class OptimizationFailedError(OptimizationError):
    """Optimization process failed."""
    pass
```

**Error Handling Strategy:**
- **Invalid strategy:** Raise `InvalidStrategyError` immediately
- **Transformation fails:** Log error, return original prompt, set confidence=0
- **Analysis fails:** Catch exception, wrap in `OptimizationError`, propagate
- **Negative improvement:** NOT an error, return result with negative improvement_score

---

### 3.4 VersionManager Component

**Component ID:** OPT-COMP-004
**File Location:** `src/optimizer/version_manager.py`
**Responsibility:** Manage prompt version history and comparisons

#### 3.4.1 Public Interface

```python
class VersionManager:
    """
    Manages prompt version history.

    Provides:
    - Version creation and storage
    - Version retrieval and history
    - Version comparison
    - Version validation
    """

    def __init__(self):
        """
        Initialize VersionManager with in-memory storage.

        Storage structure:
        {
            "prompt_id": {
                "versions": [PromptVersion, ...],
                "current_version": "1.2.0"
            }
        }
        """
        pass

    def create_version(
        self,
        prompt_id: str,
        text: str,
        analysis: PromptAnalysis,
        author: str,
        parent_version: Optional[str] = None,
        change_summary: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> PromptVersion:
        """
        Create a new prompt version.

        Version numbering:
        - First version: 1.0.0 (parent_version must be None)
        - Subsequent versions: Auto-increment minor version

        Args:
            prompt_id: Base prompt identifier
            text: Prompt text for this version
            analysis: Quality analysis for this version
            author: Creator (baseline, optimizer, manual)
            parent_version: Parent version number (None for first version)
            change_summary: Description of changes
            tags: Optional version tags

        Returns:
            Created PromptVersion object

        Raises:
            VersionConflictError: If parent_version doesn't exist
            ValueError: If first version has parent_version set
        """
        pass

    def get_version(
        self,
        prompt_id: str,
        version: str
    ) -> Optional[PromptVersion]:
        """
        Retrieve a specific version.

        Args:
            prompt_id: Base prompt identifier
            version: Version number (e.g., "1.2.0")

        Returns:
            PromptVersion or None if not found
        """
        pass

    def get_current_version(
        self,
        prompt_id: str
    ) -> Optional[PromptVersion]:
        """
        Get the current (latest) version of a prompt.

        Returns:
            Latest PromptVersion or None if no versions exist
        """
        pass

    def get_version_history(
        self,
        prompt_id: str
    ) -> List[PromptVersion]:
        """
        Get full version history for a prompt.

        Returns:
            List of PromptVersion objects, sorted by version (oldest first)
        """
        pass

    def compare_versions(
        self,
        prompt_id: str,
        version_a: str,
        version_b: str
    ) -> VersionComparison:
        """
        Compare two versions of a prompt.

        Args:
            prompt_id: Base prompt identifier
            version_a: First version number
            version_b: Second version number

        Returns:
            VersionComparison with detailed comparison

        Raises:
            VersionNotFoundError: If either version doesn't exist
        """
        pass

    def _generate_next_version(
        self,
        prompt_id: str,
        author: str
    ) -> str:
        """
        Generate next version number.

        Rules:
        - First version: "1.0.0"
        - Optimizer improvements: Increment minor (1.0.0 → 1.1.0)
        - Manual edits: Increment patch (1.1.0 → 1.1.1)

        Returns:
            Next version number
        """
        pass

    def _calculate_text_diff(
        self,
        text_a: str,
        text_b: str
    ) -> str:
        """
        Calculate unified diff between two text strings.

        Uses difflib.unified_diff.

        Returns:
            Unified diff string
        """
        pass
```

#### 3.4.2 Storage Structure

**In-Memory Storage (MVP):**
```python
{
    "wf_001_llm_node_1": {
        "versions": [
            PromptVersion(
                prompt_id="wf_001_llm_node_1",
                version="1.0.0",
                text="Original prompt text",
                analysis=PromptAnalysis(...),
                created_at=datetime(2025, 11, 17, 10, 0, 0),
                author="baseline",
                parent_version=None,
                change_summary="Initial baseline version",
                tags=["baseline"]
            ),
            PromptVersion(
                prompt_id="wf_001_llm_node_1",
                version="1.1.0",
                text="Optimized prompt text",
                analysis=PromptAnalysis(...),
                created_at=datetime(2025, 11, 17, 11, 0, 0),
                author="optimizer",
                parent_version="1.0.0",
                change_summary="Applied clarity_focus optimization",
                tags=["optimized"]
            )
        ],
        "current_version": "1.1.0"
    }
}
```

#### 3.4.3 Version Comparison Algorithm

```python
def compare_versions(
    self,
    prompt_id: str,
    version_a: str,
    version_b: str
) -> VersionComparison:
    """
    Compare two versions.
    """

    v_a = self.get_version(prompt_id, version_a)
    v_b = self.get_version(prompt_id, version_b)

    if not v_a or not v_b:
        raise VersionNotFoundError(f"Version not found for {prompt_id}")

    # Calculate text diff
    text_diff = self._calculate_text_diff(v_a.text, v_b.text)

    # Calculate score deltas
    score_delta = v_b.analysis.overall_score - v_a.analysis.overall_score
    clarity_delta = v_b.analysis.clarity_score - v_a.analysis.clarity_score
    efficiency_delta = v_b.analysis.efficiency_score - v_a.analysis.efficiency_score

    # Calculate issues resolved/introduced
    issues_a = set(issue.message for issue in v_a.analysis.issues)
    issues_b = set(issue.message for issue in v_b.analysis.issues)

    issues_resolved = len(issues_a - issues_b)
    issues_introduced = len(issues_b - issues_a)

    # Generate recommendation
    if score_delta > 5:
        recommendation = "prefer_b"
    elif score_delta < -5:
        recommendation = "prefer_a"
    else:
        recommendation = "equivalent"

    return VersionComparison(
        prompt_id=prompt_id,
        version_a=version_a,
        version_b=version_b,
        text_diff=text_diff,
        score_delta=score_delta,
        clarity_delta=clarity_delta,
        efficiency_delta=efficiency_delta,
        issues_resolved=issues_resolved,
        issues_introduced=issues_introduced,
        recommendation=recommendation,
        compared_at=datetime.now()
    )
```

#### 3.4.4 Error Handling

```python
class VersionConflictError(OptimizerException):
    """Version conflict or invalid version reference."""
    pass

class VersionNotFoundError(OptimizerException):
    """Requested version not found."""
    pass
```

**Error Handling Strategy:**
- **Parent version not found:** Raise `VersionConflictError`
- **First version with parent:** Raise `ValueError`
- **Get non-existent version:** Return `None` (not an error)
- **Compare non-existent versions:** Raise `VersionNotFoundError`

---

### 3.5 OptimizerService Component

**Component ID:** OPT-COMP-005
**File Location:** `src/optimizer/optimizer_service.py`
**Responsibility:** Orchestrate complete optimization workflows (facade pattern)

#### 3.5.1 Public Interface

```python
class OptimizerService:
    """
    Orchestration service for prompt optimization workflows.

    Provides high-level API for:
    - Complete optimization cycles
    - Batch optimization
    - Report generation
    - Integration with other modules
    """

    def __init__(
        self,
        workflow_catalog: WorkflowCatalog,
        config: Optional[OptimizationConfig] = None,
        yaml_parser: Optional[YamlParser] = None
    ):
        """
        Initialize OptimizerService.

        Args:
            workflow_catalog: Catalog of workflows
            config: Optimization configuration
            yaml_parser: YAML parser (optional)
        """
        self.catalog = workflow_catalog
        self.config = config or OptimizationConfig()

        # Initialize components
        self.extractor = PromptExtractor(workflow_catalog, yaml_parser)
        self.analyzer = PromptAnalyzer(config)
        self.engine = OptimizationEngine(self.analyzer, config)
        self.version_manager = VersionManager()

    def run_optimization_cycle(
        self,
        workflow_id: str,
        baseline_metrics: Optional[PerformanceMetrics] = None,
        strategy: str = "auto"
    ) -> List[PromptPatch]:
        """
        Run complete optimization cycle for a workflow.

        Steps:
        1. Extract all prompts from workflow
        2. Analyze each prompt
        3. Optimize prompts below threshold
        4. Create version records
        5. Generate PromptPatch objects

        Args:
            workflow_id: Workflow identifier
            baseline_metrics: Baseline performance metrics (optional)
            strategy: Optimization strategy or "auto" for automatic selection

        Returns:
            List of PromptPatch objects to apply

        Raises:
            WorkflowNotFoundError: If workflow_id not found
            OptimizationError: If optimization cycle fails
        """
        pass

    def optimize_prompt(
        self,
        prompt: Prompt,
        strategy: str,
        create_version: bool = True
    ) -> OptimizationResult:
        """
        Optimize a single prompt.

        Args:
            prompt: Prompt to optimize
            strategy: Optimization strategy
            create_version: Whether to create version record

        Returns:
            OptimizationResult
        """
        pass

    def get_optimization_report(
        self,
        workflow_id: str
    ) -> OptimizationReport:
        """
        Generate optimization report for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            OptimizationReport with aggregated results
        """
        pass

    def _select_strategy(
        self,
        analysis: PromptAnalysis
    ) -> str:
        """
        Auto-select optimization strategy based on analysis.

        Selection logic:
        - If clarity_score < 60: use clarity_focus
        - If efficiency_score < 60: use efficiency_focus
        - If both scores < 60: use clarity_focus (prioritize clarity)
        - If scores >= 60 but no structure: use structure_optimization

        Returns:
            Strategy name
        """
        pass

    def _generate_prompt_patch(
        self,
        prompt: Prompt,
        optimization_result: OptimizationResult
    ) -> PromptPatch:
        """
        Generate PromptPatch from optimization result.

        Returns:
            PromptPatch object for use with PromptPatchEngine
        """
        pass
```

#### 3.5.2 Workflow: run_optimization_cycle()

**Algorithm:**

```python
def run_optimization_cycle(
    self,
    workflow_id: str,
    baseline_metrics: Optional[PerformanceMetrics] = None,
    strategy: str = "auto"
) -> List[PromptPatch]:

    logger.info(f"Starting optimization cycle for workflow: {workflow_id}")

    # Step 1: Extract prompts
    prompts = self.extractor.extract_from_workflow(workflow_id)
    logger.info(f"Extracted {len(prompts)} prompts")

    patches = []

    for prompt in prompts:
        # Step 2: Analyze prompt
        analysis = self.analyzer.analyze_prompt(prompt)

        # Step 3: Check if optimization needed
        if analysis.overall_score >= 80:
            logger.info(f"Prompt {prompt.id} has high score ({analysis.overall_score:.1f}), skipping")
            continue

        # Step 4: Select strategy
        selected_strategy = strategy
        if strategy == "auto":
            selected_strategy = self._select_strategy(analysis)

        logger.info(f"Optimizing {prompt.id} with strategy: {selected_strategy}")

        # Step 5: Optimize
        try:
            result = self.engine.optimize(prompt, selected_strategy, baseline_metrics)

            # Step 6: Check improvement threshold
            if result.improvement_score < self.config.improvement_threshold:
                logger.info(
                    f"Improvement ({result.improvement_score:.1f}) below threshold "
                    f"({self.config.improvement_threshold}), skipping"
                )
                continue

            # Step 7: Check confidence threshold
            if result.confidence < self.config.confidence_threshold:
                logger.warning(
                    f"Confidence ({result.confidence:.2f}) below threshold "
                    f"({self.config.confidence_threshold}), skipping"
                )
                continue

            # Step 8: Create version
            if self.config.enable_version_tracking:
                # Create baseline version if not exists
                if not self.version_manager.get_version(prompt.id, "1.0.0"):
                    self.version_manager.create_version(
                        prompt_id=prompt.id,
                        text=prompt.text,
                        analysis=analysis,
                        author="baseline",
                        change_summary="Initial baseline version"
                    )

                # Create optimized version
                self.version_manager.create_version(
                    prompt_id=prompt.id,
                    text=result.optimized_prompt,
                    analysis=result.optimized_analysis,
                    author="optimizer",
                    parent_version="1.0.0",  # Always based on baseline for MVP
                    change_summary=f"Applied {selected_strategy} optimization"
                )

            # Step 9: Generate patch
            patch = self._generate_prompt_patch(prompt, result)
            patches.append(patch)

            logger.info(
                f"Successfully optimized {prompt.id}: "
                f"improvement={result.improvement_score:.1f}, "
                f"confidence={result.confidence:.2f}"
            )

        except OptimizationError as e:
            logger.error(f"Failed to optimize {prompt.id}: {e}")
            continue

    logger.info(f"Optimization cycle complete: generated {len(patches)} patches")

    return patches
```

#### 3.5.3 Auto-Strategy Selection

```python
def _select_strategy(self, analysis: PromptAnalysis) -> str:
    """
    Auto-select strategy based on analysis.
    """

    clarity = analysis.clarity_score
    efficiency = analysis.efficiency_score

    # Priority 1: Fix critical clarity issues
    if clarity < 60:
        return "clarity_focus"

    # Priority 2: Fix efficiency issues
    if efficiency < 60:
        return "efficiency_focus"

    # Priority 3: Check for structure issues
    structure_issues = [
        issue for issue in analysis.issues
        if issue.category == "structure"
    ]

    if structure_issues:
        return "structure_optimization"

    # Default: Use clarity focus (safest option)
    return "clarity_focus"
```

#### 3.5.4 PromptPatch Generation

```python
def _generate_prompt_patch(
    self,
    prompt: Prompt,
    optimization_result: OptimizationResult
) -> PromptPatch:
    """
    Generate PromptPatch from optimization result.
    """

    # Create selector targeting specific node
    selector = PromptSelector(
        by_id=prompt.node_id,
        constraints={
            "workflow_type": prompt.context.get("workflow_type"),
            "if_missing": "skip"
        }
    )

    # Create strategy for replacement
    strategy = PromptStrategy(
        mode="replace",
        content=optimization_result.optimized_prompt,
        fallback_value=prompt.text  # Fallback to original if patch fails
    )

    return PromptPatch(
        selector=selector,
        strategy=strategy
    )
```

#### 3.5.5 Error Handling

**Error Handling Strategy:**
- **Workflow not found:** Propagate `WorkflowNotFoundError` from extractor
- **Individual prompt optimization fails:** Log error, continue with next prompt
- **No prompts extracted:** Return empty list (not an error)
- **All optimizations rejected:** Return empty list, log warning
- **Version creation fails:** Log error, continue without versioning

---

## 4. Interface Requirements

### 4.1 Integration with Config Module

**Interface ID:** OPT-INT-001
**Dependency:** `src/config`

#### 4.1.1 Reading Workflow Configuration

**Required Imports:**
```python
from src.config import (
    ConfigLoader,
    WorkflowCatalog,
    WorkflowEntry,
    NodeMeta
)
```

**Usage Pattern:**
```python
# Initialize optimizer with catalog
loader = ConfigLoader()
catalog = loader.load_workflow_catalog("config/workflows.yaml")

# Pass catalog to optimizer service
optimizer = OptimizerService(catalog)
```

**Contract:**
- Optimizer MUST NOT modify `WorkflowCatalog` objects
- Optimizer MUST use `WorkflowEntry.dsl_path` to locate DSL files
- Optimizer MUST respect `NodeMeta.prompt_fields` for prompt extraction
- Optimizer MUST handle missing DSL files gracefully (raise `PromptExtractionError`)

#### 4.1.2 Generating PromptPatch Objects

**Required Imports:**
```python
from src.config import (
    PromptPatch,
    PromptSelector,
    PromptStrategy,
    PromptTemplate
)
```

**Output Contract:**
```python
# Optimizer generates patches that are compatible with PromptPatchEngine
patches: List[PromptPatch] = optimizer.run_optimization_cycle(workflow_id)

# These patches can be directly used with existing PromptPatchEngine
from src.optimizer import PromptPatchEngine
patch_engine = PromptPatchEngine(catalog, yaml_parser)
modified_dsl = patch_engine.apply_patches(workflow_id, original_dsl, patches)
```

**Validation Requirements:**
- Generated `PromptPatch` MUST pass Pydantic validation
- `PromptSelector.by_id` MUST reference valid node IDs from catalog
- `PromptStrategy.mode` MUST be `"replace"` (MVP only supports replacement)
- `PromptStrategy.content` MUST contain optimized prompt text

#### 4.1.3 Applying Patches to Test Plans

**Integration Flow:**
```python
# 1. Load test plan
test_plan = loader.load_test_plan("config/test_plan.yaml")

# 2. Run optimization
patches = optimizer.run_optimization_cycle(workflow_id, baseline_metrics)

# 3. Update test plan with patches
for workflow_entry in test_plan.workflows:
    if workflow_entry.catalog_id == workflow_id:
        # Add optimization variant
        variant = PromptVariant(
            variant_id=f"optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Optimizer-generated variant (strategy: {strategy})",
            weight=1.0,
            nodes=patches
        )

        if workflow_entry.prompt_optimization is None:
            workflow_entry.prompt_optimization = []

        workflow_entry.prompt_optimization.append(variant)

# 4. Save updated test plan
# (requires ConfigLoader.save_test_plan() - feature request)
```

**Contract:**
- Optimizer MUST generate patches that fit into `PromptVariant.nodes`
- Optimizer SHOULD provide metadata for variant description
- Test plan updates MUST preserve existing variants

---

### 4.2 Integration with Executor Module

**Interface ID:** OPT-INT-002
**Dependency:** `src/executor`

#### 4.2.1 Consuming Test Execution Results

**Required Imports:**
```python
from src.executor import (
    ExecutorService,
    TaskResult,
    RunExecutionResult
)
```

**Usage Pattern:**
```python
# 1. Executor runs tests and produces results
executor = ExecutorService()
run_result: RunExecutionResult = executor.execute_test_plan(manifest)

# 2. Optimizer receives aggregated results (does NOT re-run executor)
# Optimizer gets results via external orchestration (e.g., main.py)

# 3. Optimizer uses baseline metrics for context (optional)
baseline_metrics = collector.get_statistics(workflow_id)
patches = optimizer.run_optimization_cycle(workflow_id, baseline_metrics)
```

**Contract:**
- Optimizer MUST NOT trigger test execution directly
- Optimizer MAY use `PerformanceMetrics` for strategy selection hints
- Optimizer MUST be executable WITHOUT executor results (baseline_metrics is optional)

#### 4.2.2 Data Flow

```
┌─────────────┐
│  Executor   │ → Produces TaskResult, RunExecutionResult
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Collector  │ → Aggregates to PerformanceMetrics
└─────┬───────┘
      │
      ▼ (optional input)
┌─────────────┐
│  Optimizer  │ → Analyzes prompts, generates patches
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Test Plan   │ → Updated with optimization patches
└─────────────┘
      │
      ▼ (next iteration)
┌─────────────┐
│  Executor   │ → Re-runs with optimized prompts
└─────────────┘
```

**Interface Methods:**
- Optimizer does NOT call executor methods directly
- Communication is via shared data structures (config files, metrics)

---

### 4.3 Integration with Collector Module

**Interface ID:** OPT-INT-003
**Dependency:** `src/collector`

#### 4.3.1 Reading Performance Metrics

**Required Imports:**
```python
from src.collector import (
    DataCollector,
    PerformanceMetrics,
    TestResult
)
```

**Usage Pattern:**
```python
# 1. Collector aggregates test results
collector = DataCollector()
for result in test_results:
    collector.collect_result(result)

# 2. Get metrics for specific workflow
metrics: PerformanceMetrics = collector.get_statistics(workflow_id)

# 3. Pass to optimizer as baseline context
patches = optimizer.run_optimization_cycle(
    workflow_id,
    baseline_metrics=metrics,  # Optional parameter
    strategy="auto"
)
```

**Contract:**
- Optimizer MUST handle `baseline_metrics=None` gracefully
- Optimizer MAY use metrics for strategy hints but MUST NOT require them
- Optimizer MUST NOT modify `PerformanceMetrics` objects

#### 4.3.2 Metrics Usage in Optimization

**How Metrics Influence Optimization:**

```python
def _select_strategy(
    self,
    analysis: PromptAnalysis,
    baseline_metrics: Optional[PerformanceMetrics]
) -> str:
    """
    Select strategy considering both analysis and performance metrics.
    """

    # Primary decision based on prompt analysis
    strategy = self._select_strategy_from_analysis(analysis)

    # Optional: Adjust based on performance metrics
    if baseline_metrics:
        # If success rate is very low, prioritize clarity
        if baseline_metrics.success_rate < 0.5:
            return "clarity_focus"

        # If avg execution time is high, consider efficiency
        if baseline_metrics.avg_execution_time > 10.0:
            if strategy != "clarity_focus":  # Don't override clarity
                return "efficiency_focus"

    return strategy
```

**Metrics Fields Used:**
- `success_rate`: Low success rate → prioritize clarity
- `avg_execution_time`: High latency → consider efficiency (if appropriate)
- Other fields (token usage, cost) → for future enhancements

---

### 4.4 External Interfaces

**Interface ID:** OPT-INT-004

#### 4.4.1 CLI Integration

**Command Specification:**

```bash
# Basic optimization
python src/main.py --mode optimize --workflow-id wf_001

# With strategy specification
python src/main.py --mode optimize --workflow-id wf_001 --strategy clarity_focus

# With baseline metrics file
python src/main.py --mode optimize --workflow-id wf_001 --baseline-metrics results/metrics.json

# Generate optimization report
python src/main.py --mode optimize-report --workflow-id wf_001 --output report.json
```

**Main.py Integration Point:**

```python
# In src/main.py

def run_optimization_mode(args):
    """
    Run optimization workflow.
    """
    from src.config import ConfigLoader
    from src.optimizer import OptimizerService
    from src.collector import DataCollector

    # Load configuration
    loader = ConfigLoader()
    catalog = loader.load_workflow_catalog(args.workflow_catalog)
    test_plan = loader.load_test_plan(args.test_plan)

    # Initialize optimizer
    optimizer = OptimizerService(catalog)

    # Load baseline metrics if provided
    baseline_metrics = None
    if args.baseline_metrics:
        collector = DataCollector()
        # ... load metrics from file
        baseline_metrics = collector.get_statistics(args.workflow_id)

    # Run optimization
    logger.info(f"Starting optimization for workflow: {args.workflow_id}")
    patches = optimizer.run_optimization_cycle(
        workflow_id=args.workflow_id,
        baseline_metrics=baseline_metrics,
        strategy=args.strategy or "auto"
    )

    # Update test plan
    # ... (add patches to test plan)

    # Save updated test plan
    # ... (save to file)

    logger.info(f"Optimization complete: generated {len(patches)} patches")

    # Optionally generate report
    if args.generate_report:
        report = optimizer.get_optimization_report(args.workflow_id)
        # ... save report
```

**CLI Arguments:**

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--mode` | str | Yes | - | Mode: `optimize` or `optimize-report` |
| `--workflow-id` | str | Yes | - | Workflow identifier |
| `--strategy` | str | No | `auto` | Optimization strategy |
| `--baseline-metrics` | path | No | None | Path to baseline metrics file |
| `--output` | path | No | `stdout` | Output file for patches/report |
| `--apply-patches` | flag | No | False | Apply patches to test plan immediately |
| `--dry-run` | flag | No | False | Generate patches without applying |

#### 4.4.2 Configuration File Interface

**YAML Configuration:**

**File:** `config/optimizer.yaml` (NEW)

```yaml
# Optimizer Module Configuration

# Default optimization settings
defaults:
  strategy: clarity_focus
  improvement_threshold: 5.0  # Minimum improvement percentage
  confidence_threshold: 0.5   # Minimum confidence to accept
  max_iterations: 3
  enable_version_tracking: true

# Scoring weights
scoring:
  clarity_weight: 0.6
  efficiency_weight: 0.4

# Strategy-specific configurations
strategies:
  clarity_focus:
    enabled: true
    max_sentence_length: 30
    jargon_replacement: true
    add_structure: true

  efficiency_focus:
    enabled: true
    remove_filler: true
    target_compression: 0.7  # Target 30% reduction

  structure_optimization:
    enabled: true
    add_headers: true
    bulletize_lists: true

# Issue detection thresholds
issue_detection:
  max_token_count: 1000
  min_clarity_score: 60
  min_efficiency_score: 60

# Version management
versioning:
  enable: true
  max_versions_per_prompt: 50
  auto_cleanup: false
```

**Loading Configuration:**

```python
class OptimizerService:
    def __init__(
        self,
        workflow_catalog: WorkflowCatalog,
        config_path: Optional[str] = None
    ):
        # Load from file if provided
        if config_path:
            self.config = self._load_config_from_yaml(config_path)
        else:
            self.config = OptimizationConfig()  # Use defaults
```

---

## 5. Non-Functional Requirements

### 5.1 Performance

**Requirement ID:** OPT-NFR-001

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Prompt Extraction** | < 2 seconds for 100 nodes | Time from `extract_from_workflow()` call to return |
| **Prompt Analysis** | < 0.5 seconds per prompt | Time from `analyze_prompt()` call to return |
| **Optimization** | < 1 second per prompt | Time from `optimize()` call to return |
| **Full Optimization Cycle** | < 30 seconds for 20 prompts | Time from `run_optimization_cycle()` call to return |
| **Memory Usage** | < 200MB for 1000 prompts | Peak memory during optimization cycle |
| **Version Storage** | < 10MB for 1000 versions | In-memory storage size |

**Performance Testing:**
```python
# Performance test example
def test_optimization_cycle_performance():
    workflow_id = "large_workflow"  # Contains 20 prompts

    start_time = time.time()
    patches = optimizer.run_optimization_cycle(workflow_id)
    duration = time.time() - start_time

    assert duration < 30.0, f"Optimization cycle too slow: {duration:.2f}s"
```

---

### 5.2 Testability

**Requirement ID:** OPT-NFR-002

**Test Coverage Targets:**
- **Minimum:** 80% line coverage
- **Target:** 90% line coverage
- **Branch Coverage:** 80% minimum

**Testability Requirements:**

1. **Dependency Injection:**
   ```python
   # All components MUST accept dependencies via constructor
   class PromptExtractor:
       def __init__(
           self,
           workflow_catalog: WorkflowCatalog,
           yaml_parser: Optional[YamlParser] = None  # Injectable
       ):
           self.yaml_parser = yaml_parser or YamlParser()
   ```

2. **Mock-Friendly Design:**
   ```python
   # All external dependencies MUST be mockable
   def test_optimization_engine_with_mock_analyzer():
       mock_analyzer = Mock(spec=PromptAnalyzer)
       mock_analyzer.analyze_prompt.return_value = PromptAnalysis(...)

       engine = OptimizationEngine(analyzer=mock_analyzer)
       result = engine.optimize(prompt, "clarity_focus")

       mock_analyzer.analyze_prompt.assert_called()
   ```

3. **Deterministic Behavior:**
   - All randomness MUST be seedable
   - All timestamps MUST be injectable (not hardcoded `datetime.now()`)
   - All file I/O MUST be abstracted (use FileSystemReader pattern)

4. **Test Fixtures:**
   ```python
   # Fixtures MUST be provided in src/test/optimizer/fixtures/
   - sample_workflow_dsl.yaml
   - sample_prompts.yaml
   - expected_analysis_results.yaml
   - expected_optimization_results.yaml
   ```

---

### 5.3 Maintainability

**Requirement ID:** OPT-NFR-003

**Code Quality Requirements:**

1. **Documentation:**
   - All public classes MUST have docstrings
   - All public methods MUST have docstrings with Args/Returns/Raises
   - Complex algorithms MUST have inline comments

2. **Naming Conventions:**
   - Follow PEP 8 naming conventions
   - Use descriptive names (no single-letter variables except loop iterators)
   - Prefix private methods with `_`

3. **Complexity Limits:**
   - Maximum cyclomatic complexity: 10 per method
   - Maximum method length: 50 lines
   - Maximum class length: 500 lines

4. **Type Hints:**
   - All public methods MUST have type hints
   - Use `Optional[T]` for nullable parameters
   - Use `List[T]`, `Dict[K, V]` for collections

5. **No Circular Dependencies:**
   - Optimizer MUST NOT import from executor/collector (only their models)
   - Use dependency injection to break circular dependencies

**Code Review Checklist:**
```markdown
- [ ] All methods have docstrings
- [ ] Type hints present on all public methods
- [ ] No methods exceed 50 lines
- [ ] No circular imports
- [ ] All exceptions are custom (inherit from OptimizerException)
- [ ] Logging at appropriate levels
- [ ] No hardcoded paths or magic numbers
```

---

### 5.4 Error Handling

**Requirement ID:** OPT-NFR-004

**Exception Hierarchy:**

```python
# src/optimizer/exceptions.py

class OptimizerException(Exception):
    """Base exception for optimizer module."""
    pass

# Extraction errors
class PromptExtractionError(OptimizerException):
    """Prompt extraction failed."""
    pass

class WorkflowNotFoundError(PromptExtractionError):
    """Workflow ID not found in catalog."""
    pass

class NodeNotFoundError(PromptExtractionError):
    """Node ID not found in workflow."""
    pass

class DSLParseError(PromptExtractionError):
    """DSL YAML parsing failed."""
    pass

# Analysis errors
class PromptAnalysisError(OptimizerException):
    """Prompt analysis failed."""
    pass

class ScoringError(PromptAnalysisError):
    """Scoring calculation failed."""
    pass

# Optimization errors
class OptimizationError(OptimizerException):
    """Optimization process failed."""
    pass

class InvalidStrategyError(OptimizationError):
    """Invalid strategy name."""
    pass

class OptimizationFailedError(OptimizationError):
    """Optimization execution failed."""
    pass

# Version errors
class VersionConflictError(OptimizerException):
    """Version conflict occurred."""
    pass

class VersionNotFoundError(OptimizerException):
    """Version not found."""
    pass
```

**Error Handling Principles:**

1. **Fail Fast:** Validate inputs early, raise exceptions immediately
2. **Contextual Messages:** Include relevant IDs, file paths, etc. in error messages
3. **Graceful Degradation:** Continue processing other prompts if one fails
4. **Logging:** Log all errors with context before raising/catching
5. **No Silent Failures:** Never catch and ignore exceptions without logging

**Example:**
```python
def extract_from_workflow(self, workflow_id: str) -> List[Prompt]:
    try:
        workflow = self.catalog.get_workflow(workflow_id)
        if not workflow:
            raise WorkflowNotFoundError(
                f"Workflow '{workflow_id}' not found in catalog"
            )

        # ... extraction logic

    except WorkflowNotFoundError:
        logger.error(f"Workflow not found: {workflow_id}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during extraction: {e}",
            extra={"workflow_id": workflow_id},
            exc_info=True
        )
        raise PromptExtractionError(
            f"Failed to extract prompts from workflow '{workflow_id}': {e}"
        ) from e
```

---

## 6. Constraints and Assumptions

### 6.1 Technical Constraints

**Constraint ID:** OPT-CON-001

1. **Python Version:** Python 3.9+
   - Reason: Type hint features (e.g., `list[T]`, `dict[K, V]`) require 3.9+

2. **Pydantic Version:** Pydantic V2 (>= 2.0.0)
   - Reason: Project standard, incompatible with V1 syntax

3. **No External LLM APIs in MVP:**
   - Reason: Avoid API costs, ensure deterministic testing
   - Post-MVP: Can add GPT-4 / Claude analysis

4. **In-Memory Storage Only (MVP):**
   - Reason: Simplicity, no database dependency
   - Post-MVP: Add SQLite or PostgreSQL persistence

5. **English Language Only (MVP):**
   - Reason: Readability algorithms (Flesch Reading Ease) are English-specific
   - Post-MVP: Add multi-language NLP libraries

6. **Linear Version History (MVP):**
   - Reason: Avoid complexity of branching/merging
   - Post-MVP: Add Git-like version graph

### 6.2 Design Assumptions

**Assumption ID:** OPT-ASM-001

1. **Workflow DSL Structure:**
   - ASSUME: All LLM nodes have `data.prompt_template` field
   - ASSUME: Prompt template follows `[{role, text}]` structure
   - MITIGATION: Handle missing fields gracefully, log warnings

2. **Prompt Size:**
   - ASSUME: Most prompts are < 2000 tokens
   - ASSUME: Maximum prompt size is 50,000 characters
   - MITIGATION: Add max_length validation

3. **Optimization Threshold:**
   - ASSUME: 5% improvement is meaningful
   - ASSUME: Confidence > 0.5 indicates reliable improvement
   - MITIGATION: Make thresholds configurable

4. **Version Frequency:**
   - ASSUME: Max 10-20 versions per prompt (in typical usage)
   - ASSUME: No need for version cleanup in MVP
   - MITIGATION: Add max_versions limit, warn at threshold

5. **Integration Patterns:**
   - ASSUME: Executor and collector are already implemented and stable
   - ASSUME: Config module models are stable (no breaking changes)
   - MITIGATION: Use explicit version pins in imports

6. **Performance Expectations:**
   - ASSUME: Users will optimize workflows with < 100 prompts
   - ASSUME: Optimization runs are manual, not real-time
   - MITIGATION: Log performance metrics, warn on large workflows

### 6.3 Out of Scope

**What Optimizer Module Will NOT Do (MVP):**

1. **Execute Tests:** Optimizer does NOT trigger workflow execution
2. **Manage Workflows:** Optimizer does NOT create/delete workflows
3. **Authentication:** Optimizer does NOT handle Dify API authentication
4. **Real-Time Updates:** Optimizer does NOT watch for DSL file changes
5. **Multi-User Coordination:** Optimizer does NOT handle concurrent edits
6. **Rollback Automation:** Optimizer does NOT auto-rollback failed optimizations
7. **A/B Testing:** Optimizer does NOT run statistical comparisons
8. **Custom Strategies:** Users cannot define custom optimization strategies (MVP)

---

## 7. Acceptance Criteria

### 7.1 Feature Acceptance

**Criteria ID:** OPT-AC-001

**Must-Have Features:**

| Feature | Acceptance Test | Status |
|---------|----------------|--------|
| **Prompt Extraction** | Extract all LLM prompts from 5-node workflow in < 2s | ☐ |
| **Variable Detection** | Detect `{{var}}`, `{var}`, `$var` patterns with 100% accuracy | ☐ |
| **Clarity Scoring** | Score 10 sample prompts, variance < 5 points from manual review | ☐ |
| **Efficiency Scoring** | Score based on token count and redundancy detection | ☐ |
| **Issue Detection** | Detect at least 5 common issue types with 80% precision | ☐ |
| **Suggestion Generation** | Generate at least 3 relevant suggestions per low-scoring prompt | ☐ |
| **Clarity Optimization** | Improve clarity score by 10+ points on verbose prompts | ☐ |
| **Efficiency Optimization** | Reduce token count by 20%+ on redundant prompts | ☐ |
| **Structure Optimization** | Add markdown structure to unformatted prompts | ☐ |
| **Version Creation** | Create baseline and optimized versions | ☐ |
| **Version Comparison** | Generate text diff and score delta | ☐ |
| **Full Optimization Cycle** | Extract → Analyze → Optimize → Version → Patch in < 30s for 20 prompts | ☐ |
| **CLI Integration** | Run `python main.py --mode optimize --workflow-id wf_001` successfully | ☐ |
| **PromptPatch Generation** | Generate valid patches that PromptPatchEngine can apply | ☐ |

**Validation Tests:**

```python
# Example acceptance test
def test_full_optimization_cycle_acceptance():
    """
    Acceptance Test: Full optimization cycle completes successfully.

    Given:
    - A workflow with 10 LLM nodes
    - Prompts with known quality issues

    When:
    - OptimizerService.run_optimization_cycle() is called

    Then:
    - At least 5 prompts are optimized
    - Generated patches are valid PromptPatch objects
    - Versions are created for optimized prompts
    - Process completes in < 30 seconds
    """
    workflow_id = "test_workflow_10_nodes"

    # Arrange
    optimizer = OptimizerService(catalog)

    # Act
    start_time = time.time()
    patches = optimizer.run_optimization_cycle(workflow_id)
    duration = time.time() - start_time

    # Assert
    assert len(patches) >= 5, "Should optimize at least 5 prompts"
    assert duration < 30.0, f"Should complete in < 30s, took {duration:.2f}s"

    for patch in patches:
        assert isinstance(patch, PromptPatch)
        assert patch.strategy.mode == "replace"
        assert len(patch.strategy.content) > 0

    # Verify versions were created
    report = optimizer.get_optimization_report(workflow_id)
    assert report.total_prompts >= 10
    assert report.optimized_prompts >= 5
```

---

### 7.2 Quality Metrics

**Criteria ID:** OPT-AC-002

**Code Quality Gates:**

| Metric | Target | Measurement | Status |
|--------|--------|-------------|--------|
| **Test Coverage** | ≥ 90% line coverage | `pytest --cov=src/optimizer --cov-report=term` | ☐ |
| **Branch Coverage** | ≥ 80% branch coverage | `pytest --cov-branch` | ☐ |
| **Complexity** | Max 10 per method | `radon cc src/optimizer -a` | ☐ |
| **Type Coverage** | 100% public methods | Manual review | ☐ |
| **Docstring Coverage** | 100% public classes/methods | `interrogate src/optimizer` | ☐ |
| **Linting** | 0 errors, < 10 warnings | `flake8 src/optimizer` | ☐ |
| **Type Checking** | 0 errors | `mypy src/optimizer` | ☐ |

**Quality Validation Commands:**

```bash
# Run all quality checks
pytest src/test/optimizer --cov=src/optimizer --cov-report=html --cov-report=term
flake8 src/optimizer --max-line-length=120
mypy src/optimizer --strict
radon cc src/optimizer -a --min B
interrogate src/optimizer -vv
```

---

### 7.3 Integration Acceptance

**Criteria ID:** OPT-AC-003

**Integration Tests:**

| Integration Point | Test Scenario | Expected Result | Status |
|-------------------|---------------|-----------------|--------|
| **Config Module** | Load WorkflowCatalog, extract prompts | Successful extraction | ☐ |
| **Config Module** | Generate PromptPatch, apply via PromptPatchEngine | DSL modified correctly | ☐ |
| **Executor Module** | Receive PerformanceMetrics, use in strategy selection | Strategy influenced by metrics | ☐ |
| **Collector Module** | Read aggregated statistics | Metrics loaded correctly | ☐ |
| **Logger Module** | Log optimization events | Logs at appropriate levels | ☐ |
| **CLI** | Run optimization via main.py | Patches generated and saved | ☐ |

**Integration Test Example:**

```python
def test_integration_with_prompt_patch_engine():
    """
    Integration Test: Generated patches can be applied via PromptPatchEngine.
    """
    # Arrange
    optimizer = OptimizerService(catalog)
    patch_engine = PromptPatchEngine(catalog, yaml_parser)

    original_dsl = load_dsl_file("workflows/wf_001.yml")

    # Act: Generate patches
    patches = optimizer.run_optimization_cycle("wf_001")

    # Act: Apply patches
    modified_dsl = patch_engine.apply_patches("wf_001", original_dsl, patches)

    # Assert: DSL was modified
    assert modified_dsl != original_dsl

    # Assert: Modified DSL is valid YAML
    modified_tree = yaml_parser.load(modified_dsl)
    assert modified_tree is not None

    # Assert: Optimized prompts are in DSL
    for patch in patches:
        node_path = patch.selector.by_path or _resolve_path(patch.selector.by_id)
        node = yaml_parser.get_node_by_path(modified_tree, node_path)
        prompt_text = yaml_parser.get_field_value(node, "data.prompt_template.0.text")

        assert patch.strategy.content in prompt_text
```

---

### 7.4 Documentation Acceptance

**Criteria ID:** OPT-AC-004

**Required Documentation:**

| Document | Content | Status |
|----------|---------|--------|
| **Module README** | Usage examples, API overview | ☐ |
| **API Documentation** | All public classes and methods | ☐ |
| **Configuration Guide** | optimizer.yaml structure and options | ☐ |
| **Integration Guide** | How to use with config/executor/collector | ☐ |
| **Testing Guide** | How to run tests, write new tests | ☐ |
| **SRS (this document)** | Complete requirements specification | ☑ |

**Documentation Quality Checks:**

```bash
# Generate API documentation
pydoc-markdown -p src.optimizer > docs/optimizer_api.md

# Check docstring coverage
interrogate src/optimizer -vv --fail-under 90
```

---

## 8. Appendix

### 8.1 Example Workflows

**Example 1: Complete Optimization Workflow**

```python
from src.config import ConfigLoader
from src.executor import ExecutorService
from src.collector import DataCollector
from src.optimizer import OptimizerService

# Step 1: Load configuration
loader = ConfigLoader()
env_config = loader.load_env_config("config/env.yaml")
catalog = loader.load_workflow_catalog("config/workflows.yaml")
test_plan = loader.load_test_plan("config/test_plan.yaml")

# Step 2: Run baseline tests
executor = ExecutorService()
manifest = test_plan.to_manifest()
baseline_results = executor.execute_test_plan(manifest)

# Step 3: Collect baseline metrics
collector = DataCollector()
for result in baseline_results:
    collector.collect_result(result)
baseline_metrics = collector.get_statistics(workflow_id="wf_001")

print(f"Baseline Success Rate: {baseline_metrics.success_rate:.2%}")

# Step 4: Run optimization
optimizer = OptimizerService(catalog)
patches = optimizer.run_optimization_cycle(
    workflow_id="wf_001",
    baseline_metrics=baseline_metrics,
    strategy="auto"
)

print(f"Generated {len(patches)} optimization patches")

# Step 5: Generate optimization report
report = optimizer.get_optimization_report("wf_001")
print(f"Total Prompts: {report.total_prompts}")
print(f"Optimized Prompts: {report.optimized_prompts}")
print(f"Average Improvement: {report.avg_improvement:.1f} points")

# Step 6: Apply patches to test plan
# (Update test_plan with patches, save to file)

# Step 7: Re-run tests with optimized prompts
# (Re-run executor, compare metrics)
```

---

### 8.2 Sample Data Structures

**Sample Prompt (Extracted):**

```python
Prompt(
    id="wf_customer_support_intent_classifier",
    workflow_id="wf_customer_support",
    node_id="intent_classifier",
    node_type="llm",
    text="""You are a customer support intent classifier.
Classify the user's message into one of these categories:
- product_inquiry
- technical_support
- billing_question
- complaint

User message: {{user_message}}

Respond with the category name only.""",
    role="system",
    variables=["user_message"],
    context={
        "label": "Intent Classification",
        "position": 0,
        "model_provider": "openai",
        "model_name": "gpt-4"
    },
    extracted_at=datetime(2025, 11, 17, 10, 30, 0)
)
```

**Sample PromptAnalysis:**

```python
PromptAnalysis(
    prompt_id="wf_customer_support_intent_classifier",
    clarity_score=75.0,
    efficiency_score=82.0,
    overall_score=77.8,  # 0.6 * 75 + 0.4 * 82
    issues=[
        PromptIssue(
            severity="info",
            category="clarity",
            message="Consider adding examples to clarify expected output format",
            location=None
        ),
        PromptIssue(
            severity="warning",
            category="structure",
            message="Add structure (bullet points, sections) for clarity",
            location=None
        )
    ],
    suggestions=[
        PromptSuggestion(
            category="clarity",
            priority=2,
            suggestion="Add examples to clarify expected output format",
            example="Example:\nInput: 'My product is broken'\nOutput: complaint"
        ),
        PromptSuggestion(
            category="structure",
            priority=2,
            suggestion="Use markdown formatting to organize content into sections",
            example="## Task\nClassify user messages...\n\n## Categories\n- product_inquiry\n..."
        )
    ],
    metrics={
        "token_count": 58,
        "character_count": 287,
        "sentence_count": 5,
        "avg_sentence_length": 11.4,
        "variable_count": 1,
        "readability_index": 72.3,
        "information_density": 0.68
    },
    analyzed_at=datetime(2025, 11, 17, 10, 31, 0)
)
```

**Sample OptimizationResult:**

```python
OptimizationResult(
    prompt_id="wf_customer_support_intent_classifier",
    original_prompt="You are a customer support intent classifier...",
    optimized_prompt="""# Customer Support Intent Classification

## Task
Classify the user's message into one of the following categories:

- **product_inquiry**: Questions about products
- **technical_support**: Technical issues or help requests
- **billing_question**: Payment or billing inquiries
- **complaint**: Complaints or negative feedback

## Input
User message: {{user_message}}

## Output Format
Respond with only the category name.

## Example
Input: "My product stopped working"
Output: technical_support""",
    strategy_used="structure_optimization",
    improvement_score=12.5,  # Overall score: 77.8 → 90.3
    original_analysis=PromptAnalysis(...),  # From above
    optimized_analysis=PromptAnalysis(
        prompt_id="wf_customer_support_intent_classifier",
        clarity_score=92.0,  # Improved from 75
        efficiency_score=80.0,  # Slightly decreased due to added structure
        overall_score=90.3,  # 0.6 * 92 + 0.4 * 80
        issues=[],  # All issues resolved
        suggestions=[],
        metrics={...},
        analyzed_at=datetime(2025, 11, 17, 10, 32, 0)
    ),
    confidence=0.75,
    optimized_at=datetime(2025, 11, 17, 10, 32, 0),
    metadata={
        "improvements": {
            "clarity": 17.0,
            "efficiency": -2.0,
            "structure": "added markdown headers and examples"
        }
    }
)
```

---

### 8.3 Test Case Examples

**Test Case 1: Prompt Extraction**

```python
def test_extract_prompts_from_workflow():
    """
    Test: Extract all prompts from a workflow with multiple LLM nodes.
    """
    # Arrange
    workflow_id = "wf_test_multi_llm"
    extractor = PromptExtractor(catalog)

    # Act
    prompts = extractor.extract_from_workflow(workflow_id)

    # Assert
    assert len(prompts) == 3, "Should extract 3 prompts"
    assert all(isinstance(p, Prompt) for p in prompts)
    assert all(p.workflow_id == workflow_id for p in prompts)
    assert all(len(p.text) > 0 for p in prompts)

    # Verify variables detected
    assert "user_message" in prompts[0].variables
```

**Test Case 2: Clarity Scoring**

```python
def test_clarity_scoring_verbose_prompt():
    """
    Test: Clarity score should be low for verbose, complex prompt.
    """
    # Arrange
    analyzer = PromptAnalyzer()
    verbose_prompt = Prompt(
        id="test_verbose",
        workflow_id="test",
        node_id="test",
        node_type="llm",
        text="""You are hereby instructed to perform a comprehensive analysis of the
aforementioned textual content, utilizing advanced natural language processing
methodologies in order to ascertain the underlying semantic structure and
subsequently generate a detailed summary that encapsulates the salient points,
taking into consideration the contextual nuances and maintaining fidelity to
the original author's intent.""",
        role="system",
        variables=[],
        context={},
        extracted_at=datetime.now()
    )

    # Act
    analysis = analyzer.analyze_prompt(verbose_prompt)

    # Assert
    assert analysis.clarity_score < 60, "Verbose prompt should have low clarity score"
    assert any(
        "complex" in issue.message.lower() or "simplify" in issue.message.lower()
        for issue in analysis.issues
    ), "Should detect complexity issues"
```

**Test Case 3: Optimization with Confidence Threshold**

```python
def test_optimization_respects_confidence_threshold():
    """
    Test: Optimization should be rejected if confidence is below threshold.
    """
    # Arrange
    config = OptimizationConfig(
        confidence_threshold=0.8  # High threshold
    )
    service = OptimizerService(catalog, config)

    # Act
    patches = service.run_optimization_cycle("wf_low_confidence")

    # Assert
    # Assuming optimizations have confidence < 0.8
    assert len(patches) == 0, "Should reject low-confidence optimizations"
```

---

### 8.4 Glossary

| Term | Definition |
|------|------------|
| **Baseline Version** | Initial version (1.0.0) of a prompt before any optimization |
| **Clarity Score** | Metric measuring prompt readability and instruction clarity (0-100) |
| **Confidence** | Probability that an optimization will improve prompt performance (0-1) |
| **DSL** | Domain Specific Language - YAML format used by Dify for workflows |
| **Efficiency Score** | Metric measuring token usage efficiency (0-100) |
| **Improvement Score** | Delta in overall_score after optimization (can be negative) |
| **Information Density** | Ratio of unique words to total words |
| **Issue** | Detected problem in prompt quality (vague, redundant, etc.) |
| **Node** | Single component in a workflow graph (LLM, tool, code, etc.) |
| **Overall Score** | Weighted average of clarity and efficiency scores |
| **Prompt** | Text instruction provided to an LLM node |
| **Prompt Patch** | Modification instruction for updating a prompt in DSL |
| **Strategy** | Optimization approach (clarity_focus, efficiency_focus, structure) |
| **Suggestion** | Actionable recommendation for improving prompt quality |
| **Token** | Smallest unit of text processed by LLM (approx. 4 characters) |
| **Variable** | Placeholder in prompt text (e.g., {{variable_name}}) |
| **Version** | Specific state of a prompt with associated metadata |

---

### 8.5 Requirements Traceability Matrix (RTM)

| Requirement ID | Component | Test Case | Status |
|----------------|-----------|-----------|--------|
| OPT-COMP-001 | PromptExtractor | test_extract_from_workflow | ☐ |
| OPT-COMP-001 | PromptExtractor | test_extract_from_node | ☐ |
| OPT-COMP-001 | PromptExtractor | test_variable_detection | ☐ |
| OPT-COMP-002 | PromptAnalyzer | test_calculate_clarity_score | ☐ |
| OPT-COMP-002 | PromptAnalyzer | test_calculate_efficiency_score | ☐ |
| OPT-COMP-002 | PromptAnalyzer | test_detect_issues | ☐ |
| OPT-COMP-002 | PromptAnalyzer | test_generate_suggestions | ☐ |
| OPT-COMP-003 | OptimizationEngine | test_apply_clarity_focus | ☐ |
| OPT-COMP-003 | OptimizationEngine | test_apply_efficiency_focus | ☐ |
| OPT-COMP-003 | OptimizationEngine | test_apply_structure_optimization | ☐ |
| OPT-COMP-003 | OptimizationEngine | test_confidence_calculation | ☐ |
| OPT-COMP-004 | VersionManager | test_create_version | ☐ |
| OPT-COMP-004 | VersionManager | test_get_version_history | ☐ |
| OPT-COMP-004 | VersionManager | test_compare_versions | ☐ |
| OPT-COMP-005 | OptimizerService | test_run_optimization_cycle | ☐ |
| OPT-COMP-005 | OptimizerService | test_get_optimization_report | ☐ |
| OPT-INT-001 | Config Integration | test_integration_with_workflow_catalog | ☐ |
| OPT-INT-001 | Config Integration | test_generate_valid_prompt_patches | ☐ |
| OPT-INT-002 | Executor Integration | test_use_performance_metrics | ☐ |
| OPT-INT-003 | Collector Integration | test_read_performance_metrics | ☐ |
| OPT-INT-004 | CLI Integration | test_cli_optimize_command | ☐ |
| OPT-NFR-001 | Performance | test_extraction_performance | ☐ |
| OPT-NFR-001 | Performance | test_optimization_cycle_performance | ☐ |
| OPT-NFR-002 | Testability | test_coverage_report | ☐ |
| OPT-NFR-003 | Maintainability | test_code_quality_metrics | ☐ |
| OPT-NFR-004 | Error Handling | test_exception_handling | ☐ |

---

## End of Document

**Document Version:** 1.0.0
**Total Pages:** 65
**Last Updated:** 2025-11-17

**Approval Signatures:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Requirements Analyst | [Name] | _________ | ____ |
| Backend Developer | [Name] | _________ | ____ |
| QA Engineer | [Name] | _________ | ____ |
| Project Manager | [Name] | _________ | ____ |

---

**Next Steps:**
1. Review SRS with stakeholders
2. Create GitHub issues for each component
3. Set up test infrastructure
4. Begin Phase 1 implementation (models.py + PromptExtractor)
