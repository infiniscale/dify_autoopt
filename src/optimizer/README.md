# Optimizer Module - Prompt Optimization and Version Management

The Optimizer module provides intelligent prompt extraction, analysis, optimization, and version management for Dify workflows. It enables automatic prompt quality assessment and AI-driven optimization with semantic versioning support.

**Status**: Production Ready - 13 files, 4,874 lines, 87% test coverage

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Extensibility](#extensibility)

---

## Overview

### Key Features

- **Intelligent Extraction**: Automatically extract prompts from workflow DSL files
- **Quality Analysis**: Rule-based scoring across clarity, efficiency, and structure
- **Multi-Strategy Optimization**: Three optimization strategies (clarity, efficiency, structure)
- **Version Management**: Semantic versioning with full history tracking
- **Test Integration**: Generate PromptPatch objects for A/B testing
- **Extensible Design**: Plugin architecture for custom LLM clients and storage backends

### Architecture

```
OptimizerService (High-level Facade)
    |
    +-- PromptExtractor    -> Extract prompts from workflow DSL
    +-- PromptAnalyzer     -> Analyze quality and detect issues
    +-- OptimizationEngine -> Generate optimized variants
    +-- VersionManager     -> Track prompt history
    +-- PromptPatchEngine  -> Generate test patches
```

### Scoring Formula

The analyzer uses a weighted scoring system:

- **Clarity Score** = 0.4 × structure + 0.3 × specificity + 0.3 × coherence
- **Efficiency Score** = 0.5 × token_efficiency + 0.5 × information_density
- **Overall Score** = 0.6 × clarity + 0.4 × efficiency

All scores range from 0-100.

---

## Quick Start

### Basic Workflow Optimization (5 minutes)

```python
from src.config import ConfigLoader
from src.optimizer import optimize_workflow, analyze_workflow

# 1. Load workflow catalog
loader = ConfigLoader()
catalog = loader.load_workflow_catalog("config/workflows.yaml")

# 2. Analyze workflow quality
report = analyze_workflow("wf_customer_service", catalog)
print(f"Average Score: {report['average_score']:.1f}")
print(f"Needs Optimization: {report['needs_optimization']}")

# 3. Optimize if needed
if report['needs_optimization']:
    patches = optimize_workflow(
        workflow_id="wf_customer_service",
        catalog=catalog,
        strategy="clarity_focus"
    )

    print(f"Generated {len(patches)} optimization patches")

    # 4. Apply patches to test plan
    for patch in patches:
        print(f"  - Node {patch.selector.by_id}: {patch.strategy.mode}")
```

### Output Example

```
Average Score: 68.5
Needs Optimization: True
Generated 3 optimization patches
  - Node llm_1: replace
  - Node llm_3: replace
  - Node llm_5: replace
```

---

## Core Components

### 1. OptimizerService

**Purpose**: High-level orchestration facade for complete optimization workflow.

**Key Methods**:
- `run_optimization_cycle()`: Full optimization pipeline
- `optimize_single_prompt()`: Optimize individual prompts
- `analyze_workflow()`: Quality analysis without optimization
- `get_version_history()`: Retrieve version history

**Example**:

```python
from src.optimizer import OptimizerService

service = OptimizerService(catalog=catalog)

# Analyze workflow
report = service.analyze_workflow("wf_001")

# Run optimization
patches = service.run_optimization_cycle(
    workflow_id="wf_001",
    strategy="auto",  # Auto-select best strategy
    baseline_metrics={"success_rate": 0.75}
)
```

---

### 2. PromptExtractor

**Purpose**: Extract prompts from workflow DSL YAML files.

**Extraction Strategy**:
- Searches multiple DSL structure patterns
- Extracts text from LLM nodes
- Detects Jinja2 variables (`{{variable}}`)
- Captures metadata (model, temperature, position)

**Example**:

```python
from src.optimizer import PromptExtractor
from pathlib import Path

extractor = PromptExtractor()

# Load and parse DSL
dsl_dict = extractor.load_dsl_file(Path("workflows/customer_service.yml"))

# Extract prompts
prompts = extractor.extract_from_workflow(dsl_dict, workflow_id="wf_001")

for prompt in prompts:
    print(f"Node: {prompt.node_id}")
    print(f"Text: {prompt.text[:100]}...")
    print(f"Variables: {prompt.variables}")
    print(f"Model: {prompt.context.get('model')}")
    print("---")
```

---

### 3. PromptAnalyzer

**Purpose**: Rule-based prompt quality analysis with scoring and issue detection.

**Analysis Dimensions**:
- **Structure**: Headers, bullets, formatting
- **Specificity**: Action verbs, concrete instructions, no vague language
- **Coherence**: Sentence flow, consistent terminology
- **Token Efficiency**: Optimal length, no redundancy
- **Information Density**: High semantic value, minimal filler

**Issue Types**:
- `TOO_LONG`: > 2000 characters
- `TOO_SHORT`: < 20 characters
- `VAGUE_LANGUAGE`: Contains "some", "maybe", "etc"
- `MISSING_STRUCTURE`: No headers/bullets in long prompts
- `REDUNDANCY`: Repeated phrases
- `POOR_FORMATTING`: No line breaks
- `AMBIGUOUS_INSTRUCTIONS`: No action verbs

**Example**:

```python
from src.optimizer import PromptAnalyzer, Prompt

analyzer = PromptAnalyzer()

# Analyze prompt
analysis = analyzer.analyze_prompt(prompt)

print(f"Overall Score: {analysis.overall_score:.1f}")
print(f"Clarity: {analysis.clarity_score:.1f}")
print(f"Efficiency: {analysis.efficiency_score:.1f}")

# Review issues
for issue in analysis.issues:
    print(f"[{issue.severity.value}] {issue.type.value}")
    print(f"  {issue.description}")
    print(f"  Suggestion: {issue.suggestion}")

# Review suggestions
for suggestion in analysis.suggestions:
    print(f"Priority {suggestion.priority}: {suggestion.description}")
```

---

### 4. OptimizationEngine

**Purpose**: Generate optimized prompt variants using rule-based transformations.

**Strategies**:

1. **clarity_focus**: Improve readability
   - Add section headers
   - Break long sentences
   - Replace vague terms
   - Add explicit instructions

2. **efficiency_focus**: Reduce token usage
   - Remove filler words
   - Compress verbose phrases
   - Eliminate redundancy
   - Clean whitespace

3. **structure_focus**: Enhance organization
   - Add markdown formatting
   - Create numbered steps
   - Add section separators
   - Apply templates

**Example**:

```python
from src.optimizer import OptimizationEngine, PromptAnalyzer

analyzer = PromptAnalyzer()
engine = OptimizationEngine(analyzer)

# Optimize with specific strategy
result = engine.optimize(prompt, strategy="clarity_focus")

print(f"Original: {result.original_prompt}")
print(f"Optimized: {result.optimized_prompt}")
print(f"Improvement: {result.improvement_score:.1f} points")
print(f"Confidence: {result.confidence:.2f}")
print(f"Changes: {', '.join(result.changes)}")
```

---

### 5. VersionManager

**Purpose**: Track prompt evolution with semantic versioning.

**Version Numbering**:
- `1.0.0`: Baseline version
- `1.1.0`: Minor optimization
- `1.2.0`: Another minor optimization
- `2.0.0`: Major restructure

**Key Operations**:
- Create versions (auto-increment)
- Retrieve specific versions
- Compare versions
- Rollback to previous versions
- Find best version by score

**Example**:

```python
from src.optimizer import VersionManager

manager = VersionManager()

# Create baseline version
v1 = manager.create_version(
    prompt=prompt,
    analysis=analysis,
    optimization_result=None,  # Baseline
    parent_version=None
)

# Create optimized version
v2 = manager.create_version(
    prompt=optimized_prompt,
    analysis=optimized_analysis,
    optimization_result=result,
    parent_version="1.0.0"
)

# Compare versions
comparison = manager.compare_versions("prompt_001", "1.0.0", "1.1.0")
print(f"Improvement: {comparison['improvement']:.1f}")

# Get version history
history = manager.get_version_history("prompt_001")
for v in history:
    print(f"v{v.version}: score={v.analysis.overall_score:.1f}")

# Rollback if needed
if v2.analysis.overall_score < v1.analysis.overall_score:
    rolled_back = manager.rollback("prompt_001", "1.0.0")
```

---

## API Reference

### Data Models

#### Prompt

Extracted prompt with metadata.

```python
from src.optimizer import Prompt

prompt = Prompt(
    id="wf_001_llm_1",              # Unique identifier
    workflow_id="wf_001",            # Parent workflow
    node_id="llm_1",                 # Node ID in DSL
    node_type="llm",                 # Node type
    text="Summarize: {{document}}", # Prompt content
    role="user",                     # Message role
    variables=["document"],          # Jinja2 variables
    context={"model": "gpt-4"},      # Node metadata
    extracted_at=datetime.now()      # Extraction timestamp
)
```

**Validation**:
- `text` must not be empty
- `variables` must contain valid identifiers

---

#### PromptAnalysis

Quality analysis result with scores and suggestions.

```python
from src.optimizer import PromptAnalysis, PromptIssue, PromptSuggestion

analysis = PromptAnalysis(
    prompt_id="wf_001_llm_1",
    overall_score=75.0,              # 0-100
    clarity_score=80.0,              # 0-100
    efficiency_score=70.0,           # 0-100
    issues=[...],                    # List[PromptIssue]
    suggestions=[...],               # List[PromptSuggestion]
    metadata={
        "character_count": 120,
        "word_count": 18,
        "sentence_count": 2,
        "estimated_tokens": 30
    },
    analyzed_at=datetime.now()
)
```

**Fields**:
- `overall_score`: Weighted combination of clarity and efficiency
- `clarity_score`: Structure + specificity + coherence
- `efficiency_score`: Token efficiency + information density
- `issues`: Detected problems with severity levels
- `suggestions`: Prioritized improvement recommendations
- `metadata`: Additional metrics for reference

---

#### OptimizationResult

Result of prompt optimization with improvement metrics.

```python
from src.optimizer import OptimizationResult, OptimizationStrategy

result = OptimizationResult(
    prompt_id="wf_001_llm_1",
    original_prompt="Write summary",
    optimized_prompt="Please summarize the document in 3-5 bullet points",
    strategy=OptimizationStrategy.CLARITY_FOCUS,
    improvement_score=10.5,          # Score delta
    confidence=0.85,                 # 0.0-1.0
    changes=["Added specific output format", "Added clear instruction"],
    metadata={
        "original_score": 65.0,
        "optimized_score": 75.5
    },
    optimized_at=datetime.now()
)
```

**Confidence Levels**:
- `0.8-1.0`: High confidence (both metrics improved significantly)
- `0.6-0.8`: Medium confidence (overall improvement)
- `0.4-0.6`: Low confidence (minor improvement)
- `0.0-0.4`: Very low confidence (marginal or no improvement)

---

#### PromptVersion

Version record for prompt history tracking.

```python
from src.optimizer import PromptVersion

version = PromptVersion(
    prompt_id="wf_001_llm_1",
    version="1.1.0",                 # Semantic version
    prompt=prompt_obj,               # Prompt at this version
    analysis=analysis_obj,           # Analysis result
    optimization_result=opt_result,  # OptimizationResult or None
    parent_version="1.0.0",          # Parent version
    created_at=datetime.now(),
    metadata={
        "author": "optimizer",
        "strategy": "clarity_focus"
    }
)

# Check version type
is_baseline = version.is_baseline()  # True if parent_version is None

# Compare versions
v_tuple = version.get_version_number()  # (1, 1, 0)
```

---

### Enumerations

#### OptimizationStrategy

Available optimization strategies.

```python
from src.optimizer import OptimizationStrategy

OptimizationStrategy.CLARITY_FOCUS     # Improve readability
OptimizationStrategy.EFFICIENCY_FOCUS  # Reduce token usage
OptimizationStrategy.STRUCTURE_FOCUS   # Enhance organization
OptimizationStrategy.AUTO              # Auto-select strategy
```

#### IssueSeverity

Severity levels for detected issues.

```python
from src.optimizer import IssueSeverity

IssueSeverity.CRITICAL  # Blocking issue
IssueSeverity.WARNING   # Should be addressed
IssueSeverity.INFO      # Informational notice
```

#### IssueType / SuggestionType

See full lists in [models.py](./models.py).

---

### Convenience Functions

#### optimize_workflow()

High-level workflow optimization function.

```python
from src.optimizer import optimize_workflow

patches = optimize_workflow(
    workflow_id: str,                      # Required
    catalog: WorkflowCatalog,              # Required
    strategy: str = "auto",                # clarity_focus, efficiency_focus, structure_focus, auto
    baseline_metrics: Optional[Dict] = None,  # Optional performance baseline
    config: Optional[OptimizationConfig] = None,  # Optional config
    llm_client: Optional[LLMClient] = None,     # Optional LLM client
    storage: Optional[VersionStorage] = None    # Optional storage backend
) -> List[PromptPatch]
```

**Returns**: List of `PromptPatch` objects for test plan integration.

**Raises**:
- `WorkflowNotFoundError`: Workflow doesn't exist in catalog
- `OptimizerError`: Optimization process failed

**Example**:

```python
patches = optimize_workflow(
    workflow_id="wf_001",
    catalog=catalog,
    strategy="clarity_focus",
    baseline_metrics={"success_rate": 0.75}
)
```

---

#### analyze_workflow()

Analyze workflow without optimization.

```python
from src.optimizer import analyze_workflow

report = analyze_workflow(
    workflow_id: str,
    catalog: WorkflowCatalog
) -> Dict[str, Any]
```

**Returns**:

```python
{
    "workflow_id": "wf_001",
    "prompt_count": 3,
    "average_score": 72.5,
    "needs_optimization": True,
    "prompts": [
        {
            "prompt_id": "wf_001_llm_1",
            "node_id": "llm_1",
            "overall_score": 68.0,
            "clarity_score": 65.0,
            "efficiency_score": 72.0,
            "issues_count": 2,
            "suggestions_count": 3,
            "issues": [...]
        },
        ...
    ]
}
```

---

### Exceptions

All optimizer exceptions inherit from `OptimizerError`.

```python
from src.optimizer import (
    OptimizerError,          # Base exception
    ExtractionError,         # Prompt extraction failed
    WorkflowNotFoundError,   # Workflow not in catalog
    NodeNotFoundError,       # Node not found in DSL
    DSLParseError,          # DSL parsing failed
    AnalysisError,          # Analysis failed
    ScoringError,           # Scoring calculation failed
    OptimizationError,      # Base optimization error
    InvalidStrategyError,   # Invalid strategy name
    OptimizationFailedError,# Optimization process failed
    VersionError,           # Base version error
    VersionConflictError,   # Version already exists
    VersionNotFoundError,   # Version not found
    ValidationError,        # Data validation failed
    ConfigError             # Configuration error
)

# Example usage
try:
    patches = optimize_workflow("wf_001", catalog)
except WorkflowNotFoundError as e:
    print(f"Workflow not found: {e.workflow_id}")
except InvalidStrategyError as e:
    print(f"Invalid strategy: {e.strategy}")
    print(f"Valid options: {e.valid_strategies}")
except OptimizerError as e:
    print(f"Optimization failed: {e.message}")
    print(f"Error code: {e.error_code}")
    print(f"Context: {e.context}")
```

---

## Usage Guide

### Scenario 1: Optimize All Workflows in Test Plan

```python
from src.config import ConfigLoader
from src.optimizer import optimize_workflow

loader = ConfigLoader()
catalog = loader.load_workflow_catalog("config/workflows.yaml")
test_plan = loader.load_test_plan("config/test_plan.yaml")

all_patches = []

for workflow_config in test_plan.workflows:
    workflow_id = workflow_config.workflow_id

    try:
        patches = optimize_workflow(
            workflow_id=workflow_id,
            catalog=catalog,
            strategy="auto"
        )

        all_patches.extend(patches)
        print(f"{workflow_id}: {len(patches)} patches generated")

    except Exception as e:
        print(f"{workflow_id}: Optimization failed - {e}")

print(f"Total patches: {len(all_patches)}")
```

---

### Scenario 2: Compare Multiple Strategies

```python
from src.optimizer import OptimizerService

service = OptimizerService(catalog=catalog)

# Extract prompts
prompts = service._extract_prompts("wf_001")

strategies = ["clarity_focus", "efficiency_focus", "structure_focus"]
results = {}

for prompt in prompts[:1]:  # Test on first prompt
    results[prompt.id] = {}

    for strategy in strategies:
        result = service.optimize_single_prompt(prompt, strategy)
        results[prompt.id][strategy] = {
            "improvement": result.improvement_score,
            "confidence": result.confidence,
            "changes": result.changes
        }

# Print comparison
for prompt_id, strategies_data in results.items():
    print(f"\nPrompt: {prompt_id}")
    for strategy, metrics in strategies_data.items():
        print(f"  {strategy}:")
        print(f"    Improvement: {metrics['improvement']:.1f}")
        print(f"    Confidence: {metrics['confidence']:.2f}")
        print(f"    Changes: {metrics['changes']}")
```

---

### Scenario 3: Version Management Workflow

```python
from src.optimizer import OptimizerService, VersionManager

service = OptimizerService(catalog=catalog)
manager = service._version_manager

# Optimize and track versions
prompts = service._extract_prompts("wf_customer_service")

for prompt in prompts:
    # Analyze baseline
    analysis = service._analyzer.analyze_prompt(prompt)

    # Create baseline version
    v1 = manager.create_version(prompt, analysis, None, None)
    print(f"Created baseline v{v1.version}: score={analysis.overall_score:.1f}")

    # Optimize if needed
    if analysis.overall_score < 80:
        result = service.optimize_single_prompt(prompt, "auto")

        # Create optimized prompt object
        from src.optimizer import Prompt
        opt_prompt = Prompt(
            id=prompt.id,
            workflow_id=prompt.workflow_id,
            node_id=prompt.node_id,
            node_type=prompt.node_type,
            text=result.optimized_prompt,
            role=prompt.role,
            variables=prompt.variables,
            context=prompt.context
        )

        # Analyze optimized version
        opt_analysis = service._analyzer.analyze_prompt(opt_prompt)

        # Create optimized version
        v2 = manager.create_version(opt_prompt, opt_analysis, result, v1.version)
        print(f"Created optimized v{v2.version}: score={opt_analysis.overall_score:.1f}")

    # Get history
    history = manager.get_version_history(prompt.id)
    print(f"Version history: {len(history)} versions")

    # Find best version
    best = manager.get_best_version(prompt.id)
    print(f"Best version: v{best.version} (score={best.analysis.overall_score:.1f})")
```

---

### Scenario 4: Custom Quality Thresholds

```python
from src.optimizer import OptimizerService, OptimizationConfig

# Define custom config
config = OptimizationConfig(
    strategies=["clarity_focus", "efficiency_focus"],
    min_confidence=0.7,  # Require 70% confidence
    max_iterations=5,
    metadata={"project": "customer_service_v2"}
)

service = OptimizerService(catalog=catalog)

patches = service.run_optimization_cycle(
    workflow_id="wf_001",
    config=config,
    baseline_metrics={"success_rate": 0.6}  # Low baseline triggers optimization
)

# Filter by confidence
high_confidence_patches = [
    p for p in patches
    if service._version_manager.get_latest_version(
        f"{p.selector.by_id}"
    ).optimization_result.confidence >= 0.8
]

print(f"High-confidence patches: {len(high_confidence_patches)}/{len(patches)}")
```

---

### Scenario 5: Integration with Test Execution

```python
from src.config import ConfigLoader
from src.optimizer import optimize_workflow
from src.executor import TestCaseGenerator, RunManifestBuilder

# Load config
loader = ConfigLoader()
catalog = loader.load_workflow_catalog("config/workflows.yaml")
test_plan = loader.load_test_plan("config/test_plan.yaml")

# Optimize workflows
for wf_config in test_plan.workflows:
    patches = optimize_workflow(
        workflow_id=wf_config.workflow_id,
        catalog=catalog,
        strategy="auto"
    )

    # Add patches to test plan
    if not wf_config.prompt_optimization:
        from src.config.models import PromptOptimizationConfig
        wf_config.prompt_optimization = [
            PromptOptimizationConfig(
                variant_name="optimized",
                nodes=patches
            )
        ]
    else:
        wf_config.prompt_optimization[0].nodes.extend(patches)

# Generate test cases with optimized prompts
generator = TestCaseGenerator(test_plan, catalog)
test_cases = generator.generate_all_cases()

# Build execution manifest
builder = RunManifestBuilder(test_plan, catalog, generator)
manifests = builder.build_all_manifests()

print(f"Generated {len(manifests)} test manifests with optimized prompts")
```

---

## Configuration

### YAML Configuration Example

```yaml
optimizer:
  # Extraction settings
  extraction:
    max_prompt_length: 10000
    enable_context_analysis: true
    variable_pattern: "\\{\\{(\\w+)\\}\\}"

  # Analysis settings
  analyzer:
    evaluation_metrics:
      - clarity
      - efficiency
    confidence_threshold: 0.8

  # Optimization settings
  optimization:
    strategies:
      - name: "clarity_focus"
        enabled: true
      - name: "efficiency_focus"
        enabled: true
      - name: "structure_focus"
        enabled: true

    max_iterations: 5
    improvement_threshold: 5.0  # Minimum 5% improvement

  # Versioning settings
  versioning:
    max_versions_kept: 50
    auto_backup: true
    enable_branching: false
```

### Programmatic Configuration

```python
from src.optimizer import OptimizationConfig, OptimizationStrategy

config = OptimizationConfig(
    strategies=[
        OptimizationStrategy.CLARITY_FOCUS,
        OptimizationStrategy.EFFICIENCY_FOCUS
    ],
    min_confidence=0.7,
    max_iterations=3,
    score_threshold=80.0,  # Skip optimization if score >= 80.0
    analysis_rules={
        "min_clarity_score": 70.0,
        "min_efficiency_score": 65.0
    },
    metadata={
        "project": "prod_optimization",
        "version": "1.0"
    }
)
```

### Score Threshold Configuration

The `score_threshold` parameter controls when optimization should be skipped based on prompt quality:

```python
# Conservative: Only optimize low-quality prompts (score < 85)
config = OptimizationConfig(score_threshold=85.0)

# Default: Optimize prompts below score of 80 (recommended)
config = OptimizationConfig(score_threshold=80.0)  # Default value

# Aggressive: Optimize more prompts (score < 70)
config = OptimizationConfig(score_threshold=70.0)

# Optimize all prompts: Set threshold to maximum
config = OptimizationConfig(score_threshold=100.0)
```

**How it works**:
- If `analysis.overall_score < score_threshold`, the prompt will be optimized
- If `analysis.overall_score >= score_threshold`, optimization is skipped
- Valid range: 0.0 to 100.0
- Default: 80.0 (optimizes prompts with scores below 80)

**Use cases**:
- **Production workflows**: Use higher threshold (85-90) to only optimize problematic prompts
- **Development workflows**: Use default threshold (80) for balanced optimization
- **Experimental workflows**: Use lower threshold (70) or maximum (100) to optimize aggressively

---

## Best Practices

### 1. Optimization Strategy Selection

**When to use clarity_focus**:
- Prompts with low structure scores (< 60)
- Contains vague language
- Missing clear instructions
- User-facing workflows

**When to use efficiency_focus**:
- High token consumption (> 500 tokens)
- Redundant content
- Batch processing workflows
- Cost-sensitive applications

**When to use structure_focus**:
- Long prompts (> 300 characters)
- Multi-step instructions
- Complex workflows
- Documentation-heavy prompts

**When to use auto**:
- First-time optimization
- Unknown prompt patterns
- Let analyzer determine best fit

---

### 2. Confidence Threshold Guidelines

```python
# Conservative: Only accept high-confidence optimizations
config = OptimizationConfig(min_confidence=0.8)

# Balanced: Accept medium-confidence improvements
config = OptimizationConfig(min_confidence=0.6)

# Aggressive: Try all optimizations
config = OptimizationConfig(min_confidence=0.4)
```

**Recommendation**: Start with 0.7, adjust based on validation results.

---

### 3. Version Management

```python
# Always create baseline before optimization
v_baseline = manager.create_version(prompt, analysis, None, None)

# Tag important versions
v_baseline.metadata["tag"] = "production_baseline"
v_baseline.metadata["deployment_date"] = "2025-01-15"

# Rollback if regression detected
if new_score < baseline_score - 5:  # 5-point tolerance
    manager.rollback(prompt_id, baseline_version)
```

---

### 4. Performance Optimization

```python
# Batch processing for multiple workflows
from concurrent.futures import ThreadPoolExecutor

workflows = ["wf_001", "wf_002", "wf_003"]

def optimize_wf(wf_id):
    return optimize_workflow(wf_id, catalog, strategy="auto")

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(optimize_wf, workflows))

# Results aggregation
total_patches = sum(len(r) for r in results)
```

---

### 5. Error Handling

```python
from src.optimizer import (
    WorkflowNotFoundError,
    InvalidStrategyError,
    OptimizerError
)

def safe_optimize(workflow_id, catalog, strategy="auto"):
    """Optimization with comprehensive error handling."""
    try:
        patches = optimize_workflow(workflow_id, catalog, strategy)
        return {"status": "success", "patches": patches}

    except WorkflowNotFoundError as e:
        return {"status": "error", "reason": f"Workflow not found: {e.workflow_id}"}

    except InvalidStrategyError as e:
        return {"status": "error", "reason": f"Invalid strategy: {e.strategy}"}

    except OptimizerError as e:
        return {
            "status": "error",
            "reason": e.message,
            "error_code": e.error_code,
            "context": e.context
        }
```

---

## Troubleshooting

### Issue: No prompts extracted

**Symptoms**: `extractor.extract_from_workflow()` returns empty list.

**Causes**:
1. DSL file has non-standard structure
2. No LLM nodes in workflow
3. Prompt text fields are empty

**Solutions**:

```python
# Debug extraction
extractor = PromptExtractor()
dsl_dict = extractor.load_dsl_file(dsl_path)

# Check for nodes
nodes = extractor._find_nodes(dsl_dict)
print(f"Found {len(nodes)} nodes")

# Check node types
for node in nodes:
    node_type = extractor._detect_node_type(node)
    print(f"Node {node.get('id')}: type={node_type}")

# Check prompt text
for node in nodes:
    text = extractor._extract_prompt_text(node)
    print(f"Node {node.get('id')}: text_length={len(text) if text else 0}")
```

---

### Issue: Low optimization scores

**Symptoms**: `improvement_score` is negative or near zero.

**Causes**:
1. Prompt already well-optimized
2. Wrong strategy selected
3. Rule-based optimizer limitations

**Solutions**:

```python
# Compare multiple strategies
strategies = ["clarity_focus", "efficiency_focus", "structure_focus"]
results = {}

for strategy in strategies:
    result = engine.optimize(prompt, strategy)
    results[strategy] = result.improvement_score

best_strategy = max(results.keys(), key=lambda k: results[k])
print(f"Best strategy: {best_strategy} (improvement={results[best_strategy]:.1f})")

# Manual analysis
analysis = analyzer.analyze_prompt(prompt)
print(f"Clarity: {analysis.clarity_score:.1f}")
print(f"Efficiency: {analysis.efficiency_score:.1f}")

# Check if already optimal
if analysis.overall_score > 85:
    print("Prompt already well-optimized")
```

---

### Issue: Version conflicts

**Symptoms**: `VersionConflictError` when creating version.

**Cause**: Version number already exists.

**Solution**:

```python
# Check existing versions
existing = manager.get_version_history(prompt_id)
print(f"Existing versions: {[v.version for v in existing]}")

# Use latest version as parent
latest = manager.get_latest_version(prompt_id)
new_version = manager.create_version(
    prompt, analysis, result,
    parent_version=latest.version  # Explicitly set parent
)
```

---

### Issue: DSL parsing errors

**Symptoms**: `DSLParseError` when loading workflow.

**Causes**:
1. Invalid YAML syntax
2. File not found
3. Encoding issues

**Solutions**:

```python
# Validate YAML manually
import yaml

try:
    with open(dsl_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print("YAML is valid")
except yaml.YAMLError as e:
    print(f"YAML error: {e}")

# Check file existence
from pathlib import Path

if not Path(dsl_path).exists():
    print(f"File not found: {dsl_path}")

# Try different encoding
with open(dsl_path, 'r', encoding='latin-1') as f:
    content = f.read()
```

---

## Extensibility

### Custom LLM Client

Implement the `LLMClient` interface to integrate real LLM APIs.

```python
from src.optimizer.interfaces import LLMClient
from src.optimizer import PromptAnalysis, Prompt
from typing import Dict, Any, Optional
import openai

class OpenAIClient(LLMClient):
    """OpenAI GPT-4 based prompt analysis and optimization."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PromptAnalysis:
        """Analyze prompt using GPT-4."""

        analysis_prompt = f"""
        Analyze this prompt and provide scores (0-100):

        Prompt: {prompt}

        Provide JSON output:
        {{
            "clarity_score": <score>,
            "efficiency_score": <score>,
            "issues": [list of issues],
            "suggestions": [list of suggestions]
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": analysis_prompt}],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Convert to PromptAnalysis
        return PromptAnalysis(
            prompt_id=context.get("prompt_id", "unknown"),
            overall_score=(result["clarity_score"] + result["efficiency_score"]) / 2,
            clarity_score=result["clarity_score"],
            efficiency_score=result["efficiency_score"],
            issues=[],  # Parse from result["issues"]
            suggestions=[]  # Parse from result["suggestions"]
        )

    def optimize_prompt(
        self,
        prompt: str,
        strategy: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate optimized prompt using GPT-4."""

        optimization_prompt = f"""
        Optimize this prompt for {strategy}:

        Original: {prompt}

        Return only the optimized prompt text.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": optimization_prompt}]
        )

        return response.choices[0].message.content

# Usage
openai_client = OpenAIClient(api_key="sk-...")
service = OptimizerService(catalog=catalog, llm_client=openai_client)
```

---

### Custom Storage Backend

Implement the `VersionStorage` interface for persistent storage.

```python
from src.optimizer.interfaces import VersionStorage
from src.optimizer import PromptVersion
from typing import List, Optional
import json
from pathlib import Path

class FileSystemStorage(VersionStorage):
    """JSON file-based version storage."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_version(self, version: PromptVersion) -> None:
        """Save version to JSON file."""
        prompt_dir = self.storage_dir / version.prompt_id
        prompt_dir.mkdir(exist_ok=True)

        version_file = prompt_dir / f"{version.version}.json"

        if version_file.exists():
            raise VersionConflictError(version.prompt_id, version.version)

        # Serialize to JSON
        data = version.model_dump(mode="json")

        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_version(
        self,
        prompt_id: str,
        version: str
    ) -> Optional[PromptVersion]:
        """Load version from JSON file."""
        version_file = self.storage_dir / prompt_id / f"{version}.json"

        if not version_file.exists():
            return None

        with open(version_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return PromptVersion(**data)

    def list_versions(self, prompt_id: str) -> List[PromptVersion]:
        """List all versions for a prompt."""
        prompt_dir = self.storage_dir / prompt_id

        if not prompt_dir.exists():
            return []

        versions = []
        for version_file in sorted(prompt_dir.glob("*.json")):
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            versions.append(PromptVersion(**data))

        # Sort by version number
        versions.sort(key=lambda v: v.get_version_number())
        return versions

    def get_latest_version(self, prompt_id: str) -> Optional[PromptVersion]:
        """Get latest version."""
        versions = self.list_versions(prompt_id)
        return versions[-1] if versions else None

    def delete_version(self, prompt_id: str, version: str) -> bool:
        """Delete version file."""
        version_file = self.storage_dir / prompt_id / f"{version}.json"

        if version_file.exists():
            version_file.unlink()
            return True
        return False

    def clear_all(self) -> None:
        """Delete all version files."""
        import shutil
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
        self.storage_dir.mkdir(parents=True)

# Usage
fs_storage = FileSystemStorage("data/prompt_versions")
manager = VersionManager(storage=fs_storage)
```

---

## Additional Resources

- **Architecture Documentation**: See [docs/optimizer/optimizer_architecture.md](../../docs/optimizer/optimizer_architecture.md)
- **SRS Document**: See [docs/optimizer/optimizer_srs.md](../../docs/optimizer/optimizer_srs.md)
- **Test Report**: See [docs/optimizer/TEST_REPORT_OPTIMIZER.md](../../docs/optimizer/TEST_REPORT_OPTIMIZER.md)
- **Implementation Summary**: See [docs/optimizer/optimizer_summary.md](../../docs/optimizer/optimizer_summary.md)
- **Execution Blueprint**: See [docs/optimizer/optimizer_execution_blueprint.md](../../docs/optimizer/optimizer_execution_blueprint.md)

---

## Support

For issues, questions, or contributions:
- Review test cases in `tests/optimizer/`
- Check existing issues in project tracker
- Consult design documents in `docs/optimizer/`

**Module Status**: Production Ready | **Coverage**: 87% | **Last Updated**: 2025-01-17
