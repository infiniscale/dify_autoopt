# Optimizer Module - API Quick Reference

**Version**: 1.0.0 | **Last Updated**: 2025-01-17

---

## Quick Start (30 seconds)

```python
from src.config import ConfigLoader
from src.optimizer import optimize_workflow, analyze_workflow

loader = ConfigLoader()
catalog = loader.load_workflow_catalog("config/workflows.yaml")

# Analyze
report = analyze_workflow("wf_001", catalog)
print(f"Score: {report['average_score']:.1f}")

# Optimize
patches = optimize_workflow("wf_001", catalog, strategy="auto")
print(f"Patches: {len(patches)}")
```

---

## Core API Signatures

### Convenience Functions

```python
# High-level workflow optimization
optimize_workflow(
    workflow_id: str,
    catalog: WorkflowCatalog,
    strategy: str = "auto",  # auto|clarity_focus|efficiency_focus|structure_focus
    baseline_metrics: Optional[Dict] = None,
    config: Optional[OptimizationConfig] = None,
    llm_client: Optional[LLMClient] = None,
    storage: Optional[VersionStorage] = None
) -> List[PromptPatch]

# Workflow analysis without optimization
analyze_workflow(
    workflow_id: str,
    catalog: WorkflowCatalog
) -> Dict[str, Any]  # Returns: {workflow_id, prompt_count, average_score, needs_optimization, prompts}
```

---

### OptimizerService

```python
from src.optimizer import OptimizerService

service = OptimizerService(
    catalog: Optional[WorkflowCatalog] = None,
    llm_client: Optional[LLMClient] = None,
    storage: Optional[VersionStorage] = None,
    custom_logger: Optional[Any] = None
)

# Main optimization cycle
service.run_optimization_cycle(
    workflow_id: str,
    baseline_metrics: Optional[Dict] = None,
    strategy: str = "auto",
    config: Optional[OptimizationConfig] = None
) -> List[PromptPatch]

# Optimize single prompt
service.optimize_single_prompt(
    prompt: Prompt,
    strategy: str = "auto"
) -> OptimizationResult

# Analyze workflow
service.analyze_workflow(
    workflow_id: str
) -> Dict[str, Any]

# Get version history
service.get_version_history(
    prompt_id: str
) -> List[Dict[str, Any]]
```

---

### PromptExtractor

```python
from src.optimizer import PromptExtractor

extractor = PromptExtractor(custom_logger: Optional[Any] = None)

# Load DSL file
dsl_dict = extractor.load_dsl_file(
    dsl_path: Path
) -> Dict[str, Any]  # Raises: DSLParseError

# Extract from workflow
prompts = extractor.extract_from_workflow(
    workflow_dict: Dict[str, Any],
    workflow_id: Optional[str] = None
) -> List[Prompt]  # Raises: ExtractionError

# Extract from single node
prompt = extractor.extract_from_node(
    node: Dict[str, Any],
    workflow_id: str
) -> Optional[Prompt]
```

---

### PromptAnalyzer

```python
from src.optimizer import PromptAnalyzer

analyzer = PromptAnalyzer(
    llm_client: Optional[LLMClient] = None,
    custom_logger: Optional[Any] = None
)

# Analyze prompt quality
analysis = analyzer.analyze_prompt(
    prompt: Prompt
) -> PromptAnalysis  # Raises: AnalysisError

# Analysis result contains:
# - overall_score: float (0-100)
# - clarity_score: float (0-100)
# - efficiency_score: float (0-100)
# - issues: List[PromptIssue]
# - suggestions: List[PromptSuggestion]
# - metadata: Dict[str, Any]
```

---

### OptimizationEngine

```python
from src.optimizer import OptimizationEngine, PromptAnalyzer

analyzer = PromptAnalyzer()
engine = OptimizationEngine(
    analyzer: PromptAnalyzer,
    llm_client: Optional[LLMClient] = None,
    custom_logger: Optional[Any] = None
)

# Optimize prompt
result = engine.optimize(
    prompt: Prompt,
    strategy: str,  # clarity_focus|efficiency_focus|structure_focus
    config: Optional[OptimizationConfig] = None
) -> OptimizationResult  # Raises: InvalidStrategyError, OptimizationFailedError

# Public transformation methods (used by StubLLMClient)
engine.apply_clarity_focus(text: str) -> str
engine.apply_efficiency_focus(text: str) -> str
engine.apply_structure_optimization(text: str) -> str
```

---

### VersionManager

```python
from src.optimizer import VersionManager

manager = VersionManager(
    storage: Optional[VersionStorage] = None,
    custom_logger: Optional[Any] = None
)

# Create version
version = manager.create_version(
    prompt: Prompt,
    analysis: PromptAnalysis,
    optimization_result: Optional[OptimizationResult] = None,
    parent_version: Optional[str] = None
) -> PromptVersion  # Raises: VersionConflictError

# Get specific version
version = manager.get_version(
    prompt_id: str,
    version: str
) -> PromptVersion  # Raises: VersionNotFoundError

# Get latest version
version = manager.get_latest_version(
    prompt_id: str
) -> PromptVersion  # Raises: VersionNotFoundError

# Get version history
history = manager.get_version_history(
    prompt_id: str
) -> List[PromptVersion]

# Compare versions
comparison = manager.compare_versions(
    prompt_id: str,
    version1: str,
    version2: str
) -> Dict[str, Any]  # Raises: VersionNotFoundError

# Rollback
new_version = manager.rollback(
    prompt_id: str,
    target_version: str
) -> PromptVersion  # Raises: VersionNotFoundError

# Get best version
best = manager.get_best_version(
    prompt_id: str
) -> PromptVersion  # Raises: VersionNotFoundError

# Delete version
deleted = manager.delete_version(
    prompt_id: str,
    version: str
) -> bool

# Clear all
manager.clear_all_versions() -> None
```

---

## Data Models

### Prompt

```python
from src.optimizer import Prompt
from datetime import datetime

prompt = Prompt(
    id: str,                       # Unique identifier (required)
    workflow_id: str,              # Parent workflow (required)
    node_id: str,                  # Node ID in DSL (required)
    node_type: str,                # Node type (required)
    text: str,                     # Prompt content (required, non-empty)
    role: str = "system",          # Message role
    variables: List[str] = [],     # Jinja2 variables
    context: Dict[str, Any] = {},  # Node metadata
    extracted_at: datetime = datetime.now()
)

# Validators
# - text must not be empty
# - variables must contain valid identifiers
```

---

### PromptAnalysis

```python
from src.optimizer import PromptAnalysis, PromptIssue, PromptSuggestion

analysis = PromptAnalysis(
    prompt_id: str,                         # Required
    overall_score: float,                   # 0-100 (required)
    clarity_score: float,                   # 0-100 (required)
    efficiency_score: float,                # 0-100 (required)
    issues: List[PromptIssue] = [],
    suggestions: List[PromptSuggestion] = [],
    metadata: Dict[str, Any] = {},
    analyzed_at: datetime = datetime.now()
)

# Metadata typically contains:
# - character_count: int
# - word_count: int
# - sentence_count: int
# - estimated_tokens: float
# - avg_word_length: float
# - avg_sentence_length: float
# - variable_count: int
```

---

### PromptIssue

```python
from src.optimizer import PromptIssue, IssueSeverity, IssueType

issue = PromptIssue(
    severity: IssueSeverity,  # CRITICAL|WARNING|INFO (required)
    type: IssueType,          # Required (see enums below)
    description: str,         # Required
    location: Optional[str] = None,
    suggestion: Optional[str] = None
)
```

---

### PromptSuggestion

```python
from src.optimizer import PromptSuggestion, SuggestionType

suggestion = PromptSuggestion(
    type: SuggestionType,  # Required (see enums below)
    description: str,      # Required
    priority: int          # 1-10 (required)
)
```

---

### OptimizationResult

```python
from src.optimizer import OptimizationResult, OptimizationStrategy

result = OptimizationResult(
    prompt_id: str,                      # Required
    original_prompt: str,                # Required
    optimized_prompt: str,               # Required
    strategy: OptimizationStrategy,      # Required
    improvement_score: float,            # Score delta (can be negative)
    confidence: float,                   # 0.0-1.0 (required)
    changes: List[str] = [],
    metadata: Dict[str, Any] = {},
    optimized_at: datetime = datetime.now()
)

# Metadata typically contains:
# - original_score: float
# - optimized_score: float
# - original_clarity: float
# - optimized_clarity: float
# - original_efficiency: float
# - optimized_efficiency: float
```

---

### PromptVersion

```python
from src.optimizer import PromptVersion

version = PromptVersion(
    prompt_id: str,                                # Required
    version: str,                                  # Semantic version (required)
    prompt: Prompt,                                # Required
    analysis: PromptAnalysis,                      # Required
    optimization_result: Optional[OptimizationResult] = None,
    parent_version: Optional[str] = None,
    created_at: datetime = datetime.now(),
    metadata: Dict[str, Any] = {}
)

# Methods
version.get_version_number() -> tuple  # (major, minor, patch)
version.is_baseline() -> bool          # True if parent_version is None

# Version format validator: "X.Y.Z" where X, Y, Z are non-negative integers
```

---

### OptimizationConfig

```python
from src.optimizer import OptimizationConfig, OptimizationStrategy

config = OptimizationConfig(
    strategies: List[OptimizationStrategy] = [OptimizationStrategy.AUTO],
    min_confidence: float = 0.6,            # 0.0-1.0
    max_iterations: int = 3,                # >= 1
    analysis_rules: Dict[str, Any] = {},
    metadata: Dict[str, Any] = {}
)
```

---

## Enumerations

### OptimizationStrategy

```python
from src.optimizer import OptimizationStrategy

OptimizationStrategy.CLARITY_FOCUS     = "clarity_focus"
OptimizationStrategy.EFFICIENCY_FOCUS  = "efficiency_focus"
OptimizationStrategy.STRUCTURE_FOCUS   = "structure_focus"
OptimizationStrategy.AUTO              = "auto"
```

---

### IssueSeverity

```python
from src.optimizer import IssueSeverity

IssueSeverity.CRITICAL = "critical"  # Severe issue blocking optimization
IssueSeverity.WARNING  = "warning"   # Issue should be addressed
IssueSeverity.INFO     = "info"      # Informational notice
```

---

### IssueType

```python
from src.optimizer import IssueType

IssueType.TOO_LONG              = "too_long"
IssueType.TOO_SHORT             = "too_short"
IssueType.VAGUE_LANGUAGE        = "vague_language"
IssueType.MISSING_STRUCTURE     = "missing_structure"
IssueType.REDUNDANCY            = "redundancy"
IssueType.POOR_FORMATTING       = "poor_formatting"
IssueType.AMBIGUOUS_INSTRUCTIONS = "ambiguous_instructions"
```

---

### SuggestionType

```python
from src.optimizer import SuggestionType

SuggestionType.ADD_STRUCTURE       = "add_structure"
SuggestionType.CLARIFY_INSTRUCTIONS = "clarify_instructions"
SuggestionType.REDUCE_LENGTH       = "reduce_length"
SuggestionType.ADD_EXAMPLES        = "add_examples"
SuggestionType.IMPROVE_FORMATTING  = "improve_formatting"
```

---

## Exceptions

```python
from src.optimizer import (
    # Base
    OptimizerError,

    # Extraction
    ExtractionError,
    WorkflowNotFoundError,
    NodeNotFoundError,
    DSLParseError,

    # Analysis
    AnalysisError,
    ScoringError,

    # Optimization
    OptimizationError,
    InvalidStrategyError,
    OptimizationFailedError,

    # Versioning
    VersionError,
    VersionConflictError,
    VersionNotFoundError,

    # Configuration
    ValidationError,
    ConfigError
)

# All exceptions inherit from OptimizerError
# All have: message, error_code, context attributes

# Exception attributes
exception.message: str       # Human-readable error message
exception.error_code: str    # Machine-readable error code
exception.context: Dict      # Additional error context
```

---

## Interfaces

### LLMClient

```python
from src.optimizer.interfaces import LLMClient
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def analyze_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PromptAnalysis:
        pass

    @abstractmethod
    def optimize_prompt(
        self,
        prompt: str,
        strategy: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        pass

# Built-in: StubLLMClient (rule-based, no API calls)
```

---

### VersionStorage

```python
from src.optimizer.interfaces import VersionStorage
from abc import ABC, abstractmethod

class VersionStorage(ABC):
    @abstractmethod
    def save_version(self, version: PromptVersion) -> None:
        pass

    @abstractmethod
    def get_version(self, prompt_id: str, version: str) -> Optional[PromptVersion]:
        pass

    @abstractmethod
    def list_versions(self, prompt_id: str) -> List[PromptVersion]:
        pass

    @abstractmethod
    def get_latest_version(self, prompt_id: str) -> Optional[PromptVersion]:
        pass

    @abstractmethod
    def delete_version(self, prompt_id: str, version: str) -> bool:
        pass

    @abstractmethod
    def clear_all(self) -> None:
        pass

# Built-in: InMemoryStorage (dict-based, not persisted)
```

---

## Common Code Snippets

### Snippet 1: Basic Optimization

```python
from src.config import ConfigLoader
from src.optimizer import optimize_workflow

loader = ConfigLoader()
catalog = loader.load_workflow_catalog("config/workflows.yaml")

patches = optimize_workflow("wf_001", catalog, strategy="auto")
for patch in patches:
    print(f"Node: {patch.selector.by_id}")
```

---

### Snippet 2: Custom Strategy Selection

```python
from src.optimizer import OptimizerService

service = OptimizerService(catalog=catalog)
prompts = service._extract_prompts("wf_001")

for prompt in prompts:
    analysis = service._analyzer.analyze_prompt(prompt)

    # Choose strategy based on scores
    if analysis.clarity_score < 70:
        strategy = "clarity_focus"
    elif analysis.efficiency_score < 70:
        strategy = "efficiency_focus"
    else:
        strategy = "structure_focus"

    result = service.optimize_single_prompt(prompt, strategy)
    print(f"{prompt.node_id}: {strategy} → +{result.improvement_score:.1f}")
```

---

### Snippet 3: Version Comparison

```python
from src.optimizer import VersionManager

manager = VersionManager()

# Create baseline
v1 = manager.create_version(prompt, analysis, None, None)

# Create optimized
v2 = manager.create_version(opt_prompt, opt_analysis, result, v1.version)

# Compare
comparison = manager.compare_versions(prompt.id, v1.version, v2.version)
print(f"Improvement: +{comparison['improvement']:.1f}")
print(f"Changes: {', '.join(comparison['changes'])}")
```

---

### Snippet 4: Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

workflows = ["wf_001", "wf_002", "wf_003"]

def optimize(wf_id):
    return optimize_workflow(wf_id, catalog, strategy="auto")

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(optimize, workflows))

total_patches = sum(len(r) for r in results)
print(f"Total patches: {total_patches}")
```

---

### Snippet 5: Custom LLM Client

```python
from src.optimizer.interfaces import LLMClient

class MyLLMClient(LLMClient):
    def analyze_prompt(self, prompt, context=None):
        # Call your LLM API
        response = my_llm_api.analyze(prompt)
        return PromptAnalysis(...)

    def optimize_prompt(self, prompt, strategy, context=None):
        # Call your LLM API
        response = my_llm_api.optimize(prompt, strategy)
        return response.text

# Use
service = OptimizerService(catalog=catalog, llm_client=MyLLMClient())
```

---

### Snippet 6: Error Handling

```python
from src.optimizer import WorkflowNotFoundError, InvalidStrategyError, OptimizerError

try:
    patches = optimize_workflow("wf_001", catalog, strategy="custom")
except WorkflowNotFoundError as e:
    print(f"Workflow not found: {e.workflow_id}")
except InvalidStrategyError as e:
    print(f"Invalid strategy: {e.strategy}")
    print(f"Valid options: {e.valid_strategies}")
except OptimizerError as e:
    print(f"Error: {e.message} (code: {e.error_code})")
    print(f"Context: {e.context}")
```

---

### Snippet 7: Export to DataFrame

```python
import pandas as pd

service = OptimizerService(catalog=catalog)
prompts = service._extract_prompts("wf_001")

data = []
for prompt in prompts:
    analysis = service._analyzer.analyze_prompt(prompt)
    data.append({
        "node_id": prompt.node_id,
        "overall_score": analysis.overall_score,
        "clarity_score": analysis.clarity_score,
        "efficiency_score": analysis.efficiency_score,
        "issues_count": len(analysis.issues),
        "character_count": analysis.metadata["character_count"]
    })

df = pd.DataFrame(data)
print(df.to_string(index=False))
```

---

### Snippet 8: Integration with Test Plan

```python
from src.config import ConfigLoader
from src.optimizer import optimize_workflow
from src.config.models import PromptOptimizationConfig

loader = ConfigLoader()
catalog = loader.load_workflow_catalog("config/workflows.yaml")
test_plan = loader.load_test_plan("config/test_plan.yaml")

for wf_config in test_plan.workflows:
    patches = optimize_workflow(wf_config.workflow_id, catalog)

    # Add to test plan
    opt_config = PromptOptimizationConfig(
        variant_name="optimized",
        nodes=patches
    )

    if not wf_config.prompt_optimization:
        wf_config.prompt_optimization = [opt_config]
    else:
        wf_config.prompt_optimization.append(opt_config)
```

---

## Parameter Reference Table

### Strategy Selection

| Strategy | Use When | Effect | Token Impact |
|----------|----------|--------|--------------|
| `clarity_focus` | Low clarity score (<70) | Add structure, replace vague terms | +10-20% |
| `efficiency_focus` | High token count (>500) | Remove filler, compress phrases | -20-30% |
| `structure_focus` | Missing headers/bullets | Add formatting, numbered steps | +5-15% |
| `auto` | Unknown prompt patterns | Auto-select based on analysis | Varies |

---

### Confidence Thresholds

| Threshold | Description | Recommended For |
|-----------|-------------|-----------------|
| 0.8-1.0 | High confidence | Production deployment |
| 0.6-0.8 | Medium confidence | Testing/validation |
| 0.4-0.6 | Low confidence | Experimental optimization |
| 0.0-0.4 | Very low confidence | Review manually |

---

### Score Ranges

| Score Range | Quality Level | Action Required |
|-------------|---------------|-----------------|
| 85-100 | Excellent | No optimization needed |
| 70-84 | Good | Optional optimization |
| 50-69 | Fair | Optimization recommended |
| 0-49 | Poor | Optimization required |

---

### Issue Severity Impact

| Severity | Count Threshold | Optimization Triggered |
|----------|----------------|------------------------|
| CRITICAL | ≥ 1 | Always |
| WARNING | ≥ 3 | If score < 80 |
| INFO | ≥ 5 | If score < 70 |

---

## Testing Quick Reference

```python
# Run optimizer tests
pytest tests/optimizer/ -v

# Run with coverage
pytest tests/optimizer/ --cov=src/optimizer --cov-report=html

# Run specific test file
pytest tests/optimizer/test_service.py -v

# Run specific test
pytest tests/optimizer/test_service.py::test_run_optimization_cycle -v

# Debug mode
pytest tests/optimizer/ -v -s
```

---

## Performance Benchmarks

| Operation | Avg Time | Throughput |
|-----------|----------|------------|
| Extract prompt | 5ms | 200/sec |
| Analyze prompt | 10ms | 100/sec |
| Optimize prompt | 15ms | 66/sec |
| Create version | 5ms | 200/sec |
| Full cycle (10 prompts) | 300ms | 3.3 cycles/sec |

*Benchmarks on AMD Ryzen 9 5900X, Python 3.10*

---

## Documentation Links

- **Full README**: [src/optimizer/README.md](../../src/optimizer/README.md)
- **Usage Guide**: [docs/optimizer/optimizer_usage_guide.md](./optimizer_usage_guide.md)
- **Architecture**: [docs/optimizer/optimizer_architecture.md](./optimizer_architecture.md)
- **SRS**: [docs/optimizer/optimizer_srs.md](./optimizer_srs.md)
- **Test Report**: [docs/optimizer/TEST_REPORT_OPTIMIZER.md](./TEST_REPORT_OPTIMIZER.md)

---

**Quick Reference Version**: 1.0.0
**Module Version**: 0.1.0
**Last Updated**: 2025-01-17
