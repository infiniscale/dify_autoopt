# Optimizer Module Execution Blueprint

**Project:** dify_autoopt
**Module:** src/optimizer
**Version:** MVP 1.0
**Date:** 2025-11-17
**Author:** Project Manager Agent

---

## 1. Executive Summary

This blueprint defines the execution plan for implementing the **optimizer module** in the dify_autoopt project. The optimizer module is responsible for intelligent prompt extraction, analysis, optimization, and version management within Dify workflows.

### 1.1 Current Status

**Completed Modules** (High Test Coverage):
- `src/config`: YAML configuration system with Pydantic V2 validation
- `src/executor`: Task execution and scheduling (100% coverage)
- `src/collector`: Result collection and Excel export
- `src/utils/logger`: Loguru-based logging system

**Optimizer Module Status**:
- Existing: `PromptPatchEngine` for DSL prompt modification
- Missing: Core optimization logic, analysis, and version management

### 1.2 Objectives

1. **Complete the optimization feedback loop**: Config → Executor → Collector → Optimizer → Config
2. **Deliver MVP functionality**: Core prompt extraction, analysis, optimization, and version management
3. **Ensure testability**: All components testable with stubs/mocks
4. **Maintain consistency**: Follow existing patterns (naming, error handling, logging)

---

## 2. Architecture and Data Flow

### 2.1 Overall Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Optimization Closed Loop                         │
└─────────────────────────────────────────────────────────────────────────┘

   ┌──────────────┐
   │   Config     │  (EnvConfig, WorkflowCatalog, TestPlan)
   │   Module     │
   └──────┬───────┘
          │ 1. Load test plan with prompt patches
          ▼
   ┌──────────────┐
   │   Executor   │  Executes test cases (original + variants)
   │   Module     │  - PairwiseEngine, TaskScheduler
   └──────┬───────┘  - ConcurrentExecutor, StubExecutor
          │ 2. Generate TaskResults
          ▼
   ┌──────────────┐
   │  Collector   │  Collects execution results
   │   Module     │  - DataCollector, ResultClassifier
   └──────┬───────┘  - ExcelExporter
          │ 3. Generate TestResults + PerformanceMetrics
          ▼
   ┌──────────────┐
   │  Optimizer   │  Analyzes and optimizes prompts
   │   Module     │  ┌─────────────────────────────┐
   └──────┬───────┘  │ A. PromptExtractor          │
          │          │    - Extract prompts from   │
          │          │      workflow DSL           │
          │          │                             │
          │          │ B. PromptAnalyzer           │
          │          │    - Score quality metrics  │
          │          │    - Identify issues        │
          │          │                             │
          │          │ C. OptimizationEngine       │
          │          │    - Generate candidates    │
          │          │    - Apply strategies       │
          │          │    - Compare with baseline  │
          │          │                             │
          │          │ D. VersionManager           │
          │          │    - Track versions         │
          │          │    - Store history          │
          │          └─────────────────────────────┘
          │ 4. Generate OptimizationResult + PromptPatch
          ▼
   ┌──────────────┐
   │ Test Plan    │  Update test plan with new patches
   │   Update     │  - New prompt variants
   └──────────────┘  - Performance baselines

          │ 5. Next iteration (loop back to Executor)
          ▼
     (Continuous improvement cycle)
```

### 2.2 Detailed Component Interaction

#### Phase 1: Initialization (Config Module)
```python
# User provides initial test plan
test_plan = TestPlan(
    workflows=[
        WorkflowPlanEntry(
            workflow_id="wf_001",
            datasets=[Dataset(...)],
            prompt_patches=[],  # Initially empty
        )
    ]
)
```

#### Phase 2: Baseline Execution (Executor + Collector)
```python
# ExecutorService runs baseline tests
service = ExecutorService()
test_results = service.execute_test_plan(run_manifest)

# DataCollector computes metrics
collector = DataCollector()
for result in test_results:
    collector.collect_result(result)
baseline_metrics = collector.get_statistics()
```

#### Phase 3: Optimization (Optimizer Module - NEW)
```python
# Step 1: Extract prompts from workflow DSL
extractor = PromptExtractor(workflow_catalog)
prompts = extractor.extract_from_workflow("wf_001")

# Step 2: Analyze prompt quality
analyzer = PromptAnalyzer()
analysis = analyzer.analyze_prompt(prompts[0])
# Returns: clarity_score, efficiency_score, issues, suggestions

# Step 3: Generate optimized variants
optimizer = OptimizationEngine(analyzer)
optimization_result = optimizer.optimize(
    prompt=prompts[0],
    baseline_metrics=baseline_metrics,
    strategy="clarity_focus"  # or "efficiency_focus", "multi_objective"
)

# Step 4: Version tracking
version_mgr = VersionManager()
version = version_mgr.create_version(
    prompt_id=prompts[0].id,
    original=prompts[0].text,
    optimized=optimization_result.optimized_prompt,
    metrics=optimization_result.metrics
)
```

#### Phase 4: Feedback Loop (Back to Config)
```python
# Generate PromptPatch for next iteration
patch = PromptPatch(
    selector=PromptSelector(by_id=prompts[0].node_id),
    strategy=PromptStrategy(
        mode="replace",
        content=optimization_result.optimized_prompt
    )
)

# Update test plan
test_plan.workflows[0].prompt_patches.append(patch)

# Re-run tests with optimized prompts
# Compare results with baseline
```

### 2.3 Trigger Points for Optimization

1. **Manual Trigger**: User explicitly calls optimization API
   ```python
   optimizer_service.run_optimization(workflow_id="wf_001")
   ```

2. **Threshold-Based**: After N test runs, if success rate < threshold
   ```python
   if metrics.success_rate < 0.8 and run_count >= 10:
       trigger_optimization()
   ```

3. **Scheduled**: Periodic optimization (future feature)

---

## 3. MVP Feature List

### 3.1 Core Components (Must Implement)

| Component | File | Purpose | Acceptance Criteria |
|-----------|------|---------|---------------------|
| **PromptExtractor** | `prompt_extractor.py` | Extract prompts from workflow DSL | - Extract all LLM node prompts<br>- Identify variables and context<br>- Return structured Prompt objects<br>- Handle missing/malformed DSL |
| **PromptAnalyzer** | `prompt_analyzer.py` | Analyze prompt quality | - Calculate clarity score (0-100)<br>- Calculate efficiency score<br>- Detect common issues<br>- Generate improvement suggestions |
| **OptimizationEngine** | `optimization_engine.py` | Generate optimized prompts | - Implement 2-3 basic strategies<br>- Generate optimization candidates<br>- Compare with baseline<br>- Return top candidate |
| **VersionManager** | `version_manager.py` | Track prompt versions | - Create version records<br>- Store version history<br>- Query version by ID<br>- Compare two versions |
| **OptimizerService** | `optimizer_service.py` | Orchestration facade | - Coordinate all components<br>- Provide simple API<br>- Handle errors gracefully<br>- Integrate with logger |
| **Models** | `models.py` | Data structures | - Prompt, PromptAnalysis<br>- OptimizationResult, PromptVersion<br>- OptimizationConfig |

### 3.2 MVP Feature Details

#### 3.2.1 PromptExtractor

**Functionality:**
- Parse workflow DSL YAML to find LLM nodes
- Extract prompt text from `llm.prompt_template.messages` field
- Identify variable placeholders (e.g., `{{variable_name}}`)
- Build node context (type, label, dependencies)

**Input:**
```python
workflow_catalog: WorkflowCatalog  # From config module
workflow_id: str
```

**Output:**
```python
List[Prompt]  # Structured prompt objects with metadata
```

**Limitations (MVP):**
- Only extract from LLM nodes (not code nodes, knowledge retrieval, etc.)
- Basic variable detection (simple regex patterns)
- No multi-language prompt detection

---

#### 3.2.2 PromptAnalyzer

**Functionality:**
- **Clarity Score**: Based on readability metrics (sentence length, jargon density)
- **Efficiency Score**: Based on token count vs. information density
- **Issue Detection**: Common problems (ambiguous instructions, missing context)
- **Suggestions**: Template-based improvement hints

**Input:**
```python
prompt: Prompt
```

**Output:**
```python
PromptAnalysis(
    clarity_score: float,  # 0-100
    efficiency_score: float,  # 0-100
    overall_score: float,  # Weighted average
    issues: List[str],  # ["Issue 1", "Issue 2"]
    suggestions: List[str]  # ["Suggestion 1", "Suggestion 2"]
)
```

**Limitations (MVP):**
- Rule-based scoring (no LLM-based evaluation)
- Fixed scoring weights
- English-only analysis
- Basic heuristics (token count, sentence complexity)

---

#### 3.2.3 OptimizationEngine

**Functionality:**
- **Strategy 1: Clarity Focus** - Simplify language, break down complex sentences
- **Strategy 2: Efficiency Focus** - Reduce token count, remove redundancy
- **Strategy 3: Structure Optimization** - Add markdown formatting, use bullet points

**Algorithm (MVP):**
```python
def optimize(prompt: Prompt, strategy: str) -> OptimizationResult:
    # 1. Analyze current prompt
    analysis = analyzer.analyze_prompt(prompt)

    # 2. Apply optimization strategy
    if strategy == "clarity_focus":
        optimized = apply_clarity_rules(prompt.text)
    elif strategy == "efficiency_focus":
        optimized = apply_compression_rules(prompt.text)
    elif strategy == "structure":
        optimized = apply_structure_rules(prompt.text)

    # 3. Re-analyze optimized version
    optimized_analysis = analyzer.analyze_prompt(
        Prompt(text=optimized, ...)
    )

    # 4. Calculate improvement
    improvement = optimized_analysis.overall_score - analysis.overall_score

    return OptimizationResult(
        original_prompt=prompt.text,
        optimized_prompt=optimized,
        strategy_used=strategy,
        improvement_score=improvement,
        original_analysis=analysis,
        optimized_analysis=optimized_analysis
    )
```

**Limitations (MVP):**
- Template-based transformations (no LLM generation)
- Fixed rules per strategy
- Single optimization pass (no iteration)
- No A/B testing integration

---

#### 3.2.4 VersionManager

**Functionality:**
- Store prompt versions in memory (dict-based)
- Generate semantic version numbers (v1.0.0, v1.1.0, etc.)
- Track change history (original → optimized)
- Compare two versions (diff + metrics)

**Storage (MVP):**
```python
{
    "prompt_id": {
        "wf_001_llm_node_1": {
            "versions": [
                PromptVersion(
                    version="1.0.0",
                    text="Original prompt...",
                    created_at=datetime(...),
                    metrics=PromptAnalysis(...),
                    author="baseline"
                ),
                PromptVersion(
                    version="1.1.0",
                    text="Optimized prompt...",
                    created_at=datetime(...),
                    metrics=PromptAnalysis(...),
                    author="optimizer",
                    parent_version="1.0.0"
                )
            ],
            "current_version": "1.1.0"
        }
    }
}
```

**Limitations (MVP):**
- In-memory storage only (no persistence)
- No branching/merging
- Simple linear version history
- No conflict resolution

---

#### 3.2.5 OptimizerService (Facade)

**Purpose:** High-level API for external integration

**API:**
```python
class OptimizerService:
    def run_optimization_cycle(
        self,
        workflow_id: str,
        baseline_metrics: PerformanceMetrics,
        strategy: str = "auto"
    ) -> List[PromptPatch]:
        """
        Complete optimization cycle for a workflow

        Returns:
            List of PromptPatch objects to apply in next test iteration
        """
        # 1. Extract prompts
        # 2. Analyze each prompt
        # 3. Optimize low-scoring prompts
        # 4. Create versions
        # 5. Generate patches
        pass

    def get_optimization_report(
        self,
        workflow_id: str
    ) -> OptimizationReport:
        """Generate summary report of optimization results"""
        pass
```

---

### 3.3 Data Models (models.py)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

@dataclass
class Prompt:
    """Extracted prompt with metadata"""
    id: str  # Unique identifier (workflow_id + node_id)
    workflow_id: str
    node_id: str
    node_type: str  # "llm", "code", etc.
    text: str  # Actual prompt content
    role: str  # "system", "user", "assistant"
    variables: List[str]  # ["var1", "var2"]
    context: Dict[str, Any]  # Node metadata
    extracted_at: datetime

@dataclass
class PromptAnalysis:
    """Analysis results for a prompt"""
    prompt_id: str
    clarity_score: float  # 0-100
    efficiency_score: float  # 0-100
    overall_score: float  # Weighted average
    issues: List[str]  # List of detected problems
    suggestions: List[str]  # Improvement recommendations
    metrics: Dict[str, float]  # Raw metrics (token_count, avg_sentence_length, etc.)
    analyzed_at: datetime

@dataclass
class OptimizationResult:
    """Result of optimization process"""
    prompt_id: str
    original_prompt: str
    optimized_prompt: str
    strategy_used: str
    improvement_score: float  # Delta in overall_score
    original_analysis: PromptAnalysis
    optimized_analysis: PromptAnalysis
    confidence: float  # 0-1, confidence in improvement
    optimized_at: datetime

@dataclass
class PromptVersion:
    """Version record for a prompt"""
    prompt_id: str
    version: str  # Semantic version (1.0.0, 1.1.0, etc.)
    text: str
    metrics: PromptAnalysis
    created_at: datetime
    author: str  # "baseline", "optimizer", "manual"
    parent_version: Optional[str] = None  # Parent version number
    change_summary: Optional[str] = None  # Description of changes

@dataclass
class OptimizationConfig:
    """Configuration for optimization engine"""
    default_strategy: str = "clarity_focus"
    improvement_threshold: float = 5.0  # Minimum improvement to accept (%)
    max_optimization_iterations: int = 3
    enable_version_tracking: bool = True
    scoring_weights: Dict[str, float] = None  # {"clarity": 0.6, "efficiency": 0.4}

    def __post_init__(self):
        if self.scoring_weights is None:
            self.scoring_weights = {"clarity": 0.6, "efficiency": 0.4}
```

---

### 3.4 Testing Strategy

#### Test Coverage Targets
- Minimum: 80% line coverage
- Target: 90% line coverage (matching executor/collector modules)

#### Test Structure
```
src/test/optimizer/
├── conftest.py              # Shared fixtures
├── test_prompt_extractor.py # Extraction logic tests
├── test_prompt_analyzer.py  # Analysis scoring tests
├── test_optimization_engine.py  # Optimization strategies
├── test_version_manager.py  # Version management
├── test_optimizer_service.py  # Integration tests
└── fixtures/
    ├── sample_workflow_dsl.yaml
    ├── sample_prompts.yaml
    └── expected_results.yaml
```

#### Key Test Scenarios

**PromptExtractor Tests:**
- Extract prompts from valid workflow DSL
- Handle missing LLM nodes gracefully
- Extract variables correctly
- Handle malformed YAML

**PromptAnalyzer Tests:**
- Score clarity for various prompt types
- Detect common issues (vague instructions, missing examples)
- Generate relevant suggestions
- Handle edge cases (empty prompts, very long prompts)

**OptimizationEngine Tests:**
- Apply clarity optimization strategy
- Apply efficiency optimization strategy
- Verify improvement scores are calculated correctly
- Handle prompts that can't be improved

**VersionManager Tests:**
- Create first version for a prompt
- Create subsequent versions
- Retrieve version history
- Compare two versions

**OptimizerService Tests:**
- End-to-end optimization cycle
- Integration with config module
- Error handling and logging

---

### 3.5 Error Handling

Following existing project patterns (`src/utils/exceptions.py`):

```python
# New exceptions in src/optimizer/exceptions.py
class OptimizerException(Exception):
    """Base exception for optimizer module"""
    pass

class PromptExtractionError(OptimizerException):
    """Raised when prompt extraction fails"""
    pass

class PromptAnalysisError(OptimizerException):
    """Raised when prompt analysis fails"""
    pass

class OptimizationError(OptimizerException):
    """Raised when optimization process fails"""
    pass

class VersionConflictError(OptimizerException):
    """Raised when version conflict occurs"""
    pass
```

**Error Handling Principles:**
- Log all errors with context (workflow_id, prompt_id, etc.)
- Fail gracefully (return degraded results when possible)
- Propagate critical errors to caller
- Provide actionable error messages

---

### 3.6 Logging Integration

Following existing logger patterns (`src/utils/logger`):

```python
from src.utils.logger import get_logger, log_performance, log_context

logger = get_logger("optimizer.extractor")

class PromptExtractor:
    @log_performance("Prompt Extraction")
    def extract_from_workflow(self, workflow_id: str) -> List[Prompt]:
        with log_context(workflow_id=workflow_id, operation="extract_prompts"):
            logger.info(f"Starting prompt extraction for workflow: {workflow_id}")

            try:
                prompts = self._do_extraction(workflow_id)
                logger.info(
                    f"Extracted {len(prompts)} prompts from workflow",
                    extra={"prompt_count": len(prompts)}
                )
                return prompts
            except Exception as e:
                logger.error(
                    f"Prompt extraction failed: {str(e)}",
                    extra={"error_type": type(e).__name__},
                    exc_info=True
                )
                raise PromptExtractionError(f"Failed to extract prompts: {e}")
```

---

## 4. Features Deferred to Future Iterations

### 4.1 Advanced Features (Post-MVP)

| Feature | Description | Reason for Deferral |
|---------|-------------|---------------------|
| **LLM-Based Analysis** | Use GPT-4 to evaluate prompt quality | MVP uses rule-based heuristics (faster, no API costs) |
| **Multi-Objective Optimization** | Pareto optimization balancing multiple metrics | Complex algorithm; MVP uses single-strategy focus |
| **A/B Testing Integration** | Built-in A/B test framework | Requires integration with workflow runner (complex) |
| **Prompt Template Library** | Reusable prompt templates | Needs user research and template curation |
| **Real-time Optimization** | Optimize during test execution | Requires async architecture changes |
| **Persistent Version Storage** | Database-backed version history | MVP uses in-memory storage (simpler) |
| **Branching and Merging** | Git-like version branching | Complex version management; MVP is linear |
| **Collaborative Editing** | Multi-user prompt editing | Requires authentication and conflict resolution |
| **Automated Rollback** | Auto-rollback on quality regression | Requires production monitoring integration |
| **Prompt Diff Visualization** | Visual diff tool for prompt changes | UI component; beyond CLI scope |
| **Multi-language Support** | Analyze prompts in languages other than English | Requires NLP libraries for each language |
| **Ensemble Optimization** | Combine multiple optimization strategies | Complex algorithm; MVP uses single-pass |

### 4.2 Integration Features (Post-MVP)

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| **Workflow Auto-Discovery** | Automatically find workflows needing optimization | Requires auth module completion |
| **Dify API Integration** | Fetch/update workflows via Dify API | Requires auth + workflow modules |
| **Excel Report Export** | Export optimization reports to Excel | Collector module pattern (can reuse) |
| **Scheduled Optimization** | Cron-like scheduled optimization runs | Requires task scheduling framework |
| **Slack/Email Notifications** | Alert on optimization completion | Requires notification service |

### 4.3 Rationale for Deferrals

1. **MVP Scope Focus**: Deliver core optimization loop first
2. **Avoid External Dependencies**: LLM APIs, databases add complexity
3. **Testability Priority**: Rule-based approaches easier to test deterministically
4. **Incremental Value**: Each deferred feature can be added independently later
5. **Learning Opportunity**: Gather user feedback on MVP before advanced features

---

## 5. Implementation Plan

### 5.1 Phase Breakdown

#### Phase 1: Foundation (Week 1)
- **Day 1-2**:
  - Create `models.py` with all data structures
  - Set up test framework (`src/test/optimizer/`)
  - Create fixtures (sample DSL files, expected outputs)

- **Day 3-4**:
  - Implement `PromptExtractor`
  - Write extractor tests (80%+ coverage)
  - Validate integration with WorkflowCatalog

- **Day 5**:
  - Code review and refactoring
  - Update project documentation

#### Phase 2: Analysis (Week 2)
- **Day 1-3**:
  - Implement `PromptAnalyzer`
  - Develop scoring algorithms (clarity, efficiency)
  - Write analyzer tests (90%+ coverage)

- **Day 4-5**:
  - Implement issue detection heuristics
  - Implement suggestion generation
  - Integration testing with extractor

#### Phase 3: Optimization (Week 2-3)
- **Day 1-3**:
  - Implement `OptimizationEngine`
  - Develop optimization strategies (clarity, efficiency, structure)
  - Write optimization tests

- **Day 4-5**:
  - Implement improvement scoring
  - Implement candidate comparison logic
  - Integration testing

#### Phase 4: Version Management (Week 3)
- **Day 1-2**:
  - Implement `VersionManager`
  - Implement version creation and retrieval
  - Write version management tests

- **Day 3**:
  - Implement version comparison
  - Implement history tracking

#### Phase 5: Service Layer (Week 3-4)
- **Day 1-2**:
  - Implement `OptimizerService`
  - Implement orchestration logic
  - Write service-level integration tests

- **Day 3**:
  - Error handling and logging integration
  - Performance optimization

- **Day 4-5**:
  - End-to-end integration testing
  - Documentation and examples

#### Phase 6: Integration and Polish (Week 4)
- **Day 1-2**:
  - Integration with main.py
  - Update configuration files
  - CLI command for optimization

- **Day 3-4**:
  - Performance benchmarking
  - Code quality review
  - Documentation finalization

- **Day 5**:
  - Final testing
  - Release preparation

### 5.2 Milestones and Deliverables

| Milestone | Deliverables | Success Criteria |
|-----------|--------------|------------------|
| **M1: Foundation Complete** | models.py, PromptExtractor | Can extract prompts from DSL |
| **M2: Analysis Ready** | PromptAnalyzer | Can score and analyze prompts |
| **M3: Optimization Live** | OptimizationEngine | Can generate optimized variants |
| **M4: Version Tracking** | VersionManager | Can track prompt history |
| **M5: Service API** | OptimizerService | Can run full optimization cycle |
| **M6: Production Ready** | Integration, docs, tests | 90%+ test coverage, documented |

### 5.3 Dependencies

**Internal Dependencies:**
- `src/config` models (WorkflowCatalog, PromptPatch, etc.) - ✅ Ready
- `src/collector` models (TestResult, PerformanceMetrics) - ✅ Ready
- `src/utils/logger` - ✅ Ready
- `src/utils/exceptions` - ✅ Ready

**External Dependencies (from requirements.txt):**
- `pydantic >= 2.0.0` - ✅ Available
- `PyYAML >= 6.0.0` - ✅ Available
- `jinja2 >= 3.1.0` - ✅ Available (for template-based optimization)

**New Dependencies (if needed):**
- None for MVP (using rule-based approaches)
- Future: `openai`, `anthropic` for LLM-based analysis

### 5.4 Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Unclear optimization criteria** | High | Medium | Start with simple heuristics, gather user feedback |
| **Integration complexity** | Medium | Low | Follow existing module patterns, thorough testing |
| **Performance bottlenecks** | Medium | Low | Profile early, optimize hot paths |
| **Scope creep** | High | Medium | Strictly enforce MVP feature list, defer nice-to-haves |
| **Test coverage gaps** | Medium | Low | TDD approach, automated coverage reports |

---

## 6. Acceptance Criteria

### 6.1 Functional Requirements

- [ ] PromptExtractor can extract all LLM prompts from a workflow DSL
- [ ] PromptAnalyzer produces clarity and efficiency scores
- [ ] OptimizationEngine generates at least one improved variant per strategy
- [ ] VersionManager tracks version history linearly
- [ ] OptimizerService orchestrates full optimization cycle
- [ ] All components integrate with logging system
- [ ] Error handling covers common failure scenarios

### 6.2 Quality Requirements

- [ ] Test coverage ≥ 80% (target: 90%)
- [ ] All public methods have docstrings
- [ ] Code follows project naming conventions
- [ ] No circular dependencies
- [ ] All exceptions are custom and well-documented
- [ ] Logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)

### 6.3 Integration Requirements

- [ ] Optimizer integrates with config module (reads WorkflowCatalog)
- [ ] Optimizer integrates with collector module (reads PerformanceMetrics)
- [ ] Generated PromptPatch objects can be applied via PromptPatchEngine
- [ ] CLI command `python src/main.py --mode optimize --workflow-id <id>` works

### 6.4 Documentation Requirements

- [ ] Module README updated with usage examples
- [ ] API documentation for all public classes/methods
- [ ] Example workflow demonstrating optimization cycle
- [ ] Blueprint document (this file) maintained and updated

---

## 7. Usage Examples

### 7.1 Basic Optimization Workflow

```python
from src.config import ConfigLoader, WorkflowCatalog, TestPlan
from src.executor import ExecutorService
from src.collector import DataCollector
from src.optimizer import OptimizerService

# 1. Load configuration
loader = ConfigLoader()
env_config = loader.load_env_config("config/env.yaml")
catalog = loader.load_workflow_catalog("config/workflows.yaml")
test_plan = loader.load_test_plan("config/test_plan.yaml")

# 2. Run baseline tests
executor = ExecutorService()
baseline_results = executor.execute_test_plan(test_plan.to_manifest())

# 3. Collect baseline metrics
collector = DataCollector()
for result in baseline_results:
    collector.collect_result(result)
baseline_metrics = collector.get_statistics(workflow_id="wf_001")

# 4. Run optimization
optimizer = OptimizerService(catalog)
patches = optimizer.run_optimization_cycle(
    workflow_id="wf_001",
    baseline_metrics=baseline_metrics,
    strategy="clarity_focus"
)

# 5. Apply patches to test plan
test_plan.workflows[0].prompt_patches.extend(patches)

# 6. Re-run tests with optimized prompts
optimized_results = executor.execute_test_plan(test_plan.to_manifest())

# 7. Compare results
for result in optimized_results:
    collector.collect_result(result)
optimized_metrics = collector.get_statistics(workflow_id="wf_001")

print(f"Baseline success rate: {baseline_metrics.success_rate:.2%}")
print(f"Optimized success rate: {optimized_metrics.success_rate:.2%}")
print(f"Improvement: {optimized_metrics.success_rate - baseline_metrics.success_rate:.2%}")
```

### 7.2 Standalone Prompt Analysis

```python
from src.optimizer import PromptAnalyzer, Prompt
from datetime import datetime

# Create analyzer
analyzer = PromptAnalyzer()

# Analyze a prompt
prompt = Prompt(
    id="wf_001_llm_1",
    workflow_id="wf_001",
    node_id="llm_1",
    node_type="llm",
    text="Summarize the following document in 3-5 bullet points: {{document}}",
    role="user",
    variables=["document"],
    context={},
    extracted_at=datetime.now()
)

analysis = analyzer.analyze_prompt(prompt)

print(f"Clarity Score: {analysis.clarity_score}")
print(f"Efficiency Score: {analysis.efficiency_score}")
print(f"Overall Score: {analysis.overall_score}")
print(f"Issues: {', '.join(analysis.issues)}")
print(f"Suggestions: {', '.join(analysis.suggestions)}")
```

### 7.3 Version Management

```python
from src.optimizer import VersionManager

# Create version manager
version_mgr = VersionManager()

# Create initial version
v1 = version_mgr.create_version(
    prompt_id="wf_001_llm_1",
    text="Original prompt text...",
    metrics=original_analysis,
    author="baseline"
)

# Create optimized version
v2 = version_mgr.create_version(
    prompt_id="wf_001_llm_1",
    text="Optimized prompt text...",
    metrics=optimized_analysis,
    author="optimizer",
    parent_version=v1.version
)

# Get version history
history = version_mgr.get_version_history("wf_001_llm_1")
for version in history:
    print(f"v{version.version}: {version.metrics.overall_score}")

# Compare versions
comparison = version_mgr.compare_versions(
    prompt_id="wf_001_llm_1",
    version1=v1.version,
    version2=v2.version
)
print(f"Improvement: {comparison.improvement}%")
```

---

## 8. Next Steps

### 8.1 Immediate Actions

1. **Review this blueprint** with stakeholders
2. **Create GitHub issues** for each component (6 issues)
3. **Set up test infrastructure** (`src/test/optimizer/conftest.py`)
4. **Create sample fixtures** (workflow DSL, expected outputs)
5. **Begin Phase 1 implementation** (models.py, PromptExtractor)

### 8.2 Coordination Points

- **Daily**: Update `project.md` with progress
- **Weekly**: Code review checkpoint
- **Bi-weekly**: Integration testing with other modules
- **End of sprint**: Demo to stakeholders

### 8.3 Success Metrics

- **Code Quality**: 90%+ test coverage, 0 critical linting issues
- **Performance**: Optimization cycle completes in < 5 seconds per workflow
- **Usability**: CLI command works end-to-end with real workflows
- **Integration**: Generates valid PromptPatch objects that PromptPatchEngine can apply

---

## 9. Appendix

### 9.1 Terminology

- **Prompt**: Text instructions given to LLM nodes in a workflow
- **Prompt Patch**: Modification to a prompt (via PromptPatchEngine)
- **Optimization Cycle**: Extract → Analyze → Optimize → Version → Patch
- **Baseline Metrics**: Performance metrics before optimization
- **Variant**: Alternative version of a prompt (original, optimized_v1, etc.)
- **Strategy**: Optimization approach (clarity_focus, efficiency_focus, etc.)

### 9.2 References

- Project README: `D:\Work\dify_autoopt\README.md`
- Optimizer Design: `D:\Work\dify_autoopt\src\optimizer\README.md`
- Config Models: `D:\Work\dify_autoopt\src\config\models.py`
- Executor Architecture: `D:\Work\dify_autoopt\src\executor\`
- Collector Implementation: `D:\Work\dify_autoopt\src\collector\`

### 9.3 Open Questions

1. **Scoring Algorithm Tuning**: What weights should we use for clarity vs. efficiency?
   - **Recommendation**: Start with 60/40, allow configuration override

2. **Optimization Threshold**: When is a prompt "good enough" to skip optimization?
   - **Recommendation**: Overall score > 80, or < 5% potential improvement

3. **Version Numbering**: Semantic versioning or simple incremental?
   - **Recommendation**: Semantic versioning (major.minor.patch) for clarity

4. **Patch Application Timing**: When to apply patches (immediately, after approval, scheduled)?
   - **Recommendation**: Manual approval in MVP, auto-apply in future

---

**End of Blueprint**

---

**Change Log:**

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-17 | 1.0.0 | Initial blueprint creation | Project Manager Agent |
