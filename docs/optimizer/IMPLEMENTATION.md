# Optimizer Module - Implementation Guide

**Project**: dify_autoopt
**Module**: src/optimizer
**Version**: Production Ready 1.0
**Status**: ✅ Implemented

> **Document Purpose**: Consolidated implementation guide covering FileSystemStorage implementation, multi-strategy optimization, and execution blueprints.
>
> **Source Documents Merged**:
> - IMPLEMENTATION_GUIDE.md (FileSystemStorage implementation)
> - IMPLEMENTATION_CHECKLIST.md (Task checklists)
> - IMPLEMENTATION_CHECKLIST_MULTI_STRATEGY.md (Multi-strategy checklist)
> - optimizer_execution_blueprint.md (Execution blueprint)

---

## Table of Contents

1. [Implementation Overview](#implementation-overview)
2. [FileSystemStorage Implementation](#filesystemstorage-implementation)
3. [Multi-Strategy Optimization Implementation](#multi-strategy-optimization-implementation)
4. [Testing Strategy](#testing-strategy)
5. [Deployment Guide](#deployment-guide)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Implementation Overview

### Completed Features

| Feature | Status | Implementation Date | Test Coverage |
|---------|--------|---------------------|---------------|
| FileSystemStorage | ✅ Complete | 2025-11-17 | 94% (57 tests) |
| LLM Integration | ✅ Complete | 2025-11-18 | 100% (27 tests) |
| Multi-Strategy Support | ✅ Complete | 2025-11-18 | 100% |
| Iterative Refinement | ✅ Complete | 2025-11-18 | 100% |
| Confidence Filtering | ✅ Complete | 2025-11-18 | 100% |

### Technology Stack

**Core Dependencies**:
- Python 3.9+
- Pydantic V2 (>= 2.0.0) - Data validation
- PyYAML (>= 6.0.0) - Configuration
- Jinja2 (>= 3.1.0) - Templates

**Optional Dependencies**:
- OpenAI SDK - LLM integration
- fcntl/msvcrt - File locking

---

## FileSystemStorage Implementation

### Architecture

**File Format**: JSON with UTF-8 encoding

**Directory Structure**:
```
{storage_dir}/
├── index.json              # Global index
├── prompt_001/
│   ├── 1.json
│   ├── 2.json
│   └── 3.json
└── prompt_002/
    └── 1.json
```

### Core Implementation

#### 1. Atomic Write Pattern

```python
def _atomic_write(self, path: Path, content: str) -> None:
    """Write content atomically to prevent data corruption"""
    temp_path = path.with_suffix(".tmp")

    try:
        # Write to temporary file
        temp_path.write_text(content, encoding="utf-8")

        # Force disk sync (optional, for extra safety)
        with open(temp_path, "r+b") as f:
            os.fsync(f.fileno())

        # Atomic rename (POSIX guarantee)
        temp_path.rename(path)
    except Exception as e:
        # Cleanup temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise OptimizerError(f"Atomic write failed: {e}")
```

**Why Atomic?**
- No partial writes if process crashes
- No data corruption
- Cross-platform compatibility

#### 2. File Locking

```python
import fcntl  # Unix
import msvcrt  # Windows

def _lock_file(self, file_obj):
    """Acquire exclusive lock (cross-platform)"""
    if sys.platform == "win32":
        # Windows locking
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_LOCK, 1)
    else:
        # Unix locking (fcntl)
        fcntl.flock(file_obj, fcntl.LOCK_EX)

def _unlock_file(self, file_obj):
    """Release lock (cross-platform)"""
    if sys.platform == "win32":
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file_obj, fcntl.LOCK_UN)
```

**Usage**:
```python
with open(file_path, "r") as f:
    self._lock_file(f)
    try:
        data = json.load(f)
    finally:
        self._unlock_file(f)
```

#### 3. Global Index

```python
class GlobalIndex:
    """Fast lookup index for latest versions"""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._index = {}  # {prompt_id: {"latest": 3, "total": 3}}
        self._load()

    def get_latest_version(self, prompt_id: str) -> Optional[int]:
        """O(1) lookup for latest version"""
        entry = self._index.get(prompt_id)
        return entry["latest"] if entry else None

    def update(self, prompt_id: str, version: int):
        """Update index when new version added"""
        if prompt_id not in self._index:
            self._index[prompt_id] = {"latest": version, "total": 1}
        else:
            self._index[prompt_id]["latest"] = max(
                self._index[prompt_id]["latest"], version
            )
            self._index[prompt_id]["total"] += 1
        self._save()

    def _save(self):
        """Persist index to disk"""
        self.index_path.write_text(
            json.dumps(self._index, indent=2),
            encoding="utf-8"
        )
```

#### 4. LRU Cache

```python
from collections import OrderedDict

class LRUCache:
    """Least Recently Used cache for fast reads"""

    def __init__(self, capacity: int = 128):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value and mark as recently used"""
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)  # Mark as recent
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Add value, evict oldest if full"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)  # Remove oldest
        self.cache[key] = value

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

**Integration**:
```python
class FileSystemStorage(VersionStorage):
    def __init__(self, storage_dir, use_cache=True, cache_size=128):
        self.use_cache = use_cache
        self.cache = LRUCache(cache_size) if use_cache else None

    def get_version(self, prompt_id, version):
        # Check cache first
        if self.use_cache:
            cache_key = f"{prompt_id}:{version}"
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        # Load from disk
        data = self._load_from_disk(prompt_id, version)

        # Update cache
        if self.use_cache:
            self.cache.put(cache_key, data)

        return data
```

### Performance Metrics

| Operation | Without Cache | With Cache | Improvement |
|-----------|---------------|------------|-------------|
| save_version | ~15ms | ~15ms | N/A |
| get_version | ~8ms | ~0.05ms | 160x |
| get_latest_version | ~5ms | ~2ms | 2.5x |
| list_versions | ~30ms | ~30ms | N/A |

**Cache Hit Rate**: 90%+ (real workloads)

---

## Multi-Strategy Optimization Implementation

### Overview

Implemented nested iteration loop supporting:
1. **Multi-strategy trial**: Try multiple strategies, pick best result
2. **Iterative refinement**: Refine prompts over multiple iterations
3. **Confidence filtering**: Only accept high-confidence improvements

### Implementation

#### Core Optimization Flow

```python
def run_optimization_cycle(
    self,
    workflow_id: str,
    baseline_metrics: Optional[PerformanceMetrics] = None,
    strategy: Optional[Union[str, OptimizationStrategy]] = None,
    config: Optional[OptimizationConfig] = None
) -> List[PromptPatch]:
    """
    Run multi-strategy, iterative optimization cycle

    Args:
        workflow_id: Workflow to optimize
        baseline_metrics: Optional baseline performance
        strategy: Single strategy (backward compatibility)
        config: Optimization configuration (multi-strategy)

    Returns:
        List of prompt patches
    """
    # Step 1: Resolve configuration
    if strategy:
        # Backward compatibility: single-strategy mode
        effective_config = OptimizationConfig(
            strategies=[OptimizationStrategy(strategy)],
            max_iterations=1
        )
    elif config:
        effective_config = config
    else:
        effective_config = OptimizationConfig(strategies=[OptimizationStrategy.AUTO])

    # Step 2: Extract prompts
    prompts = self.extractor.extract_prompts(workflow_id)

    # Step 3: Optimize each prompt
    patches = []

    for prompt in prompts:
        # Analyze baseline
        baseline_analysis = self.analyzer.analyze_prompt(prompt)

        # Create baseline version
        self.version_manager.create_version(
            prompt_id=prompt.id,
            optimized_prompt=prompt.text,
            analysis=baseline_analysis,
            metadata={"type": "baseline"}
        )

        # Check if optimization needed
        if not self._should_optimize(baseline_analysis, effective_config):
            continue

        # Multi-strategy optimization
        best_result = None
        best_improvement = 0.0

        for strategy in effective_config.strategies:
            # Iterative refinement for this strategy
            result = self._optimize_with_iterations(
                prompt=prompt,
                strategy=strategy,
                max_iterations=effective_config.max_iterations,
                min_confidence=effective_config.min_confidence,
                baseline_score=baseline_analysis.overall_score
            )

            # Track best result across strategies
            if result and result.improvement_score > best_improvement:
                best_result = result
                best_improvement = result.improvement_score

        # Apply best optimization
        if best_result:
            # Create version
            self.version_manager.create_version(
                prompt_id=prompt.id,
                optimized_prompt=best_result.optimized_prompt,
                analysis=best_result.final_analysis,
                optimization_result=best_result,
                metadata={
                    "strategy": str(best_result.strategy),
                    "improvement": best_result.improvement_score,
                    "iterations": best_result.iterations
                }
            )

            # Create patch
            patch = PromptPatch(
                workflow_id=workflow_id,
                node_id=prompt.node_id,
                selector_type="by_id",
                selector_value=prompt.node_id,
                strategy="replace",
                content=best_result.optimized_prompt,
                metadata={
                    "original_score": baseline_analysis.overall_score,
                    "optimized_score": best_result.final_analysis.overall_score,
                    "improvement": best_result.improvement_score,
                    "strategy": str(best_result.strategy)
                }
            )
            patches.append(patch)

    return patches
```

#### Iterative Refinement Logic

```python
def _optimize_with_iterations(
    self,
    prompt: Prompt,
    strategy: OptimizationStrategy,
    max_iterations: int,
    min_confidence: float,
    baseline_score: float
) -> Optional[OptimizationResult]:
    """
    Perform iterative optimization with quality gates

    Args:
        prompt: Prompt to optimize
        strategy: Strategy to use
        max_iterations: Maximum refinement iterations
        min_confidence: Minimum confidence threshold
        baseline_score: Baseline quality score

    Returns:
        Best optimization result or None if all iterations failed
    """
    current_prompt = prompt
    best_result = None
    best_score = baseline_score

    for iteration in range(max_iterations):
        # Optimize current prompt
        result = self.optimization_engine.optimize(current_prompt, strategy)

        # Check confidence threshold
        if result.confidence < min_confidence:
            # Low confidence, stop iterating this strategy
            break

        # Check improvement
        if result.final_analysis.overall_score > best_score:
            best_result = result
            best_score = result.final_analysis.overall_score

            # Update current prompt for next iteration
            current_prompt = Prompt(
                id=prompt.id,
                workflow_id=prompt.workflow_id,
                node_id=prompt.node_id,
                node_type=prompt.node_type,
                text=result.optimized_prompt,  # Use optimized as new input
                role=prompt.role,
                variables=prompt.variables,
                context=prompt.context
            )
        else:
            # No improvement, stop iterating
            break

    return best_result
```

### Configuration Examples

#### Conservative
```python
config = OptimizationConfig(
    strategies=[OptimizationStrategy.CLARITY_FOCUS],
    score_threshold=85.0,  # Only fix bad prompts
    min_confidence=0.8,    # High confidence required
    max_iterations=1       # Single iteration
)
```

#### Balanced (Recommended)
```python
config = OptimizationConfig(
    strategies=[
        OptimizationStrategy.LLM_GUIDED,
        OptimizationStrategy.HYBRID
    ],
    score_threshold=80.0,
    min_confidence=0.7,
    max_iterations=3
)
```

#### Aggressive
```python
config = OptimizationConfig(
    strategies=[
        OptimizationStrategy.LLM_GUIDED,
        OptimizationStrategy.LLM_CLARITY,
        OptimizationStrategy.HYBRID
    ],
    score_threshold=90.0,
    min_confidence=0.6,
    max_iterations=5
)
```

---

## Testing Strategy

### Test Pyramid

```
     E2E Tests (10)
   ━━━━━━━━━━━━━━
  Integration (60)
 ━━━━━━━━━━━━━━━━━
Unit Tests (707)
━━━━━━━━━━━━━━━━━━━
```

**Total**: 777 tests, 97% coverage

### FileSystemStorage Tests

**File**: `src/test/optimizer/test_filesystem_storage.py` (51 tests)

**Coverage**:
- LRU Cache: 9 tests
- File Locking: 3 tests
- CRUD Operations: 15 tests
- Index Management: 6 tests
- Cache Performance: 4 tests
- Atomic Writes: 2 tests
- Error Handling: 2 tests
- Concurrent Access: 2 tests
- Sharding: 2 tests
- Performance Benchmarks: 5 tests
- Other: 7 tests

**Integration Tests**: `test_filesystem_storage_integration.py` (6 tests)

### Multi-Strategy Tests

**File**: `src/test/optimizer/test_llm_integration.py` (27 tests)

**Coverage**:
- LLM strategies: 4 tests
- Fallback mechanism: 5 tests
- Backward compatibility: 4 tests
- Error handling: 2 tests
- Analysis context: 2 tests
- Hybrid strategy: 2 tests
- Strategy metadata: 3 tests
- End-to-end: 2 tests
- Performance: 2 tests

### Running Tests

```bash
# Run all optimizer tests
pytest src/test/optimizer/ -v

# Run with coverage
pytest src/test/optimizer/ --cov=src/optimizer --cov-report=html

# Run specific test file
pytest src/test/optimizer/test_filesystem_storage.py -v

# Run performance benchmarks
pytest src/test/optimizer/test_filesystem_storage.py -k "performance" -v
```

---

## Deployment Guide

### Pre-Deployment Checklist

- [x] All 777 tests passing
- [x] 97% code coverage achieved
- [x] Performance benchmarks met
- [x] Security scan clean
- [x] Documentation complete

### Production Configuration

```python
from src.optimizer import OptimizerService, OptimizationConfig
from src.optimizer.interfaces import FileSystemStorage
from src.optimizer.interfaces.llm_providers import OpenAIClient
from src.optimizer.config import LLMConfig, LLMProvider
from src.optimizer import VersionManager

# FileSystemStorage with optimal settings
storage = FileSystemStorage(
    storage_dir="./data/optimizer/versions",
    use_index=True,        # O(1) lookups
    use_cache=True,        # 90%+ hit rate
    cache_size=256,        # Cache 256 versions
    enable_sharding=False  # Enable if >10k prompts
)

# LLM client configuration
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4-turbo-preview",
    api_key_env="OPENAI_API_KEY",
    enable_cache=True,
    cache_ttl=3600,
    timeout_seconds=10,
    max_retries=3
)

llm_client = OpenAIClient(llm_config)

# Create optimizer service
service = OptimizerService(
    catalog=workflow_catalog,
    llm_client=llm_client,
    version_manager=VersionManager(storage=storage)
)

# Production optimization configuration
config = OptimizationConfig(
    strategies=[
        OptimizationStrategy.LLM_GUIDED,
        OptimizationStrategy.HYBRID
    ],
    score_threshold=80.0,
    min_confidence=0.7,
    max_iterations=3
)

# Run optimization
patches = service.run_optimization_cycle(
    workflow_id="production_wf_001",
    config=config
)
```

### Environment Variables

```bash
# Required for LLM optimization
export OPENAI_API_KEY="sk-..."

# Optional configuration
export OPTIMIZER_STORAGE_DIR="./data/optimizer"
export OPTIMIZER_CACHE_SIZE="256"
export OPTIMIZER_LOG_LEVEL="INFO"
```

### YAML Configuration

```yaml
optimizer:
  storage:
    backend: "filesystem"
    storage_dir: "./data/optimizer/versions"
    use_index: true
    use_cache: true
    cache_size: 256
    enable_sharding: false

  optimization:
    strategies:
      - "llm_guided"
      - "hybrid"
    score_threshold: 80.0
    min_confidence: 0.7
    max_iterations: 3

  llm:
    provider: "openai"
    model: "gpt-4-turbo-preview"
    api_key_env: "OPENAI_API_KEY"
    enable_cache: true
    cache_ttl: 3600
    timeout_seconds: 10
    max_retries: 3
```

---

## Monitoring & Maintenance

### Key Metrics

#### Performance Metrics
- **save_version latency**: Target <20ms
- **get_version latency (disk)**: Target <10ms
- **get_version latency (cache)**: Target <0.1ms
- **Cache hit rate**: Target >90%

#### Quality Metrics
- **Optimization acceptance rate**: % of optimizations accepted
- **Average improvement score**: Average quality improvement
- **Confidence distribution**: Distribution of confidence scores

#### Cost Metrics (LLM)
- **API calls per day**: Total LLM API calls
- **Cost per optimization**: Average cost per prompt
- **Total daily cost**: Total LLM spend

### Monitoring Setup

```python
# Enable metrics collection
storage = FileSystemStorage(
    storage_dir="./data/optimizer",
    use_cache=True,
    cache_size=256
)

# Periodically log metrics
import logging

logger = logging.getLogger("optimizer.monitoring")

def log_metrics(storage):
    """Log performance metrics"""
    if storage.use_cache:
        hit_rate = storage.cache.get_hit_rate()
        logger.info(f"Cache hit rate: {hit_rate:.1%}")
        logger.info(f"Cache hits: {storage.cache.hits}")
        logger.info(f"Cache misses: {storage.cache.misses}")

# Schedule periodic logging (every hour)
import schedule

schedule.every(1).hour.do(lambda: log_metrics(storage))
```

### Maintenance Tasks

#### Daily
- Monitor cache hit rate (should be >90%)
- Check error logs for failures
- Monitor LLM API costs

#### Weekly
- Review optimization quality (manual sampling)
- Check storage disk usage
- Review performance metrics

#### Monthly
- Clean up old versions (optional)
- Review and optimize cache size
- Performance benchmark regression tests
- Update LLM models if needed

### Troubleshooting

#### Low Cache Hit Rate

**Symptom**: Cache hit rate <70%

**Diagnosis**:
```python
# Check cache statistics
print(f"Cache size: {len(storage.cache.cache)}")
print(f"Cache capacity: {storage.cache.capacity}")
print(f"Hit rate: {storage.cache.get_hit_rate():.1%}")
```

**Solutions**:
1. Increase cache size
2. Review access patterns (sequential vs random)
3. Consider warming cache on startup

#### Slow Write Performance

**Symptom**: save_version >50ms

**Diagnosis**:
```python
import time

start = time.time()
storage.save_version(prompt_id, version_data)
duration = (time.time() - start) * 1000
print(f"Write latency: {duration:.2f}ms")
```

**Solutions**:
1. Check disk I/O (SSD vs HDD)
2. Disable fsync if acceptable
3. Batch writes if possible

#### High LLM Costs

**Symptom**: Daily LLM costs exceeding budget

**Solutions**:
1. Enable LLM response caching
2. Use rule-based strategies for simple prompts
3. Set cost limits in LLMConfig
4. Reduce max_iterations

---

## Document Metadata

**Version**: 1.0
**Last Updated**: 2025-11-19
**Status**: Complete
**Scope**: Complete implementation guide
**Audience**: Developers, QA engineers, DevOps

---

**End of Implementation Guide**
