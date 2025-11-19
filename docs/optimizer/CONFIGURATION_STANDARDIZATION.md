# Configuration Field Standardization Recommendations

**Project:** dify_autoopt
**Module:** src/optimizer
**Version:** 1.0.0
**Date:** 2025-01-17
**Author:** Senior System Architect

---

## Executive Summary

This document addresses configuration field naming inconsistencies in the Optimizer module and provides standardization recommendations aligned with system-wide best practices.

### Key Recommendations

| Current Field | Recommendation | Rationale |
|--------------|----------------|-----------|
| `improvement_threshold` | **Keep as-is** | Clear semantic meaning (delta threshold) |
| `score_threshold` | **Rename to** `min_baseline_score` | Consistent with `min_*` pattern |
| `confidence_threshold` | **Rename to** `min_confidence` | Consistent with `min_*` pattern |

---

## 1. Problem Statement

### 1.1 Current Inconsistencies

```python
# src/optimizer/models.py
class OptimizationConfig(BaseModel):
    improvement_threshold: float = 5.0      # Minimum improvement to accept
    confidence_threshold: float = 0.6       # Minimum confidence to accept

# src/optimizer/optimizer_service.py
class OptimizerService:
    def should_optimize(self, analysis: PromptAnalysis) -> bool:
        # Using implicit "score_threshold"
        return analysis.overall_score < 80.0  # Hardcoded, should be config
```

**Issues**:
1. Mixed naming patterns: `*_threshold` vs `min_*`
2. Hardcoded thresholds in service code
3. Unclear distinction between "improvement" and "score" thresholds
4. Inconsistent with other modules (Config, Executor use `min_*`)

### 1.2 Comparison with Other Modules

```python
# src/config/models/test_plan.py
class ValidationConfig(BaseModel):
    min_response_length: int = 10           # Uses min_* pattern
    max_response_length: int = 5000         # Uses max_* pattern

# src/executor/models.py
class RetryPolicy(BaseModel):
    max_attempts: int = 3                   # Uses max_* pattern
    backoff_seconds: float = 2.0

# Optimizer should follow same pattern
class OptimizationConfig(BaseModel):
    min_baseline_score: float = 80.0        # NEW: consistent pattern
    min_improvement: float = 5.0            # RENAMED
    min_confidence: float = 0.6             # RENAMED
```

---

## 2. Semantic Analysis

### 2.1 Field Meanings

| Field | Semantic Meaning | Type | Example |
|-------|-----------------|------|---------|
| **min_baseline_score** | Minimum quality score below which optimization is needed | Lower bound (absolute) | 80.0 = prompts < 80 need optimization |
| **min_improvement** | Minimum score delta required to accept optimization | Lower bound (delta) | 5.0 = must improve by ≥5 points |
| **min_confidence** | Minimum confidence level to trust optimization result | Lower bound (probability) | 0.6 = 60% confidence required |
| **max_iterations** | Maximum optimization attempts per prompt | Upper bound (count) | 3 = stop after 3 tries |

### 2.2 Decision Logic Flow

```python
def optimize_workflow(workflow_id: str, config: OptimizationConfig):
    """Optimization decision tree using standardized fields."""

    for prompt in prompts:
        # 1. Analyze current quality
        analysis = analyzer.analyze_prompt(prompt)

        # 2. Decision: Should optimize?
        if analysis.overall_score >= config.min_baseline_score:
            logger.info(f"Skipping optimization: score {analysis.overall_score} >= {config.min_baseline_score}")
            continue

        # 3. Generate optimization
        result = engine.optimize(prompt, strategy)

        # 4. Decision: Accept optimization?
        if result.improvement_score < config.min_improvement:
            logger.info(f"Rejecting optimization: improvement {result.improvement_score} < {config.min_improvement}")
            continue

        if result.confidence < config.min_confidence:
            logger.info(f"Rejecting optimization: confidence {result.confidence} < {config.min_confidence}")
            continue

        # 5. Accept optimization
        apply_optimization(result)
```

**Clarity Benefits**:
- `min_baseline_score` clearly indicates "minimum acceptable quality"
- `min_improvement` clearly indicates "minimum acceptable delta"
- `min_confidence` clearly indicates "minimum acceptable certainty"
- All use consistent `min_*` prefix for lower bounds

---

## 3. Proposed Configuration Schema

### 3.1 Complete OptimizationConfig Model

```python
"""
src/optimizer/models.py
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum


class OptimizationStrategy(str, Enum):
    """Optimization strategy options."""
    AUTO = "auto"
    CLARITY_FOCUS = "clarity_focus"
    EFFICIENCY_FOCUS = "efficiency_focus"
    STRUCTURE_FOCUS = "structure_focus"


class OptimizationConfig(BaseModel):
    """Comprehensive optimization configuration.

    Attributes:
        min_baseline_score: Prompts below this score will be optimized (0-100).
        min_improvement: Minimum score improvement to accept optimization (0-100).
        min_confidence: Minimum confidence level to accept optimization (0.0-1.0).
        max_iterations: Maximum optimization attempts per prompt.
        strategies: List of strategies to try (in order).
        scoring_weights: Component score weights (must sum to 1.0).
        enable_version_tracking: Enable version management.
        storage_backend: Storage backend type.
        storage_config: Backend-specific configuration.
    """

    # === Thresholds (Lower Bounds) ===
    min_baseline_score: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Prompts with score below this will be optimized (0-100)"
    )

    min_improvement: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Minimum score improvement to accept optimization (0-100)"
    )

    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence level to accept optimization (0.0-1.0)"
    )

    # === Limits (Upper Bounds) ===
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum optimization attempts per prompt"
    )

    max_prompt_length: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum prompt length in characters"
    )

    # === Strategy Configuration ===
    strategies: List[OptimizationStrategy] = Field(
        default_factory=lambda: [OptimizationStrategy.AUTO],
        description="Optimization strategies to use (in priority order)"
    )

    strategy_fallback: bool = Field(
        default=True,
        description="Try next strategy if current fails"
    )

    # === Scoring Configuration ===
    scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {"clarity": 0.6, "efficiency": 0.4},
        description="Component score weights (must sum to 1.0)"
    )

    # === Feature Flags ===
    enable_version_tracking: bool = Field(
        default=True,
        description="Enable version management and history"
    )

    enable_metrics_feedback: bool = Field(
        default=False,
        description="Use executor metrics for strategy selection"
    )

    # === Storage Configuration ===
    storage_backend: str = Field(
        default="memory",
        description="Storage backend: memory | filesystem | database"
    )

    storage_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific configuration"
    )

    # === Validation ===
    @field_validator('scoring_weights')
    @classmethod
    def validate_weights_sum_to_one(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Ensure scoring weights sum to 1.0."""
        total = sum(value.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating-point errors
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total}. "
                f"Weights: {value}"
            )
        return value

    @field_validator('storage_backend')
    @classmethod
    def validate_storage_backend(cls, value: str) -> str:
        """Validate storage backend name."""
        valid_backends = ["memory", "filesystem", "database"]
        if value not in valid_backends:
            raise ValueError(
                f"Invalid storage backend '{value}'. "
                f"Must be one of: {', '.join(valid_backends)}"
            )
        return value

    @field_validator('strategies')
    @classmethod
    def validate_strategies_not_empty(cls, value: List[OptimizationStrategy]) -> List[OptimizationStrategy]:
        """Ensure at least one strategy is specified."""
        if not value:
            raise ValueError("At least one optimization strategy must be specified")
        return value

    model_config = {
        "extra": "forbid",  # Reject unknown fields
        "validate_assignment": True,  # Validate on attribute assignment
        "use_enum_values": True,  # Use enum values in JSON
    }
```

### 3.2 YAML Configuration Example

```yaml
# config/env_config.yaml
optimizer:
  # === Optimization Behavior ===
  optimization:
    # Thresholds (when to optimize)
    min_baseline_score: 80.0     # Optimize prompts scoring below 80/100
    min_improvement: 5.0         # Accept optimizations improving by ≥5 points
    min_confidence: 0.6          # Accept optimizations with ≥60% confidence

    # Limits
    max_iterations: 3            # Try up to 3 optimization attempts
    max_prompt_length: 10000     # Skip prompts longer than 10k chars

    # Strategy
    strategies:
      - "auto"                   # Auto-select strategy based on analysis
    strategy_fallback: true      # Try next strategy if current fails

    # Scoring weights (must sum to 1.0)
    scoring_weights:
      clarity: 0.6               # 60% weight on clarity
      efficiency: 0.4            # 40% weight on efficiency

    # Feature flags
    enable_version_tracking: true
    enable_metrics_feedback: false

  # === Storage Configuration ===
  storage:
    backend: "filesystem"

    # Filesystem-specific config
    config:
      storage_dir: "./data/optimizer/versions"
      use_index: true            # Enable fast lookup index
      use_cache: true            # Enable in-memory cache
      cache_size: 100            # Cache up to 100 versions
      enable_sharding: false     # Disable directory sharding (< 1000 prompts)
```

---

## 4. Migration Guide

### 4.1 Code Changes Required

```python
# === BEFORE (Current MVP) ===
class OptimizationConfig(BaseModel):
    improvement_threshold: float = 5.0
    confidence_threshold: float = 0.6

class OptimizerService:
    def should_optimize(self, analysis: PromptAnalysis) -> bool:
        return analysis.overall_score < 80.0  # Hardcoded!

    def should_accept(self, result: OptimizationResult) -> bool:
        return (
            result.improvement_score >= self.config.improvement_threshold and
            result.confidence >= self.config.confidence_threshold
        )


# === AFTER (Standardized) ===
class OptimizationConfig(BaseModel):
    min_baseline_score: float = 80.0  # NEW
    min_improvement: float = 5.0      # RENAMED
    min_confidence: float = 0.6       # RENAMED

class OptimizerService:
    def should_optimize(self, analysis: PromptAnalysis) -> bool:
        # Now configurable
        return analysis.overall_score < self.config.min_baseline_score

    def should_accept(self, result: OptimizationResult) -> bool:
        return (
            result.improvement_score >= self.config.min_improvement and
            result.confidence >= self.config.min_confidence
        )
```

### 4.2 Configuration Migration Script

```python
"""
scripts/migrate_optimizer_config.py

Migrate old optimizer configuration to new standardized format.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def migrate_config(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate old config format to new standardized format."""

    new_config = {}

    # Rename fields
    if "improvement_threshold" in old_config:
        new_config["min_improvement"] = old_config["improvement_threshold"]

    if "confidence_threshold" in old_config:
        new_config["min_confidence"] = old_config["confidence_threshold"]

    # Add new required fields with defaults
    new_config.setdefault("min_baseline_score", 80.0)
    new_config.setdefault("max_iterations", 3)
    new_config.setdefault("strategies", ["auto"])
    new_config.setdefault("scoring_weights", {"clarity": 0.6, "efficiency": 0.4})
    new_config.setdefault("enable_version_tracking", True)

    # Storage config (if exists)
    if "storage" in old_config:
        if isinstance(old_config["storage"], str):
            # Old format: storage: "memory"
            new_config["storage_backend"] = old_config["storage"]
            new_config["storage_config"] = {}
        else:
            # Already new format
            new_config["storage_backend"] = old_config["storage"].get("backend", "memory")
            new_config["storage_config"] = old_config["storage"].get("config", {})
    else:
        new_config["storage_backend"] = "memory"
        new_config["storage_config"] = {}

    return new_config


def migrate_yaml_file(input_path: Path, output_path: Path = None):
    """Migrate YAML config file to new format."""

    # Load old config
    with open(input_path, 'r') as f:
        old_yaml = yaml.safe_load(f)

    # Migrate optimizer section
    if "optimizer" in old_yaml:
        old_yaml["optimizer"] = migrate_config(old_yaml["optimizer"])

    # Save new config
    output_path = output_path or input_path
    with open(output_path, 'w') as f:
        yaml.dump(old_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Migrated {input_path} → {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python migrate_optimizer_config.py <config.yaml> [output.yaml]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    migrate_yaml_file(input_file, output_file)
```

### 4.3 Backward Compatibility (Deprecation Path)

```python
"""
Support old field names with deprecation warnings (temporary).
"""

from pydantic import BaseModel, Field, field_validator
import warnings


class OptimizationConfig(BaseModel):
    # === New Fields (Preferred) ===
    min_baseline_score: float = Field(default=80.0, ge=0.0, le=100.0)
    min_improvement: float = Field(default=5.0, ge=0.0, le=100.0)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    # === Deprecated Fields (Backward Compatibility) ===
    improvement_threshold: Optional[float] = Field(
        default=None,
        deprecated="Use 'min_improvement' instead. Will be removed in v2.0.0"
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        deprecated="Use 'min_confidence' instead. Will be removed in v2.0.0"
    )

    def __init__(self, **data):
        # Handle deprecated fields
        if "improvement_threshold" in data and "min_improvement" not in data:
            warnings.warn(
                "Field 'improvement_threshold' is deprecated. "
                "Use 'min_improvement' instead. "
                "Support will be removed in v2.0.0",
                DeprecationWarning,
                stacklevel=2
            )
            data["min_improvement"] = data.pop("improvement_threshold")

        if "confidence_threshold" in data and "min_confidence" not in data:
            warnings.warn(
                "Field 'confidence_threshold' is deprecated. "
                "Use 'min_confidence' instead. "
                "Support will be removed in v2.0.0",
                DeprecationWarning,
                stacklevel=2
            )
            data["min_confidence"] = data.pop("confidence_threshold")

        super().__init__(**data)
```

---

## 5. System-Wide Naming Conventions

### 5.1 Threshold Naming Patterns

| Pattern | Meaning | Examples |
|---------|---------|----------|
| `min_*` | Minimum value (lower bound) | `min_confidence`, `min_baseline_score`, `min_improvement` |
| `max_*` | Maximum value (upper bound) | `max_iterations`, `max_attempts`, `max_prompt_length` |
| `*_threshold` | **Avoid** (ambiguous: min or max?) | Use `min_*` or `max_*` instead |
| `enable_*` | Boolean feature flag | `enable_version_tracking`, `enable_metrics_feedback` |
| `*_rate` | Ratio (0.0-1.0) | `success_rate`, `failure_rate` |

### 5.2 Cross-Module Consistency Check

```python
# Config Module
class ValidationConfig(BaseModel):
    min_response_length: int = 10        # ✅ Consistent
    max_response_length: int = 5000      # ✅ Consistent

# Executor Module
class RetryPolicy(BaseModel):
    max_attempts: int = 3                # ✅ Consistent
    backoff_seconds: float = 2.0         # ✅ Descriptive

class ExecutionPolicy(BaseModel):
    concurrency: int = 5                 # ✅ Clear meaning
    batch_size: int = 10                 # ✅ Clear meaning

# Optimizer Module (Standardized)
class OptimizationConfig(BaseModel):
    min_baseline_score: float = 80.0     # ✅ Consistent
    min_improvement: float = 5.0         # ✅ Consistent
    min_confidence: float = 0.6          # ✅ Consistent
    max_iterations: int = 3              # ✅ Consistent
```

### 5.3 Recommended Naming Guidelines

1. **Use `min_*` for lower bounds**
   - Example: `min_confidence`, `min_score`, `min_length`

2. **Use `max_*` for upper bounds**
   - Example: `max_attempts`, `max_iterations`, `max_size`

3. **Avoid `*_threshold`** unless semantically meaningful
   - Exception: `threshold` alone can mean "decision boundary" (e.g., classification threshold)
   - Prefer explicit `min_*` or `max_*`

4. **Use `enable_*` for boolean feature flags**
   - Example: `enable_caching`, `enable_logging`, `enable_retry`

5. **Use descriptive units**
   - `timeout_seconds` (not `timeout`)
   - `backoff_seconds` (not `backoff`)
   - `batch_size` (not `batch`)

6. **Use snake_case for all fields**
   - Consistent with Python conventions

7. **Use singular for counts, plural for lists**
   - `max_attempt` ❌ → `max_attempts` ✅
   - `strategy` ✅ (singular when selecting one)
   - `strategies` ✅ (plural when list)

---

## 6. Validation and Testing

### 6.1 Configuration Validation Tests

```python
# tests/optimizer/test_optimization_config.py

import pytest
from src.optimizer.models import OptimizationConfig


def test_min_baseline_score_range():
    """Test min_baseline_score is within valid range."""
    # Valid
    config = OptimizationConfig(min_baseline_score=50.0)
    assert config.min_baseline_score == 50.0

    # Invalid: too low
    with pytest.raises(ValueError, match="greater than or equal to 0.0"):
        OptimizationConfig(min_baseline_score=-1.0)

    # Invalid: too high
    with pytest.raises(ValueError, match="less than or equal to 100.0"):
        OptimizationConfig(min_baseline_score=101.0)


def test_scoring_weights_sum_to_one():
    """Test scoring weights must sum to 1.0."""
    # Valid
    config = OptimizationConfig(
        scoring_weights={"clarity": 0.7, "efficiency": 0.3}
    )
    assert sum(config.scoring_weights.values()) == 1.0

    # Invalid: doesn't sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        OptimizationConfig(
            scoring_weights={"clarity": 0.5, "efficiency": 0.3}
        )


def test_deprecated_fields_still_work():
    """Test backward compatibility with deprecated fields."""
    with pytest.warns(DeprecationWarning, match="improvement_threshold"):
        config = OptimizationConfig(improvement_threshold=10.0)
        assert config.min_improvement == 10.0

    with pytest.warns(DeprecationWarning, match="confidence_threshold"):
        config = OptimizationConfig(confidence_threshold=0.8)
        assert config.min_confidence == 0.8


def test_storage_backend_validation():
    """Test storage backend must be valid."""
    # Valid
    config = OptimizationConfig(storage_backend="filesystem")
    assert config.storage_backend == "filesystem"

    # Invalid
    with pytest.raises(ValueError, match="Invalid storage backend"):
        OptimizationConfig(storage_backend="invalid")
```

### 6.2 Integration Test with OptimizerService

```python
def test_optimizer_respects_config_thresholds():
    """Test OptimizerService uses config thresholds correctly."""

    config = OptimizationConfig(
        min_baseline_score=70.0,
        min_improvement=8.0,
        min_confidence=0.7
    )

    service = OptimizerService(catalog, env, config=config)

    # Low-quality prompt (score = 65)
    prompt = create_test_prompt()
    analysis = PromptAnalysis(
        prompt_id=prompt.id,
        overall_score=65.0,  # Below min_baseline_score
        # ...
    )

    # Should trigger optimization
    assert service.should_optimize(analysis) == True

    # High-quality prompt (score = 85)
    analysis.overall_score = 85.0  # Above min_baseline_score
    assert service.should_optimize(analysis) == False

    # Test acceptance criteria
    result = OptimizationResult(
        prompt_id=prompt.id,
        improvement_score=10.0,  # Above min_improvement
        confidence=0.8,          # Above min_confidence
        # ...
    )

    assert service.should_accept(result) == True

    # Test rejection
    result.improvement_score = 5.0  # Below min_improvement
    assert service.should_accept(result) == False
```

---

## 7. Documentation Updates Required

### 7.1 API Documentation

```python
# Update docstrings
class OptimizationConfig(BaseModel):
    """Optimizer module configuration.

    This configuration controls when and how prompt optimizations are performed.

    Thresholds:
        min_baseline_score: Prompts scoring below this value will be optimized.
                           Range: 0-100. Default: 80.0.

        min_improvement: Minimum score improvement required to accept an optimization.
                        Range: 0-100. Default: 5.0.

        min_confidence: Minimum confidence level required to accept an optimization.
                       Range: 0.0-1.0. Default: 0.6.

    Limits:
        max_iterations: Maximum number of optimization attempts per prompt.
                       Range: 1-10. Default: 3.

    Strategy:
        strategies: List of optimization strategies to use (in priority order).
                   Options: ["auto", "clarity_focus", "efficiency_focus", "structure_focus"]
                   Default: ["auto"]

    Storage:
        storage_backend: Storage backend type.
                        Options: "memory", "filesystem", "database"
                        Default: "memory"

    Example:
        >>> config = OptimizationConfig(
        ...     min_baseline_score=75.0,
        ...     min_improvement=10.0,
        ...     max_iterations=5
        ... )
        >>> service = OptimizerService(catalog, env, config=config)
    """
```

### 7.2 User Guide Updates

```markdown
# Optimizer Configuration Guide

## Threshold Configuration

### min_baseline_score (default: 80.0)

Controls which prompts are selected for optimization.

- **Range:** 0-100
- **Meaning:** Prompts with quality scores below this value will be optimized
- **Example:** Setting to 75.0 means only prompts scoring < 75/100 are optimized

**Use Cases:**
- **Conservative (85-90):** Only optimize clearly low-quality prompts
- **Moderate (75-85):** Optimize prompts with room for improvement
- **Aggressive (60-75):** Optimize most prompts to maximize quality

### min_improvement (default: 5.0)

Controls whether an optimization is accepted or rejected.

- **Range:** 0-100
- **Meaning:** Optimization must improve score by at least this amount
- **Example:** Setting to 10.0 means only accept optimizations improving by ≥10 points

**Use Cases:**
- **Low (3-5):** Accept most optimizations (risk of minor changes)
- **Medium (5-10):** Balance between quality and quantity
- **High (10+):** Only accept significant improvements

### min_confidence (default: 0.6)

Controls trust in optimization quality.

- **Range:** 0.0-1.0
- **Meaning:** Optimization must have at least this confidence level
- **Example:** Setting to 0.8 means only accept 80%+ confidence optimizations

**Use Cases:**
- **Low (0.5-0.6):** Accept more optimizations (risk of false positives)
- **Medium (0.6-0.75):** Balanced risk tolerance
- **High (0.75-0.9):** Conservative, high-confidence only
```

---

## 8. Rollout Plan

### Phase 1: Code Changes (Week 1)
- [ ] Update `OptimizationConfig` model with new fields
- [ ] Add backward compatibility for deprecated fields
- [ ] Update `OptimizerService` to use new field names
- [ ] Write migration script

### Phase 2: Testing (Week 1)
- [ ] Write validation tests for new fields
- [ ] Write integration tests
- [ ] Test migration script on sample configs

### Phase 3: Documentation (Week 2)
- [ ] Update API documentation
- [ ] Update user guide
- [ ] Write migration guide
- [ ] Update example YAML files

### Phase 4: Deployment (Week 2)
- [ ] Deploy with deprecation warnings
- [ ] Monitor for usage of deprecated fields
- [ ] Communicate migration timeline to users

### Phase 5: Cleanup (Week 4)
- [ ] Remove deprecated fields in v2.0.0
- [ ] Remove backward compatibility code

---

## 9. Summary

### Key Changes

| Old Field | New Field | Migration |
|-----------|-----------|-----------|
| `improvement_threshold` | `min_improvement` | Auto-migrate with warning |
| `confidence_threshold` | `min_confidence` | Auto-migrate with warning |
| (hardcoded 80.0) | `min_baseline_score` | New configurable field |

### Benefits

1. **Consistency**: All modules use `min_*` / `max_*` pattern
2. **Clarity**: Field names explicitly state lower/upper bound
3. **Configurability**: Previously hardcoded values now configurable
4. **Maintainability**: Easier to understand and extend

### Compatibility

- **Backward compatible** via deprecation warnings
- **Migration script** provided
- **Gradual rollout** over 4 weeks
- **Clean removal** in v2.0.0

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-17
**Status:** Ready for Review
