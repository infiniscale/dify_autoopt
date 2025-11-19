# LLM Integration - Usage Guide

**Module**: src/optimizer
**Status**: ✅ Production Ready
**Last Updated**: 2025-11-19

> **Document Purpose**: Practical usage guide for LLM configuration and optimization in the Optimizer module.
>
> **Source Documents Merged**:
> - LLM_CONFIG_USAGE.md
> - optimizer_service_llm_usage.md

---

## Quick Start

### 1. Install Dependencies

```bash
pip install openai>=1.0.0
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run LLM Optimization

```python
from src.optimizer import OptimizerService
from src.optimizer.interfaces.llm_providers import OpenAIClient
from src.optimizer.config import LLMConfig, LLMProvider

# Configure
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4-turbo-preview",
    api_key_env="OPENAI_API_KEY"
)

# Create service
llm_client = OpenAIClient(config)
service = OptimizerService(catalog=workflow_catalog, llm_client=llm_client)

# Optimize
patches = service.run_optimization_cycle("workflow_001", strategy="llm_guided")
```

---

## Configuration

### Programmatic Configuration

```python
from src.optimizer.config import LLMConfig, LLMProvider

# Basic configuration
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    api_key_env="OPENAI_API_KEY",
    model="gpt-4-turbo-preview",
    temperature=0.7,
    max_tokens=2000,
    enable_cache=True
)

# Validate configuration
is_valid = config.validate_config()
print(f"Configuration valid: {is_valid}")

# Get display config (API key masked)
display = config.get_display_config()
print(f"Provider: {display['provider']}")
print(f"Model: {display['model']}")
print(f"API Key: {display['api_key']}")  # Shows: ***MASKED***
```

### YAML Configuration

**File: `config/llm.yaml`**:
```yaml
llm:
  provider: "openai"
  api_key_env: "OPENAI_API_KEY"
  model: "gpt-4-turbo-preview"
  temperature: 0.7
  max_tokens: 2000
  enable_cache: true
  cache_ttl: 3600
  timeout_seconds: 10
  max_retries: 3
  cost_limits:
    max_cost_per_request: 0.10
    max_cost_per_day: 10.0
```

**Load from YAML**:
```python
from src.optimizer.config import LLMConfigLoader

config = LLMConfigLoader.from_yaml("config/llm.yaml")
```

### Environment Variables

```bash
# Required
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4-turbo-preview"
export OPENAI_API_KEY="sk-..."

# Optional
export LLM_TEMPERATURE="0.7"
export LLM_MAX_TOKENS="2000"
export LLM_ENABLE_CACHE="true"
```

**Load from environment**:
```python
from src.optimizer.config import LLMConfigLoader

config = LLMConfigLoader.from_env()
```

### Auto-Load with Fallback

```python
from src.optimizer.config import LLMConfigLoader

# Try env vars → YAML → default (STUB)
config = LLMConfigLoader.auto_load()

if config.provider == LLMProvider.STUB:
    print("Warning: Using stub client (rule-based only)")
```

---

## Usage Examples

### Example 1: Basic LLM Optimization

```python
from src.optimizer import OptimizerService
from src.optimizer.interfaces.llm_providers import OpenAIClient
from src.optimizer.config import LLMConfig, LLMProvider

# Configure LLM
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4-turbo-preview",
    api_key_env="OPENAI_API_KEY",
    enable_cache=True
)

# Create client and service
llm_client = OpenAIClient(config)
service = OptimizerService(
    catalog=workflow_catalog,
    llm_client=llm_client
)

# Optimize with LLM
patches = service.run_optimization_cycle(
    workflow_id="wf_001",
    strategy="llm_guided"
)

# Print results
for patch in patches:
    print(f"Node: {patch.node_id}")
    print(f"Improvement: {patch.metadata.get('improvement', 0):.1f} points")
    print(f"Strategy: {patch.metadata.get('strategy', 'N/A')}")
    print("---")
```

### Example 2: Multi-Strategy Optimization

```python
from src.optimizer import OptimizationConfig, OptimizationStrategy

# Configure multiple strategies
config = OptimizationConfig(
    strategies=[
        OptimizationStrategy.LLM_GUIDED,     # Try full rewrite first
        OptimizationStrategy.LLM_CLARITY,    # Then clarity focus
        OptimizationStrategy.HYBRID          # Finally hybrid
    ],
    score_threshold=80.0,   # Optimize prompts below 80/100
    min_confidence=0.7,     # Accept only 70%+ confidence
    max_iterations=3        # Up to 3 refinements per strategy
)

# Run optimization
patches = service.run_optimization_cycle(
    workflow_id="wf_001",
    config=config
)

# Best result across all strategies is selected
```

### Example 3: Cost-Controlled Optimization

```python
# Configure with cost limits
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4-turbo-preview",
    api_key_env="OPENAI_API_KEY",
    cost_limits={
        "max_cost_per_request": 0.10,  # Max $0.10 per optimization
        "max_cost_per_day": 10.0       # Max $10.00 per day
    }
)

llm_client = OpenAIClient(config)

# Track costs
print(f"Total tokens: {llm_client.total_tokens_used}")
print(f"Total cost: ${llm_client.total_cost:.4f}")
```

### Example 4: Fallback to Rule-Based

```python
# Service without LLM client
service_rule_only = OptimizerService(catalog=workflow_catalog)

# Request LLM strategy - falls back automatically
patches = service_rule_only.run_optimization_cycle(
    workflow_id="wf_001",
    strategy="llm_guided"  # Falls back to structure_focus
)

# Result uses rule-based strategy
for patch in patches:
    strategy_type = patch.metadata.get("strategy_type", "rule")
    print(f"Strategy type: {strategy_type}")  # "rule"
```

### Example 5: Adaptive Strategy Selection

```python
from src.optimizer import PromptAnalyzer

def optimize_adaptively(prompt, service, analyzer):
    """Select best strategy based on prompt quality"""
    analysis = analyzer.analyze_prompt(prompt)

    # Poor quality → full LLM rewrite
    if analysis.overall_score < 60:
        strategy = "llm_guided"

    # Unclear → clarity focus
    elif analysis.clarity_score < 70:
        strategy = "llm_clarity"

    # Verbose → efficiency focus
    elif analysis.efficiency_score < 70:
        strategy = "llm_efficiency"

    # Good → hybrid for polish
    else:
        strategy = "hybrid"

    return service.run_optimization_cycle(
        workflow_id=prompt.workflow_id,
        strategy=strategy
    )
```

### Example 6: Batch Optimization with Progress

```python
from src.config import WorkflowCatalog

# Load multiple workflows
catalog = WorkflowCatalog.from_yaml("workflows/")
workflow_ids = ["wf_001", "wf_002", "wf_003"]

all_patches = []
total_cost = 0.0

for workflow_id in workflow_ids:
    print(f"Optimizing {workflow_id}...")

    patches = service.run_optimization_cycle(
        workflow_id=workflow_id,
        strategy="llm_guided"
    )

    all_patches.extend(patches)
    total_cost = llm_client.total_cost

    print(f"  Patches: {len(patches)}")
    print(f"  Running cost: ${total_cost:.4f}")

print(f"\nTotal patches: {len(all_patches)}")
print(f"Total cost: ${total_cost:.4f}")
```

---

## Strategy Selection Guide

### When to Use Each Strategy

| Strategy | Best For | Latency | Cost | Quality |
|----------|----------|---------|------|---------|
| **clarity_focus** (rule) | Minor clarity issues | <5ms | $0 | Medium |
| **efficiency_focus** (rule) | Verbose prompts | <5ms | $0 | Medium |
| **structure_focus** (rule) | Poor organization | <5ms | $0 | Medium |
| **llm_guided** | Major rewrites | 1-3s | $0.02-0.03 | High |
| **llm_clarity** | Vague instructions | 1-2.5s | $0.01-0.03 | High |
| **llm_efficiency** | Redundant content | 1-2.5s | $0.01-0.03 | High |
| **hybrid** | Production balance | 1-3s | $0.01-0.03 | High |

### Strategy Selection Matrix

| Prompt Quality | Recommended Strategy |
|---------------|---------------------|
| Very poor (<50) | llm_guided |
| Poor (50-65) | llm_guided |
| Below average (65-75) | llm_clarity or llm_efficiency |
| Average (75-85) | hybrid |
| Good (85-95) | Rule-based (clarity/efficiency) |
| Excellent (95+) | Skip optimization |

---

## Caching

### Enable Caching

```python
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4-turbo-preview",
    api_key_env="OPENAI_API_KEY",
    enable_cache=True,     # Enable caching
    cache_ttl=3600         # 1 hour TTL
)
```

### Cache Behavior

- **Cache key**: Hash of (prompt_text + optimization_goal)
- **Hit**: Returns cached response instantly
- **Miss**: Calls API and caches result
- **TTL**: Entries expire after cache_ttl seconds

### Expected Cache Hit Rate

| Use Case | Expected Hit Rate | Cost Reduction |
|----------|-------------------|----------------|
| Development | 80-90% | 80-90% |
| Testing | 90-95% | 90-95% |
| Production (varied prompts) | 60-70% | 60-70% |
| Production (repeated prompts) | 80-90% | 80-90% |

---

## Error Handling

### Automatic Retry

```python
config = LLMConfig(
    max_retries=3,        # Retry up to 3 times
    timeout_seconds=10    # 10 second timeout
)

# Retries automatically for:
# - RateLimitError (with exponential backoff)
# - APIError (transient errors)
# - Timeout (network issues)
```

### Handle Errors

```python
from src.optimizer.exceptions import OptimizationFailedError, InvalidStrategyError

try:
    patches = service.run_optimization_cycle(
        workflow_id="wf_001",
        strategy="llm_guided"
    )
except OptimizationFailedError as e:
    print(f"LLM optimization failed: {e}")
    # Fall back to rule-based
    patches = service.run_optimization_cycle(
        workflow_id="wf_001",
        strategy="clarity_focus"  # Safe fallback
    )
except InvalidStrategyError as e:
    print(f"Invalid strategy: {e}")
    print(f"Valid strategies: {e.valid_strategies}")
```

### Graceful Degradation

```python
# Service automatically falls back when LLM unavailable
def optimize_with_fallback(service, workflow_id, preferred_strategy):
    """Optimize with automatic fallback"""

    # Try preferred strategy (may be LLM)
    patches = service.run_optimization_cycle(
        workflow_id=workflow_id,
        strategy=preferred_strategy
    )

    # Check if fallback occurred
    for patch in patches:
        strategy_type = patch.metadata.get("strategy_type")
        if strategy_type == "rule":
            print(f"Note: {preferred_strategy} fell back to rule-based")

    return patches
```

---

## Monitoring

### Track Token Usage

```python
# After optimization
print(f"Tokens used: {llm_client.total_tokens_used}")
print(f"Total cost: ${llm_client.total_cost:.4f}")
```

### Monitor Cache Performance

```python
# Get cache statistics (if using custom cache)
if llm_client._cache:
    hit_rate = llm_client._cache.get_hit_rate()
    print(f"Cache hit rate: {hit_rate:.1%}")
    print(f"Cache size: {len(llm_client._cache.cache)}")
```

### Log Optimization Results

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimizer")

patches = service.run_optimization_cycle(
    workflow_id="wf_001",
    strategy="llm_guided"
)

for patch in patches:
    logger.info(
        f"Optimized {patch.node_id}: "
        f"+{patch.metadata.get('improvement', 0):.1f} points, "
        f"strategy={patch.metadata.get('strategy', 'N/A')}"
    )
```

---

## Best Practices

### 1. Security

**Never hardcode API keys**:
```python
# WRONG
config = LLMConfig(api_key="sk-...")

# RIGHT
config = LLMConfig(api_key_env="OPENAI_API_KEY")
```

**Use environment-specific configs**:
```bash
# Development
export OPENAI_API_KEY="sk-dev-..."

# Production
export OPENAI_API_KEY="sk-prod-..."
```

### 2. Cost Control

**Set limits**:
```python
config = LLMConfig(
    cost_limits={
        "max_cost_per_request": 0.10,
        "max_cost_per_day": 10.0
    }
)
```

**Enable caching**:
```python
config = LLMConfig(enable_cache=True, cache_ttl=3600)
```

**Use appropriate models**:
- GPT-4 Turbo: Best quality/cost ratio
- GPT-3.5 Turbo: Faster, cheaper, lower quality

### 3. Performance

**Use caching** for repeated optimizations

**Set timeouts** to prevent hanging:
```python
config = LLMConfig(timeout_seconds=10)
```

**Use rule-based** for simple improvements

### 4. Testing

**Use StubLLMClient** for deterministic tests:
```python
from src.optimizer.interfaces.llm_client import StubLLMClient

stub_client = StubLLMClient(response_content="Optimized prompt")
service = OptimizerService(catalog, llm_client=stub_client)
```

**Mock API calls** in unit tests:
```python
from unittest.mock import patch, MagicMock

with patch("openai.OpenAI") as mock_client:
    mock_client.return_value.chat.completions.create.return_value = MockResponse()
    # Test code
```

---

## Troubleshooting

### "Authentication failed"

**Cause**: Invalid or missing API key

**Solution**:
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### "Rate limit exceeded"

**Cause**: Too many API calls

**Solution**:
- Enable caching
- Reduce max_iterations
- Add delays between calls
- Upgrade OpenAI plan

### "Timeout"

**Cause**: Network issues or slow response

**Solution**:
```python
config = LLMConfig(
    timeout_seconds=30,  # Increase timeout
    max_retries=5        # More retries
)
```

### "Fallback to rule-based" (unexpected)

**Cause**: LLM client not configured properly

**Solution**:
```python
# Verify LLM client is available
if llm_client.is_available:
    print("LLM client is available")
else:
    print("LLM client is NOT available - check configuration")

# Check logs for warnings
# "LLM strategy requested but no LLM client available..."
```

### High costs

**Cause**: Uncached repeated calls, too many iterations

**Solution**:
- Enable caching
- Reduce max_iterations
- Set cost limits
- Use GPT-3.5 Turbo for less critical prompts

---

## Document Metadata

**Version**: 1.0
**Status**: Complete
**Audience**: Developers, users

---

**End of LLM Usage Guide**
