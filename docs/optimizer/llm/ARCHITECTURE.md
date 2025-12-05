# LLM Integration - Architecture

**Module**: src/optimizer
**Status**: ✅ Production Ready
**Last Updated**: 2025-11-19

> **Document Purpose**: Architecture design for LLM integration in the Optimizer module, covering client interface, provider implementations, and design decisions.
>
> **Source Documents Merged**:
> - LLM_INTEGRATION_ARCHITECTURE.md
> - LLM_INTEGRATION_SUMMARY.md
> - LLM_CLIENT_ARCHITECTURE_DECISION.md

---

## Executive Summary

The optimizer module integrates real LLM capabilities (OpenAI GPT-4, with support for future providers) to enable intelligent, context-aware prompt optimization. The design follows:

- **Interface-First Design**: Abstract LLMClient interface for provider-agnostic integration
- **Progressive Enhancement**: LLM enhancement without breaking existing functionality
- **Cost Control**: Built-in token tracking, caching, and rate limiting
- **Testability**: Mock-friendly design for deterministic testing

---

## Architecture Overview

### Current vs Target Architecture

**MVP (Rule-Based)**:
```
OptimizerService → OptimizationEngine → Rule-based heuristics
                                      → StubLLMClient (placeholder)
```

**Production (LLM-Enabled)**:
```
OptimizerService → OptimizationEngine → Rule-based heuristics
                                      → LLMClient interface
                                          ├─ OpenAIClient (implemented)
                                          ├─ AnthropicClient (future)
                                          └─ LocalClient (future)
```

### Component Diagram

```
┌─────────────────────────────────────────────────┐
│                 OptimizerService                │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐    ┌──────────────────────┐   │
│  │PromptAnalyzer│   │  OptimizationEngine  │   │
│  └─────────────┘    └──────────────────────┘   │
│                              │                  │
│                     ┌────────┴────────┐         │
│                     │                 │         │
│             ┌───────┴───────┐ ┌───────┴───────┐ │
│             │ Rule Strategies│ │ LLM Strategies│ │
│             └───────────────┘ └───────┬───────┘ │
│                                       │         │
└───────────────────────────────────────┼─────────┘
                                        │
                               ┌────────┴────────┐
                               │  LLMClient      │
                               │  (interface)    │
                               └────────┬────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     │                  │                  │
              ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
              │ OpenAIClient │   │ StubClient  │   │ Future...   │
              └─────────────┘   └─────────────┘   └─────────────┘
```

---

## LLMClient Interface

### Interface Definition

```python
class LLMClient(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def optimize_prompt(
        self,
        prompt_text: str,
        analysis_context: Optional[Dict[str, Any]] = None,
        optimization_goal: Optional[str] = None
    ) -> LLMResponse:
        """
        Optimize a prompt using LLM

        Args:
            prompt_text: Original prompt text to optimize
            analysis_context: Context from PromptAnalyzer (scores, issues)
            optimization_goal: Specific optimization target

        Returns:
            LLMResponse with optimized text, confidence, token usage
        """
        pass

    @abstractmethod
    def analyze_prompt(
        self,
        prompt_text: str
    ) -> LLMResponse:
        """
        Analyze prompt quality using LLM

        Args:
            prompt_text: Prompt to analyze

        Returns:
            LLMResponse with analysis results
        """
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass
```

### LLMResponse Model

```python
class LLMResponse(BaseModel):
    """Response from LLM client"""

    success: bool = True
    content: str = ""
    confidence: float = 0.8
    token_usage: Dict[str, int] = {}
    cost_estimate: float = 0.0
    metadata: Dict[str, Any] = {}
    error_message: Optional[str] = None
```

---

## Provider Implementations

### OpenAIClient

**File**: `src/optimizer/interfaces/llm_providers/openai_client.py`

**Features**:
- GPT-4 Turbo integration
- Response caching with configurable TTL
- Retry with exponential backoff
- Token usage tracking
- Cost estimation

**Implementation**:
```python
class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=config.get_api_key())
        self._cache = {} if config.enable_cache else None

    def optimize_prompt(
        self,
        prompt_text: str,
        analysis_context: Optional[Dict] = None,
        optimization_goal: Optional[str] = None
    ) -> LLMResponse:
        # Check cache
        cache_key = self._get_cache_key(prompt_text, optimization_goal)
        if self._cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Build system prompt
        system_prompt = self._build_optimization_prompt(
            analysis_context, optimization_goal
        )

        # Call OpenAI API with retry
        response = self._call_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # Build response
        result = LLMResponse(
            success=True,
            content=response.choices[0].message.content,
            confidence=self._estimate_confidence(response),
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            cost_estimate=self._calculate_cost(response.usage)
        )

        # Cache result
        if self._cache:
            self._cache[cache_key] = result

        return result
```

### StubLLMClient

**Purpose**: Testing and rule-based fallback

```python
class StubLLMClient(LLMClient):
    """Stub client for testing and deterministic behavior"""

    def __init__(self, response_content: str = "Optimized prompt"):
        self._response_content = response_content

    def optimize_prompt(self, prompt_text, **kwargs) -> LLMResponse:
        return LLMResponse(
            success=True,
            content=self._response_content,
            confidence=0.8,
            token_usage={"total_tokens": 0},
            cost_estimate=0.0
        )

    @property
    def is_available(self) -> bool:
        return True
```

---

## Design Decisions

### Decision 1: Interface-Based Design

**Choice**: Abstract LLMClient interface instead of direct API calls

**Rationale**:
- Enables provider swapping (OpenAI ↔ Claude ↔ Local)
- Facilitates testing with mocks
- Supports graceful degradation to rule-based
- Future-proofs for new providers

### Decision 2: Confidence Scoring

**Choice**: LLM provides confidence estimate (0.0-1.0)

**Rationale**:
- Enables quality filtering (min_confidence threshold)
- Supports strategy comparison
- Helps identify uncertain optimizations

### Decision 3: Response Caching

**Choice**: Optional in-memory caching with configurable TTL

**Rationale**:
- Reduces API costs (identical prompts cached)
- Improves latency (cache hits ~0ms vs ~2s)
- Configurable for testing vs production

### Decision 4: Fallback to Rule-Based

**Choice**: Automatic fallback when LLM unavailable or fails

**Rationale**:
- Ensures optimization always produces results
- Handles API failures gracefully
- Supports offline operation

---

## Optimization Strategies

### LLM-Powered Strategies

| Strategy | Description | Use Case | Fallback |
|----------|-------------|----------|----------|
| `LLM_GUIDED` | Full semantic rewrite | Complex, unclear prompts | structure_focus |
| `LLM_CLARITY` | Clarity-focused optimization | Vague instructions | clarity_focus |
| `LLM_EFFICIENCY` | Compression optimization | Verbose prompts | efficiency_focus |
| `HYBRID` | LLM + rule-based cleanup | Production balance | clarity_focus |

### Strategy Selection Flow

```python
def optimize(self, prompt, strategy):
    if strategy in LLM_STRATEGIES:
        if self.llm_client and self.llm_client.is_available:
            return self._apply_llm_strategy(prompt, strategy)
        else:
            # Fallback to rule-based
            fallback = self._get_fallback_strategy(strategy)
            return self._apply_rule_strategy(prompt, fallback)
    else:
        return self._apply_rule_strategy(prompt, strategy)
```

---

## Cost Control

### Token Tracking

```python
class OpenAIClient:
    def __init__(self, config):
        self.total_tokens_used = 0
        self.total_cost = 0.0

    def _track_usage(self, response):
        tokens = response.usage.total_tokens
        cost = self._calculate_cost(response.usage)

        self.total_tokens_used += tokens
        self.total_cost += cost

        # Check limits
        if cost > self.config.cost_limits.get("max_cost_per_request", float("inf")):
            raise CostLimitExceeded(f"Single request cost ${cost} exceeds limit")
```

### Cost Estimation

| Model | Input Cost | Output Cost | Avg Optimization |
|-------|-----------|-------------|------------------|
| GPT-4 Turbo | $10/1M | $30/1M | $0.015-0.033 |
| GPT-4 | $30/1M | $60/1M | $0.045-0.099 |
| GPT-3.5 Turbo | $0.50/1M | $1.50/1M | $0.001-0.003 |

**Recommendation**: Use GPT-4 Turbo for best quality/cost ratio

### Caching Strategy

```python
# Enable caching with 1-hour TTL
config = LLMConfig(
    enable_cache=True,
    cache_ttl=3600  # 1 hour
)

# Expected cache hit rate: 60-80% (repeated prompts)
# Cost reduction: 60-80%
```

---

## Error Handling

### Retry Mechanism

```python
def _call_with_retry(self, **kwargs):
    max_retries = self.config.max_retries  # Default: 3
    base_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            return self.client.chat.completions.create(**kwargs)
        except RateLimitError:
            # Exponential backoff
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay)

    raise OptimizationFailedError("Max retries exceeded")
```

### Error Types

| Error | Handling | Fallback |
|-------|----------|----------|
| RateLimitError | Exponential backoff | Retry |
| AuthenticationError | Raise immediately | Rule-based |
| APIError | Retry with backoff | Rule-based |
| Timeout | Retry once | Rule-based |
| NetworkError | Retry with backoff | Rule-based |

---

## Performance Characteristics

| Metric | LLM (no cache) | LLM (cached) | Rule-Based |
|--------|---------------|--------------|------------|
| Latency | 1-3s | <10ms | <5ms |
| Cost | $0.01-0.03 | $0 | $0 |
| Quality | High | High | Medium |
| Reliability | 99%+ | 100% | 100% |

### Recommendations

1. **Use caching** for repeated optimization runs
2. **Set reasonable timeout** (10s recommended)
3. **Enable retries** (3 recommended)
4. **Monitor costs** with daily limits
5. **Fall back gracefully** to rule-based

---

## Testing Strategy

### Unit Tests

```python
def test_openai_client_optimize():
    """Test optimization with mocked OpenAI API"""
    with patch("openai.OpenAI") as mock_client:
        mock_response = MockResponse(
            content="Optimized text",
            usage={"total_tokens": 100}
        )
        mock_client.return_value.chat.completions.create.return_value = mock_response

        client = OpenAIClient(config)
        result = client.optimize_prompt("original prompt")

        assert result.success
        assert result.content == "Optimized text"
```

### Integration Tests

```python
@pytest.mark.integration
def test_openai_live_api():
    """Test with real OpenAI API (requires API key)"""
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key_env="OPENAI_API_KEY"
    )

    client = OpenAIClient(config)
    result = client.optimize_prompt("summarize the document")

    assert result.success
    assert len(result.content) > 0
    assert result.token_usage["total_tokens"] > 0
```

---

## Future Providers

### Anthropic Claude

```python
class AnthropicClient(LLMClient):
    """Future: Claude integration"""

    def __init__(self, config):
        self.client = anthropic.Anthropic(api_key=config.get_api_key())

    def optimize_prompt(self, prompt_text, **kwargs):
        response = self.client.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt_text}]
        )
        return LLMResponse(
            content=response.content[0].text,
            # ...
        )
```

### Local Models (Ollama)

```python
class OllamaClient(LLMClient):
    """Future: Local model integration"""

    def __init__(self, config):
        self.base_url = config.endpoint or "http://localhost:11434"

    def optimize_prompt(self, prompt_text, **kwargs):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.config.model, "prompt": prompt_text}
        )
        return LLMResponse(content=response.json()["response"])
```

---

## Document Metadata

**Version**: 1.0
**Status**: Complete
**Audience**: Architects, developers

---

**End of LLM Architecture Document**
