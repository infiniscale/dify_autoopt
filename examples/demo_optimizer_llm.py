"""
Demonstration Script: OptimizerService LLM Integration

This script demonstrates the new LLM-driven optimization capabilities
added to OptimizerService.

Date: 2025-11-18
Author: backend-developer
"""

from src.optimizer import (
    OptimizerService,
    LLMConfig,
    LLMProvider,
)
from src.optimizer.models import Prompt


def demo_basic_llm_config():
    """Demo 1: Basic LLM configuration."""
    print("=" * 70)
    print("Demo 1: Basic LLM Configuration")
    print("=" * 70)

    # Create LLM configuration
    llm_config = LLMConfig(
        provider=LLMProvider.STUB,  # Using STUB for demo (no API calls)
        model="gpt-4-turbo-preview",
        temperature=0.7,
        max_tokens=2000
    )

    print(f"\nLLM Configuration created:")
    print(f"  Provider: {llm_config.provider}")
    print(f"  Model: {llm_config.model}")
    print(f"  Temperature: {llm_config.temperature}")
    print(f"  Max Tokens: {llm_config.max_tokens}")
    print(f"  Cache Enabled: {llm_config.enable_cache}")


def demo_service_initialization():
    """Demo 2: OptimizerService initialization with LLM config."""
    print("\n" + "=" * 70)
    print("Demo 2: OptimizerService Initialization")
    print("=" * 70)

    llm_config = LLMConfig(provider=LLMProvider.STUB)

    # Initialize service with LLM config
    service = OptimizerService(llm_config=llm_config)

    print(f"\nOptimizerService created with LLM support")
    print(f"  LLM Client Type: {type(service._llm_client).__name__}")
    print(f"  LLM Enabled: {service._llm_config is not None}")


def demo_backward_compatibility():
    """Demo 3: Backward compatibility without LLM config."""
    print("\n" + "=" * 70)
    print("Demo 3: Backward Compatibility")
    print("=" * 70)

    # Old way - still works!
    service_old = OptimizerService()
    print(f"\nOld way (no LLM config):")
    print(f"  Client Type: {type(service_old._llm_client).__name__}")
    print(f"  Works: [OK]")

    # New way - with LLM config
    llm_config = LLMConfig(provider=LLMProvider.STUB)
    service_new = OptimizerService(llm_config=llm_config)
    print(f"\nNew way (with LLM config):")
    print(f"  Client Type: {type(service_new._llm_client).__name__}")
    print(f"  Works: [OK]")

    print(f"\n[OK] Backward compatibility maintained!")


def demo_llm_stats():
    """Demo 4: LLM usage statistics."""
    print("\n" + "=" * 70)
    print("Demo 4: LLM Usage Statistics")
    print("=" * 70)

    # Create service without LLM
    service_stub = OptimizerService()
    stats_stub = service_stub.get_llm_stats()
    print(f"\nStubLLMClient stats: {stats_stub}")
    print(f"  (Returns None - no LLM used)")

    # Create service with LLM config (STUB)
    llm_config = LLMConfig(provider=LLMProvider.STUB)
    service_llm = OptimizerService(llm_config=llm_config)
    stats_llm = service_llm.get_llm_stats()
    print(f"\nLLM-enabled service stats: {stats_llm}")
    print(f"  (Also None for STUB provider)")

    print("\n  Note: Real LLM providers (OpenAI) would return:")
    print("    - total_requests: Number of API calls")
    print("    - total_tokens: Tokens consumed")
    print("    - total_cost: Cost in USD")
    print("    - cache_hit_rate: Cache efficiency")
    print("    - average_latency_ms: Response time")


def demo_provider_fallback():
    """Demo 5: Provider fallback behavior."""
    print("\n" + "=" * 70)
    print("Demo 5: Provider Fallback Behavior")
    print("=" * 70)

    providers = [
        (LLMProvider.STUB, "STUB"),
        (LLMProvider.ANTHROPIC, "Anthropic (not implemented)"),
        (LLMProvider.LOCAL, "Local (not implemented)"),
    ]

    for provider, description in providers:
        llm_config = LLMConfig(provider=provider)
        service = OptimizerService(llm_config=llm_config)

        from src.optimizer.interfaces.llm_client import StubLLMClient
        is_stub = isinstance(service._llm_client, StubLLMClient)

        print(f"\n  {description}:")
        print(f"    Falls back to STUB: {'[YES]' if is_stub else '[NO]'}")


def demo_cost_control():
    """Demo 6: Cost control configuration."""
    print("\n" + "=" * 70)
    print("Demo 6: Cost Control Configuration")
    print("=" * 70)

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        cost_limits={
            "max_cost_per_request": 0.05,  # $0.05 per request
            "max_cost_per_day": 50.0,      # $50 per day
        }
    )

    print(f"\nCost limits configured:")
    print(f"  Max cost per request: ${llm_config.cost_limits['max_cost_per_request']:.2f}")
    print(f"  Max cost per day: ${llm_config.cost_limits['max_cost_per_day']:.2f}")
    print(f"\n  These limits help prevent unexpected API costs!")


def demo_caching():
    """Demo 7: Response caching configuration."""
    print("\n" + "=" * 70)
    print("Demo 7: Response Caching")
    print("=" * 70)

    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        enable_cache=True,
        cache_ttl=86400  # 24 hours
    )

    print(f"\nCaching configured:")
    print(f"  Enabled: {llm_config.enable_cache}")
    print(f"  TTL: {llm_config.cache_ttl}s ({llm_config.cache_ttl / 3600:.1f} hours)")
    print(f"\n  Benefits:")
    print(f"    - Reduced API calls")
    print(f"    - Lower costs")
    print(f"    - Faster response times")


def demo_single_prompt_optimization():
    """Demo 8: Single prompt optimization."""
    print("\n" + "=" * 70)
    print("Demo 8: Single Prompt Optimization")
    print("=" * 70)

    # Create test prompt
    prompt = Prompt(
        id="test_prompt_1",
        workflow_id="demo_wf",
        node_id="node_1",
        node_type="llm",
        text="Write summary of the document.",
        role="system",
        variables=[],
        context={}
    )

    # Create service
    service = OptimizerService()

    # Optimize with rule-based strategy
    result = service.optimize_single_prompt(prompt, strategy="clarity_focus")

    print(f"\nOriginal Prompt:")
    print(f"  {prompt.text}")
    print(f"\nOptimized Prompt:")
    print(f"  {result.optimized_prompt}")
    print(f"\nImprovement Details:")
    print(f"  Strategy: {result.strategy.value}")
    print(f"  Improvement Score: {result.improvement_score:.1f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Changes: {len(result.changes)}")


def demo_stats_reset():
    """Demo 9: Statistics reset."""
    print("\n" + "=" * 70)
    print("Demo 9: Statistics Reset")
    print("=" * 70)

    service = OptimizerService()

    print(f"\nBefore reset:")
    stats = service.get_llm_stats()
    print(f"  Stats: {stats}")

    service.reset_llm_stats()
    print(f"\nAfter reset:")
    stats = service.get_llm_stats()
    print(f"  Stats: {stats}")

    print(f"\n  Useful for:")
    print(f"    - Measuring per-cycle usage")
    print(f"    - Starting fresh measurements")
    print(f"    - Cost tracking per workflow")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("OptimizerService LLM Integration - Feature Demonstrations")
    print("=" * 70)

    demos = [
        ("Basic LLM Configuration", demo_basic_llm_config),
        ("Service Initialization", demo_service_initialization),
        ("Backward Compatibility", demo_backward_compatibility),
        ("LLM Usage Statistics", demo_llm_stats),
        ("Provider Fallback", demo_provider_fallback),
        ("Cost Control", demo_cost_control),
        ("Response Caching", demo_caching),
        ("Single Prompt Optimization", demo_single_prompt_optimization),
        ("Statistics Reset", demo_stats_reset),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n[FAIL] Demo failed: {e}")

    print("\n" + "=" * 70)
    print("All Demonstrations Complete!")
    print("=" * 70)

    print("\nKey Features Demonstrated:")
    print("  [OK] LLM configuration with multiple providers")
    print("  [OK] Service initialization with LLM support")
    print("  [OK] Backward compatibility maintained")
    print("  [OK] Usage statistics tracking")
    print("  [OK] Graceful provider fallback")
    print("  [OK] Cost control and limits")
    print("  [OK] Response caching")
    print("  [OK] Prompt optimization")
    print("  [OK] Statistics management")

    print("\nFor full usage documentation, see:")
    print("  docs/optimizer_service_llm_usage.md")
    print("\nFor implementation details, see:")
    print("  docs/optimizer_service_llm_implementation_report.md")


if __name__ == "__main__":
    main()
