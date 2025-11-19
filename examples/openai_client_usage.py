"""
OpenAI Client Usage Examples

This file demonstrates how to use the OpenAI client layer for prompt analysis and optimization.

Author: Backend Developer
Date: 2025-11-18
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimizer.config import LLMConfig, LLMProvider, LLMConfigLoader
from src.optimizer.interfaces.llm_providers import OpenAIClient
from src.optimizer.interfaces.llm_client import StubLLMClient


def example_1_basic_configuration():
    """Example 1: Basic OpenAI client configuration and initialization."""
    print("=== Example 1: Basic Configuration ===\n")

    # Set API key in environment (in production, use .env file)
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # Create configuration
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4-turbo-preview",
        temperature=0.7,
        max_tokens=2000,
        enable_cache=True
    )

    print(f"Configuration created:")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Cache enabled: {config.enable_cache}\n")

    # Initialize client (commented out - requires valid API key)
    # client = OpenAIClient(config)
    # print("Client initialized successfully\n")


def example_2_analyze_prompt():
    """Example 2: Analyze prompt quality using GPT-4."""
    print("=== Example 2: Analyze Prompt ===\n")

    # For demonstration, use StubLLMClient (no API key required)
    client = StubLLMClient()

    prompt = "Summarize the document in a few sentences."

    # Analyze prompt
    response = client.analyze_prompt(prompt)

    print(f"Original prompt: {prompt}")
    print(f"\nAnalysis response:")
    print(f"  Provider: {response.provider}")
    print(f"  Model: {response.model}")
    print(f"  Tokens used: {response.tokens_used}")
    print(f"  Cost: ${response.cost:.4f}")
    print(f"  Latency: {response.latency_ms:.0f}ms")
    print(f"  Cached: {response.cached}")

    # Parse analysis results
    analysis = json.loads(response.content)
    print(f"\nQuality scores:")
    print(f"  Overall: {analysis['overall_score']}")
    print(f"  Clarity: {analysis['clarity_score']}")
    print(f"  Efficiency: {analysis['efficiency_score']}")
    print(f"  Structure: {analysis['structure_score']}")

    if analysis['issues']:
        print(f"\nIssues found:")
        for issue in analysis['issues']:
            print(f"  - {issue}")

    if analysis['suggestions']:
        print(f"\nSuggestions:")
        for suggestion in analysis['suggestions']:
            print(f"  - {suggestion}")

    print()


def example_3_optimize_prompt():
    """Example 3: Optimize prompt using different strategies."""
    print("=== Example 3: Optimize Prompt ===\n")

    client = StubLLMClient()

    original_prompt = "Write summary."

    # Try different optimization strategies
    strategies = ["llm_clarity", "llm_efficiency", "llm_guided"]

    for strategy in strategies:
        response = client.optimize_prompt(original_prompt, strategy=strategy)

        print(f"Strategy: {strategy}")
        print(f"Original: {original_prompt}")
        print(f"Optimized: {response.content}")
        print(f"Tokens: {response.tokens_used}, Cost: ${response.cost:.4f}")
        print()


def example_4_caching_demonstration():
    """Example 4: Demonstrate caching effectiveness."""
    print("=== Example 4: Caching Demonstration ===\n")

    config = LLMConfig(
        provider=LLMProvider.STUB,
        enable_cache=True
    )

    client = StubLLMClient()

    prompt = "Analyze this text for sentiment."
    strategy = "llm_clarity"

    # First call - cache miss
    print("First call (cache miss):")
    response1 = client.optimize_prompt(prompt, strategy=strategy)
    print(f"  Cached: {response1.cached}")
    print(f"  Tokens: {response1.tokens_used}")
    print(f"  Cost: ${response1.cost:.4f}\n")

    # Second call - cache hit (would be, in real OpenAI client)
    print("Second call (cache hit in OpenAI client):")
    print("  With caching, this would return cached results")
    print("  Tokens: 0, Cost: $0.0000, Latency: ~0ms\n")


def example_5_usage_statistics():
    """Example 5: Track and display usage statistics."""
    print("=== Example 5: Usage Statistics ===\n")

    client = StubLLMClient()

    # Make several calls
    prompts = [
        "Summarize this article.",
        "Analyze the sentiment.",
        "Extract key points."
    ]

    for prompt in prompts:
        client.analyze_prompt(prompt)
        client.optimize_prompt(prompt, strategy="llm_clarity")

    # Get statistics
    stats = client.get_usage_stats()

    print("Usage statistics:")
    print(f"  Total requests: {stats.total_requests}")
    print(f"  Total tokens: {stats.total_tokens}")
    print(f"  Total cost: ${stats.total_cost:.4f}")
    print(f"  Cache hits: {stats.cache_hits}")
    print(f"  Cache misses: {stats.cache_misses}")

    if (stats.cache_hits + stats.cache_misses) > 0:
        hit_rate = stats.cache_hits / (stats.cache_hits + stats.cache_misses)
        print(f"  Cache hit rate: {hit_rate:.2%}")

    print(f"  Avg latency: {stats.average_latency_ms:.0f}ms\n")


def example_6_cost_limits():
    """Example 6: Configure cost limits for safety."""
    print("=== Example 6: Cost Limits ===\n")

    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4-turbo-preview",
        cost_limits={
            "max_cost_per_request": 0.10,  # $0.10 per request
            "max_cost_per_day": 10.00       # $10 per day
        }
    )

    print("Cost limits configured:")
    print(f"  Max per request: ${config.cost_limits['max_cost_per_request']:.2f}")
    print(f"  Max per day: ${config.cost_limits['max_cost_per_day']:.2f}")
    print("\nThe client will log warnings if limits are exceeded.")
    print("(Note: Limits are not enforced automatically, only logged)\n")


def example_7_configuration_from_yaml():
    """Example 7: Load configuration from YAML file."""
    print("=== Example 7: Configuration from YAML ===\n")

    # Example YAML content (save as config.yaml):
    yaml_content = """
llm:
  provider: openai
  model: gpt-4-turbo-preview
  temperature: 0.7
  max_tokens: 2000
  enable_cache: true
  cache_ttl: 86400
  cost_limits:
    max_cost_per_request: 0.10
    max_cost_per_day: 10.00
"""

    print("Example YAML configuration:")
    print(yaml_content)

    # In practice:
    # config = LLMConfigLoader.from_yaml("config.yaml")
    # client = OpenAIClient(config)

    print("Load with: LLMConfigLoader.from_yaml('config.yaml')\n")


def example_8_error_handling():
    """Example 8: Handle common errors gracefully."""
    print("=== Example 8: Error Handling ===\n")

    # Example 1: Invalid strategy
    client = StubLLMClient()

    try:
        client.optimize_prompt("Test", strategy="invalid_strategy")
    except Exception as e:
        print(f"Caught error: {e}")
        print(f"Error type: {type(e).__name__}\n")

    # Example 2: Missing API key (would fail in real OpenAI client)
    print("In OpenAI client, missing API key would raise:")
    print("  ValueError: API key not found: Environment variable 'OPENAI_API_KEY' is not set\n")

    # Example 3: Configuration validation
    try:
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            temperature=3.0  # Invalid: must be 0.0-2.0
        )
    except Exception as e:
        print(f"Caught validation error: {e}")
        print(f"Error type: {type(e).__name__}\n")


def example_9_migration_guide():
    """Example 9: Migration from old interface to new interface."""
    print("=== Example 9: Migration Guide ===\n")

    client = StubLLMClient()

    print("OLD CODE (returned string):")
    print("  optimized = client.optimize_prompt('Test', 'clarity_focus')")
    print("  print(optimized)  # Direct string\n")

    print("NEW CODE (returns LLMResponse):")
    print("  response = client.optimize_prompt('Test', 'clarity_focus')")
    print("  print(response.content)  # Extract content")
    print("  print(f'Cost: ${response.cost}')  # Additional metadata\n")

    # Demonstrate
    response = client.optimize_prompt("Test", "clarity_focus")
    print(f"Actual result:")
    print(f"  Content: {response.content}")
    print(f"  Cost: ${response.cost:.4f}")
    print(f"  Tokens: {response.tokens_used}\n")


def main():
    """Run all examples."""
    examples = [
        example_1_basic_configuration,
        example_2_analyze_prompt,
        example_3_optimize_prompt,
        example_4_caching_demonstration,
        example_5_usage_statistics,
        example_6_cost_limits,
        example_7_configuration_from_yaml,
        example_8_error_handling,
        example_9_migration_guide
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}\n")

    print("=== All Examples Complete ===")


if __name__ == "__main__":
    main()
