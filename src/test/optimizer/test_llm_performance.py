"""
Performance Test Suite for LLM Integration.

This test suite validates performance and resource usage:
1. Response time tests (with/without cache)
2. Cache performance tests (hit rate, cleanup, LRU eviction)
3. Token tracking accuracy
4. Concurrent safety tests (threading)

Author: QA Engineer
Date: 2025-11-18
"""

import time
import pytest
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from src.optimizer.config import LLMConfig, LLMProvider
from src.optimizer.interfaces.llm_providers import OpenAIClient
from src.optimizer.utils.token_tracker import TokenUsageTracker
from src.optimizer.utils.prompt_cache import PromptCache
from src.optimizer.optimizer_service import OptimizerService
from src.config.models import WorkflowCatalog


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_workflow_dsl():
    """Sample workflow for testing."""
    return {
        "graph": {
            "nodes": [
                {
                    "data": {
                        "type": "llm",
                        "title": "Test Node",
                        "prompt_template": [
                            {"role": "system", "text": "Test prompt for performance testing"}
                        ]
                    },
                    "id": "llm_test"
                }
            ]
        }
    }


@pytest.fixture
def mock_catalog(sample_workflow_dsl):
    """Mock workflow catalog."""
    catalog = Mock(spec=WorkflowCatalog)
    catalog.get_workflow_by_id.return_value = sample_workflow_dsl
    return catalog


# ============================================================================
# Test 1: Response Time Tests
# ============================================================================


class TestResponseTime:
    """Test response time and latency metrics."""

    def test_stub_response_time_baseline(self, mock_catalog):
        """Measure baseline response time for STUB provider."""
        config = LLMConfig(provider=LLMProvider.STUB, enable_cache=False)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")

        # Measure response time
        start = time.time()
        result = service.optimize_prompt(prompts[0], "llm_clarity")
        elapsed_ms = (time.time() - start) * 1000

        # STUB should be very fast (< 100ms)
        assert elapsed_ms < 100, f"STUB optimization took {elapsed_ms:.2f}ms (expected < 100ms)"
        assert result.optimized_text is not None

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_openai_response_time_without_cache(self, mock_openai, mock_catalog):
        """Measure OpenAI response time without caching."""
        # Mock OpenAI with realistic delay
        mock_client = MagicMock()

        def slow_create(*args, **kwargs):
            time.sleep(0.1)  # Simulate network delay
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(
                message=MagicMock(content="Optimized result"),
                finish_reason="stop"
            )]
            mock_resp.usage = MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
            return mock_resp

        mock_client.chat.completions.create.side_effect = slow_create
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            enable_cache=False
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            start = time.time()
            result = service.optimize_prompt(prompts[0], "llm_clarity")
            elapsed_ms = (time.time() - start) * 1000

            # Should include network delay
            assert elapsed_ms >= 100, f"Expected delay >= 100ms, got {elapsed_ms:.2f}ms"
            assert result.optimized_text is not None

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_cache_hit_response_time(self, mock_openai, mock_catalog):
        """Verify cache hits are significantly faster."""
        mock_client = MagicMock()

        def slow_create(*args, **kwargs):
            time.sleep(0.1)
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(
                message=MagicMock(content="Optimized result"),
                finish_reason="stop"
            )]
            mock_resp.usage = MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
            return mock_resp

        mock_client.chat.completions.create.side_effect = slow_create
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            enable_cache=True,
            cache_ttl=3600
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            # First call (cache miss)
            start = time.time()
            result1 = service.optimize_prompt(prompts[0], "llm_clarity")
            time_miss = (time.time() - start) * 1000

            # Second call (cache hit)
            start = time.time()
            result2 = service.optimize_prompt(prompts[0], "llm_clarity")
            time_hit = (time.time() - start) * 1000

            # Cache hit should be much faster
            assert time_hit < time_miss / 2, \
                f"Cache hit ({time_hit:.2f}ms) not significantly faster than miss ({time_miss:.2f}ms)"

            # Results should be identical
            assert result1.optimized_text == result2.optimized_text

    def test_concurrent_requests_performance(self, mock_catalog):
        """Test performance under concurrent load."""
        config = LLMConfig(provider=LLMProvider.STUB, enable_cache=True)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")

        def optimize_task(task_id):
            start = time.time()
            result = service.optimize_prompt(prompts[0], "llm_clarity")
            elapsed = (time.time() - start) * 1000
            return task_id, elapsed, result

        # Run concurrent optimizations
        num_threads = 10
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(optimize_task, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        total_time = (time.time() - start_time) * 1000

        # Verify all completed successfully
        assert len(results) == num_threads
        assert all(r[2].optimized_text is not None for r in results)

        # Calculate average response time
        avg_time = sum(r[1] for r in results) / num_threads
        assert avg_time < 500, f"Average response time {avg_time:.2f}ms too high under load"


# ============================================================================
# Test 2: Cache Performance
# ============================================================================


class TestCachePerformance:
    """Test cache performance metrics."""

    def test_cache_hit_rate_tracking(self):
        """Test cache hit rate calculation."""
        cache = PromptCache(ttl_seconds=3600, max_size=100)

        prompt = "Test prompt for caching"
        strategy = "llm_clarity"

        # First access - miss
        result = cache.get(prompt, strategy)
        assert result is None

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
        assert stats["hit_rate"] == 0.0

        # Set cache
        cache.set(prompt, "Optimized result", strategy)

        # Second access - hit
        result = cache.get(prompt, strategy)
        assert result == "Optimized result"

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

        # Third access - hit
        cache.get(prompt, strategy)

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.666, 0.01)

    def test_cache_cleanup_expired_entries(self):
        """Test cleanup of expired cache entries."""
        cache = PromptCache(ttl_seconds=1, max_size=100)

        # Add entries
        for i in range(5):
            cache.set(f"prompt_{i}", f"result_{i}", "test")

        stats = cache.get_stats()
        assert stats["size"] == 5

        # Wait for expiration
        time.sleep(2)

        # Clear expired
        removed = cache.clear_expired()
        assert removed == 5

        stats = cache.get_stats()
        assert stats["size"] == 0

    def test_lru_eviction_behavior(self):
        """Test LRU eviction when cache is full."""
        cache = PromptCache(ttl_seconds=3600, max_size=3)

        # Fill cache
        cache.set("prompt_1", "result_1", "test")
        cache.set("prompt_2", "result_2", "test")
        cache.set("prompt_3", "result_3", "test")

        stats = cache.get_stats()
        assert stats["size"] == 3
        assert stats["evictions"] == 0

        # Access prompt_1 to make it more recently used
        cache.get("prompt_1", "test")

        # Add new entry - should evict prompt_2 (least recently used)
        cache.set("prompt_4", "result_4", "test")

        stats = cache.get_stats()
        assert stats["size"] == 3
        assert stats["evictions"] == 1

        # Verify prompt_2 was evicted
        assert cache.get("prompt_2", "test") is None

        # Verify others still present
        assert cache.get("prompt_1", "test") == "result_1"
        assert cache.get("prompt_3", "test") == "result_3"
        assert cache.get("prompt_4", "test") == "result_4"

    def test_cache_size_estimation(self):
        """Test cache memory size estimation."""
        cache = PromptCache(ttl_seconds=3600, max_size=100)

        # Add entries
        for i in range(10):
            prompt = f"Test prompt number {i} with some content"
            result = f"Optimized result number {i} with some optimization"
            cache.set(prompt, result, "test")

        size_bytes = cache.get_size_bytes()

        # Should have non-zero size
        assert size_bytes > 0

        # Rough estimate: 10 entries * ~100 bytes each
        assert size_bytes > 500  # At least 500 bytes
        assert size_bytes < 10000  # Less than 10KB

    def test_cache_key_uniqueness(self):
        """Test that cache keys properly differentiate prompts and strategies."""
        cache = PromptCache(ttl_seconds=3600, max_size=100)

        prompt = "Same prompt text"

        # Store with different strategies
        cache.set(prompt, "Result for clarity", "llm_clarity")
        cache.set(prompt, "Result for efficiency", "llm_efficiency")

        # Retrieve - should get different results
        result_clarity = cache.get(prompt, "llm_clarity")
        result_efficiency = cache.get(prompt, "llm_efficiency")

        assert result_clarity == "Result for clarity"
        assert result_efficiency == "Result for efficiency"
        assert result_clarity != result_efficiency


# ============================================================================
# Test 3: Token Tracking Accuracy
# ============================================================================


class TestTokenTrackingAccuracy:
    """Test token usage tracking accuracy."""

    def test_token_count_accuracy(self):
        """Test accurate token counting."""
        tracker = TokenUsageTracker()

        # Track usage
        cost = tracker.track_usage(
            model="gpt-4-turbo-preview",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=100
        )

        # Verify calculation
        # gpt-4-turbo-preview: input=$0.01/1k, output=$0.03/1k
        expected_cost = (1000 / 1000 * 0.01) + (500 / 1000 * 0.03)
        assert cost == pytest.approx(expected_cost, 0.001)

        # Check stats
        stats = tracker.get_stats()
        assert stats["total_tokens"] == 1500
        assert stats["total_cost"] == pytest.approx(expected_cost, 0.001)
        assert stats["request_count"] == 1

    def test_cost_calculation_for_different_models(self):
        """Test cost calculation accuracy across different models."""
        tracker = TokenUsageTracker()

        models_to_test = [
            ("gpt-4-turbo-preview", 1000, 500, 0.01 + 0.015),
            ("gpt-4", 1000, 500, 0.03 + 0.03),
            ("gpt-3.5-turbo", 1000, 500, 0.0005 + 0.00075),
            ("claude-3-opus", 1000, 500, 0.015 + 0.0375),
            ("claude-3-sonnet", 1000, 500, 0.003 + 0.0075),
        ]

        for model, input_tok, output_tok, expected in models_to_test:
            cost = tracker.track_usage(model, input_tok, output_tok)
            assert cost == pytest.approx(expected, 0.0001), \
                f"Cost calculation incorrect for {model}"

    def test_aggregated_statistics(self):
        """Test aggregated statistics across multiple requests."""
        tracker = TokenUsageTracker()

        # Track multiple requests
        costs = []
        for i in range(5):
            cost = tracker.track_usage(
                model="gpt-4-turbo-preview",
                input_tokens=1000 * (i + 1),
                output_tokens=500 * (i + 1),
                latency_ms=100 + i * 10
            )
            costs.append(cost)

        stats = tracker.get_stats()

        # Verify aggregation
        assert stats["request_count"] == 5
        assert stats["total_tokens"] == sum((1000 + 500) * (i + 1) for i in range(5))
        assert stats["total_cost"] == pytest.approx(sum(costs), 0.001)
        assert stats["average_latency_ms"] == pytest.approx(120, 0.1)

    def test_daily_cost_calculation(self):
        """Test daily cost tracking."""
        tracker = TokenUsageTracker()

        # Add some requests
        for _ in range(3):
            tracker.track_usage("gpt-4-turbo-preview", 1000, 500)

        daily_cost = tracker.get_daily_cost()
        total_stats = tracker.get_stats()

        # Daily cost should equal total (all requests are within 24h)
        assert daily_cost == pytest.approx(total_stats["total_cost"], 0.001)

    def test_recent_requests_tracking(self):
        """Test recent request history."""
        tracker = TokenUsageTracker()

        # Add multiple requests
        for i in range(10):
            tracker.track_usage(
                model=f"model-{i % 3}",
                input_tokens=100 * i,
                output_tokens=50 * i
            )

        # Get recent requests
        recent = tracker.get_recent_requests(limit=5)

        assert len(recent) == 5

        # Should be in reverse chronological order (most recent first)
        # Verify tokens are decreasing (since we added in increasing order)
        assert recent[0]["total_tokens"] > recent[-1]["total_tokens"]


# ============================================================================
# Test 4: Concurrent Safety
# ============================================================================


class TestConcurrentSafety:
    """Test thread safety under concurrent access."""

    def test_token_tracker_thread_safety(self):
        """Test token tracker is thread-safe."""
        tracker = TokenUsageTracker()

        def track_task(task_id):
            for _ in range(100):
                tracker.track_usage(
                    model="gpt-4-turbo-preview",
                    input_tokens=100,
                    output_tokens=50
                )

        # Run concurrent tracking
        threads = []
        num_threads = 10

        for i in range(num_threads):
            t = threading.Thread(target=track_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify counts
        stats = tracker.get_stats()
        expected_requests = num_threads * 100
        assert stats["request_count"] == expected_requests
        assert stats["total_tokens"] == expected_requests * 150

    def test_cache_thread_safety(self):
        """Test cache is thread-safe under concurrent access."""
        cache = PromptCache(ttl_seconds=3600, max_size=1000)

        def cache_task(task_id):
            for i in range(50):
                prompt = f"prompt_{task_id}_{i}"
                result = f"result_{task_id}_{i}"

                # Set and get
                cache.set(prompt, result, "test")
                retrieved = cache.get(prompt, "test")

                # Verify consistency
                assert retrieved == result

        # Run concurrent operations
        threads = []
        num_threads = 10

        for i in range(num_threads):
            t = threading.Thread(target=cache_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify final state
        stats = cache.get_stats()
        assert stats["hits"] > 0
        assert stats["misses"] > 0
        assert stats["size"] > 0

    def test_concurrent_optimization_no_race_conditions(self, mock_catalog):
        """Test no race conditions in concurrent optimizations."""
        config = LLMConfig(provider=LLMProvider.STUB, enable_cache=True)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")

        results = []
        errors = []

        def optimize_task(task_id):
            try:
                result = service.optimize_prompt(prompts[0], "llm_clarity")
                results.append((task_id, result))
            except Exception as e:
                errors.append((task_id, e))

        # Run concurrent optimizations
        threads = []
        num_threads = 20

        for i in range(num_threads):
            t = threading.Thread(target=optimize_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads

        # All results should be valid
        assert all(r[1].optimized_text is not None for r in results)

    def test_stats_reset_thread_safety(self):
        """Test that stats reset is thread-safe."""
        tracker = TokenUsageTracker()

        def mixed_operations(task_id):
            for i in range(50):
                if i % 10 == 0:
                    tracker.reset()
                else:
                    tracker.track_usage("gpt-4-turbo-preview", 100, 50)
                    tracker.get_stats()

        threads = []
        for i in range(5):
            t = threading.Thread(target=mixed_operations, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        stats = tracker.get_stats()
        assert stats is not None


# ============================================================================
# Performance Benchmarks
# ============================================================================


class TestPerformanceBenchmarks:
    """Establish performance baselines and benchmarks."""

    def test_baseline_optimization_throughput(self, mock_catalog):
        """Measure baseline optimization throughput."""
        config = LLMConfig(provider=LLMProvider.STUB, enable_cache=False)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")

        # Measure throughput
        num_iterations = 100
        start = time.time()

        for _ in range(num_iterations):
            service.optimize_prompt(prompts[0], "llm_clarity")

        elapsed = time.time() - start
        throughput = num_iterations / elapsed

        # Baseline: should handle at least 50 optimizations/second
        assert throughput >= 50, \
            f"Throughput {throughput:.2f} ops/sec below baseline (expected >= 50)"

    def test_cache_performance_benefit(self, mock_catalog):
        """Quantify cache performance benefit."""
        prompts_cached = []
        prompts_uncached = []

        # Test without cache
        config = LLMConfig(provider=LLMProvider.STUB, enable_cache=False)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)
        prompts = service.extract_prompts("wf_001")

        start = time.time()
        for _ in range(10):
            service.optimize_prompt(prompts[0], "llm_clarity")
        time_uncached = time.time() - start

        # Test with cache
        config = LLMConfig(provider=LLMProvider.STUB, enable_cache=True)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        start = time.time()
        for _ in range(10):
            service.optimize_prompt(prompts[0], "llm_clarity")
        time_cached = time.time() - start

        # Cache should provide significant speedup (at least 2x)
        speedup = time_uncached / time_cached
        assert speedup >= 2, \
            f"Cache speedup {speedup:.2f}x below target (expected >= 2x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
