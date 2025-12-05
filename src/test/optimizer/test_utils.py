"""
Unit Tests for Token Tracker and Prompt Cache.

Date: 2025-11-18
Author: backend-developer
Description: Tests for token usage tracking and caching utilities.
"""

import pytest
import time
from datetime import datetime, timedelta

from src.optimizer.utils.token_tracker import TokenUsageTracker
from src.optimizer.utils.prompt_cache import PromptCache


class TestTokenUsageTracker:
    """Test token usage tracking functionality."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = TokenUsageTracker()
        assert tracker._total_tokens == 0
        assert tracker._total_cost == 0.0
        assert tracker._request_count == 0

    def test_track_usage_basic(self):
        """Test basic token tracking."""
        tracker = TokenUsageTracker()

        cost = tracker.track_usage("gpt-4-turbo-preview", 1000, 500)

        # Expected: (1000 * 0.01 / 1000) + (500 * 0.03 / 1000) = 0.01 + 0.015 = 0.025
        assert abs(cost - 0.025) < 0.001
        assert tracker._total_tokens == 1500
        assert tracker._request_count == 1

    def test_track_multiple_requests(self):
        """Test tracking multiple requests."""
        tracker = TokenUsageTracker()

        tracker.track_usage("gpt-4-turbo-preview", 1000, 500)
        tracker.track_usage("gpt-3.5-turbo", 2000, 1000)

        stats = tracker.get_stats()
        assert stats["request_count"] == 2
        assert stats["total_tokens"] == 4500
        assert stats["total_cost"] > 0

    def test_get_stats(self):
        """Test getting statistics."""
        tracker = TokenUsageTracker()

        tracker.track_usage("gpt-4-turbo-preview", 1000, 500, latency_ms=150)
        tracker.track_usage("gpt-4-turbo-preview", 2000, 1000, latency_ms=250)

        stats = tracker.get_stats()

        assert stats["request_count"] == 2
        assert stats["total_tokens"] == 4500
        assert stats["average_latency_ms"] == 200  # (150 + 250) / 2
        assert "gpt-4-turbo-preview" in stats["requests_by_model"]
        assert stats["requests_by_model"]["gpt-4-turbo-preview"] == 2

    def test_check_limits_daily(self):
        """Test daily cost limit checking."""
        tracker = TokenUsageTracker()

        # Add some usage
        tracker.track_usage("gpt-4-turbo-preview", 1000, 500)  # ~0.025

        # Should be within limit
        assert tracker.check_limits(daily_limit=1.0)

        # Should exceed limit
        assert not tracker.check_limits(daily_limit=0.01)

    def test_check_limits_per_request(self):
        """Test per-request cost limit checking."""
        tracker = TokenUsageTracker()

        # Add usage with known cost
        tracker.track_usage("gpt-4-turbo-preview", 1000, 500)  # ~0.025

        # Should be within limit
        assert tracker.check_limits(request_limit=0.1)

        # Should exceed limit
        assert not tracker.check_limits(request_limit=0.01)

    def test_get_daily_cost(self):
        """Test getting daily cost."""
        tracker = TokenUsageTracker()

        cost1 = tracker.track_usage("gpt-4-turbo-preview", 1000, 500)
        cost2 = tracker.track_usage("gpt-3.5-turbo", 2000, 1000)

        daily_cost = tracker.get_daily_cost()
        expected_cost = cost1 + cost2

        assert abs(daily_cost - expected_cost) < 0.001

    def test_reset(self):
        """Test resetting tracker."""
        tracker = TokenUsageTracker()

        tracker.track_usage("gpt-4-turbo-preview", 1000, 500)
        assert tracker._request_count > 0

        tracker.reset()

        assert tracker._total_tokens == 0
        assert tracker._total_cost == 0.0
        assert tracker._request_count == 0
        assert len(tracker._request_history) == 0

    def test_get_recent_requests(self):
        """Test getting recent request history."""
        tracker = TokenUsageTracker()

        # Add 5 requests
        for i in range(5):
            tracker.track_usage("gpt-4-turbo-preview", 100 * (i + 1), 50)

        # Get last 3
        recent = tracker.get_recent_requests(limit=3)

        assert len(recent) == 3
        # Should be in reverse order (most recent first)
        assert recent[0]["input_tokens"] == 500  # Last request
        assert recent[2]["input_tokens"] == 300  # Third-to-last

    def test_cost_calculation_different_models(self):
        """Test cost calculation for different models."""
        tracker = TokenUsageTracker()

        # GPT-4 Turbo
        cost_gpt4 = tracker.track_usage("gpt-4-turbo-preview", 1000, 1000)
        expected_gpt4 = (1000 * 0.01 / 1000) + (1000 * 0.03 / 1000)
        assert abs(cost_gpt4 - expected_gpt4) < 0.001

        # GPT-3.5 Turbo
        cost_gpt35 = tracker.track_usage("gpt-3.5-turbo", 1000, 1000)
        expected_gpt35 = (1000 * 0.0005 / 1000) + (1000 * 0.0015 / 1000)
        assert abs(cost_gpt35 - expected_gpt35) < 0.001

        # Claude Opus
        cost_claude = tracker.track_usage("claude-3-opus", 1000, 1000)
        expected_claude = (1000 * 0.015 / 1000) + (1000 * 0.075 / 1000)
        assert abs(cost_claude - expected_claude) < 0.001

    def test_unknown_model_cost(self):
        """Test that unknown models have zero cost."""
        tracker = TokenUsageTracker()

        cost = tracker.track_usage("unknown-model", 1000, 1000)
        assert cost == 0.0


class TestPromptCache:
    """Test prompt caching functionality."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = PromptCache(ttl_seconds=3600, max_size=100)

        assert cache._ttl == 3600
        assert cache._max_size == 100
        assert len(cache._cache) == 0

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = PromptCache()

        key1 = cache._generate_key("Test prompt", "strategy1")
        key2 = cache._generate_key("Test prompt", "strategy1")
        key3 = cache._generate_key("Test prompt", "strategy2")

        # Same inputs should generate same key
        assert key1 == key2

        # Different strategy should generate different key
        assert key1 != key3

        # Should be MD5 hex (32 chars)
        assert len(key1) == 32

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = PromptCache()

        cache.set("Test prompt", "Optimized result", "strategy1")
        result = cache.get("Test prompt", "strategy1")

        assert result == "Optimized result"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = PromptCache()

        result = cache.get("Nonexistent prompt", "strategy1")
        assert result is None

    def test_cache_hit_and_miss_stats(self):
        """Test hit and miss statistics."""
        cache = PromptCache()

        # Miss
        cache.get("Test", "s1")

        # Set and hit
        cache.set("Test", "Result", "s1")
        cache.get("Test", "s1")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = PromptCache(ttl_seconds=1)  # 1 second TTL

        cache.set("Test", "Result", "s1")

        # Should be cached immediately
        assert cache.get("Test", "s1") == "Result"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("Test", "s1") is None

    def test_clear_expired(self):
        """Test clearing expired entries."""
        cache = PromptCache(ttl_seconds=1)

        # Add entries
        cache.set("Test1", "Result1", "s1")
        cache.set("Test2", "Result2", "s2")

        # Wait for expiration
        time.sleep(1.1)

        # Clear expired
        removed = cache.clear_expired()

        assert removed == 2
        assert len(cache._cache) == 0

    def test_max_size_eviction(self):
        """Test LRU eviction when max size is reached."""
        cache = PromptCache(ttl_seconds=3600, max_size=3)

        # Fill cache
        cache.set("Test1", "Result1", "s1")
        cache.set("Test2", "Result2", "s2")
        cache.set("Test3", "Result3", "s3")

        assert cache.get_stats()["size"] == 3

        # Add one more - should evict LRU
        cache.set("Test4", "Result4", "s4")

        assert cache.get_stats()["size"] == 3
        assert cache.get_stats()["evictions"] == 1

        # Test1 should be evicted (oldest)
        assert cache.get("Test1", "s1") is None

    def test_lru_access_updates(self):
        """Test that accessing entries updates LRU order."""
        cache = PromptCache(ttl_seconds=3600, max_size=3)

        # Add entries
        cache.set("Test1", "Result1", "s1")
        cache.set("Test2", "Result2", "s2")
        cache.set("Test3", "Result3", "s3")

        # Access Test1 to make it more recent
        cache.get("Test1", "s1")

        # Add new entry - should evict Test2 (now LRU)
        cache.set("Test4", "Result4", "s4")

        # Test1 should still be there
        assert cache.get("Test1", "s1") == "Result1"

        # Test2 should be evicted
        assert cache.get("Test2", "s2") is None

    def test_clear(self):
        """Test clearing cache."""
        cache = PromptCache()

        cache.set("Test1", "Result1", "s1")
        cache.set("Test2", "Result2", "s2")

        assert cache.get_stats()["size"] == 2

        cache.clear()

        assert cache.get_stats()["size"] == 0
        assert cache.get_stats()["hits"] == 0
        assert cache.get_stats()["misses"] == 0

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = PromptCache(max_size=100)

        cache.set("Test", "Result", "s1")
        cache.get("Test", "s1")  # Hit
        cache.get("Nonexistent", "s2")  # Miss

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["evictions"] == 0

    def test_get_size_bytes(self):
        """Test estimating cache size in bytes."""
        cache = PromptCache()

        # Empty cache
        assert cache.get_size_bytes() == 0

        # Add entry
        cache.set("Test prompt", "Optimized result", "strategy1")

        size = cache.get_size_bytes()
        assert size > 0

    def test_different_strategies_separate_cache(self):
        """Test that different strategies maintain separate cache entries."""
        cache = PromptCache()

        cache.set("Test", "Result1", "strategy1")
        cache.set("Test", "Result2", "strategy2")

        assert cache.get("Test", "strategy1") == "Result1"
        assert cache.get("Test", "strategy2") == "Result2"

        # Should have 2 entries
        assert cache.get_stats()["size"] == 2

    def test_thread_safety_basic(self):
        """Test basic thread safety of cache operations."""
        import threading

        cache = PromptCache()
        errors = []

        def worker(i):
            try:
                cache.set(f"Test{i}", f"Result{i}", "s1")
                result = cache.get(f"Test{i}", "s1")
                assert result == f"Result{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.get_stats()["size"] == 10
