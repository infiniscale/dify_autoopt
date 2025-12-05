"""
Test suite for CRITICAL #1: LRU Cache Race Condition Fix

Tests that LRU cache operations are thread-safe and don't corrupt under
heavy concurrent load.
"""

import threading
import time
from datetime import datetime

import pytest

from src.optimizer.interfaces.filesystem_storage import LRUCache
from src.optimizer.models import PromptVersion, PromptAnalysis, Prompt


def make_test_version(version: str, prompt_id: str = "test_prompt") -> PromptVersion:
    """Helper to create test version objects."""
    prompt = Prompt(
        id=prompt_id,
        workflow_id="test_wf",
        node_id="test_node",
        node_type="llm",
        text=f"Test prompt version {version}",
        role="user",
        variables=[],
        context={},
        extracted_at=datetime.now()
    )

    analysis = PromptAnalysis(
        prompt_id=prompt_id,
        overall_score=85.0,
        clarity_score=80.0,
        efficiency_score=90.0,
        issues=[],
        suggestions=[],
        metadata={}
    )

    return PromptVersion(
        prompt_id=prompt_id,
        version=version,
        prompt=prompt,
        analysis=analysis,
        optimization_result=None,
        parent_version=None,
        metadata={}
    )


class TestLRUCacheRaceCondition:
    """Test suite for LRU cache thread safety."""

    def test_lru_cache_concurrent_put(self):
        """Test LRU cache under heavy concurrent load."""
        cache = LRUCache(max_size=5)
        errors = []

        def hammer_cache(worker_id: int):
            """Stress test cache with concurrent puts."""
            for i in range(1000):
                try:
                    key = f"key_{i % 10}"
                    cache.put(key, make_test_version(f"1.{worker_id}.{i}"))

                    # Verify cache never exceeds max size
                    stats = cache.stats()
                    if stats["size"] > cache.max_size:
                        errors.append(
                            f"Cache size {stats['size']} exceeds max {cache.max_size}"
                        )
                except Exception as e:
                    errors.append(f"Worker {worker_id} iteration {i}: {str(e)}")

        # Launch 20 concurrent workers
        threads = [
            threading.Thread(target=hammer_cache, args=(i,))
            for i in range(20)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Assert no race conditions occurred
        assert len(errors) == 0, f"Found {len(errors)} race conditions: {errors[:5]}"

    def test_lru_cache_concurrent_get_put(self):
        """Test concurrent reads and writes don't corrupt cache."""
        cache = LRUCache(max_size=10)
        errors = []

        # Pre-populate cache
        for i in range(10):
            cache.put(f"key_{i}", make_test_version(f"1.0.{i}"))

        def reader(worker_id: int):
            """Concurrent reader thread."""
            for i in range(500):
                try:
                    key = f"key_{i % 10}"
                    result = cache.get(key)
                    # Result may be None (evicted) but shouldn't raise
                except Exception as e:
                    errors.append(f"Reader {worker_id}: {str(e)}")

        def writer(worker_id: int):
            """Concurrent writer thread."""
            for i in range(500):
                try:
                    key = f"key_{i % 10}"
                    cache.put(key, make_test_version(f"2.{worker_id}.{i}"))
                except Exception as e:
                    errors.append(f"Writer {worker_id}: {str(e)}")

        # Mix of readers and writers
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=reader, args=(i,)))
            threads.append(threading.Thread(target=writer, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Race conditions detected: {errors[:5]}"

    def test_lru_cache_size_invariant(self):
        """Test that cache size never exceeds max_size under any conditions."""
        cache = LRUCache(max_size=3)

        # Add more items than max_size
        for i in range(100):
            cache.put(f"key_{i}", make_test_version(f"1.0.{i}"))

            # Verify size constraint
            stats = cache.stats()
            assert stats["size"] <= cache.max_size, \
                f"Cache size {stats['size']} exceeds max {cache.max_size}"

    def test_lru_cache_eviction_order(self):
        """Test that LRU eviction policy is maintained."""
        cache = LRUCache(max_size=3)

        cache.put("a", make_test_version("1.0.0", "prompt_a"))
        cache.put("b", make_test_version("1.0.0", "prompt_b"))
        cache.put("c", make_test_version("1.0.0", "prompt_c"))

        # Access 'a' to make it most recent
        cache.get("a")

        # Add 'd' - should evict 'b' (oldest)
        cache.put("d", make_test_version("1.0.0", "prompt_d"))

        assert cache.get("a") is not None, "Most recently used 'a' was evicted"
        assert cache.get("b") is None, "Least recently used 'b' was not evicted"
        assert cache.get("c") is not None, "'c' should still be present"
        assert cache.get("d") is not None, "Newly added 'd' should be present"

    def test_lru_cache_stats_consistency(self):
        """Test that cache statistics remain consistent under concurrent load."""
        cache = LRUCache(max_size=10)

        def stress_test():
            for i in range(100):
                cache.put(f"key_{i % 15}", make_test_version(f"1.0.{i}"))
                cache.get(f"key_{i % 15}")

        threads = [threading.Thread(target=stress_test) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = cache.stats()

        # Verify stats are internally consistent
        assert stats["hits"] + stats["misses"] > 0, "No cache operations recorded"
        assert 0.0 <= stats["hit_rate"] <= 1.0, f"Invalid hit rate: {stats['hit_rate']}"
        assert stats["size"] <= stats["max_size"], "Size exceeds max"
        assert 0.0 <= stats["utilization"] <= 1.0, f"Invalid utilization: {stats['utilization']}"
