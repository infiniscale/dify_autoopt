"""
Test suite for CODEX P1 #3: Analysis Cache Missing Fix

Tests that prompt analysis results are cached and reused.
"""

import hashlib

import pytest

from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.models import Prompt
from datetime import datetime


def make_test_prompt(text: str, prompt_id: str = "test_prompt") -> Prompt:
    """Helper to create test prompt objects."""
    return Prompt(
        id=prompt_id,
        workflow_id="test_wf",
        node_id="test_node",
        node_type="llm",
        text=text,
        role="user",
        variables=[],
        context={},
        extracted_at=datetime.now()
    )


class TestAnalysisCache:
    """Test suite for analysis caching."""

    def test_analysis_cache_hit(self):
        """Test that identical prompts reuse cached analysis."""
        service = OptimizerService()

        prompt = make_test_prompt("Test prompt")

        # First call - cache miss
        analysis1 = service._get_or_analyze(prompt)
        cache_size_1 = len(service._analysis_cache)

        # Second call - cache hit
        analysis2 = service._get_or_analyze(prompt)
        cache_size_2 = len(service._analysis_cache)

        assert analysis1 == analysis2, "Analysis objects should be identical"
        assert cache_size_1 == cache_size_2 == 1, "Cache should not grow on hit"

    def test_analysis_cache_miss_different_text(self):
        """Test that different prompts get different cache entries."""
        service = OptimizerService()

        prompt1 = make_test_prompt("First prompt")
        prompt2 = make_test_prompt("Second prompt")

        analysis1 = service._get_or_analyze(prompt1)
        analysis2 = service._get_or_analyze(prompt2)

        assert analysis1 != analysis2, "Different prompts should have different analyses"
        assert len(service._analysis_cache) == 2, "Cache should have 2 entries"

    def test_analysis_cache_key_deterministic(self):
        """Test that cache key is deterministic (MD5 of text)."""
        service = OptimizerService()

        text = "Test prompt"
        prompt = make_test_prompt(text)

        # Compute expected cache key
        expected_key = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Analyze to populate cache
        service._get_or_analyze(prompt)

        # Verify key exists
        assert expected_key in service._analysis_cache

    def test_analysis_cache_same_text_different_id(self):
        """Test that prompts with same text but different IDs share cache."""
        service = OptimizerService()

        text = "Identical text"
        prompt1 = make_test_prompt(text, "prompt_1")
        prompt2 = make_test_prompt(text, "prompt_2")

        analysis1 = service._get_or_analyze(prompt1)
        analysis2 = service._get_or_analyze(prompt2)

        # Same analysis (cached)
        assert analysis1 == analysis2
        assert len(service._analysis_cache) == 1

    def test_analysis_cache_performance_benefit(self):
        """Test that caching provides performance benefit."""
        import time

        service = OptimizerService()

        prompt = make_test_prompt("Test prompt " * 100)

        # First call (uncached)
        start = time.time()
        service._get_or_analyze(prompt)
        uncached_time = time.time() - start

        # Second call (cached)
        start = time.time()
        service._get_or_analyze(prompt)
        cached_time = time.time() - start

        # Cached should be significantly faster
        assert cached_time < uncached_time / 10, \
            f"Cache not effective: uncached={uncached_time}s, cached={cached_time}s"

    def test_analysis_cache_multiple_analyses(self):
        """Test cache with multiple different prompts."""
        service = OptimizerService()

        # Analyze 10 unique prompts
        for i in range(10):
            prompt = make_test_prompt(f"Prompt number {i}")
            service._get_or_analyze(prompt)

        assert len(service._analysis_cache) == 10

        # Re-analyze all - should all hit cache
        for i in range(10):
            prompt = make_test_prompt(f"Prompt number {i}")
            service._get_or_analyze(prompt)

        # Cache size unchanged
        assert len(service._analysis_cache) == 10

    def test_analysis_cache_empty_on_init(self):
        """Test that cache starts empty."""
        service = OptimizerService()
        assert len(service._analysis_cache) == 0

    def test_analysis_cache_thread_safety(self):
        """Test that cache is thread-safe (basic check)."""
        import threading

        service = OptimizerService()
        errors = []

        def analyze_prompt(worker_id):
            try:
                for i in range(50):
                    text = f"Prompt {i % 10}"  # Only 10 unique prompts
                    prompt = make_test_prompt(text, f"prompt_{worker_id}_{i}")
                    service._get_or_analyze(prompt)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=analyze_prompt, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety issues: {errors}"

        # Should only have 10 entries (unique prompts)
        assert len(service._analysis_cache) == 10

    def test_analysis_cache_unicode_handling(self):
        """Test cache handles unicode text correctly."""
        service = OptimizerService()

        text = "Test with unicode: 你好 مرحبا שלום"
        prompt = make_test_prompt(text)

        analysis1 = service._get_or_analyze(prompt)
        analysis2 = service._get_or_analyze(prompt)

        assert analysis1 == analysis2
        assert len(service._analysis_cache) == 1
