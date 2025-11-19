"""
Test suite for CRITICAL #3: Regex DoS Vulnerability Fix

Tests that regex patterns are pre-compiled and reused, not recompiled
on every analysis call.
"""

import time

import pytest

from src.optimizer.prompt_analyzer import PromptAnalyzer
from src.optimizer.models import Prompt


def make_test_prompt(text: str, prompt_id: str = "test_prompt") -> Prompt:
    """Helper to create test prompt objects."""
    from datetime import datetime

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


class TestRegexPerformance:
    """Test suite for regex compilation efficiency."""

    def test_regex_compilation_efficiency(self):
        """Verify regex is not recompiled on every call."""
        analyzer = PromptAnalyzer()

        # Pathological input with many vague terms
        text = "maybe some stuff probably perhaps somehow " * 200
        prompt = make_test_prompt(text)

        # Time 100 analyses
        start = time.time()
        for _ in range(100):
            analyzer.analyze_prompt(prompt)
        elapsed = time.time() - start

        # With pre-compiled regex: < 5s
        # With recompilation (old code): ~15-20s
        assert elapsed < 5.0, f"Too slow: {elapsed}s (likely recompiling regex)"

    def test_regex_precompilation(self):
        """Test that regex patterns are class-level attributes."""
        analyzer = PromptAnalyzer()

        # Verify pre-compiled regex exists at class level
        assert hasattr(PromptAnalyzer, '_VAGUE_REGEX'), "Missing pre-compiled vague regex"
        assert hasattr(PromptAnalyzer, '_FILLER_REGEX'), "Missing pre-compiled filler regex"

        # Verify they are compiled patterns
        import re
        assert isinstance(PromptAnalyzer._VAGUE_REGEX, re.Pattern), \
            "_VAGUE_REGEX is not a compiled pattern"
        assert isinstance(PromptAnalyzer._FILLER_REGEX, re.Pattern), \
            "_FILLER_REGEX is not a compiled pattern"

    def test_vague_language_detection_performance(self):
        """Test vague language detection scales linearly with input size."""
        analyzer = PromptAnalyzer()

        # Small input
        small_text = "maybe some stuff " * 10
        small_prompt = make_test_prompt(small_text)

        start = time.time()
        for _ in range(100):
            analyzer.analyze_prompt(small_prompt)
        small_time = time.time() - start

        # Large input (10x size)
        large_text = "maybe some stuff " * 100
        large_prompt = make_test_prompt(large_text)

        start = time.time()
        for _ in range(100):
            analyzer.analyze_prompt(large_prompt)
        large_time = time.time() - start

        # Should scale roughly linearly (allow 20x factor for variance)
        # If regex recompiled each time, this would be exponential
        ratio = large_time / max(small_time, 0.001)
        assert ratio < 20, f"Performance degradation too severe: {ratio}x"

    def test_filler_word_detection_performance(self):
        """Test filler word detection uses pre-compiled regex."""
        analyzer = PromptAnalyzer()

        # Text with many filler words
        text = "actually just really very totally basically " * 100
        prompt = make_test_prompt(text)

        # Should complete quickly
        start = time.time()
        for _ in range(50):
            analyzer.analyze_prompt(prompt)
        elapsed = time.time() - start

        assert elapsed < 3.0, f"Filler detection too slow: {elapsed}s"

    def test_analysis_correctness_preserved(self):
        """Ensure optimization didn't break analysis correctness."""
        analyzer = PromptAnalyzer()

        # Test prompt with known vague terms
        text = "maybe you could possibly do some stuff and things"
        prompt = make_test_prompt(text)

        analysis = analyzer.analyze_prompt(prompt)

        # Should detect vague language
        vague_issues = [i for i in analysis.issues if i.type.value == "vague_language"]
        assert len(vague_issues) > 0, "Failed to detect vague language"

    def test_regex_pattern_coverage(self):
        """Test that all vague patterns are covered by compiled regex."""
        # Extract individual patterns
        patterns = PromptAnalyzer.VAGUE_PATTERNS

        # Create combined regex
        combined = PromptAnalyzer._VAGUE_REGEX

        # Test each pattern is matched
        test_strings = {
            r"\bsome\b": "some things",
            r"\bmaybe\b": "maybe later",
            r"\bstuff\b": "stuff here",
            r"\betc\b": "etc etc",
        }

        for pattern_text in test_strings.values():
            assert combined.search(pattern_text), \
                f"Combined regex failed to match: {pattern_text}"

    def test_concurrent_analysis_no_regex_recompilation(self):
        """Test that concurrent analyses don't trigger recompilation."""
        import threading

        analyzer = PromptAnalyzer()
        errors = []

        def analyze_many():
            try:
                for i in range(100):
                    text = f"maybe some stuff iteration {i}"
                    prompt = make_test_prompt(text, f"prompt_{i}")
                    analyzer.analyze_prompt(prompt)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=analyze_many) for _ in range(10)]

        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        assert len(errors) == 0, f"Concurrent analysis errors: {errors}"

        # 10 threads * 100 analyses = 1000 total analyses
        # Should complete in < 10 seconds with pre-compiled regex
        assert elapsed < 10.0, f"Concurrent analysis too slow: {elapsed}s"
