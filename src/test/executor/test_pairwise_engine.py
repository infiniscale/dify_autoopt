"""
Unit tests for PairwiseEngine

Focus: cover all code paths including caching, high-dimensional handling,
and error handling branches for PICT generation.
"""

import sys
import types
from pathlib import Path

import pytest

from src.executor.pairwise_engine import PairwiseEngine
from src.config.utils.exceptions import CaseGenerationError


class TestPairwiseEngineBasic:
    """Basic behaviour and engine selection tests."""

    def test_generate_empty_dimensions_returns_empty_list(self, tmp_path: Path) -> None:
        """Empty dimensions should return empty list without touching cache."""
        engine = PairwiseEngine(cache_dir=tmp_path)

        result = engine.generate({})

        assert result == []

    def test_generate_naive_engine_cartesian_product(self) -> None:
        """Engine type 'naive' should use naive cartesian product generation."""
        dimensions = {
            "a": [1, 2],
            "b": ["x", "y"],
        }

        engine = PairwiseEngine(engine_type="naive")

        result = engine.generate(dimensions)

        # Expect 4 combinations: 2 * 2
        assert len(result) == 4
        values = {tuple(sorted(item.items())) for item in result}
        assert (("a", 1), ("b", "x")) in values
        assert (("a", 2), ("b", "y")) in values


class TestPairwiseEnginePictFallbacks:
    """Tests covering PICT fallbacks and error handling."""

    def test_generate_pict_importerror_falls_back_to_naive(self, monkeypatch) -> None:
        """ImportError when importing allpairspy should fall back to naive generation."""
        dimensions = {"x": [1, 2]}

        # Force ImportError for allpairspy
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "allpairspy":
                raise ImportError("forced for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        engine = PairwiseEngine(engine_type="PICT")

        result = engine._generate_pict(dimensions, seed=None)

        # Fallback uses naive generation; for single parameter,
        # it should simply return all values.
        assert len(result) == 2
        assert {"x": 1} in result
        assert {"x": 2} in result

    def test_generate_pict_other_exception_raises_case_generation_error(self, monkeypatch) -> None:
        """Any other exception inside PICT generation should raise CaseGenerationError."""
        dimensions = {"x": [1, 2], "y": ["a", "b"]}

        # Provide a fake allpairspy module whose AllPairs raises RuntimeError
        fake_module = types.ModuleType("allpairspy")

        class FailingAllPairs:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("boom")

        fake_module.AllPairs = FailingAllPairs
        monkeypatch.setitem(sys.modules, "allpairspy", fake_module)

        engine = PairwiseEngine(engine_type="PICT")

        with pytest.raises(CaseGenerationError) as exc_info:
            engine._generate_pict(dimensions, seed=None)

        assert "PICT generation failed" in str(exc_info.value)

    def test_generate_pict_success_with_fake_allpairspy(self, monkeypatch) -> None:
        """Successful PICT generation path using a fake allpairspy module."""
        dimensions = {"a": [1, 2], "b": [3]}

        fake_module = types.ModuleType("allpairspy")

        class FakeAllPairs:
            def __init__(self, param_values):
                self._param_values = param_values

            def __iter__(self):
                # Yield combinations in the same structure as AllPairs would
                yield [1, 3]
                yield [2, 3]

        fake_module.AllPairs = FakeAllPairs
        monkeypatch.setitem(sys.modules, "allpairspy", fake_module)

        engine = PairwiseEngine(engine_type="PICT")

        combos = engine._generate_pict(dimensions, seed=None)

        assert combos == [
            {"a": 1, "b": 3},
            {"a": 2, "b": 3},
        ]


class TestPairwiseEngineAdvanced:
    """Tests for hierarchical strategy and caching behaviour."""

    def test_generate_high_dimensional_uses_hierarchical_strategy(self, monkeypatch) -> None:
        """When dimensions > 10, engine should use hierarchical strategy."""
        # Create 11 dimensions to trigger hierarchical path
        dimensions = {f"p{i}": [i, i + 1] for i in range(11)}

        engine = PairwiseEngine(engine_type="PICT")

        # Stub _generate_pict to avoid real PICT logic / imports
        def fake_pict(self, dims, seed):
            # Return a single combination using first values
            return [{name: values[0] for name, values in dims.items()}]

        monkeypatch.setattr(PairwiseEngine, "_generate_pict", fake_pict)

        combos = engine.generate(dimensions)

        # Expect that hierarchical merged default dimensions back
        assert len(combos) == 1
        combo = combos[0]
        assert set(combo.keys()) == set(dimensions.keys())

    def test_generate_with_cache_hits_and_saves(self, tmp_path: Path, monkeypatch) -> None:
        """Verify that results are cached and reused across generate calls."""
        cache_dir = tmp_path / "cache"
        engine = PairwiseEngine(engine_type="naive", cache_dir=cache_dir)

        dimensions = {"a": [1, 2], "b": [3]}

        # Track calls to _generate_naive to ensure second call hits cache
        call_counter = {"count": 0}

        original_naive = PairwiseEngine._generate_naive

        def counting_naive(self, dims, seed):
            call_counter["count"] += 1
            return original_naive(self, dims, seed)

        monkeypatch.setattr(PairwiseEngine, "_generate_naive", counting_naive)

        # First call should generate and save to cache
        result1 = engine.generate(dimensions)
        assert len(result1) == 2
        assert call_counter["count"] == 1

        # Second call with same dimensions/seed should load from cache,
        # not call _generate_naive again.
        result2 = engine.generate(dimensions)
        assert len(result2) == 2
        assert call_counter["count"] == 1

    def test_generate_pict_path_called_from_generate(self, monkeypatch) -> None:
        """engine_type='PICT' in generate should call _generate_pict."""
        dimensions = {"a": [1]}

        engine = PairwiseEngine(engine_type="PICT")

        calls = {"count": 0}

        def fake_pict(self, dims, seed):
            calls["count"] += 1
            return [{"a": 1}]

        monkeypatch.setattr(PairwiseEngine, "_generate_pict", fake_pict)

        combos = engine.generate(dimensions)

        assert calls["count"] == 1
        assert combos == [{"a": 1}]

    def test_generate_ipo_path_calls_pict(self, monkeypatch) -> None:
        """engine_type='IPO' should route through _generate_ipo -> _generate_pict."""
        dimensions = {"a": [1]}

        engine = PairwiseEngine(engine_type="IPO")

        calls = {"count": 0}

        def fake_pict(self, dims, seed):
            calls["count"] += 1
            return [{"a": 1}]

        monkeypatch.setattr(PairwiseEngine, "_generate_pict", fake_pict)

        combos = engine.generate(dimensions)

        assert calls["count"] == 1
        assert combos == [{"a": 1}]

    def test_generate_max_cases_with_and_without_seed(self) -> None:
        """max_cases limit should work for both seeded and unseeded sampling."""
        dimensions = {
            "a": list(range(5)),
            "b": [0, 1],
        }
        # 5 * 2 = 10 combinations

        engine = PairwiseEngine(engine_type="naive")

        # With seed: uses random.sample
        seeded = engine.generate(dimensions, seed=123, max_cases=3)
        assert len(seeded) == 3

        # Without seed: uses slicing
        unseeded = engine.generate(dimensions, max_cases=3)
        assert len(unseeded) == 3

    def test_load_from_cache_without_cache_dir_returns_none(self) -> None:
        """_load_from_cache should return None when cache_dir is not set."""
        engine = PairwiseEngine(cache_dir=None)

        assert engine._load_from_cache("any") is None

    def test_load_from_cache_handles_unpickling_error(self, tmp_path: Path) -> None:
        """Errors during cache load should be swallowed and return None."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        engine = PairwiseEngine(cache_dir=cache_dir)
        key = "bad"
        cache_file = cache_dir / f"{key}.pkl"
        cache_file.write_text("not a pickle", encoding="utf-8")

        result = engine._load_from_cache(key)

        assert result is None

    def test_save_to_cache_without_cache_dir_returns_immediately(self, tmp_path: Path) -> None:
        """_save_to_cache should be a no-op when cache_dir is None."""
        engine = PairwiseEngine(cache_dir=None)

        # Should not raise
        engine._save_to_cache("key", [{"a": 1}])

    def test_save_to_cache_handles_write_error(self, tmp_path: Path, monkeypatch) -> None:
        """Errors during cache save should be logged but not raised."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        engine = PairwiseEngine(cache_dir=cache_dir)

        def failing_open(*args, **kwargs):
            raise OSError("disk error")

        # Patch built-in open used by pairwise_engine
        import builtins

        monkeypatch.setattr(builtins, "open", failing_open)

        # Should not raise despite write error
        engine._save_to_cache("key", [{"a": 1}])
