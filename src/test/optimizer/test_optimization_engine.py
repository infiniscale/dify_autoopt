"""
Test Cases for OptimizationEngine

Date: 2025-11-17
Author: qa-engineer
Description: Unit tests for prompt optimization engine
"""

import pytest

from src.optimizer.optimization_engine import OptimizationEngine
from src.optimizer.models import Prompt, OptimizationStrategy
from src.optimizer.exceptions import InvalidStrategyError, OptimizationFailedError


class TestOptimizationEngineBasic:
    """Basic test cases for OptimizationEngine."""

    def test_engine_initialization(self, engine):
        """Test that OptimizationEngine can be initialized."""
        assert engine is not None
        assert isinstance(engine, OptimizationEngine)

    def test_optimize_with_clarity_focus(self, engine, sample_prompt):
        """Test optimization with clarity_focus strategy."""
        result = engine.optimize(sample_prompt, "clarity_focus")
        assert result is not None
        assert result.strategy == OptimizationStrategy.CLARITY_FOCUS
        assert result.prompt_id == sample_prompt.id
        assert 0.0 <= result.confidence <= 1.0

    def test_optimize_with_efficiency_focus(self, engine, sample_prompt):
        """Test optimization with efficiency_focus strategy."""
        result = engine.optimize(sample_prompt, "efficiency_focus")
        assert result.strategy == OptimizationStrategy.EFFICIENCY_FOCUS

    def test_optimize_with_structure_focus(self, engine, sample_prompt):
        """Test optimization with structure_focus strategy."""
        result = engine.optimize(sample_prompt, "structure_focus")
        assert result.strategy == OptimizationStrategy.STRUCTURE_FOCUS

    def test_optimize_invalid_strategy_raises_error(self, engine, sample_prompt):
        """Test that invalid strategy raises error."""
        with pytest.raises(InvalidStrategyError):
            engine.optimize(sample_prompt, "invalid_strategy")

    def test_optimize_result_structure(self, engine, sample_prompt):
        """Test optimization result has correct structure."""
        result = engine.optimize(sample_prompt, "clarity_focus")
        assert result.original_prompt == sample_prompt.text
        assert len(result.optimized_prompt) > 0
        assert isinstance(result.improvement_score, float)
        assert isinstance(result.changes, list)
        assert "original_score" in result.metadata
        assert "optimized_score" in result.metadata


class TestClarityFocusOptimization:
    """Test cases for clarity-focused optimization."""

    def test_clarity_adds_structure(self, engine, sample_prompt_short):
        """Test that clarity focus adds structure."""
        result = engine.optimize(sample_prompt_short, "clarity_focus")
        # Should add headers or formatting
        assert len(result.optimized_prompt) >= len(result.original_prompt)

    def test_clarity_replaces_vague_terms(self, engine, sample_prompt_vague):
        """Test that vague terms are replaced."""
        result = engine.optimize(sample_prompt_vague, "clarity_focus")
        # Check that some vague words are replaced
        assert "maybe" not in result.optimized_prompt.lower() or "stuff" not in result.optimized_prompt.lower()


class TestEfficiencyFocusOptimization:
    """Test cases for efficiency-focused optimization."""

    def test_efficiency_reduces_length(self, engine, sample_prompt_long):
        """Test that efficiency focus reduces length."""
        result = engine.optimize(sample_prompt_long, "efficiency_focus")
        # Should reduce length
        assert len(result.optimized_prompt) <= len(result.original_prompt)


class TestStructureFocusOptimization:
    """Test cases for structure-focused optimization."""

    def test_structure_adds_formatting(self, engine, sample_prompt):
        """Test that structure focus adds formatting."""
        result = engine.optimize(sample_prompt, "structure_focus")
        # Should add markdown headers or structure
        assert "#" in result.optimized_prompt or "1." in result.optimized_prompt


class TestConfidenceCalculation:
    """Test cases for confidence calculation."""

    def test_confidence_in_valid_range(self, engine, sample_prompt):
        """Test that confidence is in valid range."""
        result = engine.optimize(sample_prompt, "clarity_focus")
        assert 0.0 <= result.confidence <= 1.0

    def test_high_improvement_gives_high_confidence(self, engine, sample_prompt_vague):
        """Test that high improvement gives high confidence."""
        result = engine.optimize(sample_prompt_vague, "clarity_focus")
        # Vague prompt should have significant improvement
        if result.improvement_score > 10:
            assert result.confidence >= 0.5


class TestChangeDetection:
    """Test cases for change detection."""

    def test_changes_list_not_empty(self, engine, sample_prompt):
        """Test that changes are detected."""
        result = engine.optimize(sample_prompt, "clarity_focus")
        assert len(result.changes) > 0

    def test_changes_describe_modifications(self, engine, sample_prompt):
        """Test that changes describe what was modified."""
        result = engine.optimize(sample_prompt, "structure_focus")
        changes_text = " ".join(result.changes).lower()
        # Should mention structural changes
        assert any(word in changes_text for word in ["header", "structure", "format", "list"])


class TestVariablePreservation:
    """Test cases for variable preservation during optimization."""

    def test_variables_preserved_clarity(self, engine, sample_prompt_with_multiple_vars):
        """Test that variables are preserved in clarity optimization."""
        result = engine.optimize(sample_prompt_with_multiple_vars, "clarity_focus")
        # All original variables should still be present
        for var in sample_prompt_with_multiple_vars.variables:
            assert f"{{{{{var}}}}}" in result.optimized_prompt

    def test_variables_preserved_efficiency(self, engine, sample_prompt_with_multiple_vars):
        """Test that variables are preserved in efficiency optimization."""
        result = engine.optimize(sample_prompt_with_multiple_vars, "efficiency_focus")
        for var in sample_prompt_with_multiple_vars.variables:
            assert f"{{{{{var}}}}}" in result.optimized_prompt


class TestImprovementScoring:
    """Test cases for improvement score calculation."""

    def test_improvement_score_calculated(self, engine, sample_prompt):
        """Test that improvement score is calculated."""
        result = engine.optimize(sample_prompt, "clarity_focus")
        assert isinstance(result.improvement_score, float)
        # Can be positive, negative, or zero
        assert -100.0 <= result.improvement_score <= 100.0

    def test_vague_prompt_shows_improvement(self, engine, sample_prompt_vague):
        """Test that optimizing vague prompt shows improvement."""
        result = engine.optimize(sample_prompt_vague, "clarity_focus")
        # Should improve vague prompts
        assert result.improvement_score > 0


class TestMetadata:
    """Test cases for optimization metadata."""

    def test_metadata_includes_scores(self, engine, sample_prompt):
        """Test that metadata includes before/after scores."""
        result = engine.optimize(sample_prompt, "clarity_focus")
        assert "original_score" in result.metadata
        assert "optimized_score" in result.metadata
        assert "original_clarity" in result.metadata
        assert "optimized_clarity" in result.metadata
