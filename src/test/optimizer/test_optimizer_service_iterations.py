"""Tests for OptimizerService iteration logic.

Tests for _optimize_with_iterations() method covering:
- Early stop on confidence threshold
- Early stop on convergence
- Max iterations limit
- Best result selection
- Iteration with default config
"""

import pytest
from unittest.mock import Mock, patch

from src.optimizer.models import (
    OptimizationResult,
    OptimizationStrategy,
    Prompt,
    OptimizationChange,
)
from src.optimizer.optimizer_service import OptimizerService


class TestOptimizeWithIterations:
    """Test _optimize_with_iterations() method."""

    def test_early_stop_on_confidence(self, sample_prompt, mock_catalog):
        """Test iteration stops when min_confidence is reached."""
        service = OptimizerService(catalog=mock_catalog)

        # Mock engine to return increasing confidence
        results = [
            OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt="Improved Test v1",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=5.0,
                confidence=0.5,
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_1", description="change1")
                ],
                metadata={"optimized_score": 85.0},
            ),
            OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Improved Test v1",
                optimized_prompt="Improved Test v2",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=8.0,
                confidence=0.9,  # Exceeds min_confidence
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_2", description="change2")
                ],
                metadata={"optimized_score": 90.0},
            ),
        ]

        call_count = 0

        def mock_optimize(prompt, strategy):
            nonlocal call_count
            result = results[call_count]
            call_count += 1
            return result

        service._engine.optimize = mock_optimize

        # Run with min_confidence=0.7, max_iterations=5
        result = service._optimize_with_iterations(
            prompt=sample_prompt,
            strategy="clarity_focus",
            max_iterations=5,
            min_confidence=0.7,
        )

        # Should stop after 2 iterations (not 5)
        assert call_count == 2
        assert result.confidence == 0.9
        assert result.optimized_prompt == "Improved Test v2"

    def test_early_stop_on_convergence(self, sample_prompt, mock_catalog):
        """Test iteration stops when no improvement is detected."""
        service = OptimizerService(catalog=mock_catalog)

        # Mock engine to return decreasing improvements
        results = [
            OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt="Improved Test v1",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=5.0,
                confidence=0.6,
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_1", description="change1")
                ],
                metadata={"optimized_score": 85.0},
            ),
            OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Improved Test v1",
                optimized_prompt="Improved Test v2",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=0.0,  # No improvement - convergence
                confidence=0.65,
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_2", description="change2")
                ],
                metadata={"optimized_score": 85.0},
            ),
        ]

        call_count = 0

        def mock_optimize(prompt, strategy):
            nonlocal call_count
            result = results[call_count]
            call_count += 1
            return result

        service._engine.optimize = mock_optimize

        # Run with low min_confidence, high max_iterations
        result = service._optimize_with_iterations(
            prompt=sample_prompt,
            strategy="clarity_focus",
            max_iterations=10,
            min_confidence=0.9,
        )

        # Should stop after 2 iterations due to convergence (not 10)
        assert call_count == 2
        # Should return best result (v1 with improvement_score=5.0)
        assert result.improvement_score == 5.0

    def test_max_iterations_limit(self, sample_prompt, mock_catalog):
        """Test iteration respects max_iterations limit."""
        service = OptimizerService(catalog=mock_catalog)

        # Mock engine to always return low confidence
        def mock_optimize(prompt, strategy):
            return OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt="Improved Test",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=2.0,
                confidence=0.4,  # Never reaches min_confidence
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_1", description="change1")
                ],
                metadata={"optimized_score": 82.0},
            )

        call_count = 0

        def counting_optimize(prompt, strategy):
            nonlocal call_count
            call_count += 1
            return mock_optimize(prompt, strategy)

        service._engine.optimize = counting_optimize

        # Run with max_iterations=3
        result = service._optimize_with_iterations(
            prompt=sample_prompt,
            strategy="clarity_focus",
            max_iterations=3,
            min_confidence=0.7,
        )

        # Should run exactly 3 iterations
        assert call_count == 3
        assert result is not None
        assert result.confidence == 0.4

    def test_returns_best_result(self, sample_prompt, mock_catalog):
        """Test returns best result across all iterations."""
        service = OptimizerService(catalog=mock_catalog)

        # Mock engine to return varying scores
        results = [
            OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt="Improved Test v1",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=5.0,
                confidence=0.5,
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_1", description="change1")
                ],
                metadata={"optimized_score": 85.0},
            ),
            OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Improved Test v1",
                optimized_prompt="Improved Test v2",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=10.0,  # Best score
                confidence=0.6,
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_2", description="change2")
                ],
                metadata={"optimized_score": 90.0},
            ),
            OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Improved Test v2",
                optimized_prompt="Improved Test v3",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=3.0,
                confidence=0.55,
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_3", description="change3")
                ],
                metadata={"optimized_score": 88.0},
            ),
        ]

        call_count = 0

        def mock_optimize(prompt, strategy):
            nonlocal call_count
            result = results[call_count]
            call_count += 1
            return result

        service._engine.optimize = mock_optimize

        # Run with low min_confidence
        result = service._optimize_with_iterations(
            prompt=sample_prompt,
            strategy="clarity_focus",
            max_iterations=3,
            min_confidence=0.9,
        )

        # Should return v2 (best improvement_score = 10.0)
        assert result.improvement_score == 10.0
        assert result.optimized_prompt == "Improved Test v2"

    def test_iteration_with_default_config(self, sample_prompt, mock_catalog):
        """Test iteration works with default configuration values."""
        service = OptimizerService(catalog=mock_catalog)

        # Mock engine
        def mock_optimize(prompt, strategy):
            return OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt="Improved Test",
                strategy=OptimizationStrategy.AUTO,
                improvement_score=8.0,
                confidence=0.75,
                changes=[
                    OptimizationChange(rule_id="TEST_RULE_1", description="change1")
                ],
                metadata={"optimized_score": 88.0},
            )

        service._engine.optimize = mock_optimize

        # Run with typical default values
        result = service._optimize_with_iterations(
            prompt=sample_prompt,
            strategy="auto",
            max_iterations=3,  # Default from OptimizationConfig
            min_confidence=0.6,  # Default from OptimizationConfig
        )

        # Should complete successfully
        assert result is not None
        assert result.confidence == 0.75
        assert result.improvement_score == 8.0


@pytest.fixture
def sample_prompt():
    """Create a sample prompt for testing."""
    return Prompt(
        id="test_prompt_1",
        workflow_id="wf_001",
        node_id="node_1",
        node_type="llm",
        text="Original test prompt",
        role="system",
        variables=[],
        context={},
    )


@pytest.fixture
def mock_catalog():
    """Create a mock catalog."""
    catalog = Mock()
    # Add any necessary mock setup
    return catalog
