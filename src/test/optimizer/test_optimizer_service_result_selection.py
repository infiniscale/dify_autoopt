"""Tests for OptimizerService result selection logic.

Tests for _is_better_result() method covering:
- First result selection
- Confidence threshold priority
- Score comparison priority
- Confidence as tie-breaker
"""

import pytest

from src.optimizer.models import (
    OptimizationResult,
    OptimizationStrategy,
    OptimizationChange,
)
from src.optimizer.optimizer_service import OptimizerService


class TestIsBetterResult:
    """Test _is_better_result() method."""

    def test_is_better_result_first_result(self, mock_catalog):
        """Test first result is always selected (None comparison)."""
        service = OptimizerService(catalog=mock_catalog)

        candidate = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Improved Test",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=5.0,
            confidence=0.5,
            changes=[
                OptimizationChange(rule_id="TEST_RULE_1", description="change1")
            ],
            metadata={"optimized_score": 85.0},
        )

        # First result vs None
        result = service._is_better_result(
            candidate=candidate, current_best=None, min_confidence=0.6
        )

        assert result is True

    def test_is_better_result_confidence_priority(self, mock_catalog):
        """Test confidence threshold takes priority over score."""
        service = OptimizerService(catalog=mock_catalog)

        # Candidate meets threshold, current doesn't
        candidate = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Candidate",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=5.0,
            confidence=0.7,  # Meets threshold
            changes=[
                OptimizationChange(rule_id="TEST_RULE_1", description="change1")
            ],
            metadata={"optimized_score": 85.0},
        )

        current_best = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Current",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=10.0,  # Higher score but...
            confidence=0.5,  # ...doesn't meet threshold
            changes=[
                OptimizationChange(rule_id="TEST_RULE_2", description="change2")
            ],
            metadata={"optimized_score": 90.0},
        )

        # Candidate should win (meets threshold)
        result = service._is_better_result(
            candidate=candidate,
            current_best=current_best,
            min_confidence=0.6,
        )

        assert result is True

        # Reverse: current meets threshold, candidate doesn't
        result2 = service._is_better_result(
            candidate=current_best,  # Swap
            current_best=candidate,
            min_confidence=0.6,
        )

        assert result2 is False

    def test_is_better_result_score_priority(self, mock_catalog):
        """Test score comparison when both meet/fail threshold."""
        service = OptimizerService(catalog=mock_catalog)

        # Both meet threshold, candidate has higher score
        candidate = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Candidate",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=8.0,
            confidence=0.8,
            changes=[
                OptimizationChange(rule_id="TEST_RULE_1", description="change1")
            ],
            metadata={"optimized_score": 90.0},  # Higher score
        )

        current_best = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Current",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=5.0,
            confidence=0.75,
            changes=[
                OptimizationChange(rule_id="TEST_RULE_2", description="change2")
            ],
            metadata={"optimized_score": 85.0},
        )

        # Candidate should win (higher score, both meet threshold)
        result = service._is_better_result(
            candidate=candidate,
            current_best=current_best,
            min_confidence=0.7,
        )

        assert result is True

        # Both fail threshold, candidate has higher score
        result2 = service._is_better_result(
            candidate=candidate,
            current_best=current_best,
            min_confidence=0.9,  # Neither meets this
        )

        assert result2 is True

    def test_is_better_result_confidence_tiebreaker(self, mock_catalog):
        """Test confidence breaks ties when scores are equal."""
        service = OptimizerService(catalog=mock_catalog)

        # Same score, candidate has higher confidence
        candidate = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Candidate",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=8.0,
            confidence=0.85,  # Higher confidence
            changes=[
                OptimizationChange(rule_id="TEST_RULE_1", description="change1")
            ],
            metadata={"optimized_score": 88.0},
        )

        current_best = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Current",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=5.0,
            confidence=0.75,
            changes=[
                OptimizationChange(rule_id="TEST_RULE_2", description="change2")
            ],
            metadata={"optimized_score": 88.0},  # Same score
        )

        # Candidate should win (higher confidence as tie-breaker)
        result = service._is_better_result(
            candidate=candidate,
            current_best=current_best,
            min_confidence=0.7,
        )

        assert result is True

    def test_is_better_result_tolerance(self, mock_catalog):
        """Test 1-point tolerance in score comparison."""
        service = OptimizerService(catalog=mock_catalog)

        # Scores within 1-point tolerance
        candidate = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Candidate",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=8.0,
            confidence=0.75,
            changes=[
                OptimizationChange(rule_id="TEST_RULE_1", description="change1")
            ],
            metadata={"optimized_score": 88.5},
        )

        current_best = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Current",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=5.0,
            confidence=0.70,
            changes=[
                OptimizationChange(rule_id="TEST_RULE_2", description="change2")
            ],
            metadata={"optimized_score": 88.0},
        )

        # Within tolerance, should fall through to confidence tie-breaker
        result = service._is_better_result(
            candidate=candidate,
            current_best=current_best,
            min_confidence=0.6,
        )

        # Candidate wins on confidence (0.75 > 0.70)
        assert result is True


@pytest.fixture
def mock_catalog():
    """Create a mock catalog."""
    from unittest.mock import Mock

    catalog = Mock()
    return catalog
