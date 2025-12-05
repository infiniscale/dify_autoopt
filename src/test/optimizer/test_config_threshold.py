"""
Test Cases for OptimizationConfig score_threshold Field

Date: 2025-11-17
Author: backend-developer
Description: Comprehensive tests for score_threshold configuration and _should_optimize logic
"""

import pytest
from pydantic import ValidationError

from src.optimizer.models import OptimizationConfig, OptimizationStrategy, PromptAnalysis
from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.interfaces.llm_client import StubLLMClient
from src.optimizer.interfaces.storage import InMemoryStorage


class TestScoreThresholdDefaultValue:
    """Test default score_threshold value."""

    def test_default_score_threshold(self):
        """Test default score_threshold is 80.0"""
        config = OptimizationConfig()
        assert config.score_threshold == 80.0

    def test_default_score_threshold_with_other_params(self):
        """Test default score_threshold when other params are specified"""
        config = OptimizationConfig(
            strategies=[OptimizationStrategy.CLARITY_FOCUS],
            min_confidence=0.7,
            max_iterations=5
        )
        assert config.score_threshold == 80.0


class TestScoreThresholdCustomValue:
    """Test custom score_threshold configuration."""

    def test_custom_score_threshold_low(self):
        """Test custom score_threshold with low value"""
        config = OptimizationConfig(score_threshold=70.0)
        assert config.score_threshold == 70.0

    def test_custom_score_threshold_high(self):
        """Test custom score_threshold with high value"""
        config = OptimizationConfig(score_threshold=90.0)
        assert config.score_threshold == 90.0

    def test_custom_score_threshold_zero(self):
        """Test score_threshold can be set to 0.0"""
        config = OptimizationConfig(score_threshold=0.0)
        assert config.score_threshold == 0.0

    def test_custom_score_threshold_hundred(self):
        """Test score_threshold can be set to 100.0"""
        config = OptimizationConfig(score_threshold=100.0)
        assert config.score_threshold == 100.0


class TestScoreThresholdValidation:
    """Test score_threshold range validation."""

    def test_score_threshold_minimum_bound(self):
        """Test score_threshold accepts minimum valid value (0.0)"""
        config = OptimizationConfig(score_threshold=0.0)
        assert config.score_threshold == 0.0

    def test_score_threshold_maximum_bound(self):
        """Test score_threshold accepts maximum valid value (100.0)"""
        config = OptimizationConfig(score_threshold=100.0)
        assert config.score_threshold == 100.0

    def test_score_threshold_negative_raises_error(self):
        """Test score_threshold rejects negative values"""
        with pytest.raises(ValidationError) as exc_info:
            OptimizationConfig(score_threshold=-1.0)

        # Verify the error message mentions the field and constraint
        error_msg = str(exc_info.value)
        assert "score_threshold" in error_msg.lower()

    def test_score_threshold_above_hundred_raises_error(self):
        """Test score_threshold rejects values above 100.0"""
        with pytest.raises(ValidationError) as exc_info:
            OptimizationConfig(score_threshold=101.0)

        # Verify the error message mentions the field and constraint
        error_msg = str(exc_info.value)
        assert "score_threshold" in error_msg.lower()

    def test_score_threshold_far_negative_raises_error(self):
        """Test score_threshold rejects far negative values"""
        with pytest.raises(ValidationError):
            OptimizationConfig(score_threshold=-50.0)

    def test_score_threshold_far_above_hundred_raises_error(self):
        """Test score_threshold rejects far above 100 values"""
        with pytest.raises(ValidationError):
            OptimizationConfig(score_threshold=200.0)


class TestShouldOptimizeWithThreshold:
    """Test _should_optimize method respects score_threshold."""

    def test_should_optimize_score_below_default_threshold(self, optimizer_service):
        """Test optimization triggered when score is below default threshold (80.0)"""
        # Create analysis with score 75.0 (below default 80.0)
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=75.0,
            clarity_score=75.0,
            efficiency_score=75.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # With default threshold (80.0) - should optimize
        config = OptimizationConfig()
        assert optimizer_service._should_optimize(analysis, None, config) is True

    def test_should_not_optimize_score_above_default_threshold(self, optimizer_service):
        """Test optimization skipped when score is above default threshold (80.0)"""
        # Create analysis with score 85.0 (above default 80.0)
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=85.0,
            clarity_score=85.0,
            efficiency_score=85.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # With default threshold (80.0) - should NOT optimize
        config = OptimizationConfig()
        assert optimizer_service._should_optimize(analysis, None, config) is False

    def test_should_optimize_with_custom_high_threshold(self, optimizer_service):
        """Test optimization triggered with custom high threshold"""
        # Create analysis with score 75.0
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=75.0,
            clarity_score=75.0,
            efficiency_score=75.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # With higher threshold (90.0) - should optimize (75 < 90)
        config = OptimizationConfig(score_threshold=90.0)
        assert optimizer_service._should_optimize(analysis, None, config) is True

    def test_should_not_optimize_with_custom_low_threshold(self, optimizer_service):
        """Test optimization skipped with custom low threshold"""
        # Create analysis with score 75.0
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=75.0,
            clarity_score=75.0,
            efficiency_score=75.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # With lower threshold (70.0) - should NOT optimize (75 >= 70)
        config = OptimizationConfig(score_threshold=70.0)
        assert optimizer_service._should_optimize(analysis, None, config) is False

    def test_should_optimize_score_exactly_at_threshold(self, optimizer_service):
        """Test optimization behavior when score equals threshold"""
        # Create analysis with score 80.0 (exactly at default threshold)
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=80.0,
            clarity_score=80.0,
            efficiency_score=80.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # At threshold - should NOT optimize (not below threshold)
        config = OptimizationConfig(score_threshold=80.0)
        assert optimizer_service._should_optimize(analysis, None, config) is False

    def test_should_optimize_no_config_uses_default(self, optimizer_service):
        """Test that missing config uses default threshold (80.0)"""
        # Create analysis with score 75.0 (below default 80.0)
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=75.0,
            clarity_score=75.0,
            efficiency_score=75.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # No config provided - should use default threshold 80.0
        assert optimizer_service._should_optimize(analysis, None, None) is True

    def test_should_not_optimize_no_config_high_score(self, optimizer_service):
        """Test that missing config with high score skips optimization"""
        # Create analysis with score 85.0 (above default 80.0)
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=85.0,
            clarity_score=85.0,
            efficiency_score=85.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # No config provided - should use default threshold 80.0
        assert optimizer_service._should_optimize(analysis, None, None) is False


class TestShouldOptimizeEdgeCases:
    """Test edge cases for _should_optimize with threshold."""

    def test_should_optimize_with_zero_threshold(self, optimizer_service):
        """Test optimization with threshold set to 0.0"""
        # Create analysis with score 50.0
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=50.0,
            clarity_score=50.0,
            efficiency_score=50.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # With threshold 0.0 - should NOT optimize (50 >= 0)
        config = OptimizationConfig(score_threshold=0.0)
        assert optimizer_service._should_optimize(analysis, None, config) is False

    def test_should_optimize_with_hundred_threshold(self, optimizer_service):
        """Test optimization with threshold set to 100.0"""
        # Create analysis with score 95.0
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=95.0,
            clarity_score=95.0,
            efficiency_score=95.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # With threshold 100.0 - should optimize (95 < 100)
        config = OptimizationConfig(score_threshold=100.0)
        assert optimizer_service._should_optimize(analysis, None, config) is True

    def test_should_optimize_very_low_score_any_threshold(self, optimizer_service):
        """Test that very low scores always trigger optimization"""
        # Create analysis with score 10.0
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=10.0,
            clarity_score=10.0,
            efficiency_score=10.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # Test with various thresholds
        for threshold in [20.0, 50.0, 80.0, 90.0]:
            config = OptimizationConfig(score_threshold=threshold)
            assert optimizer_service._should_optimize(analysis, None, config) is True

    def test_should_not_optimize_perfect_score(self, optimizer_service):
        """Test that perfect score (100.0) skips optimization with default threshold"""
        # Create analysis with score 100.0
        analysis = PromptAnalysis(
            prompt_id="test_001",
            overall_score=100.0,
            clarity_score=100.0,
            efficiency_score=100.0,
            issues=[],
            suggestions=[],
            metadata={}
        )

        # With default threshold (80.0) - should NOT optimize (100 >= 80)
        config = OptimizationConfig()
        assert optimizer_service._should_optimize(analysis, None, config) is False


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_config_without_score_threshold_uses_default(self):
        """Test that existing OptimizationConfig usage still works"""
        # Old way of creating config (without score_threshold)
        config = OptimizationConfig(
            strategies=[OptimizationStrategy.CLARITY_FOCUS],
            min_confidence=0.7,
            max_iterations=3
        )

        # Should have default value
        assert config.score_threshold == 80.0

    def test_existing_service_calls_still_work(self, optimizer_service, sample_analysis):
        """Test that existing service usage patterns still work"""
        # Old pattern: calling _should_optimize with None config
        result = optimizer_service._should_optimize(sample_analysis, None, None)
        assert isinstance(result, bool)

    def test_config_with_all_fields_explicit(self):
        """Test config creation with all fields explicitly set"""
        config = OptimizationConfig(
            strategies=[OptimizationStrategy.EFFICIENCY_FOCUS],
            min_confidence=0.8,
            max_iterations=5,
            score_threshold=85.0,
            analysis_rules={"custom": "rule"},
            metadata={"author": "test"}
        )

        assert config.strategies == [OptimizationStrategy.EFFICIENCY_FOCUS]
        assert config.min_confidence == 0.8
        assert config.max_iterations == 5
        assert config.score_threshold == 85.0
        assert config.analysis_rules == {"custom": "rule"}
        assert config.metadata == {"author": "test"}
