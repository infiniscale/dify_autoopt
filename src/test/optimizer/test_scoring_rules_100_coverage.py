"""
Test Coverage for ScoringRules - Achieving 100% Coverage

Date: 2025-11-18
Author: backend-developer
Description: Tests for all uncovered code paths in ScoringRules
"""

import pytest
from src.optimizer.models import IssueSeverity, IssueType, PromptAnalysis, PromptIssue
from src.optimizer.scoring_rules import ScoringRules


class TestScoringRulesFromConfig:
    """Test ScoringRules.from_config() classmethod."""

    def test_from_config_with_valid_fields(self):
        """Test loading from config dict with valid fields."""
        config_dict = {
            "optimization_threshold": 75.0,
            "critical_issue_threshold": 2,
            "clarity_efficiency_gap": 15.0,
            "min_confidence": 0.7,
        }

        rules = ScoringRules.from_config(config_dict)

        assert rules.optimization_threshold == 75.0
        assert rules.critical_issue_threshold == 2
        assert rules.clarity_efficiency_gap == 15.0
        assert rules.min_confidence == 0.7

    def test_from_config_with_invalid_fields_ignored(self):
        """Test that invalid fields are filtered out."""
        config_dict = {
            "optimization_threshold": 75.0,
            "invalid_field": "should_be_ignored",
            "another_invalid": 123,
        }

        rules = ScoringRules.from_config(config_dict)

        assert rules.optimization_threshold == 75.0
        # Invalid fields ignored, defaults used
        assert rules.critical_issue_threshold == 1  # Default
        assert not hasattr(rules, "invalid_field")

    def test_from_config_empty_dict_uses_defaults(self):
        """Test that empty config dict uses all defaults."""
        rules = ScoringRules.from_config({})

        assert rules.optimization_threshold == 80.0
        assert rules.critical_issue_threshold == 1
        assert rules.clarity_efficiency_gap == 10.0


class TestScoringRulesSelectStrategy:
    """Test select_strategy() edge cases."""

    def test_select_strategy_structure_focus_both_low(self):
        """Test structure_focus when both scores are low."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=60.0,
            clarity_score=65.0,  # < 70 (low_score_threshold)
            efficiency_score=68.0,  # < 70
            issues=[],
            suggestions=[],
        )

        rules = ScoringRules()
        strategy = rules.select_strategy(analysis)

        assert strategy == "structure_focus"

    def test_select_strategy_default_clarity_focus(self):
        """Test default to clarity_focus when scores are balanced and high."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=85.0,
            clarity_score=84.0,
            efficiency_score=86.0,  # Difference < 10
            issues=[],
            suggestions=[],
        )

        rules = ScoringRules()
        strategy = rules.select_strategy(analysis)

        assert strategy == "clarity_focus"


class TestScoringRulesVersionBumpType:
    """Test version_bump_type() method."""

    def test_version_bump_major_improvement(self):
        """Test major version bump for large improvement."""
        rules = ScoringRules()

        bump_type = rules.version_bump_type(improvement_score=20.0)

        assert bump_type == "major"

    def test_version_bump_minor_improvement(self):
        """Test minor version bump for moderate improvement."""
        rules = ScoringRules()

        bump_type = rules.version_bump_type(improvement_score=7.0)

        assert bump_type == "minor"

    def test_version_bump_patch_improvement(self):
        """Test patch version bump for small improvement."""
        rules = ScoringRules()

        bump_type = rules.version_bump_type(improvement_score=2.0)

        assert bump_type == "patch"

    def test_version_bump_exact_threshold(self):
        """Test exact threshold values."""
        rules = ScoringRules(
            major_version_min_improvement=15.0,
            minor_version_min_improvement=5.0,
        )

        assert rules.version_bump_type(15.0) == "major"
        assert rules.version_bump_type(5.0) == "minor"
        assert rules.version_bump_type(4.99) == "patch"


class TestScoringRulesIsHighQuality:
    """Test is_high_quality() method."""

    def test_is_high_quality_both_thresholds_met(self):
        """Test high quality when both confidence and improvement meet thresholds."""
        rules = ScoringRules()

        result = rules.is_high_quality(confidence=0.85, improvement=7.0)

        assert result is True

    def test_is_high_quality_low_confidence(self):
        """Test not high quality when confidence is low."""
        rules = ScoringRules()

        result = rules.is_high_quality(confidence=0.75, improvement=10.0)

        assert result is False

    def test_is_high_quality_low_improvement(self):
        """Test not high quality when improvement is low."""
        rules = ScoringRules()

        result = rules.is_high_quality(confidence=0.9, improvement=3.0)

        assert result is False

    def test_is_high_quality_both_low(self):
        """Test not high quality when both are low."""
        rules = ScoringRules()

        result = rules.is_high_quality(confidence=0.6, improvement=2.0)

        assert result is False

    def test_is_high_quality_exact_thresholds(self):
        """Test exact threshold values."""
        rules = ScoringRules(
            high_confidence=0.8,
            minor_version_min_improvement=5.0,
        )

        assert rules.is_high_quality(0.8, 5.0) is True
        assert rules.is_high_quality(0.79, 5.0) is False
        assert rules.is_high_quality(0.8, 4.99) is False
