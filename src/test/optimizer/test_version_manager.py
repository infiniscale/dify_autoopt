"""
Test Cases for VersionManager

Date: 2025-11-17
Author: qa-engineer
Description: Unit tests for prompt version management
"""

import pytest
from datetime import datetime

from src.optimizer.version_manager import VersionManager
from src.optimizer.models import PromptVersion, Prompt, OptimizationResult, OptimizationStrategy
from src.optimizer.exceptions import VersionNotFoundError, VersionConflictError
from src.optimizer.scoring_rules import ScoringRules


class TestVersionManagerBasic:
    """Basic test cases for VersionManager."""

    def test_manager_initialization(self, version_manager):
        """Test that VersionManager can be initialized."""
        assert version_manager is not None
        assert isinstance(version_manager, VersionManager)

    def test_create_baseline_version(self, version_manager, sample_prompt, sample_analysis):
        """Test creating baseline version."""
        version = version_manager.create_version(
            prompt=sample_prompt,
            analysis=sample_analysis,
            optimization_result=None,
            parent_version=None,
        )
        assert version.version == "1.0.0"
        assert version.is_baseline()
        assert version.parent_version is None

    def test_create_second_version(self, version_manager, sample_prompt, sample_analysis):
        """Test creating second version after baseline."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        # Create second version
        v2 = version_manager.create_version(sample_prompt, sample_analysis, None, "1.0.0")
        assert v2.version == "1.0.1"

    def test_get_version(self, version_manager, sample_prompt, sample_analysis):
        """Test retrieving specific version."""
        created = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        retrieved = version_manager.get_version(sample_prompt.id, "1.0.0")
        assert retrieved.version == created.version
        assert retrieved.prompt_id == created.prompt_id

    def test_get_nonexistent_version_raises_error(self, version_manager):
        """Test that getting non-existent version raises error."""
        with pytest.raises(VersionNotFoundError):
            version_manager.get_version("nonexistent_id", "1.0.0")


class TestVersionHistory:
    """Test cases for version history."""

    def test_get_version_history(self, version_manager, sample_prompt, sample_analysis):
        """Test getting version history."""
        # Create multiple versions
        version_manager.create_version(sample_prompt, sample_analysis, None, None)
        version_manager.create_version(sample_prompt, sample_analysis, None, "1.0.0")
        version_manager.create_version(sample_prompt, sample_analysis, None, "1.0.1")

        history = version_manager.get_version_history(sample_prompt.id)
        assert len(history) == 3
        # Should be sorted by version
        assert history[0].version == "1.0.0"
        assert history[1].version == "1.0.1"
        assert history[2].version == "1.0.2"

    def test_get_latest_version(self, version_manager, sample_prompt, sample_analysis):
        """Test getting latest version."""
        version_manager.create_version(sample_prompt, sample_analysis, None, None)
        version_manager.create_version(sample_prompt, sample_analysis, None, "1.0.0")
        latest = version_manager.get_latest_version(sample_prompt.id)
        assert latest.version == "1.0.1"

    def test_get_latest_version_no_versions_raises_error(self, version_manager):
        """Test that getting latest with no versions raises error."""
        with pytest.raises(VersionNotFoundError):
            version_manager.get_latest_version("nonexistent")


class TestVersionComparison:
    """Test cases for version comparison."""

    def test_compare_versions(self, version_manager, sample_prompt, sample_analysis, sample_analysis_high_score):
        """Test comparing two versions."""
        version_manager.create_version(sample_prompt, sample_analysis, None, None)

        # Update prompt for second version
        updated_prompt = Prompt(
            id=sample_prompt.id,
            workflow_id=sample_prompt.workflow_id,
            node_id=sample_prompt.node_id,
            node_type=sample_prompt.node_type,
            text="Updated text",
            variables=sample_prompt.variables,
        )
        version_manager.create_version(updated_prompt, sample_analysis_high_score, None, "1.0.0")

        comparison = version_manager.compare_versions(sample_prompt.id, "1.0.0", "1.0.1")
        assert "improvement" in comparison
        assert "clarity_improvement" in comparison
        assert "efficiency_improvement" in comparison
        assert comparison["improvement"] > 0  # High score - low score


class TestRollback:
    """Test cases for version rollback."""

    def test_rollback_to_previous_version(self, version_manager, sample_prompt, sample_analysis):
        """Test rolling back to previous version."""
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        v2 = version_manager.create_version(sample_prompt, sample_analysis, None, "1.0.0")

        rolled_back = version_manager.rollback(sample_prompt.id, "1.0.0")
        assert rolled_back.version == "1.0.2"  # New version created
        assert rolled_back.metadata.get("rollback_to") == "1.0.0"


class TestBestVersion:
    """Test cases for finding best version."""

    def test_get_best_version(self, version_manager, sample_prompt, sample_analysis, sample_analysis_high_score):
        """Test getting version with highest score."""
        version_manager.create_version(sample_prompt, sample_analysis, None, None)

        updated_prompt = Prompt(
            id=sample_prompt.id,
            workflow_id=sample_prompt.workflow_id,
            node_id=sample_prompt.node_id,
            node_type=sample_prompt.node_type,
            text="Better text",
            variables=sample_prompt.variables,
        )
        version_manager.create_version(updated_prompt, sample_analysis_high_score, None, "1.0.0")

        best = version_manager.get_best_version(sample_prompt.id)
        assert best.analysis.overall_score == sample_analysis_high_score.overall_score


class TestVersionDeletion:
    """Test cases for version deletion."""

    def test_delete_version(self, version_manager, sample_prompt, sample_analysis):
        """Test deleting a version."""
        version_manager.create_version(sample_prompt, sample_analysis, None, None)
        deleted = version_manager.delete_version(sample_prompt.id, "1.0.0")
        assert deleted is True

        with pytest.raises(VersionNotFoundError):
            version_manager.get_version(sample_prompt.id, "1.0.0")

    def test_clear_all_versions(self, version_manager, sample_prompt, sample_analysis):
        """Test clearing all versions."""
        version_manager.create_version(sample_prompt, sample_analysis, None, None)
        version_manager.clear_all_versions()

        history = version_manager.get_version_history(sample_prompt.id)
        assert len(history) == 0


class TestSemanticVersioning:
    """Test cases for semantic versioning logic."""

    def test_baseline_version_is_1_0_0(self, version_manager, sample_prompt, sample_analysis):
        """Test that first version is always 1.0.0."""
        version = version_manager.create_version(
            prompt=sample_prompt,
            analysis=sample_analysis,
            optimization_result=None,
            parent_version=None
        )
        assert version.version == "1.0.0"

    def test_manual_edit_increments_patch(self, version_manager, sample_prompt, sample_analysis):
        """Test that manual edit without optimization increments patch version."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        assert v1.version == "1.0.0"

        # Manual edit (no optimization result)
        v2 = version_manager.create_version(sample_prompt, sample_analysis, None, "1.0.0")
        assert v2.version == "1.0.1"

        # Another manual edit
        v3 = version_manager.create_version(sample_prompt, sample_analysis, None, "1.0.1")
        assert v3.version == "1.0.2"

    def test_major_version_bump_for_large_improvement(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test major version bump for improvement >= 15.0 points."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        assert v1.version == "1.0.0"

        # Large improvement (20 points) -> major bump
        result = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="Optimized version",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=20.0,
            confidence=0.9,
            changes=[],
            metadata={"original_score": 70.0, "optimized_score": 90.0}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result, "1.0.0")
        assert v2.version == "2.0.0"

    def test_minor_version_bump_for_medium_improvement(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test minor version bump for improvement in [5.0, 15.0) range."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        assert v1.version == "1.0.0"

        # Medium improvement (10 points) -> minor bump
        result = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="Optimized version",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=10.0,
            confidence=0.8,
            changes=[],
            metadata={"original_score": 70.0, "optimized_score": 80.0}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result, "1.0.0")
        assert v2.version == "1.1.0"

    def test_patch_version_bump_for_small_improvement(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test patch version bump for improvement < 5.0 points."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        assert v1.version == "1.0.0"

        # Small improvement (2 points) -> patch bump
        result = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="Optimized version",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=2.0,
            confidence=0.7,
            changes=[],
            metadata={"original_score": 75.0, "optimized_score": 77.0}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result, "1.0.0")
        assert v2.version == "1.0.1"

    def test_major_bump_boundary_at_15_points(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test that improvement_score=15.0 triggers major bump."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)

        # Exactly 15.0 points -> major bump
        result = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="Optimized version",
            strategy=OptimizationStrategy.STRUCTURE_FOCUS,
            improvement_score=15.0,
            confidence=0.85,
            changes=[],
            metadata={}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result, "1.0.0")
        assert v2.version == "2.0.0"

    def test_minor_bump_boundary_at_5_points(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test that improvement_score=5.0 triggers minor bump."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)

        # Exactly 5.0 points -> minor bump
        result = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="Optimized version",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=5.0,
            confidence=0.75,
            changes=[],
            metadata={}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result, "1.0.0")
        assert v2.version == "1.1.0"

    def test_minor_bump_just_below_major_threshold(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test that 14.9 points triggers minor bump (not major)."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)

        # 14.9 points -> minor bump
        result = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="Optimized version",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=14.9,
            confidence=0.82,
            changes=[],
            metadata={}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result, "1.0.0")
        assert v2.version == "1.1.0"

    def test_patch_bump_just_below_minor_threshold(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test that 4.9 points triggers patch bump (not minor)."""
        # Create baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)

        # 4.9 points -> patch bump
        result = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="Optimized version",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=4.9,
            confidence=0.71,
            changes=[],
            metadata={}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result, "1.0.0")
        assert v2.version == "1.0.1"

    def test_multiple_major_bumps(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test multiple major version bumps: 1.0.0 -> 2.0.0 -> 3.0.0."""
        # Baseline
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        assert v1.version == "1.0.0"

        # First major improvement
        result1 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="Version 2",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=20.0,
            confidence=0.9,
            changes=[],
            metadata={}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result1, "1.0.0")
        assert v2.version == "2.0.0"

        # Second major improvement
        result2 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt="Version 2",
            optimized_prompt="Version 3",
            strategy=OptimizationStrategy.STRUCTURE_FOCUS,
            improvement_score=18.0,
            confidence=0.88,
            changes=[],
            metadata={}
        )
        v3 = version_manager.create_version(sample_prompt, sample_analysis, result2, "2.0.0")
        assert v3.version == "3.0.0"

    def test_mixed_version_bumps(
            self, version_manager, sample_prompt, sample_analysis
    ):
        """Test sequence of mixed version bumps."""
        # 1.0.0 (baseline)
        v1 = version_manager.create_version(sample_prompt, sample_analysis, None, None)
        assert v1.version == "1.0.0"

        # 1.1.0 (minor improvement: 8 points)
        result1 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="v1.1.0",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=8.0,
            confidence=0.8,
            changes=[],
            metadata={}
        )
        v2 = version_manager.create_version(sample_prompt, sample_analysis, result1, "1.0.0")
        assert v2.version == "1.1.0"

        # 1.1.1 (patch improvement: 3 points)
        result2 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt="v1.1.0",
            optimized_prompt="v1.1.1",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=3.0,
            confidence=0.75,
            changes=[],
            metadata={}
        )
        v3 = version_manager.create_version(sample_prompt, sample_analysis, result2, "1.1.0")
        assert v3.version == "1.1.1"

        # 2.0.0 (major improvement: 16 points)
        result3 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt="v1.1.1",
            optimized_prompt="v2.0.0",
            strategy=OptimizationStrategy.STRUCTURE_FOCUS,
            improvement_score=16.0,
            confidence=0.92,
            changes=[],
            metadata={}
        )
        v4 = version_manager.create_version(sample_prompt, sample_analysis, result3, "1.1.1")
        assert v4.version == "2.0.0"

        # 2.1.0 (minor improvement: 7 points)
        result4 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt="v2.0.0",
            optimized_prompt="v2.1.0",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=7.0,
            confidence=0.81,
            changes=[],
            metadata={}
        )
        v5 = version_manager.create_version(sample_prompt, sample_analysis, result4, "2.0.0")
        assert v5.version == "2.1.0"

    def test_custom_scoring_rules(self, in_memory_storage, sample_prompt, sample_analysis):
        """Test version manager with custom scoring rules."""
        # Custom rules: major=20, minor=10
        custom_rules = ScoringRules(
            major_version_min_improvement=20.0,
            minor_version_min_improvement=10.0
        )
        manager = VersionManager(
            storage=in_memory_storage,
            scoring_rules=custom_rules
        )

        # Baseline
        v1 = manager.create_version(sample_prompt, sample_analysis, None, None)
        assert v1.version == "1.0.0"

        # 15 points with custom rules -> minor (not major)
        result1 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt=sample_prompt.text,
            optimized_prompt="v1.1.0",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=15.0,
            confidence=0.85,
            changes=[],
            metadata={}
        )
        v2 = manager.create_version(sample_prompt, sample_analysis, result1, "1.0.0")
        assert v2.version == "1.1.0"

        # 8 points with custom rules -> patch (not minor)
        result2 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt="v1.1.0",
            optimized_prompt="v1.1.1",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=8.0,
            confidence=0.78,
            changes=[],
            metadata={}
        )
        v3 = manager.create_version(sample_prompt, sample_analysis, result2, "1.1.0")
        assert v3.version == "1.1.1"

        # 25 points with custom rules -> major
        result3 = OptimizationResult(
            prompt_id=sample_prompt.id,
            original_prompt="v1.1.1",
            optimized_prompt="v2.0.0",
            strategy=OptimizationStrategy.STRUCTURE_FOCUS,
            improvement_score=25.0,
            confidence=0.95,
            changes=[],
            metadata={}
        )
        v4 = manager.create_version(sample_prompt, sample_analysis, result3, "1.1.1")
        assert v4.version == "2.0.0"


class TestVersionIncrementMethod:
    """Test cases for _increment_version method."""

    def test_increment_major_version(self, version_manager):
        """Test major version increment."""
        assert version_manager._increment_version("1.2.3", "major") == "2.0.0"
        assert version_manager._increment_version("5.10.20", "major") == "6.0.0"
        assert version_manager._increment_version("0.1.0", "major") == "1.0.0"

    def test_increment_minor_version(self, version_manager):
        """Test minor version increment."""
        assert version_manager._increment_version("1.2.3", "minor") == "1.3.0"
        assert version_manager._increment_version("5.10.20", "minor") == "5.11.0"
        assert version_manager._increment_version("0.0.1", "minor") == "0.1.0"

    def test_increment_patch_version(self, version_manager):
        """Test patch version increment."""
        assert version_manager._increment_version("1.2.3", "patch") == "1.2.4"
        assert version_manager._increment_version("5.10.20", "patch") == "5.10.21"
        assert version_manager._increment_version("0.0.0", "patch") == "0.0.1"

    def test_invalid_bump_type_raises_error(self, version_manager):
        """Test that invalid bump_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bump_type"):
            version_manager._increment_version("1.0.0", "invalid")

        with pytest.raises(ValueError, match="Invalid bump_type"):
            version_manager._increment_version("1.0.0", "MAJOR")  # Case-sensitive

        with pytest.raises(ValueError, match="Invalid bump_type"):
            version_manager._increment_version("1.0.0", "")
