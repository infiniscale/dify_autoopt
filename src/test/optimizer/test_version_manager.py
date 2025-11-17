"""
Test Cases for VersionManager

Date: 2025-11-17
Author: qa-engineer
Description: Unit tests for prompt version management
"""

import pytest
from datetime import datetime

from src.optimizer.version_manager import VersionManager
from src.optimizer.models import PromptVersion, Prompt
from src.optimizer.exceptions import VersionNotFoundError, VersionConflictError


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
