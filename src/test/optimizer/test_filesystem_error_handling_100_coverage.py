"""
Test Coverage for FileSystemStorage Error Handling - Achieving 100% Coverage

Date: 2025-11-18
Author: backend-developer
Description: Tests for all uncovered error handling paths in FileSystemStorage
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from src.optimizer.exceptions import OptimizerError, VersionConflictError
from src.optimizer.interfaces.filesystem_storage import (
    FileSystemStorage,
    FileLock,
)
from src.optimizer.models import Prompt, PromptAnalysis, PromptVersion


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage for testing."""
    return FileSystemStorage(storage_dir=tmp_path, use_cache=False)


@pytest.fixture
def sample_version():
    """Create sample version for testing."""
    prompt = Prompt(
        id="test_prompt",
        workflow_id="wf_001",
        node_id="node_1",
        node_type="llm",
        text="Test prompt",
        role="user",
        variables=[],
    )
    analysis = PromptAnalysis(
        prompt_id="test_prompt",
        overall_score=85.0,
        clarity_score=80.0,
        efficiency_score=90.0,
        issues=[],
        suggestions=[],
    )
    return PromptVersion(
        prompt_id="test_prompt",
        version="1.0.0",
        prompt=prompt,
        analysis=analysis,
    )


class TestFileLockStaleDetection:
    """Test FileLock stale lock detection."""

    def test_is_stale_lock_old_file(self, tmp_path):
        """Test stale lock detection for old lock file."""
        lock_path = tmp_path / "test.lock"
        lock_path.touch()

        # Make lock file appear old (> 5 minutes)
        old_time = time.time() - 400  # 6 minutes 40 seconds ago
        import os

        os.utime(lock_path, (old_time, old_time))

        lock = FileLock(lock_path, timeout=1.0)

        assert lock._is_stale_lock() is True

    def test_is_stale_lock_recent_file(self, tmp_path):
        """Test stale lock detection for recent lock file."""
        lock_path = tmp_path / "test.lock"
        lock_path.touch()

        lock = FileLock(lock_path, timeout=1.0)

        assert lock._is_stale_lock() is False

    def test_is_stale_lock_missing_file(self, tmp_path):
        """Test stale lock detection when file doesn't exist."""
        lock_path = tmp_path / "nonexistent.lock"

        lock = FileLock(lock_path, timeout=1.0)

        # Should return False if file doesn't exist (exception caught)
        assert lock._is_stale_lock() is False


class TestFileLockCleanupErrors:
    """Test FileLock cleanup error handling."""

    def test_lock_cleanup_ignores_errors(self, tmp_path):
        """Test that lock cleanup ignores exceptions."""
        lock_path = tmp_path / "test.lock"

        lock = FileLock(lock_path, timeout=1.0)

        # Acquire lock
        with lock:
            # Mock unlink to raise exception
            with patch.object(Path, "unlink", side_effect=OSError("Cleanup error")):
                pass  # Should not raise

        # Test should complete without exception


class TestFileSystemStorageRetryLogic:
    """Test save_version retry logic with transient errors."""

    def test_save_version_retry_on_io_error(self, temp_storage, sample_version):
        """Test retry logic on transient IOError."""
        call_count = 0

        def failing_save(self, version):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IOError("Transient error")
            # Succeed on second attempt
            self._save_version_internal_original(version)

        # Store original method
        temp_storage._save_version_internal_original = (
            temp_storage._save_version_internal
        )

        # Patch with failing version
        with patch.object(
                FileSystemStorage, "_save_version_internal", failing_save, create=False
        ):
            temp_storage.save_version(sample_version)

        assert call_count == 2  # Failed once, succeeded on retry

    def test_save_version_retry_exhaustion(self, temp_storage, sample_version):
        """Test that retries are exhausted after max attempts."""

        def always_failing_save(self, version):
            raise IOError("Persistent error")

        with patch.object(
                FileSystemStorage, "_save_version_internal", always_failing_save
        ):
            with pytest.raises(OptimizerError) as exc_info:
                temp_storage.save_version(sample_version)

            assert "Failed to save version" in str(exc_info.value)
            assert exc_info.value.error_code == "FS-SAVE-001"

    def test_save_version_no_retry_on_conflict(self, temp_storage, sample_version):
        """Test that VersionConflictError is never retried."""
        # Save version first to create a conflict
        temp_storage.save_version(sample_version)

        # Try to save the same version again (will conflict)
        with pytest.raises(VersionConflictError):
            temp_storage.save_version(sample_version)

    def test_save_version_no_retry_on_unexpected_error(
            self, temp_storage, sample_version
    ):
        """Test that unexpected exceptions are not retried."""
        call_count = 0

        def unexpected_error_save(self, version):
            nonlocal call_count
            call_count += 1
            raise ValueError("Unexpected error")

        with patch.object(
                FileSystemStorage, "_save_version_internal", unexpected_error_save
        ):
            with pytest.raises(OptimizerError) as exc_info:
                temp_storage.save_version(sample_version)

            assert exc_info.value.error_code == "FS-SAVE-002"

        # Should only be called once (no retries for unexpected errors)
        assert call_count == 1


class TestFileSystemStorageShardingEdgeCases:
    """Test directory sharding edge cases."""

    def test_get_prompt_dir_with_sharding_enabled(self, tmp_path):
        """Test prompt directory calculation when sharding is enabled."""
        storage = FileSystemStorage(
            storage_dir=tmp_path, enable_sharding=True, shard_depth=2
        )

        # Get prompt directory (should be within a shard)
        prompt_dir = storage._get_prompt_dir("test_prompt_001")

        # Should be inside storage_dir, may be sharded
        assert prompt_dir.is_relative_to(tmp_path)

    def test_save_and_retrieve_with_sharding(self, tmp_path, sample_version):
        """Test that sharding doesn't break save/retrieve."""
        storage = FileSystemStorage(
            storage_dir=tmp_path, enable_sharding=True, shard_depth=2
        )

        # Save version (should go into shard directory)
        storage.save_version(sample_version)

        # Retrieve version (should find it in shard directory)
        retrieved = storage.get_version("test_prompt", "1.0.0")

        assert retrieved is not None
        assert retrieved.version == "1.0.0"


class TestFileSystemStorageIndexRebuildEdgeCases:
    """Test index rebuild edge cases."""

    def test_rebuild_index_with_corrupted_files(self, temp_storage, sample_version):
        """Test rebuild_index skips corrupted files gracefully."""
        # Save a valid version
        temp_storage.save_version(sample_version)

        # Create a corrupted file
        prompt_dir = temp_storage.storage_dir / "test_prompt"
        corrupted_file = prompt_dir / "corrupted.json"
        corrupted_file.write_text("invalid json {{{")

        # Rebuild index (should skip corrupted file)
        temp_storage.rebuild_index()

        # Valid version should still be in index
        versions = temp_storage.list_versions("test_prompt")
        assert len(versions) == 1
        assert versions[0].version == "1.0.0"

    def test_rebuild_index_empty_storage(self, temp_storage):
        """Test rebuild_index on empty storage."""
        # Should not raise
        temp_storage.rebuild_index()

        assert temp_storage._index["version"] == "1.0.0"
        assert temp_storage._index["index"] == {}


class TestFileSystemStorageEdgeCaseCoverage:
    """Additional edge case tests for complete coverage."""

    def test_list_versions_with_limit(self, temp_storage):
        """Test list_versions with limit parameter."""
        # Create multiple versions
        for i in range(5):
            prompt = Prompt(
                id="test_prompt",
                workflow_id="wf_001",
                node_id="node_1",
                node_type="llm",
                text=f"Test prompt {i}",
                role="user",
                variables=[],
            )
            analysis = PromptAnalysis(
                prompt_id="test_prompt",
                overall_score=85.0,
                clarity_score=80.0,
                efficiency_score=90.0,
                issues=[],
                suggestions=[],
            )
            version = PromptVersion(
                prompt_id="test_prompt",
                version=f"1.{i}.0",
                prompt=prompt,
                analysis=analysis,
            )
            temp_storage.save_version(version)

        # List with limit
        versions = temp_storage.list_versions("test_prompt", limit=3)

        assert len(versions) == 3

    def test_get_storage_stats_comprehensive(self, temp_storage, sample_version):
        """Test get_storage_stats with actual data."""
        # Save multiple versions for multiple prompts
        for prompt_num in range(3):
            for ver_num in range(2):
                prompt = Prompt(
                    id=f"prompt_{prompt_num}",
                    workflow_id="wf_001",
                    node_id=f"node_{prompt_num}",
                    node_type="llm",
                    text=f"Test prompt {prompt_num}",
                    role="user",
                    variables=[],
                )
                analysis = PromptAnalysis(
                    prompt_id=f"prompt_{prompt_num}",
                    overall_score=85.0,
                    clarity_score=80.0,
                    efficiency_score=90.0,
                    issues=[],
                    suggestions=[],
                )
                version = PromptVersion(
                    prompt_id=f"prompt_{prompt_num}",
                    version=f"1.{ver_num}.0",
                    prompt=prompt,
                    analysis=analysis,
                )
                temp_storage.save_version(version)

        stats = temp_storage.get_storage_stats()

        assert stats["total_prompts"] == 3
        assert stats["total_versions"] == 6
        assert stats["cache_enabled"] is False
