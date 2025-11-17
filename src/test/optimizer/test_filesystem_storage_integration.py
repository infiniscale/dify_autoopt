"""
Integration Test: FileSystemStorage with VersionManager

Date: 2025-11-17
Author: backend-developer
Description: Demonstrates FileSystemStorage as drop-in replacement for InMemoryStorage.
"""

import pytest
import shutil
from pathlib import Path

from src.optimizer.interfaces.filesystem_storage import FileSystemStorage
from src.optimizer.version_manager import VersionManager
from src.optimizer.models import Prompt, PromptAnalysis, OptimizationResult, OptimizationStrategy


class TestFileSystemStorageIntegration:
    """Integration tests for FileSystemStorage with VersionManager."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def filesystem_storage(self, temp_storage_dir):
        """FileSystemStorage instance."""
        return FileSystemStorage(
            str(temp_storage_dir),
            use_index=True,
            use_cache=True,
        )

    @pytest.fixture
    def version_manager_with_fs(self, filesystem_storage):
        """VersionManager with FileSystemStorage."""
        return VersionManager(storage=filesystem_storage)

    def test_version_manager_with_filesystem_storage(
        self,
        version_manager_with_fs,
        sample_prompt,
        sample_analysis,
    ):
        """Test VersionManager works with FileSystemStorage."""
        # Create baseline version
        v1 = version_manager_with_fs.create_version(
            prompt=sample_prompt,
            analysis=sample_analysis,
            optimization_result=None,
            parent_version=None,
        )

        assert v1.version == "1.0.0"
        assert v1.is_baseline()

        # Create second version
        v2 = version_manager_with_fs.create_version(
            prompt=sample_prompt,
            analysis=sample_analysis,
            optimization_result=None,
            parent_version="1.0.0",
        )

        assert v2.version == "1.0.1"

        # Get version
        retrieved = version_manager_with_fs.get_version(sample_prompt.id, "1.0.0")
        assert retrieved.version == "1.0.0"

        # Get latest
        latest = version_manager_with_fs.get_latest_version(sample_prompt.id)
        assert latest.version == "1.0.1"

        # Get history
        history = version_manager_with_fs.get_version_history(sample_prompt.id)
        assert len(history) == 2

    def test_persistence_across_restarts(
        self,
        temp_storage_dir,
        sample_prompt,
        sample_analysis,
    ):
        """Test versions persist across 'restarts' (new storage instances)."""
        # First instance - save versions
        storage1 = FileSystemStorage(str(temp_storage_dir))
        manager1 = VersionManager(storage=storage1)

        manager1.create_version(sample_prompt, sample_analysis, None, None)
        manager1.create_version(sample_prompt, sample_analysis, None, "1.0.0")

        # "Restart" - create new instances
        storage2 = FileSystemStorage(str(temp_storage_dir))
        manager2 = VersionManager(storage=storage2)

        # Should be able to retrieve versions
        history = manager2.get_version_history(sample_prompt.id)
        assert len(history) == 2

        latest = manager2.get_latest_version(sample_prompt.id)
        assert latest.version == "1.0.1"

    def test_filesystem_storage_performance_with_version_manager(
        self,
        version_manager_with_fs,
        sample_prompt,
        sample_analysis,
    ):
        """Test performance with multiple versions."""
        import time

        # Create 50 versions
        start = time.time()
        for i in range(50):
            parent = f"1.{i-1}.0" if i > 0 else None
            version_manager_with_fs.create_version(
                sample_prompt,
                sample_analysis,
                None,
                parent,
            )
        elapsed = time.time() - start

        print(f"\nCreated 50 versions in {elapsed:.2f}s")

        # Get history (should be fast with index)
        start = time.time()
        history = version_manager_with_fs.get_version_history(sample_prompt.id)
        elapsed = time.time() - start

        assert len(history) == 50
        print(f"Retrieved 50 versions in {elapsed*1000:.1f}ms")

    def test_switch_from_memory_to_filesystem(
        self,
        temp_storage_dir,
        sample_prompt,
        sample_analysis,
    ):
        """Test migrating from InMemoryStorage to FileSystemStorage."""
        from src.optimizer.interfaces.storage import InMemoryStorage

        # Start with InMemoryStorage
        memory_storage = InMemoryStorage()
        manager_mem = VersionManager(storage=memory_storage)

        # Create some versions
        manager_mem.create_version(sample_prompt, sample_analysis, None, None)
        manager_mem.create_version(sample_prompt, sample_analysis, None, "1.0.0")

        # "Migrate" to FileSystemStorage
        fs_storage = FileSystemStorage(str(temp_storage_dir))

        # Manual migration (copy versions)
        for version in memory_storage.list_versions(sample_prompt.id):
            fs_storage.save_version(version)

        # Create new manager with FileSystemStorage
        manager_fs = VersionManager(storage=fs_storage)

        # Should have same versions
        history = manager_fs.get_version_history(sample_prompt.id)
        assert len(history) == 2

    def test_concurrent_operations_with_version_manager(
        self,
        version_manager_with_fs,
        sample_prompt,
        sample_analysis,
    ):
        """Test concurrent version creation with FileSystemStorage."""
        import threading

        errors = []
        created = []

        def create_version(version_num):
            try:
                # Create unique prompts
                prompt = Prompt(
                    id=f"prompt_{version_num:03d}",
                    workflow_id="wf_001",
                    node_id="llm_1",
                    node_type="llm",
                    text=f"Test prompt {version_num}",
                )
                version = version_manager_with_fs.create_version(
                    prompt=prompt,
                    analysis=sample_analysis,
                    optimization_result=None,
                    parent_version=None,
                )
                created.append(version.version)
            except Exception as e:
                errors.append(e)

        # Create 10 versions concurrently
        threads = [
            threading.Thread(target=create_version, args=(i,))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(created) == 10

    def test_storage_stats_integration(
        self,
        version_manager_with_fs,
        filesystem_storage,
        sample_prompt,
        sample_analysis,
    ):
        """Test storage statistics with VersionManager."""
        # Create some versions
        for i in range(5):
            parent = f"1.{i-1}.0" if i > 0 else None
            version_manager_with_fs.create_version(
                sample_prompt,
                sample_analysis,
                None,
                parent,
            )

        # Get stats
        stats = filesystem_storage.get_storage_stats()

        assert stats["total_prompts"] == 1
        assert stats["total_versions"] == 5
        assert stats["index_enabled"] is True
        assert stats["cache_enabled"] is True
        assert stats["disk_usage_bytes"] > 0

        print(f"\nStorage stats:")
        print(f"  Total prompts: {stats['total_prompts']}")
        print(f"  Total versions: {stats['total_versions']}")
        print(f"  Disk usage: {stats['disk_usage_mb']:.2f} MB")
        if "cache_stats" in stats:
            print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
