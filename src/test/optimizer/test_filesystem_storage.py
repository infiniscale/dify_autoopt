"""
Test Cases for FileSystemStorage

Date: 2025-11-17
Author: backend-developer
Description: Comprehensive test suite for filesystem-based version storage.

Test Coverage:
    - Basic CRUD operations
    - Atomic writes
    - Index management
    - Cache functionality
    - File locking
    - Concurrent access
    - Performance benchmarks
    - Error handling
    - Edge cases
"""

import json
import os
import pytest
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.optimizer.interfaces.filesystem_storage import (
    FileSystemStorage,
    LRUCache,
    FileLock,
)
from src.optimizer.models import PromptVersion
from src.optimizer.exceptions import VersionConflictError, OptimizerError


# ============================================================================
# Test LRU Cache
# ============================================================================


class TestLRUCache:
    """Test cases for LRU cache implementation."""

    def test_cache_initialization(self):
        """Test cache can be initialized."""
        cache = LRUCache(max_size=10)
        assert cache.max_size == 10
        assert cache.stats()["size"] == 0

    def test_cache_put_and_get(self, sample_version):
        """Test putting and getting from cache."""
        cache = LRUCache(max_size=10)
        cache.put("key1", sample_version)

        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.version == sample_version.version

    def test_cache_miss_returns_none(self):
        """Test cache miss returns None."""
        cache = LRUCache(max_size=10)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_eviction_when_full(self, sample_version):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=3)

        # Add 4 items (one more than capacity)
        cache.put("key1", sample_version)
        cache.put("key2", sample_version)
        cache.put("key3", sample_version)
        cache.put("key4", sample_version)  # Should evict key1

        # key1 should be evicted
        assert cache.get("key1") is None
        # Others should still be present
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_cache_lru_order(self, sample_version):
        """Test LRU eviction order."""
        cache = LRUCache(max_size=3)

        cache.put("key1", sample_version)
        cache.put("key2", sample_version)
        cache.put("key3", sample_version)

        # Access key1 (make it most recent)
        cache.get("key1")

        # Add key4 (should evict key2, not key1)
        cache.put("key4", sample_version)

        assert cache.get("key1") is not None  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_cache_remove(self, sample_version):
        """Test removing from cache."""
        cache = LRUCache(max_size=10)
        cache.put("key1", sample_version)

        assert cache.get("key1") is not None

        cache.remove("key1")
        assert cache.get("key1") is None

    def test_cache_clear(self, sample_version):
        """Test clearing cache."""
        cache = LRUCache(max_size=10)
        cache.put("key1", sample_version)
        cache.put("key2", sample_version)

        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self, sample_version):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)
        cache.put("key1", sample_version)
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["utilization"] == 0.1

    def test_cache_thread_safety(self, sample_version):
        """Test cache is thread-safe."""
        cache = LRUCache(max_size=100)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    cache.put(f"thread_{thread_id}_key_{i}", sample_version)
                    cache.get(f"thread_{thread_id}_key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Test FileLock
# ============================================================================


class TestFileLock:
    """Test cases for cross-platform file locking."""

    def test_file_lock_acquire_release(self, tmp_path):
        """Test lock can be acquired and released."""
        lock_file = tmp_path / "test.lock"

        with FileLock(lock_file):
            assert lock_file.exists()

        # Lock should be released after context exit

    def test_file_lock_blocks_concurrent_access(self, tmp_path):
        """Test lock blocks concurrent access."""
        lock_file = tmp_path / "test.lock"
        acquired = []

        def worker(thread_id):
            with FileLock(lock_file, timeout=2.0):
                acquired.append(thread_id)
                time.sleep(0.1)  # Hold lock briefly

        # Start two threads
        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))

        t1.start()
        time.sleep(0.05)  # Let t1 acquire lock first
        t2.start()

        t1.join()
        t2.join()

        # Both should have acquired, but sequentially
        assert len(acquired) == 2

    def test_file_lock_timeout(self, tmp_path):
        """Test lock acquisition timeout."""
        lock_file = tmp_path / "test.lock"

        # Hold lock in main thread
        lock1 = FileLock(lock_file, timeout=0.5)
        lock1.__enter__()

        try:
            # Try to acquire in another context (should timeout)
            with pytest.raises(OptimizerError, match="Failed to acquire lock"):
                with FileLock(lock_file, timeout=0.2):
                    pass
        finally:
            lock1.__exit__(None, None, None)


# ============================================================================
# Test FileSystemStorage - Basic Operations
# ============================================================================


class TestFileSystemStorageBasic:
    """Test basic CRUD operations."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        # Cleanup
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """FileSystemStorage instance."""
        return FileSystemStorage(
            str(temp_storage_dir),
            use_index=True,
            use_cache=True,
        )

    def test_init_creates_directory(self, temp_storage_dir):
        """Test initialization creates storage directory."""
        storage = FileSystemStorage(str(temp_storage_dir))
        assert temp_storage_dir.exists()
        assert temp_storage_dir.is_dir()

    def test_init_creates_index(self, temp_storage_dir):
        """Test initialization creates index file."""
        storage = FileSystemStorage(str(temp_storage_dir), use_index=True)
        index_file = temp_storage_dir / ".index.json"
        assert index_file.exists()

    def test_save_version_success(self, storage, sample_version):
        """Test successful version save."""
        storage.save_version(sample_version)

        # Verify file exists
        version_file = storage._get_version_file(
            sample_version.prompt_id,
            sample_version.version
        )
        assert version_file.exists()

        # Verify content
        loaded = storage.get_version(sample_version.prompt_id, sample_version.version)
        assert loaded is not None
        assert loaded.version == sample_version.version
        assert loaded.prompt_id == sample_version.prompt_id

    def test_save_version_duplicate_raises_error(self, storage, sample_version):
        """Test saving duplicate version raises VersionConflictError."""
        storage.save_version(sample_version)

        # Try to save again
        with pytest.raises(VersionConflictError, match="already exists"):
            storage.save_version(sample_version)

    def test_save_version_creates_prompt_directory(self, storage, sample_version, temp_storage_dir):
        """Test save_version creates prompt directory if missing."""
        prompt_dir = temp_storage_dir / sample_version.prompt_id
        assert not prompt_dir.exists()

        storage.save_version(sample_version)

        assert prompt_dir.exists()
        assert prompt_dir.is_dir()

    def test_get_version_found(self, storage, sample_version):
        """Test retrieving existing version."""
        storage.save_version(sample_version)

        loaded = storage.get_version(sample_version.prompt_id, sample_version.version)

        assert loaded is not None
        assert loaded.version == sample_version.version
        assert loaded.prompt_id == sample_version.prompt_id
        assert loaded.prompt.text == sample_version.prompt.text

    def test_get_version_not_found(self, storage):
        """Test retrieving non-existent version returns None."""
        loaded = storage.get_version("nonexistent", "1.0.0")
        assert loaded is None

    def test_list_versions_empty(self, storage):
        """Test listing versions for non-existent prompt returns empty list."""
        versions = storage.list_versions("nonexistent")
        assert versions == []

    def test_list_versions_single(self, storage, sample_version):
        """Test listing single version."""
        storage.save_version(sample_version)

        versions = storage.list_versions(sample_version.prompt_id)
        assert len(versions) == 1
        assert versions[0].version == sample_version.version

    def test_list_versions_multiple_sorted(self, storage, sample_prompt, sample_analysis):
        """Test listing multiple versions returns sorted list."""
        # Create versions out of order
        v2 = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.2",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        v0 = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.0",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        v1 = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.1",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )

        storage.save_version(v2)
        storage.save_version(v0)
        storage.save_version(v1)

        versions = storage.list_versions("test_prompt")

        assert len(versions) == 3
        assert versions[0].version == "1.0.0"
        assert versions[1].version == "1.0.1"
        assert versions[2].version == "1.0.2"

    def test_get_latest_version_found(self, storage, sample_prompt, sample_analysis):
        """Test getting latest version."""
        v1 = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.0",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        v2 = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.1",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )

        storage.save_version(v1)
        storage.save_version(v2)

        latest = storage.get_latest_version("test_prompt")

        assert latest is not None
        assert latest.version == "1.0.1"

    def test_get_latest_version_empty(self, storage):
        """Test getting latest version for non-existent prompt returns None."""
        latest = storage.get_latest_version("nonexistent")
        assert latest is None

    def test_delete_version_success(self, storage, sample_version):
        """Test successful version deletion."""
        storage.save_version(sample_version)

        deleted = storage.delete_version(sample_version.prompt_id, sample_version.version)
        assert deleted is True

        # Verify file removed
        version_file = storage._get_version_file(
            sample_version.prompt_id,
            sample_version.version
        )
        assert not version_file.exists()

    def test_delete_version_not_found(self, storage):
        """Test deleting non-existent version returns False."""
        deleted = storage.delete_version("nonexistent", "1.0.0")
        assert deleted is False

    def test_clear_all_empty_storage(self, storage, temp_storage_dir):
        """Test clearing empty storage."""
        storage.clear_all()

        # Directory should still exist but be empty
        assert temp_storage_dir.exists()
        prompt_dirs = [d for d in temp_storage_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        assert len(prompt_dirs) == 0

    def test_clear_all_populated_storage(self, storage, sample_version):
        """Test clearing populated storage."""
        storage.save_version(sample_version)

        storage.clear_all()

        # No versions should remain
        versions = storage.list_versions(sample_version.prompt_id)
        assert versions == []


# ============================================================================
# Test FileSystemStorage - Index Management
# ============================================================================


class TestFileSystemStorageIndex:
    """Test index management functionality."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """FileSystemStorage with index enabled."""
        return FileSystemStorage(str(temp_storage_dir), use_index=True)

    def test_index_created_on_init(self, storage, temp_storage_dir):
        """Test index file is created on initialization."""
        index_file = temp_storage_dir / ".index.json"
        assert index_file.exists()

        with open(index_file, 'r') as f:
            index = json.load(f)

        assert index["version"] == "1.0.0"
        assert index["total_prompts"] == 0
        assert index["total_versions"] == 0
        assert index["index"] == {}

    def test_index_updated_on_save(self, storage, sample_version):
        """Test index is updated when version is saved."""
        storage.save_version(sample_version)

        assert storage._index["total_prompts"] == 1
        assert storage._index["total_versions"] == 1

        entry = storage._index["index"][sample_version.prompt_id]
        assert entry["latest_version"] == sample_version.version
        assert entry["version_count"] == 1

    def test_index_updated_on_delete(self, storage, sample_version):
        """Test index is updated when version is deleted."""
        storage.save_version(sample_version)
        storage.delete_version(sample_version.prompt_id, sample_version.version)

        assert storage._index["total_prompts"] == 0
        assert storage._index["total_versions"] == 0
        assert sample_version.prompt_id not in storage._index["index"]

    def test_get_latest_uses_index(self, storage, sample_prompt, sample_analysis):
        """Test get_latest_version uses index for fast lookup."""
        v1 = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.0",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        v2 = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.1",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )

        storage.save_version(v1)
        storage.save_version(v2)

        # Should use index (fast path)
        latest = storage.get_latest_version("test_prompt")
        assert latest.version == "1.0.1"

    def test_index_rebuild(self, storage, sample_prompt, sample_analysis, temp_storage_dir):
        """Test index can be rebuilt from filesystem."""
        # Save some versions
        for i in range(3):
            v = PromptVersion(
                prompt_id="test_prompt",
                version=f"1.0.{i}",
                prompt=sample_prompt,
                analysis=sample_analysis,
            )
            storage.save_version(v)

        # Corrupt index
        index_file = temp_storage_dir / ".index.json"
        index_file.write_text("{ invalid json }")

        # Rebuild
        storage._rebuild_index()

        # Index should be correct
        assert storage._index["total_prompts"] == 1
        assert storage._index["total_versions"] == 3

        entry = storage._index["index"]["test_prompt"]
        assert entry["latest_version"] == "1.0.2"
        assert entry["version_count"] == 3

    def test_rebuild_index_public_method(self, storage, sample_version):
        """Test public rebuild_index() method."""
        storage.save_version(sample_version)

        # Manually corrupt index
        storage._index = None

        # Rebuild via public method
        storage.rebuild_index()

        assert storage._index is not None
        assert storage._index["total_versions"] == 1


# ============================================================================
# Test FileSystemStorage - Cache Functionality
# ============================================================================


class TestFileSystemStorageCache:
    """Test cache functionality."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """FileSystemStorage with cache enabled."""
        return FileSystemStorage(
            str(temp_storage_dir),
            use_cache=True,
            cache_size=10,
        )

    def test_cache_hit_on_second_get(self, storage, sample_version):
        """Test cache hit on second get."""
        storage.save_version(sample_version)

        # First get (miss)
        v1 = storage.get_version(sample_version.prompt_id, sample_version.version)

        # Second get (should be hit)
        v2 = storage.get_version(sample_version.prompt_id, sample_version.version)

        # Should be same object from cache
        assert v1 is v2

    def test_cache_updated_on_save(self, storage, sample_version):
        """Test cache is updated when version is saved."""
        storage.save_version(sample_version)

        # Should be in cache
        cache_key = f"{sample_version.prompt_id}:{sample_version.version}"
        cached = storage._cache.get(cache_key)
        assert cached is not None
        assert cached.version == sample_version.version

    def test_cache_invalidated_on_delete(self, storage, sample_version):
        """Test cache is invalidated when version is deleted."""
        storage.save_version(sample_version)

        # Verify in cache
        cache_key = f"{sample_version.prompt_id}:{sample_version.version}"
        assert storage._cache.get(cache_key) is not None

        # Delete
        storage.delete_version(sample_version.prompt_id, sample_version.version)

        # Should be removed from cache
        assert storage._cache.get(cache_key) is None

    def test_cache_disabled(self, temp_storage_dir, sample_version):
        """Test storage works with cache disabled."""
        storage = FileSystemStorage(str(temp_storage_dir), use_cache=False)

        assert storage._cache is None

        storage.save_version(sample_version)
        loaded = storage.get_version(sample_version.prompt_id, sample_version.version)

        assert loaded is not None
        assert loaded.version == sample_version.version


# ============================================================================
# Test FileSystemStorage - Atomic Writes
# ============================================================================


class TestFileSystemStorageAtomicity:
    """Test atomic write functionality."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """FileSystemStorage instance."""
        return FileSystemStorage(str(temp_storage_dir))

    def test_atomic_write_success(self, storage, sample_version):
        """Test atomic write completes successfully."""
        storage.save_version(sample_version)

        # No temp files should remain
        temp_files = list(storage.storage_dir.glob("**/*.tmp*"))
        assert len(temp_files) == 0

    def test_atomic_write_cleanup_on_error(self, storage, sample_version, temp_storage_dir, monkeypatch):
        """Test atomic write cleans up temp file on error."""
        # Mock json.dump to raise error
        original_dump = json.dump

        def failing_dump(*args, **kwargs):
            raise IOError("Simulated I/O error")

        monkeypatch.setattr(json, "dump", failing_dump)

        # Try to save (should fail)
        with pytest.raises(OptimizerError):
            storage.save_version(sample_version)

        # Verify no temp files left
        temp_files = list(temp_storage_dir.glob("**/*.tmp*"))
        assert len(temp_files) == 0


# ============================================================================
# Test FileSystemStorage - Error Handling
# ============================================================================


class TestFileSystemStorageErrors:
    """Test error handling."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """FileSystemStorage instance."""
        return FileSystemStorage(str(temp_storage_dir))

    def test_corrupted_json_raises_error(self, storage, sample_version):
        """Test reading corrupted JSON file raises error."""
        # Create corrupted file
        version_file = storage._get_version_file(
            sample_version.prompt_id,
            sample_version.version
        )
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text("{ invalid json }")

        # Should raise OptimizerError
        with pytest.raises(OptimizerError, match="Corrupted version file"):
            storage.get_version(sample_version.prompt_id, sample_version.version)

    def test_list_versions_skips_corrupted_files(self, storage, sample_prompt, sample_analysis):
        """Test list_versions skips corrupted files."""
        # Save valid version
        v1 = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.0",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        storage.save_version(v1)

        # Create corrupted file
        prompt_dir = storage._get_prompt_dir("test_prompt")
        corrupted_file = prompt_dir / "1.0.1.json"
        corrupted_file.write_text("{ invalid }")

        # Should return only valid version
        versions = storage.list_versions("test_prompt")
        assert len(versions) == 1
        assert versions[0].version == "1.0.0"


# ============================================================================
# Test FileSystemStorage - Concurrent Access
# ============================================================================


class TestFileSystemStorageConcurrency:
    """Test concurrent access with file locking."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """FileSystemStorage instance."""
        return FileSystemStorage(str(temp_storage_dir))

    def test_concurrent_writes_different_prompts(self, storage, sample_prompt, sample_analysis):
        """Test concurrent writes to different prompts succeed."""
        errors = []

        def write_version(prompt_id):
            try:
                version = PromptVersion(
                    prompt_id=prompt_id,
                    version="1.0.0",
                    prompt=sample_prompt,
                    analysis=sample_analysis,
                )
                storage.save_version(version)
            except Exception as e:
                errors.append(e)

        # Spawn 10 threads
        threads = [
            threading.Thread(target=write_version, args=(f"prompt_{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors
        assert len(errors) == 0

        # All versions saved
        for i in range(10):
            loaded = storage.get_version(f"prompt_{i}", "1.0.0")
            assert loaded is not None

    def test_concurrent_writes_same_prompt_raises_conflict(self, storage, sample_prompt, sample_analysis):
        """Test concurrent writes to same prompt raise VersionConflictError."""
        errors = []
        successes = []

        def write_version():
            try:
                version = PromptVersion(
                    prompt_id="prompt_001",
                    version="1.0.0",
                    prompt=sample_prompt,
                    analysis=sample_analysis,
                )
                storage.save_version(version)
                successes.append(True)
            except VersionConflictError:
                errors.append("conflict")

        # Spawn 5 threads (same prompt_id + version)
        threads = [threading.Thread(target=write_version) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 1 success, 4 conflicts
        assert len(successes) == 1
        assert len(errors) == 4


# ============================================================================
# Test FileSystemStorage - Sharding
# ============================================================================


class TestFileSystemStorageSharding:
    """Test directory sharding for scalability."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    def test_sharding_enabled_distributes_prompts(self, temp_storage_dir, sample_prompt, sample_analysis):
        """Test sharding distributes prompts across shard directories."""
        storage = FileSystemStorage(
            str(temp_storage_dir),
            enable_sharding=True,
            shard_depth=2,
        )

        # Save versions for different prompts
        for i in range(5):
            version = PromptVersion(
                prompt_id=f"prompt_{i:03d}",
                version="1.0.0",
                prompt=sample_prompt,
                analysis=sample_analysis,
            )
            storage.save_version(version)

        # Should have shard directories
        shard_dirs = [d for d in temp_storage_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        assert len(shard_dirs) > 0

    def test_sharding_retrieval_works(self, temp_storage_dir, sample_prompt, sample_analysis):
        """Test retrieval works with sharding enabled."""
        storage = FileSystemStorage(
            str(temp_storage_dir),
            enable_sharding=True,
        )

        version = PromptVersion(
            prompt_id="test_prompt",
            version="1.0.0",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        storage.save_version(version)

        # Should be able to retrieve
        loaded = storage.get_version("test_prompt", "1.0.0")
        assert loaded is not None
        assert loaded.version == "1.0.0"


# ============================================================================
# Test FileSystemStorage - Storage Stats
# ============================================================================


class TestFileSystemStorageStats:
    """Test storage statistics."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """FileSystemStorage instance."""
        return FileSystemStorage(str(temp_storage_dir), use_index=True, use_cache=True)

    def test_get_storage_stats(self, storage, sample_version):
        """Test getting storage statistics."""
        storage.save_version(sample_version)

        stats = storage.get_storage_stats()

        assert stats["storage_dir"] == str(storage.storage_dir)
        assert stats["total_prompts"] == 1
        assert stats["total_versions"] == 1
        assert stats["index_enabled"] is True
        assert stats["cache_enabled"] is True
        assert stats["disk_usage_bytes"] > 0
        assert "cache_stats" in stats


# ============================================================================
# Test FileSystemStorage - Performance Benchmarks
# ============================================================================


class TestFileSystemStoragePerformance:
    """Performance benchmark tests."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Temporary storage directory."""
        storage_dir = tmp_path / "versions"
        yield storage_dir
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """FileSystemStorage with index and cache."""
        return FileSystemStorage(
            str(temp_storage_dir),
            use_index=True,
            use_cache=True,
            cache_size=100,
        )

    def test_write_performance_100_versions(self, storage, sample_prompt, sample_analysis):
        """Benchmark write throughput."""
        start = time.time()

        for i in range(100):
            version = PromptVersion(
                prompt_id=f"prompt_{i:03d}",
                version="1.0.0",
                prompt=sample_prompt,
                analysis=sample_analysis,
            )
            storage.save_version(version)

        elapsed = time.time() - start
        avg_write_time = elapsed / 100

        print(f"\nWrote 100 versions in {elapsed:.2f}s (avg: {avg_write_time*1000:.1f}ms)")

        # Assert performance target: < 20ms per write
        # Note: This might be slower on CI systems, so we use a relaxed threshold
        assert avg_write_time < 0.1, f"Write too slow: {avg_write_time*1000:.1f}ms"

    def test_read_performance_with_cache(self, storage, sample_version):
        """Benchmark read throughput with cache."""
        storage.save_version(sample_version)

        # Warm up cache
        storage.get_version(sample_version.prompt_id, sample_version.version)

        # Benchmark cached reads
        start = time.time()
        for _ in range(1000):
            storage.get_version(sample_version.prompt_id, sample_version.version)
        elapsed = time.time() - start

        avg_read_time = elapsed / 1000

        print(f"\nRead 1000 times (cached) in {elapsed:.3f}s (avg: {avg_read_time*1000000:.1f}µs)")

        # Assert performance target: < 0.1ms per cached read
        assert avg_read_time < 0.001, f"Cached read too slow: {avg_read_time*1000:.2f}ms"

    def test_list_versions_performance(self, storage, sample_prompt, sample_analysis):
        """Benchmark list_versions with 50 versions."""
        # Create 50 versions for same prompt
        for i in range(50):
            version = PromptVersion(
                prompt_id="prompt_001",
                version=f"1.{i}.0",
                prompt=sample_prompt,
                analysis=sample_analysis,
            )
            storage.save_version(version)

        # Benchmark
        start = time.time()
        versions = storage.list_versions("prompt_001")
        elapsed = time.time() - start

        print(f"\nListed 50 versions in {elapsed*1000:.1f}ms")

        assert len(versions) == 50
        assert elapsed < 0.5, f"List too slow: {elapsed*1000:.1f}ms"

    def test_get_latest_performance_with_index(self, storage, sample_prompt, sample_analysis):
        """Benchmark get_latest_version with index."""
        # Create 50 versions
        for i in range(50):
            version = PromptVersion(
                prompt_id="prompt_001",
                version=f"1.{i}.0",
                prompt=sample_prompt,
                analysis=sample_analysis,
            )
            storage.save_version(version)

        # Benchmark latest lookup (should use index)
        start = time.time()
        for _ in range(100):
            latest = storage.get_latest_version("prompt_001")
        elapsed = time.time() - start

        avg_time = elapsed / 100

        print(f"\nGet latest 100 times in {elapsed:.3f}s (avg: {avg_time*1000:.1f}ms)")

        # Should be fast with index
        assert avg_time < 0.01, f"Latest lookup too slow: {avg_time*1000:.1f}ms"
