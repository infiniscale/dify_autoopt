"""
Test suite for CRITICAL #4: Index Corruption on Concurrent Writes Fix

Tests that index updates use copy-on-write semantics and remain consistent
under heavy concurrent load.
"""

import tempfile
import threading
from datetime import datetime
from pathlib import Path

import pytest

from src.optimizer.interfaces.filesystem_storage import FileSystemStorage
from src.optimizer.models import PromptVersion, PromptAnalysis, Prompt


def make_test_version(version: str, prompt_id: str = "test_prompt") -> PromptVersion:
    """Helper to create test version objects."""
    prompt = Prompt(
        id=prompt_id,
        workflow_id="test_wf",
        node_id="test_node",
        node_type="llm",
        text=f"Test prompt version {version}",
        role="user",
        variables=[],
        context={},
        extracted_at=datetime.now()
    )

    analysis = PromptAnalysis(
        prompt_id=prompt_id,
        overall_score=85.0,
        clarity_score=80.0,
        efficiency_score=90.0,
        issues=[],
        suggestions=[],
        metadata={}
    )

    return PromptVersion(
        prompt_id=prompt_id,
        version=version,
        prompt=prompt,
        analysis=analysis,
        optimization_result=None,
        parent_version=None,
        metadata={}
    )


class TestIndexConcurrency:
    """Test suite for index integrity under concurrent writes."""

    def test_index_concurrent_updates(self):
        """Test index integrity under concurrent writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir, use_index=True)

            def save_versions(worker_id):
                """Save 10 versions per worker."""
                for i in range(10):
                    version = make_test_version(
                        f"1.{worker_id}.{i}",
                        f"prompt_{worker_id}"
                    )
                    storage.save_version(version)

            # Launch 10 workers = 100 total versions
            threads = [
                threading.Thread(target=save_versions, args=(i,))
                for i in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Verify: all versions saved correctly
            stats = storage.get_storage_stats()
            assert stats["total_prompts"] == 10, \
                f"Expected 10 prompts, got {stats['total_prompts']}"
            assert stats["total_versions"] == 100, \
                f"Expected 100 versions, got {stats['total_versions']}"

            # Verify each prompt has correct version count
            for worker_id in range(10):
                prompt_id = f"prompt_{worker_id}"
                versions = storage.list_versions(prompt_id)
                assert len(versions) == 10, \
                    f"Prompt {prompt_id} has {len(versions)} versions, expected 10"

    def test_index_single_prompt_concurrent_versions(self):
        """Test concurrent version additions to same prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir, use_index=True)
            prompt_id = "shared_prompt"
            errors = []

            def add_version(worker_id):
                """Add version to shared prompt."""
                try:
                    for i in range(10):
                        version = make_test_version(
                            f"1.{worker_id}.{i}",
                            prompt_id
                        )
                        storage.save_version(version)
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {str(e)}")

            # 10 workers adding to same prompt
            threads = [
                threading.Thread(target=add_version, args=(i,))
                for i in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors occurred: {errors}"

            # Verify all 100 versions present
            versions = storage.list_versions(prompt_id)
            assert len(versions) == 100, \
                f"Expected 100 versions, got {len(versions)}"

            # Verify index consistency
            stats = storage.get_storage_stats()
            assert stats["total_prompts"] == 1
            assert stats["total_versions"] == 100

    def test_index_copy_on_write_semantics(self):
        """Test that index updates don't modify original data structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir, use_index=True)

            # Save initial version
            v1 = make_test_version("1.0.0", "test_prompt")
            storage.save_version(v1)

            # Get index entry reference
            with storage._index_lock:
                entry_before = storage._index["index"]["test_prompt"].copy()
                versions_list_id_before = id(entry_before["versions"])

            # Add another version
            v2 = make_test_version("1.0.1", "test_prompt")
            storage.save_version(v2)

            # Verify new list was created (copy-on-write)
            with storage._index_lock:
                entry_after = storage._index["index"]["test_prompt"]
                versions_list_id_after = id(entry_after["versions"])

            # List IDs should differ (new list created)
            assert versions_list_id_before != versions_list_id_after, \
                "Index update didn't create new list (not copy-on-write)"

    def test_index_async_save_no_blocking(self):
        """Test that async index save doesn't block operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir, use_index=True)

            # Save many versions rapidly
            start = datetime.now()
            for i in range(50):
                version = make_test_version(f"1.0.{i}", "test_prompt")
                storage.save_version(version)
            elapsed = (datetime.now() - start).total_seconds()

            # Should complete quickly (not blocked by synchronous saves)
            assert elapsed < 2.0, f"Save operations blocked: {elapsed}s"

            # Wait for async saves to complete
            import time
            time.sleep(0.5)

            # Verify all versions indexed
            stats = storage.get_storage_stats()
            assert stats["total_versions"] == 50

    def test_index_version_list_ordering(self):
        """Test that version list maintains correct ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir, use_index=True)

            # Add versions in random order
            versions_to_add = ["1.0.5", "1.0.1", "1.0.3", "1.0.2", "1.0.4"]
            for ver in versions_to_add:
                storage.save_version(make_test_version(ver, "test_prompt"))

            # Verify index has sorted list
            with storage._index_lock:
                indexed_versions = storage._index["index"]["test_prompt"]["versions"]

            expected = ["1.0.1", "1.0.2", "1.0.3", "1.0.4", "1.0.5"]
            assert indexed_versions == expected, \
                f"Version list not sorted: {indexed_versions}"

    def test_index_latest_version_tracking(self):
        """Test that latest_version is always correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir, use_index=True)

            # Add versions
            for ver in ["1.0.0", "1.1.0", "1.2.0"]:
                storage.save_version(make_test_version(ver, "test_prompt"))

            # Verify latest
            latest = storage.get_latest_version("test_prompt")
            assert latest.version == "1.2.0", \
                f"Latest version incorrect: {latest.version}"

            # Add higher version
            storage.save_version(make_test_version("2.0.0", "test_prompt"))

            latest = storage.get_latest_version("test_prompt")
            assert latest.version == "2.0.0"

    def test_index_dirty_flag_behavior(self):
        """Test that _index_dirty flag is managed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir, use_index=True)

            # Initially not dirty (just loaded)
            assert storage._index_dirty == False

            # Save version should set dirty flag
            storage.save_version(make_test_version("1.0.0", "test_prompt"))

            # Flag should be set (before async save)
            assert storage._index_dirty == True

            # Wait for async save
            import time
            time.sleep(0.5)

            # Flag should be cleared after save
            assert storage._index_dirty == False

    def test_index_rebuild_after_corruption(self):
        """Test that corrupted index can be rebuilt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(tmpdir, use_index=True)

            # Add versions
            for i in range(10):
                storage.save_version(make_test_version(f"1.0.{i}", "test_prompt"))

            # Corrupt index
            storage._index["total_versions"] = 999

            # Rebuild
            storage.rebuild_index()

            # Verify correct values
            stats = storage.get_storage_stats()
            assert stats["total_versions"] == 10, "Index rebuild failed"
