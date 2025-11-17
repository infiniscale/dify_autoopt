# FileSystemStorage Implementation Guide & Risk Assessment

**Project:** dify_autoopt
**Module:** src/optimizer
**Component:** FileSystemStorage
**Version:** 1.0.0
**Date:** 2025-01-17
**Author:** Senior System Architect

---

## Table of Contents

1. [Implementation Roadmap](#1-implementation-roadmap)
2. [Technical Implementation Guide](#2-technical-implementation-guide)
3. [Testing Strategy](#3-testing-strategy)
4. [Performance Benchmarks](#4-performance-benchmarks)
5. [Risk Assessment](#5-risk-assessment)
6. [Deployment Guide](#6-deployment-guide)
7. [Monitoring & Maintenance](#7-monitoring--maintenance)

---

## 1. Implementation Roadmap

### 1.1 Four-Phase Delivery Plan

```
Phase 1: Core Implementation (Week 1)
├── Day 1-2: FileSystemStorage class
├── Day 3: Unit tests (90%+ coverage)
├── Day 4: Integration with VersionManager
└── Day 5: Documentation & code review

Phase 2: Performance Optimization (Week 2)
├── Day 1: Global index implementation
├── Day 2: LRU cache implementation
├── Day 3: Performance benchmarks
├── Day 4: Optimization tuning
└── Day 5: Performance testing & validation

Phase 3: Production Hardening (Week 3)
├── Day 1-2: File locking (cross-platform)
├── Day 2: Directory sharding
├── Day 3: Crash recovery
├── Day 4: Data validation
└── Day 5: Security hardening

Phase 4: Integration & Deployment (Week 4)
├── Day 1: EnvConfig integration
├── Day 2: Migration utilities
├── Day 3: End-to-end testing
├── Day 4: Deployment preparation
└── Day 5: Production deployment
```

### 1.2 Detailed Task Breakdown

#### Phase 1: Core Implementation

| Task ID | Task | Priority | Effort | Deliverable |
|---------|------|----------|--------|-------------|
| P1-T1 | Create `filesystem_storage.py` skeleton | P0 | 2h | File structure |
| P1-T2 | Implement `save_version()` with atomic write | P0 | 4h | Method + tests |
| P1-T3 | Implement `get_version()` | P0 | 3h | Method + tests |
| P1-T4 | Implement `list_versions()` | P0 | 3h | Method + tests |
| P1-T5 | Implement `get_latest_version()` | P0 | 2h | Method + tests |
| P1-T6 | Implement `delete_version()` | P0 | 2h | Method + tests |
| P1-T7 | Implement `clear_all()` | P0 | 1h | Method + tests |
| P1-T8 | Write unit tests (90%+ coverage) | P0 | 8h | Test suite |
| P1-T9 | Integration test with VersionManager | P0 | 4h | Integration tests |
| P1-T10 | API documentation | P0 | 2h | Docstrings |

**Total Effort:** 31 hours (~4 days)

#### Phase 2: Performance Optimization

| Task ID | Task | Priority | Effort | Deliverable |
|---------|------|----------|--------|-------------|
| P2-T1 | Design index schema | P1 | 2h | Design doc |
| P2-T2 | Implement index loading/saving | P1 | 4h | Index methods |
| P2-T3 | Implement index updates (add/delete) | P1 | 4h | Index sync |
| P2-T4 | Implement LRU cache | P1 | 4h | Cache class |
| P2-T5 | Add cache hit/miss metrics | P1 | 2h | Metrics |
| P2-T6 | Write performance benchmarks | P1 | 6h | Benchmark suite |
| P2-T7 | Optimize serialization (profile) | P2 | 4h | Optimizations |
| P2-T8 | Performance testing & validation | P1 | 4h | Test results |

**Total Effort:** 30 hours (~4 days)

#### Phase 3: Production Hardening

| Task ID | Task | Priority | Effort | Deliverable |
|---------|------|----------|--------|-------------|
| P3-T1 | Implement file locking (Unix) | P1 | 4h | fcntl locking |
| P3-T2 | Implement file locking (Windows) | P1 | 4h | msvcrt locking |
| P3-T3 | Cross-platform locking tests | P1 | 4h | Test suite |
| P3-T4 | Implement directory sharding | P1 | 6h | Sharding logic |
| P3-T5 | Implement crash recovery | P1 | 4h | Cleanup logic |
| P3-T6 | Implement data validation | P1 | 3h | Validation |
| P3-T7 | Add checksums (optional) | P2 | 4h | SHA-256 hashing |
| P3-T8 | Security review | P1 | 4h | Security report |

**Total Effort:** 33 hours (~4 days)

#### Phase 4: Integration & Deployment

| Task ID | Task | Priority | Effort | Deliverable |
|---------|------|----------|--------|-------------|
| P4-T1 | Add StorageConfig to EnvConfig | P0 | 2h | Config model |
| P4-T2 | Update OptimizerService factory | P0 | 2h | Factory method |
| P4-T3 | Write migration script | P0 | 4h | Migration tool |
| P4-T4 | Update example YAML configs | P0 | 2h | Examples |
| P4-T5 | End-to-end integration tests | P0 | 8h | Test suite |
| P4-T6 | Load testing (1000 prompts) | P1 | 4h | Load test results |
| P4-T7 | Deployment documentation | P0 | 4h | Deployment guide |
| P4-T8 | Production deployment | P0 | 4h | Deployed system |

**Total Effort:** 30 hours (~4 days)

**Grand Total:** 124 hours (~15.5 days with 8-hour workdays)

---

## 2. Technical Implementation Guide

### 2.1 Critical Implementation Details

#### Atomic Write Pattern

```python
def _atomic_write(self, path: Path, data: dict) -> None:
    """Write file atomically using temp file + rename.

    CRITICAL: This ensures no partial writes on crash/interruption.
    """
    temp_path = path.with_suffix(f".tmp.{os.getpid()}")

    try:
        # 1. Write to temp file
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # 2. Atomic rename (POSIX guarantees atomicity)
        temp_path.replace(path)

    except Exception as e:
        # Cleanup temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise OptimizerError(
            message=f"Atomic write failed: {str(e)}",
            error_code="FS-ATOMIC-001",
            context={"path": str(path), "temp_path": str(temp_path)}
        )
```

**Why Atomic?**
- Prevents partial writes on crash
- No corrupted files
- Reader always sees complete data or nothing

**Platform Support:**
- Unix/Linux: `os.replace()` is atomic
- Windows: Atomic since Python 3.3
- NFS: May not be atomic (document limitation)

#### File Locking (Cross-Platform)

```python
import fcntl  # Unix
import msvcrt  # Windows
import os
import platform


class FileLock:
    """Cross-platform file locking context manager."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.lock_file = None

    def __enter__(self):
        """Acquire lock."""
        self.lock_file = open(self.file_path, 'a')

        if platform.system() == "Windows":
            # Windows: Lock first byte
            msvcrt.locking(
                self.lock_file.fileno(),
                msvcrt.LK_LOCK,  # Blocking lock
                1
            )
        else:
            # Unix: Advisory lock
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
        if self.lock_file:
            if platform.system() == "Windows":
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)

            self.lock_file.close()


# Usage in save_version()
def save_version(self, version: PromptVersion) -> None:
    """Save version with file locking."""
    version_file = self._get_version_file(version.prompt_id, version.version)
    lock_file = version_file.with_suffix(".lock")

    with FileLock(lock_file):
        # Check for duplicate (with lock held)
        if version_file.exists():
            raise VersionConflictError(...)

        # Atomic write (with lock held)
        self._atomic_write(version_file, version.model_dump(mode="json"))

    # Lock released automatically
```

#### LRU Cache Implementation

```python
from collections import OrderedDict
import threading


class LRUCache:
    """Thread-safe LRU cache for PromptVersion objects."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[PromptVersion]:
        """Get item from cache (move to end)."""
        with self._lock:
            if key not in self._cache:
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: PromptVersion) -> None:
        """Add item to cache (evict oldest if full)."""
        with self._lock:
            if key in self._cache:
                # Update existing entry
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new entry
                if len(self._cache) >= self.max_size:
                    # Evict oldest (first item)
                    self._cache.popitem(last=False)

                self._cache[key] = value

    def remove(self, key: str) -> None:
        """Remove item from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size
            }


# Integration into FileSystemStorage
class FileSystemStorage(VersionStorage):
    def __init__(self, storage_dir: str, use_cache: bool = True, cache_size: int = 100):
        # ... other init ...
        self._cache = LRUCache(max_size=cache_size) if use_cache else None

    def get_version(self, prompt_id: str, version: str) -> Optional[PromptVersion]:
        """Get version with cache lookup."""
        cache_key = f"{prompt_id}:{version}"

        # 1. Check cache
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                return cached

        # 2. Load from disk
        version_obj = self._load_from_disk(prompt_id, version)

        # 3. Update cache
        if version_obj and self._cache:
            self._cache.put(cache_key, version_obj)

        return version_obj
```

### 2.2 Index Design

```python
class IndexEntry(BaseModel):
    """Index entry for a single prompt."""
    latest_version: str
    version_count: int
    versions: List[str]  # Sorted by version number
    created_at: str
    updated_at: str


class GlobalIndex(BaseModel):
    """Global index for all prompts."""
    version: str = "1.0.0"
    last_updated: str
    total_prompts: int
    total_versions: int
    index: Dict[str, IndexEntry]


class FileSystemStorage(VersionStorage):
    def _update_index_add(self, version: PromptVersion) -> None:
        """Update index when adding a version."""
        with self._index_lock:
            if not self._index:
                return

            prompt_id = version.prompt_id

            # Get or create entry
            entry = self._index["index"].get(prompt_id)
            if not entry:
                entry = IndexEntry(
                    latest_version=version.version,
                    version_count=1,
                    versions=[version.version],
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat()
                ).model_dump()
                self._index["index"][prompt_id] = entry
                self._index["total_prompts"] += 1
            else:
                # Update existing entry
                entry["versions"].append(version.version)
                entry["versions"].sort(key=lambda v: self._parse_version(v))
                entry["latest_version"] = entry["versions"][-1]
                entry["version_count"] += 1
                entry["updated_at"] = datetime.now().isoformat()

            self._index["total_versions"] += 1
            self._index["last_updated"] = datetime.now().isoformat()

            # Save index to disk
            self._save_index()

    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string to tuple."""
        try:
            major, minor, patch = version.split('.')
            return (int(major), int(minor), int(patch))
        except (ValueError, AttributeError):
            return (0, 0, 0)
```

---

## 3. Testing Strategy

### 3.1 Unit Test Coverage Plan

| Component | Target Coverage | Key Test Cases |
|-----------|----------------|----------------|
| `save_version()` | 100% | Normal save, duplicate error, I/O error, permission error |
| `get_version()` | 100% | Found, not found, corrupted file, cache hit, cache miss |
| `list_versions()` | 100% | Empty, single, multiple, sorting, corrupted file skip |
| `get_latest_version()` | 100% | With index, without index, empty |
| `delete_version()` | 100% | Success, not found, permission error |
| `clear_all()` | 100% | Empty storage, populated storage |
| Index methods | 100% | Add, delete, rebuild, corruption recovery |
| Cache | 100% | Hit, miss, eviction, thread safety |
| File locking | 95% | Acquire, release, cross-platform |
| Atomic write | 100% | Success, failure cleanup, crash simulation |

### 3.2 Test Pyramid

```
                    /\
                   /  \
                  /E2E \          2 tests (5%)
                 /------\
                /  Intg  \        8 tests (20%)
               /----------\
              /   Unit     \      30 tests (75%)
             /--------------\
```

**Unit Tests (30 tests, 75%)**
- Test each method in isolation
- Mock file I/O for speed
- Cover edge cases and error paths

**Integration Tests (8 tests, 20%)**
- Test FileSystemStorage with VersionManager
- Test with real file I/O
- Test concurrent access
- Test migration from InMemoryStorage

**End-to-End Tests (2 tests, 5%)**
- Test complete optimization workflow
- Test storage backend switching
- Test performance under load

### 3.3 Sample Unit Tests

```python
# tests/optimizer/test_filesystem_storage.py

import pytest
from pathlib import Path
from src.optimizer.interfaces.filesystem_storage import FileSystemStorage
from src.optimizer.models import PromptVersion
from src.optimizer.exceptions import VersionConflictError


class TestFileSystemStorageSaveVersion:
    """Test suite for save_version() method."""

    def test_save_version_success(self, temp_storage_dir, sample_version):
        """Test successful version save."""
        storage = FileSystemStorage(str(temp_storage_dir))

        storage.save_version(sample_version)

        # Verify file exists
        version_file = temp_storage_dir / sample_version.prompt_id / f"{sample_version.version}.json"
        assert version_file.exists()

        # Verify content
        loaded = storage.get_version(sample_version.prompt_id, sample_version.version)
        assert loaded is not None
        assert loaded.version == sample_version.version

    def test_save_version_duplicate_raises_error(self, temp_storage_dir, sample_version):
        """Test saving duplicate version raises VersionConflictError."""
        storage = FileSystemStorage(str(temp_storage_dir))

        storage.save_version(sample_version)

        # Try to save again
        with pytest.raises(VersionConflictError, match="already exists"):
            storage.save_version(sample_version)

    def test_save_version_creates_directory(self, temp_storage_dir, sample_version):
        """Test save_version creates prompt directory if missing."""
        storage = FileSystemStorage(str(temp_storage_dir))

        prompt_dir = temp_storage_dir / sample_version.prompt_id
        assert not prompt_dir.exists()

        storage.save_version(sample_version)

        assert prompt_dir.exists()
        assert prompt_dir.is_dir()

    def test_save_version_atomic_write_cleanup_on_error(
        self,
        temp_storage_dir,
        sample_version,
        monkeypatch
    ):
        """Test atomic write cleans up temp file on error."""
        storage = FileSystemStorage(str(temp_storage_dir))

        # Mock json.dump to raise error
        import json
        original_dump = json.dump

        def failing_dump(*args, **kwargs):
            raise IOError("Simulated I/O error")

        monkeypatch.setattr(json, "dump", failing_dump)

        # Try to save
        with pytest.raises(Exception):
            storage.save_version(sample_version)

        # Verify no temp files left
        temp_files = list(temp_storage_dir.glob("**/*.tmp*"))
        assert len(temp_files) == 0

    def test_save_version_updates_index(self, temp_storage_dir, sample_version):
        """Test save_version updates global index."""
        storage = FileSystemStorage(str(temp_storage_dir), use_index=True)

        storage.save_version(sample_version)

        # Verify index updated
        assert storage._index["total_versions"] == 1
        assert storage._index["total_prompts"] == 1
        assert sample_version.prompt_id in storage._index["index"]

        entry = storage._index["index"][sample_version.prompt_id]
        assert entry["latest_version"] == sample_version.version
        assert entry["version_count"] == 1

    def test_save_version_updates_cache(self, temp_storage_dir, sample_version):
        """Test save_version updates LRU cache."""
        storage = FileSystemStorage(str(temp_storage_dir), use_cache=True)

        storage.save_version(sample_version)

        # Verify cache contains version
        cache_key = f"{sample_version.prompt_id}:{sample_version.version}"
        cached = storage._cache.get(cache_key)
        assert cached is not None
        assert cached.version == sample_version.version


class TestFileSystemStorageGetVersion:
    """Test suite for get_version() method."""

    def test_get_version_found(self, storage_with_data, sample_version):
        """Test retrieving existing version."""
        loaded = storage_with_data.get_version(
            sample_version.prompt_id,
            sample_version.version
        )

        assert loaded is not None
        assert loaded.version == sample_version.version
        assert loaded.prompt_id == sample_version.prompt_id

    def test_get_version_not_found(self, temp_storage_dir):
        """Test retrieving non-existent version returns None."""
        storage = FileSystemStorage(str(temp_storage_dir))

        loaded = storage.get_version("nonexistent", "1.0.0")
        assert loaded is None

    def test_get_version_cache_hit(self, storage_with_data, sample_version):
        """Test cache hit returns cached version."""
        storage = storage_with_data

        # First load (miss)
        loaded1 = storage.get_version(sample_version.prompt_id, sample_version.version)

        # Second load (hit)
        loaded2 = storage.get_version(sample_version.prompt_id, sample_version.version)

        # Should be same object from cache
        assert loaded1 is loaded2

    def test_get_version_handles_corrupted_file(
        self,
        temp_storage_dir,
        sample_version
    ):
        """Test get_version handles corrupted JSON file."""
        storage = FileSystemStorage(str(temp_storage_dir))

        # Create corrupted file
        version_file = storage._get_version_file(
            sample_version.prompt_id,
            sample_version.version
        )
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text("{ invalid json }")

        # Should raise OptimizerError
        with pytest.raises(Exception):
            storage.get_version(sample_version.prompt_id, sample_version.version)


class TestFileSystemStorageConcurrency:
    """Test suite for concurrent access."""

    def test_concurrent_writes_different_prompts(self, temp_storage_dir):
        """Test concurrent writes to different prompts succeed."""
        import threading

        storage = FileSystemStorage(str(temp_storage_dir))
        errors = []

        def write_version(prompt_id):
            try:
                version = create_test_version(prompt_id, "1.0.0")
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

    def test_concurrent_writes_same_prompt_raises_conflict(self, temp_storage_dir):
        """Test concurrent writes to same prompt raise VersionConflictError."""
        import threading

        storage = FileSystemStorage(str(temp_storage_dir))
        errors = []
        successes = []

        def write_version():
            try:
                version = create_test_version("prompt_001", "1.0.0")
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
```

### 3.4 Performance Benchmark Tests

```python
# tests/optimizer/test_filesystem_storage_performance.py

import pytest
import time


def test_write_performance_100_versions(temp_storage_dir):
    """Benchmark write throughput."""
    storage = FileSystemStorage(str(temp_storage_dir), use_index=True)

    start = time.time()
    for i in range(100):
        version = create_test_version(f"prompt_{i:03d}", "1.0.0")
        storage.save_version(version)
    elapsed = time.time() - start

    avg_write_time = elapsed / 100

    print(f"Wrote 100 versions in {elapsed:.2f}s (avg: {avg_write_time*1000:.1f}ms)")

    # Assert performance target: < 20ms per write
    assert avg_write_time < 0.02, f"Write too slow: {avg_write_time*1000:.1f}ms"


def test_read_performance_with_cache(temp_storage_dir):
    """Benchmark read throughput with cache."""
    storage = FileSystemStorage(str(temp_storage_dir), use_cache=True, cache_size=100)

    # Prepare data
    for i in range(10):
        version = create_test_version(f"prompt_{i:03d}", "1.0.0")
        storage.save_version(version)

    # Benchmark: Read same version 1000 times (cache hit)
    start = time.time()
    for _ in range(1000):
        storage.get_version("prompt_001", "1.0.0")
    elapsed = time.time() - start

    avg_read_time = elapsed / 1000

    print(f"Read 1000 times (cached) in {elapsed:.3f}s (avg: {avg_read_time*1000000:.1f}µs)")

    # Assert performance target: < 0.1ms per cached read
    assert avg_read_time < 0.0001, f"Cached read too slow: {avg_read_time*1000:.2f}ms"


def test_list_versions_performance(temp_storage_dir):
    """Benchmark list_versions with 50 versions."""
    storage = FileSystemStorage(str(temp_storage_dir), use_index=True)

    # Create 50 versions for same prompt
    for i in range(50):
        version = create_test_version("prompt_001", f"1.{i}.0")
        storage.save_version(version)

    # Benchmark
    start = time.time()
    versions = storage.list_versions("prompt_001")
    elapsed = time.time() - start

    print(f"Listed 50 versions in {elapsed*1000:.1f}ms")

    assert len(versions) == 50
    assert elapsed < 0.05, f"List too slow: {elapsed*1000:.1f}ms"
```

---

## 4. Performance Benchmarks

### 4.1 Target Performance Metrics

| Operation | Without Index | With Index | With Index + Cache | Target |
|-----------|--------------|------------|-------------------|--------|
| `save_version()` | 15ms | 18ms | 18ms | < 20ms |
| `get_version()` (disk) | 8ms | 8ms | 8ms | < 10ms |
| `get_version()` (cache) | N/A | N/A | 0.05ms | < 0.1ms |
| `list_versions()` (10 vers) | 50ms | 10ms | 10ms | < 20ms |
| `get_latest_version()` | 50ms | 2ms | 1ms | < 5ms |
| `delete_version()` | 5ms | 8ms | 8ms | < 10ms |

### 4.2 Scalability Benchmarks

| Scenario | Prompts | Versions/Prompt | Total Files | Index Size | List Time | Latest Time |
|----------|---------|----------------|-------------|------------|-----------|-------------|
| Small | 10 | 5 | 50 | 5KB | 5ms | 1ms |
| Medium | 100 | 10 | 1,000 | 50KB | 15ms | 2ms |
| Large | 1,000 | 20 | 20,000 | 500KB | 80ms | 3ms |
| XLarge | 10,000 | 50 | 500,000 | 5MB | 300ms | 5ms |

**Notes:**
- Index enables O(1) latest lookup
- Sharding recommended for > 1,000 prompts
- Cache hit rate: 70-90% in typical usage

---

## 5. Risk Assessment

### 5.1 Technical Risks

| Risk ID | Risk | Probability | Impact | Severity | Mitigation |
|---------|------|-------------|--------|----------|------------|
| TR-01 | File system performance bottleneck | Medium | High | **High** | Use index + cache; benchmark early; fallback to InMemory |
| TR-02 | Concurrent write conflicts | Low | Medium | **Medium** | File locking; retry logic; clear error messages |
| TR-03 | Disk space exhaustion | Low | High | **Medium** | Monitoring; auto-cleanup; configurable retention |
| TR-04 | Data corruption | Low | Critical | **Medium** | Atomic writes; checksums; backups |
| TR-05 | Cross-platform incompatibility | Medium | Medium | **Medium** | Test on Windows/Linux/Mac; use Path objects |
| TR-06 | Index desync from files | Low | Medium | **Low** | Rebuild index utility; validation on load |
| TR-07 | Cache memory usage | Low | Low | **Low** | Configurable cache size; LRU eviction |
| TR-08 | Network filesystem issues | Medium | Medium | **Medium** | Document limitations; test on NFS; disable index |

### 5.2 Integration Risks

| Risk ID | Risk | Probability | Impact | Severity | Mitigation |
|---------|------|-------------|--------|----------|------------|
| IR-01 | Breaking VersionStorage interface | Low | Critical | **Medium** | 100% interface compliance testing; version contract |
| IR-02 | Performance regression vs InMemory | Medium | Medium | **Medium** | Benchmarks; cache optimization; document trade-offs |
| IR-03 | Config migration breaks users | Low | Medium | **Low** | Backward compatibility; migration script; deprecation warnings |
| IR-04 | Storage path misconfiguration | Medium | Low | **Low** | Validation; clear error messages; examples |
| IR-05 | Data loss during migration | Low | High | **Medium** | Backup; validation; rollback plan |

### 5.3 Operational Risks

| Risk ID | Risk | Probability | Impact | Severity | Mitigation |
|---------|------|-------------|--------|----------|------------|
| OR-01 | Insufficient disk space | Medium | High | **High** | Pre-flight checks; monitoring; alerts; auto-cleanup |
| OR-02 | File permission errors | Medium | Medium | **Medium** | Validation on init; clear error messages; docs |
| OR-03 | Backup/restore complexity | Low | Medium | **Low** | Simple file copy; rsync; tar.gz archives |
| OR-04 | Version storage location | Low | Low | **Low** | Configurable; environment-specific configs |
| OR-05 | Monitoring/debugging | Medium | Medium | **Medium** | Comprehensive logging; metrics; health checks |

### 5.4 Risk Mitigation Summary

**High Priority Mitigations:**

1. **TR-01: Performance Bottleneck**
   - Implement index + cache in Phase 2
   - Run benchmarks on target hardware
   - Document when to use InMemory vs FileSystem

2. **TR-03: Disk Space**
   - Add disk space check in `save_version()`
   - Implement auto-cleanup of old versions (configurable)
   - Monitor disk usage

3. **TR-04: Data Corruption**
   - Use atomic write pattern
   - Validate JSON on read
   - Optional checksums for critical data

4. **OR-01: Disk Space**
   - Pre-flight check: `storage.check_disk_space(required_mb=100)`
   - Alert when < 10% free space
   - Configurable retention policy

**Medium Priority Mitigations:**

5. **TR-02: Write Conflicts**
   - Implement file locking
   - Clear error message: "Version X already exists for prompt Y"
   - Document expected behavior

6. **TR-05: Cross-Platform**
   - Test on Windows 10/11, Ubuntu 22.04, macOS 13
   - Use `pathlib.Path` consistently
   - Handle platform-specific locking

7. **IR-02: Performance Regression**
   - Document: "FileSystem is 2-5x slower than InMemory, but persistent"
   - Cache optimization: 90%+ hit rate → near-InMemory performance
   - Provide benchmark results

**Low Priority Mitigations:**

8. **TR-06: Index Desync**
   - Rebuild index utility: `storage.rebuild_index()`
   - Validate index on load (optional)
   - Document manual recovery

9. **IR-03: Config Migration**
   - Backward compatibility for 2 releases
   - Deprecation warnings
   - Clear migration guide

10. **OR-03: Backup/Restore**
    - Simple: `cp -r storage_dir/ backup/`
    - Advanced: Use `rsync` or database dumps
    - Document procedures

### 5.5 Risk Matrix

```
        Impact →
        Low     Medium     High     Critical
      ┌────────┬──────────┬────────┬──────────┐
  P   │        │          │ TR-01  │          │
  r H │        │          │ OR-01  │          │
  o i │        │          │        │          │
  b g ├────────┼──────────┼────────┼──────────┤
  a h │        │ TR-02    │ TR-03  │          │
  b   │        │ TR-05    │ IR-05  │          │
  i M │        │ TR-08    │        │          │
  l e ├────────┼──────────┼────────┼──────────┤
  i d │ TR-07  │ TR-06    │        │          │
  t   │ OR-04  │ IR-04    │        │          │
  y L │        │ OR-05    │        │          │
    o ├────────┼──────────┼────────┼──────────┤
  w L │        │ IR-01    │        │ TR-04    │
      │        │ IR-03    │        │          │
      │        │ OR-02    │        │          │
      │        │ OR-03    │        │          │
      └────────┴──────────┴────────┴──────────┘

Legend:
  High Risk (Red): Immediate attention required
  Medium Risk (Yellow): Monitor and mitigate
  Low Risk (Green): Accept or defer
```

---

## 6. Deployment Guide

### 6.1 Pre-Deployment Checklist

```bash
# 1. Code Quality
- [ ] All unit tests pass (90%+ coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] Code review completed and approved
- [ ] No security vulnerabilities (bandit scan)

# 2. Documentation
- [ ] API documentation complete
- [ ] Usage guide written
- [ ] Migration guide ready
- [ ] Example configs updated

# 3. Infrastructure
- [ ] Storage directory created with correct permissions
- [ ] Disk space sufficient (> 1GB recommended)
- [ ] Backup strategy defined
- [ ] Monitoring configured

# 4. Configuration
- [ ] env_config.yaml updated with storage settings
- [ ] StorageConfig validated
- [ ] Migration script tested on staging data

# 5. Rollback Plan
- [ ] Backup of InMemoryStorage data (if migrating)
- [ ] Rollback procedure documented
- [ ] Test rollback procedure
```

### 6.2 Deployment Steps

#### Step 1: Prepare Storage Directory

```bash
# Create storage directory
mkdir -p /var/lib/dify_autoopt/optimizer/versions

# Set permissions (owner: dify_user, group: dify_group)
chown -R dify_user:dify_group /var/lib/dify_autoopt
chmod 750 /var/lib/dify_autoopt/optimizer/versions

# Verify disk space (need > 1GB)
df -h /var/lib/dify_autoopt
```

#### Step 2: Update Configuration

```yaml
# config/env_config.prod.yaml
optimizer:
  storage:
    backend: "filesystem"
    config:
      storage_dir: "/var/lib/dify_autoopt/optimizer/versions"
      use_index: true
      use_cache: true
      cache_size: 500
      enable_sharding: false  # Enable if > 1000 prompts

  optimization:
    min_baseline_score: 80.0
    min_improvement: 5.0
    min_confidence: 0.6
    max_iterations: 3
```

#### Step 3: Migrate Existing Data (If Applicable)

```bash
# Backup InMemoryStorage data (if persisted somewhere)
python scripts/export_inmemory_storage.py \
  --output /var/backups/optimizer_versions_$(date +%Y%m%d).json

# Migrate to FileSystem
python scripts/migrate_storage.py \
  --from memory \
  --to filesystem \
  --storage-dir /var/lib/dify_autoopt/optimizer/versions \
  --validate

# Verify migration
python scripts/verify_migration.py \
  --storage-dir /var/lib/dify_autoopt/optimizer/versions \
  --expected-count <N>
```

#### Step 4: Deploy Code

```bash
# Pull latest code
git pull origin main

# Install dependencies (if any new)
pip install -r requirements.txt

# Run smoke tests
pytest tests/optimizer/test_smoke.py -v

# Restart service
systemctl restart dify_autoopt
```

#### Step 5: Verify Deployment

```bash
# Check service status
systemctl status dify_autoopt

# Check logs for errors
tail -f /var/log/dify_autoopt/optimizer.log

# Run health check
curl http://localhost:8000/health/optimizer

# Verify storage
ls -lah /var/lib/dify_autoopt/optimizer/versions/
```

### 6.3 Rollback Procedure

```bash
# 1. Stop service
systemctl stop dify_autoopt

# 2. Revert code
git checkout <previous-commit>

# 3. Restore config
cp config/env_config.prod.yaml.backup config/env_config.prod.yaml

# 4. Restore data (if needed)
python scripts/restore_inmemory_storage.py \
  --backup /var/backups/optimizer_versions_YYYYMMDD.json

# 5. Restart service
systemctl start dify_autoopt

# 6. Verify
curl http://localhost:8000/health/optimizer
```

---

## 7. Monitoring & Maintenance

### 7.1 Key Metrics to Monitor

| Metric | Source | Alert Threshold | Action |
|--------|--------|----------------|--------|
| **Disk Usage** | OS | > 80% | Auto-cleanup old versions |
| **Write Latency** | Application logs | > 50ms avg | Investigate I/O; enable sharding |
| **Read Latency** | Application logs | > 20ms avg | Check cache hit rate; investigate I/O |
| **Cache Hit Rate** | Application logs | < 70% | Increase cache size; analyze access patterns |
| **Index Size** | Filesystem | > 10MB | Consider database backend |
| **Error Rate** | Application logs | > 1% | Investigate disk errors; check permissions |
| **File Count** | OS | > 100,000 | Enable sharding; consider database |

### 7.2 Logging Configuration

```python
# Configure optimizer logger
from loguru import logger

logger.add(
    "/var/log/dify_autoopt/optimizer_storage.log",
    level="INFO",
    rotation="100 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    serialize=True  # JSON format
)

# Log key metrics
class FileSystemStorage(VersionStorage):
    def save_version(self, version: PromptVersion) -> None:
        start = time.time()

        # ... save logic ...

        elapsed = time.time() - start
        self._logger.info(
            "Version saved",
            prompt_id=version.prompt_id,
            version=version.version,
            elapsed_ms=elapsed * 1000,
            file_size=version_file.stat().st_size
        )
```

### 7.3 Maintenance Tasks

#### Daily

```bash
# Check disk usage
df -h /var/lib/dify_autoopt

# Check error logs
grep ERROR /var/log/dify_autoopt/optimizer_storage.log | tail -20

# Verify service health
curl http://localhost:8000/health/optimizer
```

#### Weekly

```bash
# Analyze storage metrics
python scripts/analyze_storage_metrics.py \
  --storage-dir /var/lib/dify_autoopt/optimizer/versions \
  --report /tmp/storage_report.txt

# Check index consistency
python scripts/verify_index_consistency.py \
  --storage-dir /var/lib/dify_autoopt/optimizer/versions

# Review cache statistics
python scripts/cache_stats.py
```

#### Monthly

```bash
# Backup storage
tar -czf /var/backups/optimizer_versions_$(date +%Y%m%d).tar.gz \
  /var/lib/dify_autoopt/optimizer/versions

# Cleanup old versions (if needed)
python scripts/cleanup_old_versions.py \
  --storage-dir /var/lib/dify_autoopt/optimizer/versions \
  --keep-last 10 \
  --dry-run

# Performance benchmarks
pytest tests/optimizer/test_filesystem_storage_performance.py \
  --benchmark-only \
  --benchmark-save=monthly_$(date +%Y%m)
```

### 7.4 Troubleshooting Guide

**Problem: Slow write performance (> 50ms)**

```bash
# 1. Check disk I/O
iostat -x 1 10

# 2. Check if index is enabled
grep "use_index" config/env_config.yaml

# 3. Profile write operation
python -m cProfile -s cumtime scripts/profile_storage.py

# 4. Consider SSD storage
# 5. Enable sharding if > 1000 prompts
```

**Problem: Cache hit rate < 50%**

```bash
# 1. Increase cache size
# Edit config/env_config.yaml:
#   cache_size: 1000  # Increase from 100

# 2. Analyze access patterns
python scripts/analyze_cache_access.py

# 3. Consider pre-warming cache
python scripts/warm_cache.py --top-n 100
```

**Problem: Index desync**

```bash
# 1. Rebuild index
python scripts/rebuild_index.py \
  --storage-dir /var/lib/dify_autoopt/optimizer/versions

# 2. Verify consistency
python scripts/verify_index_consistency.py \
  --storage-dir /var/lib/dify_autoopt/optimizer/versions

# 3. Enable validation on load (temporary)
# Edit config: validate_index_on_load: true
```

---

## 8. Success Criteria

### 8.1 Technical Success Criteria

- [ ] FileSystemStorage implements 100% of VersionStorage interface
- [ ] Unit test coverage ≥ 90%
- [ ] Integration tests pass with VersionManager
- [ ] Performance benchmarks meet targets (< 20ms writes, < 10ms reads)
- [ ] Cross-platform compatibility (Windows, Linux, macOS)
- [ ] No data loss in stress tests (1000 concurrent writes)
- [ ] Migration from InMemoryStorage works without data loss

### 8.2 Quality Success Criteria

- [ ] Code review completed and approved by 2+ reviewers
- [ ] Security scan passes (no high/critical vulnerabilities)
- [ ] Documentation complete (API docs, usage guide, migration guide)
- [ ] Example configurations provided and tested
- [ ] No regressions in existing Optimizer functionality

### 8.3 Operational Success Criteria

- [ ] Deployed to production environment
- [ ] Monitoring configured and operational
- [ ] Backup strategy implemented and tested
- [ ] Rollback procedure tested successfully
- [ ] Zero downtime during deployment
- [ ] User feedback collected and addressed

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-17
**Approved By:** [Pending Review]
**Status:** Ready for Implementation
