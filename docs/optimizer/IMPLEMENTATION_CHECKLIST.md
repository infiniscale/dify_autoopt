# FileSystemStorage Implementation Checklist

**Project:** dify_autoopt
**Module:** src/optimizer
**Component:** FileSystemStorage
**Developer:** [Assign to backend-developer]
**Start Date:** [TBD]
**Target Completion:** 4 weeks (124 hours)

---

## Quick Start

### Prerequisites

```bash
# 1. Read architecture documents (3 hours)
- [ ] ARCHITECTURE_SUMMARY.md (15 min) - Get overview
- [ ] IMPLEMENTATION_GUIDE.md Section 2 (90 min) - Technical details
- [ ] FILESYSTEM_STORAGE_ARCHITECTURE.md Section 3 (45 min) - Design
- [ ] CONFIGURATION_STANDARDIZATION.md Section 4 (20 min) - Config changes

# 2. Set up development environment
- [ ] Clone latest develop branch
- [ ] Create feature branch: feature/filesystem-storage
- [ ] Install dependencies: pip install -r requirements.txt
- [ ] Run existing tests: pytest src/test/optimizer/ -v
- [ ] Verify all 340 tests pass

# 3. Review existing code
- [ ] src/optimizer/interfaces/storage.py - VersionStorage interface
- [ ] src/optimizer/version_manager.py - Usage of storage
- [ ] src/optimizer/optimizer_service.py - Service integration
- [ ] src/test/optimizer/test_version_manager.py - Test patterns
```

---

## Phase 1: Core Implementation (Week 1)

### Day 1: Project Setup & Skeleton

**Effort:** 8 hours

```bash
# 1. Create FileSystemStorage file
- [ ] Create: src/optimizer/interfaces/filesystem_storage.py
- [ ] Import dependencies:
      from pathlib import Path
      from typing import Optional, List, Dict, Any
      import json
      import threading
      from ..models import PromptVersion
      from ..exceptions import VersionConflictError, OptimizerError
      from .storage import VersionStorage

# 2. Implement class skeleton
- [ ] Class: FileSystemStorage(VersionStorage)
- [ ] __init__() method with parameters:
      - storage_dir: str
      - use_index: bool = True
      - use_cache: bool = True
      - cache_size: int = 100
      - enable_sharding: bool = False
- [ ] Helper methods:
      - _get_prompt_dir()
      - _get_version_file()
      - _atomic_write()

# 3. Write basic tests
- [ ] Create: src/test/optimizer/test_filesystem_storage.py
- [ ] Test fixtures:
      - temp_storage_dir (pytest tmpdir)
      - sample_version (from conftest.py)
- [ ] Smoke test: test_init_creates_storage_dir()
```

**Deliverable:** FileSystemStorage skeleton compiles, basic test passes

### Day 2: save_version() Implementation

**Effort:** 8 hours

```bash
# 1. Implement save_version()
- [ ] Check for duplicate version (raise VersionConflictError)
- [ ] Create prompt directory if missing
- [ ] Serialize PromptVersion to JSON (model_dump(mode="json"))
- [ ] Atomic write pattern:
      - Write to temp file (.tmp)
      - Call fsync() to force disk write
      - Atomic rename to final path
- [ ] Error handling:
      - PermissionError → OptimizerError("FS-PERM-001")
      - IOError → OptimizerError("FS-IO-001")
      - Cleanup temp file on failure

# 2. Write comprehensive tests (8 tests)
- [ ] test_save_version_success()
- [ ] test_save_version_duplicate_raises_error()
- [ ] test_save_version_creates_directory()
- [ ] test_save_version_atomic_write_cleanup_on_error()
- [ ] test_save_version_handles_permission_error()
- [ ] test_save_version_handles_io_error()
- [ ] test_save_version_serializes_correctly()
- [ ] test_save_version_fsync_called()

# 3. Coverage check
- [ ] Run: pytest src/test/optimizer/test_filesystem_storage.py::TestSaveVersion --cov
- [ ] Target: 95%+ coverage for save_version()
```

**Deliverable:** save_version() fully implemented and tested

### Day 3: get_version() & list_versions()

**Effort:** 8 hours

```bash
# 1. Implement get_version()
- [ ] Check if version file exists (return None if not)
- [ ] Read JSON file
- [ ] Deserialize to PromptVersion
- [ ] Error handling:
      - FileNotFoundError → return None
      - JSONDecodeError → OptimizerError("FS-LOAD-001")
      - ValidationError → OptimizerError("FS-LOAD-002")

# 2. Write tests for get_version() (6 tests)
- [ ] test_get_version_found()
- [ ] test_get_version_not_found()
- [ ] test_get_version_handles_corrupted_file()
- [ ] test_get_version_handles_invalid_json()
- [ ] test_get_version_handles_validation_error()
- [ ] test_get_version_returns_correct_data()

# 3. Implement list_versions()
- [ ] Check if prompt directory exists (return [] if not)
- [ ] Iterate over *.json files in directory
- [ ] Skip metadata files (starting with '.')
- [ ] Load each version
- [ ] Sort by version number (using get_version_number())
- [ ] Return sorted list

# 4. Write tests for list_versions() (6 tests)
- [ ] test_list_versions_empty()
- [ ] test_list_versions_single()
- [ ] test_list_versions_multiple()
- [ ] test_list_versions_sorted_correctly()
- [ ] test_list_versions_skips_corrupted_files()
- [ ] test_list_versions_skips_metadata_files()

# 5. Coverage check
- [ ] Target: 95%+ coverage for both methods
```

**Deliverable:** get_version() and list_versions() fully implemented

### Day 4: Remaining Methods & Integration

**Effort:** 8 hours

```bash
# 1. Implement get_latest_version()
- [ ] Call list_versions()
- [ ] Return last element (newest) or None
- [ ] Optimize later with index (Phase 2)

# 2. Write tests for get_latest_version() (4 tests)
- [ ] test_get_latest_version_found()
- [ ] test_get_latest_version_empty()
- [ ] test_get_latest_version_multiple()
- [ ] test_get_latest_version_sorted()

# 3. Implement delete_version()
- [ ] Check if version file exists (return False if not)
- [ ] Delete file using unlink()
- [ ] Return True on success
- [ ] Error handling:
      - PermissionError → OptimizerError("FS-DELETE-001")

# 4. Write tests for delete_version() (4 tests)
- [ ] test_delete_version_success()
- [ ] test_delete_version_not_found()
- [ ] test_delete_version_handles_permission_error()
- [ ] test_delete_version_removes_file()

# 5. Implement clear_all()
- [ ] Remove entire storage directory (shutil.rmtree)
- [ ] Recreate storage directory
- [ ] Handle errors gracefully

# 6. Write tests for clear_all() (2 tests)
- [ ] test_clear_all_empty_storage()
- [ ] test_clear_all_populated_storage()

# 7. Integration test with VersionManager
- [ ] test_filesystem_storage_with_version_manager()
- [ ] Create version via VersionManager
- [ ] Retrieve via storage.get_version()
- [ ] Verify data integrity

# 8. Coverage check
- [ ] Run: pytest src/test/optimizer/ --cov=src/optimizer/interfaces/filesystem_storage
- [ ] Target: 90%+ overall coverage
```

**Deliverable:** All 6 VersionStorage methods implemented, 30 unit tests passing

### Day 5: Documentation & Code Review

**Effort:** 8 hours

```bash
# 1. Write comprehensive docstrings
- [ ] Class docstring with usage examples
- [ ] Method docstrings with Args, Returns, Raises
- [ ] Helper method docstrings
- [ ] Type hints for all parameters

# 2. Update __init__.py
- [ ] Export FileSystemStorage
- [ ] Add to __all__ list

# 3. Write usage examples
- [ ] Example 1: Basic usage
- [ ] Example 2: With custom config
- [ ] Example 3: Error handling

# 4. Code review preparation
- [ ] Run linter: ruff check src/optimizer/interfaces/filesystem_storage.py
- [ ] Run type checker: mypy src/optimizer/interfaces/filesystem_storage.py
- [ ] Run formatter: black src/optimizer/interfaces/filesystem_storage.py
- [ ] Remove debug prints and commented code

# 5. Create PR
- [ ] Title: "feat(optimizer): implement FileSystemStorage (Phase 1)"
- [ ] Description: Link to IMPLEMENTATION_GUIDE.md
- [ ] Checklist: Tests pass, coverage 90%+, docs complete
- [ ] Request reviews from 2+ developers

# 6. Address review feedback
- [ ] Fix issues identified in review
- [ ] Update tests as needed
- [ ] Rerun coverage checks
```

**Deliverable:** Phase 1 complete, PR ready for merge

---

## Phase 2: Performance Optimization (Week 2)

### Day 6: Index Design & Implementation

**Effort:** 8 hours

```bash
# 1. Design index schema
- [ ] Review FILESYSTEM_STORAGE_ARCHITECTURE.md Section 3.3
- [ ] Define IndexEntry model (Pydantic)
- [ ] Define GlobalIndex model (Pydantic)

# 2. Implement index loading
- [ ] _load_index() method
- [ ] Read .index.json file
- [ ] Deserialize to GlobalIndex
- [ ] Handle corruption (rebuild index)

# 3. Implement index saving
- [ ] _save_index() method
- [ ] Atomic write pattern (same as version files)
- [ ] Thread-safe with self._index_lock

# 4. Write tests (6 tests)
- [ ] test_load_index_success()
- [ ] test_load_index_missing_creates_new()
- [ ] test_load_index_corrupted_rebuilds()
- [ ] test_save_index_atomic()
- [ ] test_index_thread_safe()
- [ ] test_index_format_correct()
```

### Day 7: Index Updates

**Effort:** 8 hours

```bash
# 1. Implement _update_index_add()
- [ ] Add new entry or update existing
- [ ] Update latest_version
- [ ] Increment version_count
- [ ] Update timestamps
- [ ] Call _save_index()

# 2. Implement _update_index_delete()
- [ ] Remove version from versions list
- [ ] Decrement version_count
- [ ] Update latest_version (recalculate)
- [ ] Remove entry if no versions left
- [ ] Call _save_index()

# 3. Integrate with existing methods
- [ ] save_version() → _update_index_add()
- [ ] delete_version() → _update_index_delete()
- [ ] clear_all() → reset index

# 4. Optimize get_latest_version()
- [ ] Use index.latest_version if index enabled
- [ ] Fallback to list_versions() if index disabled
- [ ] Benchmark improvement (should be < 5ms)

# 5. Write tests (6 tests)
- [ ] test_update_index_add_new_entry()
- [ ] test_update_index_add_existing_entry()
- [ ] test_update_index_delete_removes_version()
- [ ] test_update_index_delete_updates_latest()
- [ ] test_update_index_delete_removes_entry_when_empty()
- [ ] test_get_latest_version_uses_index()
```

### Day 8: LRU Cache Implementation

**Effort:** 8 hours

```bash
# 1. Implement LRUCache class
- [ ] Use collections.OrderedDict
- [ ] Thread-safe with threading.RLock
- [ ] get() method (move to end on hit)
- [ ] put() method (evict oldest if full)
- [ ] remove() method
- [ ] clear() method
- [ ] stats() method

# 2. Integrate with FileSystemStorage
- [ ] Initialize cache in __init__() if use_cache=True
- [ ] get_version() → check cache first
- [ ] get_version() → update cache on disk read
- [ ] save_version() → update cache after write
- [ ] delete_version() → invalidate cache

# 3. Write cache tests (6 tests)
- [ ] test_cache_hit()
- [ ] test_cache_miss()
- [ ] test_cache_eviction()
- [ ] test_cache_thread_safe()
- [ ] test_cache_stats()
- [ ] test_cache_disabled()
```

### Day 9: Performance Benchmarks

**Effort:** 8 hours

```bash
# 1. Create performance benchmark suite
- [ ] Create: src/test/optimizer/test_filesystem_storage_performance.py
- [ ] test_write_performance_100_versions()
- [ ] test_read_performance_disk()
- [ ] test_read_performance_cache()
- [ ] test_list_versions_performance()
- [ ] test_latest_version_performance()
- [ ] test_delete_version_performance()

# 2. Run benchmarks
- [ ] Without index, without cache
- [ ] With index, without cache
- [ ] With index, with cache
- [ ] Record results in benchmark report

# 3. Verify targets met
- [ ] Write: < 20ms avg
- [ ] Read (disk): < 10ms avg
- [ ] Read (cache): < 0.1ms avg
- [ ] List 50 versions: < 50ms
- [ ] Latest version: < 5ms

# 4. Optimize if needed
- [ ] Profile slow operations
- [ ] Implement optimizations
- [ ] Re-run benchmarks
```

### Day 10: Performance Testing & Validation

**Effort:** 8 hours

```bash
# 1. Load testing
- [ ] test_large_scale_1000_prompts()
- [ ] test_many_versions_per_prompt_50()
- [ ] test_concurrent_reads_10_threads()
- [ ] Measure: throughput, latency, memory usage

# 2. Cache effectiveness
- [ ] Measure cache hit rate in typical usage
- [ ] Target: > 70% hit rate
- [ ] Analyze access patterns
- [ ] Optimize cache size if needed

# 3. Index effectiveness
- [ ] Measure index size (should be < 10MB for 10k prompts)
- [ ] Verify O(1) latest version lookup
- [ ] Test index rebuild performance

# 4. Create performance report
- [ ] Document all benchmark results
- [ ] Compare to targets
- [ ] Identify bottlenecks
- [ ] Recommend optimizations

# 5. Code review & merge
- [ ] Create PR for Phase 2
- [ ] Review with team
- [ ] Merge to feature branch
```

**Deliverable:** Phase 2 complete, performance targets met

---

## Phase 3: Production Hardening (Week 3)

### Day 11-12: File Locking

**Effort:** 16 hours

```bash
# 1. Implement FileLock class
- [ ] Cross-platform lock context manager
- [ ] Unix: fcntl.flock(LOCK_EX)
- [ ] Windows: msvcrt.locking(LK_LOCK)
- [ ] __enter__() - acquire lock
- [ ] __exit__() - release lock

# 2. Integrate with FileSystemStorage
- [ ] save_version() - lock before duplicate check
- [ ] delete_version() - lock before delete
- [ ] Use .lock files (e.g., version_file.with_suffix(".lock"))

# 3. Write locking tests (8 tests)
- [ ] test_lock_acquire_release()
- [ ] test_lock_blocks_concurrent_access()
- [ ] test_lock_released_on_exception()
- [ ] test_lock_cross_platform()
- [ ] test_concurrent_writes_same_prompt_conflict()
- [ ] test_concurrent_writes_different_prompts_succeed()
- [ ] test_lock_timeout() (optional)
- [ ] test_lock_deadlock_prevention() (optional)

# 4. Test on multiple platforms
- [ ] Windows 10/11 - msvcrt locking works
- [ ] Ubuntu 22.04 - fcntl locking works
- [ ] macOS 13 - fcntl locking works
```

### Day 13: Directory Sharding

**Effort:** 8 hours

```bash
# 1. Implement sharding logic
- [ ] _get_shard_dir() method
- [ ] SHA-256 hash of prompt_id
- [ ] Use first N chars as shard directory
- [ ] Configurable shard_depth (default: 2)

# 2. Update _get_prompt_dir()
- [ ] If enable_sharding: storage_dir / shard / prompt_id
- [ ] Else: storage_dir / prompt_id

# 3. Write sharding tests (4 tests)
- [ ] test_sharding_distributes_prompts()
- [ ] test_sharding_retrieval_works()
- [ ] test_sharding_disabled_by_default()
- [ ] test_migration_to_sharded()

# 4. Document when to enable sharding
- [ ] Disable: < 1,000 prompts
- [ ] Enable: > 1,000 prompts
- [ ] Add to configuration guide
```

### Day 14: Crash Recovery & Validation

**Effort:** 8 hours

```bash
# 1. Implement crash recovery
- [ ] _recover_from_crash() method
- [ ] Clean up .tmp files older than 1 hour
- [ ] Call on __init__()
- [ ] Log recovered files

# 2. Implement data validation
- [ ] _validate_version_file() method
- [ ] Check JSON structure
- [ ] Validate with Pydantic
- [ ] Optional: SHA-256 checksums

# 3. Write recovery tests (4 tests)
- [ ] test_recovery_cleans_tmp_files()
- [ ] test_recovery_preserves_recent_tmp()
- [ ] test_validation_detects_corruption()
- [ ] test_validation_accepts_valid()

# 4. Optional: Backup on delete
- [ ] Move to .trash/ instead of delete
- [ ] Configurable retention (7 days)
- [ ] Auto-cleanup old trash
```

### Day 15: Security Review

**Effort:** 8 hours

```bash
# 1. Security audit
- [ ] Path traversal prevention
- [ ] Filename sanitization
- [ ] File permission checks (chmod 0o600)
- [ ] No arbitrary code execution
- [ ] No SQL injection (N/A for filesystem)

# 2. Run security scanners
- [ ] bandit src/optimizer/interfaces/filesystem_storage.py
- [ ] Safety check for dependencies
- [ ] Fix all high/critical issues

# 3. Write security tests (4 tests)
- [ ] test_path_traversal_prevention()
- [ ] test_filename_sanitization()
- [ ] test_file_permissions()
- [ ] test_no_arbitrary_code_execution()

# 4. Documentation
- [ ] Security considerations section
- [ ] Best practices guide
- [ ] Known limitations (e.g., NFS)

# 5. Code review & merge
- [ ] Create PR for Phase 3
- [ ] Security review by 2+ developers
- [ ] Merge to feature branch
```

**Deliverable:** Phase 3 complete, production-hardened

---

## Phase 4: Integration & Deployment (Week 4)

### Day 16: Configuration Integration

**Effort:** 8 hours

```bash
# 1. Update EnvConfig model
- [ ] Add StorageConfig class (Pydantic)
- [ ] Fields: backend, config dict
- [ ] Validation: backend in ["memory", "filesystem", "database"]
- [ ] Add to OptimizerConfig in EnvConfig

# 2. Update OptimizerService
- [ ] Add storage factory method
- [ ] _create_storage_from_env(env: EnvConfig) -> VersionStorage
- [ ] Support memory, filesystem backends
- [ ] Inject storage into VersionManager

# 3. Update example configs
- [ ] config/env_config.example.yaml
- [ ] Add optimizer.storage section
- [ ] Document all config options

# 4. Write config tests (4 tests)
- [ ] test_storage_config_validation()
- [ ] test_storage_factory_memory()
- [ ] test_storage_factory_filesystem()
- [ ] test_invalid_backend_raises_error()
```

### Day 17: Migration Utilities

**Effort:** 8 hours

```bash
# 1. Write migration script
- [ ] scripts/migrate_optimizer_storage.py
- [ ] migrate_inmemory_to_filesystem()
- [ ] migrate_config() - update YAML
- [ ] Validate migration (count check)
- [ ] Rollback on error

# 2. Write migration tests (4 tests)
- [ ] test_migrate_inmemory_to_filesystem()
- [ ] test_migration_preserves_data()
- [ ] test_migration_rollback_on_error()
- [ ] test_config_migration()

# 3. Write verification script
- [ ] scripts/verify_storage_migration.py
- [ ] Compare version counts
- [ ] Validate data integrity
- [ ] Generate migration report

# 4. Documentation
- [ ] Migration guide
- [ ] Rollback procedure
- [ ] Troubleshooting tips
```

### Day 18: End-to-End Testing

**Effort:** 8 hours

```bash
# 1. Write E2E tests (2 tests)
- [ ] test_optimization_workflow_with_filesystem_storage()
      - Load config with FileSystemStorage
      - Run optimization cycle
      - Verify versions persisted
      - Restart service (simulate)
      - Verify versions still exist
- [ ] test_storage_backend_switching()
      - Start with InMemoryStorage
      - Migrate to FileSystemStorage
      - Verify data preserved
      - Switch back to InMemory
      - Verify data preserved

# 2. Load testing
- [ ] test_filesystem_storage_1000_prompts()
- [ ] Measure: throughput, latency, disk usage
- [ ] Verify performance targets met

# 3. Integration testing
- [ ] Test with actual WorkflowCatalog
- [ ] Test with actual TestPlan
- [ ] Test with Executor integration
- [ ] Test with Collector integration (optional)

# 4. Stress testing
- [ ] Concurrent writes (10 threads)
- [ ] Large prompts (10k chars)
- [ ] Many versions (100 per prompt)
- [ ] Verify no data loss or corruption
```

### Day 19: Deployment Preparation

**Effort:** 8 hours

```bash
# 1. Create deployment guide
- [ ] Pre-deployment checklist
- [ ] Step-by-step deployment
- [ ] Rollback procedure
- [ ] Monitoring setup

# 2. Create monitoring dashboards
- [ ] Disk usage alert (> 80%)
- [ ] Write latency alert (> 50ms)
- [ ] Error rate alert (> 1%)
- [ ] Cache hit rate metric

# 3. Create maintenance scripts
- [ ] scripts/cleanup_old_versions.py
- [ ] scripts/rebuild_index.py
- [ ] scripts/verify_index_consistency.py
- [ ] scripts/analyze_storage_metrics.py

# 4. Write operational runbook
- [ ] Daily checks
- [ ] Weekly maintenance
- [ ] Monthly backups
- [ ] Troubleshooting guide
```

### Day 20: Production Deployment

**Effort:** 8 hours

```bash
# 1. Pre-deployment
- [ ] All tests pass (380 tests)
- [ ] Coverage ≥ 90%
- [ ] Security scan clean
- [ ] Code review approved (2+)
- [ ] Documentation complete

# 2. Staging deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests
- [ ] Verify monitoring works
- [ ] Load test with production-like data

# 3. Production deployment
- [ ] Create storage directory
- [ ] Set permissions
- [ ] Update config
- [ ] Deploy code
- [ ] Restart service
- [ ] Verify health checks

# 4. Post-deployment
- [ ] Monitor for 24 hours
- [ ] Check error logs
- [ ] Verify performance metrics
- [ ] Collect user feedback

# 5. Final PR & merge
- [ ] Create PR: feature/filesystem-storage → develop
- [ ] Final review
- [ ] Merge to develop
- [ ] Tag release: v2.0.0
```

**Deliverable:** FileSystemStorage in production

---

## Success Criteria

### Technical

- [x] Architecture designed (5 documents, 118 pages)
- [ ] All 6 VersionStorage methods implemented
- [ ] 40 tests written (30 unit, 8 integration, 2 E2E)
- [ ] Coverage ≥ 90%
- [ ] Performance targets met:
  - [ ] Write: < 20ms
  - [ ] Read (disk): < 10ms
  - [ ] Read (cache): < 0.1ms
  - [ ] Cache hit rate: > 70%
- [ ] Cross-platform tested (Windows, Linux, macOS)
- [ ] No data loss in stress tests

### Quality

- [ ] Code review approved by 2+ developers
- [ ] Security scan passes (0 high/critical)
- [ ] Documentation complete (docstrings, guides, examples)
- [ ] No regressions in existing functionality

### Operational

- [ ] Deployed to production
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Zero downtime during deployment
- [ ] Rollback tested successfully

---

## Daily Standup Template

```markdown
### What I did yesterday
- [List completed tasks]

### What I'm doing today
- [List planned tasks from checklist]

### Blockers
- [Any issues or questions]

### Coverage
- Current: X%
- Target: 90%+

### Tests
- Passing: X/40
- Failing: Y
```

---

## Common Issues & Solutions

### Issue: Tests failing due to file permissions

```bash
# Solution: Use pytest tmpdir fixture
def test_save_version(tmp_path):
    storage = FileSystemStorage(str(tmp_path))
    # tmp_path is automatically cleaned up
```

### Issue: Atomic rename fails on Windows

```bash
# Solution: Use Path.replace() (atomic since Python 3.3)
temp_path.replace(final_path)  # Not temp_path.rename()
```

### Issue: File locking not working

```bash
# Solution: Use platform-specific locking
import platform
if platform.system() == "Windows":
    # Use msvcrt
else:
    # Use fcntl
```

### Issue: Tests too slow

```bash
# Solution: Mock file I/O for unit tests
from unittest.mock import patch, mock_open

@patch("builtins.open", mock_open(read_data="{}"))
def test_fast():
    # No real file I/O
```

---

## Resources

### Documentation

- [ARCHITECTURE_SUMMARY.md](./ARCHITECTURE_SUMMARY.md) - Executive overview
- [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) - Detailed implementation guide
- [FILESYSTEM_STORAGE_ARCHITECTURE.md](./FILESYSTEM_STORAGE_ARCHITECTURE.md) - Complete design

### Code Examples

- `src/optimizer/interfaces/storage.py` - VersionStorage interface
- `src/optimizer/version_manager.py` - Usage patterns
- `src/test/optimizer/test_version_manager.py` - Test patterns

### Tools

- pytest - Testing framework
- pytest-cov - Coverage reporting
- black - Code formatter
- ruff - Linter
- mypy - Type checker
- bandit - Security scanner

---

**Checklist Version:** 1.0.0
**Last Updated:** 2025-01-17
**Assigned To:** [backend-developer]
**Status:** Ready to Start

---

*This checklist provides a day-by-day task breakdown for implementing FileSystemStorage. Check off items as you complete them. Update daily standup with progress.*
