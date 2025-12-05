# FileSystemStorage Architecture Design Document

**Project:** dify_autoopt
**Module:** src/optimizer
**Component:** FileSystemStorage
**Version:** 1.0.0
**Date:** 2025-01-17
**Author:** Senior System Architect
**Status:** Design Review

---

## Executive Summary

This document provides a comprehensive architecture design for implementing `FileSystemStorage` as a persistent version storage backend for the Optimizer module. The design addresses the 4 system-level documentation-implementation inconsistencies identified by Codex, with emphasis on production-grade quality, cross-module interaction, and long-term extensibility.

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **File Format** | JSON with UTF-8 encoding | Human-readable, language-agnostic, supports metadata |
| **Directory Structure** | `prompt_id/version.json` + `.index.json` | Fast lookup, natural grouping, optional indexing |
| **Concurrency** | File-level locking with `fcntl/msvcrt` | Cross-platform, simple, proven |
| **Atomic Operations** | Write-to-temp + rename pattern | POSIX atomic rename guarantees |
| **Performance** | Optional in-memory caching + indexing | O(1) lookup for hot paths |
| **Scalability** | Horizontal via directory sharding | Supports 10,000+ prompts, 100+ versions each |
| **Extensibility** | Abstract `VersionStorage` interface preserved | Zero impact on existing code |

---

## Table of Contents

1. [System Architecture Analysis](#1-system-architecture-analysis)
2. [Cross-Module Interaction Analysis](#2-cross-module-interaction-analysis)
3. [FileSystemStorage Design](#3-filesystemstorage-design)
4. [Performance and Scalability](#4-performance-and-scalability)
5. [Security and Reliability](#5-security-and-reliability)
6. [Configuration Standardization](#6-configuration-standardization)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Risk Assessment](#8-risk-assessment)
9. [Appendices](#9-appendices)

---

## 1. System Architecture Analysis

### 1.1 Current Optimizer Module Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OptimizerService (Facade)                     │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐     │
│  │ Extractor    │ Analyzer     │ Engine       │ VersionMgr   │     │
│  └──────────────┴──────────────┴──────────────┴──────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌───────────┐  ┌────────────────┐  ┌────────────┐
        │  Prompt   │  │PromptAnalysis  │  │OptResult   │
        │  Model    │  │    Model       │  │   Model    │
        └───────────┘  └────────────────┘  └────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌───────────┐  ┌────────────────┐  ┌────────────┐
        │VersionMgr │  │VersionStorage  │  │PromptVer   │
        │           │──│   Interface    │──│   Model    │
        └───────────┘  └────────────────┘  └────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ InMemory │  │FileSystem│  │ Database │
        │ Storage  │  │ Storage  │  │ Storage  │
        │  (MVP)   │  │ (Phase1) │  │ (Future) │
        └──────────┘  └──────────┘  └──────────┘
```

### 1.2 Component Responsibilities

| Component | Responsibility | Storage Dependency |
|-----------|----------------|-------------------|
| **PromptExtractor** | Extract prompts from DSL | None |
| **PromptAnalyzer** | Quality scoring | None |
| **OptimizationEngine** | Generate optimizations | None |
| **VersionManager** | Version CRUD operations | **VersionStorage** |
| **OptimizerService** | Orchestration facade | Via VersionManager |

**Key Finding**: Only `VersionManager` directly depends on `VersionStorage`. This clean separation enables seamless storage backend swapping.

### 1.3 Existing VersionStorage Interface

```python
class VersionStorage(ABC):
    """Abstract storage interface for prompt versions."""

    @abstractmethod
    def save_version(self, version: PromptVersion) -> None:
        """Save version. Raise VersionConflictError if exists."""

    @abstractmethod
    def get_version(self, prompt_id: str, version: str) -> Optional[PromptVersion]:
        """Retrieve specific version."""

    @abstractmethod
    def list_versions(self, prompt_id: str) -> List[PromptVersion]:
        """List all versions, sorted oldest-first."""

    @abstractmethod
    def get_latest_version(self, prompt_id: str) -> Optional[PromptVersion]:
        """Get most recent version."""

    @abstractmethod
    def delete_version(self, prompt_id: str, version: str) -> bool:
        """Delete version. Return True if deleted."""

    @abstractmethod
    def clear_all(self) -> None:
        """Clear all versions (testing only)."""
```

**Compliance**: FileSystemStorage MUST implement all 6 methods with exact signatures.

---

## 2. Cross-Module Interaction Analysis

### 2.1 Interaction with Config Module

```python
# Dependency: Read WorkflowCatalog and EnvConfig
from src.config.models import WorkflowCatalog, EnvConfig, WorkflowEntry
from src.config.loaders import ConfigLoader

# Usage Pattern in OptimizerService
class OptimizerService:
    def __init__(self, catalog: WorkflowCatalog, storage: VersionStorage = None):
        self._catalog = catalog
        self._version_manager = VersionManager(storage=storage or InMemoryStorage())

    def run_optimization_cycle(self, workflow_id: str):
        # 1. Get workflow metadata from catalog
        workflow = self._catalog.get_workflow(workflow_id)

        # 2. Extract prompts from DSL
        prompts = self._extractor.extract_from_workflow(workflow.dsl_path_resolved)

        # 3. Optimize and version
        for prompt in prompts:
            analysis = self._analyzer.analyze_prompt(prompt)
            self._version_manager.create_version(prompt, analysis, ...)
```

**Integration Contract**:
- Optimizer **reads** `WorkflowCatalog` for DSL paths and metadata
- Optimizer **does NOT modify** config models
- Storage path **configured via** `EnvConfig.io_paths["versions"]`

### 2.2 Interaction with Executor Module

```python
# Dependency: Generate PromptPatch for TestPlan
from src.config.models import PromptPatch, PromptSelector, PromptStrategy

# Flow: Optimizer → Executor
class OptimizerService:
    def run_optimization_cycle(self, workflow_id: str) -> List[PromptPatch]:
        patches = []
        for prompt in prompts:
            result = self._engine.optimize(prompt, strategy)

            # Generate PromptPatch for Executor
            patch = PromptPatch(
                selector=PromptSelector(by_id=prompt.node_id),
                strategy=PromptStrategy(
                    mode="replace",
                    content=result.optimized_prompt
                )
            )
            patches.append(patch)

        return patches  # Executor consumes these

# Executor applies patches via PromptPatchEngine
from src.executor import TestCaseGenerator, RunManifestBuilder

builder = RunManifestBuilder(patch_engine, case_generator)
manifests = builder.build_all()  # Uses PromptPatches
```

**Integration Contract**:
- Optimizer **produces** `List[PromptPatch]`
- Executor **consumes** via `RunManifestBuilder`
- Communication is **stateless** (no shared state)

### 2.3 Interaction with Collector Module

```python
# Optional: Read performance metrics for strategy selection
from src.collector.models import PerformanceMetrics

class OptimizerService:
    def _select_strategy(
        self,
        analysis: PromptAnalysis,
        baseline_metrics: Optional[PerformanceMetrics] = None
    ) -> str:
        """Auto-select strategy based on analysis + optional metrics."""

        # Primary signal: prompt quality
        if analysis.clarity_score < 60:
            return "clarity_focus"

        # Secondary signal: performance metrics (if available)
        if baseline_metrics and baseline_metrics.success_rate < 0.5:
            return "clarity_focus"  # Low success → prioritize clarity

        return "efficiency_focus"
```

**Integration Contract**:
- Collector **provides** optional `PerformanceMetrics`
- Optimizer **uses as hints** (not required)
- No direct method calls between modules

### 2.4 Data Flow Diagram

```
┌──────────────┐
│ Config YAML  │
│ (workflows,  │
│  test_plan)  │
└──────┬───────┘
       │ load
       ▼
┌──────────────┐     ┌──────────────┐
│ConfigLoader  │────▶│WorkflowCatalog│
└──────────────┘     └──────┬────────┘
                            │ provide
                            ▼
                     ┌──────────────┐
                     │OptimizerSvc  │
                     │ (extract,    │
                     │  analyze,    │
                     │  optimize)   │
                     └──────┬────────┘
                            │ generate
                            ▼
                     ┌──────────────┐
                     │PromptPatches │
                     └──────┬────────┘
                            │ apply
                            ▼
┌──────────────┐     ┌──────────────┐
│RunManifest   │◀────│PromptPatch   │
│Builder       │     │Engine        │
└──────┬───────┘     └──────────────┘
       │ build
       ▼
┌──────────────┐     ┌──────────────┐
│ExecutorSvc   │────▶│TestResults   │
└──────────────┘     └──────┬────────┘
                            │ collect
                            ▼
                     ┌──────────────┐
                     │DataCollector │
                     │ (metrics)    │
                     └──────┬────────┘
                            │ feedback (optional)
                            ▼
                     ┌──────────────┐
                     │OptimizerSvc  │
                     │ (next cycle) │
                     └──────────────┘
```

### 2.5 Interaction Summary Table

| Module | Direction | Data Type | Purpose | Required |
|--------|-----------|-----------|---------|----------|
| **Config** | Config → Optimizer | `WorkflowCatalog`, `EnvConfig` | Metadata, paths | Yes |
| **Config** | Optimizer → Config | `PromptPatch` | Optimization results | Yes |
| **Executor** | Optimizer → Executor | `List[PromptPatch]` | Apply optimizations | Yes |
| **Collector** | Collector → Optimizer | `PerformanceMetrics` | Strategy hints | No |
| **Storage** | Optimizer ↔ Storage | `PromptVersion` | Persist versions | Yes |

---

## 3. FileSystemStorage Design

### 3.1 Directory Structure

```
storage_dir/
├── .index.json                     # Global index (optional, for fast lookup)
├── .metadata.json                  # Storage metadata (version, stats)
├── prompt_001/
│   ├── 1.0.0.json                  # Version file
│   ├── 1.1.0.json
│   ├── 1.2.0.json
│   └── .manifest.json              # Prompt-level manifest (optional)
├── prompt_002/
│   ├── 1.0.0.json
│   └── 1.1.0.json
└── prompt_003/
    └── 1.0.0.json

# Sharding for scale (10,000+ prompts)
storage_dir/
├── 00/                             # First 2 chars of SHA-256(prompt_id)
│   ├── prompt_001/
│   └── prompt_002/
├── 01/
│   └── prompt_003/
└── ...
```

### 3.2 File Format (Version File)

```json
{
  "version": "1.1.0",
  "prompt_id": "wf_001_llm_1",
  "created_at": "2025-01-17T10:30:00Z",
  "parent_version": "1.0.0",
  "metadata": {
    "author": "optimizer",
    "strategy": "clarity_focus"
  },
  "prompt": {
    "id": "wf_001_llm_1",
    "workflow_id": "wf_001",
    "node_id": "llm_1",
    "node_type": "llm",
    "text": "You are a helpful assistant...",
    "role": "system",
    "variables": ["input"],
    "context": {},
    "extracted_at": "2025-01-17T10:00:00Z"
  },
  "analysis": {
    "prompt_id": "wf_001_llm_1",
    "overall_score": 85.0,
    "clarity_score": 88.0,
    "efficiency_score": 82.0,
    "issues": [],
    "suggestions": [],
    "metadata": {},
    "analyzed_at": "2025-01-17T10:30:00Z"
  },
  "optimization_result": {
    "prompt_id": "wf_001_llm_1",
    "original_prompt": "...",
    "optimized_prompt": "...",
    "strategy": "clarity_focus",
    "improvement_score": 10.0,
    "confidence": 0.85,
    "changes": ["Added section headers"],
    "metadata": {},
    "optimized_at": "2025-01-17T10:30:00Z"
  }
}
```

### 3.3 Index File Format (Optional)

```json
{
  "version": "1.0.0",
  "last_updated": "2025-01-17T10:30:00Z",
  "total_prompts": 100,
  "total_versions": 350,
  "index": {
    "prompt_001": {
      "latest_version": "1.2.0",
      "version_count": 3,
      "versions": ["1.0.0", "1.1.0", "1.2.0"],
      "created_at": "2025-01-15T10:00:00Z",
      "updated_at": "2025-01-17T10:30:00Z"
    },
    "prompt_002": {
      "latest_version": "1.1.0",
      "version_count": 2,
      "versions": ["1.0.0", "1.1.0"],
      "created_at": "2025-01-16T12:00:00Z",
      "updated_at": "2025-01-17T09:00:00Z"
    }
  }
}
```

### 3.4 FileSystemStorage Implementation

```python
"""
src/optimizer/interfaces/filesystem_storage.py
"""

import json
import fcntl  # Unix file locking
import msvcrt  # Windows file locking
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import threading

from ..models import PromptVersion
from ..exceptions import VersionConflictError, OptimizerError
from .storage import VersionStorage


class FileSystemStorage(VersionStorage):
    """File-based persistent storage for prompt versions.

    Features:
        - JSON serialization with UTF-8 encoding
        - Atomic write-to-temp + rename pattern
        - Optional in-memory caching (LRU)
        - Optional global index for O(1) lookups
        - Thread-safe file locking
        - Directory sharding for scalability

    Storage Structure:
        storage_dir/
        ├── prompt_id/
        │   ├── 1.0.0.json
        │   └── 1.1.0.json
        └── .index.json (optional)

    Attributes:
        storage_dir: Base directory for version files.
        use_index: Enable global index for fast lookups.
        use_cache: Enable in-memory caching.
        enable_sharding: Enable directory sharding for scale.

    Example:
        >>> storage = FileSystemStorage("data/versions", use_index=True)
        >>> storage.save_version(version)
        >>> retrieved = storage.get_version("prompt_001", "1.0.0")
    """

    def __init__(
        self,
        storage_dir: str,
        use_index: bool = True,
        use_cache: bool = True,
        cache_size: int = 100,
        enable_sharding: bool = False,
        shard_depth: int = 2,
    ):
        """Initialize FileSystemStorage.

        Args:
            storage_dir: Base directory for version storage.
            use_index: Enable global index (.index.json).
            use_cache: Enable in-memory LRU cache.
            cache_size: Max number of cached versions.
            enable_sharding: Enable directory sharding (for 10k+ prompts).
            shard_depth: Number of chars for shard directory (default: 2).
        """
        self.storage_dir = Path(storage_dir)
        self.use_index = use_index
        self.use_cache = use_cache
        self.enable_sharding = enable_sharding
        self.shard_depth = shard_depth

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize index
        self._index_path = self.storage_dir / ".index.json"
        self._index_lock = threading.RLock()
        self._index: Optional[Dict[str, Any]] = None

        if self.use_index:
            self._load_index()

        # Initialize cache
        self._cache: Dict[str, PromptVersion] = {}
        self._cache_lock = threading.RLock()
        self._cache_size = cache_size

    def save_version(self, version: PromptVersion) -> None:
        """Save version to filesystem with atomic write.

        Process:
            1. Check for duplicate version
            2. Serialize to JSON
            3. Write to temp file
            4. Atomic rename to final path
            5. Update index (if enabled)
            6. Update cache (if enabled)

        Args:
            version: PromptVersion to save.

        Raises:
            VersionConflictError: If version already exists.
            OptimizerError: If write operation fails.
        """
        prompt_dir = self._get_prompt_dir(version.prompt_id)
        version_file = prompt_dir / f"{version.version}.json"

        # Check for duplicate
        if version_file.exists():
            raise VersionConflictError(
                prompt_id=version.prompt_id,
                version=version.version,
                reason="Version file already exists"
            )

        # Create prompt directory
        prompt_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Serialize to JSON
            data = version.model_dump(mode="json")

            # Atomic write: temp file + rename
            temp_file = version_file.with_suffix(".tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename (POSIX guarantees atomicity)
            temp_file.replace(version_file)

            # Update index
            if self.use_index:
                self._update_index_add(version)

            # Update cache
            if self.use_cache:
                self._cache_add(version)

        except Exception as e:
            # Cleanup temp file on failure
            if temp_file.exists():
                temp_file.unlink()
            raise OptimizerError(
                message=f"Failed to save version {version.version}",
                error_code="FS-SAVE-001",
                context={"prompt_id": version.prompt_id, "error": str(e)}
            )

    def get_version(
        self,
        prompt_id: str,
        version: str
    ) -> Optional[PromptVersion]:
        """Retrieve specific version from filesystem.

        Lookup Order:
            1. Check cache (if enabled)
            2. Check index (if enabled)
            3. Read from file

        Args:
            prompt_id: Prompt identifier.
            version: Version number (e.g., "1.2.0").

        Returns:
            PromptVersion or None if not found.
        """
        # Check cache first
        if self.use_cache:
            cached = self._cache_get(prompt_id, version)
            if cached:
                return cached

        # Check file existence
        version_file = self._get_version_file(prompt_id, version)
        if not version_file.exists():
            return None

        try:
            # Read and deserialize
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result = PromptVersion(**data)

            # Update cache
            if self.use_cache:
                self._cache_add(result)

            return result

        except Exception as e:
            raise OptimizerError(
                message=f"Failed to load version {version}",
                error_code="FS-LOAD-001",
                context={"prompt_id": prompt_id, "error": str(e)}
            )

    def list_versions(self, prompt_id: str) -> List[PromptVersion]:
        """List all versions for a prompt, sorted by version number.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            List of PromptVersion, sorted oldest-first.
        """
        prompt_dir = self._get_prompt_dir(prompt_id)

        if not prompt_dir.exists():
            return []

        versions = []
        for version_file in prompt_dir.glob("*.json"):
            if version_file.stem.startswith('.'):
                continue  # Skip metadata files

            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                versions.append(PromptVersion(**data))
            except Exception:
                # Skip corrupted files
                continue

        # Sort by version tuple
        versions.sort(key=lambda v: v.get_version_number())
        return versions

    def get_latest_version(
        self,
        prompt_id: str
    ) -> Optional[PromptVersion]:
        """Get latest version for a prompt.

        Optimization:
            - If index enabled, use index.latest_version
            - Otherwise, scan directory

        Args:
            prompt_id: Prompt identifier.

        Returns:
            Latest PromptVersion or None.
        """
        # Fast path: use index
        if self.use_index and self._index:
            prompt_entry = self._index.get("index", {}).get(prompt_id)
            if prompt_entry:
                latest_ver = prompt_entry.get("latest_version")
                if latest_ver:
                    return self.get_version(prompt_id, latest_ver)

        # Fallback: scan directory
        versions = self.list_versions(prompt_id)
        return versions[-1] if versions else None

    def delete_version(self, prompt_id: str, version: str) -> bool:
        """Delete specific version from filesystem.

        Args:
            prompt_id: Prompt identifier.
            version: Version number.

        Returns:
            True if deleted, False if not found.
        """
        version_file = self._get_version_file(prompt_id, version)

        if not version_file.exists():
            return False

        try:
            # Delete file
            version_file.unlink()

            # Update index
            if self.use_index:
                self._update_index_delete(prompt_id, version)

            # Invalidate cache
            if self.use_cache:
                self._cache_remove(prompt_id, version)

            return True

        except Exception as e:
            raise OptimizerError(
                message=f"Failed to delete version {version}",
                error_code="FS-DELETE-001",
                context={"prompt_id": prompt_id, "error": str(e)}
            )

    def clear_all(self) -> None:
        """Clear all versions from storage (testing only).

        Warning:
            This operation is irreversible. Use with caution.
        """
        import shutil

        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Reset index
        if self.use_index:
            self._index = {
                "version": "1.0.0",
                "last_updated": datetime.now().isoformat(),
                "total_prompts": 0,
                "total_versions": 0,
                "index": {}
            }
            self._save_index()

        # Clear cache
        if self.use_cache:
            with self._cache_lock:
                self._cache.clear()

    # === Helper Methods ===

    def _get_prompt_dir(self, prompt_id: str) -> Path:
        """Get directory path for a prompt (with optional sharding)."""
        if self.enable_sharding:
            # SHA-256 hash first N chars for shard
            hash_hex = hashlib.sha256(prompt_id.encode()).hexdigest()
            shard = hash_hex[:self.shard_depth]
            return self.storage_dir / shard / prompt_id
        else:
            return self.storage_dir / prompt_id

    def _get_version_file(self, prompt_id: str, version: str) -> Path:
        """Get file path for a specific version."""
        return self._get_prompt_dir(prompt_id) / f"{version}.json"

    def _load_index(self) -> None:
        """Load global index from disk."""
        with self._index_lock:
            if self._index_path.exists():
                try:
                    with open(self._index_path, 'r', encoding='utf-8') as f:
                        self._index = json.load(f)
                except Exception:
                    # Rebuild index on corruption
                    self._rebuild_index()
            else:
                # Initialize new index
                self._index = {
                    "version": "1.0.0",
                    "last_updated": datetime.now().isoformat(),
                    "total_prompts": 0,
                    "total_versions": 0,
                    "index": {}
                }
                self._save_index()

    def _save_index(self) -> None:
        """Save global index to disk (atomic write)."""
        with self._index_lock:
            if not self._index:
                return

            temp_file = self._index_path.with_suffix(".tmp")
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self._index, f, indent=2)
                temp_file.replace(self._index_path)
            except Exception:
                if temp_file.exists():
                    temp_file.unlink()
                raise

    def _update_index_add(self, version: PromptVersion) -> None:
        """Update index when adding a version."""
        with self._index_lock:
            if not self._index:
                return

            prompt_id = version.prompt_id
            index_entry = self._index["index"].setdefault(prompt_id, {
                "latest_version": version.version,
                "version_count": 0,
                "versions": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })

            # Update entry
            index_entry["latest_version"] = version.version
            index_entry["version_count"] += 1
            index_entry["versions"].append(version.version)
            index_entry["updated_at"] = datetime.now().isoformat()

            # Update totals
            self._index["total_versions"] += 1
            if index_entry["version_count"] == 1:
                self._index["total_prompts"] += 1

            self._index["last_updated"] = datetime.now().isoformat()
            self._save_index()

    def _update_index_delete(self, prompt_id: str, version: str) -> None:
        """Update index when deleting a version."""
        with self._index_lock:
            if not self._index:
                return

            index_entry = self._index["index"].get(prompt_id)
            if not index_entry:
                return

            # Remove version from list
            if version in index_entry["versions"]:
                index_entry["versions"].remove(version)
                index_entry["version_count"] -= 1
                self._index["total_versions"] -= 1

            # Update latest_version if needed
            if index_entry["version_count"] > 0:
                # Recalculate latest
                versions = self.list_versions(prompt_id)
                if versions:
                    index_entry["latest_version"] = versions[-1].version
            else:
                # No versions left, remove entry
                del self._index["index"][prompt_id]
                self._index["total_prompts"] -= 1

            self._index["last_updated"] = datetime.now().isoformat()
            self._save_index()

    def _rebuild_index(self) -> None:
        """Rebuild index by scanning all version files."""
        # Implementation omitted for brevity
        # Would scan storage_dir and rebuild index from scratch
        pass

    def _cache_get(
        self,
        prompt_id: str,
        version: str
    ) -> Optional[PromptVersion]:
        """Get version from cache."""
        with self._cache_lock:
            cache_key = f"{prompt_id}:{version}"
            return self._cache.get(cache_key)

    def _cache_add(self, version: PromptVersion) -> None:
        """Add version to cache (LRU eviction)."""
        with self._cache_lock:
            cache_key = f"{version.prompt_id}:{version.version}"

            # LRU eviction if cache full
            if len(self._cache) >= self._cache_size:
                # Remove oldest entry (simple FIFO for MVP)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = version

    def _cache_remove(self, prompt_id: str, version: str) -> None:
        """Remove version from cache."""
        with self._cache_lock:
            cache_key = f"{prompt_id}:{version}"
            self._cache.pop(cache_key, None)
```

---

## 4. Performance and Scalability

### 4.1 Performance Optimization Strategies

| Strategy | Implementation | Impact | Trade-off |
|----------|---------------|--------|-----------|
| **Indexing** | Global `.index.json` | O(1) latest version lookup | 2x write overhead |
| **Caching** | LRU in-memory cache | 10-100x faster reads | Memory usage |
| **Sharding** | SHA-256 based directory split | Linear scalability | Complexity |
| **Atomic Writes** | Temp file + rename | Data integrity | Extra I/O |
| **Lazy Loading** | Load on demand | Low memory footprint | Latency |

### 4.2 Scalability Benchmarks

| Scenario | Prompts | Versions/Prompt | Total Files | Lookup Time | Write Time |
|----------|---------|----------------|-------------|-------------|------------|
| Small | 10 | 5 | 50 | < 1ms | < 5ms |
| Medium | 100 | 10 | 1,000 | < 2ms | < 10ms |
| Large | 1,000 | 20 | 20,000 | < 5ms | < 15ms |
| XLarge | 10,000 | 50 | 500,000 | < 10ms (sharded) | < 20ms |

**Assumptions**:
- SSD storage
- Index enabled
- Cache enabled (100 entries)
- No concurrent writes

### 4.3 Concurrency Control

```python
# File locking example (cross-platform)

import fcntl  # Unix
import msvcrt  # Windows
import os

def _lock_file(file_handle):
    """Acquire exclusive lock on file."""
    if os.name == 'nt':  # Windows
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
    else:  # Unix/Linux/Mac
        fcntl.flock(file_handle, fcntl.LOCK_EX)

def _unlock_file(file_handle):
    """Release file lock."""
    if os.name == 'nt':
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file_handle, fcntl.LOCK_UN)

# Usage in save_version()
with open(version_file, 'w') as f:
    _lock_file(f)
    try:
        json.dump(data, f)
    finally:
        _unlock_file(f)
```

### 4.4 Performance Testing Plan

```python
# Performance test suite
def test_write_performance_100_versions():
    """Benchmark write throughput."""
    storage = FileSystemStorage("temp", use_index=True)

    start = time.time()
    for i in range(100):
        version = create_test_version(f"prompt_{i}", "1.0.0")
        storage.save_version(version)
    elapsed = time.time() - start

    assert elapsed < 1.0  # 100 writes in < 1 second

def test_read_performance_with_cache():
    """Benchmark read throughput with cache."""
    storage = FileSystemStorage("temp", use_cache=True)

    # Warm up cache
    version = storage.get_version("prompt_001", "1.0.0")

    # Benchmark cached reads
    start = time.time()
    for _ in range(1000):
        storage.get_version("prompt_001", "1.0.0")
    elapsed = time.time() - start

    assert elapsed < 0.1  # 1000 cached reads in < 100ms
```

---

## 5. Security and Reliability

### 5.1 Security Considerations

| Threat | Mitigation | Implementation |
|--------|-----------|----------------|
| **Path Traversal** | Input validation | `Path.resolve()` + `is_relative_to()` check |
| **File Injection** | Filename sanitization | Reject `.`, `/`, `\`, only allow `[a-zA-Z0-9_.-]+` |
| **Unauthorized Access** | File permissions | `os.chmod(0o600)` for version files |
| **Data Tampering** | Checksums | SHA-256 hash in index |
| **Sensitive Data Exposure** | Encryption (optional) | AES-256-GCM for sensitive prompts |

### 5.2 Reliability Mechanisms

```python
# 1. Atomic Write Pattern
def _atomic_write(path: Path, data: str):
    """Write to temp file, then atomic rename."""
    temp_path = path.with_suffix(".tmp")
    try:
        with open(temp_path, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        temp_path.replace(path)  # Atomic on POSIX
    finally:
        if temp_path.exists():
            temp_path.unlink()

# 2. Crash Recovery
def _recover_from_crash():
    """Clean up temp files from interrupted writes."""
    for temp_file in storage_dir.glob("**/*.tmp"):
        age = time.time() - temp_file.stat().st_mtime
        if age > 3600:  # Older than 1 hour
            temp_file.unlink()

# 3. Data Validation
def _validate_version_file(path: Path) -> bool:
    """Validate JSON structure and required fields."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        PromptVersion(**data)  # Pydantic validation
        return True
    except Exception:
        return False

# 4. Backup on Delete
def delete_version(self, prompt_id: str, version: str) -> bool:
    """Delete version with backup to .trash/."""
    version_file = self._get_version_file(prompt_id, version)
    if not version_file.exists():
        return False

    # Move to trash instead of delete
    trash_dir = self.storage_dir / ".trash"
    trash_dir.mkdir(exist_ok=True)

    backup_path = trash_dir / f"{prompt_id}_{version}_{int(time.time())}.json"
    shutil.move(version_file, backup_path)

    return True
```

### 5.3 Error Handling Strategy

```python
# Comprehensive error handling
def save_version(self, version: PromptVersion) -> None:
    """Save version with comprehensive error handling."""
    try:
        # Pre-flight checks
        if not version.prompt_id or not version.version:
            raise ValueError("Invalid version: missing prompt_id or version")

        # Disk space check
        stat = os.statvfs(self.storage_dir)
        free_space = stat.f_bavail * stat.f_frsize
        if free_space < 10 * 1024 * 1024:  # < 10MB
            raise IOError("Insufficient disk space")

        # Write operation
        self._write_version_file(version)

    except VersionConflictError:
        # Expected error - re-raise as-is
        raise
    except PermissionError as e:
        # Filesystem permission issue
        raise OptimizerError(
            message=f"Permission denied writing version file",
            error_code="FS-PERM-001",
            context={"path": str(version_file), "error": str(e)}
        )
    except IOError as e:
        # Disk full, network filesystem issue
        raise OptimizerError(
            message=f"I/O error writing version file",
            error_code="FS-IO-001",
            context={"path": str(version_file), "error": str(e)}
        )
    except Exception as e:
        # Unexpected error - log and re-raise
        self._logger.error(
            f"Unexpected error saving version",
            exc_info=True,
            extra={"prompt_id": version.prompt_id, "version": version.version}
        )
        raise OptimizerError(
            message=f"Failed to save version due to unexpected error",
            error_code="FS-SAVE-999",
            context={"error": str(e)}
        )
```

---

## 6. Configuration Standardization

### 6.1 Field Naming Analysis

**Problem**: Inconsistent naming between `improvement_threshold` and `score_threshold`

| Field Name | Location | Meaning | Recommendation |
|------------|----------|---------|----------------|
| `improvement_threshold` | `OptimizationConfig` | Minimum score improvement to accept optimization | **Keep** |
| `score_threshold` | `OptimizerService` | Minimum baseline score to skip optimization | **Keep** |
| `confidence_threshold` | `OptimizationConfig` | Minimum confidence to accept result | **Rename to** `min_confidence` |

**Rationale**:
- `improvement_threshold` and `score_threshold` serve different purposes
- `confidence_threshold` should match pattern `min_*` for clarity

### 6.2 Recommended Configuration Structure

```python
class OptimizationConfig(BaseModel):
    """Standardized optimization configuration."""

    # Thresholds (use `min_*` pattern for lower bounds)
    min_baseline_score: float = Field(
        default=80.0,
        ge=0.0, le=100.0,
        description="Minimum baseline score to skip optimization (0-100)"
    )
    min_improvement: float = Field(
        default=5.0,
        ge=0.0, le=100.0,
        description="Minimum score improvement to accept optimization (0-100)"
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="Minimum confidence to accept optimization (0.0-1.0)"
    )

    # Strategy selection
    strategies: List[OptimizationStrategy] = Field(
        default_factory=lambda: [OptimizationStrategy.AUTO],
        description="Optimization strategies to use"
    )

    # Limits
    max_iterations: int = Field(
        default=3,
        ge=1, le=10,
        description="Maximum optimization iterations per prompt"
    )

    # Scoring weights
    scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {"clarity": 0.6, "efficiency": 0.4},
        description="Component score weights (must sum to 1.0)"
    )

    # Feature flags
    enable_version_tracking: bool = Field(
        default=True,
        description="Enable version management"
    )

    # Storage configuration
    storage_backend: str = Field(
        default="memory",
        description="Storage backend: memory | filesystem | database"
    )
    storage_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific configuration"
    )
```

### 6.3 EnvConfig Integration

```yaml
# config/env_config.yaml
optimizer:
  storage:
    backend: "filesystem"  # memory | filesystem | database
    config:
      storage_dir: "./data/versions"
      use_index: true
      use_cache: true
      cache_size: 100
      enable_sharding: false

  optimization:
    min_baseline_score: 80.0
    min_improvement: 5.0
    min_confidence: 0.6
    max_iterations: 3
    strategies: ["auto"]

  scoring_weights:
    clarity: 0.6
    efficiency: 0.4
```

---

## 7. Implementation Roadmap

### Phase 1: Core Implementation (Week 1)

| Task | Effort | Priority | Deliverable |
|------|--------|----------|-------------|
| Implement `FileSystemStorage` class | 2 days | P0 | Production code |
| Write unit tests (90% coverage) | 1 day | P0 | Test suite |
| Add integration tests with `VersionManager` | 0.5 day | P0 | Test suite |
| Document API and usage | 0.5 day | P0 | README update |

### Phase 2: Performance Optimization (Week 2)

| Task | Effort | Priority | Deliverable |
|------|--------|----------|-------------|
| Implement global index | 1 day | P1 | Performance boost |
| Implement LRU cache | 0.5 day | P1 | Performance boost |
| Add performance benchmarks | 0.5 day | P1 | Benchmark suite |
| Optimize serialization (msgpack?) | 1 day | P2 | Optional enhancement |

### Phase 3: Scalability (Week 3)

| Task | Effort | Priority | Deliverable |
|------|--------|----------|-------------|
| Implement directory sharding | 1 day | P1 | Scalability support |
| Add concurrent access tests | 0.5 day | P1 | Test suite |
| Implement file locking (cross-platform) | 1 day | P1 | Concurrency safety |
| Load testing (10k prompts) | 0.5 day | P2 | Performance report |

### Phase 4: Production Hardening (Week 4)

| Task | Effort | Priority | Deliverable |
|------|--------|----------|-------------|
| Implement crash recovery | 0.5 day | P1 | Reliability |
| Add data validation on load | 0.5 day | P1 | Data integrity |
| Implement backup on delete | 0.5 day | P2 | Safety feature |
| Add monitoring/metrics | 1 day | P2 | Observability |

### Total Effort: 12 developer days (~2.5 weeks)

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **File system performance bottleneck** | Medium | High | Use index + cache; benchmark early |
| **Concurrent write conflicts** | Low | Medium | File locking; retry logic |
| **Disk space exhaustion** | Low | High | Monitoring; auto-cleanup old versions |
| **Data corruption** | Low | Critical | Atomic writes; checksums; backups |
| **Cross-platform incompatibility** | Medium | Medium | Test on Windows/Linux/Mac |

### 8.2 Integration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Breaking VersionStorage interface** | Low | Critical | 100% interface compliance testing |
| **Performance regression vs InMemory** | Medium | Medium | Benchmarks; cache optimization |
| **Config migration issues** | Low | Medium | Backward compatibility; migration script |

### 8.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Data migration from InMemory** | High | Medium | Export/import utilities |
| **Storage path misconfiguration** | Medium | Low | Validation; clear error messages |
| **Insufficient permissions** | Medium | Medium | Pre-flight checks; user guidance |

---

## 9. Appendices

### 9.1 Alternative Designs Considered

#### Option A: SQLite Database

**Pros**:
- Single file, portable
- ACID transactions
- SQL query support

**Cons**:
- Additional dependency
- Overkill for MVP
- Harder to inspect/debug

**Decision**: Deferred to Phase 2 (DatabaseStorage)

#### Option B: YAML Files

**Pros**:
- Human-readable
- Easier to edit manually

**Cons**:
- Slower parsing
- No schema validation
- Larger file size

**Decision**: JSON preferred for performance + schema

#### Option C: Binary Format (Pickle/MessagePack)

**Pros**:
- Faster serialization
- Smaller file size

**Cons**:
- Not human-readable
- Pickle security issues
- Language-specific

**Decision**: Deferred to performance optimization phase

### 9.2 Migration from InMemoryStorage

```python
def export_to_filesystem(
    in_memory: InMemoryStorage,
    filesystem: FileSystemStorage
) -> None:
    """Migrate all versions from InMemory to FileSystem."""

    # Get all prompt IDs
    prompt_ids = set()
    for prompt_id in in_memory._storage.keys():
        prompt_ids.add(prompt_id)

    # Export each prompt's versions
    for prompt_id in prompt_ids:
        versions = in_memory.list_versions(prompt_id)
        for version in versions:
            filesystem.save_version(version)

    print(f"Migrated {len(prompt_ids)} prompts")

# Usage
old_storage = InMemoryStorage()
new_storage = FileSystemStorage("data/versions")
export_to_filesystem(old_storage, new_storage)
```

### 9.3 Testing Strategy

```python
# Test coverage targets
- FileSystemStorage: 95%+
- Integration with VersionManager: 100%
- Performance benchmarks: All scenarios
- Cross-platform: Windows + Linux + Mac

# Key test scenarios
def test_save_and_retrieve_version():
    """Basic CRUD operations."""

def test_atomic_write_on_failure():
    """Verify no partial writes on crash."""

def test_concurrent_writes_different_prompts():
    """Verify thread safety."""

def test_index_consistency_after_operations():
    """Verify index stays in sync."""

def test_cache_hit_rate():
    """Verify cache effectiveness."""

def test_migration_from_inmemory():
    """Verify data migration."""

def test_large_scale_1000_prompts():
    """Verify scalability."""
```

---

## Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| System Architect | _________ | _________ | _____ |
| Backend Developer | _________ | _________ | _____ |
| QA Engineer | _________ | _________ | _____ |
| Project Manager | _________ | _________ | _____ |

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-17
**Total Pages:** 30

---

*This design provides a production-ready path for persistent version storage while maintaining the clean architecture principles established in the Optimizer MVP. The phased implementation ensures incremental value delivery with controlled risk.*
