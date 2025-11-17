"""
Optimizer Module - FileSystem Version Storage Implementation

Date: 2025-11-17
Author: backend-developer
Description: Production-grade file-based persistent storage for prompt versions.
"""

import hashlib
import json
import os
import platform
import shutil
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..exceptions import OptimizerError, VersionConflictError
from ..models import PromptVersion
from .storage import VersionStorage


# ============================================================================
# LRU Cache Implementation
# ============================================================================


class LRUCache:
    """Thread-safe LRU cache for PromptVersion objects.

    Features:
        - O(1) get and put operations
        - Thread-safe with RLock
        - Automatic eviction when full
        - Cache statistics tracking

    Example:
        >>> cache = LRUCache(max_size=100)
        >>> cache.put("key", version_obj)
        >>> cached = cache.get("key")
        >>> stats = cache.stats()
    """

    def __init__(self, max_size: int = 100):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries to cache.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, PromptVersion] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[PromptVersion]:
        """Get item from cache (move to end if hit).

        Args:
            key: Cache key.

        Returns:
            Cached PromptVersion or None if not found.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def put(self, key: str, value: PromptVersion) -> None:
        """Add item to cache (evict oldest if full).

        Args:
            key: Cache key.
            value: PromptVersion to cache.
        """
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
        """Remove item from cache.

        Args:
            key: Cache key to remove.
        """
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, hits, misses, hit_rate).
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# ============================================================================
# File Lock Implementation (Cross-Platform)
# ============================================================================


class FileLock:
    """Cross-platform file locking context manager.

    Supports:
        - Windows: msvcrt.locking()
        - Unix/Linux/Mac: fcntl.flock()

    Example:
        >>> lock_file = Path("/tmp/myfile.lock")
        >>> with FileLock(lock_file):
        ...     # Critical section protected by lock
        ...     perform_operation()
    """

    def __init__(self, lock_path: Path, timeout: float = 10.0):
        """Initialize file lock.

        Args:
            lock_path: Path to lock file.
            timeout: Maximum seconds to wait for lock acquisition.
        """
        self.lock_path = lock_path
        self.timeout = timeout
        self.lock_file: Optional[Any] = None
        self._system = platform.system()

    def __enter__(self):
        """Acquire lock."""
        # Create lock file if it doesn't exist
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        while True:
            try:
                # Open lock file in append mode
                self.lock_file = open(self.lock_path, 'a')

                if self._system == "Windows":
                    # Windows: Use msvcrt.locking()
                    import msvcrt
                    msvcrt.locking(
                        self.lock_file.fileno(),
                        msvcrt.LK_NBLCK,  # Non-blocking lock
                        1
                    )
                else:
                    # Unix: Use fcntl.flock()
                    import fcntl
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Lock acquired successfully
                return self

            except (IOError, OSError) as e:
                # Lock acquisition failed
                if self.lock_file:
                    self.lock_file.close()
                    self.lock_file = None

                # Check timeout
                if time.time() - start_time >= self.timeout:
                    raise OptimizerError(
                        message=f"Failed to acquire lock after {self.timeout}s",
                        error_code="FS-LOCK-001",
                        context={"lock_path": str(self.lock_path)}
                    )

                # Wait a bit before retrying
                time.sleep(0.01)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
        if self.lock_file:
            try:
                if self._system == "Windows":
                    import msvcrt
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass  # Ignore errors during unlock
            finally:
                self.lock_file.close()


# ============================================================================
# FileSystemStorage Implementation
# ============================================================================


class FileSystemStorage(VersionStorage):
    """File-based persistent storage for prompt versions.

    Features:
        - JSON serialization with UTF-8 encoding
        - Atomic write-to-temp + rename pattern
        - Optional in-memory LRU caching
        - Optional global index for O(1) lookups
        - Thread-safe file locking
        - Directory sharding for scalability
        - Crash recovery (cleanup stale temp files)

    Storage Structure:
        storage_dir/
        ├── .index.json                     # Global index (optional)
        ├── prompt_001/
        │   ├── 1.0.0.json                  # Version file
        │   ├── 1.1.0.json
        │   └── 1.2.0.json
        └── prompt_002/
            └── 1.0.0.json

    Sharding (for > 1000 prompts):
        storage_dir/
        ├── 00/
        │   ├── prompt_001/
        │   └── prompt_002/
        └── 01/
            └── prompt_003/

    Attributes:
        storage_dir: Base directory for version files.
        use_index: Enable global index for fast lookups.
        use_cache: Enable in-memory caching.
        enable_sharding: Enable directory sharding for scale.

    Example:
        >>> storage = FileSystemStorage("data/versions", use_index=True)
        >>> storage.save_version(version)
        >>> latest = storage.get_latest_version("prompt_001")
        >>> all_versions = storage.list_versions("prompt_001")
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
        self._cache: Optional[LRUCache] = None
        if self.use_cache:
            self._cache = LRUCache(max_size=cache_size)

        # Perform crash recovery on init
        self._recover_from_crash()

        logger.info(
            "FileSystemStorage initialized",
            storage_dir=str(self.storage_dir),
            use_index=self.use_index,
            use_cache=self.use_cache,
            enable_sharding=self.enable_sharding,
        )

    # ========================================================================
    # VersionStorage Interface Implementation
    # ========================================================================

    def save_version(self, version: PromptVersion) -> None:
        """Save version to filesystem with atomic write.

        Process:
            1. Check for duplicate version (with file lock)
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

        Example:
            >>> storage = FileSystemStorage("data/versions")
            >>> storage.save_version(version)
        """
        prompt_dir = self._get_prompt_dir(version.prompt_id)
        version_file = prompt_dir / f"{version.version}.json"
        lock_file = prompt_dir / ".lock"

        # Acquire lock for this prompt
        with FileLock(lock_file):
            # Check for duplicate version (with lock held)
            if version_file.exists():
                raise VersionConflictError(
                    prompt_id=version.prompt_id,
                    version=version.version,
                    reason="Version file already exists"
                )

            # Create prompt directory
            prompt_dir.mkdir(parents=True, exist_ok=True)

            try:
                start_time = time.time()

                # Serialize to JSON
                data = version.model_dump(mode="json")

                # Atomic write: temp file + rename
                self._atomic_write(version_file, data)

                # Update index
                if self.use_index:
                    self._update_index_add(version)

                # Update cache
                if self._cache:
                    cache_key = f"{version.prompt_id}:{version.version}"
                    self._cache.put(cache_key, version)

                elapsed = time.time() - start_time
                logger.debug(
                    "Version saved",
                    prompt_id=version.prompt_id,
                    version=version.version,
                    elapsed_ms=elapsed * 1000,
                    file_size=version_file.stat().st_size,
                )

            except VersionConflictError:
                # Re-raise version conflicts
                raise
            except Exception as e:
                logger.error(
                    "Failed to save version",
                    prompt_id=version.prompt_id,
                    version=version.version,
                    error=str(e),
                )
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
            2. Read from file
            3. Update cache on cache miss

        Args:
            prompt_id: Prompt identifier.
            version: Version number (e.g., "1.2.0").

        Returns:
            PromptVersion or None if not found.

        Example:
            >>> version = storage.get_version("prompt_001", "1.0.0")
            >>> if version:
            ...     print(version.analysis.overall_score)
        """
        cache_key = f"{prompt_id}:{version}"

        # Check cache first
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug("Cache hit", cache_key=cache_key)
                return cached

        # Check file existence
        version_file = self._get_version_file(prompt_id, version)
        if not version_file.exists():
            logger.debug("Version not found", prompt_id=prompt_id, version=version)
            return None

        try:
            start_time = time.time()

            # Read and deserialize
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result = PromptVersion(**data)

            # Update cache
            if self._cache:
                self._cache.put(cache_key, result)

            elapsed = time.time() - start_time
            logger.debug(
                "Version loaded",
                prompt_id=prompt_id,
                version=version,
                elapsed_ms=elapsed * 1000,
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON in version file",
                file=str(version_file),
                error=str(e),
            )
            raise OptimizerError(
                message=f"Corrupted version file: {version_file}",
                error_code="FS-LOAD-001",
                context={"prompt_id": prompt_id, "version": version, "error": str(e)}
            )
        except Exception as e:
            logger.error(
                "Failed to load version",
                prompt_id=prompt_id,
                version=version,
                error=str(e),
            )
            raise OptimizerError(
                message=f"Failed to load version {version}",
                error_code="FS-LOAD-002",
                context={"prompt_id": prompt_id, "error": str(e)}
            )

    def list_versions(self, prompt_id: str) -> List[PromptVersion]:
        """List all versions for a prompt, sorted by version number.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            List of PromptVersion, sorted oldest-first.

        Example:
            >>> versions = storage.list_versions("prompt_001")
            >>> for v in versions:
            ...     print(f"v{v.version}: score={v.analysis.overall_score}")
        """
        prompt_dir = self._get_prompt_dir(prompt_id)

        if not prompt_dir.exists():
            logger.debug("Prompt directory not found", prompt_id=prompt_id)
            return []

        versions = []
        for version_file in prompt_dir.glob("*.json"):
            if version_file.stem.startswith('.'):
                continue  # Skip metadata files like .lock

            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                versions.append(PromptVersion(**data))
            except Exception as e:
                # Skip corrupted files
                logger.warning(
                    "Skipping corrupted version file",
                    file=str(version_file),
                    error=str(e),
                )
                continue

        # Sort by version tuple (major, minor, patch)
        versions.sort(key=lambda v: self._parse_version(v.version))

        logger.debug(
            "Listed versions",
            prompt_id=prompt_id,
            count=len(versions),
        )

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

        Example:
            >>> latest = storage.get_latest_version("prompt_001")
            >>> if latest:
            ...     print(f"Latest: v{latest.version}")
        """
        # Fast path: use index
        if self.use_index and self._index:
            with self._index_lock:
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

        Example:
            >>> deleted = storage.delete_version("prompt_001", "1.0.0")
            >>> if deleted:
            ...     print("Version deleted successfully")
        """
        version_file = self._get_version_file(prompt_id, version)
        lock_file = self._get_prompt_dir(prompt_id) / ".lock"

        if not version_file.exists():
            logger.debug("Version not found for deletion", prompt_id=prompt_id, version=version)
            return False

        # Acquire lock for this prompt
        with FileLock(lock_file):
            try:
                # Delete file
                version_file.unlink()

                # Update index
                if self.use_index:
                    self._update_index_delete(prompt_id, version)

                # Invalidate cache
                if self._cache:
                    cache_key = f"{prompt_id}:{version}"
                    self._cache.remove(cache_key)

                logger.info(
                    "Version deleted",
                    prompt_id=prompt_id,
                    version=version,
                )

                return True

            except Exception as e:
                logger.error(
                    "Failed to delete version",
                    prompt_id=prompt_id,
                    version=version,
                    error=str(e),
                )
                raise OptimizerError(
                    message=f"Failed to delete version {version}",
                    error_code="FS-DELETE-001",
                    context={"prompt_id": prompt_id, "error": str(e)}
                )

    def clear_all(self) -> None:
        """Clear all versions from storage (testing only).

        Warning:
            This operation is irreversible. Use with caution.

        Example:
            >>> storage.clear_all()
            >>> assert storage.list_versions("prompt_001") == []
        """
        logger.warning("Clearing all versions from storage")

        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Reset index
        if self.use_index:
            with self._index_lock:
                self._index = {
                    "version": "1.0.0",
                    "last_updated": datetime.now().isoformat(),
                    "total_prompts": 0,
                    "total_versions": 0,
                    "index": {}
                }
                self._save_index()

        # Clear cache
        if self._cache:
            self._cache.clear()

        logger.info("All versions cleared")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_prompt_dir(self, prompt_id: str) -> Path:
        """Get directory path for a prompt (with optional sharding).

        Args:
            prompt_id: Prompt identifier.

        Returns:
            Path to prompt directory.
        """
        if self.enable_sharding:
            # SHA-256 hash first N chars for shard
            hash_hex = hashlib.sha256(prompt_id.encode()).hexdigest()
            shard = hash_hex[:self.shard_depth]
            return self.storage_dir / shard / prompt_id
        else:
            return self.storage_dir / prompt_id

    def _get_version_file(self, prompt_id: str, version: str) -> Path:
        """Get file path for a specific version.

        Args:
            prompt_id: Prompt identifier.
            version: Version number.

        Returns:
            Path to version file.
        """
        return self._get_prompt_dir(prompt_id) / f"{version}.json"

    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write file atomically using temp file + rename.

        CRITICAL: This ensures no partial writes on crash/interruption.

        Args:
            path: Final file path.
            data: Data to write (will be JSON serialized).

        Raises:
            OptimizerError: If write operation fails.
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

    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string to tuple.

        Args:
            version: Semantic version string (e.g., "1.2.0").

        Returns:
            Tuple of (major, minor, patch).
        """
        try:
            major, minor, patch = version.split('.')
            return (int(major), int(minor), int(patch))
        except (ValueError, AttributeError):
            logger.warning("Invalid version format", version=version)
            return (0, 0, 0)

    # ========================================================================
    # Index Management
    # ========================================================================

    def _load_index(self) -> None:
        """Load global index from disk."""
        with self._index_lock:
            if self._index_path.exists():
                try:
                    with open(self._index_path, 'r', encoding='utf-8') as f:
                        self._index = json.load(f)
                    logger.debug("Index loaded", total_prompts=self._index.get("total_prompts", 0))
                except Exception as e:
                    logger.warning("Failed to load index, rebuilding", error=str(e))
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

            try:
                self._atomic_write(self._index_path, self._index)
                logger.debug("Index saved")
            except Exception as e:
                logger.error("Failed to save index", error=str(e))

    def _update_index_add(self, version: PromptVersion) -> None:
        """Update index when adding a version.

        Args:
            version: PromptVersion being added.
        """
        with self._index_lock:
            if not self._index:
                return

            prompt_id = version.prompt_id
            index_entry = self._index["index"].get(prompt_id)

            if not index_entry:
                # New prompt
                self._index["index"][prompt_id] = {
                    "latest_version": version.version,
                    "version_count": 1,
                    "versions": [version.version],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                self._index["total_prompts"] += 1
            else:
                # Existing prompt
                index_entry["versions"].append(version.version)
                index_entry["versions"].sort(key=self._parse_version)
                index_entry["latest_version"] = index_entry["versions"][-1]
                index_entry["version_count"] += 1
                index_entry["updated_at"] = datetime.now().isoformat()

            self._index["total_versions"] += 1
            self._index["last_updated"] = datetime.now().isoformat()
            self._save_index()

    def _update_index_delete(self, prompt_id: str, version: str) -> None:
        """Update index when deleting a version.

        Args:
            prompt_id: Prompt identifier.
            version: Version being deleted.
        """
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

            # Update latest_version or remove entry
            if index_entry["version_count"] > 0:
                # Recalculate latest
                index_entry["latest_version"] = index_entry["versions"][-1]
                index_entry["updated_at"] = datetime.now().isoformat()
            else:
                # No versions left, remove entry
                del self._index["index"][prompt_id]
                self._index["total_prompts"] -= 1

            self._index["last_updated"] = datetime.now().isoformat()
            self._save_index()

    def _rebuild_index(self) -> None:
        """Rebuild index by scanning all version files.

        Called when index is corrupted or missing.
        """
        logger.info("Rebuilding index from filesystem")

        with self._index_lock:
            self._index = {
                "version": "1.0.0",
                "last_updated": datetime.now().isoformat(),
                "total_prompts": 0,
                "total_versions": 0,
                "index": {}
            }

            # Scan all version files
            if self.enable_sharding:
                # Scan all shard directories
                for shard_dir in self.storage_dir.iterdir():
                    if shard_dir.is_dir() and not shard_dir.name.startswith('.'):
                        self._scan_shard_directory(shard_dir)
            else:
                # Scan storage directory directly
                self._scan_shard_directory(self.storage_dir)

            self._save_index()
            logger.info(
                "Index rebuilt",
                total_prompts=self._index["total_prompts"],
                total_versions=self._index["total_versions"],
            )

    def _scan_shard_directory(self, shard_dir: Path) -> None:
        """Scan a shard directory and update index.

        Args:
            shard_dir: Directory to scan.
        """
        for prompt_dir in shard_dir.iterdir():
            if not prompt_dir.is_dir() or prompt_dir.name.startswith('.'):
                continue

            prompt_id = prompt_dir.name
            versions_list = []

            for version_file in prompt_dir.glob("*.json"):
                if version_file.stem.startswith('.'):
                    continue

                try:
                    with open(version_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    version_obj = PromptVersion(**data)
                    versions_list.append(version_obj.version)
                except Exception:
                    continue

            if versions_list:
                versions_list.sort(key=self._parse_version)
                self._index["index"][prompt_id] = {
                    "latest_version": versions_list[-1],
                    "version_count": len(versions_list),
                    "versions": versions_list,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                self._index["total_prompts"] += 1
                self._index["total_versions"] += len(versions_list)

    # ========================================================================
    # Crash Recovery
    # ========================================================================

    def _recover_from_crash(self) -> None:
        """Clean up temp files from interrupted writes.

        Removes .tmp files older than 1 hour.
        """
        try:
            temp_files = list(self.storage_dir.glob("**/*.tmp*"))
            cleaned_count = 0

            for temp_file in temp_files:
                try:
                    age = time.time() - temp_file.stat().st_mtime
                    if age > 3600:  # Older than 1 hour
                        temp_file.unlink()
                        cleaned_count += 1
                except Exception:
                    pass  # Ignore errors

            if cleaned_count > 0:
                logger.info("Crash recovery completed", cleaned_files=cleaned_count)

        except Exception as e:
            logger.warning("Crash recovery failed", error=str(e))

    # ========================================================================
    # Public Utilities
    # ========================================================================

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage stats (total prompts, versions, disk usage, cache stats).

        Example:
            >>> stats = storage.get_storage_stats()
            >>> print(f"Total versions: {stats['total_versions']}")
        """
        stats = {
            "storage_dir": str(self.storage_dir),
            "total_prompts": 0,
            "total_versions": 0,
            "disk_usage_bytes": 0,
            "index_enabled": self.use_index,
            "cache_enabled": self.use_cache,
            "sharding_enabled": self.enable_sharding,
        }

        # Get index stats if available
        if self.use_index and self._index:
            with self._index_lock:
                stats["total_prompts"] = self._index.get("total_prompts", 0)
                stats["total_versions"] = self._index.get("total_versions", 0)

        # Get cache stats if available
        if self._cache:
            stats["cache_stats"] = self._cache.stats()

        # Calculate disk usage
        try:
            total_size = sum(
                f.stat().st_size
                for f in self.storage_dir.rglob("*.json")
                if not f.name.startswith('.')
            )
            stats["disk_usage_bytes"] = total_size
            stats["disk_usage_mb"] = total_size / (1024 * 1024)
        except Exception:
            pass

        return stats

    def rebuild_index(self) -> None:
        """Rebuild index from filesystem (public utility).

        Example:
            >>> storage.rebuild_index()
        """
        if not self.use_index:
            logger.warning("Index is not enabled")
            return

        self._rebuild_index()
