"""
Optimizer Module - Version Storage Interface

Date: 2025-11-17
Author: backend-developer
Description: Abstract interface for prompt version storage.
"""

import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class VersionStorage(ABC):
    """Abstract storage interface for prompt versions.

    This interface enables future integration with different storage backends
    (filesystem, database, etc.) without modifying VersionManager logic.

    Implementations:
        - InMemoryStorage (MVP): Dict-based in-memory storage.
        - FileSystemStorage (Future): JSON file-based storage.
        - DatabaseStorage (Future): SQLite/PostgreSQL storage.

    Example:
        >>> storage = InMemoryStorage()
        >>> storage.save_version(version)
        >>> retrieved = storage.get_version("prompt_001", "1.0.0")
    """

    @abstractmethod
    def save_version(self, version: "PromptVersion") -> None:  # type: ignore
        """Save a version to storage.

        Args:
            version: PromptVersion to save.

        Raises:
            VersionConflictError: If version already exists.
            OptimizerError: If save operation fails.

        Example:
            >>> storage.save_version(version)
        """
        pass

    @abstractmethod
    def get_version(
        self, prompt_id: str, version: str
    ) -> Optional["PromptVersion"]:  # type: ignore
        """Retrieve a specific version.

        Args:
            prompt_id: Prompt identifier.
            version: Version number (e.g., "1.2.0").

        Returns:
            PromptVersion or None if not found.

        Example:
            >>> version = storage.get_version("prompt_001", "1.0.0")
            >>> if version:
            ...     print(version.text)
        """
        pass

    @abstractmethod
    def list_versions(self, prompt_id: str) -> List["PromptVersion"]:  # type: ignore
        """List all versions for a prompt.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            List of PromptVersion, sorted by version number (oldest first).

        Example:
            >>> versions = storage.list_versions("prompt_001")
            >>> for v in versions:
            ...     print(f"v{v.version}: {v.analysis.overall_score}")
        """
        pass

    @abstractmethod
    def get_latest_version(
        self, prompt_id: str
    ) -> Optional["PromptVersion"]:  # type: ignore
        """Get the latest (most recent) version for a prompt.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            Latest PromptVersion or None if no versions exist.

        Example:
            >>> latest = storage.get_latest_version("prompt_001")
        """
        pass

    @abstractmethod
    def delete_version(self, prompt_id: str, version: str) -> bool:
        """Delete a specific version.

        Args:
            prompt_id: Prompt identifier.
            version: Version number.

        Returns:
            True if deleted, False if not found.

        Example:
            >>> deleted = storage.delete_version("prompt_001", "1.0.0")
        """
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """Clear all stored versions.

        Used primarily for testing purposes.

        Example:
            >>> storage.clear_all()
        """
        pass


class InMemoryStorage(VersionStorage):
    """In-memory storage for MVP.

    Uses a dictionary-based storage structure. Data is NOT persisted
    across restarts.

    Storage Structure:
        {
            "prompt_id": {
                "versions": [PromptVersion, ...],
                "current_version": "1.2.0"
            }
        }

    Attributes:
        _storage: Internal dictionary storing versions.
        _lock: Reentrant lock for thread-safe operations.

    Thread Safety:
        All operations are protected by RLock for safe concurrent access.

    Example:
        >>> storage = InMemoryStorage()
        >>> storage.save_version(version)
        >>> versions = storage.list_versions("prompt_001")
    """

    def __init__(self) -> None:
        """Initialize InMemoryStorage with empty storage and thread lock."""
        self._storage: Dict[str, Dict] = {}
        self._lock = threading.RLock()

    def save_version(self, version: "PromptVersion") -> None:  # type: ignore
        """Save a version to in-memory storage (thread-safe).

        Args:
            version: PromptVersion to save.

        Raises:
            VersionConflictError: If version already exists for this prompt.

        Example:
            >>> storage = InMemoryStorage()
            >>> storage.save_version(version)
        """
        with self._lock:
            # Import here to avoid circular dependency
            from ..exceptions import VersionConflictError

            prompt_id = version.prompt_id

            # Initialize storage for this prompt if not exists
            if prompt_id not in self._storage:
                self._storage[prompt_id] = {
                    "versions": [],
                    "current_version": version.version,
                }

            # Check for duplicate version
            existing = [
                v
                for v in self._storage[prompt_id]["versions"]
                if v.version == version.version
            ]
            if existing:
                raise VersionConflictError(
                    prompt_id=prompt_id,
                    version=version.version,
                    reason="Version already exists",
                )

            # Add version
            self._storage[prompt_id]["versions"].append(version)

            # Update current version
            self._storage[prompt_id]["current_version"] = version.version

    def get_version(
        self, prompt_id: str, version: str
    ) -> Optional["PromptVersion"]:  # type: ignore
        """Retrieve a specific version from storage (thread-safe).

        Args:
            prompt_id: Prompt identifier.
            version: Version number (e.g., "1.2.0").

        Returns:
            PromptVersion or None if not found.
        """
        with self._lock:
            if prompt_id not in self._storage:
                return None

            for v in self._storage[prompt_id]["versions"]:
                if v.version == version:
                    return v

            return None

    def list_versions(
        self,
        prompt_id: str,
        limit: Optional[int] = None
    ) -> List["PromptVersion"]:  # type: ignore
        """List all versions for a prompt, sorted by version number (thread-safe).

        Args:
            prompt_id: Prompt identifier.
            limit: Max versions to return (None = all).

        Returns:
            List of PromptVersion, sorted by version (oldest first).
        """
        with self._lock:
            if prompt_id not in self._storage:
                return []

            versions = self._storage[prompt_id]["versions"]

            # Sort by version tuple (major, minor, patch)
            def version_key(v):
                """Extract version tuple for sorting."""
                parts = v.version.split(".")
                return (int(parts[0]), int(parts[1]), int(parts[2]))

            sorted_versions = sorted(versions, key=version_key)

            # Apply limit
            if limit is not None:
                return sorted_versions[:limit]

            return sorted_versions

    def get_latest_version(
        self, prompt_id: str
    ) -> Optional["PromptVersion"]:  # type: ignore
        """Get the latest version for a prompt (thread-safe).

        Args:
            prompt_id: Prompt identifier.

        Returns:
            Latest PromptVersion or None if no versions exist.
        """
        with self._lock:
            versions = self.list_versions(prompt_id)
            if not versions:
                return None

            # Return the last one (highest version number)
            return versions[-1]

    def delete_version(self, prompt_id: str, version: str) -> bool:
        """Delete a specific version from storage (thread-safe).

        Args:
            prompt_id: Prompt identifier.
            version: Version number.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if prompt_id not in self._storage:
                return False

            versions = self._storage[prompt_id]["versions"]
            original_count = len(versions)

            self._storage[prompt_id]["versions"] = [
                v for v in versions if v.version != version
            ]

            deleted = len(self._storage[prompt_id]["versions"]) < original_count

            # Update current_version if needed
            if deleted and self._storage[prompt_id]["versions"]:
                # Set current to latest remaining version
                latest = self.get_latest_version(prompt_id)
                if latest:
                    self._storage[prompt_id]["current_version"] = latest.version

            return deleted

    def clear_all(self) -> None:
        """Clear all stored versions (thread-safe).

        Example:
            >>> storage.clear_all()
            >>> assert storage.list_versions("prompt_001") == []
        """
        with self._lock:
            self._storage.clear()
