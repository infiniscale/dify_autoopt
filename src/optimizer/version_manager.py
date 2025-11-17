"""
Optimizer Module - Version Manager

Date: 2025-11-17
Author: backend-developer
Description: Manages prompt version history with semantic versioning.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from .exceptions import VersionConflictError, VersionError, VersionNotFoundError
from .interfaces.storage import InMemoryStorage, VersionStorage
from .models import (
    OptimizationResult,
    Prompt,
    PromptAnalysis,
    PromptVersion,
)


class VersionManager:
    """Manage prompt version history.

    Tracks prompt evolution using semantic versioning (major.minor.patch).

    Version Numbering:
        - 1.0.0: Baseline version
        - 1.1.0: Minor optimization
        - 1.2.0: Another minor optimization
        - 2.0.0: Major restructure

    Attributes:
        _storage: Version storage backend.
        _logger: Loguru logger instance.

    Example:
        >>> manager = VersionManager()
        >>> version = manager.create_version(prompt, analysis)
        >>> history = manager.get_version_history("prompt_001")
    """

    def __init__(
        self,
        storage: Optional[VersionStorage] = None,
        custom_logger: Optional[Any] = None,
    ) -> None:
        """Initialize VersionManager.

        Args:
            storage: Version storage backend (defaults to InMemoryStorage).
            custom_logger: Optional custom logger instance.
        """
        self._storage = storage or InMemoryStorage()
        self._logger = custom_logger or logger.bind(module="optimizer.version")

    def create_version(
        self,
        prompt: Prompt,
        analysis: PromptAnalysis,
        optimization_result: Optional[OptimizationResult] = None,
        parent_version: Optional[str] = None,
    ) -> PromptVersion:
        """Create a new version for a prompt.

        Automatically assigns the next version number based on existing versions.

        Args:
            prompt: Prompt object for this version.
            analysis: PromptAnalysis for this version.
            optimization_result: OptimizationResult if this is an optimized version.
            parent_version: Parent version number (for lineage tracking).

        Returns:
            Created PromptVersion object.

        Raises:
            VersionConflictError: If version conflict occurs.

        Example:
            >>> version = manager.create_version(
            ...     prompt=prompt,
            ...     analysis=analysis,
            ...     optimization_result=None,  # Baseline
            ...     parent_version=None
            ... )
        """
        self._logger.info(f"Creating version for prompt '{prompt.id}'")

        # Determine next version number
        existing_versions = self._storage.list_versions(prompt.id)

        if not existing_versions:
            # First version (baseline)
            version_number = "1.0.0"
        else:
            # Increment version
            latest = existing_versions[-1]
            version_number = self._increment_version(
                latest.version, is_major=optimization_result is not None
            )

        # Create version object
        version = PromptVersion(
            prompt_id=prompt.id,
            version=version_number,
            prompt=prompt,
            analysis=analysis,
            optimization_result=optimization_result,
            parent_version=parent_version,
            created_at=datetime.now(),
            metadata={
                "author": "baseline" if optimization_result is None else "optimizer",
                "strategy": (
                    optimization_result.strategy.value if optimization_result else None
                ),
            },
        )

        # Save to storage
        self._storage.save_version(version)

        self._logger.info(f"Created version {version_number} for prompt '{prompt.id}'")

        return version

    def get_version(self, prompt_id: str, version: str) -> PromptVersion:
        """Get a specific version of a prompt.

        Args:
            prompt_id: Prompt identifier.
            version: Version number (e.g., "1.2.0").

        Returns:
            PromptVersion object.

        Raises:
            VersionNotFoundError: If version doesn't exist.

        Example:
            >>> version = manager.get_version("prompt_001", "1.0.0")
            >>> print(version.prompt.text)
        """
        result = self._storage.get_version(prompt_id, version)

        if result is None:
            raise VersionNotFoundError(prompt_id, version)

        return result

    def get_latest_version(self, prompt_id: str) -> PromptVersion:
        """Get the latest version of a prompt.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            Latest PromptVersion object.

        Raises:
            VersionNotFoundError: If no versions exist.

        Example:
            >>> latest = manager.get_latest_version("prompt_001")
        """
        result = self._storage.get_latest_version(prompt_id)

        if result is None:
            raise VersionNotFoundError(
                prompt_id=prompt_id,
                version="latest",
                context={"reason": "No versions exist for this prompt"},
            )

        return result

    def get_version_history(self, prompt_id: str) -> List[PromptVersion]:
        """Get all versions for a prompt, sorted chronologically.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            List of PromptVersion objects (oldest first).

        Example:
            >>> history = manager.get_version_history("prompt_001")
            >>> for v in history:
            ...     print(f"v{v.version}: score={v.analysis.overall_score}")
        """
        return self._storage.list_versions(prompt_id)

    def compare_versions(
        self,
        prompt_id: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """Compare two versions of a prompt.

        Args:
            prompt_id: Prompt identifier.
            version1: First version number.
            version2: Second version number.

        Returns:
            Comparison dictionary with diffs and metrics.

        Raises:
            VersionNotFoundError: If either version doesn't exist.

        Example:
            >>> comparison = manager.compare_versions(
            ...     "prompt_001", "1.0.0", "1.1.0"
            ... )
            >>> print(comparison["improvement"])
        """
        v1 = self.get_version(prompt_id, version1)
        v2 = self.get_version(prompt_id, version2)

        # Calculate improvements
        score_delta = v2.analysis.overall_score - v1.analysis.overall_score
        clarity_delta = v2.analysis.clarity_score - v1.analysis.clarity_score
        efficiency_delta = v2.analysis.efficiency_score - v1.analysis.efficiency_score

        # Text diff
        text_diff = self._compute_text_diff(v1.prompt.text, v2.prompt.text)

        comparison = {
            "prompt_id": prompt_id,
            "version1": version1,
            "version2": version2,
            "improvement": score_delta,
            "clarity_improvement": clarity_delta,
            "efficiency_improvement": efficiency_delta,
            "text_diff": text_diff,
            "version1_analysis": {
                "overall_score": v1.analysis.overall_score,
                "clarity_score": v1.analysis.clarity_score,
                "efficiency_score": v1.analysis.efficiency_score,
                "issues_count": len(v1.analysis.issues),
            },
            "version2_analysis": {
                "overall_score": v2.analysis.overall_score,
                "clarity_score": v2.analysis.clarity_score,
                "efficiency_score": v2.analysis.efficiency_score,
                "issues_count": len(v2.analysis.issues),
            },
            "changes": v2.optimization_result.changes if v2.optimization_result else [],
        }

        return comparison

    def rollback(self, prompt_id: str, target_version: str) -> PromptVersion:
        """Rollback to a specific version.

        Creates a new version with the content of the target version.

        Args:
            prompt_id: Prompt identifier.
            target_version: Version to rollback to.

        Returns:
            New PromptVersion object (with incremented version number).

        Raises:
            VersionNotFoundError: If target version doesn't exist.

        Example:
            >>> rolled_back = manager.rollback("prompt_001", "1.0.0")
        """
        self._logger.info(
            f"Rolling back prompt '{prompt_id}' to version {target_version}"
        )

        # Get target version
        target = self.get_version(prompt_id, target_version)

        # Get current latest version
        current_latest = self.get_latest_version(prompt_id)

        # Create new version with target content
        rollback_version = self.create_version(
            prompt=target.prompt,
            analysis=target.analysis,
            optimization_result=None,  # Rollback is not an optimization
            parent_version=current_latest.version,
        )

        # Update metadata to indicate rollback
        rollback_version.metadata["rollback_from"] = current_latest.version
        rollback_version.metadata["rollback_to"] = target_version
        rollback_version.metadata["author"] = "rollback"

        self._logger.info(
            f"Rolled back to version {target_version}, "
            f"created new version {rollback_version.version}"
        )

        return rollback_version

    def get_best_version(self, prompt_id: str) -> PromptVersion:
        """Get the version with the highest overall score.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            PromptVersion with highest overall_score.

        Raises:
            VersionNotFoundError: If no versions exist.

        Example:
            >>> best = manager.get_best_version("prompt_001")
        """
        versions = self.get_version_history(prompt_id)

        if not versions:
            raise VersionNotFoundError(
                prompt_id=prompt_id,
                version="best",
                context={"reason": "No versions exist for this prompt"},
            )

        # Find version with highest overall score
        best = max(versions, key=lambda v: v.analysis.overall_score)

        return best

    def _increment_version(self, current_version: str, is_major: bool = False) -> str:
        """Increment version number.

        Args:
            current_version: Current version string (e.g., "1.2.0").
            is_major: Whether this is a major change (increments minor).

        Returns:
            Next version string.

        Example:
            >>> self._increment_version("1.2.0", is_major=False)
            "1.2.1"
            >>> self._increment_version("1.2.0", is_major=True)
            "1.3.0"
        """
        parts = current_version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        if is_major:
            # Increment minor version, reset patch
            minor += 1
            patch = 0
        else:
            # Increment patch version
            patch += 1

        return f"{major}.{minor}.{patch}"

    def _compute_text_diff(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compute text difference metrics.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Dictionary with diff metrics.
        """
        # Simple diff metrics
        len_diff = len(text2) - len(text1)
        word_diff = len(text2.split()) - len(text1.split())

        # Character-level similarity (simple Jaccard)
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        if not set1 and not set2:
            similarity = 1.0
        elif not set1 or not set2:
            similarity = 0.0
        else:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            similarity = intersection / union if union > 0 else 0.0

        return {
            "character_diff": len_diff,
            "word_diff": word_diff,
            "similarity": similarity,
            "length_ratio": len(text2) / len(text1) if len(text1) > 0 else 0.0,
        }

    def delete_version(self, prompt_id: str, version: str) -> bool:
        """Delete a specific version.

        Args:
            prompt_id: Prompt identifier.
            version: Version to delete.

        Returns:
            True if deleted, False if not found.

        Warning:
            Use with caution. This operation cannot be undone.

        Example:
            >>> deleted = manager.delete_version("prompt_001", "1.1.0")
        """
        self._logger.warning(f"Deleting version {version} for prompt '{prompt_id}'")

        deleted = self._storage.delete_version(prompt_id, version)

        if deleted:
            self._logger.info(f"Deleted version {version}")
        else:
            self._logger.warning(f"Version {version} not found, nothing deleted")

        return deleted

    def clear_all_versions(self) -> None:
        """Clear all versions from storage.

        Warning:
            This operation deletes ALL version history. Use only for testing.

        Example:
            >>> manager.clear_all_versions()
        """
        self._logger.warning("Clearing all versions from storage")
        self._storage.clear_all()
        self._logger.info("All versions cleared")
