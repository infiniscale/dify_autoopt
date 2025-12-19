"""
Executor Module - Custom Exceptions (Ownership Anchor)

Defines executor-owned exception classes that were previously colocated
under src.config.utils.exceptions. This module serves as the canonical
location, while src.config.utils.exceptions re-exports them for
backward compatibility.
"""


class CaseGenerationError(Exception):
    """Raised when test case generation fails in executor components."""


class ManifestError(Exception):
    """Raised when building or consuming RunManifest fails in executor."""
