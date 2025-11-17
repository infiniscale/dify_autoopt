"""
Optimizer Module - Interfaces Package

Date: 2025-11-17
Author: backend-developer
Description: Defines abstract interfaces for external dependencies.
"""

from .llm_client import LLMClient, StubLLMClient
from .storage import VersionStorage, InMemoryStorage
from .filesystem_storage import FileSystemStorage

__all__ = [
    "LLMClient",
    "StubLLMClient",
    "VersionStorage",
    "InMemoryStorage",
    "FileSystemStorage",
]
