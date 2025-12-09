"""
Execution Concurrency Utilities

Provides a lightweight semaphore-based concurrency manager that can be shared
across workflow execution and optimizer pipelines. Each manager controls its
own independent slot pool to avoid cross-interference between subsystems.
"""

from contextlib import contextmanager
import threading


class ConcurrencyManager:
    """Simple semaphore-based concurrency manager."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"concurrency limit must be >= 1, got {limit}")
        self.limit = limit
        self._semaphore = threading.Semaphore(limit)

    @contextmanager
    def slot(self):
        """Acquire a slot for the duration of the context."""
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()
