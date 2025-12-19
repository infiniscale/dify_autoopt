"""
Unit tests for ConcurrentExecutor

Date: 2025-11-17
Author: qa-engineer
Description: Focused tests for retry / timeout / cancellation branches to reach 100% coverage.
"""

import types

import pytest
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import patch

from src.config.models import RetryPolicy
from src.utils.exceptions import (
    TaskExecutionException,
    TaskTimeoutException,
    SchedulerException,
)
from src.executor.concurrent_executor import ConcurrentExecutor
from src.executor.models import Task, TaskStatus, CancellationToken


class TestConcurrentExecutorCoverage:
    """Focused test class for 100% coverage."""

    def test_empty_tasks_list(self, concurrent_executor, sample_manifest):
        """Empty tasks list returns empty results (early return)."""
        results = concurrent_executor._execute_tasks([], sample_manifest)
        assert results == []

    def test_successful_execution_basic(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """Basic successful execution path for a single task."""

        def successful_task_func(task: Task):
            return {"status": "success"}

        executor = ConcurrentExecutor(
            task_execution_func=successful_task_func,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_basic",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = sample_manifest.execution_policy.retry_policy

        result = executor._execute_single_task(task, retry_policy)

        assert result.task_status == TaskStatus.SUCCEEDED

    def test_retry_on_timeout_error_sets_pending_then_succeeds(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """First attempt hits TimeoutError, second attempt succeeds."""
        call_state = {"count": 0}

        class FakeFuture:
            def __init__(self):
                self._cancelled = False

            def result(self, timeout=None):
                call_state["count"] += 1
                if call_state["count"] == 1:
                    raise FuturesTimeoutError()
                return {"status": "ok"}

            def cancel(self):
                self._cancelled = True

        class FakeExecutor:
            def __init__(self, max_workers: int):
                self._max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, task):
                # We don't call fn here; we only simulate timeout + success.
                return FakeFuture()

        executor = ConcurrentExecutor(
            task_execution_func=lambda t: {"status": "ignored"},
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_timeout_error",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = RetryPolicy(
            max_attempts=2,
            backoff_seconds=0.1,
            backoff_multiplier=1.0,
        )

        with patch("src.executor.concurrent_executor.ThreadPoolExecutor", FakeExecutor):
            result = executor._execute_single_task(task, retry_policy)

        assert call_state["count"] == 2
        assert result.task_status == TaskStatus.SUCCEEDED

    def test_retry_on_task_timeout_exception_sets_pending_then_succeeds(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """First attempt raises TaskTimeoutException, second attempt succeeds."""
        call_count = {"value": 0}

        def timeout_then_success(task: Task):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise TaskTimeoutException(
                    message="Explicit timeout",
                    task_id=task.task_id,
                    timeout_seconds=task.timeout_seconds,
                    elapsed_seconds=task.timeout_seconds,
                )
            return {"status": "ok"}

        executor = ConcurrentExecutor(
            task_execution_func=timeout_then_success,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_timeout_exception",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = RetryPolicy(
            max_attempts=2,
            backoff_seconds=0.1,
            backoff_multiplier=1.0,
        )

        result = executor._execute_single_task(task, retry_policy)

        assert call_count["value"] == 2
        assert result.task_status == TaskStatus.SUCCEEDED

    def test_retry_on_generic_exception_sets_pending_then_succeeds(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """First attempt raises generic Exception, second attempt succeeds."""
        call_count = {"value": 0}

        def fail_then_success(task: Task):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise RuntimeError("Transient error")
            return {"status": "ok"}

        executor = ConcurrentExecutor(
            task_execution_func=fail_then_success,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_error",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = RetryPolicy(
            max_attempts=2,
            backoff_seconds=0.1,
            backoff_multiplier=1.0,
        )

        result = executor._execute_single_task(task, retry_policy)

        assert call_count["value"] == 2
        assert result.task_status == TaskStatus.SUCCEEDED

    def test_cancel_during_retry_backoff_marks_task_cancelled(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """Cancellation during retry backoff should mark task CANCELLED."""
        call_count = {"value": 0}

        def always_fail(task: Task):
            call_count["value"] += 1
            raise TaskExecutionException(
                message="Transient failure",
                task_id=task.task_id,
                attempt=call_count["value"],
            )

        executor = ConcurrentExecutor(
            task_execution_func=always_fail,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_cancel_backoff",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = RetryPolicy(
            max_attempts=3,
            backoff_seconds=1.0,
            backoff_multiplier=1.0,
        )

        token = CancellationToken()
        flags = {"execution_checked": False, "backoff_checked": False}

        original_is_cancelled = token.is_cancelled

        def fake_is_cancelled() -> bool:
            if not flags["execution_checked"]:
                flags["execution_checked"] = True
                return False
            if not flags["backoff_checked"]:
                flags["backoff_checked"] = True
                return True
            return original_is_cancelled()

        # Override instance method to control per-call behaviour
        token.is_cancelled = fake_is_cancelled  # type: ignore[assignment]

        result = executor._execute_single_task(task, retry_policy, token)

        assert flags["execution_checked"] is True
        assert flags["backoff_checked"] is True
        assert result.task_status == TaskStatus.CANCELLED
        assert "retry backoff" in (result.error_message or "").lower()

    def test_fallback_error_handling_when_no_attempts(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """Fallback path: no attempts performed should mark ERROR with default message."""
        execution_called = {"value": False}

        def should_not_be_called(task: Task):
            execution_called["value"] = True
            return {"status": "ok"}

        executor = ConcurrentExecutor(
            task_execution_func=should_not_be_called,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_fallback",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        # Use a minimal namespace to bypass normal RetryPolicy validation and trigger the fallback
        retry_policy = types.SimpleNamespace(
            max_attempts=0,
            backoff_seconds=0.0,
            backoff_multiplier=1.0,
        )

        result = executor._execute_single_task(task, retry_policy)

        assert execution_called["value"] is False
        assert result.task_status == TaskStatus.ERROR
        assert result.error_message == "Unknown error during task execution"

    # ---------------------------------------------------------------------
    # Additional tests for _execute_tasks and remaining branches
    # ---------------------------------------------------------------------

    def test_execute_tasks_cancelled_before_start(
            self,
            concurrent_executor,
            sample_manifest,
            sample_test_case,
            mock_id_fn,
            mock_now_fn,
    ):
        """If cancellation token is cancelled before start, all tasks are cancelled."""
        tasks = [
            Task.from_manifest_case(
                sample_test_case,
                sample_manifest.execution_policy,
                workflow_id="wf_cancel_before",
                id_fn=mock_id_fn,
                now_fn=mock_now_fn,
            )
            for _ in range(2)
        ]
        token = CancellationToken()
        token.cancel()

        results = concurrent_executor._execute_tasks(tasks, sample_manifest, token)

        assert len(results) == 2
        assert all(r.task_status == TaskStatus.CANCELLED for r in results)

    def test_execute_tasks_cancelled_during_submission(
            self,
            sample_manifest,
            sample_test_case,
            mock_id_fn,
            mock_now_fn,
            mock_sleep_fn,
    ):
        """Cancellation during submission should mark remaining tasks as cancelled."""
        tasks = [
            Task.from_manifest_case(
                sample_test_case,
                sample_manifest.execution_policy,
                workflow_id="wf_submit_cancel",
                id_fn=mock_id_fn,
                now_fn=mock_now_fn,
            )
            for _ in range(5)
        ]

        token = CancellationToken()
        call_state = {"count": 0}

        from concurrent.futures import ThreadPoolExecutor

        original_submit = ThreadPoolExecutor.submit

        def submit_with_cancel(self, fn, *args, **kwargs):
            call_state["count"] += 1
            if call_state["count"] == 2:
                token.cancel()
            return original_submit(self, fn, *args, **kwargs)

        executor = ConcurrentExecutor(
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        with patch.object(ThreadPoolExecutor, "submit", submit_with_cancel):
            results = executor._execute_tasks(tasks, sample_manifest, token)

        assert len(results) == 5
        cancelled_count = sum(1 for r in results if r.task_status == TaskStatus.CANCELLED)
        assert cancelled_count >= 1

    def test_execute_tasks_future_exception_sets_error_status(
            self,
            sample_manifest,
            sample_test_case,
            mock_id_fn,
            mock_now_fn,
            mock_sleep_fn,
    ):
        """Exceptions from futures during collection should mark task as ERROR."""
        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_future_error",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        class FakeFuture:
            def result(self):
                raise RuntimeError("unexpected")

            def cancelled(self):
                return False

            def done(self):
                return True

        class FakeExecutor:
            def __init__(self, max_workers: int):
                self._max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                return FakeFuture()

        executor = ConcurrentExecutor(
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        with patch("src.executor.concurrent_executor.ThreadPoolExecutor", FakeExecutor), patch(
                "src.executor.concurrent_executor.as_completed", lambda mapping: list(mapping.keys())
        ):
            results = executor._execute_tasks([task], sample_manifest)

        assert len(results) == 1
        assert results[0].task_status == TaskStatus.ERROR
        assert "Unexpected error during execution" in (results[0].error_message or "")

    def test_execute_tasks_cancelled_future_records_execution_time(
            self,
            sample_manifest,
            sample_test_case,
            mock_id_fn,
            mock_now_fn,
            mock_sleep_fn,
    ):
        """Cancelled futures with timestamps should compute execution_time."""
        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_cancel_future",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )
        # Pre-populate timestamps so duration path is exercised
        task.started_at = mock_now_fn()
        task.finished_at = mock_now_fn()

        class FakeFuture:
            def cancelled(self):
                return True

            def done(self):
                return False

        class FakeExecutor:
            def __init__(self, max_workers: int):
                self._max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                return FakeFuture()

        executor = ConcurrentExecutor(
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        with patch("src.executor.concurrent_executor.ThreadPoolExecutor", FakeExecutor), patch(
                "src.executor.concurrent_executor.as_completed", lambda mapping: []
        ):
            results = executor._execute_tasks([task], sample_manifest)

        assert len(results) == 1
        result = results[0]
        assert result.task_status == TaskStatus.CANCELLED
        assert result.execution_time >= 0.0

    def test_execute_tasks_thread_pool_failure_raises_scheduler_exception(
            self,
            concurrent_executor,
            sample_manifest,
            sample_test_case,
            mock_id_fn,
            mock_now_fn,
    ):
        """Failure creating thread pool should raise SchedulerException."""
        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_pool_fail",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        class FailingExecutor:
            def __init__(self, max_workers: int):
                self._max_workers = max_workers

            def __enter__(self):
                raise RuntimeError("pool init failed")

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch("src.executor.concurrent_executor.ThreadPoolExecutor", FailingExecutor):
            with pytest.raises(SchedulerException):
                concurrent_executor._execute_tasks([task], sample_manifest)

    def test_single_task_cancelled_before_execution(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """Cancellation before first attempt in _execute_single_task."""
        executor = ConcurrentExecutor(
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_single_cancel",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = RetryPolicy(
            max_attempts=3,
            backoff_seconds=0.1,
            backoff_multiplier=1.0,
        )

        token = CancellationToken()
        token.cancel()

        result = executor._execute_single_task(task, retry_policy, token)

        assert result.task_status == TaskStatus.CANCELLED
        assert "cancelled during execution" in (result.error_message or "").lower()

    def test_timeout_error_on_last_attempt_marks_timeout(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """TimeoutError on last attempt should mark task as TIMEOUT."""
        call_state = {"count": 0}

        class FakeFuture:
            def result(self, timeout=None):
                call_state["count"] += 1
                raise FuturesTimeoutError()

            def cancel(self):
                # Called by _execute_single_task on timeout
                return None

        class FakeExecutor:
            def __init__(self, max_workers: int):
                self._max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                return FakeFuture()

        executor = ConcurrentExecutor(
            task_execution_func=lambda t: {"status": "ignored"},
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_timeout_last",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = RetryPolicy(
            max_attempts=1,
            backoff_seconds=0.1,
            backoff_multiplier=1.0,
        )

        with patch("src.executor.concurrent_executor.ThreadPoolExecutor", FakeExecutor):
            result = executor._execute_single_task(task, retry_policy)

        assert result.task_status == TaskStatus.TIMEOUT
        assert "timeout" in (result.error_message or "").lower()

    def test_task_timeout_exception_on_last_attempt_marks_timeout(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """TaskTimeoutException on last attempt should mark timeout status."""

        def always_timeout(task: Task):
            raise TaskTimeoutException(
                message="Timeout",
                task_id=task.task_id,
                timeout_seconds=task.timeout_seconds,
                elapsed_seconds=task.timeout_seconds,
            )

        executor = ConcurrentExecutor(
            task_execution_func=always_timeout,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_timeout_exc_last",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = RetryPolicy(
            max_attempts=1,
            backoff_seconds=0.1,
            backoff_multiplier=1.0,
        )

        result = executor._execute_single_task(task, retry_policy)

        assert result.task_status == TaskStatus.TIMEOUT
        assert "Timeout" in (result.error_message or "Timeout")

    def test_generic_exception_on_last_attempt_marks_error(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case,
    ):
        """Generic exception on last attempt should mark ERROR status."""

        def always_fail(task: Task):
            raise RuntimeError("fatal")

        executor = ConcurrentExecutor(
            task_execution_func=always_fail,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn,
        )

        task = Task.from_manifest_case(
            sample_test_case,
            sample_manifest.execution_policy,
            workflow_id="wf_error_last",
            id_fn=mock_id_fn,
            now_fn=mock_now_fn,
        )

        retry_policy = RetryPolicy(
            max_attempts=1,
            backoff_seconds=0.1,
            backoff_multiplier=1.0,
        )

        result = executor._execute_single_task(task, retry_policy)

        assert result.task_status == TaskStatus.ERROR
        assert "Unexpected error" in (result.error_message or "")
