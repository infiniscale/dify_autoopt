"""
Test Suite for TaskScheduler

Date: 2025-11-14
Author: qa-engineer
Description: Comprehensive unit tests for TaskScheduler with 100% coverage
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from unittest import mock

import pytest

from src.config.models import RunManifest, TestCase, ExecutionPolicy, RetryPolicy, RateLimit
from src.executor.models import Task, TaskResult, TaskStatus, CancellationToken
from src.executor.task_scheduler import TaskScheduler
from src.executor.stub_executor import StubExecutor


class TestTaskSchedulerInit:
    """Test TaskScheduler initialization."""

    def test_init_with_default_params(self):
        """Test TaskScheduler initialization with default parameters.

        Arrange:
            - No custom parameters
        Act:
            - Initialize TaskScheduler
        Assert:
            - Should have no rate limiter initially
        """
        scheduler = TaskScheduler()
        assert scheduler._rate_limiter is None

    def test_init_with_custom_functions(self):
        """Test TaskScheduler initialization with custom dependency injection.

        Arrange:
            - Custom now_fn, sleep_fn, id_fn
        Act:
            - Initialize TaskScheduler with custom functions
        Assert:
            - Custom functions should be stored
        """
        custom_now = lambda: datetime(2025, 1, 1, 0, 0, 0)
        custom_sleep = lambda s: None
        custom_id = lambda: "custom_id"

        scheduler = TaskScheduler(
            now_fn=custom_now,
            sleep_fn=custom_sleep,
            id_fn=custom_id
        )

        assert scheduler._now_fn == custom_now
        assert scheduler._sleep_fn == custom_sleep
        assert scheduler._id_fn == custom_id

    def test_init_inherits_from_concurrent_executor(self):
        """Test that TaskScheduler correctly inherits from ConcurrentExecutor.

        Arrange:
            - TaskScheduler instance
        Act:
            - Check parent class
        Assert:
            - Should be subclass of ConcurrentExecutor
        """
        from src.executor.concurrent_executor import ConcurrentExecutor
        scheduler = TaskScheduler()
        assert isinstance(scheduler, ConcurrentExecutor)


class TestSplitIntoBatches:
    """Test TaskScheduler._split_into_batches() method."""

    def test_split_exact_batches(self, task_scheduler, sample_manifest):
        """Test splitting tasks into exact batches.

        Arrange:
            - 10 tasks with batch_size=5
        Act:
            - Split into batches
        Assert:
            - Should create 2 batches of 5 tasks each
        """
        tasks = [
            Task.from_manifest_case(
                test_case=sample_manifest.cases[0],
                execution_policy=sample_manifest.execution_policy,
                workflow_id=sample_manifest.workflow_id,
                id_fn=lambda: f"task_{i}"
            )
            for i in range(10)
        ]

        batches = task_scheduler._split_into_batches(tasks, batch_size=5)

        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5

    def test_split_with_remainder(self, task_scheduler, sample_manifest):
        """Test splitting tasks with remainder batch.

        Arrange:
            - 11 tasks with batch_size=5
        Act:
            - Split into batches
        Assert:
            - Should create 3 batches: [5, 5, 1]
        """
        tasks = [
            Task.from_manifest_case(
                test_case=sample_manifest.cases[0],
                execution_policy=sample_manifest.execution_policy,
                workflow_id=sample_manifest.workflow_id,
                id_fn=lambda: f"task_{i}"
            )
            for i in range(11)
        ]

        batches = task_scheduler._split_into_batches(tasks, batch_size=5)

        assert len(batches) == 3
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5
        assert len(batches[2]) == 1

    def test_split_single_batch(self, task_scheduler, sample_manifest):
        """Test splitting when all tasks fit in one batch.

        Arrange:
            - 3 tasks with batch_size=10
        Act:
            - Split into batches
        Assert:
            - Should create 1 batch with 3 tasks
        """
        tasks = [
            Task.from_manifest_case(
                test_case=sample_manifest.cases[0],
                execution_policy=sample_manifest.execution_policy,
                workflow_id=sample_manifest.workflow_id,
                id_fn=lambda: f"task_{i}"
            )
            for i in range(3)
        ]

        batches = task_scheduler._split_into_batches(tasks, batch_size=10)

        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_split_empty_list(self, task_scheduler):
        """Test splitting empty task list.

        Arrange:
            - Empty task list
        Act:
            - Split into batches
        Assert:
            - Should return empty list
        """
        batches = task_scheduler._split_into_batches([], batch_size=5)
        assert batches == []


class TestExecuteTasks:
    """Test TaskScheduler._execute_tasks() method."""

    def test_execute_tasks_with_rate_limiter(self, execution_policy_with_rate_limit):
        """Test that rate limiter is initialized when rate_control is configured.

        Arrange:
            - TaskScheduler with mocked functions
            - Manifest with rate_control configured
        Act:
            - Execute tasks
        Assert:
            - Rate limiter should be created
            - Tasks should execute successfully
        """
        from src.executor.task_scheduler import TaskScheduler

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(6)
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy_with_rate_limit,
            rate_limits=execution_policy_with_rate_limit.rate_control,
            evaluator=None,
            metadata={}
        )

        scheduler = TaskScheduler(
            task_execution_func=lambda task: {"status": "succeeded", "outputs": {}},
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=lambda s: None,
            id_fn=lambda: "test_id"
        )

        tasks = scheduler._build_tasks(manifest)
        results = scheduler._execute_tasks(tasks, manifest, None)

        # Verify rate limiter was created
        assert scheduler._rate_limiter is not None
        # All tasks should complete
        assert len(results) == 6

    def test_execute_tasks_creates_correct_batches(self, execution_policy_with_rate_limit):
        """Test that tasks are split into correct batches based on batch_size.

        Arrange:
            - 9 tasks with batch_size=3
        Act:
            - Execute and track batch execution
        Assert:
            - Should create 3 batches
        """
        from src.executor.task_scheduler import TaskScheduler

        execution_policy = ExecutionPolicy(
            concurrency=2,
            batch_size=3,
            rate_control=execution_policy_with_rate_limit.rate_control,
            backoff_seconds=0.0,
            retry_policy=RetryPolicy(),
            stop_conditions={}
        )

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(9)
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=execution_policy.rate_control,
            evaluator=None,
            metadata={}
        )

        scheduler = TaskScheduler(
            task_execution_func=lambda task: {"status": "succeeded", "outputs": {}},
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=lambda s: None,
            id_fn=lambda: "test_id"
        )

        tasks = scheduler._build_tasks(manifest)
        batches = scheduler._split_into_batches(tasks, batch_size=3)

        assert len(batches) == 3
        assert all(len(batch) == 3 for batch in batches)

    def test_execute_tasks_with_backoff(self, execution_policy_with_rate_limit):
        """Test batch backoff between batches.

        Arrange:
            - Execution policy with backoff_seconds=0.5
            - Mock sleep_fn to track calls
        Act:
            - Execute 2 batches of tasks
        Assert:
            - sleep_fn should be called for backoff between batches
        """
        from src.executor.task_scheduler import TaskScheduler

        sleep_calls = []

        def track_sleep(seconds):
            sleep_calls.append(seconds)

        execution_policy = ExecutionPolicy(
            concurrency=2,
            batch_size=2,
            rate_control=execution_policy_with_rate_limit.rate_control,
            backoff_seconds=0.5,
            retry_policy=RetryPolicy(),
            stop_conditions={}
        )

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(4)
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=execution_policy.rate_control,
            evaluator=None,
            metadata={}
        )

        scheduler = TaskScheduler(
            task_execution_func=lambda task: {"status": "succeeded", "outputs": {}},
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=track_sleep,
            id_fn=lambda: "test_id"
        )

        tasks = scheduler._build_tasks(manifest)
        scheduler._execute_tasks(tasks, manifest, None)

        # Should have exactly 1 backoff call (between 2 batches, not after last batch)
        assert sleep_calls.count(0.5) == 1

    def test_execute_tasks_empty_list(self, execution_policy_with_rate_limit):
        """Test executing empty task list.

        Arrange:
            - Empty task list
        Act:
            - Call _execute_tasks
        Assert:
            - Should return empty list
        """
        from src.executor.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=lambda s: None,
            id_fn=lambda: "test_id"
        )

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=[],
            execution_policy=execution_policy_with_rate_limit,
            rate_limits=execution_policy_with_rate_limit.rate_control,
            evaluator=None,
            metadata={}
        )

        results = scheduler._execute_tasks([], manifest, None)
        assert results == []

    def test_execute_tasks_with_cancellation(self, execution_policy_with_rate_limit):
        """Test task cancellation during batch execution.

        Arrange:
            - CancellationToken that is pre-cancelled
            - 2 batches of tasks
        Act:
            - Execute tasks with cancellation
        Assert:
            - Second batch should be cancelled
        """
        from src.executor.task_scheduler import TaskScheduler

        execution_policy = ExecutionPolicy(
            concurrency=2,
            batch_size=2,
            rate_control=execution_policy_with_rate_limit.rate_control,
            backoff_seconds=0.0,
            retry_policy=RetryPolicy(),
            stop_conditions={}
        )

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(4)
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=execution_policy.rate_control,
            evaluator=None,
            metadata={}
        )

        cancellation_token = CancellationToken()

        # Track batch execution
        batch_count = [0]

        def exec_func_with_cancel(task):
            batch_count[0] += 1
            # Cancel after first batch (2 tasks)
            if batch_count[0] == 2:
                cancellation_token.cancel()
            return {"status": "succeeded", "outputs": {}}

        scheduler = TaskScheduler(
            task_execution_func=exec_func_with_cancel,
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=lambda s: None,
            id_fn=lambda: "test_id"
        )

        tasks = scheduler._build_tasks(manifest)
        results = scheduler._execute_tasks(tasks, manifest, cancellation_token)

        # Should have some cancelled tasks
        cancelled_count = sum(
            1 for r in results
            if r.task_status == TaskStatus.CANCELLED
        )
        assert cancelled_count > 0

    def test_execute_tasks_with_cancellation_during_backoff(self, execution_policy_with_rate_limit):
        """Test task cancellation during backoff period.

        Arrange:
            - CancellationToken
            - 2 batches with backoff between them
        Act:
            - Execute first batch, cancel before calling sleep for backoff
        Assert:
            - Should break out of batch loop
            - Only first batch results should be returned
        """
        from src.executor.task_scheduler import TaskScheduler

        # Use a rate limit config that won't interfere
        rate_limit = RateLimit(per_minute=6000, burst=100)

        execution_policy = ExecutionPolicy(
            concurrency=2,
            batch_size=2,
            rate_control=rate_limit,
            backoff_seconds=1.0,
            retry_policy=RetryPolicy(),
            stop_conditions={}
        )

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(4)
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=rate_limit,
            evaluator=None,
            metadata={}
        )

        cancellation_token = CancellationToken()
        task_count = [0]

        def exec_with_cancel(task):
            task_count[0] += 1
            # Cancel after first batch completes (2 tasks)
            if task_count[0] == 2:
                cancellation_token.cancel()
            return {"status": "succeeded", "outputs": {}}

        scheduler = TaskScheduler(
            task_execution_func=exec_with_cancel,
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=lambda s: None,
            id_fn=lambda: "test_id"
        )

        tasks = scheduler._build_tasks(manifest)
        results = scheduler._execute_tasks(tasks, manifest, cancellation_token)

        # Verify execution was interrupted after cancellation during backoff
        completed_count = sum(
            1 for r in results
            if r.task_status == TaskStatus.SUCCEEDED
        )
        # First batch should be completed (some or all of 2 tasks)
        assert completed_count >= 1
        # Break during backoff means remaining batches are not added to results
        # Only completed batch results are returned
        assert len(results) == completed_count

    def test_execute_tasks_aggregates_all_batch_results(self, execution_policy_with_rate_limit):
        """Test that results from all batches are aggregated correctly.

        Arrange:
            - Multiple batches of tasks
        Act:
            - Execute all batches
        Assert:
            - All task results should be returned
        """
        from src.executor.task_scheduler import TaskScheduler

        execution_policy = ExecutionPolicy(
            concurrency=2,
            batch_size=3,
            rate_control=execution_policy_with_rate_limit.rate_control,
            backoff_seconds=0.0,
            retry_policy=RetryPolicy(),
            stop_conditions={}
        )

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(7)  # Will create 3 batches: [3, 3, 1]
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=execution_policy.rate_control,
            evaluator=None,
            metadata={}
        )

        scheduler = TaskScheduler(
            task_execution_func=lambda task: {"status": "succeeded", "outputs": {}},
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=lambda s: None,
            id_fn=lambda: "test_id"
        )

        tasks = scheduler._build_tasks(manifest)
        results = scheduler._execute_tasks(tasks, manifest, None)

        # All 7 tasks should be in results
        assert len(results) == 7


class TestStopConditions:
    """Test TaskScheduler._should_stop() method and stop condition logic."""

    def test_should_stop_max_failures(self, task_scheduler):
        """Test stop condition based on max_failures.

        Arrange:
            - Stop condition: max_failures=2
            - 3 failed tasks in results
        Act:
            - Check _should_stop
        Assert:
            - Should return True
        """
        stop_conditions = {"max_failures": 2}
        start_time = datetime(2025, 1, 1, 0, 0, 0)

        results = [
            TaskResult(
                task_id=f"task_{i}",
                workflow_id="wf_001",
                dataset="test",
                scenario="normal",
                task_status=TaskStatus.FAILED,
                created_at=start_time,
                finished_at=start_time
            )
            for i in range(3)
        ]

        should_stop = task_scheduler._should_stop(results, stop_conditions, start_time)
        assert should_stop is True

    def test_should_stop_timeout(self, mock_now_fn, mock_sleep_fn, mock_id_fn):
        """Test stop condition based on timeout.

        Arrange:
            - Stop condition: timeout=5 seconds
            - Elapsed time > 5 seconds
        Act:
            - Check _should_stop
        Assert:
            - Should return True
        """
        start_time = datetime(2025, 1, 1, 0, 0, 0)
        current_time = start_time + timedelta(seconds=6)

        def time_now():
            return current_time

        scheduler = TaskScheduler(
            now_fn=time_now,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        stop_conditions = {"timeout": 5}
        results = []

        should_stop = scheduler._should_stop(results, stop_conditions, start_time)
        assert should_stop is True

    def test_should_not_stop_when_below_threshold(self, task_scheduler):
        """Test that execution continues when below failure threshold.

        Arrange:
            - Stop condition: max_failures=3
            - Only 2 failed tasks
        Act:
            - Check _should_stop
        Assert:
            - Should return False
        """
        stop_conditions = {"max_failures": 3}
        start_time = datetime(2025, 1, 1, 0, 0, 0)

        results = [
            TaskResult(
                task_id=f"task_{i}",
                workflow_id="wf_001",
                dataset="test",
                scenario="normal",
                task_status=TaskStatus.FAILED,
                created_at=start_time,
                finished_at=start_time
            )
            for i in range(2)
        ]

        should_stop = task_scheduler._should_stop(results, stop_conditions, start_time)
        assert should_stop is False

    def test_should_stop_counts_errors_and_timeouts(self, task_scheduler):
        """Test that stop condition counts ERROR and TIMEOUT as failures.

        Arrange:
            - Stop condition: max_failures=2
            - Mix of FAILED, ERROR, TIMEOUT statuses
        Act:
            - Check _should_stop
        Assert:
            - Should count all failure types
        """
        stop_conditions = {"max_failures": 2}
        start_time = datetime(2025, 1, 1, 0, 0, 0)

        results = [
            TaskResult(
                task_id="task_1",
                workflow_id="wf_001",
                dataset="test",
                scenario="normal",
                task_status=TaskStatus.FAILED,
                created_at=start_time,
                finished_at=start_time
            ),
            TaskResult(
                task_id="task_2",
                workflow_id="wf_001",
                dataset="test",
                scenario="normal",
                task_status=TaskStatus.ERROR,
                created_at=start_time,
                finished_at=start_time
            ),
            TaskResult(
                task_id="task_3",
                workflow_id="wf_001",
                dataset="test",
                scenario="normal",
                task_status=TaskStatus.TIMEOUT,
                created_at=start_time,
                finished_at=start_time
            )
        ]

        should_stop = task_scheduler._should_stop(results, stop_conditions, start_time)
        assert should_stop is True


class TestIntegration:
    """Test complete TaskScheduler workflows."""

    def test_full_workflow_with_all_features(self, execution_policy_with_rate_limit):
        """Test complete workflow with rate limiting, batching, and backoff.

        Arrange:
            - Manifest with rate control, batching, backoff
            - Multiple tasks across multiple batches
        Act:
            - Execute full workflow via run_manifest
        Assert:
            - All tasks should complete successfully
            - Rate limiting, batching, and backoff should work together
        """
        from src.executor.task_scheduler import TaskScheduler

        sleep_calls = []

        def track_sleep(seconds):
            sleep_calls.append(seconds)

        execution_policy = ExecutionPolicy(
            concurrency=2,
            batch_size=3,
            rate_control=execution_policy_with_rate_limit.rate_control,
            backoff_seconds=0.2,
            retry_policy=RetryPolicy(max_attempts=1),
            stop_conditions={}
        )

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(9)
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=execution_policy.rate_control,
            evaluator=None,
            metadata={}
        )

        scheduler = TaskScheduler(
            task_execution_func=lambda task: {"status": "succeeded", "outputs": {}},
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=track_sleep,
            id_fn=lambda: "test_id"
        )

        results = scheduler.run_manifest(manifest)

        # Verify all tasks completed successfully
        assert len(results.task_results) == 9
        assert all(r.task_status == TaskStatus.SUCCEEDED for r in results.task_results)

        # Verify backoff calls (3 batches = 2 backoff calls)
        assert sleep_calls.count(0.2) == 2

    def test_workflow_with_stop_condition_triggered(self, execution_policy_with_rate_limit):
        """Test workflow that stops early due to max_failures.

        Arrange:
            - Stop condition: max_failures=2
            - Tasks configured to fail
        Act:
            - Execute workflow
        Assert:
            - Should stop after 2 failures
            - Remaining tasks should be cancelled
        """
        from src.executor.task_scheduler import TaskScheduler
        from src.utils.exceptions import TaskExecutionException

        call_count = [0]

        def failing_exec(task):
            call_count[0] += 1
            raise TaskExecutionException(
                message=f"Simulated failure for task {task.task_id}",
                task_id=task.task_id,
                attempt=task.attempt_count
            )

        execution_policy = ExecutionPolicy(
            concurrency=2,
            batch_size=2,
            rate_control=execution_policy_with_rate_limit.rate_control,
            backoff_seconds=0.0,
            retry_policy=RetryPolicy(max_attempts=1),
            stop_conditions={"max_failures": 2}
        )

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(6)
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=execution_policy.rate_control,
            evaluator=None,
            metadata={}
        )

        scheduler = TaskScheduler(
            task_execution_func=failing_exec,
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=lambda s: None,
            id_fn=lambda: "test_id"
        )

        results = scheduler.run_manifest(manifest)

        # Count failures and cancellations
        failure_count = sum(
            1 for r in results.task_results
            if r.task_status in {TaskStatus.FAILED, TaskStatus.ERROR}
        )
        cancelled_count = sum(
            1 for r in results.task_results
            if r.task_status == TaskStatus.CANCELLED
        )

        # Should have triggered stop condition
        assert failure_count >= 2
        assert cancelled_count > 0

    def test_workflow_with_single_batch(self, execution_policy_with_rate_limit):
        """Test workflow with all tasks fitting in single batch.

        Arrange:
            - 3 tasks with batch_size=10
        Act:
            - Execute workflow
        Assert:
            - Should complete in single batch
            - No backoff should occur
        """
        from src.executor.task_scheduler import TaskScheduler

        sleep_calls = []

        def track_sleep(seconds):
            sleep_calls.append(seconds)

        execution_policy = ExecutionPolicy(
            concurrency=2,
            batch_size=10,
            rate_control=execution_policy_with_rate_limit.rate_control,
            backoff_seconds=0.5,
            retry_policy=RetryPolicy(),
            stop_conditions={}
        )

        test_cases = [
            TestCase(
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                parameters={"query": f"test {i}"}
            )
            for i in range(3)
        ]

        manifest = RunManifest(
            workflow_id="wf_001",
            workflow_version="1.0.0",
            prompt_variant="baseline",
            dsl_payload="test: payload",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=execution_policy.rate_control,
            evaluator=None,
            metadata={}
        )

        scheduler = TaskScheduler(
            task_execution_func=lambda task: {"status": "succeeded", "outputs": {}},
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=track_sleep,
            id_fn=lambda: "test_id"
        )

        results = scheduler.run_manifest(manifest)

        # All tasks should succeed
        assert len(results.task_results) == 3
        assert all(r.task_status == TaskStatus.SUCCEEDED for r in results.task_results)

        # No backoff should occur (only 1 batch)
        assert 0.5 not in sleep_calls
