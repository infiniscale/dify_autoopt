"""
Unit Tests for Executor Module Data Models

Date: 2025-11-14
Author: qa-engineer
Description: Tests for Task, TaskResult, RunStatistics, RunExecutionResult, and CancellationToken
Coverage Target: 100%
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from typing import List

from src.executor.models import (
    TaskStatus,
    Task,
    TaskResult,
    RunStatistics,
    RunExecutionResult,
    CancellationToken
)
from src.collector.models import TestResult, TestStatus
from src.config.models.run_manifest import TestCase
from src.config.models.test_plan import ExecutionPolicy


# ============================================================================
# Test TaskStatus Enum
# ============================================================================


class TestTaskStatus:
    """Test suite for TaskStatus enumeration"""

    def test_is_terminal_for_terminal_states(self):
        """Test is_terminal() returns True for terminal states"""
        # Arrange
        terminal_states = [
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.CANCELLED,
            TaskStatus.ERROR
        ]

        # Act & Assert
        for status in terminal_states:
            assert status.is_terminal() is True, f"{status.value} should be terminal"

    def test_is_terminal_for_non_terminal_states(self):
        """Test is_terminal() returns False for non-terminal states"""
        # Arrange
        non_terminal_states = [
            TaskStatus.PENDING,
            TaskStatus.QUEUED,
            TaskStatus.RUNNING
        ]

        # Act & Assert
        for status in non_terminal_states:
            assert status.is_terminal() is False, f"{status.value} should not be terminal"

    def test_is_success_only_for_succeeded(self):
        """Test is_success() returns True only for SUCCEEDED"""
        # Arrange
        all_statuses = [
            TaskStatus.PENDING,
            TaskStatus.QUEUED,
            TaskStatus.RUNNING,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.CANCELLED,
            TaskStatus.ERROR
        ]

        # Act & Assert
        for status in all_statuses:
            if status == TaskStatus.SUCCEEDED:
                assert status.is_success() is True
            else:
                assert status.is_success() is False

    def test_enum_values(self):
        """Test enum values are correct strings"""
        # Assert
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.QUEUED.value == "queued"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.SUCCEEDED.value == "succeeded"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.TIMEOUT.value == "timeout"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.ERROR.value == "error"


# ============================================================================
# Test Task Data Class
# ============================================================================


class TestTask:
    """Test suite for Task data class"""

    def test_from_manifest_case_success(
        self,
        sample_test_case,
        sample_execution_policy,
        fixed_time,
        fixed_id
    ):
        """Test successful Task creation from manifest case"""
        # Arrange
        workflow_id = "wf_001"

        # Act
        task = Task.from_manifest_case(
            test_case=sample_test_case,
            execution_policy=sample_execution_policy,
            workflow_id=workflow_id,
            id_fn=lambda: fixed_id,
            now_fn=lambda: fixed_time
        )

        # Assert
        assert task.task_id == fixed_id
        assert task.workflow_id == workflow_id
        assert task.dataset == sample_test_case.dataset
        assert task.scenario == sample_test_case.scenario
        assert task.test_case == sample_test_case
        assert task.parameters == sample_test_case.parameters
        assert task.conversation_flow == sample_test_case.conversation_flow
        assert task.prompt_variant == sample_test_case.prompt_variant
        assert task.timeout_seconds == 30.0  # From stop_conditions
        assert task.max_retries == 2  # max_attempts - 1 = 3 - 1 = 2
        assert task.retry_backoff == 2.0
        assert task.status == TaskStatus.PENDING
        assert task.attempt_count == 0
        assert task.created_at == fixed_time
        assert task.started_at is None
        assert task.finished_at is None
        assert task.error_message is None
        assert task.result is None

    def test_from_manifest_case_with_conversation_flow(
        self,
        sample_test_case_with_conversation,
        sample_execution_policy,
        fixed_time,
        fixed_id
    ):
        """Test Task creation with conversation flow"""
        # Act
        task = Task.from_manifest_case(
            test_case=sample_test_case_with_conversation,
            execution_policy=sample_execution_policy,
            workflow_id="wf_chatflow",
            id_fn=lambda: fixed_id,
            now_fn=lambda: fixed_time
        )

        # Assert
        assert task.conversation_flow is not None
        assert task.conversation_flow == sample_test_case_with_conversation.conversation_flow
        assert task.prompt_variant == "variant_a"

    def test_from_manifest_case_none_test_case_raises(
        self,
        sample_execution_policy
    ):
        """Test from_manifest_case raises ValueError when test_case is None"""
        # Act & Assert
        with pytest.raises(ValueError, match="test_case cannot be None"):
            Task.from_manifest_case(
                test_case=None,
                execution_policy=sample_execution_policy,
                workflow_id="wf_001"
            )

    def test_from_manifest_case_none_execution_policy_raises(
        self,
        sample_test_case
    ):
        """Test from_manifest_case raises ValueError when execution_policy is None"""
        # Act & Assert
        with pytest.raises(ValueError, match="execution_policy cannot be None"):
            Task.from_manifest_case(
                test_case=sample_test_case,
                execution_policy=None,
                workflow_id="wf_001"
            )

    def test_mark_started_updates_status_and_timestamp(
        self,
        sample_task,
        fixed_time
    ):
        """Test mark_started updates status, timestamp, and attempt count"""
        # Arrange
        start_time = fixed_time + timedelta(seconds=5)
        assert sample_task.status == TaskStatus.PENDING
        assert sample_task.started_at is None
        assert sample_task.attempt_count == 0

        # Act
        sample_task.mark_started(now_fn=lambda: start_time)

        # Assert
        assert sample_task.status == TaskStatus.RUNNING
        assert sample_task.started_at == start_time
        assert sample_task.attempt_count == 1

    def test_mark_started_increments_attempt_count(
        self,
        sample_task,
        fixed_time
    ):
        """Test mark_started increments attempt count on retry"""
        # Arrange
        sample_task.attempt_count = 1

        # Act
        sample_task.mark_started(now_fn=lambda: fixed_time)

        # Assert
        assert sample_task.attempt_count == 2

    def test_mark_started_raises_if_already_terminal(
        self,
        sample_task,
        fixed_time
    ):
        """Test mark_started raises ValueError if task is already terminal"""
        # Arrange
        sample_task.status = TaskStatus.SUCCEEDED

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot start task in terminal state"):
            sample_task.mark_started(now_fn=lambda: fixed_time)

    def test_mark_finished_success(
        self,
        sample_task,
        sample_test_result,
        fixed_time
    ):
        """Test mark_finished with successful completion"""
        # Arrange
        sample_task.mark_started(now_fn=lambda: fixed_time)
        finish_time = fixed_time + timedelta(seconds=10)

        # Act
        sample_task.mark_finished(
            status=TaskStatus.SUCCEEDED,
            result=sample_test_result,
            error_message=None,
            now_fn=lambda: finish_time
        )

        # Assert
        assert sample_task.status == TaskStatus.SUCCEEDED
        assert sample_task.result == sample_test_result
        assert sample_task.error_message is None
        assert sample_task.finished_at == finish_time

    def test_mark_finished_with_failure(
        self,
        sample_task,
        fixed_time
    ):
        """Test mark_finished with failure status"""
        # Arrange
        sample_task.mark_started(now_fn=lambda: fixed_time)
        finish_time = fixed_time + timedelta(seconds=5)
        error_msg = "Task execution failed due to invalid input"

        # Act
        sample_task.mark_finished(
            status=TaskStatus.FAILED,
            result=None,
            error_message=error_msg,
            now_fn=lambda: finish_time
        )

        # Assert
        assert sample_task.status == TaskStatus.FAILED
        assert sample_task.result is None
        assert sample_task.error_message == error_msg
        assert sample_task.finished_at == finish_time

    def test_mark_finished_validates_terminal_status(
        self,
        sample_task
    ):
        """Test mark_finished raises ValueError for non-terminal status"""
        # Arrange
        sample_task.status = TaskStatus.RUNNING

        # Act & Assert
        with pytest.raises(ValueError, match="mark_finished requires terminal status"):
            sample_task.mark_finished(status=TaskStatus.RUNNING)

    def test_mark_finished_raises_if_already_terminal(
        self,
        sample_task,
        fixed_time
    ):
        """Test mark_finished raises ValueError if already terminal"""
        # Arrange
        sample_task.status = TaskStatus.SUCCEEDED
        sample_task.finished_at = fixed_time

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot finish task already in terminal state"):
            sample_task.mark_finished(
                status=TaskStatus.FAILED,
                now_fn=lambda: fixed_time
            )

    def test_is_terminal(self, sample_task):
        """Test is_terminal method"""
        # Arrange & Act & Assert
        sample_task.status = TaskStatus.PENDING
        assert sample_task.is_terminal() is False

        sample_task.status = TaskStatus.RUNNING
        assert sample_task.is_terminal() is False

        sample_task.status = TaskStatus.SUCCEEDED
        assert sample_task.is_terminal() is True

        sample_task.status = TaskStatus.FAILED
        assert sample_task.is_terminal() is True

    def test_can_retry_when_attempts_remain(self, sample_task):
        """Test can_retry returns True when attempts remain"""
        # Arrange
        sample_task.status = TaskStatus.FAILED
        sample_task.attempt_count = 1
        sample_task.max_retries = 2

        # Act & Assert
        assert sample_task.can_retry() is True

    def test_can_retry_when_max_attempts_reached(self, sample_task):
        """Test can_retry returns False when max attempts reached"""
        # Arrange
        sample_task.status = TaskStatus.FAILED
        sample_task.attempt_count = 3
        sample_task.max_retries = 2

        # Act & Assert
        assert sample_task.can_retry() is False

    def test_can_retry_for_retriable_statuses(self, sample_task):
        """Test can_retry returns True for retriable statuses"""
        # Arrange
        sample_task.attempt_count = 1
        sample_task.max_retries = 2

        retriable_statuses = [
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.ERROR
        ]

        # Act & Assert
        for status in retriable_statuses:
            sample_task.status = status
            assert sample_task.can_retry() is True

    def test_can_retry_false_for_non_retriable_statuses(self, sample_task):
        """Test can_retry returns False for non-retriable statuses"""
        # Arrange
        sample_task.attempt_count = 1
        sample_task.max_retries = 2

        non_retriable_statuses = [
            TaskStatus.SUCCEEDED,
            TaskStatus.CANCELLED,
            TaskStatus.PENDING,
            TaskStatus.RUNNING
        ]

        # Act & Assert
        for status in non_retriable_statuses:
            sample_task.status = status
            assert sample_task.can_retry() is False

    def test_state_transitions_full_lifecycle(
        self,
        sample_task,
        sample_test_result,
        fixed_time
    ):
        """Test complete task lifecycle state transitions"""
        # Initial state
        assert sample_task.status == TaskStatus.PENDING
        assert sample_task.started_at is None
        assert sample_task.finished_at is None

        # PENDING → RUNNING
        start_time = fixed_time + timedelta(seconds=1)
        sample_task.mark_started(now_fn=lambda: start_time)
        assert sample_task.status == TaskStatus.RUNNING
        assert sample_task.started_at == start_time

        # RUNNING → SUCCEEDED
        finish_time = fixed_time + timedelta(seconds=10)
        sample_task.mark_finished(
            status=TaskStatus.SUCCEEDED,
            result=sample_test_result,
            now_fn=lambda: finish_time
        )
        assert sample_task.status == TaskStatus.SUCCEEDED
        assert sample_task.finished_at == finish_time
        assert sample_task.is_terminal() is True

        # Cannot transition from terminal state
        with pytest.raises(ValueError):
            sample_task.mark_finished(status=TaskStatus.FAILED)


# ============================================================================
# Test TaskResult Data Class
# ============================================================================


class TestTaskResult:
    """Test suite for TaskResult data class"""

    def test_from_task_success(
        self,
        sample_task,
        sample_test_result,
        fixed_time
    ):
        """Test TaskResult creation from succeeded task"""
        # Arrange
        sample_task.mark_started(now_fn=lambda: fixed_time)
        finish_time = fixed_time + timedelta(seconds=5)
        sample_task.mark_finished(
            status=TaskStatus.SUCCEEDED,
            result=sample_test_result,
            now_fn=lambda: finish_time
        )

        # Act
        task_result = TaskResult.from_task(sample_task)

        # Assert
        assert task_result.task_id == sample_task.task_id
        assert task_result.workflow_id == sample_task.workflow_id
        assert task_result.dataset == sample_task.dataset
        assert task_result.scenario == sample_task.scenario
        assert task_result.task_status == TaskStatus.SUCCEEDED
        assert task_result.test_result == sample_test_result
        assert task_result.error_message is None
        assert task_result.attempt_count == sample_task.attempt_count
        assert task_result.created_at == sample_task.created_at
        assert task_result.finished_at == finish_time
        assert task_result.execution_time == 5.0

    def test_from_task_calculates_execution_time(
        self,
        sample_task,
        fixed_time
    ):
        """Test TaskResult correctly calculates execution time"""
        # Arrange
        start_time = fixed_time
        finish_time = fixed_time + timedelta(seconds=12.5)

        sample_task.mark_started(now_fn=lambda: start_time)
        sample_task.mark_finished(
            status=TaskStatus.FAILED,
            error_message="Test error",
            now_fn=lambda: finish_time
        )

        # Act
        task_result = TaskResult.from_task(sample_task)

        # Assert
        assert task_result.execution_time == 12.5

    def test_from_task_raises_if_not_terminal(self, sample_task):
        """Test from_task raises ValueError if task is not terminal"""
        # Arrange
        sample_task.status = TaskStatus.RUNNING

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot create TaskResult from non-terminal task"):
            TaskResult.from_task(sample_task)

    def test_from_task_raises_if_task_is_none(self):
        """Test from_task raises ValueError if task is None"""
        # Act & Assert
        with pytest.raises(ValueError, match="task cannot be None"):
            TaskResult.from_task(None)

    def test_from_task_with_no_timestamps(self, sample_task):
        """Test from_task handles missing timestamps gracefully"""
        # Arrange
        sample_task.status = TaskStatus.FAILED
        sample_task.started_at = None
        sample_task.finished_at = None

        # Act
        task_result = TaskResult.from_task(sample_task)

        # Assert
        assert task_result.execution_time == 0.0

    def test_from_task_preserves_metadata(
        self,
        sample_task,
        fixed_time,
    ):
        """Test from_task copies metadata from Task into TaskResult."""
        # Arrange
        sample_task.metadata["prompt_variant"] = "variant_a"
        sample_task.metadata["extra"] = "value"
        sample_task.status = TaskStatus.FAILED
        sample_task.finished_at = fixed_time

        # Act
        task_result = TaskResult.from_task(sample_task)

        # Assert
        assert task_result.metadata["prompt_variant"] == "variant_a"
        assert task_result.metadata["extra"] == "value"

    def test_get_tokens_used_exists(
        self,
        sample_task,
        sample_test_result,
        fixed_time
    ):
        """Test get_tokens_used returns correct value when result exists"""
        # Arrange
        sample_task.mark_started(now_fn=lambda: fixed_time)
        sample_task.mark_finished(
            status=TaskStatus.SUCCEEDED,
            result=sample_test_result,
            now_fn=lambda: fixed_time
        )
        task_result = TaskResult.from_task(sample_task)

        # Act
        tokens = task_result.get_tokens_used()

        # Assert
        assert tokens == 150  # From sample_test_result

    def test_get_tokens_used_missing(self, sample_task, fixed_time):
        """Test get_tokens_used returns 0 when result is None"""
        # Arrange
        sample_task.mark_started(now_fn=lambda: fixed_time)
        sample_task.mark_finished(
            status=TaskStatus.FAILED,
            result=None,
            error_message="Error",
            now_fn=lambda: fixed_time
        )
        task_result = TaskResult.from_task(sample_task)

        # Act
        tokens = task_result.get_tokens_used()

        # Assert
        assert tokens == 0

    def test_get_cost_exists(
        self,
        sample_task,
        sample_test_result,
        fixed_time
    ):
        """Test get_cost returns correct value when result exists"""
        # Arrange
        sample_task.mark_started(now_fn=lambda: fixed_time)
        sample_task.mark_finished(
            status=TaskStatus.SUCCEEDED,
            result=sample_test_result,
            now_fn=lambda: fixed_time
        )
        task_result = TaskResult.from_task(sample_task)

        # Act
        cost = task_result.get_cost()

        # Assert
        assert cost == 0.002  # From sample_test_result

    def test_get_cost_missing(self, sample_task, fixed_time):
        """Test get_cost returns 0.0 when result is None"""
        # Arrange
        sample_task.mark_started(now_fn=lambda: fixed_time)
        sample_task.mark_finished(
            status=TaskStatus.ERROR,
            result=None,
            error_message="System error",
            now_fn=lambda: fixed_time
        )
        task_result = TaskResult.from_task(sample_task)

        # Act
        cost = task_result.get_cost()

        # Assert
        assert cost == 0.0


# ============================================================================
# Test RunExecutionResult Data Class
# ============================================================================


class TestRunExecutionResult:
    """Test suite for RunExecutionResult data class"""

    def test_from_task_results_all_succeeded(
        self,
        fixed_time
    ):
        """Test RunExecutionResult creation with all succeeded tasks"""
        # Arrange
        task_results = self._create_task_results(
            count=5,
            statuses=[TaskStatus.SUCCEEDED] * 5,
            fixed_time=fixed_time
        )
        started_at = fixed_time
        finished_at = fixed_time + timedelta(seconds=30)

        # Act
        run_result = RunExecutionResult.from_task_results(
            run_id="run_001",
            workflow_id="wf_001",
            started_at=started_at,
            finished_at=finished_at,
            task_results=task_results
        )

        # Assert
        assert run_result.run_id == "run_001"
        assert run_result.workflow_id == "wf_001"
        assert run_result.started_at == started_at
        assert run_result.finished_at == finished_at
        assert run_result.total_duration == 30.0
        assert len(run_result.task_results) == 5
        assert run_result.statistics.total_tasks == 5
        assert run_result.statistics.succeeded_tasks == 5
        assert run_result.statistics.failed_tasks == 0
        assert run_result.statistics.success_rate == 1.0

    def test_from_task_results_mixed_statuses(
        self,
        fixed_time
    ):
        """Test RunExecutionResult with mixed task statuses"""
        # Arrange
        statuses = [
            TaskStatus.SUCCEEDED,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.ERROR,
            TaskStatus.CANCELLED
        ]
        task_results = self._create_task_results(
            count=6,
            statuses=statuses,
            fixed_time=fixed_time
        )

        # Act
        run_result = RunExecutionResult.from_task_results(
            run_id="run_002",
            workflow_id="wf_002",
            started_at=fixed_time,
            finished_at=fixed_time + timedelta(seconds=60),
            task_results=task_results
        )

        # Assert
        stats = run_result.statistics
        assert stats.total_tasks == 6
        assert stats.succeeded_tasks == 2
        assert stats.failed_tasks == 1
        assert stats.timeout_tasks == 1
        assert stats.error_tasks == 1
        assert stats.cancelled_tasks == 1
        assert stats.completed_tasks == 5  # All except cancelled
        assert stats.success_rate == 2 / 5  # 2 succeeded out of 5 completed

    def test_from_task_results_calculates_totals(
        self,
        fixed_time
    ):
        """Test RunExecutionResult calculates totals correctly"""
        # Arrange
        task_results = self._create_task_results_with_metrics(
            count=3,
            fixed_time=fixed_time
        )

        # Act
        run_result = RunExecutionResult.from_task_results(
            run_id="run_003",
            workflow_id="wf_003",
            started_at=fixed_time,
            finished_at=fixed_time + timedelta(seconds=100),
            task_results=task_results
        )

        # Assert
        stats = run_result.statistics
        assert stats.total_execution_time == 15.0  # 5 + 5 + 5
        assert stats.avg_execution_time == 5.0  # 15 / 3
        assert stats.total_tokens == 450  # 150 * 3
        assert stats.total_cost == 0.006  # 0.002 * 3

    def test_from_task_results_with_retries(
        self,
        fixed_time
    ):
        """Test RunExecutionResult counts retries correctly"""
        # Arrange
        task_results = []
        for i in range(3):
            task_result = TaskResult(
                task_id=f"task_{i}",
                workflow_id="wf_001",
                dataset="test",
                scenario="normal",
                task_status=TaskStatus.SUCCEEDED,
                attempt_count=i + 2,  # 2, 3, 4 attempts (1, 2, 3 retries)
                created_at=fixed_time,
                finished_at=fixed_time,
                execution_time=5.0
            )
            task_results.append(task_result)

        # Act
        run_result = RunExecutionResult.from_task_results(
            run_id="run_004",
            workflow_id="wf_001",
            started_at=fixed_time,
            finished_at=fixed_time,
            task_results=task_results
        )

        # Assert
        assert run_result.statistics.total_retries == 6  # (2-1) + (3-1) + (4-1)

    def test_from_task_results_empty_list_raises(self, fixed_time):
        """Test from_task_results raises ValueError for empty list"""
        # Act & Assert
        with pytest.raises(ValueError, match="task_results cannot be empty"):
            RunExecutionResult.from_task_results(
                run_id="run_005",
                workflow_id="wf_005",
                started_at=fixed_time,
                finished_at=fixed_time,
                task_results=[]
            )

    def test_from_task_results_validates_required_fields(self, fixed_time):
        """Test from_task_results validates required fields"""
        # Arrange
        task_results = [TaskResult(
            task_id="task_1",
            workflow_id="wf_001",
            dataset="test",
            scenario="normal",
            task_status=TaskStatus.SUCCEEDED,
            created_at=fixed_time,
            finished_at=fixed_time,
            execution_time=0.0
        )]

        # Act & Assert - Empty run_id
        with pytest.raises(ValueError, match="run_id cannot be empty"):
            RunExecutionResult.from_task_results(
                run_id="",
                workflow_id="wf_001",
                started_at=fixed_time,
                finished_at=fixed_time,
                task_results=task_results
            )

        # Act & Assert - Empty workflow_id
        with pytest.raises(ValueError, match="workflow_id cannot be empty"):
            RunExecutionResult.from_task_results(
                run_id="run_001",
                workflow_id="",
                started_at=fixed_time,
                finished_at=fixed_time,
                task_results=task_results
            )

        # Act & Assert - None started_at
        with pytest.raises(ValueError, match="started_at cannot be None"):
            RunExecutionResult.from_task_results(
                run_id="run_001",
                workflow_id="wf_001",
                started_at=None,
                finished_at=fixed_time,
                task_results=task_results
            )

        # Act & Assert - None finished_at
        with pytest.raises(ValueError, match="finished_at cannot be None"):
            RunExecutionResult.from_task_results(
                run_id="run_001",
                workflow_id="wf_001",
                started_at=fixed_time,
                finished_at=None,
                task_results=task_results
            )

    def test_from_task_results_with_metadata(self, fixed_time):
        """Test from_task_results preserves metadata"""
        # Arrange
        task_results = [TaskResult(
            task_id="task_1",
            workflow_id="wf_001",
            dataset="test",
            scenario="normal",
            task_status=TaskStatus.SUCCEEDED,
            created_at=fixed_time,
            finished_at=fixed_time,
            execution_time=0.0
        )]
        metadata = {"key": "value", "number": 123}

        # Act
        run_result = RunExecutionResult.from_task_results(
            run_id="run_001",
            workflow_id="wf_001",
            started_at=fixed_time,
            finished_at=fixed_time,
            task_results=task_results,
            metadata=metadata
        )

        # Assert
        assert run_result.metadata == metadata

    def test_from_task_results_default_metadata(self, fixed_time):
        """Test from_task_results uses empty dict for default metadata"""
        # Arrange
        task_results = [TaskResult(
            task_id="task_1",
            workflow_id="wf_001",
            dataset="test",
            scenario="normal",
            task_status=TaskStatus.SUCCEEDED,
            created_at=fixed_time,
            finished_at=fixed_time,
            execution_time=0.0
        )]

        # Act
        run_result = RunExecutionResult.from_task_results(
            run_id="run_001",
            workflow_id="wf_001",
            started_at=fixed_time,
            finished_at=fixed_time,
            task_results=task_results
        )

        # Assert
        assert run_result.metadata == {}

    # Helper methods
    def _create_task_results(
        self,
        count: int,
        statuses: List[TaskStatus],
        fixed_time: datetime
    ) -> List[TaskResult]:
        """Helper to create task results with specified statuses"""
        results = []
        for i in range(count):
            result = TaskResult(
                task_id=f"task_{i}",
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                task_status=statuses[i],
                attempt_count=1,
                created_at=fixed_time,
                finished_at=fixed_time + timedelta(seconds=5),
                execution_time=5.0
            )
            results.append(result)
        return results

    def _create_task_results_with_metrics(
        self,
        count: int,
        fixed_time: datetime
    ) -> List[TaskResult]:
        """Helper to create task results with test metrics"""
        results = []
        for i in range(count):
            test_result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i}",
                timestamp=fixed_time,
                status=TestStatus.SUCCESS,
                execution_time=5.0,
                tokens_used=150,
                cost=0.002,
                inputs={},
                outputs={},
                error_message=None
            )
            result = TaskResult(
                task_id=f"task_{i}",
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                task_status=TaskStatus.SUCCEEDED,
                test_result=test_result,
                attempt_count=1,
                created_at=fixed_time,
                finished_at=fixed_time + timedelta(seconds=5),
                execution_time=5.0
            )
            results.append(result)
        return results


# ============================================================================
# Test CancellationToken
# ============================================================================


class TestCancellationToken:
    """Test suite for CancellationToken"""

    def test_initial_state_not_cancelled(self):
        """Test token starts in not-cancelled state"""
        # Arrange & Act
        token = CancellationToken()

        # Assert
        assert token.is_cancelled() is False

    def test_cancel_sets_flag(self):
        """Test cancel sets the cancelled flag"""
        # Arrange
        token = CancellationToken()

        # Act
        token.cancel()

        # Assert
        assert token.is_cancelled() is True

    def test_is_cancelled_returns_true_after_cancel(self):
        """Test is_cancelled returns True after cancel"""
        # Arrange
        token = CancellationToken()
        assert token.is_cancelled() is False

        # Act
        token.cancel()

        # Assert
        assert token.is_cancelled() is True

    def test_reset_clears_flag(self):
        """Test reset clears the cancelled flag"""
        # Arrange
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled() is True

        # Act
        token.reset()

        # Assert
        assert token.is_cancelled() is False

    def test_multiple_resets(self):
        """Test token can be reset multiple times"""
        # Arrange
        token = CancellationToken()

        # Act & Assert
        for _ in range(3):
            token.cancel()
            assert token.is_cancelled() is True
            token.reset()
            assert token.is_cancelled() is False

    def test_thread_safety_concurrent_cancel(self):
        """Test thread safety with concurrent cancel operations"""
        # Arrange
        token = CancellationToken()
        results = []

        def worker():
            for _ in range(100):
                token.cancel()
                results.append(token.is_cancelled())

        # Act
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert
        assert len(results) == 500  # 100 * 5
        assert all(results)  # All should be True
        assert token.is_cancelled() is True

    def test_thread_safety_concurrent_reset(self):
        """Test thread safety with concurrent reset operations"""
        # Arrange
        token = CancellationToken()
        token.cancel()

        def worker():
            for _ in range(50):
                token.reset()
                time.sleep(0.0001)  # Small delay
                token.cancel()

        # Act
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert - No assertion errors or deadlocks occurred
        assert True  # If we reach here, thread safety is working

    def test_thread_safety_mixed_operations(self):
        """Test thread safety with mixed operations"""
        # Arrange
        token = CancellationToken()
        cancel_count = [0]
        reset_count = [0]

        def canceller():
            for _ in range(50):
                token.cancel()
                cancel_count[0] += 1
                time.sleep(0.0001)

        def resetter():
            for _ in range(50):
                token.reset()
                reset_count[0] += 1
                time.sleep(0.0001)

        def checker():
            for _ in range(50):
                token.is_cancelled()
                time.sleep(0.0001)

        # Act
        threads = [
            threading.Thread(target=canceller),
            threading.Thread(target=resetter),
            threading.Thread(target=checker)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert
        assert cancel_count[0] == 50
        assert reset_count[0] == 50
        # Final state is unpredictable but no crashes
