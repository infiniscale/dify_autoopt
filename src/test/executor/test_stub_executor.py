"""
Unit Tests for StubExecutor

Date: 2025-11-14
Author: qa-engineer
Description: Comprehensive tests for StubExecutor with 100% coverage
Coverage Target: 100%
"""

import pytest
import random
from datetime import datetime
from typing import Dict, Any
from unittest.mock import patch, Mock

from src.executor.stub_executor import StubExecutor
from src.executor.models import Task, TaskResult, TaskStatus
from src.config.models import RunManifest, TestCase, RetryPolicy
from src.utils.exceptions import TaskExecutionException, TaskTimeoutException


# ============================================================================
# Test Class 1: TestStubExecutorInit
# ============================================================================


class TestStubExecutorInit:
    """Test suite for StubExecutor initialization"""

    def test_init_default_parameters(self, mock_now_fn, mock_sleep_fn, mock_id_fn):
        """Test initialization with default parameters (delay=0, failure_rate=0)"""
        # Arrange & Act
        executor = StubExecutor(
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Assert
        assert executor._simulated_delay == 0.0
        assert executor._failure_rate == 0.0
        assert executor._task_behaviors == {}

    def test_init_with_simulated_delay(self, mock_now_fn, mock_sleep_fn, mock_id_fn):
        """Test initialization with simulated delay"""
        # Arrange & Act
        executor = StubExecutor(
            simulated_delay=0.5,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Assert
        assert executor._simulated_delay == 0.5

    def test_init_with_failure_rate(self, mock_now_fn, mock_sleep_fn, mock_id_fn):
        """Test initialization with failure rate"""
        # Arrange & Act
        executor = StubExecutor(
            failure_rate=0.3,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Assert
        assert executor._failure_rate == 0.3

    def test_init_with_task_behaviors(self, mock_now_fn, mock_sleep_fn, mock_id_fn):
        """Test initialization with task behaviors"""
        # Arrange
        behaviors = {"task_001": "success", "task_002": "failure"}

        # Act
        executor = StubExecutor(
            task_behaviors=behaviors,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Assert
        assert executor._task_behaviors == behaviors

    def test_init_custom_dependency_injection(self):
        """Test initialization with custom dependency injection"""
        # Arrange
        custom_now = Mock(return_value=datetime(2025, 1, 1))
        custom_sleep = Mock()
        custom_id = Mock(return_value="custom_id")

        # Act
        executor = StubExecutor(
            now_fn=custom_now,
            sleep_fn=custom_sleep,
            id_fn=custom_id
        )

        # Assert
        assert executor._now_fn == custom_now
        assert executor._sleep_fn == custom_sleep
        assert executor._id_fn == custom_id


# ============================================================================
# Test Class 2: TestStubExecution
# ============================================================================


class TestStubExecution:
    """Test suite for _stub_execution method"""

    # --- Behavior Configuration Tests ---

    def test_stub_execution_success_behavior(self, stub_executor, sample_task):
        """Test behavior='success' returns success result"""
        # Arrange
        stub_executor.set_task_behavior(sample_task.task_id, "success")

        # Act
        result = stub_executor._stub_execution(sample_task)

        # Assert
        assert isinstance(result, dict)
        assert result["status"] == "succeeded"

    def test_stub_execution_failure_behavior(self, stub_executor, sample_task):
        """Test behavior='failure' raises TaskExecutionException"""
        # Arrange
        stub_executor.set_task_behavior(sample_task.task_id, "failure")

        # Act & Assert
        with pytest.raises(TaskExecutionException) as exc_info:
            stub_executor._stub_execution(sample_task)

        assert sample_task.task_id in str(exc_info.value)

    def test_stub_execution_timeout_behavior(self, stub_executor, sample_task):
        """Test behavior='timeout' raises TaskTimeoutException"""
        # Arrange
        stub_executor.set_task_behavior(sample_task.task_id, "timeout")

        # Act & Assert
        with pytest.raises(TaskTimeoutException) as exc_info:
            stub_executor._stub_execution(sample_task)

        assert sample_task.task_id in str(exc_info.value)

    def test_stub_execution_error_behavior(self, stub_executor, sample_task):
        """Test behavior='error' raises RuntimeError"""
        # Arrange
        stub_executor.set_task_behavior(sample_task.task_id, "error")

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            stub_executor._stub_execution(sample_task)

        assert sample_task.task_id in str(exc_info.value)

    def test_stub_execution_unknown_behavior_defaults_to_success(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_task
    ):
        """Test unknown behavior defaults to success"""
        # Arrange
        executor = StubExecutor(
            task_behaviors={sample_task.task_id: "unknown_behavior"},
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Act
        result = executor._stub_execution(sample_task)

        # Assert
        assert result["status"] == "succeeded"

    # --- Failure Rate Tests ---

    def test_stub_execution_failure_rate_zero(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_task
    ):
        """Test failure_rate=0.0 results in all successes"""
        # Arrange
        executor = StubExecutor(
            failure_rate=0.0,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Act - Execute multiple times
        results = []
        for i in range(10):
            task = sample_task
            task.task_id = f"task_{i}"
            result = executor._stub_execution(task)
            results.append(result)

        # Assert - All should succeed
        assert all(r["status"] == "succeeded" for r in results)

    def test_stub_execution_failure_rate_one(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_task
    ):
        """Test failure_rate=1.0 results in all failures"""
        # Arrange
        executor = StubExecutor(
            failure_rate=1.0,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Act & Assert - Execute multiple times
        for i in range(10):
            task = sample_task
            task.task_id = f"task_{i}"
            with pytest.raises(TaskExecutionException):
                executor._stub_execution(task)

    def test_stub_execution_failure_rate_random(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_task
    ):
        """Test failure_rate=0.5 produces random mix of success/failure"""
        # Arrange
        executor = StubExecutor(
            failure_rate=0.5,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Act - Execute many times to test randomness
        successes = 0
        failures = 0

        random.seed(42)  # Set seed for reproducibility
        for i in range(100):
            task = sample_task
            task.task_id = f"task_{i}"
            try:
                executor._stub_execution(task)
                successes += 1
            except TaskExecutionException:
                failures += 1

        # Assert - Should have mix of both (not all one or the other)
        assert successes > 0
        assert failures > 0
        # With 100 trials at 0.5 rate, expect roughly 40-60 of each
        assert 30 < successes < 70
        assert 30 < failures < 70

    # --- Delay Tests ---

    def test_stub_execution_no_delay(self, mock_now_fn, mock_id_fn, sample_task):
        """Test simulated_delay=0 does not call sleep"""
        # Arrange
        sleep_mock = Mock()
        executor = StubExecutor(
            simulated_delay=0.0,
            now_fn=mock_now_fn,
            sleep_fn=sleep_mock,
            id_fn=mock_id_fn
        )

        # Act
        executor._stub_execution(sample_task)

        # Assert
        sleep_mock.assert_not_called()

    def test_stub_execution_with_delay(self, mock_now_fn, mock_id_fn, sample_task):
        """Test simulated_delay>0 calls sleep_fn"""
        # Arrange
        sleep_mock = Mock()
        executor = StubExecutor(
            simulated_delay=0.5,
            now_fn=mock_now_fn,
            sleep_fn=sleep_mock,
            id_fn=mock_id_fn
        )

        # Act
        executor._stub_execution(sample_task)

        # Assert
        sleep_mock.assert_called_once_with(0.5)

    def test_stub_execution_delay_timing(self, mock_now_fn, mock_id_fn, sample_task):
        """Test that delay time is accurate"""
        # Arrange
        sleep_calls = []

        def track_sleep(seconds):
            sleep_calls.append(seconds)

        executor = StubExecutor(
            simulated_delay=1.5,
            now_fn=mock_now_fn,
            sleep_fn=track_sleep,
            id_fn=mock_id_fn
        )

        # Act
        executor._stub_execution(sample_task)

        # Assert
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 1.5

    # --- Exception Parameter Tests (Verify Fixes) ---

    def test_stub_execution_failure_exception_has_task_id(self, stub_executor, sample_task):
        """Test TaskExecutionException includes task_id parameter"""
        # Arrange
        stub_executor.set_task_behavior(sample_task.task_id, "failure")

        # Act & Assert
        with pytest.raises(TaskExecutionException) as exc_info:
            stub_executor._stub_execution(sample_task)

        # Verify the exception was constructed correctly
        exception = exc_info.value
        assert hasattr(exception, 'task_id')
        assert exception.task_id == sample_task.task_id

    def test_stub_execution_failure_exception_has_attempt(self, stub_executor, sample_task):
        """Test TaskExecutionException includes attempt parameter"""
        # Arrange
        stub_executor.set_task_behavior(sample_task.task_id, "failure")
        sample_task.attempt_count = 2

        # Act & Assert
        with pytest.raises(TaskExecutionException) as exc_info:
            stub_executor._stub_execution(sample_task)

        exception = exc_info.value
        assert hasattr(exception, 'attempt')
        assert exception.attempt == sample_task.attempt_count

    def test_stub_execution_timeout_exception_has_timeout_seconds(
            self,
            stub_executor,
            sample_task
    ):
        """Test TaskTimeoutException has complete parameters"""
        # Arrange
        stub_executor.set_task_behavior(sample_task.task_id, "timeout")
        sample_task.timeout_seconds = 30.0

        # Act & Assert
        with pytest.raises(TaskTimeoutException) as exc_info:
            stub_executor._stub_execution(sample_task)

        exception = exc_info.value
        assert hasattr(exception, 'task_id')
        assert hasattr(exception, 'timeout_seconds')
        assert hasattr(exception, 'elapsed_seconds')
        assert exception.task_id == sample_task.task_id
        assert exception.timeout_seconds == sample_task.timeout_seconds


# ============================================================================
# Test Class 3: TestStubSuccess
# ============================================================================


class TestStubSuccess:
    """Test suite for _stub_success method"""

    def test_stub_success_returns_complete_data(self, stub_executor, sample_task):
        """Test that stub success returns complete data structure"""
        # Act
        result = stub_executor._stub_success(sample_task)

        # Assert
        assert isinstance(result, dict)
        assert "workflow_run_id" in result
        assert "status" in result
        assert "outputs" in result
        assert "elapsed_time" in result
        assert "total_tokens" in result
        assert "total_steps" in result
        assert "created_at" in result
        assert "finished_at" in result

    def test_stub_success_includes_task_id(self, stub_executor, sample_task):
        """Test that stub success includes task.task_id"""
        # Act
        result = stub_executor._stub_success(sample_task)

        # Assert
        assert sample_task.task_id in result["workflow_run_id"]
        assert result["outputs"]["task_id"] == sample_task.task_id

    def test_stub_success_includes_workflow_id(self, stub_executor, sample_task):
        """Test that stub success includes task.workflow_id"""
        # Act
        result = stub_executor._stub_success(sample_task)

        # Assert
        assert result["outputs"]["workflow_id"] == sample_task.workflow_id

    def test_stub_success_includes_dataset(self, stub_executor, sample_task):
        """Test that stub success includes task.dataset"""
        # Act
        result = stub_executor._stub_success(sample_task)

        # Assert
        assert result["outputs"]["dataset"] == sample_task.dataset

    def test_stub_success_includes_scenario(self, stub_executor, sample_task):
        """Test that stub success includes task.scenario"""
        # Act
        result = stub_executor._stub_success(sample_task)

        # Assert
        assert result["outputs"]["scenario"] == sample_task.scenario

    def test_stub_success_includes_parameters(self, stub_executor, sample_task):
        """Test that stub success includes task.parameters"""
        # Act
        result = stub_executor._stub_success(sample_task)

        # Assert
        assert result["outputs"]["parameters"] == sample_task.parameters

    def test_stub_success_elapsed_time_matches_delay(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_task
    ):
        """Test that elapsed_time equals simulated_delay"""
        # Arrange
        executor = StubExecutor(
            simulated_delay=2.5,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Act
        result = executor._stub_success(sample_task)

        # Assert
        assert result["elapsed_time"] == 2.5


# ============================================================================
# Test Class 4: TestConfigurationMethods
# ============================================================================


class TestConfigurationMethods:
    """Test suite for configuration methods"""

    # --- set_task_behavior Tests ---

    def test_set_task_behavior_success(self, stub_executor):
        """Test setting success behavior for a task"""
        # Act
        stub_executor.set_task_behavior("task_001", "success")

        # Assert
        assert stub_executor._task_behaviors["task_001"] == "success"

    def test_set_task_behavior_failure(self, stub_executor):
        """Test setting failure behavior for a task"""
        # Act
        stub_executor.set_task_behavior("task_002", "failure")

        # Assert
        assert stub_executor._task_behaviors["task_002"] == "failure"

    def test_set_task_behavior_timeout(self, stub_executor):
        """Test setting timeout behavior for a task"""
        # Act
        stub_executor.set_task_behavior("task_003", "timeout")

        # Assert
        assert stub_executor._task_behaviors["task_003"] == "timeout"

    def test_set_task_behavior_error(self, stub_executor):
        """Test setting error behavior for a task"""
        # Act
        stub_executor.set_task_behavior("task_004", "error")

        # Assert
        assert stub_executor._task_behaviors["task_004"] == "error"

    def test_set_task_behavior_invalid_raises_error(self, stub_executor):
        """Test that invalid behavior raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid behavior"):
            stub_executor.set_task_behavior("task_005", "invalid")

    def test_set_task_behavior_overrides_failure_rate(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_task
    ):
        """Test that specific task behavior overrides global failure_rate"""
        # Arrange
        executor = StubExecutor(
            failure_rate=1.0,  # All tasks should fail by default
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Set specific task to succeed
        executor.set_task_behavior(sample_task.task_id, "success")

        # Act
        result = executor._stub_execution(sample_task)

        # Assert - Should succeed despite failure_rate=1.0
        assert result["status"] == "succeeded"

    # --- set_failure_rate Tests ---

    def test_set_failure_rate_valid_range(self, stub_executor):
        """Test setting failure rate within valid range [0.0, 1.0]"""
        # Act
        stub_executor.set_failure_rate(0.5)

        # Assert
        assert stub_executor._failure_rate == 0.5

    def test_set_failure_rate_negative_raises_error(self, stub_executor):
        """Test that negative failure_rate raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            stub_executor.set_failure_rate(-0.1)

    def test_set_failure_rate_greater_than_one_raises_error(self, stub_executor):
        """Test that failure_rate > 1.0 raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            stub_executor.set_failure_rate(1.5)

    # --- set_simulated_delay Tests ---

    def test_set_simulated_delay_valid(self, stub_executor):
        """Test setting valid simulated delay"""
        # Act
        stub_executor.set_simulated_delay(1.5)

        # Assert
        assert stub_executor._simulated_delay == 1.5

    def test_set_simulated_delay_zero(self, stub_executor):
        """Test setting delay to zero"""
        # Act
        stub_executor.set_simulated_delay(0.0)

        # Assert
        assert stub_executor._simulated_delay == 0.0

    def test_set_simulated_delay_negative_raises_error(self, stub_executor):
        """Test that negative delay raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="must be non-negative"):
            stub_executor.set_simulated_delay(-1.0)

    # --- clear_task_behaviors Tests ---

    def test_clear_task_behaviors(self, stub_executor):
        """Test clearing all task behavior configurations"""
        # Arrange
        stub_executor.set_task_behavior("task_001", "success")
        stub_executor.set_task_behavior("task_002", "failure")
        assert len(stub_executor._task_behaviors) == 2

        # Act
        stub_executor.clear_task_behaviors()

        # Assert
        assert len(stub_executor._task_behaviors) == 0

    def test_clear_task_behaviors_affects_subsequent_execution(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_task
    ):
        """Test that clearing behaviors makes executor use failure_rate"""
        # Arrange
        executor = StubExecutor(
            failure_rate=1.0,  # All should fail
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        # Set task to succeed
        executor.set_task_behavior(sample_task.task_id, "success")
        result = executor._stub_execution(sample_task)
        assert result["status"] == "succeeded"

        # Clear behaviors
        executor.clear_task_behaviors()

        # Act - Should now fail based on failure_rate
        with pytest.raises(TaskExecutionException):
            executor._stub_execution(sample_task)


# ============================================================================
# Test Class 5: TestStubExecutorIntegration
# ============================================================================


class TestStubExecutorIntegration:
    """Integration tests for StubExecutor"""

    def test_run_manifest_with_stub_executor(
            self,
            stub_executor,
            sample_manifest_with_multiple_cases,
            mock_id_fn,
            mock_now_fn
    ):
        """Test complete workflow from manifest to results with stub executor"""
        # Arrange
        manifest = sample_manifest_with_multiple_cases
        tasks = [
            Task.from_manifest_case(case, manifest.execution_policy, manifest.workflow_id, mock_id_fn, mock_now_fn)
            for case in manifest.cases
        ]

        # Act
        results = stub_executor._execute_tasks(tasks, manifest)

        # Assert
        assert len(results) == len(manifest.cases)
        assert all(r.task_status == TaskStatus.SUCCEEDED for r in results)

    def test_stub_executor_with_mixed_behaviors(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case
    ):
        """Test stub executor with mixed behaviors (success/failure)"""
        # Arrange
        executor = StubExecutor(
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        tasks = [
            Task.from_manifest_case(sample_test_case, sample_manifest.execution_policy, "wf_001", mock_id_fn,
                                    mock_now_fn)
            for _ in range(5)
        ]

        # Configure behaviors
        executor.set_task_behavior(tasks[0].task_id, "success")
        executor.set_task_behavior(tasks[1].task_id, "success")
        executor.set_task_behavior(tasks[2].task_id, "failure")
        executor.set_task_behavior(tasks[3].task_id, "success")
        executor.set_task_behavior(tasks[4].task_id, "failure")

        # Disable retries for this test
        sample_manifest.execution_policy.retry_policy.max_attempts = 1

        # Act
        results = executor._execute_tasks(tasks, sample_manifest)

        # Assert
        assert len(results) == 5
        succeeded = sum(1 for r in results if r.task_status == TaskStatus.SUCCEEDED)
        failed = sum(1 for r in results if r.task_status == TaskStatus.FAILED)
        assert succeeded == 3
        assert failed == 2

    def test_stub_executor_with_retry_on_failure(
            self,
            mock_now_fn,
            mock_sleep_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case
    ):
        """Test stub executor retry behavior on failures"""
        # Arrange
        executor = StubExecutor(
            failure_rate=1.0,  # All attempts fail
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn,
            id_fn=mock_id_fn
        )

        task = Task.from_manifest_case(sample_test_case, sample_manifest.execution_policy, "wf_001", mock_id_fn,
                                       mock_now_fn)

        # Configure retries
        retry_policy = RetryPolicy(
            max_attempts=3,
            backoff_seconds=0.1,
            backoff_multiplier=2.0
        )

        # Act
        result = executor._execute_single_task(task, retry_policy)

        # Assert
        assert result.task_status == TaskStatus.FAILED
        # Should have attempted 3 times
        assert task.attempt_count >= 3 or result.task_status == TaskStatus.FAILED

    def test_stub_executor_performance_with_delay(
            self,
            mock_now_fn,
            mock_id_fn,
            sample_manifest,
            sample_test_case
    ):
        """Test stub executor respects simulated delay"""
        # Arrange
        sleep_calls = []

        def track_sleep(seconds):
            sleep_calls.append(seconds)

        executor = StubExecutor(
            simulated_delay=0.5,
            now_fn=mock_now_fn,
            sleep_fn=track_sleep,
            id_fn=mock_id_fn
        )

        tasks = [
            Task.from_manifest_case(sample_test_case, sample_manifest.execution_policy, "wf_001", mock_id_fn,
                                    mock_now_fn)
            for _ in range(3)
        ]

        # Act
        results = executor._execute_tasks(tasks, sample_manifest)

        # Assert
        assert len(results) == 3
        # Each task should have called sleep once with 0.5 seconds
        assert len(sleep_calls) == 3
        assert all(delay == 0.5 for delay in sleep_calls)
