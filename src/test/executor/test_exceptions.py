"""
Unit Tests for Executor Module Exceptions

Date: 2025-11-14
Author: qa-engineer
Description: Tests for ExecutorException and its subclasses
Coverage Target: 100%
"""

import pytest

from src.utils.exceptions import (
    ExecutorException,
    TaskExecutionException,
    TaskTimeoutException,
    SchedulerException,
    RateLimitException
)


# ============================================================================
# Test ExecutorException (Base Class)
# ============================================================================


class TestExecutorException:
    """Test suite for ExecutorException base class"""

    def test_init_with_message(self):
        """Test initialization with message only"""
        # Arrange & Act
        exc = ExecutorException("Test error message")

        # Assert
        assert str(exc) == "Test error message"
        assert exc.args == ("Test error message",)

    def test_init_without_message(self):
        """Test initialization without message"""
        # Arrange & Act
        exc = ExecutorException()

        # Assert
        assert str(exc) == ""
        assert exc.args == ()

    def test_can_be_raised(self):
        """Test exception can be raised and caught"""
        # Arrange & Act & Assert
        with pytest.raises(ExecutorException) as exc_info:
            raise ExecutorException("Test exception")

        assert str(exc_info.value) == "Test exception"

    def test_can_be_caught_as_base_exception(self):
        """Test ExecutorException can be caught as Exception"""
        # Arrange
        caught = False

        # Act
        try:
            raise ExecutorException("Test")
        except Exception as e:
            caught = True
            assert isinstance(e, ExecutorException)

        # Assert
        assert caught

    def test_multiple_args(self):
        """Test exception with multiple arguments"""
        # Arrange & Act
        exc = ExecutorException("Error", "Additional info")

        # Assert
        assert exc.args == ("Error", "Additional info")


# ============================================================================
# Test TaskExecutionException
# ============================================================================


class TestTaskExecutionException:
    """Test suite for TaskExecutionException"""

    def test_init_with_message(self):
        """Test initialization with message"""
        # Arrange & Act
        exc = TaskExecutionException("Task execution failed")

        # Assert
        assert str(exc) == "Task execution failed"
        assert isinstance(exc, ExecutorException)

    def test_inheritance_chain(self):
        """Test correct inheritance chain"""
        # Arrange & Act
        exc = TaskExecutionException("Test")

        # Assert
        assert isinstance(exc, TaskExecutionException)
        assert isinstance(exc, ExecutorException)
        assert isinstance(exc, Exception)

    def test_can_be_caught_specifically(self):
        """Test can be caught as TaskExecutionException"""
        # Arrange & Act & Assert
        with pytest.raises(TaskExecutionException) as exc_info:
            raise TaskExecutionException("Specific task error")

        assert str(exc_info.value) == "Specific task error"

    def test_can_be_caught_as_executor_exception(self):
        """Test can be caught as ExecutorException"""
        # Arrange
        caught = False

        # Act
        try:
            raise TaskExecutionException("Task error")
        except ExecutorException as e:
            caught = True
            assert isinstance(e, TaskExecutionException)

        # Assert
        assert caught

    def test_with_task_context(self):
        """Test with task context information"""
        # Arrange & Act
        task_id = "task_123"
        exc = TaskExecutionException(
            f"Task {task_id} failed: Invalid input"
        )

        # Assert
        assert "task_123" in str(exc)
        assert "Invalid input" in str(exc)


# ============================================================================
# Test TaskTimeoutException
# ============================================================================


class TestTaskTimeoutException:
    """Test suite for TaskTimeoutException"""

    def test_init_with_message(self):
        """Test initialization with message"""
        # Arrange & Act
        exc = TaskTimeoutException("Task timed out after 30 seconds")

        # Assert
        assert str(exc) == "Task timed out after 30 seconds"
        assert isinstance(exc, ExecutorException)

    def test_inheritance_chain(self):
        """Test correct inheritance chain"""
        # Arrange & Act
        exc = TaskTimeoutException("Timeout")

        # Assert
        assert isinstance(exc, TaskTimeoutException)
        assert isinstance(exc, ExecutorException)
        assert isinstance(exc, Exception)

    def test_can_be_caught_specifically(self):
        """Test can be caught as TaskTimeoutException"""
        # Arrange & Act & Assert
        with pytest.raises(TaskTimeoutException) as exc_info:
            raise TaskTimeoutException("Timeout occurred")

        assert str(exc_info.value) == "Timeout occurred"

    def test_with_timeout_details(self):
        """Test with timeout details"""
        # Arrange & Act
        task_id = "task_456"
        timeout = 30.0
        exc = TaskTimeoutException(
            f"Task {task_id} exceeded timeout of {timeout}s"
        )

        # Assert
        assert "task_456" in str(exc)
        assert "30.0" in str(exc)

    def test_distinguishable_from_task_execution_exception(self):
        """Test timeout exception is different from execution exception"""
        # Arrange
        timeout_exc = TaskTimeoutException("Timeout")
        exec_exc = TaskExecutionException("Failed")

        # Act & Assert
        assert type(timeout_exc) != type(exec_exc)
        assert isinstance(timeout_exc, ExecutorException)
        assert isinstance(exec_exc, ExecutorException)


# ============================================================================
# Test SchedulerException
# ============================================================================


class TestSchedulerException:
    """Test suite for SchedulerException"""

    def test_init_with_message(self):
        """Test initialization with message"""
        # Arrange & Act
        exc = SchedulerException("Scheduler error")

        # Assert
        assert str(exc) == "Scheduler error"
        assert isinstance(exc, ExecutorException)

    def test_inheritance_chain(self):
        """Test correct inheritance chain"""
        # Arrange & Act
        exc = SchedulerException("Test")

        # Assert
        assert isinstance(exc, SchedulerException)
        assert isinstance(exc, ExecutorException)
        assert isinstance(exc, Exception)

    def test_can_be_caught_specifically(self):
        """Test can be caught as SchedulerException"""
        # Arrange & Act & Assert
        with pytest.raises(SchedulerException) as exc_info:
            raise SchedulerException("Queue is full")

        assert str(exc_info.value) == "Queue is full"

    def test_with_scheduler_context(self):
        """Test with scheduler context"""
        # Arrange & Act
        exc = SchedulerException("Failed to schedule task: Queue capacity exceeded")

        # Assert
        assert "schedule" in str(exc)
        assert "Queue capacity" in str(exc)

    def test_init_without_message(self):
        """Test initialization without message"""
        # Arrange & Act
        exc = SchedulerException()

        # Assert
        assert str(exc) == ""


# ============================================================================
# Test RateLimitException
# ============================================================================


class TestRateLimitException:
    """Test suite for RateLimitException"""

    def test_init_with_message(self):
        """Test initialization with message"""
        # Arrange & Act
        exc = RateLimitException("Rate limit exceeded")

        # Assert
        assert str(exc) == "Rate limit exceeded"
        assert isinstance(exc, ExecutorException)

    def test_inheritance_chain(self):
        """Test correct inheritance chain"""
        # Arrange & Act
        exc = RateLimitException("Test")

        # Assert
        assert isinstance(exc, RateLimitException)
        assert isinstance(exc, ExecutorException)
        assert isinstance(exc, Exception)

    def test_can_be_caught_specifically(self):
        """Test can be caught as RateLimitException"""
        # Arrange & Act & Assert
        with pytest.raises(RateLimitException) as exc_info:
            raise RateLimitException("Too many requests")

        assert str(exc_info.value) == "Too many requests"

    def test_with_rate_limit_details(self):
        """Test with rate limit details"""
        # Arrange & Act
        retry_after = 5.0
        exc = RateLimitException(
            f"Rate limit exceeded. Retry after {retry_after} seconds"
        )

        # Assert
        assert "Rate limit exceeded" in str(exc)
        assert "5.0" in str(exc)

    def test_init_without_message(self):
        """Test initialization without message"""
        # Arrange & Act
        exc = RateLimitException()

        # Assert
        assert str(exc) == ""


# ============================================================================
# Test Exception Hierarchy and Catching
# ============================================================================


class TestExceptionHierarchy:
    """Test suite for exception inheritance and catching behavior"""

    def test_all_inherit_from_executor_exception(self):
        """Test all exceptions inherit from ExecutorException"""
        # Arrange
        exceptions = [
            TaskExecutionException("test"),
            TaskTimeoutException("test"),
            SchedulerException("test"),
            RateLimitException("test")
        ]

        # Act & Assert
        for exc in exceptions:
            assert isinstance(exc, ExecutorException)

    def test_all_inherit_from_base_exception(self):
        """Test all exceptions inherit from Exception"""
        # Arrange
        exceptions = [
            ExecutorException("test"),
            TaskExecutionException("test"),
            TaskTimeoutException("test"),
            SchedulerException("test"),
            RateLimitException("test")
        ]

        # Act & Assert
        for exc in exceptions:
            assert isinstance(exc, Exception)

    def test_can_catch_all_as_executor_exception(self):
        """Test can catch all specific exceptions as ExecutorException"""
        # Arrange
        caught_count = 0

        # Act
        for exc_class in [
            TaskExecutionException,
            TaskTimeoutException,
            SchedulerException,
            RateLimitException
        ]:
            try:
                raise exc_class("Test error")
            except ExecutorException:
                caught_count += 1

        # Assert
        assert caught_count == 4

    def test_specific_exceptions_are_distinct(self):
        """Test specific exception types are distinct"""
        # Arrange
        exc1 = TaskExecutionException("test")
        exc2 = TaskTimeoutException("test")
        exc3 = SchedulerException("test")
        exc4 = RateLimitException("test")

        # Act & Assert
        assert type(exc1) != type(exc2)
        assert type(exc1) != type(exc3)
        assert type(exc1) != type(exc4)
        assert type(exc2) != type(exc3)
        assert type(exc2) != type(exc4)
        assert type(exc3) != type(exc4)

    def test_catch_order_matters(self):
        """Test exception catching order is important"""
        # Arrange
        caught_as = None

        # Act
        try:
            raise TaskExecutionException("Test")
        except TaskExecutionException:
            caught_as = "specific"
        except ExecutorException:
            caught_as = "base"

        # Assert
        assert caught_as == "specific"

    def test_base_catch_works_for_all(self):
        """Test catching ExecutorException catches all subclasses"""
        # Arrange
        caught_types = []

        # Act
        for exc_class in [
            ExecutorException,
            TaskExecutionException,
            TaskTimeoutException,
            SchedulerException,
            RateLimitException
        ]:
            try:
                raise exc_class(f"Test {exc_class.__name__}")
            except ExecutorException as e:
                caught_types.append(type(e).__name__)

        # Assert
        assert len(caught_types) == 5
        assert "ExecutorException" in caught_types
        assert "TaskExecutionException" in caught_types
        assert "TaskTimeoutException" in caught_types
        assert "SchedulerException" in caught_types
        assert "RateLimitException" in caught_types
