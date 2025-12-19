"""
Unit tests for ResultConverter

Date: 2025-11-14
Author: backend-developer
Description: 测试 ResultConverter 的状态映射和数据转换功能
"""

import unittest
from datetime import datetime

from src.collector.models import TestResult, TestStatus
from src.executor.models import TaskResult, TaskStatus as TaskStatusEnum
from src.executor.result_converter import ResultConverter


class TestResultConverter(unittest.TestCase):
    """测试 ResultConverter 类。"""

    def setUp(self) -> None:
        """测试前置准备。"""
        self.timestamp = datetime(2025, 11, 14, 10, 0, 0)

    def test_convert_succeeded_task(self) -> None:
        """测试转换成功的任务。"""
        # Arrange: 创建 SUCCEEDED 状态的 TaskResult(带 test_result)
        embedded_test_result = TestResult(
            workflow_id="wf_001",
            execution_id="task_001",
            timestamp=self.timestamp,
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=100,
            cost=0.002,
            inputs={"query": "test"},
            outputs={"result": "success"}
        )

        task_result = TaskResult(
            task_id="task_001",
            workflow_id="wf_001",
            dataset="test_dataset",
            scenario="normal",
            task_status=TaskStatusEnum.SUCCEEDED,
            test_result=embedded_test_result,
            error_message=None,
            attempt_count=1,
            created_at=self.timestamp,
            finished_at=self.timestamp,
            execution_time=1.5
        )

        # Act
        test_result = ResultConverter.convert(task_result)

        # Assert
        self.assertEqual(test_result.workflow_id, "wf_001")
        self.assertEqual(test_result.execution_id, "task_001")
        self.assertEqual(test_result.status, TestStatus.SUCCESS)
        self.assertEqual(test_result.execution_time, 1.5)
        self.assertEqual(test_result.tokens_used, 100)
        self.assertEqual(test_result.cost, 0.002)
        self.assertEqual(test_result.inputs, {"query": "test"})
        self.assertEqual(test_result.outputs, {"result": "success"})
        self.assertIsNone(test_result.error_message)
        self.assertEqual(test_result.dataset, "test_dataset")
        self.assertEqual(test_result.metadata["scenario"], "normal")
        self.assertEqual(test_result.metadata["attempt_count"], 1)

    def test_convert_failed_task(self) -> None:
        """测试转换失败的任务。"""
        # Arrange
        task_result = TaskResult(
            task_id="task_002",
            workflow_id="wf_001",
            dataset="test_dataset",
            scenario="error",
            task_status=TaskStatusEnum.FAILED,
            test_result=None,
            error_message="Business validation failed",
            attempt_count=2,
            created_at=self.timestamp,
            finished_at=self.timestamp,
            execution_time=0.5
        )

        # Act
        test_result = ResultConverter.convert(task_result)

        # Assert
        self.assertEqual(test_result.status, TestStatus.FAILED)
        self.assertEqual(test_result.error_message, "Business validation failed")
        self.assertEqual(test_result.tokens_used, 0)
        self.assertEqual(test_result.cost, 0.0)
        self.assertEqual(test_result.inputs, {})
        self.assertEqual(test_result.outputs, {})
        self.assertEqual(test_result.metadata["attempt_count"], 2)

    def test_convert_timeout_task(self) -> None:
        """测试转换超时的任务。"""
        # Arrange
        task_result = TaskResult(
            task_id="task_003",
            workflow_id="wf_001",
            dataset="test_dataset",
            scenario="boundary",
            task_status=TaskStatusEnum.TIMEOUT,
            test_result=None,
            error_message="Task execution timeout",
            attempt_count=3,
            created_at=self.timestamp,
            finished_at=self.timestamp,
            execution_time=300.0
        )

        # Act
        test_result = ResultConverter.convert(task_result)

        # Assert
        self.assertEqual(test_result.status, TestStatus.TIMEOUT)
        self.assertEqual(test_result.error_message, "Task execution timeout")
        self.assertEqual(test_result.execution_time, 300.0)

    def test_convert_error_task(self) -> None:
        """测试转换错误的任务。"""
        # Arrange
        task_result = TaskResult(
            task_id="task_004",
            workflow_id="wf_001",
            dataset="test_dataset",
            scenario="error",
            task_status=TaskStatusEnum.ERROR,
            test_result=None,
            error_message="System exception occurred",
            attempt_count=1,
            created_at=self.timestamp,
            finished_at=self.timestamp,
            execution_time=0.1
        )

        # Act
        test_result = ResultConverter.convert(task_result)

        # Assert
        self.assertEqual(test_result.status, TestStatus.ERROR)
        self.assertEqual(test_result.error_message, "System exception occurred")

    def test_convert_cancelled_task(self) -> None:
        """测试转换取消的任务(应映射为 ERROR)。"""
        # Arrange
        task_result = TaskResult(
            task_id="task_005",
            workflow_id="wf_001",
            dataset="test_dataset",
            scenario="normal",
            task_status=TaskStatusEnum.CANCELLED,
            test_result=None,
            error_message="Task cancelled by user",
            attempt_count=1,
            created_at=self.timestamp,
            finished_at=self.timestamp,
            execution_time=0.0
        )

        # Act
        test_result = ResultConverter.convert(task_result)

        # Assert
        self.assertEqual(test_result.status, TestStatus.ERROR)
        self.assertEqual(test_result.error_message, "Task cancelled by user")

    def test_map_all_status_values(self) -> None:
        """测试所有 TaskStatus 到 TestStatus 的映射。"""
        # Test all 8 TaskStatus values
        test_cases = [
            (TaskStatusEnum.SUCCEEDED, TestStatus.SUCCESS),
            (TaskStatusEnum.FAILED, TestStatus.FAILED),
            (TaskStatusEnum.TIMEOUT, TestStatus.TIMEOUT),
            (TaskStatusEnum.ERROR, TestStatus.ERROR),
            (TaskStatusEnum.CANCELLED, TestStatus.ERROR),
            (TaskStatusEnum.PENDING, TestStatus.ERROR),
            (TaskStatusEnum.QUEUED, TestStatus.ERROR),
            (TaskStatusEnum.RUNNING, TestStatus.ERROR),
        ]

        for task_status, expected_test_status in test_cases:
            with self.subTest(task_status=task_status):
                result = ResultConverter._map_status(task_status)
                self.assertEqual(result, expected_test_status)

    def test_convert_batch_empty_list(self) -> None:
        """测试批量转换空列表。"""
        # Act
        results = ResultConverter.convert_batch([])

        # Assert
        self.assertEqual(results, [])

    def test_convert_batch_multiple_results(self) -> None:
        """测试批量转换多个结果。"""
        # Arrange
        task_results = [
            TaskResult(
                task_id=f"task_{i}",
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                task_status=TaskStatusEnum.SUCCEEDED,
                test_result=TestResult(
                    workflow_id="wf_001",
                    execution_id=f"task_{i}",
                    timestamp=self.timestamp,
                    status=TestStatus.SUCCESS,
                    execution_time=1.0,
                    tokens_used=50,
                    cost=0.001,
                    inputs={},
                    outputs={}
                ),
                error_message=None,
                attempt_count=1,
                created_at=self.timestamp,
                finished_at=self.timestamp,
                execution_time=1.0
            )
            for i in range(5)
        ]

        # Act
        test_results = ResultConverter.convert_batch(task_results)

        # Assert
        self.assertEqual(len(test_results), 5)
        for i, test_result in enumerate(test_results):
            self.assertEqual(test_result.execution_id, f"task_{i}")
            self.assertEqual(test_result.status, TestStatus.SUCCESS)

    def test_convert_batch_mixed_statuses(self) -> None:
        """测试批量转换混合状态的结果。"""
        # Arrange
        task_results = [
            TaskResult(
                task_id="task_1",
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                task_status=TaskStatusEnum.SUCCEEDED,
                test_result=None,
                attempt_count=1,
                created_at=self.timestamp,
                finished_at=self.timestamp,
                execution_time=1.0
            ),
            TaskResult(
                task_id="task_2",
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="error",
                task_status=TaskStatusEnum.FAILED,
                test_result=None,
                error_message="Failed",
                attempt_count=2,
                created_at=self.timestamp,
                finished_at=self.timestamp,
                execution_time=0.5
            ),
            TaskResult(
                task_id="task_3",
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="boundary",
                task_status=TaskStatusEnum.TIMEOUT,
                test_result=None,
                error_message="Timeout",
                attempt_count=3,
                created_at=self.timestamp,
                finished_at=self.timestamp,
                execution_time=300.0
            ),
        ]

        # Act
        test_results = ResultConverter.convert_batch(task_results)

        # Assert
        self.assertEqual(len(test_results), 3)
        self.assertEqual(test_results[0].status, TestStatus.SUCCESS)
        self.assertEqual(test_results[1].status, TestStatus.FAILED)
        self.assertEqual(test_results[2].status, TestStatus.TIMEOUT)

    def test_convert_none_task_result_raises_error(self) -> None:
        """测试转换 None 时抛出异常。"""
        with self.assertRaises(ValueError) as context:
            ResultConverter.convert(None)

        self.assertIn("task_result cannot be None", str(context.exception))

    def test_convert_batch_none_list_raises_error(self) -> None:
        """测试批量转换 None 列表时抛出异常。"""
        with self.assertRaises(ValueError) as context:
            ResultConverter.convert_batch(None)

        self.assertIn("task_results cannot be None", str(context.exception))

    def test_convert_preserves_metadata(self) -> None:
        """测试转换过程保留元数据。"""
        # Arrange
        task_result = TaskResult(
            task_id="task_001",
            workflow_id="wf_001",
            dataset="production_dataset",
            scenario="custom",
            task_status=TaskStatusEnum.SUCCEEDED,
            test_result=None,
            attempt_count=1,
            created_at=self.timestamp,
            finished_at=self.timestamp,
            execution_time=2.5
        )

        # Act
        test_result = ResultConverter.convert(task_result)

        # Assert
        self.assertEqual(test_result.metadata["dataset"], "production_dataset")
        self.assertEqual(test_result.metadata["scenario"], "custom")
        self.assertEqual(test_result.metadata["attempt_count"], 1)


if __name__ == '__main__':
    unittest.main()
