"""
Unit tests for ExecutorService

Date: 2025-11-14
Author: backend-developer
Description: 测试 ExecutorService 的集成功能和高层 API
"""

import unittest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from src.config.models import RunManifest, ExecutionPolicy, TestCase
from src.collector.models import TestResult, TestStatus
from src.executor.executor_service import ExecutorService
from src.executor.models import (
    CancellationToken,
    TaskResult,
    TaskStatus as TaskStatusEnum,
    RunExecutionResult,
    RunStatistics
)


class TestExecutorService(unittest.TestCase):
    """测试 ExecutorService 类。"""

    def setUp(self) -> None:
        """测试前置准备。"""
        self.timestamp = datetime(2025, 11, 14, 10, 0, 0)
        self.mock_now = MagicMock(return_value=self.timestamp)
        self.mock_sleep = MagicMock()
        self.mock_id = MagicMock(side_effect=[f"id_{i}" for i in range(200)])  # Increased for large batch test

    def _create_test_manifest(self, num_cases: int = 1) -> RunManifest:
        """创建测试用的 RunManifest。

        Args:
            num_cases: 测试用例数量

        Returns:
            RunManifest: 运行清单
        """
        test_cases = [
            TestCase(
                workflow_id="wf_test",
                dataset=f"dataset_{i}",
                scenario="normal",
                parameters={"query": f"test_{i}"},
                conversation_flow=None,
                prompt_variant=None
            )
            for i in range(num_cases)
        ]

        retry_policy = MagicMock()
        retry_policy.max_attempts = 1
        retry_policy.backoff_seconds = 0.0

        execution_policy = MagicMock()
        execution_policy.concurrency = 4
        execution_policy.batch_size = 10
        execution_policy.stop_conditions = {}
        execution_policy.backoff_seconds = 0.0
        execution_policy.rate_control = None
        execution_policy.retry_policy = retry_policy

        rate_limits = MagicMock()
        evaluator = MagicMock()

        return RunManifest(
            workflow_id="wf_test",
            workflow_version="v1.0",
            prompt_variant=None,
            dsl_payload="test_dsl",
            cases=test_cases,
            execution_policy=execution_policy,
            rate_limits=rate_limits,
            evaluator=evaluator,
            metadata={}
        )

    def test_init_default_parameters(self) -> None:
        """测试使用默认参数初始化。"""
        # Act
        service = ExecutorService()

        # Assert
        self.assertIsNotNone(service.scheduler)

    def test_init_with_custom_functions(self) -> None:
        """测试使用自定义函数初始化。"""
        # Arrange
        mock_exec_func = MagicMock()

        # Act
        service = ExecutorService(
            task_execution_func=mock_exec_func,
            now_fn=self.mock_now,
            sleep_fn=self.mock_sleep,
            id_fn=self.mock_id
        )

        # Assert
        self.assertIsNotNone(service.scheduler)

    def test_execute_test_plan_success(self) -> None:
        """测试成功执行测试计划。"""
        # Arrange
        manifest = self._create_test_manifest(num_cases=2)
        service = ExecutorService(
            now_fn=self.mock_now,
            sleep_fn=self.mock_sleep,
            id_fn=self.mock_id
        )

        # Act
        test_results = service.execute_test_plan(manifest)

        # Assert
        self.assertEqual(len(test_results), 2)
        for test_result in test_results:
            self.assertIsInstance(test_result, TestResult)
            self.assertEqual(test_result.workflow_id, "wf_test")
            # StubExecutor 默认返回 SUCCESS
            self.assertEqual(test_result.status, TestStatus.SUCCESS)

    def test_execute_test_plan_with_cancellation_token(self) -> None:
        """测试使用取消令牌执行测试计划。"""
        # Arrange
        manifest = self._create_test_manifest(num_cases=5)
        token = CancellationToken()
        service = ExecutorService(
            now_fn=self.mock_now,
            sleep_fn=self.mock_sleep,
            id_fn=self.mock_id
        )

        # Act
        test_results = service.execute_test_plan(manifest, token)

        # Assert
        self.assertEqual(len(test_results), 5)

    def test_execute_test_plan_cancel_during_execution(self) -> None:
        """测试执行过程中取消任务。"""
        # Arrange
        manifest = self._create_test_manifest(num_cases=10)
        token = CancellationToken()

        # 创建自定义执行函数,在第3个任务后取消
        call_count = [0]

        def custom_exec_func(task: Any) -> TestResult:
            call_count[0] += 1
            if call_count[0] == 3:
                token.cancel()
            return TestResult(
                workflow_id=task.workflow_id,
                execution_id=task.task_id,
                timestamp=self.timestamp,
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=50,
                cost=0.001,
                inputs=task.parameters,
                outputs={"result": "success"}
            )

        service = ExecutorService(
            task_execution_func=custom_exec_func,
            now_fn=self.mock_now,
            sleep_fn=self.mock_sleep,
            id_fn=self.mock_id
        )

        # Act
        test_results = service.execute_test_plan(manifest, token)

        # Assert
        # 由于是批量执行,可能会有部分任务完成,部分被取消
        self.assertGreater(len(test_results), 0)
        self.assertLessEqual(len(test_results), 10)

    def test_execute_test_plan_none_manifest_raises_error(self) -> None:
        """测试传入 None manifest 时抛出异常。"""
        # Arrange
        service = ExecutorService()

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            service.execute_test_plan(None)

        self.assertIn("manifest cannot be None", str(context.exception))

    def test_execute_test_plan_empty_test_cases(self) -> None:
        """测试执行空测试用例列表(应抛出异常)。"""
        # Arrange
        manifest = self._create_test_manifest(num_cases=0)
        service = ExecutorService(
            now_fn=self.mock_now,
            sleep_fn=self.mock_sleep,
            id_fn=self.mock_id
        )

        # Act & Assert
        with self.assertRaises(Exception):  # ExecutorException
            service.execute_test_plan(manifest)

    def test_scheduler_property(self) -> None:
        """测试 scheduler 属性访问。"""
        # Arrange
        service = ExecutorService()

        # Act
        scheduler = service.scheduler

        # Assert
        self.assertIsNotNone(scheduler)
        from src.executor.task_scheduler import TaskScheduler
        self.assertIsInstance(scheduler, TaskScheduler)

    def test_execute_test_plan_integration_with_result_converter(self) -> None:
        """测试 ExecutorService 与 ResultConverter 的集成。"""
        # Arrange
        manifest = self._create_test_manifest(num_cases=3)
        service = ExecutorService(
            now_fn=self.mock_now,
            sleep_fn=self.mock_sleep,
            id_fn=self.mock_id
        )

        # Act
        test_results = service.execute_test_plan(manifest)

        # Assert - verify conversion works and returns TestResult objects
        self.assertEqual(len(test_results), 3)
        for result in test_results:
            # Verify it's a TestResult from collector module
            self.assertIsInstance(result, TestResult)
            self.assertEqual(result.workflow_id, "wf_test")
            # StubExecutor returns SUCCESS by default
            self.assertEqual(result.status, TestStatus.SUCCESS)
            # Verify all required fields are present
            self.assertIsNotNone(result.execution_id)
            self.assertIsNotNone(result.timestamp)
            self.assertGreaterEqual(result.execution_time, 0.0)
            self.assertGreaterEqual(result.tokens_used, 0)
            self.assertGreaterEqual(result.cost, 0.0)

    @patch('src.executor.executor_service.TaskScheduler')
    def test_execute_test_plan_calls_scheduler_correctly(
            self,
            mock_scheduler_class: MagicMock
    ) -> None:
        """测试 execute_test_plan 正确调用调度器。"""
        # Arrange
        manifest = self._create_test_manifest(num_cases=2)

        # 模拟 RunExecutionResult
        mock_task_results = [
            TaskResult(
                task_id=f"task_{i}",
                workflow_id="wf_test",
                dataset=f"dataset_{i}",
                scenario="normal",
                task_status=TaskStatusEnum.SUCCEEDED,
                test_result=TestResult(
                    workflow_id="wf_test",
                    execution_id=f"task_{i}",
                    timestamp=self.timestamp,
                    status=TestStatus.SUCCESS,
                    execution_time=1.0,
                    tokens_used=50,
                    cost=0.001,
                    inputs={},
                    outputs={}
                ),
                attempt_count=1,
                created_at=self.timestamp,
                finished_at=self.timestamp,
                execution_time=1.0
            )
            for i in range(2)
        ]

        mock_run_result = RunExecutionResult(
            run_id="run_001",
            workflow_id="wf_test",
            started_at=self.timestamp,
            finished_at=self.timestamp,
            total_duration=2.0,
            task_results=mock_task_results,
            statistics=RunStatistics()
        )

        mock_scheduler_instance = MagicMock()
        mock_scheduler_instance.run_manifest.return_value = mock_run_result
        mock_scheduler_class.return_value = mock_scheduler_instance

        service = ExecutorService(
            now_fn=self.mock_now,
            sleep_fn=self.mock_sleep,
            id_fn=self.mock_id
        )

        # Act
        test_results = service.execute_test_plan(manifest)

        # Assert
        mock_scheduler_instance.run_manifest.assert_called_once_with(manifest, None)
        self.assertEqual(len(test_results), 2)

    def test_execute_test_plan_large_batch(self) -> None:
        """测试执行大批量测试。"""
        # Arrange
        manifest = self._create_test_manifest(num_cases=100)
        service = ExecutorService(
            now_fn=self.mock_now,
            sleep_fn=self.mock_sleep,
            id_fn=self.mock_id
        )

        # Act
        test_results = service.execute_test_plan(manifest)

        # Assert
        self.assertEqual(len(test_results), 100)
        for test_result in test_results:
            self.assertEqual(test_result.workflow_id, "wf_test")


if __name__ == '__main__':
    unittest.main()
