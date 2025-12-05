"""
Executor Module - Result Converter

Date: 2025-11-14
Author: backend-developer
Description: 将执行器的 TaskResult 转换为 Collector 的 TestResult
"""

from typing import List

from src.collector.models import TestResult, TestStatus
from .models import TaskResult, TaskStatus as TaskStatusEnum


class ResultConverter:
    """结果转换器(静态工具类)。

    负责将 Executor 的 TaskResult 转换为 Collector 的 TestResult。

    状态映射:
        TaskStatus.SUCCEEDED  → TestStatus.SUCCESS
        TaskStatus.FAILED     → TestStatus.FAILED
        TaskStatus.TIMEOUT    → TestStatus.TIMEOUT
        TaskStatus.ERROR      → TestStatus.ERROR
        TaskStatus.CANCELLED  → TestStatus.ERROR
        TaskStatus.PENDING    → TestStatus.ERROR
        TaskStatus.QUEUED     → TestStatus.ERROR
        TaskStatus.RUNNING    → TestStatus.ERROR
    """

    @staticmethod
    def convert(task_result: TaskResult) -> TestResult:
        """将 TaskResult 转换为 TestResult。

        Args:
            task_result: 执行器的任务结果

        Returns:
            TestResult: Collector 的测试结果

        Raises:
            ValueError: 当 task_result 为 None 时

        Implementation:
            1. 映射状态(TaskStatus → TestStatus)
            2. 提取基本字段(workflow_id, execution_time, etc)
            3. 提取 tokens 和 cost(从 test_result 或默认值)
            4. 提取 error_message(从 task_result)
            5. 构建 TestResult
        """
        if task_result is None:
            raise ValueError("task_result cannot be None")

        # Map status
        test_status = ResultConverter._map_status(task_result.task_status)

        # Extract tokens and cost from embedded test_result if available
        tokens_used = 0
        cost = 0.0
        outputs = {}
        inputs = {}

        if task_result.test_result:
            tokens_used = task_result.test_result.tokens_used
            cost = task_result.test_result.cost
            outputs = task_result.test_result.outputs
            inputs = task_result.test_result.inputs

        # Extract error message
        error_message = task_result.error_message

        # Build metadata (preserve dataset and scenario info)
        metadata = {
            'dataset': task_result.dataset,
            'scenario': task_result.scenario,
            'attempt_count': task_result.attempt_count,
        }

        # Extract prompt_variant from metadata if available
        prompt_variant = task_result.metadata.get('prompt_variant') if task_result.metadata else None

        return TestResult(
            workflow_id=task_result.workflow_id,
            execution_id=task_result.task_id,
            timestamp=task_result.created_at,
            status=test_status,
            execution_time=task_result.execution_time,
            tokens_used=tokens_used,
            cost=cost,
            inputs=inputs,
            outputs=outputs,
            error_message=error_message,
            prompt_variant=prompt_variant,
            dataset=task_result.dataset,
            metadata=metadata
        )

    @staticmethod
    def _map_status(task_status: TaskStatusEnum) -> TestStatus:
        """映射 TaskStatus 到 TestStatus。

        Args:
            task_status: 任务状态

        Returns:
            TestStatus: 测试状态

        Notes:
            - SUCCEEDED → SUCCESS (唯一成功状态)
            - FAILED → FAILED (业务失败)
            - TIMEOUT → TIMEOUT (超时)
            - ERROR → ERROR (系统错误)
            - CANCELLED → ERROR (取消视为错误)
            - PENDING/QUEUED/RUNNING → ERROR (未完成状态视为错误)
        """
        mapping = {
            TaskStatusEnum.SUCCEEDED: TestStatus.SUCCESS,
            TaskStatusEnum.FAILED: TestStatus.FAILED,
            TaskStatusEnum.TIMEOUT: TestStatus.TIMEOUT,
            TaskStatusEnum.ERROR: TestStatus.ERROR,
            TaskStatusEnum.CANCELLED: TestStatus.ERROR,
            TaskStatusEnum.PENDING: TestStatus.ERROR,
            TaskStatusEnum.QUEUED: TestStatus.ERROR,
            TaskStatusEnum.RUNNING: TestStatus.ERROR,
        }
        return mapping.get(task_status, TestStatus.ERROR)

    @staticmethod
    def convert_batch(task_results: List[TaskResult]) -> List[TestResult]:
        """批量转换 TaskResult 到 TestResult。

        Args:
            task_results: 任务结果列表

        Returns:
            List[TestResult]: 测试结果列表

        Raises:
            ValueError: 当 task_results 为 None 时

        Notes:
            - 空列表返回空列表(不抛出异常)
            - 批量转换过程中任何单个转换失败会传播异常
        """
        if task_results is None:
            raise ValueError("task_results cannot be None")

        return [ResultConverter.convert(tr) for tr in task_results]
