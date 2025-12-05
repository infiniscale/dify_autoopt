"""
Executor Module - Executor Service

Date: 2025-11-14
Author: backend-developer
Description: 执行器服务,提供统一的测试执行入口
"""

import time
from datetime import datetime
from typing import Callable, List, Optional
from uuid import uuid4

from src.config.models import RunManifest
from src.collector.models import TestResult

from .task_scheduler import TaskScheduler
from .result_converter import ResultConverter
from .models import CancellationToken


class ExecutorService:
    """执行器服务(统一入口点)。

    整合 TaskScheduler 和 ResultConverter,提供高层 API。

    流程:
        RunManifest → TaskScheduler.run_manifest()
        → RunExecutionResult → TaskResults → ResultConverter.convert_batch()
        → TestResults

    Attributes:
        _scheduler: 任务调度器

    Example:
        >>> service = ExecutorService()
        >>> manifest = RunManifest(...)
        >>> test_results = service.execute_test_plan(manifest)
        >>> print(f"Executed {len(test_results)} tests")
    """

    def __init__(
        self,
        task_execution_func=None,
        now_fn: Callable[[], datetime] = datetime.now,
        sleep_fn: Callable[[float], None] = time.sleep,
        id_fn: Callable[[], str] = lambda: str(uuid4())
    ) -> None:
        """初始化执行器服务。

        Args:
            task_execution_func: 任务执行函数(可选,默认使用 StubExecutor)
            now_fn: 时间获取函数(用于依赖注入和测试)
            sleep_fn: 休眠函数(用于依赖注入和测试)
            id_fn: ID生成函数(用于依赖注入和测试)

        Notes:
            - 所有函数参数都支持依赖注入,便于单元测试
            - task_execution_func 为 None 时,调度器会使用 StubExecutor
        """
        self._scheduler = TaskScheduler(
            task_execution_func=task_execution_func,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            id_fn=id_fn
        )

    def execute_test_plan(
        self,
        manifest: RunManifest,
        cancellation_token: Optional[CancellationToken] = None
    ) -> List[TestResult]:
        """执行测试计划(主入口方法)。

        Args:
            manifest: 运行清单
            cancellation_token: 取消令牌(可选)

        Returns:
            List[TestResult]: 测试结果列表

        Raises:
            ExecutorException: 当执行失败时
            SchedulerException: 当调度失败时
            ValueError: 当 manifest 为 None 或无效时

        Notes:
            - 此方法会阻塞直到所有测试执行完成或被取消
            - 返回的 TestResult 列表顺序与任务执行顺序一致
            - 如果所有任务都被取消,返回空列表

        Example:
            >>> service = ExecutorService()
            >>> manifest = RunManifest(...)
            >>> token = CancellationToken()
            >>> results = service.execute_test_plan(manifest, token)
        """
        if manifest is None:
            raise ValueError("manifest cannot be None")

        # 1. Execute tasks via scheduler
        run_result = self._scheduler.run_manifest(manifest, cancellation_token)

        # 2. Convert TaskResults to TestResults
        test_results = ResultConverter.convert_batch(run_result.task_results)

        return test_results

    @property
    def scheduler(self) -> TaskScheduler:
        """获取内部调度器(用于测试和高级用法)。

        Returns:
            TaskScheduler: 任务调度器实例

        Notes:
            - 此属性主要用于测试和高级用法
            - 一般用户应使用 execute_test_plan 方法
            - 直接使用 scheduler 需要了解 TaskScheduler 的内部实现
        """
        return self._scheduler
