"""
Executor Module - Abstract Base Executor

Date: 2025-11-14
Author: backend-developer
Description: 定义执行器抽象基类，提供任务执行的模板流程
"""

# 标准库
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional
from uuid import uuid4

# 项目模块
from src.config.models import RunManifest, TestCase, ExecutionPolicy
from src.utils.exceptions import ExecutorException, SchedulerException

from .models import (
    Task,
    TaskResult,
    TaskStatus,
    RunExecutionResult,
    CancellationToken
)


class ExecutorBase(ABC):
    """执行器抽象基类。

    提供任务执行的模板流程，子类需实现具体的调度和执行逻辑。

    模板流程：
        1. 验证 RunManifest
        2. 构建 Task 列表（from_manifest_case）
        3. 执行任务（_execute_tasks，子类实现）
        4. 聚合结果（from_task_results）

    Attributes:
        _now_fn: 时间获取函数（依赖注入）
        _sleep_fn: 休眠函数（依赖注入）
        _id_fn: ID生成函数（依赖注入）
    """

    def __init__(
        self,
        now_fn: Callable[[], datetime] = datetime.now,
        sleep_fn: Callable[[float], None] = time.sleep,
        id_fn: Callable[[], str] = lambda: str(uuid4())
    ) -> None:
        """初始化执行器。

        Args:
            now_fn: 时间获取函数（用于依赖注入和测试）
            sleep_fn: 休眠函数（用于依赖注入和测试）
            id_fn: ID生成函数（用于依赖注入和测试）
        """
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn
        self._id_fn = id_fn

    def run_manifest(
        self,
        manifest: RunManifest,
        cancellation_token: Optional[CancellationToken] = None
    ) -> RunExecutionResult:
        """执行 RunManifest 的主入口方法（模板方法）。

        执行流程：
            1. 验证 manifest 完整性
            2. 构建任务列表
            3. 调用子类的 _execute_tasks() 执行任务
            4. 聚合结果并返回

        Args:
            manifest: 运行清单，包含所有测试用例和执行策略
            cancellation_token: 取消令牌（可选）

        Returns:
            RunExecutionResult: 执行结果对象，包含所有任务结果和统计数据

        Raises:
            ExecutorException: 当 manifest 验证失败时
            SchedulerException: 当任务构建或执行失败时
        """
        # Step 1: 验证 RunManifest
        self._validate_manifest(manifest)

        # Step 2: 构建任务列表
        tasks = self._build_tasks(manifest)

        if not tasks:
            raise SchedulerException("No tasks to execute")

        # 记录开始时间
        started_at = self._now_fn()
        run_id = self._id_fn()

        # Step 3: 执行任务（由子类实现）
        task_results = self._execute_tasks(
            tasks=tasks,
            manifest=manifest,
            cancellation_token=cancellation_token
        )

        # 记录完成时间
        finished_at = self._now_fn()

        # Step 4: 聚合结果
        run_result = RunExecutionResult.from_task_results(
            run_id=run_id,
            workflow_id=manifest.workflow_id,
            started_at=started_at,
            finished_at=finished_at,
            task_results=task_results,
            metadata={
                "workflow_version": manifest.workflow_version,
                "prompt_variant": manifest.prompt_variant,
                "total_cases": len(manifest.cases),
                "execution_policy": {
                    "concurrency": manifest.execution_policy.concurrency,
                    "batch_size": manifest.execution_policy.batch_size,
                    "max_retries": manifest.execution_policy.retry_policy.max_attempts
                }
            }
        )

        return run_result

    def _validate_manifest(self, manifest: RunManifest) -> None:
        """验证 RunManifest 的完整性。

        Args:
            manifest: 运行清单对象

        Raises:
            ExecutorException: 当验证失败时
        """
        if manifest is None:
            raise ExecutorException("RunManifest cannot be None")

        if not manifest.workflow_id:
            raise ExecutorException("RunManifest.workflow_id cannot be empty")

        if not manifest.cases:
            raise ExecutorException("RunManifest.cases cannot be empty")

        if manifest.execution_policy is None:
            raise ExecutorException("RunManifest.execution_policy cannot be None")

        # 验证执行策略的关键参数
        policy = manifest.execution_policy
        if policy.concurrency < 1:
            raise ExecutorException(
                f"Invalid concurrency: {policy.concurrency}, must be >= 1"
            )
        if policy.batch_size < 1:
            raise ExecutorException(
                f"Invalid batch_size: {policy.batch_size}, must be >= 1"
            )

    def _build_tasks(self, manifest: RunManifest) -> List[Task]:
        """从 RunManifest 构建任务列表（默认实现）。

        子类可以覆盖此方法以实现自定义的任务构建逻辑。

        Args:
            manifest: 运行清单对象

        Returns:
            List[Task]: 任务对象列表

        Raises:
            SchedulerException: 当任务构建失败时
        """
        tasks: List[Task] = []

        try:
            for test_case in manifest.cases:
                task = Task.from_manifest_case(
                    test_case=test_case,
                    execution_policy=manifest.execution_policy,
                    workflow_id=manifest.workflow_id,
                    id_fn=self._id_fn,
                    now_fn=self._now_fn
                )
                tasks.append(task)
        except Exception as e:
            raise SchedulerException(f"Failed to build tasks: {str(e)}") from e

        return tasks

    @abstractmethod
    def _execute_tasks(
        self,
        tasks: List[Task],
        manifest: RunManifest,
        cancellation_token: Optional[CancellationToken] = None
    ) -> List[TaskResult]:
        """执行任务列表（抽象方法，由子类实现）。

        子类需要实现具体的调度策略，例如：
        - 串行执行（SequentialExecutor）
        - 并行执行（ConcurrentExecutor）
        - 批量执行（BatchExecutor）

        Args:
            tasks: 待执行的任务列表
            manifest: 运行清单（包含执行策略和配置）
            cancellation_token: 取消令牌（可选）

        Returns:
            List[TaskResult]: 所有任务的执行结果列表

        Raises:
            SchedulerException: 当调度或执行失败时
            TaskExecutionException: 当单个任务执行失败时
            TaskTimeoutException: 当任务超时时
            RateLimitException: 当超出速率限制时
        """
        pass
