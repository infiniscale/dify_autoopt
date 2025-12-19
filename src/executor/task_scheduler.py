"""
Executor Module - Task Scheduler

Date: 2025-11-14
Author: backend-developer
Description: 任务调度器，实现批次管理和高级调度策略
"""

import time
from datetime import datetime
from typing import Callable, List, Optional
from uuid import uuid4

from src.config.models import RunManifest, ExecutionPolicy
from src.utils.exceptions import SchedulerException

from .concurrent_executor import ConcurrentExecutor
from .models import Task, TaskResult, TaskStatus, CancellationToken
from .rate_limiter import RateLimiter


class TaskScheduler(ConcurrentExecutor):
    """任务调度器（增强型并发执行器）。

    在 ConcurrentExecutor 的基础上增加：
    - 批次管理（按 batch_size 分批执行）
    - 速率限制（RateLimiter）
    - 批次间退避（backoff_seconds）
    - 停止条件（max_failures, timeout）

    调度流程：
    1. 将任务分批（batch_size）
    2. 对每批任务：
       a. 检查停止条件（max_failures, timeout）
       b. 检查取消令牌
       c. 速率限制（每批执行前获取令牌）
       d. 执行批次（调用父类 _execute_tasks）
       e. 批次间退避（backoff_seconds）
    3. 聚合所有批次结果

    Attributes:
        _rate_limiter: 速率限制器（可选）
    """

    def __init__(
            self,
            task_execution_func=None,
            now_fn: Callable[[], datetime] = datetime.now,
            sleep_fn: Callable[[float], None] = time.sleep,
            id_fn: Callable[[], str] = lambda: str(uuid4())
    ) -> None:
        """初始化任务调度器。

        Args:
            task_execution_func: 任务执行函数
            now_fn: 时间获取函数
            sleep_fn: 休眠函数
            id_fn: ID生成函数
        """
        super().__init__(task_execution_func, now_fn, sleep_fn, id_fn)
        self._rate_limiter: Optional[RateLimiter] = None

    def _execute_tasks(
            self,
            tasks: List[Task],
            manifest: RunManifest,
            cancellation_token: Optional[CancellationToken] = None
    ) -> List[TaskResult]:
        """执行任务列表（使用批次调度）。

        覆盖父类方法，增加批次管理、速率限制和停止条件。

        Args:
            tasks: 待执行的任务列表
            manifest: 运行清单
            cancellation_token: 取消令牌

        Returns:
            List[TaskResult]: 所有任务的执行结果

        Raises:
            SchedulerException: 当调度失败时
        """
        if not tasks:
            return []

        execution_policy = manifest.execution_policy
        batch_size = execution_policy.batch_size

        # 初始化速率限制器
        if execution_policy.rate_control:
            self._rate_limiter = RateLimiter(
                rate_limit=execution_policy.rate_control,
                now_fn=self._now_fn,
                sleep_fn=self._sleep_fn
            )

        # 分批
        batches = self._split_into_batches(tasks, batch_size)

        # 执行所有批次
        all_results: List[TaskResult] = []
        start_time = self._now_fn()

        for batch_idx, batch in enumerate(batches):
            # 检查停止条件
            if self._should_stop(all_results, execution_policy.stop_conditions, start_time):
                # 标记剩余任务为取消
                for task in batch:
                    task.status = TaskStatus.CANCELLED
                    task.metadata["error_message"] = "Stopped due to stop conditions"
                    all_results.append(TaskResult.from_task(task))
                continue

            # 检查取消令牌
            if cancellation_token and cancellation_token.is_cancelled():
                for task in batch:
                    task.status = TaskStatus.CANCELLED
                    task.metadata["error_message"] = "Execution cancelled"
                    all_results.append(TaskResult.from_task(task))
                continue

            # 速率限制（每批次获取 batch_size 个令牌）
            if self._rate_limiter:
                self._rate_limiter.acquire(tokens=len(batch))

            # 执行批次（调用父类方法）
            batch_results = super()._execute_tasks(batch, manifest, cancellation_token)
            all_results.extend(batch_results)

            # 批次间退避（最后一批不需要退避）
            if batch_idx < len(batches) - 1 and execution_policy.backoff_seconds > 0:
                if cancellation_token and cancellation_token.is_cancelled():
                    break
                self._sleep_fn(execution_policy.backoff_seconds)

        return all_results

    def _split_into_batches(self, tasks: List[Task], batch_size: int) -> List[List[Task]]:
        """将任务列表分批。

        Args:
            tasks: 任务列表
            batch_size: 批次大小

        Returns:
            List[List[Task]]: 分批后的任务列表
        """
        batches = []
        for i in range(0, len(tasks), batch_size):
            batches.append(tasks[i:i + batch_size])
        return batches

    def _should_stop(
            self,
            results: List[TaskResult],
            stop_conditions: dict,
            start_time: datetime
    ) -> bool:
        """检查是否应该停止执行。

        支持的停止条件：
        - max_failures: 最大失败数
        - timeout: 总超时时间（秒）

        Args:
            results: 已完成的任务结果
            stop_conditions: 停止条件配置
            start_time: 开始时间

        Returns:
            bool: 应该停止返回 True
        """
        # 检查最大失败数
        if "max_failures" in stop_conditions:
            max_failures = stop_conditions["max_failures"]
            failures = sum(
                1 for r in results
                if r.task_status in {TaskStatus.FAILED, TaskStatus.ERROR, TaskStatus.TIMEOUT}
            )
            if failures >= max_failures:
                return True

        # 检查总超时
        if "timeout" in stop_conditions:
            timeout_seconds = stop_conditions["timeout"]
            elapsed = (self._now_fn() - start_time).total_seconds()
            if elapsed >= timeout_seconds:
                return True

        return False
