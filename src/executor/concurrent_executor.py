"""
Executor Module - Concurrent Executor

Date: 2025-11-14
Author: backend-developer
Description: 基于线程池的并发执行器实现
"""

# 标准库
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, TimeoutError
from datetime import datetime
from typing import Callable, List, Optional, Dict, Any
from uuid import uuid4

# 项目模块
from src.config.models import RunManifest, RetryPolicy
from src.utils.exceptions import (
    TaskExecutionException,
    TaskTimeoutException,
    SchedulerException
)

from .executor_base import ExecutorBase
from .models import Task, TaskResult, TaskStatus, CancellationToken


# 任务执行函数签名
TaskExecutionFunc = Callable[[Task], Dict[str, Any]]


class ConcurrentExecutor(ExecutorBase):
    """基于线程池的并发执行器。

    特性：
        - 使用 ThreadPoolExecutor 实现并发执行
        - 支持动态并发度控制（从 ExecutionPolicy 读取）
        - 支持超时和重试机制
        - 支持取消令牌
        - 可注入自定义任务执行函数（默认使用 stub）

    Attributes:
        _task_execution_func: 任务执行函数（可注入）
    """

    def __init__(
        self,
        task_execution_func: Optional[TaskExecutionFunc] = None,
        now_fn: Callable[[], datetime] = datetime.now,
        sleep_fn: Callable[[float], None] = time.sleep,
        id_fn: Callable[[], str] = lambda: str(uuid4())
    ) -> None:
        """初始化并发执行器。

        Args:
            task_execution_func: 任务执行函数（默认使用 _default_stub_execution）
            now_fn: 时间获取函数
            sleep_fn: 休眠函数
            id_fn: ID生成函数
        """
        super().__init__(now_fn, sleep_fn, id_fn)
        self._task_execution_func = task_execution_func or self._default_stub_execution

    def _execute_tasks(
        self,
        tasks: List[Task],
        manifest: RunManifest,
        cancellation_token: Optional[CancellationToken] = None
    ) -> List[TaskResult]:
        """并发执行任务列表（实现抽象方法）。

        实现要点：
            1. 从 manifest.execution_policy 读取并发度
            2. 创建 ThreadPoolExecutor
            3. 提交所有任务到线程池
            4. 使用 as_completed() 收集结果
            5. 检查 cancellation_token，如果被取消则提前退出
            6. 处理超时和重试逻辑

        Args:
            tasks: 待执行的任务列表
            manifest: 运行清单（包含执行策略）
            cancellation_token: 取消令牌（可选）

        Returns:
            List[TaskResult]: 所有任务的执行结果

        Raises:
            SchedulerException: 当调度失败时
        """
        if not tasks:
            return []

        # 从 ExecutionPolicy 读取并发度
        concurrency = manifest.execution_policy.concurrency
        retry_policy = manifest.execution_policy.retry_policy

        # 存储任务结果
        task_results: List[TaskResult] = []

        # 检查取消令牌
        if cancellation_token and cancellation_token.is_cancelled():
            # 所有任务标记为取消
            for task in tasks:
                task.status = TaskStatus.CANCELLED
                task.metadata["error_message"] = "Execution cancelled before start"
                task_results.append(TaskResult.from_task(task))
            return task_results

        try:
            # 创建线程池
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # 提交所有任务到线程池，并存储 Future 到 Task 的映射
                future_to_task: Dict[Future, Task] = {}

                for task in tasks:
                    # 检查是否被取消
                    if cancellation_token and cancellation_token.is_cancelled():
                        task.status = TaskStatus.CANCELLED
                        task.metadata["error_message"] = "Execution cancelled during submission"
                        task_results.append(TaskResult.from_task(task))
                        continue

                    # 提交任务到线程池
                    future = executor.submit(
                        self._execute_single_task,
                        task,
                        retry_policy,
                        cancellation_token
                    )
                    future_to_task[future] = task

                # 收集所有已完成的任务结果
                for future in as_completed(future_to_task):
                    task = future_to_task[future]

                    try:
                        # 获取任务结果
                        task_result = future.result()
                        task_results.append(task_result)

                    except Exception as e:
                        # 处理执行过程中的异常
                        task.status = TaskStatus.ERROR
                        task.metadata["error_message"] = f"Unexpected error during execution: {str(e)}"
                        task.finished_at = self._now_fn()
                        task_results.append(TaskResult.from_task(task))

                    # 检查取消令牌
                    if cancellation_token and cancellation_token.is_cancelled():
                        # 取消剩余未完成的任务
                        for remaining_future in future_to_task:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break

                # 处理被取消的任务（未开始执行的）
                for future, task in future_to_task.items():
                    if future.cancelled():
                        task.status = TaskStatus.CANCELLED
                        task.metadata["error_message"] = "Task cancelled"
                        task.finished_at = self._now_fn()
                        # 避免重复添加已完成的任务
                        if not any(tr.task_id == task.task_id for tr in task_results):
                            # 计算执行时间
                            execution_time = 0.0
                            if task.started_at and task.finished_at:
                                execution_time = (task.finished_at - task.started_at).total_seconds()

                            task_results.append(TaskResult(
                                task_id=task.task_id,
                                workflow_id=task.workflow_id,
                                dataset=task.dataset,
                                scenario=task.scenario,
                                task_status=task.status,
                                error_message=task.metadata.get("error_message"),
                                attempt_count=task.attempt_count,
                                created_at=task.created_at,
                                finished_at=task.finished_at,
                                execution_time=execution_time,
                                metadata=task.metadata
                            ))

        except Exception as e:
            raise SchedulerException(f"Thread pool execution failed: {str(e)}") from e

        return task_results

    def _execute_single_task(
        self,
        task: Task,
        retry_policy: RetryPolicy,
        cancellation_token: Optional[CancellationToken] = None
    ) -> TaskResult:
        """执行单个任务（支持重试）。

        实现要点：
            1. 循环重试（最多 retry_policy.max_attempts 次）
            2. 每次尝试前检查 cancellation_token
            3. 调用 task_execution_func 执行任务
            4. 处理超时（使用 task.timeout_seconds）
            5. 失败后根据 retry_policy 重试（指数退避）
            6. 返回 TaskResult

        Args:
            task: 待执行的任务
            retry_policy: 重试策略
            cancellation_token: 取消令牌

        Returns:
            TaskResult: 任务执行结果
        """
        last_error = None

        # 重试循环（最多 max_attempts 次）
        for attempt in range(retry_policy.max_attempts):
            # 检查取消令牌
            if cancellation_token and cancellation_token.is_cancelled():
                task.status = TaskStatus.CANCELLED
                task.metadata["error_message"] = "Task cancelled during execution"
                task.finished_at = self._now_fn()
                return TaskResult.from_task(task)

            try:
                # 标记任务开始
                task.mark_started(now_fn=self._now_fn)

                # 使用 ThreadPoolExecutor 实现超时控制
                with ThreadPoolExecutor(max_workers=1) as timeout_executor:
                    future = timeout_executor.submit(self._task_execution_func, task)

                    try:
                        # 等待任务完成，带超时
                        result_data = future.result(timeout=task.timeout_seconds)

                        # 任务成功执行
                        task.mark_finished(
                            status=TaskStatus.SUCCEEDED,
                            now_fn=self._now_fn
                        )

                        return TaskResult.from_task(task)

                    except TimeoutError:
                        # 任务超时
                        future.cancel()
                        last_error = f"Task execution timeout after {task.timeout_seconds}s"

                        # 如果这是最后一次尝试，标记为超时
                        if attempt == retry_policy.max_attempts - 1:
                            task.mark_finished(
                                status=TaskStatus.TIMEOUT,
                                now_fn=self._now_fn
                            )
                            task.metadata["error_message"] = last_error
                            return TaskResult.from_task(task)

                        # 否则重置状态准备重试
                        task.status = TaskStatus.PENDING

            except TaskTimeoutException as e:
                # 显式超时异常
                last_error = str(e)

                if attempt == retry_policy.max_attempts - 1:
                    task.mark_finished(
                        status=TaskStatus.TIMEOUT,
                        now_fn=self._now_fn
                    )
                    task.metadata["error_message"] = last_error
                    return TaskResult.from_task(task)

                task.status = TaskStatus.PENDING

            except TaskExecutionException as e:
                # 任务执行失败（业务逻辑错误）
                last_error = str(e)

                if attempt == retry_policy.max_attempts - 1:
                    task.mark_finished(
                        status=TaskStatus.FAILED,
                        now_fn=self._now_fn
                    )
                    task.metadata["error_message"] = last_error
                    return TaskResult.from_task(task)

                task.status = TaskStatus.PENDING

            except Exception as e:
                # 系统异常
                last_error = f"Unexpected error: {str(e)}"

                if attempt == retry_policy.max_attempts - 1:
                    task.mark_finished(
                        status=TaskStatus.ERROR,
                        now_fn=self._now_fn
                    )
                    task.metadata["error_message"] = last_error
                    return TaskResult.from_task(task)

                task.status = TaskStatus.PENDING

            # 如果不是最后一次尝试，执行退避等待
            if attempt < retry_policy.max_attempts - 1:
                # 计算退避时间（指数退避）
                backoff_time = retry_policy.backoff_seconds * (
                    retry_policy.backoff_multiplier ** attempt
                )

                # 等待（检查取消令牌）
                if cancellation_token and cancellation_token.is_cancelled():
                    task.status = TaskStatus.CANCELLED
                    task.metadata["error_message"] = "Task cancelled during retry backoff"
                    task.finished_at = self._now_fn()
                    return TaskResult.from_task(task)

                self._sleep_fn(backoff_time)

        # 理论上不应该到达这里（所有情况都在循环中处理）
        task.mark_finished(
            status=TaskStatus.ERROR,
            now_fn=self._now_fn
        )
        task.metadata["error_message"] = last_error or "Unknown error during task execution"
        return TaskResult.from_task(task)

    def _default_stub_execution(self, task: Task) -> Dict[str, Any]:
        """默认的桩执行函数（测试用）。

        模拟成功执行，返回简单输出。

        Args:
            task: 待执行的任务

        Returns:
            Dict[str, Any]: 模拟的输出数据
        """
        return {
            "workflow_run_id": f"stub-run-{task.task_id}",
            "status": "succeeded",
            "outputs": {"result": "stub execution success"},
            "elapsed_time": 0.1
        }
