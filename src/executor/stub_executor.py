"""
Executor Module - Stub Executor for Testing

Date: 2025-11-14
Author: backend-developer
Description: 测试用的桩执行器，不调用真实 Dify API
"""

# 标准库
import random
from datetime import datetime
from typing import Callable, Dict, Any, Optional
import time
from uuid import uuid4

# 项目模块
from src.utils.exceptions import TaskExecutionException, TaskTimeoutException

from .concurrent_executor import ConcurrentExecutor
from .models import Task


class StubExecutor(ConcurrentExecutor):
    """测试用的桩执行器。

    特性：
        - 不调用真实 Dify API
        - 可配置模拟延迟
        - 可配置失败率
        - 可配置特定任务的行为（成功/失败/超时）

    Attributes:
        _simulated_delay: 模拟执行延迟（秒）
        _failure_rate: 失败率（0.0-1.0）
        _task_behaviors: 特定任务的行为配置 {task_id: behavior}
    """

    def __init__(
            self,
            simulated_delay: float = 0.0,
            failure_rate: float = 0.0,
            task_behaviors: Optional[Dict[str, str]] = None,
            now_fn: Callable[[], datetime] = datetime.now,
            sleep_fn: Callable[[float], None] = time.sleep,
            id_fn: Callable[[], str] = lambda: str(uuid4())
    ) -> None:
        """初始化桩执行器。

        Args:
            simulated_delay: 模拟执行延迟（秒）
            failure_rate: 失败率（0.0-1.0）
            task_behaviors: 特定任务行为 {task_id: "success"|"failure"|"timeout"|"error"}
            now_fn: 时间获取函数
            sleep_fn: 休眠函数
            id_fn: ID生成函数
        """

        # 创建自定义的桩执行函数
        def stub_func(task: Task) -> Dict[str, Any]:
            return self._stub_execution(task)

        super().__init__(
            task_execution_func=stub_func,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            id_fn=id_fn
        )

        self._simulated_delay = simulated_delay
        self._failure_rate = failure_rate
        self._task_behaviors = task_behaviors or {}

    def _stub_execution(self, task: Task) -> Dict[str, Any]:
        """桩执行函数（模拟 Dify API 调用）。

        行为：
            1. 检查 task_behaviors 是否有特定配置
            2. 否则根据 failure_rate 随机决定成功/失败
            3. 模拟延迟（simulated_delay）
            4. 返回模拟输出

        Args:
            task: 待执行的任务

        Returns:
            Dict[str, Any]: 模拟的执行输出

        Raises:
            TaskExecutionException: 当模拟失败时
            TaskTimeoutException: 当模拟超时时
        """
        # 模拟执行延迟
        if self._simulated_delay > 0:
            self._sleep_fn(self._simulated_delay)

        # 确定任务行为
        behavior = self._task_behaviors.get(task.task_id)

        if behavior is None:
            # 没有特定配置，根据失败率随机决定
            if random.random() < self._failure_rate:
                behavior = "failure"
            else:
                behavior = "success"

        # 根据行为执行
        if behavior == "success":
            return self._stub_success(task)
        elif behavior == "failure":
            raise TaskExecutionException(
                message=f"Simulated failure for task {task.task_id}",
                task_id=task.task_id,
                attempt=task.attempt_count
            )
        elif behavior == "timeout":
            raise TaskTimeoutException(
                message=f"Simulated timeout for task {task.task_id}",
                task_id=task.task_id,
                timeout_seconds=task.timeout_seconds,
                elapsed_seconds=task.timeout_seconds
            )
        elif behavior == "error":
            raise RuntimeError(
                f"Simulated system error for task {task.task_id}"
            )
        else:
            # 未知行为，默认成功
            return self._stub_success(task)

    def _stub_success(self, task: Task) -> Dict[str, Any]:
        """生成模拟成功的输出数据。

        Args:
            task: 待执行的任务

        Returns:
            Dict[str, Any]: 模拟的成功输出
        """
        return {
            "workflow_run_id": f"stub-run-{task.task_id}",
            "status": "succeeded",
            "outputs": {
                "result": "stub execution success",
                "task_id": task.task_id,
                "workflow_id": task.workflow_id,
                "dataset": task.dataset,
                "scenario": task.scenario,
                "parameters": task.parameters
            },
            "elapsed_time": self._simulated_delay,
            "total_tokens": 100,
            "total_steps": 1,
            "created_at": self._now_fn().isoformat(),
            "finished_at": self._now_fn().isoformat()
        }

    def set_task_behavior(self, task_id: str, behavior: str) -> None:
        """设置特定任务的行为。

        Args:
            task_id: 任务ID
            behavior: 行为类型（"success", "failure", "timeout", "error"）

        Raises:
            ValueError: 当 behavior 不是有效值时
        """
        valid_behaviors = {"success", "failure", "timeout", "error"}
        if behavior not in valid_behaviors:
            raise ValueError(
                f"Invalid behavior: {behavior}. Must be one of {valid_behaviors}"
            )

        self._task_behaviors[task_id] = behavior

    def clear_task_behaviors(self) -> None:
        """清除所有特定任务行为配置。"""
        self._task_behaviors.clear()

    def set_failure_rate(self, failure_rate: float) -> None:
        """设置全局失败率。

        Args:
            failure_rate: 失败率（0.0-1.0）

        Raises:
            ValueError: 当 failure_rate 不在 [0.0, 1.0] 范围内时
        """
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError(
                f"failure_rate must be between 0.0 and 1.0, got {failure_rate}"
            )

        self._failure_rate = failure_rate

    def set_simulated_delay(self, delay: float) -> None:
        """设置模拟延迟。

        Args:
            delay: 延迟时间（秒）

        Raises:
            ValueError: 当 delay 为负数时
        """
        if delay < 0:
            raise ValueError(
                f"simulated_delay must be non-negative, got {delay}"
            )

        self._simulated_delay = delay
