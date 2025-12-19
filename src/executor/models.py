"""
Executor Module - Core Data Models

Date: 2025-11-14
Author: backend-developer
Description: 定义任务执行的核心数据结构、状态枚举和统计模型
"""

# 标准库
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

# 项目模块
from src.config.models import TestCase, ExecutionPolicy
from src.collector.models import TestResult, TestStatus


class TaskStatus(Enum):
    """任务执行状态枚举。

    状态流转图:
        PENDING → QUEUED → RUNNING → {SUCCEEDED, FAILED, TIMEOUT, ERROR}
        任意状态 → CANCELLED

    Attributes:
        PENDING: 任务已创建但未加入队列
        QUEUED: 任务已加入执行队列
        RUNNING: 任务正在执行
        SUCCEEDED: 任务执行成功
        FAILED: 任务执行失败（业务失败）
        TIMEOUT: 任务执行超时
        CANCELLED: 任务被取消
        ERROR: 任务执行错误（系统异常）
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ERROR = "error"

    def is_terminal(self) -> bool:
        """判断是否为终态。

        Returns:
            bool: SUCCEEDED, FAILED, TIMEOUT, CANCELLED, ERROR 返回 True
        """
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.CANCELLED,
            TaskStatus.ERROR
        }

    def is_success(self) -> bool:
        """判断是否为成功状态。

        Returns:
            bool: 仅 SUCCEEDED 返回 True
        """
        return self == TaskStatus.SUCCEEDED


@dataclass
class Task:
    """单个测试任务的完整描述。

    包含任务的标识信息、业务数据、执行策略和运行时状态。

    Attributes:
        # 标识字段
        task_id: 任务唯一标识符
        workflow_id: 工作流ID
        dataset: 数据集名称
        scenario: 场景类型（normal/boundary/error/custom）

        # 业务数据字段
        test_case: 原始测试用例数据
        parameters: 执行参数字典
        conversation_flow: 对话流（可选，用于chatflow）
        prompt_variant: 提示词变体ID（可选）

        # 执行策略字段
        timeout_seconds: 超时时间（秒）
        max_retries: 最大重试次数
        retry_backoff: 重试退避时间（秒）

        # 运行时状态字段
        status: 当前任务状态
        attempt_count: 已尝试次数（含当前）
        created_at: 创建时间
        started_at: 开始执行时间（可选）
        finished_at: 完成时间（可选）
        error_message: 错误信息（可选）
        result: 执行结果（可选）
    """

    # 标识字段
    task_id: str
    workflow_id: str
    dataset: str
    scenario: str

    # 业务数据字段
    test_case: TestCase
    parameters: Dict[str, Any]
    conversation_flow: Optional[Any] = None
    prompt_variant: Optional[str] = None

    # 执行策略字段
    timeout_seconds: float = 300.0
    max_retries: int = 3
    retry_backoff: float = 2.0

    # 运行时状态字段
    status: TaskStatus = TaskStatus.PENDING
    attempt_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[TestResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest_case(
            cls,
            test_case: TestCase,
            execution_policy: ExecutionPolicy,
            workflow_id: str,
            id_fn: Callable[[], str] = lambda: str(uuid4()),
            now_fn: Callable[[], datetime] = datetime.now
    ) -> "Task":
        """从 RunManifest 的 TestCase 构建 Task 对象（工厂方法）。

        Args:
            test_case: 测试用例对象
            execution_policy: 执行策略配置
            workflow_id: 工作流ID
            id_fn: ID生成函数（用于依赖注入和测试）
            now_fn: 时间获取函数（用于依赖注入和测试）

        Returns:
            Task: 构建的任务对象

        Raises:
            ValueError: 当 test_case 或 execution_policy 为 None 时
        """
        if test_case is None:
            raise ValueError("test_case cannot be None")
        if execution_policy is None:
            raise ValueError("execution_policy cannot be None")

        # 从执行策略中提取超时和重试配置
        timeout = execution_policy.stop_conditions.get("timeout_per_task", 300.0)
        retry_policy = execution_policy.retry_policy

        return cls(
            task_id=id_fn(),
            workflow_id=workflow_id,
            dataset=test_case.dataset,
            scenario=test_case.scenario,
            test_case=test_case,
            parameters=test_case.parameters,
            conversation_flow=test_case.conversation_flow,
            prompt_variant=test_case.prompt_variant,
            timeout_seconds=timeout,
            max_retries=retry_policy.max_attempts - 1,  # max_attempts 包含首次执行
            retry_backoff=retry_policy.backoff_seconds,
            status=TaskStatus.PENDING,
            attempt_count=0,
            created_at=now_fn()
        )

    def mark_started(
            self,
            now_fn: Callable[[], datetime] = datetime.now
    ) -> None:
        """标记任务开始执行，更新状态和时间戳。

        Args:
            now_fn: 时间获取函数（用于依赖注入和测试）

        Raises:
            ValueError: 当任务已处于终态时
        """
        if self.status.is_terminal():
            raise ValueError(
                f"Cannot start task in terminal state: {self.status.value}"
            )

        self.status = TaskStatus.RUNNING
        self.started_at = now_fn()
        self.attempt_count += 1

    def mark_finished(
            self,
            status: TaskStatus,
            result: Optional[TestResult] = None,
            error_message: Optional[str] = None,
            now_fn: Callable[[], datetime] = datetime.now
    ) -> None:
        """标记任务完成，更新状态、结果和时间戳。

        Args:
            status: 最终状态（必须为终态）
            result: 执行结果对象（可选）
            error_message: 错误信息（失败时提供）
            now_fn: 时间获取函数（用于依赖注入和测试）

        Raises:
            ValueError: 当 status 不是终态，或任务已处于终态时
        """
        if not status.is_terminal():
            raise ValueError(
                f"mark_finished requires terminal status, got: {status.value}"
            )
        if self.status.is_terminal():
            raise ValueError(
                f"Cannot finish task already in terminal state: {self.status.value}"
            )

        self.status = status
        self.result = result
        self.error_message = error_message
        self.finished_at = now_fn()

    def is_terminal(self) -> bool:
        """判断任务是否已处于终态。

        Returns:
            bool: 任务处于终态时返回 True
        """
        return self.status.is_terminal()

    def can_retry(self) -> bool:
        """判断任务是否可以重试。

        重试条件：
        1. 状态为 FAILED 或 TIMEOUT 或 ERROR
        2. 尝试次数未超过最大重试次数

        Returns:
            bool: 可以重试时返回 True
        """
        retriable_states = {TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.ERROR}
        return (
                self.status in retriable_states
                and self.attempt_count <= self.max_retries
        )


@dataclass
class TaskResult:
    """任务执行结果的封装。

    将 Task 和 TestResult 整合为统一的结果对象，便于下游模块使用。

    Attributes:
        task_id: 任务ID
        workflow_id: 工作流ID
        dataset: 数据集名称
        scenario: 场景类型
        task_status: 任务状态
        test_result: 测试结果对象（可选，成功时提供）
        error_message: 错误信息（失败时提供）
        attempt_count: 执行尝试次数
        created_at: 任务创建时间
        finished_at: 任务完成时间
        execution_time: 执行耗时（秒）
    """

    task_id: str
    workflow_id: str
    dataset: str
    scenario: str
    task_status: TaskStatus
    test_result: Optional[TestResult] = None
    error_message: Optional[str] = None
    attempt_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: Task) -> "TaskResult":
        """从 Task 对象构建 TaskResult（工厂方法）。

        Args:
            task: 任务对象

        Returns:
            TaskResult: 构建的任务结果对象

        Raises:
            ValueError: 当 task 为 None 或未处于终态时
        """
        if task is None:
            raise ValueError("task cannot be None")
        if not task.is_terminal():
            raise ValueError(
                f"Cannot create TaskResult from non-terminal task: {task.status.value}"
            )

        # 计算执行时间
        execution_time = 0.0
        if task.started_at and task.finished_at:
            execution_time = (task.finished_at - task.started_at).total_seconds()

        # 获取错误信息（优先使用 error_message，否则从 metadata 读取）
        error_msg = task.error_message or task.metadata.get("error_message")

        return cls(
            task_id=task.task_id,
            workflow_id=task.workflow_id,
            dataset=task.dataset,
            scenario=task.scenario,
            task_status=task.status,
            test_result=task.result,
            error_message=error_msg,
            attempt_count=task.attempt_count,
            created_at=task.created_at,
            finished_at=task.finished_at,
            execution_time=execution_time,
            metadata=task.metadata,
        )

    def get_tokens_used(self) -> int:
        """获取任务消耗的 Token 数量。

        Returns:
            int: Token 数量，失败时返回 0
        """
        if self.test_result:
            return self.test_result.tokens_used
        return 0

    def get_cost(self) -> float:
        """获取任务执行成本。

        Returns:
            float: 执行成本（美元），失败时返回 0.0
        """
        if self.test_result:
            return self.test_result.cost
        return 0.0


@dataclass
class RunStatistics:
    """执行批次的聚合统计数据。

    Attributes:
        total_tasks: 总任务数
        completed_tasks: 已完成任务数（包括成功和失败）
        succeeded_tasks: 成功任务数
        failed_tasks: 失败任务数
        timeout_tasks: 超时任务数
        cancelled_tasks: 取消任务数
        error_tasks: 错误任务数
        success_rate: 成功率（0-1）
        total_execution_time: 总执行时间（秒）
        avg_execution_time: 平均执行时间（秒）
        total_tokens: 总 Token 消耗
        total_cost: 总成本（美元）
        total_retries: 总重试次数
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    succeeded_tasks: int = 0
    failed_tasks: int = 0
    timeout_tasks: int = 0
    cancelled_tasks: int = 0
    error_tasks: int = 0
    success_rate: float = 0.0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_retries: int = 0


@dataclass
class RunExecutionResult:
    """单次 RunManifest 执行的完整结果。

    Attributes:
        run_id: 执行批次ID
        workflow_id: 工作流ID
        started_at: 开始时间
        finished_at: 完成时间
        total_duration: 总耗时（秒）
        task_results: 所有任务的执行结果列表
        statistics: 聚合统计数据
        metadata: 额外元数据
    """

    run_id: str
    workflow_id: str
    started_at: datetime
    finished_at: datetime
    total_duration: float
    task_results: List[TaskResult]
    statistics: RunStatistics
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task_results(
            cls,
            run_id: str,
            workflow_id: str,
            started_at: datetime,
            finished_at: datetime,
            task_results: List[TaskResult],
            metadata: Optional[Dict[str, Any]] = None
    ) -> "RunExecutionResult":
        """从任务结果列表构建 RunExecutionResult（工厂方法）。

        Args:
            run_id: 执行批次ID
            workflow_id: 工作流ID
            started_at: 开始时间
            finished_at: 完成时间
            task_results: 任务结果列表
            metadata: 额外元数据（可选）

        Returns:
            RunExecutionResult: 构建的执行结果对象

        Raises:
            ValueError: 当必需参数为 None 或 task_results 为空时
        """
        if not run_id:
            raise ValueError("run_id cannot be empty")
        if not workflow_id:
            raise ValueError("workflow_id cannot be empty")
        if started_at is None:
            raise ValueError("started_at cannot be None")
        if finished_at is None:
            raise ValueError("finished_at cannot be None")
        if not task_results:
            raise ValueError("task_results cannot be empty")

        # 计算统计数据
        stats = RunStatistics()
        stats.total_tasks = len(task_results)

        for task_result in task_results:
            # 计数各类状态
            if task_result.task_status == TaskStatus.SUCCEEDED:
                stats.succeeded_tasks += 1
            elif task_result.task_status == TaskStatus.FAILED:
                stats.failed_tasks += 1
            elif task_result.task_status == TaskStatus.TIMEOUT:
                stats.timeout_tasks += 1
            elif task_result.task_status == TaskStatus.CANCELLED:
                stats.cancelled_tasks += 1
            elif task_result.task_status == TaskStatus.ERROR:
                stats.error_tasks += 1

            # 累计执行时间、Token 和成本
            stats.total_execution_time += task_result.execution_time
            stats.total_tokens += task_result.get_tokens_used()
            stats.total_cost += task_result.get_cost()

            # 累计重试次数（attempt_count - 1 = 重试次数）
            if task_result.attempt_count > 1:
                stats.total_retries += (task_result.attempt_count - 1)

        # 计算完成任务数和成功率
        stats.completed_tasks = (
                stats.succeeded_tasks
                + stats.failed_tasks
                + stats.timeout_tasks
                + stats.error_tasks
        )

        if stats.completed_tasks > 0:
            stats.success_rate = stats.succeeded_tasks / stats.completed_tasks

        # 计算平均执行时间
        if stats.completed_tasks > 0:
            stats.avg_execution_time = stats.total_execution_time / stats.completed_tasks

        # 计算总耗时
        total_duration = (finished_at - started_at).total_seconds()

        return cls(
            run_id=run_id,
            workflow_id=workflow_id,
            started_at=started_at,
            finished_at=finished_at,
            total_duration=total_duration,
            task_results=task_results,
            statistics=stats,
            metadata=metadata or {}
        )


class CancellationToken:
    """线程安全的取消令牌。

    用于在并发执行中优雅地取消任务。使用 threading.Lock 保证原子性。

    Attributes:
        _is_cancelled: 取消标志（内部状态）
        _lock: 线程锁

    Example:
        >>> token = CancellationToken()
        >>> token.cancel()
        >>> assert token.is_cancelled() == True
    """

    def __init__(self) -> None:
        """初始化取消令牌。"""
        self._is_cancelled: bool = False
        self._lock: Lock = Lock()

    def cancel(self) -> None:
        """设置取消标志为 True（线程安全）。"""
        with self._lock:
            self._is_cancelled = True

    def is_cancelled(self) -> bool:
        """检查是否已取消（线程安全）。

        Returns:
            bool: 已取消返回 True
        """
        with self._lock:
            return self._is_cancelled

    def reset(self) -> None:
        """重置取消标志为 False（用于令牌复用）。"""
        with self._lock:
            self._is_cancelled = False
