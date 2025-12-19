"""
Executor Module - Task Execution and Scheduling

This module handles test case generation, concurrent execution, and task scheduling.
"""

# 现有模块
from .pairwise_engine import PairwiseEngine
from .test_case_generator import TestCaseGenerator
from .run_manifest_builder import RunManifestBuilder

# Phase 1 核心模块
from .models import (
    TaskStatus,
    Task,
    TaskResult,
    RunStatistics,
    RunExecutionResult,
    CancellationToken
)
from .executor_base import ExecutorBase

# Phase 2 核心模块
from .concurrent_executor import ConcurrentExecutor, TaskExecutionFunc
from .stub_executor import StubExecutor

# Phase 3 核心模块
from .rate_limiter import RateLimiter
from .task_scheduler import TaskScheduler

# Phase 4 核心模块
from .result_converter import ResultConverter
from .executor_service import ExecutorService

__all__ = [
    # 现有导出
    'PairwiseEngine',
    'TestCaseGenerator',
    'RunManifestBuilder',

    # Phase 1 导出
    'TaskStatus',
    'Task',
    'TaskResult',
    'RunStatistics',
    'RunExecutionResult',
    'CancellationToken',
    'ExecutorBase',

    # Phase 2 导出
    'ConcurrentExecutor',
    'TaskExecutionFunc',
    'StubExecutor',

    # Phase 3 导出
    'RateLimiter',
    'TaskScheduler',

    # Phase 4 导出
    'ResultConverter',
    'ExecutorService',
]
