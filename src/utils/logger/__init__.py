"""
Dify自动化测试工具 - 日志管理模块

高性能、线程安全的日志系统，基于loguru实现，专为高并发测试环境设计。
支持异步操作、结构化日志和动态配置。

主要特性:
- 基于loguru的高性能日志系统
- 线程安全，支持高并发环境
- 异步操作支持，不阻塞主线程
- 结构化日志，支持上下文信息
- 按日期自动分割的日志文件
- 动态日志级别控制
- 完整的异常追踪和错误处理
- 与YAML配置系统无缝集成
"""

from .logger import (
    SimpleLogManager,
    get_logger,
    setup_logging,
    log_context,
    log_performance,
    log_exception,
    log_workflow_trace,
    LoggingException,
    _log_manager,
)

__version__ = "1.0.0"
__all__ = [
    "SimpleLogManager",
    "get_logger",
    "setup_logging",
    "log_context",
    "log_performance",
    "log_exception",
    "log_workflow_trace",
    "_log_manager",
    "LoggingException"
]
