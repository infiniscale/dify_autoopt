"""
日期: 2025-01-12
作者: rrong
描述: 日志模块的使用示例和最佳实践指南
"""

"""
日志模块使用指南
===============

## 目录
1. 基础使用
2. 配置管理
3. 结构化日志
4. 上下文管理
5. 性能监控
6. 异步操作
7. 错误处理
8. 最佳实践
"""

# ===== 1. 基础使用 =====
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# 导入日志模块
from src.utils.logger import (
    setup_logging,
    get_logger,
    get_logger_with_context,
    log_performance,
    log_exception,
    log_workflow_trace,
    log_context
)


async def basic_usage_example():
    """基础使用示例"""

    # 1. 初始化日志系统（在主函数开始时调用）
    await setup_logging()

    # 2. 获取logger实例
    logger = get_logger("basic_example")

    # 3. 基础日志记录
    logger.debug("这是一条调试信息")
    logger.info("应用启动成功")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")


async def config_example():
    """配置示例"""
    config_path = "config/config.yaml"

    # 使用自定义配置初始化
    log_manager = await setup_logging(config_path)

    # 获取logger
    logger = get_logger("config_example")

    # 记录配置信息
    logger.info("日志系统已使用自定义配置初始化")


# ===== 2. 结构化日志 =====
async def structured_logging_example():
    """结构化日志示例"""
    logger = get_logger("structured_example")

    # 使用extra参数记录结构化数据
    logger.info(
        "用户操作记录",
        extra={
            "user_id": "user_123",
            "action": "login",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0...",
            "timestamp": datetime.now().isoformat(),
            "session_id": "sess_456"
        }
    )

    # 工作流执行日志
    logger.info(
        "工作流执行状态",
        extra={
            "workflow_id": "workflow_001",
            "status": "running",
            "progress": 0.75,
            "start_time": "2025-01-12T10:30:00Z",
            "estimated_completion": "2025-01-12T10:45:00Z",
            "node_details": {
                "current_node": "text_processing",
                "completed_nodes": ["data_input", "validation"],
                "pending_nodes": ["result_formatting"]
            }
        }
    )


# ===== 3. 上下文管理 =====
async def context_management_example():
    """上下文管理示例"""
    logger = get_logger("context_example")

    # 1. 设置全局上下文（在整个应用执行期间有效）
    from src.utils.logger import _log_manager
    _log_manager.set_global_context(
        application="dify_autoopt",
        version="1.0.0",
        environment="production"
    )

    # 2. 使用上下文管理器（临时上下文）
    with log_context(request_id="req_123", user_id="user_456"):
        logger.info("处理用户请求开始")

        # 嵌套上下文
        with log_context(operation="validate_input", step=1):
            logger.info("验证用户输入")

        with log_context(operation="process_data", step=2):
            logger.info("处理业务数据")

        logger.info("处理用户请求完成")

    # 3. 获取带上下文的logger
    context_logger = get_logger_with_context(
        "context_logger",
        workflow_id="wf_789",
        execution_type="test"
    )

    context_logger.info("这是来自特定上下文的日志")


# ===== 4. 性能监控 =====
@log_performance("数据库查询")
async def database_query_example():
    """性能监控示例"""
    logger = get_logger("performance_example")

    # 模拟耗时操作
    await asyncio.sleep(0.1)

    logger.info("数据库查询完成")


def sync_performance_example():
    """同步函数性能监控"""
    logger = get_logger("performance_example")

    # 模拟同步耗时操作
    import time
    time.sleep(0.05)

    logger.info("同步操作完成")


# ===== 5. 异常处理 =====
@log_exception(reraise=False)  # 记录异常但不重新抛出
async def async_exception_example():
    """异步异常处理示例"""
    logger = get_logger("exception_example")

    # 模拟异常
    raise ValueError("这是一个示例异常")


@log_exception(level="WARNING")  # 记录为警告级别
def sync_exception_example():
    """同步异常处理示例"""
    logger = get_logger("exception_example")

    # 模拟异常
    raise RuntimeError("这是一个运行时异常")


async def try_catch_logging_example():
    """try-catch中的日志记录示例"""
    logger = get_logger("exception_example")

    try:
        # 模拟可能失败的操作
        raise ConnectionError("数据库连接失败")

    except ConnectionError as e:
        logger.error(
            "数据库操作失败",
            extra={
                "operation": "database_connect",
                "database": "dify_db",
                "retry_count": 3,
                "error_code": "CONN_FAILED",
                "error_details": str(e)
            },
            exc_info=True  # 包含完整的异常堆栈
        )

        # 记录恢复策略
        logger.info(
            "尝试恢复数据库连接",
            extra={
                "recovery_strategy": "reconnect",
                "max_retries": 5,
                "retry_delay": 2.0
            }
        )


# ===== 6. 工作流跟踪 =====
async def workflow_tracing_example():
    """工作流跟踪示例"""
    logger = get_logger("workflow_example")

    workflow_id = "workflow_test_001"

    # 使用工作流跟踪上下文管理器
    with log_workflow_trace(
        workflow_id=workflow_id,
        operation="full_workflow_execution",
        logger=logger
    ):
        # 数据输入阶段
        with log_workflow_trace(
            workflow_id=workflow_id,
            operation="data_input",
            logger=logger
        ):
            logger.info("开始输入数据")
            await asyncio.sleep(0.1)

        # 处理阶段
        with log_workflow_trace(
            workflow_id=workflow_id,
            operation="text_processing",
            logger=logger
        ):
            logger.info("开始文本处理")
            await asyncio.sleep(0.2)

        # 结果输出阶段
        with log_workflow_trace(
            workflow_id=workflow_id,
            operation="output_generation",
            logger=logger
        ):
            logger.info("生成输出结果")
            await asyncio.sleep(0.1)


# ===== 7. 高级用法 =====
class WorkflowLogger:
    """工作流专用日志器类"""

    def __init__(self, workflow_id: str, execution_id: str):
        self.workflow_id = workflow_id
        self.execution_id = execution_id
        self.logger = get_logger_with_context(
            "workflow",
            workflow_id=workflow_id,
            execution_id=execution_id
        )
        self.start_time = datetime.now()

    def log_node_start(self, node_id: str, node_type: str):
        """记录节点开始"""
        self.logger.info(
            f"节点开始执行: {node_id}",
            extra={
                "node_id": node_id,
                "node_type": node_type,
                "status": "started",
                "execution_time": 0
            }
        )

    def log_node_complete(self, node_id: str, duration: float, result_count: int = 0):
        """记录节点完成"""
        self.logger.info(
            f"节点执行完成: {node_id}",
            extra={
                "node_id": node_id,
                "status": "completed",
                "execution_time": duration,
                "result_count": result_count
            }
        )

    def log_node_error(self, node_id: str, error: Exception):
        """记录节点错误"""
        self.logger.error(
            f"节点执行失败: {node_id}",
            extra={
                "node_id": node_id,
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)
            },
            exc_info=True
        )

    def log_workflow_summary(self, total_nodes: int, success_nodes: int, total_duration: float):
        """记录工作流摘要"""
        success_rate = success_nodes / total_nodes if total_nodes > 0 else 0

        self.logger.info(
            "工作流执行摘要",
            extra={
                "total_nodes": total_nodes,
                "success_nodes": success_nodes,
                "failed_nodes": total_nodes - success_nodes,
                "success_rate": success_rate,
                "total_duration": total_duration,
                "status": "completed"
            }
        )


async def advanced_workflow_logging_example():
    """高级工作流日志示例"""
    workflow_logger = WorkflowLogger(
        workflow_id="advanced_workflow_001",
        execution_id="exec_20250112_001"
    )

    # 模拟工作流执行
    nodes = [
        ("node_input", "data_input"),
        ("node_validate", "validation"),
        ("node_process", "text_processing"),
        ("node_format", "formatting"),
        ("node_output", "output")
    ]

    for node_id, node_type in nodes:
        start_time = datetime.now()

        try:
            workflow_logger.log_node_start(node_id, node_type)

            # 模拟节点执行
            await asyncio.sleep(0.1)

            if node_id == "node_validate":
                # 模拟验证失败
                raise ValueError("数据格式不正确")

            duration = (datetime.now() - start_time).total_seconds()
            result_count = 10 if node_type == "text_processing" else 1

            workflow_logger.log_node_complete(node_id, duration, result_count)

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            workflow_logger.log_node_error(node_id, e)
            break  # 在实际应用中可能继续执行其他节点

    total_duration = (datetime.now() - workflow_logger.start_time).total_seconds()
    workflow_logger.log_workflow_summary(
        total_nodes=len(nodes),
        success_nodes=1,  # 只有第一个节点成功
        total_duration=total_duration
    )


# ===== 8. 监控和统计 =====
async def logging_monitoring_example():
    """日志监控示例"""
    from src.utils.logger import _log_manager

    # 获取日志系统统计信息
    stats = _log_manager.get_stats()

    logger = get_logger("monitoring_example")
    logger.info(
        "日志系统状态",
        extra={
            "logging_stats": stats,
            "system_health": "healthy"
        }
    )


# ===== 9. 配置文件示例 =====
# config/logging_config.yaml 示例内容:
"""
logging:
  global:
    level: "INFO"
    format: "structured"
    date_format: "%Y-%m-%d %H:%M:%S"
    structured_format: true
    catch_exceptions: true
    include_traceback: true

  outputs:
    console:
      enabled: true
      level: "INFO"
      include_colors: true

    file:
      enabled: true
      level: "DEBUG"
      path: "logs"
      rotation: "daily"
      retention: "30 days"
      compression: "zip"

  performance:
    async_mode: true
    buffer_size: 8192
    flush_interval: 5
    max_queue_size: 10000

  context:
    auto_fields: ["module", "version", "request_id"]
    custom_fields:
      service: "dify_autoopt"
      team: "optimization"

  health:
    max_exception_length: 1000
    error_notification_webhook: null
    auto_retry_on_exception: true
    max_auto_retry_attempts: 3
"""


# ===== 10. 主程序集成示例 =====
async def main():
    """主程序集成示例"""
    # 1. 初始化日志系统（必须在程序开始时调用）
    await setup_logging("config/config.yaml")

    logger = get_logger("main")

    try:
        # 2. 记录应用启动信息
        logger.info(
            "Dify自动优化工具启动",
            extra={
                "version": "1.0.0",
                "environment": "production",
                "python_version": "3.8+",
                "startup_time": datetime.now().isoformat()
            }
        )

        # 3. 设置全局上下文
        from src.utils.logger import _log_manager
        _log_manager.set_global_context(
            application_version="1.0.0",
            deployment_environment="production",
            instance_id="prod_001"
        )

        # 4. 执行各种示例
        await basic_usage_example()
        await structured_logging_example()
        await context_management_example()
        await database_query_example()
        await try_catch_logging_example()
        await workflow_tracing_example()
        await advanced_workflow_logging_example()
        await logging_monitoring_example()

        # 5. 记录应用关闭信息
        logger.info(
            "Dify自动优化工具正常关闭",
            extra={
                "shutdown_time": datetime.now().isoformat(),
                "status": "graceful_shutdown"
            }
        )

    except Exception as e:
        logger.critical(
            "应用运行时发生严重错误",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "shutdown_reason": "error"
            },
            exc_info=True
        )
        raise

    finally:
        # 6. 关闭日志系统
        from src.utils.logger import _log_manager
        await _log_manager.shutdown()


# ===== 11. 测试用例示例 =====
import pytest
from unittest.mock import patch, MagicMock
from src.utils.logger import setup_logging, get_logger


@pytest.fixture
async def test_logger():
    """测试日志器fixture"""
    await setup_logging()
    return get_logger("test_logger")


@pytest.mark.asyncio
async def test_basic_logging(test_logger):
    """基础日志测试"""
    test_logger.info("测试信息日志")
    test_logger.debug("测试调试日志")
    test_logger.warning("测试警告日志")
    test_logger.error("测试错误日志")


@pytest.mark.asyncio
async def test_structured_logging(test_logger):
    """结构化日志测试"""
    test_logger.info(
        "结构化日志测试",
        extra={
            "test_id": "test_001",
            "test_type": "integration",
            "test_data": {"key": "value"},
            "timestamp": datetime.now().isoformat()
        }
    )


@pytest.mark.asyncio
async def test_context_logging(test_logger):
    """上下文日志测试"""
    with log_context(test_id="ctx_test", session_id="sess_123"):
        test_logger.info("上下文中的日志")

        with log_context(operation="nested"):
            test_logger.info("嵌套上下文中的日志")


@pytest.mark.asyncio
async def test_performance_logging():
    """性能日志测试"""
    await database_query_example()


@pytest.mark.asyncio
async def test_exception_logging():
    """异常日志测试"""
    await async_exception_example()

    # 测试同步异常
    with pytest.raises(SystemExit):
        sync_exception_example()


# ===== 12. 性能基准测试 =====
import time
from concurrent.futures import ThreadPoolExecutor


async def benchmark_logging():
    """日志性能基准测试"""
    logger = get_logger("benchmark")

    # 预热
    for i in range(100):
        logger.info(f"warmup {i}")

    # 同步日志性能测试
    start_time = time.time()
    for i in range(10000):
        logger.info(f"test message {i}")
    sync_duration = time.time() - start_time

    logger.info(
        "同步日志性能测试完成",
        extra={
            "messages_count": 10000,
            "duration_seconds": sync_duration,
            "messages_per_second": 10000 / sync_duration
        }
    )

    # 并发日志性能测试
    def concurrent_logging(test_id: int, num_messages: int):
        local_logger = get_logger(f"concurrent_test_{test_id}")
        for i in range(num_messages):
            local_logger.info(f"concurrent message {test_id}-{i}")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(10):
            future = executor.submit(concurrent_logging, i, 1000)
            futures.append(future)

        for future in futures:
            future.result()

    concurrent_duration = time.time() - start_time
    logger.info(
        "并发日志性能测试完成",
        extra={
            "threads": 10,
            "messages_per_thread": 1000,
            "total_messages": 10000,
            "duration_seconds": concurrent_duration,
            "messages_per_second": 10000 / concurrent_duration
        }
    )


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())