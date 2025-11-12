"""
日期: 2025-01-12
作者: rrong
描述: Dify自动化测试工具主程序，集成日志系统示例
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入业务模块
from src.utils.logger import setup_logging, get_logger
from src.config.yaml_loader import load_config


async def main():
    """主程序入口，演示日志系统集成"""

    # 1. 初始化日志系统（必须在程序开始时调用）
    config_path = "config/logging_config.yaml"
    await setup_logging(config_path)

    logger = get_logger("main")

    try:
        # 2. 记录应用启动信息
        logger.info(
            "Dify自动优化工具启动",
            extra={
                "version": "1.0.0",
                "environment": "development",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "startup_time": datetime.now().isoformat()
            }
        )

        # 3. 设置全局上下文
        from src.utils.logger import _log_manager
        _log_manager.set_global_context(
            application_version="1.0.0",
            deployment_environment="development",
            instance_id="dev_001",
            process_id=str(Path.cwd())
        )

        # 4. 记录配置加载
        try:
            config = load_config("config/config.yaml")
            logger.info(
                "配置文件加载成功",
                extra={
                    "config_file": "config/config.yaml",
                    "config_keys": list(config.keys()) if config else []
                }
            )
        except Exception as e:
            logger.warning(
                "配置文件加载失败，使用默认配置",
                extra={
                    "config_file": "config/config.yaml",
                    "error": str(e)
                }
            )

        # 5. 模拟业务流程
        await simulate_workflow_execution()

        # 6. 记录应用关闭信息
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
        return 1

    finally:
        # 7. 关闭日志系统
        from src.utils.logger import _log_manager
        await _log_manager.shutdown()

    return 0


async def simulate_workflow_execution():
    """模拟工作流执行，演示日志功能"""
    from src.utils.logger import (
        log_context,
        log_performance,
        log_workflow_trace,
        log_exception
    )

    # 1. 设置工作流上下文
    workflow_logger = get_logger("workflow")

    with log_context(
        operation="workflow_simulation",
        session_id="sess_20250112_001",
        user_id="demo_user"
    ):
        workflow_logger.info("开始模拟工作流执行")

        # 2. 工作流级别的跟踪
        with log_workflow_trace(
            workflow_id="demo_workflow_001",
            operation="full_execution",
            logger=workflow_logger
        ):
            # 3. 数据输入阶段
            await execute_data_input_phase()

            # 4. 处理阶段
            await execute_processing_phase()

            # 5. 输出生成阶段
            await execute_output_generation_phase()

        # 6. 性能统计
        await log_workflow_performance()


@log_performance("数据输入阶段")
async def execute_data_input_phase():
    """执行数据输入阶段"""
    logger = get_logger("data_input")

    with log_context(phase="input", step="validation"):
        logger.info("开始数据输入验证")

        # 模拟数据验证
        await asyncio.sleep(0.1)

        logger.info(
            "数据验证完成",
            extra={
                "input_records": 100,
                "validated_records": 95,
                "invalid_records": 5
            }
        )


@log_performance("数据处理阶段")
async def execute_processing_phase():
    """执行数据处理阶段"""
    logger = get_logger("processing")

    # 模拟多个处理节点
    nodes = [
        ("text_preprocessing", "文本预处理"),
        ("feature_extraction", "特征提取"),
        ("model_inference", "模型推理")
    ]

    for node_id, node_name in nodes:
        await execute_processing_node(node_id, node_name)


async def execute_processing_node(node_id: str, node_name: str):
    """执行单个处理节点"""
    logger = get_logger(f"processing.{node_id}")

    try:
        logger.info(f"开始执行节点: {node_name}")

        with log_context(node_id=node_id, node_name=node_name):
            # 模拟节点处理
            processing_time = 0.05 + (node_id.count("_") * 0.02)
            await asyncio.sleep(processing_time)

            logger.info(
                f"节点执行完成: {node_name}",
                extra={
                    "processing_time": processing_time,
                    "output_records": 80 if node_id == "model_inference" else 95
                }
            )

    except Exception as e:
        logger.error(
            f"节点执行失败: {node_name}",
            extra={
                "node_id": node_id,
                "error_details": str(e)
            },
            exc_info=True
        )
        raise


@log_performance("输出生成阶段")
async def execute_output_generation_phase():
    """执行输出生成阶段"""
    logger = get_logger("output")

    try:
        logger.info("开始生成输出结果")

        # 模拟输出生成
        await asyncio.sleep(0.08)

        logger.info(
            "输出生成完成",
            extra={
                "output_files": 3,
                "file_types": ["csv", "json", "html"],
                "total_size_mb": 2.5
            }
        )

    except Exception as e:
        logger.error(
            "输出生成失败",
            extra={
                "error_details": str(e)
            },
            exc_info=True
        )


async def log_workflow_performance():
    """记录工作流性能统计"""
    logger = get_logger("performance")

    # 获取日志系统统计
    from src.utils.logger import _log_manager
    stats = _log_manager.get_stats()

    logger.info(
        "工作流性能统计",
        extra={
            "logging_stats": stats,
            "workflow_performance": {
                "total_nodes": 3,
                "successful_nodes": 3,
                "failed_nodes": 0,
                "success_rate": 1.0
            },
            "system_metrics": {
                "memory_usage_mb": 128,  # 模拟值
                "cpu_usage_percent": 15,  # 模拟值
                "disk_usage_percent": 45  # 模拟值
            }
        }
    )


@log_exception(reraise=False)
async def simulate_error_scenario():
    """模拟错误场景"""
    logger = get_logger("error_simulation")

    logger.warning("开始错误场景模拟")

    # 模拟网络超时
    await asyncio.sleep(0.02)

    # 模拟一个连接错误
    raise ConnectionError("模拟网络连接超时")


if __name__ == "__main__":
    # 运行主程序
    exit_code = asyncio.run(main())
    sys.exit(exit_code)