"""
日期: 2025-01-12
作者: rrong
描述: Dify自动化测试工具主程序，集成日志系统示例
"""

import os
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入业务模块
from src.utils.logger import (
    setup_logging,
    get_logger,
    log_context,
    log_performance,
    log_workflow_trace,
    log_exception,
)
from src.config.loaders.config_loader import FileSystemReader, ConfigLoader
import argparse
try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    def _load_dotenv(*args, **kwargs):
        return False


async def main(argv: list[str] | None = None) -> int:
    """主程序入口，提供最小可用 CLI 与日志演示。"""

    # CLI 参数
    parser = argparse.ArgumentParser(description="Dify自动化测试与提示词优化工具")
    parser.add_argument("--mode", "-m", choices=["test", "optimize", "report"], default="test")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="环境配置文件路径")
    parser.add_argument("--catalog", default="config/workflow_repository.yaml", help="工作流目录文件路径（可选）")
    parser.add_argument("--report", default=None, help="报告输出路径（json，可选）")
    parser.add_argument("--log-config", default=None, help="日志配置文件路径（留空将自动探测 logging_config.yaml 或 config.yaml）")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="覆盖配置项，格式: a.b.c=value，可重复（仅影响运行时配置，引导后生效）",
    )
    args = parser.parse_args(argv)

    # 0. 预加载 .env 环境变量，供 ${VAR} 展开
    try:
        _load_dotenv(dotenv_path=Path('.env'), override=False)
    except Exception:
        pass

    # 1. 初始化日志系统（必须在程序开始时调用）
    # 如果未指定，将自动探测 logging_config.yaml 或 config.yaml
    await setup_logging(args.log_config)

    logger = get_logger("main")

    try:
        # 2. 记录应用启动信息
        logger.info(
            "Dify自动优化工具启动",
            extra={
                "version": "1.0.0",
                "environment": "development",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "startup_time": datetime.now().isoformat(),
                "env": {
                    "env_file": ".env",
                    "has_DIFY_API_TOKEN": bool(os.getenv("DIFY_API_TOKEN")),
                    "has_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
                }
            }
        )

        # 3. 设置全局上下文（初始）
        from src.utils.logger import _log_manager
        _log_manager.set_global_context(
            application_version="1.0.0",
            deployment_environment="development",
            instance_id="dev_001",
            process_id=str(Path.cwd())
        )

        # 4. 记录配置加载
        # 4. 配置加载与初始化（统一 config.yaml）
        from src.config.bootstrap import bootstrap_from_unified
        effective_config = Path(args.config)

        # 4.1 处理 --set 覆盖项（将覆盖写入临时文件，仅影响本次运行时引导）
        if args.set:
            try:
                import yaml, tempfile

                def _apply(d: dict, path: str, value: str):
                    cur = d
                    parts = path.split('.') if path else []
                    for key in parts[:-1]:
                        if key not in cur or not isinstance(cur[key], dict):
                            cur[key] = {}
                        cur = cur[key]
                    # 尝试将字符串转换为基本类型
                    v = value
                    if value.lower() in {"true", "false"}:
                        v = value.lower() == "true"
                    else:
                        try:
                            if "." in value:
                                v = float(value)
                            else:
                                v = int(value)
                        except Exception:
                            v = value
                    if parts:
                        cur[parts[-1]] = v

                raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
                for item in args.set:
                    if "=" not in item:
                        continue
                    k, v = item.split("=", 1)
                    _apply(raw, k.strip(), v.strip())

                tf = tempfile.NamedTemporaryFile(prefix="config_overrides_", suffix=".yaml", delete=False)
                Path(tf.name).write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
                effective_config = Path(tf.name)
                logger.info("已应用配置覆盖项", extra={"overrides": args.set, "effective_config": str(effective_config)})
            except Exception as e:
                logger.warning("应用配置覆盖项失败，继续使用原始配置", extra={"error": str(e)})
        try:
            runtime = bootstrap_from_unified(effective_config)

            # 全局上下文增强
            meta = runtime.app.meta or {}
            _log_manager.set_global_context(
                deployment_environment=meta.get("environment", "development"),
                application_version=meta.get("version", "1.0.0"),
                dify_base_url=runtime.dify_base_url,
            )

            # 详细日志
            logger.info(
                "配置初始化完成",
                extra={
                    "config_file": str(effective_config),
                    "meta": {
                        "version": meta.get("version"),
                        "environment": meta.get("environment"),
                    },
                    "dify": {"base_url": runtime.dify_base_url},
                    "auth": {
                        "has_api_key": runtime.has_token_auth,
                        "has_username_password": runtime.has_password_auth,
                    },
                    "workflows": runtime.workflows_count,
                    "execution": runtime.app.execution or {},
                    "optimization": runtime.app.optimization or {},
                    "io_paths": runtime.app.io_paths or {},
                    "logging": runtime.app.logging or {},
                },
            )
        except Exception as e:
            logger.warning(
                "配置文件加载失败，使用默认配置",
                extra={
                    "config_file": args.config,
                    "error": str(e)
                }
            )

        # 5. 按模式执行最小流程
        if args.mode == "test":
            results = await run_test_mode(args.catalog, args.config)
            if args.report:
                from src.report import generate_report, save_report_json
                rep = generate_report([r for r in results])
                save_report_json(rep, args.report)
                logger.info("测试报告已生成", extra={"path": args.report})

        elif args.mode == "optimize":
            logger.info("进入优化模式（占位实现）", extra={"mode": args.mode})
            # 仍使用模拟流程，以保持运行演示
            await simulate_workflow_execution()

        elif args.mode == "report":
            logger.info("进入报告模式（占位实现）", extra={"mode": args.mode})
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


# ============== 附加：最小 test 模式编排 ==============
async def run_test_mode(catalog_path: str, unified_config_path: str):
    """最小可运行 test 模式：尝试加载 catalog 并运行 stub 工作流。

    若 catalog 不存在，回退到内建模拟工作流执行。
    返回值为结果字典列表。
    """
    logger = get_logger("cli.test")
    from pathlib import Path
    from src.workflow import discover_workflows, run_workflow

    p = Path(catalog_path)
    if not p.exists():
        logger.warning("catalog 文件不存在，尝试使用统一配置中的 workflows", extra={"catalog": catalog_path})
        try:
            from src.config.bootstrap import get_runtime
            rt = get_runtime()
            from src.workflow.runner import run_inline_workflow
            results = []
            if rt.workflows_count == 0:
                logger.warning("统一配置中未定义 workflows，回退到模拟流程")
                await simulate_workflow_execution()
                return []
            for w in rt.app.workflows[:3]:
                r = await run_inline_workflow(w.id, w.name or w.id)
                results.append({
                    "workflow_id": r.workflow_id,
                    "label": r.label,
                    "status": r.status,
                    "started_at": r.started_at,
                    "ended_at": r.ended_at,
                    "metrics": r.metrics,
                })
            return results
        except Exception as e:
            logger.warning("使用统一配置运行失败，回退到模拟流程", extra={"error": str(e)})
            await simulate_workflow_execution()
            return []

    try:
        # 使用 ConfigLoader 解析 catalog
        loader = ConfigLoader()
        catalog = loader.load_catalog(p)
        wfs = discover_workflows(catalog)
        logger.info("发现工作流", extra={"count": len(wfs)})

        results = []
        for wf in wfs[:3]:  # 限制最多执行3个，防止时间过长
            r = await run_workflow(wf)
            results.append({
                "workflow_id": r.workflow_id,
                "label": r.label,
                "status": r.status,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "metrics": r.metrics,
            })
        return results
    except Exception as e:
        logger.warning("catalog 加载或运行失败，回退到模拟流程", extra={"error": str(e)})
        await simulate_workflow_execution()
        return []
