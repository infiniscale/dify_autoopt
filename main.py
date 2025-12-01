"""
应用入口（根目录）
- 单一配置文件：--config 指定（默认自动尝试 config/config.yaml，不存在则使用内置默认）
- 日志：自动探测 logging_config.yaml 或 config.yaml 的 logging
- .env：在初始化日志之前加载
"""

import os
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import argparse

try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # 兼容未安装 dotenv 的环境
    def _load_dotenv(*args, **kwargs):
        return False

from src.utils.logger import (
    setup_logging,
    get_logger,
    log_context,
    log_performance,
    log_workflow_trace,
    log_exception,
)
from src.config.loaders.config_loader import ConfigLoader


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dify自动化测试与提示词优化工具")
    parser.add_argument("--mode", "-m", choices=["test", "optimize", "report"], default="test")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="环境配置文件路径（未设置则自动尝试 config/config.yaml，不存在时使用内置默认配置）",
    )
    parser.add_argument("--catalog", default="config/workflow_repository.yaml", help="工作流目录文件路径（可选）")
    parser.add_argument("--report", default=None, help="报告输出路径（json，可选）")
    parser.add_argument("--log-config", default=None, help="日志配置文件路径（留空自动探测）")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="覆盖配置项，格式: a.b.c=value，可重复（仅影响本次运行时）",
    )
    args = parser.parse_args(argv)

    # 0. 预加载 .env 环境变量
    try:
        _load_dotenv(dotenv_path=Path('.env'), override=False)
    except Exception:
        pass

    # 1. 日志初始化
    await setup_logging(args.log_config)
    logger = get_logger("main")

    try:
        # 2. 启动日志
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

        # 3. 全局上下文初始化（初始）
        from src.utils.logger import _log_manager
        _log_manager.set_global_context(
            application_version="1.0.0",
            deployment_environment="development",
            instance_id="dev_001",
            process_id=str(Path.cwd())
        )

        # 4. 加载配置与引导
        from src.config.bootstrap import bootstrap_from_unified
        # 4.0 确定有效配置路径
        effective_config: Path
        if args.config:
            candidate = Path(args.config)
            if candidate.exists():
                effective_config = candidate
            else:
                logger.warning("指定的配置文件不存在，尝试默认路径", extra={"config": str(candidate)})
                effective_config = Path("config/config.yaml")
        else:
            effective_config = Path("config/config.yaml")
        if not effective_config.exists():
            # 使用内置默认配置
            import yaml, tempfile
            default_cfg = {
                "meta": {"version": "1.0.0", "environment": "development"},
                "dify": {"base_url": None},
                "auth": {},
                "variables": {},
                "workflows": [],
                "execution": {"concurrency": 5, "timeout": 300, "retry_count": 3},
                "optimization": {"strategy": "auto", "max_iterations": 3},
                "io_paths": {"output_dir": "./outputs", "logs_dir": "./logs"},
                "logging": {"level": "INFO", "format": "simple", "console_enabled": True, "file_enabled": True},
            }
            tf = tempfile.NamedTemporaryFile(prefix="default_config_", suffix=".yaml", delete=False)
            Path(tf.name).write_text(yaml.safe_dump(default_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            effective_config = Path(tf.name)
            logger.info("未找到配置文件，已使用内置默认配置", extra={"effective_config": str(effective_config)})

        # 4.1 应用 --set 覆盖项
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
                base_raw = {}
                try:
                    if effective_config.exists():
                        import yaml
                        base_raw = yaml.safe_load(effective_config.read_text(encoding="utf-8")) or {}
                except Exception:
                    base_raw = {}
                raw = base_raw
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

        # 4.2 引导运行态
        runtime = bootstrap_from_unified(effective_config)
        meta = runtime.app.meta or {}
        _log_manager.set_global_context(
            deployment_environment=meta.get("environment", "development"),
            application_version=meta.get("version", "1.0.0"),
            dify_base_url=runtime.dify_base_url,
        )
        logger.info(
            "配置初始化完成",
            extra={
                "config_file": str(effective_config),
                "meta": {"version": meta.get("version"), "environment": meta.get("environment")},
                "dify": {"base_url": runtime.dify_base_url},
                "auth": {"has_api_key": runtime.has_token_auth, "has_username_password": runtime.has_password_auth},
                "workflows": runtime.workflows_count,
                "execution": runtime.app.execution or {},
                "optimization": runtime.app.optimization or {},
                "io_paths": runtime.app.io_paths or {},
                "logging": runtime.app.logging or {},
            },
        )

        # 5. 模式执行
        if args.mode == "test":
            results = await run_test_mode(args.catalog, str(effective_config))
            if args.report:
                from src.report import generate_report, save_report_json
                rep = generate_report([r for r in results])
                save_report_json(rep, args.report)
                logger.info("测试报告已生成", extra={"path": args.report})

        elif args.mode == "optimize":
            logger.info("进入优化模式（占位实现）", extra={"mode": args.mode})
            await simulate_workflow_execution()

        elif args.mode == "report":
            logger.info("进入报告模式（占位实现）", extra={"mode": args.mode})
            await simulate_workflow_execution()

        logger.info("Dify自动优化工具正常关闭", extra={"shutdown_time": datetime.now().isoformat(), "status": "graceful_shutdown"})
    except Exception as e:
        logger.critical(
            "应用运行时发生严重错误",
            extra={"error_type": type(e).__name__, "error_message": str(e), "shutdown_reason": "error"},
            exc_info=True,
        )
        return 1
    finally:
        from src.utils.logger import _log_manager
        await _log_manager.shutdown()
    return 0


async def run_test_mode(catalog_path: str, unified_config_path: str):
    logger = get_logger("cli.test")
    from src.workflow import discover_workflows, run_workflow
    from src.workflow.runner import run_inline_workflow

    p = Path(catalog_path)
    if not p.exists():
        logger.warning("catalog 文件不存在，尝试使用统一配置中的 workflows", extra={"catalog": catalog_path})
        try:
            from src.config.bootstrap import get_runtime
            rt = get_runtime()
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
        loader = ConfigLoader()
        catalog = loader.load_catalog(p)
        wfs = discover_workflows(catalog)
        logger.info("发现工作流", extra={"count": len(wfs)})
        results = []
        for wf in wfs[:3]:
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


@log_performance("数据输入阶段")
async def execute_data_input_phase():
    logger = get_logger("data_input")
    with log_context(phase="input", step="validation"):
        logger.info("开始数据输入验证")
        await asyncio.sleep(0.1)
        logger.info("数据验证完成", extra={"input_records": 100, "validated_records": 95, "invalid_records": 5})


@log_performance("数据处理阶段")
async def execute_processing_phase():
    logger = get_logger("processing")
    nodes = [("text_preprocessing", "文本预处理"), ("feature_extraction", "特征提取"), ("model_inference", "模型推理")]
    for node_id, node_name in nodes:
        await execute_processing_node(node_id, node_name)


async def execute_processing_node(node_id: str, node_name: str):
    logger = get_logger(f"processing.{node_id}")
    try:
        logger.info(f"开始执行节点: {node_name}")
        with log_context(node_id=node_id, node_name=node_name):
            processing_time = 0.05 + (node_id.count("_") * 0.02)
            await asyncio.sleep(processing_time)
            logger.info(f"节点执行完成: {node_name}", extra={"processing_time": processing_time, "output_records": 80 if node_id == "model_inference" else 95})
    except Exception as e:
        logger.error(f"节点执行失败: {node_name}", extra={"node_id": node_id, "error_details": str(e)}, exc_info=True)
        raise


@log_performance("输出生成阶段")
async def execute_output_generation_phase():
    logger = get_logger("output")
    try:
        logger.info("开始生成输出结果")
        await asyncio.sleep(0.08)
        logger.info("输出生成完成", extra={"output_files": 3, "file_types": ["csv", "json", "html"], "total_size_mb": 2.5})
    except Exception as e:
        logger.error("输出生成失败", extra={"error_details": str(e)}, exc_info=True)


async def log_workflow_performance():
    logger = get_logger("performance")
    from src.utils.logger import _log_manager
    stats = _log_manager.get_stats()
    logger.info(
        "工作流性能统计",
        extra={
            "logging_stats": stats,
            "workflow_performance": {"total_nodes": 3, "successful_nodes": 3, "failed_nodes": 0, "success_rate": 1.0},
            "system_metrics": {"memory_usage_mb": 128, "cpu_usage_percent": 15, "disk_usage_percent": 45},
        },
    )


@log_exception(reraise=False)
async def simulate_error_scenario():
    logger = get_logger("error_simulation")
    logger.warning("开始错误场景模拟")
    await asyncio.sleep(0.02)
    raise ConnectionError("模拟网络连接超时")


async def simulate_workflow_execution():
    workflow_logger = get_logger("workflow")
    with log_context(operation="workflow_simulation", session_id="sess_20250112_001", user_id="demo_user"):
        workflow_logger.info("开始模拟工作流执行")
        with log_workflow_trace(workflow_id="demo_workflow_001", operation="full_execution", logger=workflow_logger):
            await execute_data_input_phase()
            await execute_processing_phase()
            await execute_output_generation_phase()
        await log_workflow_performance()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
