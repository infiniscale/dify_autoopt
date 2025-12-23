"""
应用入口（根目录）
- 单一配置文件：--config 指定（默认自动尝试 config/config.yaml，不存在则使用内置默认）
- 日志：自动探测 logging_config.yaml 或 config.yaml 的 logging
- .env：在初始化日志之前加载
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import requests

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


def _expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser()


def _config_has_logging(path: Path) -> bool:
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return isinstance(raw, dict) and "logging" in raw
    except Exception:
        return False


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dify自动化测试与提示词优化工具")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["run", "opt", "all", "loop"],
        default="run",
        help="运行模式：run=仅执行工作流并保存结果；opt=基于已有运行结果生成优化建议；all=先执行后优化；loop=运行-优化-发布循环",
    )
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
    parser.add_argument("--max-cycles", type=int, default=None, help="loop 模式：最大迭代轮数（默认取 optimization.max_iterations）")
    parser.add_argument("--loop-no-patch", type=int, default=1, help="loop 模式：连续无补丁轮数达到此值则退出")
    parser.add_argument("--target-failure-rate", type=float, default=None, help="loop 模式：当失败率不高于该值时退出")
    args = parser.parse_args(argv)

    # 0. 预加载 .env 环境变量
    try:
        _load_dotenv(dotenv_path=Path('.env'), override=False)
    except Exception:
        pass

    logger = None
    try:
        # 1. 确定有效配置路径（提前处理，用于日志初始化）
        deferred_logs: list[tuple[str, str, dict]] = []

        def _defer(level: str, message: str, extra: dict) -> None:
            deferred_logs.append((level, message, extra))

        from src.config.bootstrap import bootstrap_from_unified
        effective_config: Path
        if args.config:
            candidate = _expand_path(args.config)
            if candidate.exists():
                effective_config = candidate
            else:
                _defer("warning", "指定的配置文件不存在，尝试默认路径", {"config": str(candidate)})
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
            _defer("info", "未找到配置文件，已使用内置默认配置", {"effective_config": str(effective_config)})

        # 1.1 应用 --set 覆盖项（生成临时配置文件）
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
                _defer("info", "已应用配置覆盖项", {"overrides": args.set, "effective_config": str(effective_config)})
            except Exception as e:
                _defer("warning", "应用配置覆盖项失败，继续使用原始配置", {"error": str(e)})

        # 2. 日志初始化（优先使用显式 --log-config，其次使用有效配置文件的 logging 段）
        log_config_path: str | None = args.log_config
        if log_config_path:
            log_config_path = str(_expand_path(log_config_path))
        elif effective_config.exists() and _config_has_logging(effective_config):
            log_config_path = str(effective_config)
        await setup_logging(log_config_path)
        logger = get_logger("main")

        for level, message, extra in deferred_logs:
            getattr(logger, level)(message, extra=extra)

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
        # 4.1 引导运行态
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

        # 4.3 Dify 登录与状态校验
        async def _validate_token_async(base_url: str, token: str, timeout: int = 10) -> bool:
            def _do_validate() -> bool:
                try:
                    url = f"{base_url.rstrip('/')}/console/api/apps"
                    headers = {"Authorization": f"Bearer {token}"}
                    params = {"page": 1, "limit": 1}
                    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
                    return resp.status_code == 200
                except Exception:
                    return False
            return await asyncio.to_thread(_do_validate)

        try:
            base_url = runtime.dify_base_url
            auth_cfg = runtime.app.auth or {}
            if base_url and (auth_cfg.get("api_key") or (auth_cfg.get("username") and auth_cfg.get("password"))):
                logger.info("开始Dify登录校验", extra={"base_url": base_url, "auth_mode": "api_key" if auth_cfg.get("api_key") else "password"})

                access_token: str | None = None
                if auth_cfg.get("api_key"):
                    access_token = str(auth_cfg.get("api_key"))
                else:
                    from src.auth.login import DifyAuthClient
                    client = DifyAuthClient(base_url=base_url, email=auth_cfg.get("username"), password=auth_cfg.get("password"), timeout=10)
                    # 使用线程避免阻塞事件循环
                    def _do_login():
                        return client.login()
                    login_result = await asyncio.to_thread(_do_login)
                    if isinstance(login_result, dict):
                        # login() 约定返回包含 access_token 的 data 字典
                        access_token = login_result.get("access_token") or login_result.get("data", {}).get("access_token")

                if access_token:
                    ok = await _validate_token_async(base_url, access_token)
                    if ok:
                        logger.info("Dify 登录验证成功")
                        # 登录成功后获取并展示 App 列表（分页汇总）
                        try:
                            from src.workflow import list_all_apps, export_app_dsl
                            # 1) 拉取全部应用列表（线程中执行同步 requests）
                            def _fetch_all():
                                return list_all_apps(base_url=base_url, limit=30, name="", is_created_by_me=False, token=access_token, timeout=10)
                            apps = await asyncio.to_thread(_fetch_all)
                            # 提取部分关键信息进行展示
                            names = []
                            ids = []
                            for it in apps:
                                if isinstance(it, dict):
                                    n = it.get("name") or (it.get("app") or {}).get("name") if isinstance(it.get("app"), dict) else None
                                    i = it.get("id") or (it.get("app") or {}).get("id") if isinstance(it.get("app"), dict) else None
                                    if n:
                                        names.append(n)
                                    if i:
                                        ids.append(i)
                            logger.info(
                                "已获取应用列表",
                                extra={
                                    "count": len(apps),
                                    "sample_names": names[:10],
                                    "sample_ids": ids[:10],
                                },
                            )

                            # 2) 遍历配置中的 workflows，导出对应 DSL 至配置 io_paths.output_dir
                            workflow_ids = []
                            try:
                                workflow_ids = [w.id for w in (runtime.app.workflows or []) if getattr(w, 'id', None)]
                            except Exception:
                                workflow_ids = []

                            exported_paths = []
                            if workflow_ids:
                                if args.mode == "opt":
                                    logger.info("opt 模式仅使用已有运行结果，跳过在线 DSL 导出与工作流执行", extra={"count": len(workflow_ids)})
                                else:
                                    logger.info("开始导出工作流 DSL", extra={"count": len(workflow_ids)})

                                    def _export_one(app_id: str):
                                        return export_app_dsl(app_id, base_url=base_url, token=access_token, include_secret=False)

                                    for wid in workflow_ids:
                                        try:
                                            p = await asyncio.to_thread(_export_one, wid)
                                            exported_paths.append(str(p))
                                        except Exception as ex:
                                            logger.warning("导出 DSL 失败", extra={"workflow_id": wid, "error": str(ex)})

                                    logger.info("DSL 导出完成", extra={"exported": len(exported_paths), "paths": exported_paths[:10]})
                            else:
                                logger.info("配置中未发现 workflows，跳过 DSL 导出")
                        except Exception as e:
                            logger.warning("获取应用列表或导出 DSL 失败", extra={"error": str(e)})
                    else:
                        logger.warning("Dify 登录验证失败，请检查凭据或网络")
                else:
                    logger.warning("未获取到访问令牌，跳过验证")
            else:
                logger.info("未配置 Dify 认证信息，跳过登录校验")
        except Exception as e:
            logger.warning("Dify 登录校验出现异常", extra={"error": str(e)})
        # 5. 模式执行
        if args.mode == "run":
            logger.info("进入运行模式（仅执行工作流，不做优化）", extra={"mode": args.mode})
            results = await run_optimize_mode(run_workflows=True, optimize=False)
            if args.report and results is not None:
                from src.report import generate_report, save_report_json
                rep = generate_report([r for r in results])
                save_report_json(rep, args.report)
                logger.info("测试报告已生成", extra={"path": args.report})

        elif args.mode == "opt":
            logger.info("进入优化模式（仅使用历史运行结果）", extra={"mode": args.mode})
            await run_optimize_mode(run_workflows=False, optimize=True)

        elif args.mode == "all":
            logger.info("进入全流程模式（先运行后优化）", extra={"mode": args.mode})
            await run_optimize_mode(run_workflows=True, optimize=True)
        elif args.mode == "loop":
            logger.info(
                "进入循环优化模式（运行-优化-发布-再运行）",
                extra={
                    "mode": args.mode,
                    "max_cycles": args.max_cycles,
                    "no_patch_rounds": args.loop_no_patch,
                    "target_failure_rate": args.target_failure_rate,
                },
            )
            from src.optimizer import run_optimize_loop
            from src.config.bootstrap import get_runtime

            rt = get_runtime()
            opt_cfg = (rt.app.optimization or {}) if rt and getattr(rt, "app", None) else {}
            cfg_max_cycles = opt_cfg.get("max_iterations")
            cfg_exit_ratio = opt_cfg.get("exit_ratio")

            max_cycles = args.max_cycles if args.max_cycles is not None else (cfg_max_cycles if isinstance(cfg_max_cycles, int) and cfg_max_cycles > 0 else 3)
            exit_ratio = cfg_exit_ratio if isinstance(cfg_exit_ratio, (int, float)) and 0 <= cfg_exit_ratio <= 1 else None

            loop_summary = run_optimize_loop(
                max_cycles=max_cycles,
                allow_no_patch_rounds=args.loop_no_patch,
                target_failure_rate=args.target_failure_rate,
                exit_ratio=exit_ratio,
            )
            try:
                logger.info("循环优化完成", extra={"summary": loop_summary[:5], "total_cycles": len(loop_summary)})
            except Exception:
                pass

        logger.info("Dify自动优化工具正常关闭", extra={"shutdown_time": datetime.now().isoformat(), "status": "graceful_shutdown"})
        return 0
    except Exception as e:
        if logger:
            logger.critical(
                "应用运行时发生严重错误",
                extra={"error_type": type(e).__name__, "error_message": str(e), "shutdown_reason": "error"},
                exc_info=True,
            )
        else:
            print(f"Fatal error before logger initialized: {type(e).__name__}: {e}")
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


def _resolve_reference_file(opt_cfg: dict, workflow_id: str) -> str | None:
    """Resolve reference expectation file for a workflow."""
    if not opt_cfg:
        return None
    ref_path = opt_cfg.get("reference_path")
    if ref_path:
        p = Path(ref_path)
        return str(p) if p.exists() else None
    ref_dir = opt_cfg.get("reference_dir")
    if not ref_dir:
        return None
    base = Path(ref_dir)
    candidates = [
        base / f"{_safe_dirname_from_id(workflow_id)}.yaml",
        base / f"{_safe_dirname_from_id(workflow_id)}.yml",
        base / f"{_safe_dirname_from_id(workflow_id)}.json",
        base / f"{workflow_id}.yaml",
        base / f"{workflow_id}.yml",
        base / f"{workflow_id}.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _load_runs_from_disk(workflow_id: str, output_dir: str | Path) -> list[dict]:
    """Load persisted run results from disk for offline optimization."""
    root = Path(output_dir).expanduser()
    run_dir = root / _safe_dirname_from_id(workflow_id) / "runs"
    if not run_dir.exists():
        return []
    runs = []
    for fp in sorted(run_dir.glob("run_*.json")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            runs.append(payload.get("result", payload))
        except Exception:
            continue
    return runs


def _safe_dirname_from_id(app_id: str) -> str:
    """Filesystem-safe directory name (local copy to avoid early workflow imports)."""
    try:
        import re

        name = str(app_id)
        name = name.replace("/", "_").replace("\\", "_")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        name = name.strip("._-") or "app"
        return name
    except Exception:
        return "app"


async def run_optimize_mode(*, run_workflows: bool = True, optimize: bool = True):
    """
    Execute workflows and optionally generate prompt optimization patches.

    Args:
        run_workflows: Whether to execute workflows to produce fresh runs (persisted to disk).
        optimize: Whether to run the prompt optimizer using available runs.
    """
    from src.config.bootstrap import get_runtime
    from src.optimizer import PromptOptimizer, load_workflow_yaml
    from src.workflow import execute_workflow_from_config

    rt = get_runtime()
    logger = get_logger("cli.optimize")

    workflows = rt.app.workflows or []
    if not workflows:
        logger.info("未配置 workflows，跳过优化")
        return

    output_dir = (rt.app.io_paths or {}).get("output_dir") or "./outputs"
    exec_timeout = rt.app.execution.get("timeout", 60) if rt.app.execution else 60
    exec_retry = rt.app.execution.get("retry_count", 0) if rt.app.execution else 0
    exec_meta = {
        "execution_config": {
            "timeout": exec_timeout,
            "retry_count": exec_retry,
            "concurrency": (rt.app.execution or {}).get("concurrency"),
            "response_mode": "blocking",
        },
        "source": "config",
    }
    ref_cfg = rt.app.optimization or {}

    optimizer = PromptOptimizer(default_output_root=output_dir, llm_config=ref_cfg.get("llm"))
    summaries = []
    run_records: list[dict] = []

    for wf in workflows:
        wid = getattr(wf, "id", None)
        if not wid:
            continue
        wid = str(wid)
        logger.info(
            "开始处理工作流",
            extra={"workflow_id": wid, "run_workflows": run_workflows, "optimize": optimize},
        )

        run_results: list[dict] = []
        if run_workflows:
            try:
                def _run_one():
                    return execute_workflow_from_config(
                        wid,
                        base_url=rt.dify_base_url,
                        timeout=exec_timeout,
                        persist_results=True,
                        retry_count=exec_retry,
                        persist_metadata=exec_meta,
                    )
                run_results = await asyncio.to_thread(_run_one)
            except Exception as ex:
                logger.warning("执行工作流失败", extra={"workflow_id": wid, "error": str(ex)})

        if not run_results:
            try:
                run_results = _load_runs_from_disk(wid, output_dir)
            except Exception as ex:
                logger.warning("加载历史运行结果失败", extra={"workflow_id": wid, "error": str(ex)})

        # 记录运行结果供报告使用
        try:
            from src.optimizer.prompt_optimizer import _status_from_run  # local import to avoid early logger use
        except Exception:
            _status_from_run = None  # type: ignore

        for run in run_results:
            metrics = {}
            if isinstance(run, dict):
                metrics = run.get("metrics") or {}
                if not metrics:
                    if isinstance(run.get("data"), dict):
                        metrics = run.get("data", {}).get("metrics", {}) or {}
                    elif isinstance(run.get("result"), dict) and isinstance(run["result"].get("data"), dict):
                        metrics = run["result"]["data"].get("metrics", {}) or {}
            status = "unknown"
            if _status_from_run:
                try:
                    status = _status_from_run(run)
                except Exception:
                    status = str(run.get("status", "unknown")) if isinstance(run, dict) else "unknown"
            elif isinstance(run, dict):
                status = str(run.get("status", "unknown"))
            run_records.append({"workflow_id": wid, "status": status, "metrics": metrics})

        if not optimize:
            summaries.append({"workflow_id": wid, "runs": len(run_results)})
            logger.info("运行完成（未执行优化）", extra={"workflow_id": wid, "runs": len(run_results)})
            continue

        try:
            yaml_tree, yaml_path = load_workflow_yaml(wid, output_dir=output_dir)
        except FileNotFoundError as ex:
            logger.warning("未找到工作流 DSL", extra={"workflow_id": wid, "error": str(ex)})
            yaml_tree, yaml_path = None, None
        except Exception as ex:
            logger.warning("加载工作流 DSL 失败", extra={"workflow_id": wid, "error": str(ex)})
            yaml_tree, yaml_path = None, None

        reference_path = _resolve_reference_file(ref_cfg, wid)

        reference_texts: list[str | None] | None = None
        if optimize and run_results:
            wf_ref = getattr(wf, "reference", None)
            if wf_ref is not None:
                ref_list = wf_ref if isinstance(wf_ref, list) else [wf_ref]
                if len(ref_list) == 1 and len(run_results) > 1:
                    ref_list = ref_list * len(run_results)
                elif len(ref_list) != len(run_results):
                    logger.warning(
                        "reference 数量与运行结果数量不匹配，按最小长度对齐",
                        extra={"workflow_id": wid, "references": len(ref_list), "runs": len(run_results)},
                    )
                reference_texts = []
                for path in ref_list[: len(run_results)]:
                    if not path:
                        reference_texts.append(None)
                        continue
                    try:
                        p = Path(path)
                        reference_texts.append(p.read_text(encoding="utf-8") if p.exists() else None)
                    except Exception as ex:
                        logger.warning("读取 reference 失败", extra={"workflow_id": wid, "reference": path, "error": str(ex)})
                        reference_texts.append(None)

        if not run_results or yaml_tree is None or yaml_path is None:
            summaries.append({"workflow_id": wid, "runs": len(run_results), "patches": 0})
            continue

        try:
            opt_strategy = str(ref_cfg.get("strategy") or "").lower()
            use_prompt_state = bool(run_workflows and opt_strategy == "prompt_state")
            if use_prompt_state:
                try:
                    logger.info("PromptState optimizer enabled (offline)", extra={"workflow_id": wid})
                except Exception:
                    pass

                def _select_representative_run(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
                    for item in results:
                        try:
                            status = _status_from_run(item) if _status_from_run else str(item.get("status", "")).lower()
                        except Exception:
                            status = str(item.get("status", "")).lower() if isinstance(item, dict) else "unknown"
                        if status not in {"success", "succeeded"}:
                            return item
                    return results[0] if results else {}

                sample_run = _select_representative_run(run_results)

                def _offline_run_once(prompt_text: str, *_args, **_kwargs):
                    llm = (ref_cfg.get("llm") or {}) if isinstance(ref_cfg, dict) else {}
                    url = llm.get("url")
                    model = llm.get("model")
                    api_key = llm.get("api_key") or llm.get("key")
                    try:
                        timeout = float(llm.get("timeout", 60))
                    except Exception:
                        timeout = 60

                    if not url or not model:
                        return sample_run or {}

                    try:
                        workflow_input = None
                        if isinstance(sample_run, dict):
                            workflow_input = (
                                    sample_run.get("input")
                                    or sample_run.get("inputs")
                                    or sample_run.get("workflow_input")
                            )

                        user_content = ""
                        if workflow_input is not None:
                            user_content += f"WORKFLOW_INPUT:\n{workflow_input}\n\n"
                        user_content += f"PROMPT:\n{prompt_text}\n\nReturn the workflow output text only."

                        payload = {
                            "model": model,
                            "messages": [
                                {"role": "system",
                                 "content": "You simulate the workflow output based on input and prompt."},
                                {"role": "user", "content": user_content},
                            ],
                            "stream": False,
                        }
                        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                        resp.raise_for_status()
                        data = resp.json()

                        text = ""
                        try:
                            text = (((data.get("choices") or [])[0] or {}).get("message") or {}).get("content") or ""
                        except Exception:
                            text = str(data)

                        return {"output": text, "status": "success"}
                    except Exception:
                        return sample_run or {}

                max_steps = ref_cfg.get("max_iterations")
                max_steps = int(max_steps) if isinstance(max_steps, int) and max_steps > 0 else 5
                beam_width = ref_cfg.get("beam_width")
                beam_width = int(beam_width) if isinstance(beam_width, int) and beam_width > 0 else 1
                variant_count = ref_cfg.get("variant_count")
                variant_count = int(variant_count) if isinstance(variant_count, int) and variant_count > 0 else 1
                action_budget = ref_cfg.get("action_budget")
                max_concurrency = ref_cfg.get("concurrency")
                max_concurrency = int(max_concurrency) if isinstance(max_concurrency,
                                                                     int) and max_concurrency > 0 else None
                checkpoint_dir = Path(output_dir) / _safe_dirname_from_id(wid) / "prompt_state"
                optimizer_cm = rt.optimizer_concurrency if rt else None

                report = optimizer.optimize_with_prompt_state(
                    workflow_id=wid,
                    run_results=run_results,
                    workflow_yaml=(yaml_tree, yaml_path),
                    reference_path=reference_path,
                    reference_texts=reference_texts,
                    output_root=output_dir,
                    run_once_fn=_offline_run_once,
                    max_steps=max_steps,
                    beam_width=beam_width,
                    variant_count=variant_count,
                    action_budget=action_budget,
                    concurrency_manager=optimizer_cm,
                    max_concurrency=max_concurrency,
                    checkpoint_dir=checkpoint_dir,
                    resume_from_checkpoint=bool(ref_cfg.get("resume_prompt_state")),
                )
            else:
                report = optimizer.optimize_from_runs(
                    workflow_id=wid,
                    run_results=run_results,
                    workflow_yaml=(yaml_tree, yaml_path),
                    reference_path=reference_path,
                    reference_texts=reference_texts,
                    output_root=output_dir,
                )
            summaries.append(
                {
                    "workflow_id": wid,
                    "runs": len(run_results),
                    "issues": len(report.issues),
                    "patches": len(report.patches),
                    "yaml_path": str(report.yaml_path),
                    "patched_path": str(report.patched_path) if report.patched_path else None,
                    "reference_path": reference_path,
                }
            )
            logger.info(
                "优化完成",
                extra={
                    "workflow_id": wid,
                    "runs": len(run_results),
                    "issues": len(report.issues),
                    "patches": len(report.patches),
                },
            )
        except Exception as ex:
            summaries.append({"workflow_id": wid, "runs": len(run_results), "patches": 0, "error": str(ex)})
            logger.warning("生成优化报告失败", extra={"workflow_id": wid, "error": str(ex)})

    logger.info(
        "运行/优化流程完成",
        extra={
            "workflows": len(summaries),
            "summaries": summaries[:10],
            "optimize": optimize,
            "run_workflows": run_workflows,
        },
    )
    return run_records if run_records else summaries


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
