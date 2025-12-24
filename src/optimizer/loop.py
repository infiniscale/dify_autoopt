"""Self-loop optimizer orchestrator.

Runs: execute workflow -> optimize prompts -> import/publish -> fetch api key -> repeat.
Stops on: no patches, target failure rate met, or max cycles reached.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from src.config.bootstrap import get_runtime
from src.optimizer.prompt_optimizer import PromptOptimizer, _set_by_pointer, _status_from_run, prompt_hash
from src.optimizer.yaml_loader import load_workflow_yaml
from src.utils.logger import get_logger
from src.workflow.api_keys import list_api_keys
from src.workflow.apps import _resolve_base_url, _resolve_token, _login_token, list_all_apps
from src.workflow.execute import _normalize_inputs, execute_workflow_v1
from src.workflow.export import _safe_dirname_from_id, export_app_dsl
from src.workflow.imports import import_app_yaml
from src.workflow.publish import publish_workflow


@dataclass
class LoopCycleResult:
    cycle: int
    app_id: str
    runs: int
    failure_rate: float
    patches: int
    patched_path: Optional[str]
    reference_path: Optional[str]
    stop_reason: Optional[str] = None


def _compute_failure_rate(run_results: Sequence[Dict[str, Any]]) -> float:
    if not run_results:
        return 1.0
    failures = 0
    for run in run_results:
        try:
            status = _status_from_run(run)
        except Exception:
            status = str(run.get("status", "")).lower() if isinstance(run, dict) else "unknown"
        if status not in {"success", "succeeded"}:
            failures += 1
    return failures / max(1, len(run_results))


def _resolve_reference_file(opt_cfg: dict, workflow_id: str) -> Optional[str]:
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


def _resolve_reference_texts(wf_entry, run_results: Sequence[Dict[str, Any]], logger) -> Optional[List[Optional[str]]]:
    ref = getattr(wf_entry, "reference", None)
    if ref is None:
        return None
    ref_list = ref if isinstance(ref, list) else [ref]
    if len(ref_list) == 1 and len(run_results) > 1:
        ref_list = ref_list * len(run_results)
    elif len(ref_list) != len(run_results):
        try:
            logger.warning(
                "reference 数量与运行结果数量不匹配，按最小长度对齐",
                extra={"references": len(ref_list), "runs": len(run_results)},
            )
        except Exception:
            pass
    texts: List[Optional[str]] = []
    for path in ref_list[: len(run_results)]:
        if not path:
            texts.append(None)
            continue
        try:
            p = Path(path)
            texts.append(p.read_text(encoding="utf-8") if p.exists() else None)
        except Exception as ex:
            try:
                logger.warning("读取 reference 失败", extra={"reference": path, "error": str(ex)})
            except Exception:
                pass
            texts.append(None)
    return texts


def _extract_app_id_from_import(resp: Dict[str, Any], fallback: str) -> str:
    if not isinstance(resp, dict):
        return fallback
    # Top-level response
    if resp.get("app_id"):
        return str(resp["app_id"])
    if resp.get("id") and not resp.get("data"):
        return str(resp["id"])
    data = resp.get("data")
    if isinstance(data, dict):
        if data.get("id"):
            return str(data["id"])
        if isinstance(data.get("app"), dict) and data["app"].get("id"):
            return str(data["app"]["id"])
        if data.get("app_id"):
            return str(data["app_id"])
    return fallback


def _tag_yaml_name(yaml_path: str, suffix: str, logger) -> Tuple[str, Optional[str]]:
    """
    Load YAML and append suffix to the app/workflow name if possible.
    Returns (yaml_content, new_name_if_changed).
    """
    content = Path(yaml_path).read_text(encoding="utf-8")
    new_name: Optional[str] = None
    try:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            return content, None

        def _update_name(container: Dict[str, Any], key: str) -> bool:
            if key in container and isinstance(container[key], str):
                orig = container[key]
                container[key] = f"{orig}-{suffix}"
                return True
            return False

        updated = False
        for key in ("name",):
            updated = updated or _update_name(data, key)
        if not updated and isinstance(data.get("app"), dict):
            updated = updated or _update_name(data["app"], "name")
        if not updated and isinstance(data.get("workflow"), dict):
            updated = updated or _update_name(data["workflow"], "name")

        if updated:
            new_name = data.get("name")
            content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
            try:
                logger.info("已为导入的 YAML 添加后缀", extra={"suffix": suffix, "new_name": new_name})
            except Exception:
                pass
    except Exception:
        return content, None
    return content, new_name


def _choose_api_key(keys: List[Dict[str, Any]], fallback: str) -> str:
    for item in keys:
        for key_name in ("api_key", "key", "apiKey"):
            if key_name in item and item[key_name]:
                return str(item[key_name])
    return fallback


def run_optimize_loop(
    *,
    max_cycles: Optional[int] = None,
    allow_no_patch_rounds: int = 1,
    target_failure_rate: Optional[float] = None,
    exit_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Execute -> Optimize -> Import/Publish -> Repeat, until stop conditions met.

    Stop conditions:
      - patches==0 for allow_no_patch_rounds
      - failure_rate <= target_failure_rate (if provided)
      - cycle >= max_cycles
    """
    logger = get_logger("optimizer.loop")
    rt = get_runtime()
    output_dir = (rt.app.io_paths or {}).get("output_dir") or "./outputs"
    opt_cfg = rt.app.optimization or {}
    exec_timeout = rt.app.execution.get("timeout", 9000) if rt.app.execution else 9000
    run_concurrency = rt.app.execution.get("concurrency", 1) if rt.app.execution else 1
    exec_retry = rt.app.execution.get("retry_count", 0) if rt.app.execution else 0
    exec_meta = {
        "execution_config": {
            "timeout": exec_timeout,
            "retry_count": exec_retry,
            "concurrency": run_concurrency,
            "response_mode": "blocking",
        }
    }
    if isinstance(opt_cfg.get("run_concurrency"), int) and opt_cfg.get("run_concurrency") > 0:
        run_concurrency = opt_cfg.get("run_concurrency")
    llm_cfg = opt_cfg.get("llm")

    cfg_max_cycles = opt_cfg.get("max_iterations")
    if max_cycles is None:
        max_cycles = cfg_max_cycles if isinstance(cfg_max_cycles, int) and cfg_max_cycles > 0 else 3

    cfg_exit_ratio = opt_cfg.get("exit_ratio")
    if exit_ratio is None and isinstance(cfg_exit_ratio, (int, float)):
        exit_ratio = cfg_exit_ratio
    if exit_ratio is not None and not (0 <= exit_ratio <= 1):
        exit_ratio = None
    base_url = _resolve_base_url(rt.dify_base_url)
    api_base = getattr(rt, "dify_api_base", None) or base_url
    # 控制台 token（导出/导入/发布用），与执行用 app_key 区分
    console_token = _resolve_token(None)
    if (not console_token or (isinstance(console_token, str) and console_token.startswith("app-"))) and base_url:
        console_token = _login_token(base_url) or console_token

    if not base_url:
        logger.warning("缺少 base_url，无法执行循环优化", extra={"base_url": base_url})
        return []
    try:
        workflows_ctx = rt.app.workflows or []
        logger.info(
            "Loop 初始上下文",
            extra={
                "base_url": base_url,
                "api_base": api_base,
                "has_token": bool(console_token),
                "token_prefix": console_token[:4] + "****" if console_token and isinstance(console_token, str) and len(console_token) >= 8 else "<short/empty>",
                "workflows": len(workflows_ctx),
                "run_concurrency": run_concurrency,
            },
        )
        # 简单校验 token 是否可用：拉一次 apps 列表（仅在 token 可用时）
        if console_token:
            try:
                list_all_apps(base_url=base_url, token=console_token, limit=1, max_pages=1)
            except Exception as ex:
                logger.warning("控制台 token 验证失败，后续导出/发布可能 401", extra={"error": str(ex)})
    except Exception:
        pass

    workflows = rt.app.workflows or []
    summaries: List[Dict[str, Any]] = []

    for wf in workflows:
        wid = getattr(wf, "id", None)
        if not wid:
            continue
        current_app_id = str(wid)
        current_api_key = getattr(wf, "api_key", None) or ""
        ref_path = _resolve_reference_file(opt_cfg, current_app_id)
        no_patch_rounds = 0

        # Build workflow context for injection
        wf_name = getattr(wf, "name", None)
        wf_desc = getattr(wf, "description", None)
        workflow_context: Optional[str] = None
        if wf_name or wf_desc:
            context_parts = []
            if wf_name:
                context_parts.append(f"Workflow Name: {wf_name}")
            if wf_desc:
                context_parts.append(f"Workflow Description: {wf_desc}")
            workflow_context = "\n".join(context_parts)

        current_yaml_path: Optional[str] = None
        # 确保初始 DSL 可用：优先导出最新 DSL（若有控制台 token），否则尝试读取本地已导出的 DSL
        if console_token:
            try:
                exported = export_app_dsl(current_app_id, base_url=base_url, token=console_token, include_secret=False)
                current_yaml_path = str(exported)
            except Exception as ex:  # noqa: BLE001
                logger.warning(
                    "导出 DSL 失败，尝试读取本地 DSL",
                    extra={
                        "workflow_id": current_app_id,
                        "error": str(ex),
                        "base_url": base_url,
                        "has_token": bool(console_token),
                    },
                )
        if current_yaml_path is None:
            try:
                tree, path = load_workflow_yaml(current_app_id, output_dir=output_dir)
                current_yaml_path = str(path)
            except Exception:
                logger.warning(
                    "未找到可用的 DSL，跳过该工作流",
                    extra={"workflow_id": current_app_id, "has_token": bool(console_token)},
                )
                continue

        for cycle in range(1, max_cycles + 1):
            # 1) 运行工作流
            rows, declared_types = _normalize_inputs(getattr(wf, "inputs", {}) or {})
            try:
                run_results = execute_workflow_v1(
                    current_app_id,
                    rows,
                    base_url=api_base,
                    api_key=current_api_key,
                    timeout=exec_timeout,
                    retry_count=exec_retry,
                    input_types=declared_types,
                    output_dir=output_dir,
                    persist_results=True,
                    concurrency=run_concurrency,
                    persist_metadata=exec_meta,
                )
            except Exception as ex:  # noqa: BLE001
                logger.warning(
                    "执行工作流失败，终止自循环",
                    extra={"workflow_id": current_app_id, "cycle": cycle, "error": str(ex)},
                )
                break

            failure_rate = _compute_failure_rate(run_results)
            reference_texts = _resolve_reference_texts(wf, run_results, logger)

            # 2) 优化提示词
            try:
                yaml_tree = yaml.safe_load(Path(current_yaml_path).read_text(encoding="utf-8"))
            except Exception as ex:  # noqa: BLE001
                logger.warning("读取 DSL 失败，终止自循环", extra={"workflow_id": current_app_id, "error": str(ex)})
                break

            optimizer = PromptOptimizer(default_output_root=output_dir, llm_config=llm_cfg)
            try:
                opt_strategy = str(opt_cfg.get("strategy") or "").lower()
                if opt_strategy == "prompt_state" and console_token:
                    try:
                        logger.info(
                            "PromptState optimizer enabled (online)",
                            extra={"workflow_id": current_app_id, "cycle": cycle},
                        )
                    except Exception:
                        pass

                    prompt_eval_cache: Dict[str, Tuple[str, str]] = {}
                    fallback_run = run_results[0] if run_results else {}
                    max_steps = opt_cfg.get("max_iterations")
                    max_steps = int(max_steps) if isinstance(max_steps, int) and max_steps > 0 else 3
                    beam_width = opt_cfg.get("beam_width")
                    beam_width = int(beam_width) if isinstance(beam_width, int) and beam_width > 0 else 1
                    variant_count = opt_cfg.get("variant_count")
                    variant_count = int(variant_count) if isinstance(variant_count, int) and variant_count > 0 else 1
                    action_budget = opt_cfg.get("action_budget")
                    max_concurrency = opt_cfg.get("concurrency")
                    max_concurrency = int(max_concurrency) if isinstance(max_concurrency,
                                                                         int) and max_concurrency > 0 else None
                    checkpoint_dir = Path(output_dir) / _safe_dirname_from_id(current_app_id) / f"prompt_state_c{cycle}"
                    optimizer_cm = rt.optimizer_concurrency if rt else None

                    def _online_run_once(prompt_text: str, block=None, message=None):
                        if not message or not message.get("path"):
                            return fallback_run
                        key = prompt_hash(prompt_text)
                        if key in prompt_eval_cache:
                            app_id, api_key = prompt_eval_cache[key]
                        else:
                            try:
                                temp_tree = copy.deepcopy(yaml_tree)
                                _set_by_pointer(temp_tree, message["path"], prompt_text)
                                yaml_content = yaml.safe_dump(temp_tree, allow_unicode=True, sort_keys=False)
                                import_resp = import_app_yaml(
                                    yaml_content=yaml_content, base_url=base_url, token=console_token
                                )
                                app_id = _extract_app_id_from_import(import_resp, current_app_id)
                                publish_workflow(app_id, base_url=base_url, token=console_token)
                                new_keys = list_api_keys(
                                    base_url,
                                    app_id,
                                    console_token,
                                    create_when_missing=True,
                                    create_name=f"autoopt-eval-{cycle}",
                                )
                                api_key = _choose_api_key(new_keys, current_api_key)
                                prompt_eval_cache[key] = (app_id, api_key)
                            except Exception as exc:  # noqa: BLE001
                                try:
                                    logger.warning(
                                        "PromptState 在线评估导入失败，回退到现有结果",
                                        extra={"workflow_id": current_app_id, "error": str(exc)},
                                    )
                                except Exception:
                                    pass
                                return fallback_run
                        eval_rows = rows[:1] if rows else []
                        if not eval_rows:
                            return fallback_run
                        try:
                            results = execute_workflow_v1(
                                app_id,
                                eval_rows,
                                base_url=api_base,
                                api_key=api_key,
                                timeout=exec_timeout,
                                retry_count=exec_retry,
                                input_types=declared_types,
                                output_dir=None,
                                persist_results=False,
                                concurrency=1,
                                persist_metadata=exec_meta,
                            )
                            return results[0] if results else fallback_run
                        except Exception as exc:  # noqa: BLE001
                            try:
                                logger.warning(
                                    "PromptState 在线评估执行失败，回退到现有结果",
                                    extra={"workflow_id": current_app_id, "error": str(exc)},
                                )
                            except Exception:
                                pass
                            return fallback_run

                    report = optimizer.optimize_with_prompt_state(
                        workflow_id=current_app_id,
                        run_results=run_results,
                        workflow_yaml=(yaml_tree, Path(current_yaml_path)),
                        reference_path=ref_path,
                        reference_texts=reference_texts,
                        output_root=output_dir,
                        run_once_fn=_online_run_once,
                        max_steps=max_steps,
                        beam_width=beam_width,
                        variant_count=variant_count,
                        action_budget=action_budget,
                        concurrency_manager=optimizer_cm,
                        max_concurrency=max_concurrency,
                        checkpoint_dir=checkpoint_dir,
                        resume_from_checkpoint=bool(opt_cfg.get("resume_prompt_state")),
                        workflow_context=workflow_context,
                    )
                else:
                    if opt_strategy == "prompt_state" and not console_token:
                        try:
                            logger.warning(
                                "PromptState 在线评估缺少 console token，回退到旧优化路径",
                                extra={"workflow_id": current_app_id, "cycle": cycle},
                            )
                        except Exception:
                            pass
                    report = optimizer.optimize_from_runs(
                        workflow_id=current_app_id,
                        run_results=run_results,
                        workflow_yaml=(yaml_tree, Path(current_yaml_path)),
                        reference_path=ref_path,
                        reference_texts=reference_texts,
                        output_root=output_dir,
                        workflow_context=workflow_context,
                    )
            except Exception as ex:  # noqa: BLE001
                logger.warning(
                    "生成优化报告失败，终止自循环",
                    extra={"workflow_id": current_app_id, "cycle": cycle, "error": str(ex)},
                )
                break

            patches_count = len(report.patches)
            cycle_result = LoopCycleResult(
                cycle=cycle,
                app_id=current_app_id,
                runs=len(run_results),
                failure_rate=failure_rate,
                patches=patches_count,
                patched_path=str(report.patched_path) if report.patched_path else None,
                reference_path=ref_path,
            )
            summaries.append(asdict(cycle_result))
            try:
                logger.info(
                    "自循环完成一轮",
                    extra={
                        "workflow_id": current_app_id,
                        "cycle": cycle,
                        "runs": len(run_results),
                        "failure_rate": failure_rate,
                        "patches": patches_count,
                        "skip_ratio": report.stats.get("prompts_skip_ratio") if isinstance(report.stats, dict) else None,
                    },
                )
            except Exception:
                pass

            # 3) 退出条件
            stop_reason = None
            if patches_count == 0:
                no_patch_rounds += 1
            else:
                no_patch_rounds = 0
            if no_patch_rounds >= allow_no_patch_rounds:
                stop_reason = "no_patches"
            if target_failure_rate is not None and failure_rate <= target_failure_rate:
                stop_reason = "target_met"
            skip_ratio = report.stats.get("prompts_skip_ratio") if isinstance(report.stats, dict) else None
            if exit_ratio is not None and skip_ratio is not None and skip_ratio >= exit_ratio:
                stop_reason = "skip_ratio"
            if cycle >= max_cycles:
                if stop_reason is None:
                    stop_reason = "max_cycles"

            if stop_reason:
                try:
                    logger.info(
                        "自循环结束",
                        extra={
                            "workflow_id": current_app_id,
                            "cycle": cycle,
                            "reason": stop_reason,
                            "failure_rate": failure_rate,
                            "patches": patches_count,
                        },
                    )
                except Exception:
                    pass
                summaries[-1]["stop_reason"] = stop_reason
                break

            if not report.patched_path:
                logger.warning("未生成 patched DSL，无法导入发布，终止自循环", extra={"workflow_id": current_app_id})
                break

            # 4) 导入 / 发布 / 获取 API Key
            if not console_token:
                logger.warning(
                    "缺少控制台 token，无法导入/发布优化后的 DSL，终止自循环",
                    extra={"workflow_id": current_app_id},
                )
                break
            tag = f"{datetime.now().strftime('%Y%m%d')}-c{cycle}"
            yaml_content, tagged_name = _tag_yaml_name(report.patched_path, tag, logger)
            try:
                import_resp = import_app_yaml(yaml_content=yaml_content, base_url=base_url, token=console_token)
                new_app_id = _extract_app_id_from_import(import_resp, current_app_id)
                if not new_app_id or new_app_id == current_app_id:
                    raise RuntimeError("导入未产生新 app_id")
            except Exception as ex:  # noqa: BLE001
                logger.warning(
                    "导入优化后的 DSL 失败，终止自循环",
                    extra={"workflow_id": current_app_id, "cycle": cycle, "error": str(ex)},
                )
                break

            try:
                publish_workflow(new_app_id, base_url=base_url, token=console_token)
            except Exception as ex:  # noqa: BLE001
                logger.warning(
                    "发布优化后的工作流失败，终止自循环",
                    extra={"workflow_id": new_app_id, "cycle": cycle, "error": str(ex)},
                )
                break

            new_keys = list_api_keys(
                base_url,
                new_app_id,
                console_token,
                create_when_missing=True,
                create_name=f"auto-loop-{tag}",
            )
            chosen_key = _choose_api_key(new_keys, "")
            if not chosen_key:
                logger.warning(
                    "发布后的工作流未获取到有效 API Key，终止自循环",
                    extra={"workflow_id": new_app_id, "cycle": cycle},
                )
                break
            current_api_key = chosen_key
            current_app_id = new_app_id
            current_yaml_path = str(report.patched_path)

    return summaries
