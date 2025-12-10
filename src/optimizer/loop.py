"""Self-loop optimizer orchestrator.

Runs: execute workflow -> optimize prompts -> import/publish -> fetch api key -> repeat.
Stops on: no patches, target failure rate met, or max cycles reached.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from src.config.bootstrap import get_runtime
from src.optimizer.prompt_optimizer import PromptOptimizer, _status_from_run
from src.utils.logger import get_logger
from src.workflow.apps import _resolve_base_url, _resolve_token
from src.workflow.api_keys import list_api_keys
from src.workflow.export import _safe_dirname_from_id, export_app_dsl
from src.workflow.execute import _normalize_inputs, execute_workflow_v1
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
    data = resp.get("data")
    if isinstance(data, dict):
        if data.get("id"):
            return str(data["id"])
        if isinstance(data.get("app"), dict) and data["app"].get("id"):
            return str(data["app"]["id"])
    return fallback


def _choose_api_key(keys: List[Dict[str, Any]], fallback: str) -> str:
    for item in keys:
        for key_name in ("api_key", "key", "apiKey"):
            if key_name in item and item[key_name]:
                return str(item[key_name])
    return fallback


def run_optimize_loop(
    *,
    max_cycles: int = 3,
    allow_no_patch_rounds: int = 1,
    target_failure_rate: Optional[float] = None,
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
    exec_timeout = rt.app.execution.get("timeout", 9000) if rt.app.execution else 9000
    opt_cfg = rt.app.optimization or {}
    llm_cfg = opt_cfg.get("llm")
    base_url = _resolve_base_url(rt.dify_base_url)
    token = _resolve_token(None)

    if not base_url or not token:
        logger.warning("缺少 base_url 或 token，无法执行发布/导入流程", extra={"base_url": base_url, "has_token": bool(token)})
        return []

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

        # 确保初始 DSL 可用：优先导出最新 DSL
        try:
            exported = export_app_dsl(current_app_id, base_url=base_url, token=token, include_secret=False)
            current_yaml_path = str(exported)
        except Exception as ex:  # noqa: BLE001
            logger.warning("导出 DSL 失败，跳过该工作流", extra={"workflow_id": current_app_id, "error": str(ex)})
            continue

        for cycle in range(1, max_cycles + 1):
            # 1) 运行工作流
            rows, declared_types = _normalize_inputs(getattr(wf, "inputs", {}) or {})
            try:
                run_results = execute_workflow_v1(
                    current_app_id,
                    rows,
                    base_url=rt.dify_base_url or base_url,
                    api_key=current_api_key,
                    timeout=exec_timeout,
                    input_types=declared_types,
                    output_dir=output_dir,
                    persist_results=True,
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
                report = optimizer.optimize_from_runs(
                    workflow_id=current_app_id,
                    run_results=run_results,
                    workflow_yaml=(yaml_tree, Path(current_yaml_path)),
                    reference_path=ref_path,
                    reference_texts=reference_texts,
                    output_root=output_dir,
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
            try:
                import_resp = import_app_yaml(yaml_path=report.patched_path, base_url=base_url, token=token)
                new_app_id = _extract_app_id_from_import(import_resp, current_app_id)
            except Exception as ex:  # noqa: BLE001
                logger.warning(
                    "导入优化后的 DSL 失败，终止自循环",
                    extra={"workflow_id": current_app_id, "cycle": cycle, "error": str(ex)},
                )
                break

            try:
                publish_workflow(new_app_id, base_url=base_url, token=token)
            except Exception as ex:  # noqa: BLE001
                logger.warning(
                    "发布优化后的工作流失败，终止自循环",
                    extra={"workflow_id": new_app_id, "cycle": cycle, "error": str(ex)},
                )
                break

            new_keys = list_api_keys(base_url, new_app_id, token)
            current_api_key = _choose_api_key(new_keys, current_api_key)
            current_app_id = new_app_id
            current_yaml_path = str(report.patched_path)

    return summaries
