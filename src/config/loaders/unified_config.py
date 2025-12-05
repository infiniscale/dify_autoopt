"""
Unified Application Configuration Loader

Single-file config loader designed to work with a consolidated `config/config.yaml`:
  - meta, dify, auth, variables, workflows, execution, optimization, io_paths, logging

This complements existing Env/Catalog/TestPlan loaders and does not break them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class WorkflowInline:
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    reference: Optional[Any] = None
    api_key: Optional[str] = None


@dataclass
class AppConfig:
    meta: Dict[str, Any]
    dify: Dict[str, Any]
    auth: Dict[str, Any]
    variables: Dict[str, Any]
    workflows: List[WorkflowInline]
    execution: Dict[str, Any]
    optimization: Dict[str, Any]
    io_paths: Dict[str, Any]
    logging: Dict[str, Any]


class UnifiedConfigLoader:
    """Loader for single-file consolidated config (config.yaml)."""

    def _expand_env_vars(self, data: Any) -> Any:
        import os, re

        if isinstance(data, str):
            def replacer(match):
                var = match.group(1)
                val = os.getenv(var)
                return match.group(0) if val is None else val

            return re.sub(r"\$\{(\w+)\}", replacer, data)
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._expand_env_vars(x) for x in data]
        return data

    def load(self, path: Path | str) -> AppConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Unified config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        # Debug: raw top-level keys (do not log sensitive values)
        try:
            from src.utils.logger import get_logger
            get_logger("config.loader").debug(
                "unified_config loaded (raw)", extra={"path": str(p.resolve()), "top_keys": list(raw.keys())}
            )
        except Exception:
            pass
        if not isinstance(raw, dict):
            raise ValueError("config.yaml must be a mapping at root level")

        expanded = self._expand_env_vars(raw)
        try:
            from src.utils.logger import get_logger
            get_logger("config.loader").debug(
                "unified_config expanded", extra={"has_logging": bool((expanded or {}).get("logging"))}
            )
        except Exception:
            pass

        # 基础类型校验（更严格的提示）
        def _expect_mapping(name: str):
            v = expanded.get(name)
            if v is not None and not isinstance(v, dict):
                raise ValueError(f"config.yaml: '{name}' must be a mapping")

        def _expect_list(name: str):
            v = expanded.get(name)
            if v is not None and not isinstance(v, list):
                raise ValueError(f"config.yaml: '{name}' must be a list")

        for k in ["meta", "dify", "auth", "variables", "execution", "optimization", "io_paths", "logging"]:
            _expect_mapping(k)
        _expect_list("workflows")

        def _get_dict(key: str) -> Dict[str, Any]:
            v = expanded.get(key) or {}
            return v if isinstance(v, dict) else {}

        def _get_list(key: str) -> List[Any]:
            v = expanded.get(key) or []
            return v if isinstance(v, list) else []

        workflows: List[WorkflowInline] = []
        for w in _get_list("workflows"):
            if isinstance(w, dict) and "id" in w:
                workflows.append(
                    WorkflowInline(
                        id=str(w.get("id")),
                        name=w.get("name"),
                        description=w.get("description"),
                        inputs=w.get("inputs"),
                        parameters=w.get("parameters"),
                        reference=w.get("reference"),
                        api_key=w.get("api_key"),
                    )
                )

        app = AppConfig(
            meta=_get_dict("meta"),
            dify=_get_dict("dify"),
            auth=_get_dict("auth"),
            variables=_get_dict("variables"),
            workflows=workflows,
            execution=_get_dict("execution"),
            optimization=_get_dict("optimization"),
            io_paths=_get_dict("io_paths"),
            logging=_get_dict("logging"),
        )

        # 尝试记录详细日志（如果日志已初始化）
        try:
            from src.utils.logger import get_logger
            lg = get_logger("config.loader")
            details = {
                "path": str(p.resolve()),
                "meta": {
                    "version": app.meta.get("version"),
                    "environment": app.meta.get("environment"),
                },
                "dify": {
                    "base_url": app.dify.get("base_url"),
                    "tenant_id": app.dify.get("tenant_id"),
                },
                "auth": {
                    "has_username": bool(app.auth.get("username")),
                    "has_password": bool(app.auth.get("password")),
                    "has_api_key": bool(app.auth.get("api_key")),
                },
                "workflows": len(app.workflows),
                "variables_count": len(app.variables or {}),
                "execution": {
                    "concurrency": app.execution.get("concurrency"),
                    "timeout": app.execution.get("timeout"),
                    "retry_count": app.execution.get("retry_count"),
                },
                "optimization": {
                    "strategy": (app.optimization or {}).get("strategy"),
                    "max_iterations": (app.optimization or {}).get("max_iterations"),
                },
                "logging": {
                    "level": (app.logging or {}).get("level"),
                    "format": (app.logging or {}).get("format"),
                    "console_enabled": (app.logging or {}).get("console_enabled"),
                    "file_enabled": (app.logging or {}).get("file_enabled"),
                },
            }
            lg.info("已加载统一配置文件", extra=details)
            # 进一步调试输出：各段键数量
            lg.debug(
                "配置段统计",
                extra={
                    "sections": {
                        "meta": len(app.meta or {}),
                        "dify": len(app.dify or {}),
                        "auth": len(app.auth or {}),
                        "variables": len(app.variables or {}),
                        "workflows": len(app.workflows or []),
                        "execution": len(app.execution or {}),
                        "optimization": len(app.optimization or {}),
                        "io_paths": len(app.io_paths or {}),
                        "logging": len(app.logging or {}),
                    }
                },
            )
        except Exception:
            # 不阻断主流程
            pass

        # 额外校验：工作流 inputs/reference 的多元素配对规则
        self._validate_workflow_multiplicity(app)

        return app

    # ---------------------- Validation Helpers ----------------------
    def validate(self, app: AppConfig) -> Dict[str, Any]:
        """Return {'errors': [...], 'warnings': [...]} with human-readable messages."""
        errors: List[str] = []
        warnings: List[str] = []

        # dify.base_url must exist and look like URL
        base_url = (app.dify or {}).get("base_url")
        if not base_url:
            errors.append("missing required: dify.base_url")
        elif not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            errors.append("invalid value: dify.base_url must start with http:// or https://")

        # auth: either api_key or username/password
        auth = app.auth or {}
        if not (auth.get("api_key") or (auth.get("username") and auth.get("password"))):
            errors.append("missing required: auth.api_key or auth.username/password")

        # logging level/format
        logging_cfg = app.logging or {}
        level = logging_cfg.get("level")
        if level and level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            warnings.append("unknown logging.level (expected DEBUG/INFO/WARNING/ERROR/CRITICAL)")

        fmt = logging_cfg.get("format")
        if fmt and fmt not in {"simple", "structured"}:
            warnings.append("unknown logging.format (expected simple/structured)")

        # execution basic checks
        exec_cfg = app.execution or {}
        conc = exec_cfg.get("concurrency")
        if conc is not None and (not isinstance(conc, int) or conc <= 0):
            warnings.append("execution.concurrency should be positive integer")
        timeout = exec_cfg.get("timeout")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            warnings.append("execution.timeout should be positive integer seconds")
        retry = exec_cfg.get("retry_count")
        if retry is not None and (not isinstance(retry, int) or retry < 0):
            warnings.append("execution.retry_count should be integer >= 0")

        # optimization strategy
        opt = app.optimization or {}
        strat = opt.get("strategy")
        allowed = {"auto", "clarity_focus", "efficiency_focus", "structure_focus", "llm_guided"}
        if strat and strat not in allowed:
            warnings.append("optimization.strategy not in allowed set")

        try:
            from src.utils.logger import get_logger
            get_logger("config.loader").debug(
                "配置校验完成", extra={"errors": len(errors), "warnings": len(warnings)}
            )
        except Exception:
            pass
        return {"errors": errors, "warnings": warnings}

    # ---------------------- Workflow Inputs Validation ----------------------
    def _validate_workflow_multiplicity(self, app: AppConfig) -> None:
        """Ensure list-typed inputs are aligned in length and reference matches.

        Rules:
        - Each workflow's inputs can be scalars or lists.
        - If any input is a list, then all list-typed inputs must have the same length N.
        - Scalars are allowed and are considered broadcastable.
        - If `reference` is provided, it must be either a scalar or a list of length N (when N exists).
        Violations raise ValueError with a helpful message.
        """
        for wf in app.workflows or []:
            inputs = (wf.inputs or {}) if isinstance(wf.inputs, dict) else {}

            def _extract_value(x):
                # Support new shape: {'type': 'file|string|number', 'value': ...}
                if isinstance(x, dict) and 'value' in x:
                    return x.get('value')
                return x

            # Collect lengths for list-typed values after extraction
            list_lengths = []
            for k, v in inputs.items():
                val = _extract_value(v)
                if isinstance(val, list):
                    list_lengths.append((k, len(val)))

            if not list_lengths:
                # All scalars -> ok
                continue

            lengths = [n for _, n in list_lengths]
            uniq = set(lengths)
            if len(uniq) > 1:
                raise ValueError(
                    f"Workflow '{wf.id}': list inputs must have equal length, got: "
                    + ", ".join(f"{k}={n}" for k, n in list_lengths)
                )

            N = lengths[0]
            # Ensure every list-typed input length == N
            for k, v in inputs.items():
                val = _extract_value(v)
                if isinstance(val, list) and len(val) != N:
                    raise ValueError(
                        f"Workflow '{wf.id}': input '{k}' length {len(val)} != expected {N}"
                    )

            # Validate reference field length if list
            ref = getattr(wf, 'reference', None)
            if ref is not None and isinstance(ref, list) and len(ref) != N:
                raise ValueError(
                    f"Workflow '{wf.id}': reference length {len(ref)} != expected {N}"
                )
