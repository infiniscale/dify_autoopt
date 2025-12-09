"""
Config Bootstrap - Initialize application runtime from unified config.yaml

Responsibilities:
- Load consolidated config (config/config.yaml) via UnifiedConfigLoader
- Validate essential fields (dify.base_url, auth presence, logging presence)
- Expose a process-wide runtime accessor for other modules
- Emit detailed structured logs about the configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .loaders.unified_config import UnifiedConfigLoader, AppConfig
from src.execution import ConcurrencyManager


@dataclass
class AppRuntime:
    app: AppConfig
    concurrency: Dict[str, ConcurrencyManager] = field(default_factory=dict)

    @property
    def dify_base_url(self) -> Optional[str]:
        return (self.app.dify or {}).get("base_url")

    @property
    def has_token_auth(self) -> bool:
        return bool((self.app.auth or {}).get("api_key"))

    @property
    def has_password_auth(self) -> bool:
        a = self.app.auth or {}
        return bool(a.get("username") and a.get("password"))

    @property
    def logging_present(self) -> bool:
        return bool(self.app.logging)

    @property
    def workflows_count(self) -> int:
        return len(self.app.workflows or [])

    @property
    def workflow_concurrency(self) -> Optional[ConcurrencyManager]:
        return (self.concurrency or {}).get("workflow")

    @property
    def optimizer_concurrency(self) -> Optional[ConcurrencyManager]:
        return (self.concurrency or {}).get("optimizer")


_runtime: Optional[AppRuntime] = None


def get_runtime() -> AppRuntime:
    if _runtime is None:
        raise RuntimeError("Config runtime not initialized. Call bootstrap_from_unified() first.")
    return _runtime


def bootstrap_from_unified(path: Path | str) -> AppRuntime:
    """Load unified config.yaml and initialize runtime with detailed logs."""
    cfg_path = Path(path)
    # Debug: bootstrap start
    try:
        from src.utils.logger import get_logger
        get_logger("config.bootstrap").debug("bootstrap_from_unified start", extra={"path": str(cfg_path.resolve())})
    except Exception:
        pass
    loader = UnifiedConfigLoader()
    app_cfg = loader.load(cfg_path)

    # Emit validation and summary logs (logger may already be initialized)
    try:
        from src.utils.logger import get_logger

        lg = get_logger("config.bootstrap")
        v = loader.validate(app_cfg)
        errors = v.get("errors", [])
        warnings = v.get("warnings", [])

        payload = {
            "path": str(cfg_path.resolve()),
            "workflows": len(app_cfg.workflows),
            "logging": {
                "present": bool(app_cfg.logging),
                "level": (app_cfg.logging or {}).get("level"),
                "format": (app_cfg.logging or {}).get("format"),
            },
            "dify": {
                "base_url": (app_cfg.dify or {}).get("base_url"),
                "tenant_id": (app_cfg.dify or {}).get("tenant_id"),
            },
            "auth": {
                "has_api_key": bool((app_cfg.auth or {}).get("api_key")),
                "has_username": bool((app_cfg.auth or {}).get("username")),
            },
            "errors": errors,
            "warnings": warnings,
        }
        if errors:
            lg.warning("配置引导存在必填项缺失", extra=payload)
        else:
            lg.info("配置引导完成", extra=payload)
        lg.debug("配置引导校验结果", extra={"errors": errors, "warnings": warnings})
    except Exception:
        pass

    cm_map = _build_concurrency_managers(app_cfg)

    global _runtime
    _runtime = AppRuntime(app=app_cfg, concurrency=cm_map)
    try:
        from src.utils.logger import get_logger
        get_logger("config.bootstrap").debug("runtime initialized")
    except Exception:
        pass
    return _runtime


def _build_concurrency_managers(app_cfg: AppConfig) -> Dict[str, ConcurrencyManager]:
    """Create independent concurrency pools for workflow execution and optimizer."""

    def _sanitize(value: Any, default: int) -> int:
        try:
            iv = int(value)
            return iv if iv > 0 else default
        except Exception:
            return default

    exec_cfg = app_cfg.execution or {}
    opt_cfg = app_cfg.optimization or {}
    workflow_limit = _sanitize(exec_cfg.get("concurrency"), default=1)
    optimizer_limit = _sanitize(opt_cfg.get("concurrency"), default=workflow_limit)

    try:
        from src.utils.logger import get_logger

        get_logger("config.bootstrap").debug(
            "Initialized concurrency managers",
            extra={
                "workflow_limit": workflow_limit,
                "optimizer_limit": optimizer_limit,
            },
        )
    except Exception:
        pass

    return {
        "workflow": ConcurrencyManager(workflow_limit),
        "optimizer": ConcurrencyManager(optimizer_limit),
    }
