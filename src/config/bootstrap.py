"""
Config Bootstrap - Initialize application runtime from unified config.yaml

Responsibilities:
- Load consolidated config (config/config.yaml) via UnifiedConfigLoader
- Validate essential fields (dify.base_url, auth presence, logging presence)
- Expose a process-wide runtime accessor for other modules
- Emit detailed structured logs about the configuration
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .loaders.unified_config import UnifiedConfigLoader, AppConfig


@dataclass
class AppRuntime:
    app: AppConfig

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

    global _runtime
    _runtime = AppRuntime(app=app_cfg)
    try:
        from src.utils.logger import get_logger
        get_logger("config.bootstrap").debug("runtime initialized")
    except Exception:
        pass
    return _runtime
