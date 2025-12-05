"""
Workflow Import Utilities

Upload a YAML DSL to Dify via console API.

Endpoint: POST {base_url}/console/api/apps/imports
Body (JSON): {"mode": "yaml-content", "yaml_content": "<file content>"}
Auth: Authorization: Bearer <token>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import requests

from src.utils.logger import get_logger, log_performance
from .apps import _resolve_token, _resolve_base_url, _mask_token


@log_performance("workflow_import_app_yaml")
def import_app_yaml(
    *,
    yaml_path: Optional[str | Path] = None,
    yaml_content: Optional[str] = None,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Import an app from YAML content.

    Either `yaml_path` or `yaml_content` must be provided.
    Returns the parsed JSON response.
    """
    logger = get_logger("workflow.imports")

    if not yaml_content and not yaml_path:
        raise ValueError("Either yaml_path or yaml_content must be provided")

    content: str
    if yaml_content is not None:
        content = str(yaml_content)
    else:
        p = Path(yaml_path)  # type: ignore[arg-type]
        if not p.exists():
            raise FileNotFoundError(f"YAML file not found: {p}")
        content = p.read_text(encoding="utf-8")

    resolved_token = _resolve_token(token)
    if not resolved_token:
        raise RuntimeError("No access token available. Provide `token` or set DIFY_API_TOKEN or configure token store.")

    resolved_base = _resolve_base_url(base_url)
    if not resolved_base:
        raise RuntimeError("No base_url provided and runtime not initialized.")

    url = f"{resolved_base}/console/api/apps/imports"
    headers = {
        "Authorization": f"Bearer {resolved_token}",
        "Content-Type": "application/json",
    }
    body = {"mode": "yaml-content", "yaml_content": content}

    try:
        logger.info(
            "Importing app YAML",
            extra={
                "url": url,
                "mode": body["mode"],
                "token": _mask_token(resolved_token),
                "yaml_length": len(content),
            },
        )
    except Exception:
        pass

    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    try:
        # Common success response might include new app id or status
        app_id = None
        if isinstance(data, dict):
            d = data.get("data")
            if isinstance(d, dict):
                app_id = d.get("id") or d.get("app", {}).get("id") if isinstance(d.get("app"), dict) else None
        logger.info("App YAML imported", extra={"app_id": app_id})
    except Exception:
        pass

    return data

