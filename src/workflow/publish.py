"""
Workflow Publish Utilities

Publish a workflow for an app via the console API.

Endpoint example:
  POST {base_url}/console/api/apps/{app_id}/workflows/publish

Auth: Authorization: Bearer <token>
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import requests

from src.utils.logger import get_logger, log_performance
from .apps import _resolve_token, _resolve_base_url, _mask_token


@log_performance("workflow_publish")
def publish_workflow(
    app_id: str,
    *,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 30,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Publish the workflow of an app via Dify console API.

    Some deployments accept an empty JSON body; `payload` can be provided if required.
    Returns the parsed JSON response.
    """
    logger = get_logger("workflow.publish")

    resolved_token = _resolve_token(token)
    if not resolved_token:
        raise RuntimeError("No access token available. Provide `token` or set DIFY_API_TOKEN or configure token store.")

    resolved_base = _resolve_base_url(base_url)
    if not resolved_base:
        raise RuntimeError("No base_url provided and runtime not initialized.")

    url = f"{resolved_base}/console/api/apps/{app_id}/workflows/publish"
    headers = {
        "Authorization": f"Bearer {resolved_token}",
        "Content-Type": "application/json",
    }
    body = payload if isinstance(payload, dict) else {}

    try:
        logger.info(
            "Publishing workflow",
            extra={
                "app_id": app_id,
                "url": url,
                "token": _mask_token(resolved_token),
                "has_payload": bool(body),
            },
        )
    except Exception:
        pass

    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    try:
        status = None
        if isinstance(data, dict):
            d = data.get("data")
            if isinstance(d, dict):
                status = d.get("status") or d.get("result")
        logger.info("Workflow published", extra={"status": status})
    except Exception:
        pass

    return data

