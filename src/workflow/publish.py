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
from .apps import _resolve_token, _resolve_base_url, _mask_token, _login_token


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

    resolved_base = _resolve_base_url(base_url)
    if not resolved_base:
        raise RuntimeError("No base_url provided and runtime not initialized.")

    resolved_token = _resolve_token(token)
    if not resolved_token:
        resolved_token = _login_token(resolved_base)
    if not resolved_token:
        raise RuntimeError("No access token available. Please login (username/password) or set DIFY_API_TOKEN.")

    url = f"{resolved_base}/console/api/apps/{app_id}/workflows/publish"
    body = payload if isinstance(payload, dict) else {}

    def _post_with_token(tok: str):
        headers = {
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
        }
        return requests.post(url, headers=headers, json=body, timeout=timeout)

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

    resp = _post_with_token(resolved_token)
    if resp.status_code == 401:
        refreshed_token = _login_token(resolved_base)
        if refreshed_token and refreshed_token != resolved_token:
            try:
                logger.info(
                    "Token expired, retrying publish with refreshed token",
                    extra={
                        "app_id": app_id,
                        "url": url,
                        "old_token": _mask_token(resolved_token),
                        "new_token": _mask_token(refreshed_token),
                    },
                )
            except Exception:
                pass
            resolved_token = refreshed_token
            resp = _post_with_token(resolved_token)

    try:
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        try:
            logger.warning(
                "Publish workflow failed",
                extra={
                    "status": resp.status_code,
                    "body": resp.text[:300],
                    "url": url,
                    "app_id": app_id,
                },
            )
        except Exception:
            pass
        raise
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
