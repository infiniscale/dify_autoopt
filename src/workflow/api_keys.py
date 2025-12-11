"""
Workflow API key helper.

Provides helpers to fetch or create API keys for a given app id from the Dify console API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import get_logger
from .apps import _resolve_token, _login_token

logger = get_logger("workflow.api_keys")


def _parse_keys(payload: Any) -> List[Dict[str, Any]]:
    """Normalize API key response into a list of dicts."""
    if isinstance(payload, dict):
        if "data" in payload:
            payload = payload.get("data")
        elif "keys" in payload:
            payload = payload.get("keys")
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return []

    result: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # 兼容字段名 token/api_key/key
        if item.get("token") and not item.get("api_key"):
            item["api_key"] = item["token"]
        if item.get("key") and not item.get("api_key"):
            item["api_key"] = item["key"]
        result.append(item)
    return result


def create_api_key(
    base_url: str,
    app_id: str,
    token: Optional[str],
    *,
    name: Optional[str] = None,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    Create an API key for the given app. Returns the created key dict on success.
    """
    resolved_token = _resolve_token(token) or _login_token(base_url)
    if not resolved_token:
        logger.warning("无法创建 API Key，缺少 token", extra={"app_id": app_id})
        return None
    url = f"{base_url.rstrip('/')}/console/api/apps/{app_id}/api-keys"
    headers = {"Authorization": f"Bearer {resolved_token}"}
    payload = {"name": name or f"auto-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}", "type": "app"}
    resp = None
    try:
        logger.info("Creating API key", extra={"url": url, "app_id": app_id, "payload": payload})
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        body_snippet = resp.text[:300] if resp.text else ""
        logger.debug("Create API key response", extra={"status": status, "body": body_snippet})
        resp.raise_for_status()
        data = resp.json()
        parsed = _parse_keys(data)
        if parsed:
            logger.info("API key created", extra={"app_id": app_id, "sample": parsed[:1]})
            return parsed[0]
        logger.warning("API key creation returned empty payload", extra={"app_id": app_id, "raw": data})
    except Exception as exc:  # noqa: BLE001
        try:
            logger.warning(
                "Failed to create API key",
                extra={
                    "url": url,
                    "app_id": app_id,
                    "status": resp.status_code if resp else None,
                    "body": resp.text[:300] if resp else None,
                    "error": str(exc),
                    "headers": headers,
                    "payload": payload,
                },
            )
        except Exception:
            pass
    return None


def list_api_keys(
    base_url: str,
    app_id: str,
    token: Optional[str],
    *,
    timeout: float = 10.0,
    create_when_missing: bool = False,
    create_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch API keys for a given app id from the Dify console API. Optionally create one if missing.

    Args:
        base_url: Dify base URL, e.g. http://xy.dnset.com:1280
        app_id: application id
        token: console access token (Bearer)
        timeout: request timeout in seconds
        create_when_missing: if True, create a new key when none exist
        create_name: optional name used when creating a key

    Returns:
        A list of API key dicts (best-effort parsing). Returns [] on errors.
    """
    resolved_token = _resolve_token(token) or _login_token(base_url)
    if not resolved_token:
        logger.warning("无法获取 API Keys，缺少 token", extra={"app_id": app_id})
        return []

    url = f"{base_url.rstrip('/')}/console/api/apps/{app_id}/api-keys"
    headers = {"Authorization": f"Bearer {resolved_token}"}
    resp = None
    try:
        logger.info("Fetching API keys", extra={"url": url, "app_id": app_id, "headers": headers})
        resp = requests.get(url, headers=headers, timeout=timeout)
        status = resp.status_code
        body_snippet = resp.text[:300] if resp.text else ""
        logger.debug("API keys raw response", extra={"status": status, "body": body_snippet})
        resp.raise_for_status()
        data = resp.json()
        keys = _parse_keys(data)
        if keys:
            logger.info(
                "API keys fetched",
                extra={
                    "app_id": app_id,
                    "count": len(keys),
                    "sample": keys[:1],
                },
            )
            return keys
        logger.warning("API keys empty", extra={"app_id": app_id, "raw": data})
        if create_when_missing:
            created = create_api_key(base_url, app_id, resolved_token, name=create_name, timeout=timeout)
            return [created] if created else []
    except Exception as exc:  # noqa: BLE001
        try:
            logger.warning(
                "Failed to fetch API keys",
                extra={
                    "url": url,
                    "app_id": app_id,
                    "status": resp.status_code if resp else None,
                    "body": resp.text[:300] if resp else None,
                    "error": str(exc),
                    "headers": headers,
                },
            )
        except Exception:
            pass
    return []
